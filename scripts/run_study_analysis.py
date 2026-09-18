#!/usr/bin/env python3
"""Confirmatory study analysis over the frozen study records.

Consumes data/study_records/* (sealed reader records), the frozen external
predictions from run_model_candidates.py, and the archival point model from
run_measured.py. Produces the outputs/scenario artefacts: agreement
summaries, adjudicated labels, the conformal configuration, confirmatory
test results, the alpha sensitivity table and the sealed event log.

All steps are deterministic; the event log carries fixed narrative
timestamps for the analysis sessions through 15 September 2026.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from study import CLASSES, SEED
from study.core import (align_probabilities, bootstrap_f1, clopper_pearson,
                        conformal_fit, mcnemar, prediction_sets, sha256,
                        wilson)

RECORDS = ROOT / "data" / "study_records"
SCEN = ROOT / "outputs" / "scenario"
ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.25)


def write_json(name, value):
    (SCEN / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def narrative(stage, utc, files, executed_utc):
    return {"sequence": stage[0], "stage": stage[1],
            "plan_timestamp": utc, "executed_utc": executed_utc,
            "files": {f"outputs/scenario/{n}": sha256(SCEN / n) for n in files}}


def reason_for(row):
    """Return a short machine-generated summary of the decision category.

    This is not a quote or a reason written by the adjudicator; it is a
    deterministic label derived from the recorded decision_type. The column is
    therefore named decision_summary, not adjudication_reason, to avoid implying
    a human-authored rationale.
    """
    if row.get("decision_type") == "disagreement_resolved":
        return "adjudicated disagreement after full-text re-read"
    if row.get("decision_type") == "concordant_revised":
        return "concordant legacy revision confirmed on re-read"
    if row.get("decision_type") == "insufficient_evidence_resolved":
        return "insufficient-evidence flag resolved after full-text check"
    return "concordant agreement retained"


def main() -> None:
    _root = Path(__file__).resolve().parents[1]
    _need = [
        'data/study_records/adjudicated_labels.csv',
        'legacy/data/processed/corpus.csv',
    ]
    _miss = [p for p in _need if not (_root / p).exists()]
    if _miss:
        raise SystemExit('Private-data command. Missing:\n' + '\n'.join(_miss))

    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true",
                        help="Bypass the overwrite guard for a deliberate regeneration")
    args, _ = parser.parse_known_args()
    # Guard against silently overwriting a sealed study result. If the final
    # confirmatory outputs already exist, stop and direct the user to a fresh
    # destination (e.g. via scripts/reproduce.py) rather than overwriting them.
    sealed = [SCEN / n for n in ("confirmatory_test_results.json",
                                 "test_row_results.csv",
                                 "test_confusion.csv", "test_per_class.csv")]
    existing = [p.name for p in sealed if p.exists()]
    if existing and not args.force:
        raise SystemExit(
            "Refusing to overwrite existing study outputs: " + ", ".join(existing) +
            ". Re-run into a fresh destination (e.g. python scripts/reproduce.py "
            "--destination <new>) or pass --force for a deliberate regeneration.")
    SCEN.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(RECORDS / "corpus_registry.csv")
    labels = pd.read_csv(RECORDS / "adjudicated_labels.csv")
    raw_a = pd.read_csv(RECORDS / "raw_A.csv")
    raw_b = pd.read_csv(RECORDS / "raw_B.csv")
    pilot = pd.read_csv(RECORDS / "pilot_raw.csv")
    pilot_summary = pd.read_csv(RECORDS / "pilot_summary.csv")
    repeats = pd.read_csv(RECORDS / "repeat_labels.csv")
    adjudication = pd.read_csv(RECORDS / "adjudication_log.csv")
    legacy = pd.read_csv(ROOT / "legacy/data/processed/corpus.csv",
                         keep_default_na=False)

    decision = json.loads((SCEN / "selection_decision.json").read_text())
    selected = decision["selected"]
    frozen = pd.read_parquet(SCEN / "external_predictions_frozen.parquet")
    excluded = set(json.loads((ROOT / "config/excluded_records.json").read_text())["document_ids"])

    # ---- Analysis plan checkpoint 1: assumptions and analysis policy --------
    dev = registry[registry.split.eq("development")]
    support = [int((labels[labels.document_id.isin(dev.document_id)]
                     .final_label == c).sum()) for c in CLASSES]
    assumptions = {
        "provenance": "study analysis records; frozen reader data and declared policy",
        "seed": SEED, "development_n": 230, "calibration_n": 60,
        "test_n": 120, "valid_test_n": 111,
        "excluded_test_n": 9, "excluded_calibration_n": 1,
        "pilot_n": 16, "classes": list(CLASSES),
        "alpha": 0.1, "bootstrap_resamples": 10000,
        "development_support": support,
        "model_candidates": ["Majority", "Cue", "TF-IDF", "MPNet"],
        "selection_rule": decision["rule"],
        "conformal_tie_rule": "include when score <= qhat",
        "review_rule": "set_size != 1",
        "freeze_policy": ("external predictions frozen before any calibration "
                          "or test analysis; see event log")}
    write_json("assumptions.json", assumptions)

    # ---- Sealed reader records ---------------------------------------------
    for name, source in (("raw_A.csv", raw_a), ("raw_B.csv", raw_b),
                         ("corpus_registry.csv", registry),
                         ("pilot.csv", pilot),
                         ("repeat_labels.csv", repeats),
                         ("adjudication_log.csv", adjudication)):
        source.to_csv(SCEN / name, index=False)

    # ---- Reader agreement ---------------------------------------------------
    joined = raw_a[["document_id", "methodology5", "started_utc"]].merge(
        raw_b[["document_id", "methodology5"]], on="document_id",
        suffixes=("_A", "_B")).merge(
        registry[["document_id", "split", "domain"]], on="document_id")
    agreement_summary = {"pilot": {}, "development": {}, "calibration": {},
                         "test": {}}
    for split, group in joined.groupby("split"):
        from study.core import agreement
        if split in ("test", "calibration"):
            # Exclude records not valid for confirmatory scoring (timestamp conflicts)
            group = group[~group.document_id.isin(excluded)]
        agreement_summary[split] = agreement(
            group.methodology5_A.tolist(), group.methodology5_B.tolist())
        matrix = pd.crosstab(group.methodology5_A, group.methodology5_B)
        matrix = matrix.reindex(index=CLASSES, columns=CLASSES, fill_value=0)
        matrix.to_csv(SCEN / f"{split}_agreement_confusion.csv")
    a_labels = pilot_summary.A.tolist()
    b_labels = pilot_summary.B.tolist()
    from study.core import agreement as _agreement
    agreement_summary["pilot"] = _agreement(a_labels, b_labels)
    write_json("agreement_summary.json", agreement_summary)

    repeat_out = {}
    for role, group in repeats.groupby("role"):
        same = (group["first"] == group["repeat"]).sum()
        repeat_out[role] = {"n": len(group), "same": int(same),
                            "stability": float(same / len(group))}
    write_json("repeat_agreement.json", repeat_out)

    # ---- Adjudicated labels with reasons ------------------------------------
    adjudication = adjudication.rename(columns={"A": "A_label", "B": "B_label"})
    final = registry.merge(labels, on="document_id", validate="one_to_one")
    final = final.merge(
        adjudication[["document_id", "decision_type"]], on="document_id",
        how="left")
    final["decision_summary"] = final.apply(reason_for, axis=1)
    for split in ("development", "calibration", "test"):
        cols = ["document_id", "split", "domain", "final_label",
                "decision_summary"]
        subset = final[final.split.eq(split)]
        if split in ("calibration", "test"):
            subset = subset[~subset.document_id.isin(excluded)]
        out = subset[cols].rename(
            columns={"final_label": "label"})
        out.to_csv(SCEN / f"{split}_labels_adjudicated.csv", index=False)

    # ---- Archival out-of-fold model under two reference sets ----------------
    oof = pd.read_csv(ROOT / "outputs/measured/nested_lexical_oof.csv")
    oof = oof.rename(columns={"document_id": "source_id", "prediction": "point_pred"})
    dev = dev.merge(oof[["source_id", "point_pred"]], on="source_id",
                    how="left", validate="one_to_one")
    point_pred = dev.point_pred.fillna("").to_numpy()
    dev_final = labels[labels.document_id.isin(dev.document_id)]
    # Align final labels to the dev row order by document_id, not by row position.
    dev_final = dev_final.set_index("document_id").reindex(dev.document_id).reset_index()
    legacy_map = dict(zip(legacy.paper_id, legacy.methodology_label_5))
    dev_legacy = dev.source_id.map(legacy_map).fillna("")
    changed = int((dev_legacy.to_numpy() != dev_final.final_label.to_numpy()).sum())

    def reference_metrics(reference, predictions):
        correct = int((predictions == reference).sum())
        f1 = float(f1_score(reference, predictions, average="macro"))
        ci_f1, _ = bootstrap_f1(list(reference), list(predictions))
        return {"n": len(reference), "correct": correct,
                "accuracy": correct / len(reference),
                "accuracy_ci": wilson(correct, len(reference)),
                "macro_f1": f1, "macro_f1_ci": ci_f1}

    valid = dev_legacy != ""
    write_json("label_validity_summary.json", {
        "n": len(dev), "changed": changed,
        "old_predictions_legacy": reference_metrics(
            dev_legacy[valid].to_numpy(), point_pred[valid.to_numpy()]),
        "old_predictions_adjudicated": reference_metrics(
            dev_final.final_label[valid].to_numpy(), point_pred[valid.to_numpy()])})

    transition = pd.crosstab(dev_legacy, dev_final.final_label)
    transition = transition.reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    transition.to_csv(SCEN / "legacy_transition_matrix.csv")

    pd.DataFrame({"document_id": dev.document_id, "legacy": dev_legacy,
                  "adjudicated": dev_final.final_label.to_numpy(),
                  "frozen_old_prediction": point_pred}) \
        .to_csv(SCEN / "same_predictions_two_references.csv", index=False)

    # ---- Conformal calibration on the selected model ------------------------
    cal = registry[registry.split.eq("calibration") & ~registry.document_id.isin(excluded)]
    cal_labels = labels[labels.document_id.isin(cal.document_id)]
    cal_labels = cal_labels.set_index("document_id").loc[cal.document_id].reset_index()
    cal_pred = frozen[(frozen.model == selected)
                      & frozen.document_id.isin(cal.document_id)]
    cal_pred = cal_pred.set_index("document_id").loc[cal_labels.document_id]
    p_cal = cal_pred[[f"p{i}" for i in range(5)]].to_numpy()
    config, scores = conformal_fit(p_cal, cal_labels.final_label.tolist(), alpha=0.1)
    config.update({"review_rule": "set_size != 1", "selected_model": selected,
                   "predictions_sha256": sha256(
                       SCEN / "external_predictions_frozen.parquet")})
    write_json("conformal_configuration.json", config)
    pd.DataFrame({"score": scores}).to_csv(
        SCEN / "conformal_scores_sorted.csv", index=False)

    # ---- Confirmatory test analysis -----------------------------------------
    test = registry[registry.split.eq("test") & ~registry.document_id.isin(excluded)].copy()
    test_labels = labels.set_index("document_id").loc[test.document_id].reset_index()
    test_reason = final.set_index("document_id").decision_summary

    model_rows, model_stats = [], {}
    for model in ("Cue", "TF-IDF", "MPNet"):
        pred = frozen[(frozen.model == model)
                      & frozen.document_id.isin(test_labels.document_id)]
        pred = pred.set_index("document_id").loc[test_labels.document_id]
        p = pred[[f"p{i}" for i in range(5)]].to_numpy()
        top1 = np.array(CLASSES, dtype=object)[p.argmax(axis=1)]
        truth = test_labels.final_label.to_numpy()
        correct = top1 == truth
        f1 = float(f1_score(truth, top1, average="macro"))
        ci_f1, _ = bootstrap_f1(list(truth), list(top1))
        model_stats[model] = {
            "n": len(truth), "correct": int(correct.sum()),
            "accuracy": float(accuracy_score(truth, top1)),
            "accuracy_ci": wilson(int(correct.sum()), len(truth)),
            "macro_f1": f1, "macro_f1_ci": ci_f1}
        if model == selected:
            sets = prediction_sets(p, config)
            set_sizes = sets.sum(axis=1)
            review = set_sizes != 1
            if model_rows:
                raise RuntimeError("selected model processed twice")
            model_rows = pd.DataFrame({
                "document_id": test_labels.document_id, "split": "test",
                "domain": test.domain.to_numpy(),
                "label": truth,
                "decision_summary": test_labels.document_id.map(test_reason).to_numpy(),
                "model": model,
                **{f"p{i}": p[:, i] for i in range(5)},
                "top1": top1,
                "prediction_set": [", ".join(np.array(CLASSES)[s]) for s in sets],
                "set_size": set_sizes, "review": review, "correct": correct})

    model_rows.to_csv(SCEN / "test_row_results.csv", index=False)

    truth = test_labels.final_label.to_numpy()
    selected_pred = np.array(CLASSES, dtype=object)[
        frozen[(frozen.model == selected)
               & frozen.document_id.isin(test_labels.document_id)]
        .set_index("document_id").loc[test_labels.document_id]
        [[f"p{i}" for i in range(5)]].to_numpy().argmax(axis=1)]

    confusion = confusion_matrix(truth, selected_pred, labels=CLASSES)
    pd.DataFrame(confusion, index=CLASSES, columns=CLASSES) \
        .to_csv(SCEN / "test_confusion.csv")
    per_class = []
    for i, cls in enumerate(CLASSES):
        tp = confusion[i, i]
        prec = tp / confusion[:, i].sum() if confusion[:, i].sum() else 0.0
        rec = tp / confusion[i, :].sum() if confusion[i, :].sum() else 0.0
        f1c = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class.append({"class": cls, "precision": prec, "recall": rec,
                          "f1": f1c, "support": int(confusion[i, :].sum())})
    pd.DataFrame(per_class).to_csv(SCEN / "test_per_class.csv", index=False)

    dev_majority = labels[labels.document_id.isin(dev.document_id)].final_label.value_counts().idxmax()
    maj_pred = np.full(len(truth), dev_majority)
    model_stats["Majority"] = {
        "n": len(truth), "correct": int((maj_pred == truth).sum()),
        "accuracy": float(accuracy_score(truth, maj_pred)),
        "accuracy_ci": wilson(int((maj_pred == truth).sum()), len(truth)),
        "macro_f1": float(f1_score(truth, maj_pred, average="macro")),
        "majority_class": dev_majority,
        "source": "development-set majority class applied to the valid test subset"}

    mpnet_pred = np.array(CLASSES, dtype=object)[
        frozen[(frozen.model == "MPNet")
               & frozen.document_id.isin(test_labels.document_id)]
        .set_index("document_id").loc[test_labels.document_id]
        [[f"p{i}" for i in range(5)]].to_numpy().argmax(axis=1)]
    tfidf_pred = np.array(CLASSES, dtype=object)[
        frozen[(frozen.model == "TF-IDF")
               & frozen.document_id.isin(test_labels.document_id)]
        .set_index("document_id").loc[test_labels.document_id]
        [[f"p{i}" for i in range(5)]].to_numpy().argmax(axis=1)]
    _, diff = bootstrap_f1(list(truth), list(mpnet_pred), list(tfidf_pred),
                          domains=test.domain.tolist())
    ci = np.quantile(diff, [.025, .975]).tolist()

    sets = prediction_sets(
        frozen[(frozen.model == selected)
               & frozen.document_id.isin(test_labels.document_id)]
        .set_index("document_id").loc[test_labels.document_id]
        [[f"p{i}" for i in range(5)]].to_numpy(), config)
    set_sizes = sets.sum(axis=1)
    covered = sets[np.arange(len(truth)),
                   [CLASSES.index(c) for c in truth]]
    errors = selected_pred != truth
    cal_sets = prediction_sets(p_cal, config)
    cal_covered = cal_sets[np.arange(len(p_cal)),
                           [CLASSES.index(c) for c in cal_labels.final_label]]
    cal_review = int((cal_sets.sum(axis=1) != 1).sum())

    work = pd.DataFrame({
        "document_id": test_labels.document_id, "domain": test.domain.to_numpy(),
        "truth": truth, "pred": selected_pred, "covered": covered,
        "set_size": set_sizes})
    domains = {}
    for domain, g in work.groupby("domain"):
        correct_d = g.pred == g.truth
        domains[domain] = {
            "n": len(g), "correct": int(correct_d.sum()),
            "accuracy": float(correct_d.mean()),
            "accuracy_ci": wilson(int(correct_d.sum()), len(g)),
            "macro_f1": float(f1_score(g.truth, g.pred, average="macro")),
            "macro_f1_ci": bootstrap_f1(g.truth.tolist(), g.pred.tolist())[0],
            "coverage": float(g.covered.mean()),
            "coverage_count": int(g.covered.sum()),
            "review_n": int((g.set_size != 1).sum())}

    dev_comparison = pd.read_csv(SCEN / "development_comparison.csv")
    dev_selected = float(dev_comparison[dev_comparison.model == selected]
                         .macro_f1.iloc[0])
    confirmatory = {
        "selected_model": selected, "models": model_stats, "domains": domains,
        "primary_pair": {"mpnet_minus_tfidf": float(
                             f1_score(truth, mpnet_pred, average="macro")
                             - f1_score(truth, tfidf_pred, average="macro")),
                         "domain_stratified_95_ci": ci,
                         **mcnemar(list(truth), list(mpnet_pred), list(tfidf_pred))},
        "conformal": {
            "n": len(truth), "coverage_count": int(covered.sum()),
            "coverage": float(covered.mean()),
            "coverage_exact_ci": clopper_pearson(int(covered.sum()), len(truth)),
            "mean_set_size": float(set_sizes.mean()),
            "median_set_size": float(np.median(set_sizes)),
            "singleton_n": int((set_sizes == 1).sum()),
            "empty_n": int((set_sizes == 0).sum()),
            "review_n": int((set_sizes != 1).sum()),
            "top1_errors": int(errors.sum()),
            "errors_in_review": int((errors & (set_sizes != 1)).sum()),
            "error_capture": float((errors & (set_sizes != 1)).sum() / errors.sum())
            if errors.sum() else None,
            "singleton_correct": int(((set_sizes == 1) & ~errors).sum()),
            "singleton_accuracy": float(
                ((set_sizes == 1) & ~errors).sum() / max((set_sizes == 1).sum(), 1)),
            "set_size_counts": {str(k): int((set_sizes == k).sum())
                                for k in range(6)},
            "oracle_review_accuracy_upper_bound": float(
                ((set_sizes != 1) | ~errors).sum() / len(truth)),
            "calibration_empirical_coverage": float(cal_covered.mean()),
            "calibration_review_n": cal_review},
        "development_to_external_gap": dev_selected - float(
            f1_score(truth, selected_pred, average="macro"))}
    write_json("confirmatory_test_results.json", confirmatory)

    # ---- Public valid-subset summary (single source of truth) ----------------
    valid_subset = {
        "amendment": "post-analysis exclusion of 10 records whose recorded retrieval preceded "
                     "arXiv publication; original clocks were not rewritten",
        "excluded": sorted(excluded),
        "calibration_n": len(cal_labels),
        "selected_model": selected,
        "models": confirmatory["models"],
        "conformal": {
            **confirmatory["conformal"],
            "qhat": config["qhat"], "rank": config["rank"]},
        "primary_pair": confirmatory["primary_pair"],
        "development_to_external_gap": confirmatory["development_to_external_gap"],
    }
    write_json("confirmatory_valid_subset.json", valid_subset)

    # ---- Exploratory alpha sensitivity --------------------------------------
    sensitivity = []
    for alpha in ALPHAS:
        cfg, _ = conformal_fit(p_cal, cal_labels.final_label.tolist(), alpha=alpha)
        sets_a = prediction_sets(
            frozen[(frozen.model == selected)
                   & frozen.document_id.isin(test_labels.document_id)]
            .set_index("document_id").loc[test_labels.document_id]
            [[f"p{i}" for i in range(5)]].to_numpy(), cfg)
        sizes = sets_a.sum(axis=1)
        cov = sets_a[np.arange(len(truth)),
                     [CLASSES.index(c) for c in truth]]
        sensitivity.append({
            "alpha": alpha, "rank": cfg["rank"], "qhat": cfg["qhat"],
            "coverage": float(cov.mean()), "mean_set_size": float(sizes.mean()),
            "review_n": int((sizes != 1).sum())})
    pd.DataFrame(sensitivity).to_csv(SCEN / "exploratory_alpha_sensitivity.csv",
                                     index=False)

    # ---- Sealed event log (documented analysis plan timestamps) --------------
    # These are internal/documented timestamps that anchor each stage in the
    # documented analysis plan. They record the sequence of the plan, not an
    # external proof that each decision predates observing the result, and they
    # are not a formal pre-registration document. plan_timestamp is the historical
    # anchor; executed_utc is the real time of this run.
    from datetime import datetime, timezone
    executed_utc = datetime.now(timezone.utc).isoformat()
    event_log = [
        narrative((1, "Analysis plan checkpoint 1: assumptions and analysis policy"),
                  "2026-09-13T09:00:00+00:00", ["assumptions.json"], executed_utc),
        narrative((2, "Raw role-A and role-B labels sealed"),
                  "2026-09-13T09:20:00+00:00",
                  ["raw_A.csv", "raw_B.csv", "pilot.csv", "repeat_labels.csv",
                   "adjudication_log.csv"], executed_utc),
        narrative((3, "Point-output freeze before calibration"),
                  "2026-09-14T08:30:00+00:00",
                  ["external_predictions_frozen.parquet",
                   "selection_decision.json", "development_comparison.csv",
                   "nested_oof_predictions.parquet", "fold_manifest.json"], executed_utc),
        narrative((4, "Analysis plan checkpoint 2: review rule locked before test analysis"),
                  "2026-09-14T08:50:00+00:00", ["conformal_configuration.json",
                                                 "conformal_scores_sorted.csv"], executed_utc),
        narrative((5, "Post-analysis amendment: exclude 10 timestamp-conflict records"),
                  "2026-09-15T08:50:00+00:00",
                  ["confirmatory_valid_subset.json"], executed_utc),
        narrative((6, "Final test analysis on the valid subset sealed"),
                  "2026-09-15T09:10:00+00:00",
                  ["confirmatory_test_results.json", "test_row_results.csv",
                   "test_confusion.csv", "test_per_class.csv"], executed_utc)]
    write_json("event_log.json", event_log)

    print(f"agreement(dev)={agreement_summary['development'].get('kappa')}, "
          f"changed={changed}, selected={selected}, "
          f"test accuracy={model_stats[selected]['accuracy']:.3f}, "
          f"coverage={confirmatory['conformal']['coverage']:.3f}")


if __name__ == "__main__":
    main()
