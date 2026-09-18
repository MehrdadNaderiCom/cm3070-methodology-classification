#!/usr/bin/env python3
"""Train and compare candidate text-classification models (real runs).

Development records carry the adjudicated study labels; four candidate
families are compared with nested out-of-fold predictions on the development
set, a declared selection rule is applied, and the selected model's
external predictions are frozen before any calibration or test analysis.

Outputs (outputs/scenario/):
  development_comparison.csv    dev OOF accuracy / macro-F1 for each candidate
  selection_decision.json       declared rule and paired difference CI
  nested_oof_predictions.parquet OOF probabilities (TF-IDF and MPNet)
  external_predictions_frozen.parquet  external predictions (TF-IDF and MPNet)
  fold_manifest.json            outer and inner fold memberships
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from study import CLASSES, SEED
from study.core import (align_probabilities, bootstrap_f1, group_folds,
                        lexical_pipeline, nested_oof, validated_probabilities,
                        wilson)
from study.cues import cue_frame_scores

SCEN = ROOT / "outputs" / "scenario"
MODEL_RUNS = ROOT / "outputs" / "model_runs"
SELECTION_RULE = ("Highest development out-of-fold macro-F1; ties broken "
                  "toward the simpler model")
CUE_SMOOTHING = 1.0


def cue_probabilities(titles, abstracts):
    """Additive-smoothed cue scores as class probabilities."""
    scores = cue_frame_scores(titles, abstracts)
    raw = np.array([[s[c] for c in CLASSES] for s in scores]) + CUE_SMOOTHING
    return validated_probabilities(raw / raw.sum(axis=1, keepdims=True))


def mpnet_frame():
    """Load cached MPNet embeddings and attach study labels."""
    bundle = np.load(MODEL_RUNS / "mpnet_embeddings.npz", allow_pickle=False)
    ids = bundle["document_ids"].astype(str)
    emb = bundle["embeddings"]
    labels = pd.read_csv(ROOT / "data" / "study_records" / "adjudicated_labels.csv")
    registry = pd.read_csv(ROOT / "data" / "study_records" / "corpus_registry.csv")
    frame = registry[["document_id", "split", "domain", "group", "title", "abstract"]]
    frame = frame.merge(labels, on="document_id", validate="one_to_one")
    frame = frame.rename(columns={"final_label": "label"})
    emb_frame = pd.DataFrame({"document_id": ids})
    emb_frame["embedding"] = list(emb)
    return frame.merge(emb_frame, on="document_id", validate="one_to_one")


def mpnet_oof(dev):
    """Nested OOF probabilities for the logistic regression on MPNet vectors."""
    X = np.stack(dev.embedding.to_numpy()).astype(np.float64)
    y = dev.label.to_numpy(dtype=object)
    groups = dev.group.to_numpy(dtype=object)
    outer = group_folds(y, groups)
    p = np.zeros((len(y), 5))
    fold_ids = np.full(len(y), -1, dtype=int)
    manifest = []
    for fold, (tr, te) in enumerate(outer):
        inner = group_folds(y[tr], groups[tr], requested=3, seed=SEED + fold)
        search = GridSearchCV(
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=SEED),
            {"C": [0.3, 1.0, 3.0]}, scoring="f1_macro", cv=inner, n_jobs=1,
            error_score="raise")
        search.fit(X[tr], y[tr])
        p[te] = align_probabilities(search.predict_proba(X[te]), search.classes_)
        fold_ids[te] = fold
        manifest.append({"outer_fold": fold,
                         "train_ids": dev.document_id.iloc[tr].tolist(),
                         "test_ids": dev.document_id.iloc[te].tolist(),
                         "best_params": {k.replace("lr__", ""): v for k, v in
                                         search.best_params_.items()}})
    assert (fold_ids >= 0).all()
    return p, fold_ids, manifest


def evaluate_oof(frame, p, name):
    pred = np.array(CLASSES, dtype=object)[p.argmax(axis=1)]
    correct = int((pred == frame.label.to_numpy()).sum())
    accuracy = correct / len(frame)
    f1 = float(f1_score(frame.label, pred, average="macro"))
    ci_acc = wilson(correct, len(frame))
    ci_f1, _ = bootstrap_f1(frame.label.tolist(), pred.tolist())
    return {"model": name, "n": len(frame), "correct": correct,
            "accuracy": accuracy, "accuracy_ci": ci_acc, "macro_f1": f1,
            "macro_f1_ci": ci_f1}, pred


def main() -> None:
    _root = Path(__file__).resolve().parents[1]
    _need = [
        'data/study_records/adjudicated_labels.csv',
        'legacy/data/processed/corpus.csv',
    ]
    _miss = [p for p in _need if not (_root / p).exists()]
    if _miss:
        raise SystemExit('Private-data command. Missing:\n' + '\n'.join(_miss))

    # Guard against silently overwriting the frozen external predictions. If the
    # frozen predictions already exist, stop and direct to a fresh destination.
    frozen_file = SCEN / "external_predictions_frozen.parquet"
    if frozen_file.exists():
        raise SystemExit(
            "Refusing to overwrite frozen external predictions ("
            "outputs/scenario/external_predictions_frozen.parquet). Re-run into a "
            "fresh destination (e.g. python scripts/reproduce.py --destination <new>).")
    SCEN.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(ROOT / "data" / "study_records" / "adjudicated_labels.csv")
    registry = pd.read_csv(ROOT / "data" / "study_records" / "corpus_registry.csv")
    frame = registry[["document_id", "split", "domain", "group", "title", "abstract"]]
    frame = frame.merge(labels, on="document_id", validate="one_to_one")
    frame = frame.rename(columns={"final_label": "label"})
    dev = frame[frame.split.eq("development")].reset_index(drop=True)
    ext = frame[frame.split.ne("development") & frame.split.ne("pilot")]
    ext = ext.reset_index(drop=True)

    # ---- Candidate 1: majority class --------------------------------------
    support = dev.label.value_counts()
    majority = support.index[0]
    p_majority = np.tile(np.array([support.get(c, 0) for c in CLASSES]) / len(dev),
                         (len(dev), 1))

    # ---- Candidate 2: transparent cue rules -------------------------------
    p_cue = cue_probabilities(dev.title.tolist(), dev.abstract.tolist())

    # ---- Candidate 3: TF-IDF lexical pipeline (nested OOF) -----------------
    tfidf_frame = dev[["document_id", "title", "abstract", "label", "group"]].copy()
    p_tfidf, folds_tfidf, manifest_tfidf = nested_oof(tfidf_frame)

    # ---- Candidate 4: MPNet embeddings + logistic regression ---------------
    mpnet = mpnet_frame()
    mpnet_dev = mpnet[mpnet.split.eq("development")].reset_index(drop=True)
    mpnet_ext = mpnet[mpnet.split.isin(["calibration", "test"])].reset_index(drop=True)
    p_mpnet, folds_mpnet, manifest_mpnet = mpnet_oof(mpnet_dev)

    rows, preds = [], {}
    for name, p in (("Majority", p_majority), ("Cue", p_cue),
                    ("TF-IDF", p_tfidf), ("MPNet", p_mpnet)):
        row, pred = evaluate_oof(dev, p, name)
        rows.append(row)
        preds[name] = p
    comparison = pd.DataFrame(rows)
    comparison.to_csv(SCEN / "development_comparison.csv", index=False)

    # ---- Declared selection rule ------------------------------------
    _, diff_samples = bootstrap_f1(dev.label.tolist(),
                                   np.array(CLASSES, dtype=object)[
                                       p_mpnet.argmax(axis=1)].tolist(),
                                   np.array(CLASSES, dtype=object)[
                                       p_tfidf.argmax(axis=1)].tolist())
    gain = float(f1_score(dev.label, np.array(CLASSES, dtype=object)[
        p_mpnet.argmax(axis=1)], average="macro")
        - f1_score(dev.label, np.array(CLASSES, dtype=object)[
            p_tfidf.argmax(axis=1)], average="macro"))
    ci = np.quantile(diff_samples, [.025, .975]).tolist()
    # Explicit simplicity preference for ties: Majority < Cue < TF-IDF < MPNet.
    # Sort by macro-F1 descending, then by simplicity rank ascending.
    simplicity = {"Majority": 0, "Cue": 1, "TF-IDF": 2, "MPNet": 3}
    comparison = comparison.copy()
    comparison["_simplicity"] = comparison["model"].map(simplicity)
    selected = (comparison
                .sort_values(["macro_f1", "_simplicity"],
                             ascending=[False, True]).iloc[0]["model"])
    decision = {"selected": selected, "rule": SELECTION_RULE,
                "development_macro_f1": {r["model"]: r["macro_f1"]
                                         for r in rows},
                "mpnet_minus_tfidf": gain, "paired_95_ci": ci}
    (SCEN / "selection_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    # ---- Freeze external predictions before any calibration or test use ----
    text_ext = (ext.title + ". " + ext.abstract).to_numpy(dtype=object)
    tfidf_fit = lexical_pipeline()
    tfidf_fit.fit((dev.title + ". " + dev.abstract).to_numpy(dtype=object),
                  dev.label.to_numpy())
    p_ext_tfidf = align_probabilities(tfidf_fit.predict_proba(text_ext),
                                      tfidf_fit.classes_)

    X_dev = np.stack(mpnet_dev.embedding.to_numpy()).astype(np.float64)
    X_ext = np.stack(mpnet_ext.embedding.to_numpy()).astype(np.float64)
    mpnet_fit = LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=SEED).fit(X_dev, mpnet_dev.label)
    p_ext_mpnet = align_probabilities(mpnet_fit.predict_proba(X_ext),
                                      mpnet_fit.classes_)

    p_majority_ext = np.tile(
        np.array([support.get(c, 0) for c in CLASSES]) / len(dev), (len(ext), 1))
    p_cue_ext = cue_probabilities(ext.title.tolist(), ext.abstract.tolist())
    frozen = pd.concat([
        pd.DataFrame({"document_id": ext.document_id, "model": "Majority",
                     **{f"p{i}": p_majority_ext[:, i] for i in range(5)}}),
        pd.DataFrame({"document_id": ext.document_id, "model": "Cue",
                     **{f"p{i}": p_cue_ext[:, i] for i in range(5)}}),
        pd.DataFrame({"document_id": ext.document_id, "model": "TF-IDF",
                     **{f"p{i}": p_ext_tfidf[:, i] for i in range(5)}}),
        pd.DataFrame({"document_id": ext.document_id, "model": "MPNet",
                     **{f"p{i}": p_ext_mpnet[:, i] for i in range(5)}})])
    frozen.to_parquet(SCEN / "external_predictions_frozen.parquet", index=False)

    oof = pd.concat([
        pd.DataFrame({"document_id": dev.document_id, "model": "TF-IDF",
                      "outer_fold": folds_tfidf, "label": dev.label,
                     **{f"p{i}": p_tfidf[:, i] for i in range(5)}}),
        pd.DataFrame({"document_id": mpnet_dev.document_id, "model": "MPNet",
                      "outer_fold": folds_mpnet, "label": mpnet_dev.label,
                     **{f"p{i}": p_mpnet[:, i] for i in range(5)}})])
    oof.to_parquet(SCEN / "nested_oof_predictions.parquet", index=False)

    folds_out = {"outer": [], "tfidf": manifest_tfidf, "mpnet": manifest_mpnet}
    for fold, (tr, te) in enumerate(group_folds(dev.label.to_numpy(dtype=object),
                                                dev.group.to_numpy(dtype=object))):
        folds_out["outer"].append({
            "fold": fold, "train_ids": dev.document_id.iloc[tr].tolist(),
            "test_ids": dev.document_id.iloc[te].tolist()})
    (SCEN / "fold_manifest.json").write_text(
        json.dumps(folds_out, indent=2) + "\n", encoding="utf-8")

    print(comparison.to_string(index=False))
    print(f"selected: {selected} (gain {gain:+.4f}, CI [{ci[0]:+.4f}, {ci[1]:+.4f}])")


if __name__ == "__main__":
    main()
