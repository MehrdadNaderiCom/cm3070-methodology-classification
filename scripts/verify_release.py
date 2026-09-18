"""Verify numerical evidence, contracts, freeze hashes and optional release inventory."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from study import CLASSES
from study.core import (sha256, validated_probabilities, prediction_sets, conformal_fit,
                        strict_join)

def is_public_tree():
    return (ROOT / "data/demo/PUBLIC_RELEASE").exists()


def scan_forbidden_text():
    forbidden = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or any(x in p.parts for x in ("__pycache__", ".git")):
            continue
        if p.suffix.lower() not in {".csv", ".json", ".md", ".txt", ".jsonl"}:
            continue
        if p.name == "release_manifest.json":
            continue
        head = p.read_text(encoding="utf-8", errors="ignore")[:200]
        if p.suffix.lower() == ".csv" and "demo" not in p.parts:
            first = head.splitlines()[0] if head else ""
            if "abstract" in {c.strip().strip('"').lower() for c in first.split(",")}:
                forbidden.append(str(p.relative_to(ROOT)))
    return forbidden


def verify_public(check, checks):
    check(not (ROOT / "outputs/measured/point_model.bin").exists(), "no archival model binary")
    check(not (ROOT / "outputs/model_runs/mpnet_embeddings.npz").exists(), "no text embeddings")
    check(not (ROOT / "data/study_records/raw_A.csv").exists(), "no raw reader-A forms")
    check(not (ROOT / "data/study_records/raw_B.csv").exists(), "no raw reader-B forms")
    check(not scan_forbidden_text(), "no abstract column in published tables")
    demo = ROOT / "data/demo/demo_model.bin"
    manifest = json.loads((ROOT / "data/demo/demo_model.manifest.json").read_text(encoding="utf-8"))
    check(demo.exists() and sha256(demo) == manifest["model_sha256"], "public demo model hash")
    table = pd.read_csv(ROOT / "data/public/paper_index.csv")
    check("abstract" not in table.columns, "paper index has no abstracts")
    check({"document_id", "source_id", "split", "final_label", "redistribution_status",
           "confirmatory_status"} <= set(table.columns),
          "paper index has required public columns")
    check(len(table) == 426 and table.document_id.is_unique, "426 public paper identifiers")
    check(int((table.confirmatory_status == "excluded").sum()) == 10, "ten records excluded from confirmatory")
    agg = json.loads((ROOT / "outputs/public/confirmatory_test_results.json").read_text(encoding="utf-8"))
    check(agg["models"]["Cue"]["correct"] == 63, "published Cue correct count")
    check(agg["models"]["Cue"]["n"] == 111, "published Cue test n")
    check(agg["models"]["Majority"]["correct"] == 37, "majority uses development class")
    rows = pd.read_csv(ROOT / "outputs/public/test_row_results.csv")
    confu = pd.read_csv(ROOT / "outputs/public/test_confusion.csv", index_col=0)
    per = pd.read_csv(ROOT / "outputs/public/test_per_class.csv")
    check(len(rows) == 111, "test_row_results n=111")
    check(int(confu.to_numpy().sum()) == 111, "confusion matrix sums to 111")
    check(int(per.support.sum()) == 111, "per-class support sums to 111")
    release = ROOT / "release_manifest.json"
    inventory = json.loads(release.read_text(encoding="utf-8"))
    for row in inventory["files"]:
        check(sha256(ROOT / row["path"]) == row["sha256"], f"release hash: {row['path']}")
    checks.append("public inventory hashed")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["auto", "study", "public"], default="auto",
                        help="Force the verification profile instead of auto-detection")
    args = parser.parse_args()
    checks = []
    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    if args.profile == "study":
        is_public = False
    elif args.profile == "public":
        is_public = True
    else:
        is_public = is_public_tree()
    if is_public:
        verify_public(check, checks)
        print(json.dumps({"status": "PASS", "profile": "public", "checks": len(checks),
                          "details": checks}, indent=2))
        return
    s, m = ROOT / "outputs/scenario", ROOT / "outputs/measured"
    r = ROOT / "outputs/model_runs"
    registry = pd.read_csv(s / "corpus_registry.csv")
    analytical = registry[registry.split.ne("pilot")]
    check(len(analytical) == 410 and analytical.document_id.is_unique, "410 unique analytical IDs")
    check(analytical.split.value_counts().to_dict() ==
          {"development": 230, "test": 120, "calibration": 60}, "230/60/120 split counts")
    check(int((registry.split == "pilot").sum()) == 16, "16 pilot records outside the analysis")
    for role in ("A", "B"):
        raw = pd.read_csv(s / f"raw_{role}.csv")
        strict_join(analytical[["document_id"]], raw)
        check(set(raw.methodology5) <= set(CLASSES), f"reader {role} canonical labels")
        alts = raw.candidate_alternative.fillna("")
        check(set(alts) <= set(CLASSES) | {""}, f"reader {role} alternatives are canonical")
        started = pd.to_datetime(raw.started_utc)
        saved = pd.to_datetime(raw.saved_utc)
        check((saved > started).all() and (saved - started).dt.total_seconds().between(60, 1800).all(),
              f"reader {role} plausible form durations")
    p = pd.read_parquet(s / "external_predictions_frozen.parquet")
    config = json.loads((s / "conformal_configuration.json").read_text())
    check(sha256(s / "external_predictions_frozen.parquet") == config["predictions_sha256"],
          "frozen external prediction hash")
    for model, g in p.groupby("model"):
        check(len(g) == 180 and g.document_id.is_unique, f"{model} external ID contract")
        validated_probabilities(g[[f"p{i}" for i in range(5)]].to_numpy())
        checks.append(f"{model} probability contract")
    selected = p[p.model == config["selected_model"]]
    cal = pd.read_csv(s / "calibration_labels_adjudicated.csv")
    cp = selected[selected.document_id.isin(cal.document_id)].drop(columns=["split", "domain"], errors="ignore")
    c = strict_join(cal, cp)
    recal, _ = conformal_fit(c[[f"p{i}" for i in range(5)]].to_numpy(), c.label.to_numpy(), .1)
    check(recal["rank"] == config["rank"] and np.isclose(recal["qhat"], config["qhat"]),
          "rank and quantile recomputation")
    test = pd.read_csv(s / "test_labels_adjudicated.csv")
    tp = selected[selected.document_id.isin(test.document_id)].drop(columns=["split", "domain"], errors="ignore")
    t = strict_join(test, tp)
    probs = t[[f"p{i}" for i in range(5)]].to_numpy()
    pred = np.array(CLASSES)[probs.argmax(axis=1)]
    result = json.loads((s / "confirmatory_test_results.json").read_text())
    check(np.isclose(f1_score(t.label, pred, average="macro"),
                     result["models"][config["selected_model"]]["macro_f1"], atol=1e-12, rtol=0),
          "selected test macro-F1 matches stored result")
    correct = int((t.label == pred).sum())
    check(correct == result["models"][config["selected_model"]]["correct"],
          f"{correct} correct top-one decisions")
    sets = prediction_sets(probs, config)
    truth = np.array([CLASSES.index(x) for x in t.label])
    covered = int(sets[np.arange(len(t)), truth].sum())
    check(covered == result["conformal"]["coverage_count"], "covered test labels match stored count")
    check(int((sets.sum(axis=1) != 1).sum()) == result["conformal"]["review_n"],
          "review flags match stored count")
    same = pd.read_csv(s / "same_predictions_two_references.csv")
    lv = json.loads((s / "label_validity_summary.json").read_text())
    check(int((same.legacy != same.adjudicated).sum()) == lv["changed"], "changed reference labels")
    folds_measured = json.loads((m / "fold_manifest.json").read_text())
    for f in folds_measured:
        train, held = set(f["train_ids"]), set(f["test_ids"])
        check(not train & held, f"measured fold {f.get('fold', f.get('outer_fold'))} no ID overlap")
    folds_scenario = json.loads((s / "fold_manifest.json").read_text())
    for group, fold_list in folds_scenario.items():
        for f in fold_list:
            train, held = set(f["train_ids"]), set(f["test_ids"])
            check(not train & held, f"scenario {group} fold {f.get('outer_fold', f.get('fold'))} no ID overlap")
    manifest = json.loads((m / "training_manifest.json").read_text())
    check(sha256(m / "point_model.bin") == manifest["model_sha256"], "actual fitted-model hash")
    encode_log = json.loads((r / "mpnet_encode_log.json").read_text())
    check(encode_log["executed"] and encode_log["revision"].startswith("e8c3b32edf54"),
          "MPNet actually executed at the pinned revision")
    check(sha256(r / "mpnet_embeddings.npz") == encode_log["embeddings_sha256"],
          "MPNet embedding cache hash")
    events = json.loads((s / "event_log.json").read_text())
    for e in events:
        for rel, expected in e["files"].items():
            check(sha256(ROOT / rel) == expected, f"freeze hash verified: {Path(rel).name}")
    release = ROOT / "release_manifest.json"
    if release.exists():
        inventory = json.loads(release.read_text())
        for row in inventory["files"]:
            check(sha256(ROOT / row["path"]) == row["sha256"], f"release hash: {row['path']}")
    print(json.dumps({"status": "PASS", "checks": len(checks), "details": checks}, indent=2))

if __name__ == "__main__":
    main()
