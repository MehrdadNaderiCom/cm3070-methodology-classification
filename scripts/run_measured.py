"""Re-execute archival lexical experiments and measure the revised pipeline."""
from pathlib import Path
import importlib.metadata
import json
import platform
import sys
import time
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "legacy/src")]
import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from study import CLASSES, SEED
from study.core import (fingerprint, group_folds, lexical_pipeline, nested_oof,
                        normalized_title, sha256)
from cdfm import tasks
from cdfm.evaluation.folds import make_folds
from cdfm.evaluation.metrics import evaluate
from cdfm.models.baseline import build_tfidf_baseline

OUT = ROOT / "outputs/measured"
OUT.mkdir(parents=True, exist_ok=True)


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    _root = Path(__file__).resolve().parents[1]
    _need = [
        'data/study_records/adjudicated_labels.csv',
        'legacy/data/processed/corpus.csv',
    ]
    _miss = [p for p in _need if not (_root / p).exists()]
    if _miss:
        raise SystemExit('Private-data command. Missing:\n' + '\n'.join(_miss))

    if (OUT / "point_model.bin").exists():
        raise FileExistsError("Measured output exists. Use reproduce.py with a new destination.")
    corpus = pd.read_csv(ROOT / "legacy/data/processed/corpus.csv", keep_default_na=False)
    audit = {"rows": len(corpus), "duplicate_ids": int(corpus.paper_id.duplicated().sum()),
             "duplicate_titles": int(corpus.title.map(normalized_title).duplicated().sum()),
             "discipline_support": corpus.discipline_label.value_counts().to_dict(),
             "methodology5_support": corpus.methodology_label_5.value_counts().to_dict(),
             "methodology7_support": corpus.methodology_label_7.value_counts().to_dict(),
             "original_corpus_sha256": sha256(ROOT / "legacy/data/processed/corpus.csv")}
    write_json("corpus_audit.json", audit)
    rows = []
    for key in ("discipline", "field", "methodology5", "methodology7"):
        # Preserve the archival field subset for comparison; all seven
        # methodology classes are now retained, never mislabeled as seven.
        prepared = tasks.prepare(corpus, tasks.TASKS[key],
                                 minimum_support=0 if key == "methodology7" else 15)
        result = evaluate(build_tfidf_baseline(), prepared.text.to_numpy(dtype=object),
                          prepared.labels.to_numpy(dtype=object), make_folds(prepared.labels),
                          task=key, model="tfidf")
        rows.append(result.summary())
        pd.DataFrame({"document_id": prepared.frame.paper_id, "label": result.y_true,
                      "prediction": result.y_pred}).to_csv(OUT / f"{key}_archival_oof.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "lexical_reproduction.csv", index=False)
    f = corpus[corpus.methodology_label_5.ne("")].copy()
    f = f.rename(columns={"paper_id": "document_id", "methodology_label_5": "label"})
    f["group"] = f.title.map(normalized_title)
    probabilities, folds, manifest = nested_oof(f)
    prediction = np.array(CLASSES)[probabilities.argmax(axis=1)]
    summary = {"n": len(f), "macro_f1": float(f1_score(f.label, prediction, average="macro")),
               "accuracy": float(accuracy_score(f.label, prediction)),
               "outer_folds": len(manifest), "label_reference": "archival single-reader labels"}
    write_json("nested_lexical_summary.json", summary)
    write_json("fold_manifest.json", manifest)
    oof = f[["document_id", "label"]].copy()
    oof["outer_fold"] = folds
    oof["prediction"] = prediction
    for j in range(5): oof[f"p{j}"] = probabilities[:, j]
    oof.to_csv(OUT / "nested_lexical_oof.csv", index=False)
    text = (f.title + ". " + f.abstract).to_numpy(dtype=object)
    y = f.label.to_numpy(dtype=object)
    cv = group_folds(y, f.group.to_numpy(dtype=object), requested=3)
    search = GridSearchCV(lexical_pipeline(), {"lr__C": [.3, 1., 3.]},
                          scoring="f1_macro", cv=cv, n_jobs=1, error_score="raise")
    start = time.perf_counter()
    search.fit(text, y)
    training_seconds = time.perf_counter() - start
    model_path = OUT / "point_model.bin"
    joblib.dump(search.best_estimator_, model_path)
    configuration = {"family": "tfidf-logistic", "parameters": search.best_params_,
                     "classes": list(search.classes_), "seed": SEED,
                     "label_reference": "archival single-reader labels"}
    write_json("training_manifest.json", {
        **configuration, "training_ids": f.document_id.tolist(),
        "corpus_sha256": audit["original_corpus_sha256"], "model_sha256": sha256(model_path),
        "training_fingerprint": fingerprint(f[["document_id", "title", "abstract", "label"]]
                                            .to_dict("records"), configuration),
    })
    timing = []
    rng = np.random.default_rng(SEED)
    for repeat in range(5):
        order = rng.permutation(min(30, len(text)))
        for i in order:
            start = time.perf_counter_ns()
            search.predict_proba([text[i]])
            timing.append({"repeat": repeat, "document_id": f.document_id.iloc[i],
                           "condition": "warm-process-single-record",
                           "latency_ms": (time.perf_counter_ns() - start) / 1_000_000})
    pd.DataFrame(timing).to_csv(OUT / "latency_raw.csv", index=False)
    latencies = np.array([r["latency_ms"] for r in timing])
    write_json("cost_summary.json", {
        "n_calls": len(timing), "unique_documents": 30, "repeats": 5,
        "median_ms": float(np.median(latencies)), "p95_ms": float(np.quantile(latencies, .95)),
        "sequential_throughput_per_second": 1000 / float(latencies.mean()),
        "training_and_selection_seconds": training_seconds, "model_bytes": model_path.stat().st_size,
        "process_rss_after_fit_bytes": psutil.Process().memory_info().rss,
        "rss_measurement": "snapshot after fit, not peak memory",
        "platform": platform.platform(), "python": platform.python_version(),
        "processor": platform.processor(), "logical_cpus": psutil.cpu_count(),
        "cpu_affinity_count": len(psutil.Process().cpu_affinity()),
        "threads": "1 (environment controlled)", "network_included": False,
        "mpnet": "no encoder latency or memory benchmark in this step; the executed encoding run is logged separately in outputs/model_runs/",
    })
    versions = {p: importlib.metadata.version(p) for p in
                ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "pytest", "psutil",
                 "pyarrow", "matplotlib", "PyYAML", "requests", "tqdm"]}
    write_json("environment.json", versions)
    (ROOT / "environment.lock").write_text(
        "\n".join(f"{p}=={v}" for p, v in versions.items()) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit, "nested": summary, "reproduction": rows}, indent=2))


if __name__ == "__main__":
    main()
