"""Strict contracts, group-aware evaluation and split conformal prediction."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from . import CLASSES, SEED


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(records, configuration):
    payload = {"records": records, "configuration": configuration, "schema": 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def normalized_title(text):
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).casefold()).split())


def validate_frame(frame, *, require_labels=True, max_rows=1000):
    needed = {"document_id", "title", "abstract", "group"}
    if require_labels:
        needed.add("label")
    if not needed <= set(frame.columns):
        raise ValueError(f"missing columns: {sorted(needed - set(frame.columns))}")
    if not 1 <= len(frame) <= max_rows:
        raise ValueError("row count outside allowed limits")
    for key in ("document_id", "group"):
        if frame[key].isna().any() or frame[key].astype(str).str.strip().eq("").any():
            raise ValueError(f"empty {key}")
    if frame.document_id.duplicated().any():
        raise ValueError("duplicate document ID")
    for key in ("title", "abstract"):
        if not frame[key].map(lambda v: isinstance(v, str)).all():
            raise ValueError(f"{key} must contain strings")
        if frame[key].str.len().gt(50_000).any():
            raise ValueError(f"{key} exceeds text limit")
    if (frame.title.str.strip() + frame.abstract.str.strip()).eq("").any():
        raise ValueError("empty title and abstract")
    if require_labels and not set(frame.label) <= set(CLASSES):
        raise ValueError("unknown or missing label")
    return frame


def load_records(paths, *, require_labels=False, max_bytes=5_000_000):
    if not 1 <= len(paths) <= 1:
        raise ValueError("exactly one input file is permitted")
    path = Path(paths[0])
    if path.is_symlink() or not path.is_file():
        raise ValueError("input must be a regular non-symlink file")
    if path.suffix.lower() not in {".csv", ".json"}:
        raise ValueError("only uncompressed CSV or JSON is accepted")
    if path.stat().st_size > max_bytes:
        raise ValueError("input exceeds byte limit")
    try:
        if path.suffix.lower() == ".json":
            rows = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(rows, dict):
                rows = [rows]
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                raise ValueError("JSON must contain records")
            frame = pd.DataFrame(rows)
        else:
            frame = pd.read_csv(path, keep_default_na=False)
    except (UnicodeError, pd.errors.ParserError, json.JSONDecodeError) as exc:
        raise ValueError("malformed UTF-8 input") from exc
    return validate_frame(frame, require_labels=require_labels)


def validated_probabilities(values, classes=CLASSES):
    p = np.asarray(values, dtype=float)
    if tuple(classes) != CLASSES:
        raise ValueError("class order does not match the frozen contract")
    if p.ndim != 2 or p.shape[1] != len(classes) or not len(p):
        raise ValueError("probability shape mismatch")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("invalid probability values")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0):
        raise ValueError("probabilities must sum to one")
    return p


def align_probabilities(values, estimator_classes):
    if set(estimator_classes) != set(CLASSES) or len(estimator_classes) != len(CLASSES):
        raise ValueError("fitted model must contain all five classes")
    order = [list(estimator_classes).index(c) for c in CLASSES]
    return validated_probabilities(np.asarray(values)[:, order])


def strict_join(labels, predictions):
    for f in (labels, predictions):
        if f.document_id.isna().any() or f.document_id.duplicated().any():
            raise ValueError("duplicate or missing ID")
    if set(labels.document_id) != set(predictions.document_id):
        raise ValueError("ID sets differ")
    return labels.merge(predictions, on="document_id", validate="one_to_one")


def group_folds(y, groups, requested=5, seed=SEED):
    y, groups = np.asarray(y), np.asarray(groups)
    if len(y) != len(groups) or requested < 2 or len(set(y)) < 2:
        raise ValueError("invalid fold inputs")
    support = [len(set(groups[y == c])) for c in set(y)]
    k = min(requested, min(support))
    if k < 2:
        raise ValueError("singleton class/group cannot be split")
    # Group constraints can make nominal support insufficient in a particular
    # assignment. Reduce k and validate each realised fold, never silently leak.
    for count in range(k, 1, -1):
        splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.zeros(len(y)), y, groups))
        if all(set(y[tr]) == set(y) and set(y[te]) == set(y)
               and not (set(groups[tr]) & set(groups[te])) for tr, te in folds):
            return folds
    raise ValueError("no class-complete group-disjoint split is available")


def lexical_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ])


def nested_oof(frame):
    validate_frame(frame)
    text = (frame.title + ". " + frame.abstract).to_numpy(dtype=object)
    y = frame.label.to_numpy(dtype=object)
    groups = frame.group.to_numpy(dtype=object)
    outer = group_folds(y, groups)
    p = np.zeros((len(y), 5))
    fold_ids = np.full(len(y), -1, dtype=int)
    manifest = []
    for fold, (tr, te) in enumerate(outer):
        inner = group_folds(y[tr], groups[tr], requested=3, seed=SEED + fold)
        search = GridSearchCV(lexical_pipeline(), {"lr__C": [0.3, 1.0, 3.0]},
                              scoring="f1_macro", cv=inner, n_jobs=1, error_score="raise")
        search.fit(text[tr], y[tr])
        p[te] = align_probabilities(search.predict_proba(text[te]), search.classes_)
        fold_ids[te] = fold
        manifest.append({"outer_fold": fold, "train_ids": frame.document_id.iloc[tr].tolist(),
                         "test_ids": frame.document_id.iloc[te].tolist(),
                         "best_params": search.best_params_,
                         "inner_folds": [{"train_ids": frame.document_id.iloc[tr[a]].tolist(),
                                          "test_ids": frame.document_id.iloc[tr[b]].tolist()}
                                         for a, b in inner]})
    assert (fold_ids >= 0).all()
    return p, fold_ids, manifest


def conformal_fit(probabilities, labels, alpha=0.1):
    p = validated_probabilities(probabilities)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if len(labels) != len(p) or not set(labels) <= set(CLASSES):
        raise ValueError("calibration labels do not match probabilities")
    indexes = np.array([CLASSES.index(c) for c in labels])
    scores = np.sort(1 - p[np.arange(len(p)), indexes])
    rank = math.ceil((len(p) + 1) * (1 - alpha))
    # Score range is [0,1]; using 1 at rank n+1 includes every class.
    q = 1.0 if rank > len(scores) else float(scores[rank - 1])
    return {"alpha": alpha, "n": len(p), "rank": rank, "qhat": q,
            "classes": list(CLASSES), "tie_rule": "include when score <= qhat",
            "review_rule": "set_size != 1"}, scores


def prediction_sets(probabilities, configuration):
    p = validated_probabilities(probabilities, configuration["classes"])
    q = float(configuration["qhat"])
    if not np.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("invalid conformal threshold")
    return (1 - p) <= q


def wilson(successes, n):
    if not 0 <= successes <= n or n < 1:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    phat = successes / n
    d = 1 + z * z / n
    middle = (phat + z * z / (2 * n)) / d
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / d
    return [middle - half, middle + half]


def clopper_pearson(k, n):
    if not 0 <= k <= n or n < 1:
        raise ValueError("invalid binomial counts")
    return [0.0 if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(.975, k + 1, n - k))]


def matrix_f1(matrix):
    c = np.asarray(matrix, dtype=float)
    tp = np.diagonal(c, axis1=-2, axis2=-1)
    denominator = c.sum(axis=-1) + c.sum(axis=-2)
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp),
                     where=denominator > 0).mean(axis=-1)


def bootstrap_f1(y, pred_a, pred_b=None, domains=None, resamples=10_000, seed=SEED):
    y = np.asarray([CLASSES.index(c) for c in y])
    a = np.asarray([CLASSES.index(c) for c in pred_a])
    b = None if pred_b is None else np.asarray([CLASSES.index(c) for c in pred_b])
    if not len(y) or len(y) != len(a) or (b is not None and len(b) != len(y)):
        raise ValueError("bootstrap row mismatch")
    if domains is not None and len(domains) != len(y):
        raise ValueError("bootstrap domains mismatch")
    rng = np.random.default_rng(seed)
    groups = [np.arange(len(y))] if domains is None else [
        np.flatnonzero(np.asarray(domains) == d) for d in sorted(set(domains))]
    values = []
    for start in range(0, resamples, 500):
        size = min(500, resamples - start)
        ix = np.concatenate([rng.choice(g, (size, len(g)), replace=True) for g in groups], axis=1)
        cm_a = np.zeros((size, 25), dtype=int)
        np.add.at(cm_a, (np.arange(size)[:, None], y[ix] * 5 + a[ix]), 1)
        score = matrix_f1(cm_a.reshape(size, 5, 5))
        if b is not None:
            cm_b = np.zeros((size, 25), dtype=int)
            np.add.at(cm_b, (np.arange(size)[:, None], y[ix] * 5 + b[ix]), 1)
            score -= matrix_f1(cm_b.reshape(size, 5, 5))
        values.extend(score.tolist())
    return np.quantile(values, [.025, .975]).tolist(), np.asarray(values)


def mcnemar(y, a, b):
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    aw = int(((a == y) & (b != y)).sum())
    bw = int(((a != y) & (b == y)).sum())
    return {"a_only_correct": aw, "b_only_correct": bw,
            "p_exact": float(binomtest(aw, aw + bw, .5).pvalue) if aw + bw else 1.0}


def agreement(a, b, resamples=2000, seed=SEED):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or not len(a):
        raise ValueError("agreement rows mismatch")
    n = len(a)
    counts_a = np.array([(a == c).mean() for c in CLASSES])
    counts_b = np.array([(b == c).mean() for c in CLASSES])
    po = float((a == b).mean())
    pe = float(counts_a @ counts_b)
    pooled = (counts_a + counts_b) / 2
    pe_ac1 = float((pooled * (1 - pooled)).sum() / (len(CLASSES) - 1))
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(resamples):
        ix = rng.integers(n, size=n)
        aa, bb = a[ix], b[ix]
        e = sum((aa == c).mean() * (bb == c).mean() for c in CLASSES)
        if e < 1:
            ks.append(((aa == bb).mean() - e) / (1 - e))
    return {"n": n, "agree": int((a == b).sum()), "observed": po,
            "observed_ci": wilson(int((a == b).sum()), n),
            "kappa": None if pe == 1 else (po - pe) / (1 - pe),
            "kappa_ci": np.quantile(ks, [.025, .975]).tolist() if ks else [None, None],
            "ac1": (po - pe_ac1) / (1 - pe_ac1),
            "positive_agreement": {c: (2 * int(((a == c) & (b == c)).sum()) /
                                      int((a == c).sum() + (b == c).sum())
                                      if int((a == c).sum() + (b == c).sum()) else None)
                                   for c in CLASSES}}
