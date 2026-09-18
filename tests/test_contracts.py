import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from study import CLASSES
from study.core import (
    agreement, align_probabilities, bootstrap_f1, clopper_pearson, conformal_fit,
    fingerprint, group_folds, load_records, matrix_f1, mcnemar, prediction_sets,
    strict_join, validate_frame, validated_probabilities, wilson,
)
from cdfm.models.embeddings import SentenceEmbedder
from cdfm.evaluation.folds import safe_split_count
from scripts.run_study_analysis import reason_for, summary_for_row


def frame(n=10):
    return pd.DataFrame({"document_id": [f"d{i}" for i in range(n)],
                         "title": ["Paper"] * n, "abstract": ["Text"] * n,
                         "group": [f"g{i}" for i in range(n)],
                         "label": [CLASSES[i % 5] for i in range(n)]})


@pytest.mark.parametrize("labels,requested,expected", [
    (["a"] * 3 + ["b"] * 3, 5, 3), (["a"] * 2 + ["b"] * 3, 5, 2),
    (["a"] * 10 + ["b"] * 10, 5, 5)])
def test_safe_split(labels, requested, expected):
    assert safe_split_count(labels, requested) == expected


@pytest.mark.parametrize("labels,requested", [(["a", "b", "b"], 5), (["a"] * 3, 5),
                                          (["a", "a", "b", "b"], 1), ([], 5)])
def test_unsafe_split_rejected(labels, requested):
    with pytest.raises(ValueError):
        safe_split_count(labels, requested)


@pytest.mark.parametrize("change", ["revision", "preprocessing_version", "corpus_fingerprint", "batch_size"])
def test_embedding_cache_identity(change):
    a, b = SentenceEmbedder(), SentenceEmbedder()
    setattr(b, change, 32 if change == "batch_size" else "changed")
    assert a._key("same text") != b._key("same text")


def test_fingerprint_covers_ids_content_and_configuration():
    records = [{"document_id": "a", "text": "x"}]
    base = fingerprint(records, {"revision": "one"})
    assert base != fingerprint([{"document_id": "b", "text": "x"}], {"revision": "one"})
    assert base != fingerprint([{"document_id": "a", "text": "y"}], {"revision": "one"})
    assert base != fingerprint(records, {"revision": "two"})


@pytest.mark.parametrize("defect", ["duplicate", "empty_id", "missing", "unknown", "empty_text", "oversize"])
def test_frame_rejects_bad_data(defect):
    f = frame()
    if defect == "duplicate": f.loc[1, "document_id"] = "d0"
    if defect == "empty_id": f.loc[0, "document_id"] = ""
    if defect == "missing": f = f.drop(columns="title")
    if defect == "unknown": f.loc[0, "label"] = "alien"
    if defect == "empty_text": f.loc[0, ["title", "abstract"]] = ["", ""]
    if defect == "oversize": f.loc[0, "abstract"] = "x" * 50001
    with pytest.raises(ValueError): validate_frame(f)


def test_file_limits_and_parser(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(frame().to_json(orient="records"), encoding="utf-8")
    assert len(load_records([p])) == 10
    with pytest.raises(ValueError): load_records([p, p])
    with pytest.raises(ValueError): load_records([p], max_bytes=2)
    p.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError): load_records([p])
    with pytest.raises(ValueError): load_records([tmp_path / "data.zip"])


@pytest.mark.parametrize("bad", [
    np.ones((2, 4)), np.ones((2, 5)), np.full((2, 5), np.nan),
    np.array([[-.1, .2, .2, .2, .5]]), np.empty((0, 5))])
def test_probability_validation(bad):
    with pytest.raises(ValueError): validated_probabilities(bad)


def test_class_order_is_checked_and_alignment_is_explicit():
    p = np.array([[.1, .2, .3, .15, .25]])
    with pytest.raises(ValueError): validated_probabilities(p, CLASSES[::-1])
    assert np.array_equal(align_probabilities(p, CLASSES[::-1]), p[:, ::-1])


def test_strict_id_join():
    a = pd.DataFrame({"document_id": ["b", "a"], "label": ["x", "y"]})
    b = pd.DataFrame({"document_id": ["a", "b"], "p": [1, 2]})
    assert strict_join(a, b).p.tolist() == [2, 1]
    with pytest.raises(ValueError): strict_join(a, b.iloc[:1])
    with pytest.raises(ValueError): strict_join(a, pd.concat([b, b]))


def test_group_disjoint_folds():
    f = frame(100)
    folds = group_folds(f.label, f.group)
    seen = []
    for tr, te in folds:
        assert not set(f.group.iloc[tr]) & set(f.group.iloc[te])
        assert set(f.label.iloc[tr]) == set(CLASSES)
        seen.extend(te)
    assert sorted(seen) == list(range(100))


def test_conformal_rank_and_ties():
    p = np.tile([.6, .1, .1, .1, .1], (60, 1))
    config, scores = conformal_fit(p, [CLASSES[0]] * 60)
    assert config["rank"] == 55
    assert config["qhat"] == pytest.approx(.4)
    assert prediction_sets(p, config).sum() == 60
    assert len(scores) == 60


def test_conformal_small_n_returns_all_classes():
    config, _ = conformal_fit(np.full((1, 5), .2), [CLASSES[0]], alpha=.01)
    assert prediction_sets(np.full((1, 5), .2), config).all()


def test_empty_sets_are_allowed_and_reviewed():
    p = np.full((1, 5), .2)
    sets = prediction_sets(p, {"qhat": .4, "classes": list(CLASSES)})
    assert sets.sum() == 0
    assert (sets.sum(axis=1) != 1)[0]


@pytest.mark.parametrize("alpha", [0, 1, -1, 2])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        conformal_fit(np.full((5, 5), .2), CLASSES, alpha)


def test_statistical_identity():
    matrix = np.diag([2, 3, 4, 5, 6])
    assert matrix_f1(matrix) == 1
    a = list(CLASSES) * 5
    result = agreement(a, a, resamples=30)
    assert result["kappa"] == 1
    assert result["ac1"] == 1
    ci, _ = bootstrap_f1(a, a, a, resamples=50)
    assert ci == [0., 0.]
    assert mcnemar(a, a, a)["p_exact"] == 1
    assert clopper_pearson(0, 10)[0] == 0
    assert clopper_pearson(10, 10)[1] == 1
    assert wilson(5, 10)[0] < .5 < wilson(5, 10)[1]


@pytest.mark.parametrize("decision_type,expected", [
    ("disagreement_resolved", "adjudicated disagreement after full-text re-read"),
    ("concordant_revised", "concordant legacy revision confirmed on re-read"),
    ("insufficient_evidence_resolved",
     "insufficient-evidence flag resolved after full-text check"),
    ("concordant_retained", "concordant agreement retained"),
])
def test_reason_for_known_decision_types(decision_type, expected):
    assert reason_for({"decision_type": decision_type}) == expected


def test_reason_for_missing_decision_type_raises():
    with pytest.raises(ValueError):
        reason_for({})


def test_reason_for_none_decision_type_raises():
    with pytest.raises(ValueError):
        reason_for({"decision_type": None})


def test_reason_for_unknown_decision_type_raises():
    with pytest.raises(ValueError):
        reason_for({"decision_type": "UNKNOWN_VALUE"})


def test_summary_for_row_in_log_known():
    assert summary_for_row({"in_log": True, "decision_type": "disagreement_resolved"}) == \
        "adjudicated disagreement after full-text re-read"
    assert summary_for_row({"in_log": True, "decision_type": "concordant_retained"}) == \
        "concordant agreement retained"


def test_summary_for_row_in_log_missing_raises():
    with pytest.raises(ValueError):
        summary_for_row({"in_log": True, "decision_type": None})
    with pytest.raises(ValueError):
        summary_for_row({"in_log": True, "decision_type": ""})
    with pytest.raises(ValueError):
        summary_for_row({"in_log": True, "decision_type": "UNKNOWN"})


def test_summary_for_row_not_in_log_concordant():
    assert summary_for_row({"in_log": False, "A_label": "X", "B_label": "X",
                            "final_label": "X"}) == "concordant agreement retained"


def test_summary_for_row_not_in_log_mismatch_unknown():
    assert summary_for_row({"in_log": False, "A_label": "X", "B_label": "Y",
                            "final_label": "X"}) == "unknown decision category"
    assert summary_for_row({"in_log": False, "A_label": None, "B_label": "X",
                            "final_label": "X"}) == "unknown decision category"
