"""Guard the numbers that already appear in the submitted Preliminary Report.

Chapter 4 of that report states an accuracy of 0.860, a macro F1 of 0.862, a
majority floor of 0.200, a per class table and a confusion matrix. Those figures
are now fixed in a document that cannot be edited. If a refactor, a library
upgrade or a change of preprocessing moves them, the final report would quietly
contradict the preliminary one.

This module fails loudly instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from cdfm import RANDOM_STATE
from cdfm.models import build_majority_baseline, build_tfidf_baseline
from cdfm.paths import GOLD_V1

PUBLISHED_MAJORITY = 0.200
PUBLISHED_ACCURACY = 0.860
PUBLISHED_MACRO_F1 = 0.862
PUBLISHED_ERRORS = 7

PUBLISHED_PER_CLASS = {
    "Case Study": (1.00, 0.88, 0.93, 8),
    "Conceptual": (0.78, 0.88, 0.82, 8),
    "Design/Engineering": (0.89, 0.80, 0.84, 10),
    "Experimental": (0.83, 0.83, 0.83, 12),
    "Review": (0.85, 0.92, 0.88, 12),
}

PUBLISHED_CONFUSION = np.array(
    [
        [7, 1, 0, 0, 0],
        [0, 7, 0, 0, 1],
        [0, 0, 8, 1, 1],
        [0, 1, 1, 10, 0],
        [0, 0, 0, 1, 11],
    ]
)


@pytest.fixture(scope="module")
def prototype_predictions():
    """Rerun the prototype exactly as Chapter 4 describes it."""
    if not GOLD_V1.exists():
        pytest.skip("gold_v1_50.csv is private and is not shipped in the public tree")
    frame = pd.read_csv(GOLD_V1)
    frame = frame.dropna(subset=["title", "abstract", "methodology_label"]).reset_index(drop=True)
    text = (frame["title"].fillna("") + ". " + frame["abstract"].fillna("")).str.strip()

    features = text.to_numpy(dtype=object)
    labels = frame["methodology_label"].to_numpy(dtype=object)

    smallest_class = int(frame["methodology_label"].value_counts().min())
    folds = min(5, smallest_class)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    majority = cross_val_predict(build_majority_baseline(), features, labels, cv=splitter)
    predicted = cross_val_predict(build_tfidf_baseline(), features, labels, cv=splitter)

    return {
        "labels": labels,
        "majority": majority,
        "predicted": predicted,
        "folds": folds,
        "classes": sorted(frame["methodology_label"].unique()),
    }


def test_sample_is_the_published_fifty(prototype_predictions):
    assert len(prototype_predictions["labels"]) == 50


def test_fold_count_matches_the_report(prototype_predictions):
    assert prototype_predictions["folds"] == 5


def test_majority_floor(prototype_predictions):
    observed = accuracy_score(prototype_predictions["labels"], prototype_predictions["majority"])
    assert round(observed, 3) == PUBLISHED_MAJORITY


def test_headline_accuracy(prototype_predictions):
    observed = accuracy_score(prototype_predictions["labels"], prototype_predictions["predicted"])
    assert round(observed, 3) == PUBLISHED_ACCURACY


def test_headline_macro_f1(prototype_predictions):
    observed = f1_score(
        prototype_predictions["labels"], prototype_predictions["predicted"], average="macro"
    )
    assert round(observed, 3) == PUBLISHED_MACRO_F1


def test_error_count(prototype_predictions):
    errors = int((prototype_predictions["labels"] != prototype_predictions["predicted"]).sum())
    assert errors == PUBLISHED_ERRORS


@pytest.mark.parametrize("class_name", sorted(PUBLISHED_PER_CLASS))
def test_per_class_table(prototype_predictions, class_name):
    classes = prototype_predictions["classes"]
    precision, recall, f1, support = precision_recall_fscore_support(
        prototype_predictions["labels"],
        prototype_predictions["predicted"],
        labels=classes,
        zero_division=0,
    )
    index = classes.index(class_name)
    expected = PUBLISHED_PER_CLASS[class_name]
    observed = (
        round(float(precision[index]), 2),
        round(float(recall[index]), 2),
        round(float(f1[index]), 2),
        int(support[index]),
    )
    assert observed == expected


def test_confusion_matrix_matches_figure_4_1(prototype_predictions):
    observed = confusion_matrix(
        prototype_predictions["labels"],
        prototype_predictions["predicted"],
        labels=prototype_predictions["classes"],
    )
    assert np.array_equal(observed, PUBLISHED_CONFUSION)
