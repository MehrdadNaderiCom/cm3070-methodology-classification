"""The lexical baseline: TF-IDF features with logistic regression.

This is the fixed reference point for the whole project. Every later approach is
measured against it on identical folds, so its configuration is frozen here and
changed only through a decision log entry.

`PROTOTYPE_CONFIG` records the exact settings used for the feature prototype
reported in the Preliminary Report. `tests/test_prototype_parity.py` asserts that
those settings still reproduce the published figures.
"""

from __future__ import annotations

from types import MappingProxyType

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cdfm import RANDOM_STATE

PROTOTYPE_CONFIG = MappingProxyType(
    {
        "lowercase": True,
        "stop_words": "english",
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_iter": 1000,
        "class_weight": "balanced",
    }
)
"""Settings behind the published prototype figures. Read only on purpose.

Changing any value here changes numbers that already appear in a submitted
document, so the change belongs in the decision log before it belongs in code.
"""


def build_tfidf_baseline(config: dict | None = None, random_state: int = RANDOM_STATE) -> Pipeline:
    """Return the TF-IDF and logistic regression pipeline.

    Args:
        config: Overrides applied on top of `PROTOTYPE_CONFIG`. Passing anything
            here means the result is no longer the published baseline, which is
            why the default is to pass nothing.
        random_state: Seed for the solver. Shared across the project.
    """
    settings = dict(PROTOTYPE_CONFIG)
    if config:
        settings.update(config)

    vectoriser = TfidfVectorizer(
        lowercase=settings["lowercase"],
        stop_words=settings["stop_words"],
        ngram_range=tuple(settings["ngram_range"]),
        min_df=settings["min_df"],
    )
    classifier = LogisticRegression(
        max_iter=settings["max_iter"],
        class_weight=settings["class_weight"],
        random_state=random_state,
    )
    return Pipeline([("tfidf", vectoriser), ("lr", classifier)])


def build_majority_baseline() -> DummyClassifier:
    """Return the floor that any reported model has to clear.

    Quoting an accuracy without this number next to it says nothing. On a five
    class problem with the largest class at 24 percent, an accuracy of 0.86 is a
    strong result; on a two class problem it could be worse than guessing.
    """
    return DummyClassifier(strategy="most_frequent")
