"""Fold generation, shared by every task and every model.

The comparison in this project is only meaningful if the models are scored on
identical splits. Folds are therefore produced here, from the labels and the
project seed, and passed to whichever estimator is being evaluated. No model
builds its own.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from cdfm import RANDOM_STATE

DEFAULT_SPLITS = 5


def safe_split_count(labels, requested: int = DEFAULT_SPLITS) -> int:
    """Largest usable fold count: never more than the smallest class holds.

    Stratified splitting needs at least one example of every class in every
    fold. Asking for five folds when a class has three examples raises, and
    silently dropping to three folds would make two models incomparable if they
    were run at different times against different data.
    """
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    if len(values) < 2:
        raise ValueError("stratified evaluation needs at least two classes")
    if requested < 2:
        raise ValueError("requested folds must be at least two")
    if int(counts.min()) < 2:
        raise ValueError("singleton class: stratified evaluation is not possible")
    return min(requested, int(counts.min()))


def make_folds(labels, requested: int = DEFAULT_SPLITS, seed: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=safe_split_count(labels, requested), shuffle=True, random_state=seed
    )


def describe(labels, requested: int = DEFAULT_SPLITS) -> str:
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    splits = safe_split_count(labels, requested)
    smallest = values[int(np.argmin(counts))]
    return (
        f"{len(values)} classes, {len(labels)} rows, smallest class "
        f"{smallest!r} at {int(counts.min())}, using {splits} folds"
    )
