"""Text representations paired with a downstream classifier.

Every model here is a plain scikit-learn estimator. That is deliberate: the
evaluation harness calls the same `cross_val_predict` over the same folds for
each one, so a difference in reported numbers can only come from the model.
"""

from cdfm.models.baseline import (
    PROTOTYPE_CONFIG,
    build_majority_baseline,
    build_tfidf_baseline,
)

__all__ = ["PROTOTYPE_CONFIG", "build_majority_baseline", "build_tfidf_baseline"]
