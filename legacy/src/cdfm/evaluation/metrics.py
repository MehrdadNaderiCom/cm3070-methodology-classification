"""Scoring, reporting and persistence for one model on one task.

Every number the report quotes is produced here and written to a file. Nothing
is typed into prose by hand, which is the only way to stop a written result from
drifting away from what the code currently does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_val_predict

from cdfm.models import build_majority_baseline


@dataclass
class TaskResult:
    """What one model scored on one task, with everything needed to report it."""

    task: str
    model: str
    labels: list[str]
    y_true: np.ndarray
    y_pred: np.ndarray
    n_splits: int
    majority_accuracy: float = 0.0
    majority_macro_f1: float = 0.0
    cost: dict = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return float(accuracy_score(self.y_true, self.y_pred))

    @property
    def macro_f1(self) -> float:
        return float(f1_score(self.y_true, self.y_pred, average="macro", zero_division=0))

    @property
    def error_count(self) -> int:
        return int((self.y_true != self.y_pred).sum())

    def per_class(self) -> pd.DataFrame:
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, labels=self.labels, zero_division=0
        )
        return pd.DataFrame(
            {
                "class": self.labels,
                "precision": precision.round(3),
                "recall": recall.round(3),
                "f1": f1.round(3),
                "support": support,
            }
        )

    def confusion(self) -> pd.DataFrame:
        matrix = confusion_matrix(self.y_true, self.y_pred, labels=self.labels)
        return pd.DataFrame(matrix, index=self.labels, columns=self.labels)

    def summary(self) -> dict:
        return {
            "task": self.task,
            "model": self.model,
            "rows": int(len(self.y_true)),
            "classes": len(self.labels),
            "folds": self.n_splits,
            "majority_accuracy": round(self.majority_accuracy, 3),
            "majority_macro_f1": round(self.majority_macro_f1, 3),
            "accuracy": round(self.accuracy, 3),
            "macro_f1": round(self.macro_f1, 3),
            "errors": self.error_count,
            **{f"cost_{key}": value for key, value in self.cost.items()},
        }

    def to_markdown(self) -> str:
        lines = [
            f"## {self.task}, {self.model}",
            "",
            f"{len(self.y_true)} papers, {len(self.labels)} classes, "
            f"{self.n_splits}-fold stratified cross validation.",
            "",
            "| Metric | Majority floor | Model |",
            "| --- | ---: | ---: |",
            f"| Accuracy | {self.majority_accuracy:.3f} | {self.accuracy:.3f} |",
            f"| Macro F1 | {self.majority_macro_f1:.3f} | {self.macro_f1:.3f} |",
            "",
            f"Misclassified: {self.error_count} of {len(self.y_true)}.",
            "",
            "The floor is a classifier that always predicts the largest class. Its "
            "accuracy and its macro F1 differ sharply, so each model figure is read "
            "against the floor in its own units.",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in self.per_class().itertuples(index=False):
            lines.append(
                f"| {row._0} | {row.precision:.2f} | {row.recall:.2f} | "
                f"{row.f1:.2f} | {row.support} |"
            )
        return "\n".join(lines)


def evaluate(
    estimator,
    features,
    labels,
    folds,
    *,
    task: str,
    model: str,
) -> TaskResult:
    """Score an estimator by pooled cross validated prediction.

    Pooling the out of fold predictions and scoring once, rather than averaging
    per fold scores, is what the prototype did and what its published numbers
    describe. It also keeps per class support meaningful on a small dataset,
    where a fold can hold two examples of a class.
    """
    labels = np.asarray(labels)
    features = np.asarray(features)
    predicted = cross_val_predict(estimator, features, labels, cv=folds)
    majority = cross_val_predict(build_majority_baseline(), features, labels, cv=folds)

    return TaskResult(
        task=task,
        model=model,
        labels=sorted(set(labels.tolist())),
        y_true=labels,
        y_pred=predicted,
        n_splits=folds.get_n_splits(),
        majority_accuracy=float(accuracy_score(labels, majority)),
        # Both floors are recorded because they are far apart. A majority
        # classifier on three balanced-ish classes scores 0.391 accuracy and
        # 0.188 macro F1, so quoting a model's macro F1 against the accuracy
        # floor overstates the margin by a factor of two.
        majority_macro_f1=float(f1_score(labels, majority, average="macro", zero_division=0)),
    )


def save(result: TaskResult, metrics_dir: Path, errors_dir: Path, frame: pd.DataFrame) -> None:
    """Write every artefact the report will cite for this result."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result.task}__{result.model}".replace("/", "-").replace(" ", "_")

    (metrics_dir / f"{stem}.json").write_text(
        json.dumps(result.summary(), indent=2) + "\n", encoding="utf-8"
    )
    (metrics_dir / f"{stem}.md").write_text(result.to_markdown() + "\n", encoding="utf-8")
    result.per_class().to_csv(metrics_dir / f"{stem}__per_class.csv", index=False, encoding="utf-8")
    result.confusion().to_csv(metrics_dir / f"{stem}__confusion.csv", encoding="utf-8")

    wrong = result.y_true != result.y_pred
    if wrong.any():
        errors = frame.loc[wrong, ["paper_id", "title", "discipline_label", "venue"]].copy()
        errors["true"] = result.y_true[wrong]
        errors["predicted"] = result.y_pred[wrong]
        errors.to_csv(errors_dir / f"{stem}__errors.csv", index=False, encoding="utf-8")


def comparison_table(results: list[TaskResult]) -> pd.DataFrame:
    """One row per model per task, for the side by side table in the report."""
    return pd.DataFrame([result.summary() for result in results])
