"""The three classification tasks, and how a corpus is prepared for each.

A task is a label column plus the rules for deciding which rows can legitimately
be used with it. Keeping those rules here, rather than in whichever script runs
first, is what stops two experiments from quietly evaluating on different subsets
and being reported side by side as though they were comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MINIMUM_SUPPORT = 15
"""Below this, per class precision and recall stop being measurements.

Recorded in R02 of the risk register. A class with five examples under five fold
cross validation contributes one test item per fold, and its reported F1 can only
take a handful of values.
"""


@dataclass(frozen=True)
class Task:
    key: str
    label_column: str
    title: str
    description: str


TASKS: dict[str, Task] = {
    "discipline": Task(
        key="discipline",
        label_column="discipline_label",
        title="Discipline",
        description="Computer Science, Information Systems or Information Technology, "
        "inherited from the venue the paper appeared in.",
    ),
    "field": Task(
        key="field",
        label_column="field_label",
        title="Field",
        description="Computing field, mapped from arXiv primary category and OpenAlex "
        "topic subfield by config/field_mapping.yaml.",
    ),
    "methodology5": Task(
        key="methodology5",
        label_column="methodology_label_5",
        title="Methodology, five class",
        description="The reduced taxonomy used by the feature prototype.",
    ),
    "methodology7": Task(
        key="methodology7",
        label_column="methodology_label_7",
        title="Methodology, seven class",
        description="The full taxonomy named in the project design, from which the "
        "five class labels are derived.",
    ),
}


def build_text(frame: pd.DataFrame) -> pd.Series:
    """Title and abstract joined, exactly as the prototype built its input.

    Any change here changes the published prototype numbers, which is why the
    parity tests call this path.
    """
    title = frame["title"].fillna("").astype(str)
    abstract = frame["abstract"].fillna("").astype(str)
    return (title + ". " + abstract).str.strip()


@dataclass
class Prepared:
    task: Task
    frame: pd.DataFrame
    text: pd.Series
    labels: pd.Series
    dropped_classes: dict[str, int]
    under_supported: dict[str, int]

    @property
    def usable(self) -> bool:
        return len(self.labels.unique()) >= 2 and len(self.frame) >= 20

    def unusable_reason(self) -> str:
        if len(self.frame) == 0:
            return "no labelled rows"
        if len(self.labels.unique()) < 2:
            return f"{len(self.frame)} labelled rows but only one class"
        return f"{len(self.frame)} labelled rows, fewer than the 20 needed"

    def note(self) -> str:
        lines = []
        if self.dropped_classes:
            parts = ", ".join(
                f"{name} ({count})" for name, count in sorted(self.dropped_classes.items())
            )
            lines.append(
                f"excluded {len(self.dropped_classes)} class(es) below {MINIMUM_SUPPORT} "
                f"examples: {parts}"
            )
        if self.under_supported:
            parts = ", ".join(
                f"{name} ({count})" for name, count in sorted(self.under_supported.items())
            )
            lines.append(
                f"every class sits below {MINIMUM_SUPPORT} examples, so nothing was excluded "
                f"and per class figures carry wide uncertainty: {parts}"
            )
        return "; ".join(lines)


def prepare(corpus: pd.DataFrame, task: Task, minimum_support: int = MINIMUM_SUPPORT) -> Prepared:
    """Select the rows a task can honestly be evaluated on.

    Rows with no label for this task are dropped. Classes too small to report are
    dropped as well, but only while at least two classes survive the cut.

    That exception matters. The feature prototype's five class methodology task
    has no class above twelve examples, and filtering on support would empty it
    and take the published numbers with it. Refusing to evaluate a small dataset
    is not more honest than evaluating it and stating the caveat, so the thin
    classes are reported rather than removed.
    """
    frame = corpus[corpus[task.label_column].fillna("").astype(str).str.strip().ne("")].copy()

    dropped: dict[str, int] = {}
    under_supported: dict[str, int] = {}

    if not frame.empty:
        counts = frame[task.label_column].value_counts()
        thin = counts[counts < minimum_support]
        survivors = counts[counts >= minimum_support]

        if len(thin) and len(survivors) >= 2:
            dropped = {str(name): int(count) for name, count in thin.items()}
            frame = frame[~frame[task.label_column].isin(thin.index)]
        elif len(thin):
            under_supported = {str(name): int(count) for name, count in thin.items()}

    frame = frame.reset_index(drop=True)
    return Prepared(
        task=task,
        frame=frame,
        text=build_text(frame),
        labels=frame[task.label_column].astype(str),
        dropped_classes=dropped,
        under_supported=under_supported,
    )
