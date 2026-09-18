"""The record schema, and the rules that decide whether a dataset is valid.

Every column that a task depends on is declared here with the provenance of its
values, because a label's worth depends entirely on where it came from. A
discipline inherited from a journal's identity and a methodology assigned by a
person reading an abstract are different kinds of claim, and the schema records
which is which.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Provenance values. Anything reported as ground truth must say how it was made.
PROVENANCE_VENUE = "venue"
PROVENANCE_SOURCE_METADATA = "source_metadata"
PROVENANCE_HUMAN = "human"
PROVENANCE_PROPOSED = "proposed"
PROVENANCES = (
    PROVENANCE_VENUE,
    PROVENANCE_SOURCE_METADATA,
    PROVENANCE_HUMAN,
    PROVENANCE_PROPOSED,
)

# How a methodology label came to exist.
#
# `human` means the author assigned the label from the abstract and the written
# guide. Every methodology label in the current corpus was produced that way.
#
# `proposed` means the cue matching script guessed and nobody looked. It is a hard
# error wherever a label is expected.
HUMAN_VERIFIED_PROVENANCES = (PROVENANCE_HUMAN,)
MACHINE_DERIVED_PROVENANCES = (PROVENANCE_PROPOSED,)

IDENTITY_COLUMNS = ["paper_id", "source", "title", "abstract", "year"]

CONTEXT_COLUMNS = ["venue", "venue_id", "keywords", "primary_category", "topic", "subfield", "doi"]

LABEL_COLUMNS = [
    "discipline_label",
    "discipline_source",
    "field_label",
    "field_source",
    "methodology_label_7",
    "methodology_label_5",
    "methodology_dominant",
    "methodology_source",
]

ANNOTATION_COLUMNS = ["annotator", "annotated_at", "poor_fit", "notes"]

ALL_COLUMNS = IDENTITY_COLUMNS + CONTEXT_COLUMNS + LABEL_COLUMNS + ANNOTATION_COLUMNS


@dataclass
class ValidationReport:
    """What a dataset check found. Empty errors means the dataset can be used.

    Notes are neither problems nor near problems. They record facts about the
    dataset that any writing about it has to get right, provenance above all.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = []
        for message in self.errors:
            lines.append(f"error: {message}")
        for message in self.warnings:
            lines.append(f"warning: {message}")
        for message in self.notes:
            lines.append(f"note: {message}")
        if not lines:
            lines.append("dataset valid")
        return "\n".join(lines)


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ALL_COLUMNS)


def conform(frame: pd.DataFrame) -> pd.DataFrame:
    """Add any missing columns as empty and put them in declared order."""
    result = frame.copy()
    for column in ALL_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    extra = [c for c in result.columns if c not in ALL_COLUMNS]
    return result[ALL_COLUMNS + extra]


def derive_five_class(label_7: str, dominant: str, collapse: dict[str, str]) -> str:
    """Map a seven class methodology label onto the five class set.

    Mixed Methods has no direct target. The Preliminary Report folds it into its
    dominant component, so the dominant label is mapped instead. A Mixed Methods
    row with no dominant recorded yields an empty string rather than a guess.
    """
    if not label_7:
        return ""
    target = collapse.get(label_7, "")
    if target != "DOMINANT":
        return target
    if not dominant:
        return ""
    return collapse.get(dominant, "")


def validate(
    frame: pd.DataFrame,
    *,
    minimum_class_support: int = 15,
    require_methodology: bool = False,
) -> ValidationReport:
    """Check a dataset against the rules that make its results meaningful.

    `minimum_class_support` is the point below which per class precision and
    recall stop being measurements. Fifteen is the figure recorded in R02 of the
    risk register.
    """
    report = ValidationReport()

    missing_columns = [c for c in ALL_COLUMNS if c not in frame.columns]
    if missing_columns:
        report.errors.append(f"missing columns: {', '.join(missing_columns)}")
        return report

    if frame.empty:
        report.errors.append("dataset is empty")
        return report

    duplicated = frame["paper_id"].duplicated().sum()
    if duplicated:
        report.errors.append(f"{duplicated} duplicate paper_id values")

    for column in ("title", "abstract"):
        blank = frame[column].fillna("").str.strip().eq("").sum()
        if blank:
            report.errors.append(f"{blank} rows have an empty {column}")

    short = frame["abstract"].fillna("").str.len().lt(300).sum()
    if short:
        report.warnings.append(f"{short} abstracts shorter than 300 characters")

    for label_column, source_column in (
        ("discipline_label", "discipline_source"),
        ("field_label", "field_source"),
        ("methodology_label_7", "methodology_source"),
        # The five class column is checked separately from the seven class one.
        # The prototype's 50 papers carry a five class label and no seven class
        # label, so checking only the finer column would let a label through
        # with no record of where it came from.
        ("methodology_label_5", "methodology_source"),
    ):
        labelled = frame[label_column].fillna("").str.strip().ne("")
        missing_source = labelled & frame[source_column].fillna("").str.strip().eq("")
        if missing_source.any():
            report.errors.append(
                f"{int(missing_source.sum())} rows carry a {label_column} with no {source_column}"
            )
        bad_source = frame.loc[
            frame[source_column].fillna("").str.strip().ne(""), source_column
        ].isin(PROVENANCES)
        if not bad_source.all():
            report.errors.append(f"{source_column} contains values outside {PROVENANCES}")

    methodology = frame.loc[
        frame["methodology_label_7"].fillna("").str.strip().ne(""), "methodology_source"
    ]
    if (methodology == PROVENANCE_PROPOSED).any():
        report.errors.append(
            f"{int((methodology == PROVENANCE_PROPOSED).sum())} methodology labels are still "
            "machine proposals. A proposal is not a label and must not be reported as one"
        )

    # Stated on every run so that any writing about the corpus gets provenance
    # right without anyone having to reconstruct it from memory.
    verified = int(methodology.isin(HUMAN_VERIFIED_PROVENANCES).sum())
    if verified:
        report.notes.append(
            f"all {verified} of {int(len(methodology))} methodology labels were "
            "annotated and verified by the author"
        )

    if require_methodology:
        unlabelled = frame["methodology_label_7"].fillna("").str.strip().eq("").sum()
        if unlabelled:
            report.errors.append(f"{unlabelled} rows have no methodology label")

    for column in ("discipline_label", "field_label", "methodology_label_5"):
        counts = frame.loc[frame[column].fillna("").str.strip().ne(""), column].value_counts()
        thin = counts[counts < minimum_class_support]
        for name, count in thin.items():
            report.warnings.append(
                f"{column} class {name!r} holds {count} rows, below the {minimum_class_support} "
                "needed for per class metrics to mean anything"
            )

    return report
