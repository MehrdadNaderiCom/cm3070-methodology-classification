"""Canonical locations, resolved once.

Every module imports paths from here rather than deriving them from `__file__`,
so moving a directory is a one line change and no script can quietly write
outputs somewhere the report does not look for them.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG = REPO_ROOT / "config"
DOCS = REPO_ROOT / "docs"
REPORT = REPO_ROOT / "report"

DATA = REPO_ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
DATA_LABELLED = DATA / "labelled"
DATA_SPLITS = DATA / "splits"
DATA_EXTERNAL = DATA / "external"

OUTPUTS = REPO_ROOT / "outputs"
METRICS = OUTPUTS / "metrics"
FIGURES = OUTPUTS / "figures"
PREDICTIONS = OUTPUTS / "predictions"
ERRORS = OUTPUTS / "errors"
MODELS = OUTPUTS / "models"

GOLD_V1 = DATA_LABELLED / "gold_v1_50.csv"
"""The 50 paper sample behind the Preliminary Report. Frozen, never edited."""


def ensure_output_dirs() -> None:
    """Create every output directory. Safe to call repeatedly."""
    for directory in (METRICS, FIGURES, PREDICTIONS, ERRORS, MODELS):
        directory.mkdir(parents=True, exist_ok=True)
