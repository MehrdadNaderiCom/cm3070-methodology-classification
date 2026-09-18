"""Classify computing research publications by discipline, field and methodology.

The package is organised around one idea: every task shares the same records,
the same folds and the same evaluation code, so that differences between
reported numbers are differences between models rather than between harnesses.

    cdfm.data        acquisition, schema, freezing and preprocessing
    cdfm.models      text representations paired with a classifier
    cdfm.tasks       the three classification tasks and their label sources
    cdfm.evaluation  folds, metrics, confusion, error export, cost measurement
"""

__version__ = "0.1.0"

RANDOM_STATE = 42
"""Single seed for every stochastic step in the project.

Imported rather than redefined so that no module can quietly disagree.
"""
