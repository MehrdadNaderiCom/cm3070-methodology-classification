# Computing research methodology classification

CM3070 BSc Computer Science final project. This is the **public** repository tree.

Code is MIT-licensed. Paper **abstracts are not published here**. The public tables contain identifiers, titles, DOIs, splits and labels only. Full study-path reproduction (raw forms, abstracts, archival model, embeddings) is not part of this repository.

## What you can run here

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r environment.lock
python -m pip install -e .
python -m pytest
python scripts/classify.py --file data/demo_input.json
python scripts/verify_release.py
python scripts/demo_review.py
```

`classify.py --file` uses a small **authored demo model** in `data/demo/`, not the private archival classifier.

`classify.py --case-id EXT000` prints a **stored** Cue row from `outputs/public/test_row_results.csv`. It is not a fresh prediction.

Public tests: `python -m pytest` runs 46 contract tests. `python -m pytest legacy/tests -q` runs the legacy suite; tests that need the private gold sample or corpus skip with an explicit reason. Analysis commands (`scripts/run_*.py`, `scripts/reproduce.py`) are present as code but exit with a clear message unless the private archive is attached.

Confirmatory numbers in `outputs/public/` exclude ten temporally invalid external records (see `docs/RECORD_VALIDITY.md`).

## Licence

- Code: `LICENSE` (MIT)
- Data: `docs/DATA_ACCESS_AND_LICENSE.md` and `data/public/redistribution_status.csv`
