# Reproducibility (public tree)

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environment.lock
python -m pip install -e .
python -m pytest
python scripts/classify.py --file data/demo_input.json
python scripts/verify_release.py
python scripts/demo_review.py
```

These commands use authored demo data and published aggregates. They do not rebuild MPNet embeddings or the archival lexical model. That path remains in the private archive.
