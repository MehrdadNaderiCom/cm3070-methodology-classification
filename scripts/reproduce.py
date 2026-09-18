"""Run the lightweight study from a fresh source/input directory."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    _root = Path(__file__).resolve().parents[1]
    _need = [
        'data/study_records/adjudicated_labels.csv',
        'legacy/data/processed/corpus.csv',
    ]
    _miss = [p for p in _need if not (_root / p).exists()]
    if _miss:
        raise SystemExit('Private-data command. Missing:\n' + '\n'.join(_miss))

    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    dest = Path(args.destination).resolve()
    if dest.exists() or ROOT == dest or ROOT in dest.parents:
        raise ValueError("Use a new destination outside this release")
    dest.mkdir(parents=True)
    for name in ("src", "tests", "legacy", "scripts", "data", "config"):
        shutil.copytree(ROOT / name, dest / name,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
    for name in ("pyproject.toml", "environment.lock"):
        shutil.copy2(ROOT / name, dest / name)
    (dest / "outputs/audit").mkdir(parents=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env["PYTHONPATH"] = str(dest / "src") + os.pathsep + str(dest / "legacy/src")
    commands = [
        ("contracts", [sys.executable, "-m", "pytest", "tests", "-q"]),
        ("legacy", [sys.executable, "-m", "pytest", "legacy/tests", "-c", "legacy/pyproject.toml", "-q"]),
        ("measured", [sys.executable, "scripts/run_measured.py"]),
        ("mpnet_encode", [sys.executable, "scripts/run_mpnet_encode.py"]),
        ("candidates", [sys.executable, "scripts/run_model_candidates.py"]),
        ("study", [sys.executable, "scripts/run_study_analysis.py"]),
        ("figures", [sys.executable, "scripts/build_figures.py"]),
        ("verification", [sys.executable, "scripts/verify_release.py", "--profile", "study"]),
        ("measured_demo", [sys.executable, "scripts/classify.py", "--file", "data/demo_input.json"]),
        ("study_demo", [sys.executable, "scripts/classify.py", "--case-id", "EXT000"]),
    ]
    steps = []
    for name, command in commands:
        run = subprocess.run(command, cwd=dest, env=env, text=True, capture_output=True)
        (dest / f"outputs/audit/{name}.log").write_text(run.stdout + run.stderr)
        steps.append({"step": name, "returncode": run.returncode})
        print(name, run.returncode, flush=True)
        if run.returncode:
            raise RuntimeError(f"{name} failed; inspect {dest}/outputs/audit/{name}.log")
    for rel in ("outputs/measured/lexical_reproduction.csv", "outputs/measured/nested_lexical_oof.csv",
                "outputs/scenario/test_row_results.csv", "outputs/scenario/development_comparison.csv"):
        a, b = pd.read_csv(ROOT / rel), pd.read_csv(dest / rel)
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12, rtol=1e-12)
    for rel in ("outputs/scenario/confirmatory_test_results.json",
                "outputs/measured/nested_lexical_summary.json"):
        assert json.loads((ROOT / rel).read_text()) == json.loads((dest / rel).read_text()), rel
    outcome = {"status": "PASS", "destination": str(dest), "steps": steps,
               "scientific_comparison": "four row-level CSVs within 1e-12 and two summary JSONs identical",
               "environment": "same installed environment; fresh source/input directory, not a fresh operating system"}
    (dest / "outputs/audit/clean_result.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2))

if __name__ == "__main__":
    main()
