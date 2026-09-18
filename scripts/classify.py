"""Classify a local input, or inspect one frozen exercise case."""
from pathlib import Path
import argparse
import json
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from study import CLASSES
from study.core import align_probabilities, load_records, prediction_sets, sha256


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file")
    group.add_argument("--case-id")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.case_id:
        frozen = ROOT / "outputs/scenario/external_predictions_frozen.parquet"
        stored = ROOT / "outputs/public/test_row_results.csv"
        if frozen.exists():
            base = ROOT / "outputs/scenario"
            config = json.loads((base / "conformal_configuration.json").read_text(encoding="utf-8"))
            if sha256(frozen) != config["predictions_sha256"]:
                raise ValueError("frozen predictions hash mismatch")
            data = pd.read_parquet(frozen)
            row = data[(data.document_id == args.case_id) & (data.model == config["selected_model"])]
            if len(row) != 1:
                raise ValueError("unknown or non-unique case")
            p = row[[f"p{i}" for i in range(5)]].to_numpy()
            sets = prediction_sets(p, config)[0]
            records = [{"document_id": args.case_id, "mode": "study_case",
                        "top1": CLASSES[int(p[0].argmax())], "top1_score": float(p.max()),
                        "prediction_set": list(np.array(CLASSES)[sets]),
                        "review_required": bool(sets.sum() != 1),
                        "frozen_model_family": config["selected_model"]}]
        elif stored.exists():
            table = pd.read_csv(stored)
            hit = table[table.document_id == args.case_id]
            if len(hit) != 1:
                raise ValueError("unknown or excluded case in the public table")
            r = hit.iloc[0]
            records = [{"document_id": args.case_id, "mode": "stored_frozen_output",
                        "top1": r.top1, "label": r.label,
                        "set_size": int(r.set_size), "review_required": bool(r.review),
                        "correct": bool(r.correct),
                        "frozen_model_family": "Cue",
                        "note": "Stored Cue output from the frozen study path, not a fresh prediction."}]
        else:
            raise FileNotFoundError("no frozen study-path table is present in this tree")
    else:
        frame = load_records([args.file])
        archival = ROOT / "outputs/measured/point_model.bin"
        demo = ROOT / "data/demo/demo_model.bin"
        if archival.exists():
            base = ROOT / "outputs/measured"
            manifest = json.loads((base / "training_manifest.json").read_text(encoding="utf-8"))
            path = archival
            mode = "archival-trained-model"
        elif demo.exists():
            base = ROOT / "data/demo"
            manifest = json.loads((base / "demo_model.manifest.json").read_text(encoding="utf-8"))
            path = demo
            mode = "public-demo-model"
        else:
            raise FileNotFoundError("no classifier artefact is present")
        if sha256(path) != manifest["model_sha256"]:
            raise ValueError("point-model hash mismatch")
        # Load only this locally produced model; never unpickle user input.
        model = joblib.load(path)
        p = align_probabilities(model.predict_proba((frame.title + ". " + frame.abstract)
                                                   .to_numpy(dtype=object)), model.classes_)
        records = [{"document_id": row.document_id, "mode": mode,
                    "top1": CLASSES[int(p[i].argmax())], "top1_score": float(p[i].max()),
                    "prediction_set": None, "review_required": True,
                    "review_reason": "independent calibration not available for this model",
                    "model_sha256": manifest["model_sha256"]}
                   for i, row in frame.reset_index(drop=True).iterrows()]
    text = json.dumps(records, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
