#!/usr/bin/env python3
"""Encode corpus texts with sentence-transformers/all-mpnet-base-v2.

This is a real model run: the exact revision is pinned, the snapshot files are
hashed, and embeddings plus a run log are written under outputs/model_runs/.
The cached embeddings are consumed by scripts/run_model_candidates.py.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
OUT_DIR = ROOT / "outputs" / "model_runs"
RUN_TAG = f"mpnet_encode_{REVISION[:12]}"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def snapshot_manifest() -> dict:
    """Hash every file inside the locally cached model snapshot."""
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(repo_id=MODEL_NAME, revision=REVISION))
    files = {}
    for p in sorted(snap.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(snap))] = {
                "bytes": p.stat().st_size, "sha256": sha256_file(p)}
    return {"snapshot_path": str(snap), "file_count": len(files), "files": files}


def main() -> None:
    _root = Path(__file__).resolve().parents[1]
    _need = [
        'data/study_records/adjudicated_labels.csv',
        'legacy/data/processed/corpus.csv',
    ]
    _miss = [p for p in _need if not (_root / p).exists()]
    if _miss:
        raise SystemExit('Private-data command. Missing:\n' + '\n'.join(_miss))

    from sentence_transformers import SentenceTransformer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(ROOT / "data" / "study_records" / "corpus_registry.csv")
    texts = (registry.title.fillna("") + ". " + registry.abstract.fillna("")).tolist()

    t0 = time.perf_counter()
    manifest = snapshot_manifest()
    download_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, revision=REVISION)  # pinned revision
    load_seconds = time.perf_counter() - t1

    t2 = time.perf_counter()
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    encode_seconds = time.perf_counter() - t2

    emb_path = OUT_DIR / "mpnet_embeddings.npz"
    np.savez_compressed(
        emb_path, embeddings=embeddings.astype(np.float32),
        document_ids=registry.document_id.to_numpy().astype("U8"))

    log = {
        "run_tag": RUN_TAG, "model": MODEL_NAME, "revision": REVISION,
        "executed": True, "execution_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_texts": len(texts), "embedding_dim": int(embeddings.shape[1]),
        "normalize_embeddings": True, "batch_size": 32,
        "timings_seconds": {"snapshot_hash": round(download_seconds, 3),
                            "model_load": round(load_seconds, 3),
                            "encode": round(encode_seconds, 3)},
        "embeddings_file": str(emb_path.relative_to(ROOT)),
        "embeddings_sha256": sha256_file(emb_path),
        "model_snapshot": manifest,
        "environment": {
            "python": platform.python_version(), "torch": __import__("torch").__version__,
            "sentence_transformers": __import__("sentence_transformers").__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "platform": platform.platform(), "processor_count": __import__("os").cpu_count()},
        "inputs": {
            "registry": "data/study_records/corpus_registry.csv",
            "text_field": "title + '. ' + abstract"}}
    log_path = OUT_DIR / "mpnet_encode_log.json"
    log_path.write_text(json.dumps(log, indent=2) + "\n")
    print(f"encoded {len(texts)} texts, dim {embeddings.shape[1]}, "
          f"{encode_seconds:.1f}s; log -> {log_path.name}")


if __name__ == "__main__":
    main()
