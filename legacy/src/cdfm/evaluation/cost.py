"""Measure what a model costs to train and to run.

The research question names inference latency and computational cost, so those
are measured rather than described. A claim that one approach is more expensive
than another is worth nothing without a number and a unit next to it.

Two traps are avoided here, both of which produced plainly false figures on the
first attempt.

Caching. The embedding transformer caches vectors by content hash, which is what
makes cross validation affordable. Timing it with the cache warm measures a
dictionary lookup and reports a transformer as being as fast as a bag of words.
Caching is therefore switched off for the duration of the measurement.

Lazy loading. The transformer weights are loaded on first use and are not part of
the estimator's serialised state, so `joblib` writes a pipeline of a few tens of
kilobytes for a model of several hundred megabytes. External weights are measured
on disk and added.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import joblib
import psutil


def _disable_caches(estimator) -> None:
    """Turn off any embedding cache so latency reflects real work."""
    for step in getattr(estimator, "steps", []):
        component = step[1]
        if hasattr(component, "use_cache"):
            component.use_cache = False


def _external_weights_bytes(estimator) -> int:
    total = 0
    for step in getattr(estimator, "steps", []):
        component = step[1]
        if hasattr(component, "model_directory_bytes"):
            total += component.model_directory_bytes()
    return total


def _resident_bytes() -> int:
    return psutil.Process(os.getpid()).memory_info().rss


def measure(estimator, features, labels, sample, *, repeats: int = 3) -> dict:
    """Fit once, then time repeated prediction over a fixed sample.

    Memory is process resident set size rather than `tracemalloc`, which only
    sees Python level allocations and therefore misses tensor memory entirely.
    """
    _disable_caches(estimator)
    gc.collect()
    baseline_memory = _resident_bytes()

    started = time.perf_counter()
    estimator.fit(features, labels)
    train_seconds = time.perf_counter() - started

    peak_memory = max(baseline_memory, _resident_bytes())

    # One untimed call, so that lazy model loading and any first call warmup is
    # not charged to the latency figure.
    estimator.predict(sample[:1])
    peak_memory = max(peak_memory, _resident_bytes())

    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        estimator.predict(sample)
        timings.append(time.perf_counter() - started)
        peak_memory = max(peak_memory, _resident_bytes())

    best = min(timings)
    size_bytes = serialised_size(estimator) + _external_weights_bytes(estimator)

    return {
        "train_seconds": round(train_seconds, 2),
        "inference_ms_per_paper": round(best / max(1, len(sample)) * 1000, 2),
        "resident_memory_mb": round(peak_memory / (1024 * 1024), 1),
        "memory_growth_mb": round((peak_memory - baseline_memory) / (1024 * 1024), 1),
        "model_size_mb": round(size_bytes / (1024 * 1024), 2),
    }


def serialised_size(estimator, scratch: Path | None = None) -> int:
    """Bytes the fitted estimator occupies when written out."""
    import tempfile

    directory = scratch or Path(tempfile.gettempdir())
    target = directory / "cdfm_size_probe.joblib"
    try:
        joblib.dump(estimator, target)
        return target.stat().st_size
    finally:
        target.unlink(missing_ok=True)
