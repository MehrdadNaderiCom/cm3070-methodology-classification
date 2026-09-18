"""Sentence embedding representation, paired with the same downstream classifier.

The comparison this project exists to make is lexical against semantic. Holding
the classifier fixed and changing only the representation is what makes the
difference attributable to the representation.

Encoding is cached on disk by text and model. Without a cache, five fold cross
validation re-encodes every abstract five times, and the three tasks re-encode
the same corpus three more times on top of that.
"""

from __future__ import annotations

import hashlib
import json
import importlib.metadata
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cdfm import RANDOM_STATE
from cdfm.paths import MODELS

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
SCIENTIFIC_MODEL = "allenai/specter2_base"

CACHE_PATH = MODELS / "embedding_cache.joblib"


class SentenceEmbedder(BaseEstimator, TransformerMixin):
    """Encode text with a sentence transformer, caching by content hash.

    Written as a scikit-learn transformer so that it drops into the same
    Pipeline and the same `cross_val_predict` call as the TF-IDF vectoriser.
    That is not cosmetic: it is what guarantees the two approaches see identical
    folds and identical training rows.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 16,
        cache_path: Path | None = None,
        use_cache: bool = True,
        revision: str = DEFAULT_REVISION,
        preprocessing_version: str = "title-abstract-v1",
        corpus_fingerprint: str = "",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.use_cache = use_cache
        self.revision = revision
        self.preprocessing_version = preprocessing_version
        self.corpus_fingerprint = corpus_fingerprint

    def _resolved_cache_path(self) -> Path:
        return Path(self.cache_path) if self.cache_path else CACHE_PATH

    def _load_cache(self) -> dict[str, np.ndarray]:
        path = self._resolved_cache_path()
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                return {}
        return {}

    def _key(self, text: str) -> str:
        versions = {}
        for library in ("sentence-transformers", "transformers", "torch", "numpy"):
            try:
                versions[library] = importlib.metadata.version(library)
            except importlib.metadata.PackageNotFoundError:
                versions[library] = "not-installed"
        identity = {
            "model": self.model_name, "revision": self.revision,
            "preprocessing": self.preprocessing_version,
            "corpus": self.corpus_fingerprint, "schema": 2,
            "pooling": "model-defined-mean", "normalize": True,
            "versions": versions, "text": text, "batch_size": self.batch_size,
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()

    def _encoder(self):
        from sentence_transformers import SentenceTransformer

        if not hasattr(self, "_model"):
            if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
                raise ValueError("an immutable model commit SHA is required")
            if self.model_name != DEFAULT_MODEL and self.revision == DEFAULT_REVISION:
                raise ValueError("the MPNet revision cannot be used for another model")
            self._model = SentenceTransformer(self.model_name, revision=self.revision)
        return self._model

    def fit(self, X, y=None):  # noqa: N803
        return self

    def model_directory_bytes(self) -> int:
        """Size on disk of the downloaded transformer weights.

        A fitted pipeline serialised with joblib does not carry these, because
        the encoder is loaded lazily and lives outside the estimator's state.
        Reporting the joblib size alone would describe this model as forty times
        smaller than the TF-IDF baseline, which is the opposite of the truth.
        """
        try:
            from huggingface_hub import snapshot_download

            directory = Path(snapshot_download(self.model_name, revision=self.revision, local_files_only=True))
        except Exception:
            return 0
        return sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())

    def transform(self, X):  # noqa: N803
        texts = [str(item) for item in X]

        if not self.use_cache:
            unique = list(dict.fromkeys(texts))
            vectors = self._encoder().encode(
                unique,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            lookup = dict(zip(unique, vectors))
            return np.vstack([lookup[text] for text in texts])

        cache = self._load_cache()

        missing = [text for text in texts if self._key(text) not in cache]
        if missing:
            unique = list(dict.fromkeys(missing))
            vectors = self._encoder().encode(
                unique,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for text, vector in zip(unique, vectors):
                cache[self._key(text)] = vector.astype(np.float32)
            path = self._resolved_cache_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(cache, path)

        return np.vstack([cache[self._key(text)] for text in texts])


def build_embedding_model(
    model_name: str = DEFAULT_MODEL,
    random_state: int = RANDOM_STATE,
    cache_path: Path | None = None,
) -> Pipeline:
    """Sentence embeddings into the same logistic regression as the baseline.

    The classifier settings match `cdfm.models.baseline` deliberately. If the
    embedding model wins, the win should belong to the representation and not to
    a differently tuned classifier.
    """
    return Pipeline(
        [
            ("embed", SentenceEmbedder(model_name=model_name, cache_path=cache_path)),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=random_state
                ),
            ),
        ]
    )
