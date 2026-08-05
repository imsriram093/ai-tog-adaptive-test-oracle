from __future__ import annotations
import numpy as np

from .base import Embedder, _l2_normalize


class SentenceTransformerEmbedder(Embedder):
    """
    Production embedding backend using a real sentence-embedding model.

    Requires the optional `semantic` extra (sentence-transformers + torch)
    and network access to the model hub on first run to download weights,
    which is why this project's CI/sandbox tests default to TfidfEmbedder
    instead. Locally, this is what you want: swap it in via
    `pipeline.build_pipeline(embedder_backend="semantic")`.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Install the optional "
                "'semantic' extra: pip install -e '.[semantic]'"
            ) from e
        try:
            self._model = SentenceTransformer(model_name)
        except OSError as e:
            raise OSError(
                f"Failed to load or download the sentence-transformer model "
                f"'{model_name}'. If you are offline or behind a firewall, "
                f"either pre-download the model while connected, or switch to "
                f"the TF-IDF backend: "
                f"build_pipeline(data_dir, embedder_backend='tfidf')"
            ) from e
        self._dim = self._model.get_sentence_embedding_dimension()
        self._name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True)
        return _l2_normalize(vecs)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name
