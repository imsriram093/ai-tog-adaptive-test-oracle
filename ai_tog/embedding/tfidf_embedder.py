from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import Embedder, _l2_normalize


class TfidfEmbedder(Embedder):
    """
    Deterministic, dependency-light embedding backend.

    This is NOT a semantic embedding model - it is a lexical bag-of-words
    vectorizer. It exists so the full pipeline can run and be unit-tested in
    environments with no access to a model hub (e.g. a locked-down CI runner
    or this sandbox), without changing a single line of oracle/threshold/
    verdict logic. Swap in SentenceTransformerEmbedder for real semantic
    similarity in any environment that can reach the model hub.
    """

    def __init__(self, corpus: list[str], max_features: int = 512):
        self._vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self._vectorizer.fit(corpus)
        self._dim = len(self._vectorizer.vocabulary_)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._vectorizer.transform(texts).toarray().astype(np.float64)
        return _l2_normalize(vecs)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "tfidf-fallback"
