from abc import ABC, abstractmethod
import numpy as np


class Embedder(ABC):
    """
    Backend-agnostic text embedding interface. Every stage of the pipeline
    depends only on this interface, never on a concrete embedding model, so
    the embedding backend can be swapped without touching the oracle logic.
    """

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) array of L2-normalized embeddings."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms
