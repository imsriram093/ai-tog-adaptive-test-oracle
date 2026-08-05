import numpy as np
from ai_tog.embedding.tfidf_embedder import TfidfEmbedder


def test_encode_returns_unit_vectors():
    corpus = ["the cat sat on the mat", "a dog ran in the park", "paris is the capital of france"]
    embedder = TfidfEmbedder(corpus)
    vecs = embedder.encode(corpus)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_similar_texts_score_higher_than_dissimilar():
    corpus = [
        "the capital of france is paris",
        "paris is the capital city of france",
        "the weather today is sunny and warm",
    ]
    embedder = TfidfEmbedder(corpus)
    vecs = embedder.encode(corpus)
    sim_related = vecs[0] @ vecs[1]
    sim_unrelated = vecs[0] @ vecs[2]
    assert sim_related > sim_unrelated
