import pytest
from ai_tog.baseline import TestCategory, BaselineExample, load_categories
from ai_tog.embedding.tfidf_embedder import TfidfEmbedder
from ai_tog.profile import ProfileBuilder

DATA_DIR = "data/categories"


def test_category_requires_at_least_three_baseline_examples():
    with pytest.raises(ValueError):
        TestCategory(
            category_id="too_small",
            prompt="x",
            baseline_examples=[BaselineExample(text="a"), BaselineExample(text="b")],
        )


def test_load_categories_finds_all_three():
    categories = load_categories(DATA_DIR)
    assert set(categories.keys()) == {
        "refund_policy_explanation",
        "capital_city_fact",
        "out_of_scope_decline",
    }


def test_profile_centroid_is_unit_length():
    categories = load_categories(DATA_DIR)
    cat = categories["capital_city_fact"]
    corpus = [ex.text for ex in cat.baseline_examples]
    embedder = TfidfEmbedder(corpus)
    builder = ProfileBuilder(embedder)
    profile = builder.build(cat)
    import numpy as np
    assert abs(np.linalg.norm(profile.centroid) - 1.0) < 1e-6
    assert profile.n_examples == 5
    assert profile.version == 1


def test_leave_one_out_seed_scores_are_lower_than_naive_self_similarity():
    """
    The whole point of leave-one-out seeding is that it must NOT be as
    optimistic as scoring each baseline example against a centroid that
    already includes itself.
    """
    categories = load_categories(DATA_DIR)
    cat = categories["capital_city_fact"]
    corpus = [ex.text for ex in cat.baseline_examples]
    embedder = TfidfEmbedder(corpus)
    builder = ProfileBuilder(embedder)

    profile = builder.build(cat)  # centroid includes every example
    vecs = embedder.encode(corpus)
    naive_self_sim = [float(v @ profile.centroid) for v in vecs]

    loo_scores = builder.leave_one_out_seed_scores(cat)

    assert sum(loo_scores) / len(loo_scores) <= sum(naive_self_sim) / len(naive_self_sim)


def test_incremental_update_increases_n_examples_and_version():
    categories = load_categories(DATA_DIR)
    cat = categories["capital_city_fact"]
    corpus = [ex.text for ex in cat.baseline_examples]
    embedder = TfidfEmbedder(corpus)
    builder = ProfileBuilder(embedder)
    profile = builder.build(cat)

    before_n, before_v = profile.n_examples, profile.version
    builder.update_incremental(profile, "Paris is definitely the capital of France.")
    assert profile.n_examples == before_n + 1
    assert profile.version == before_v + 1
    assert len(profile.history) == 2
