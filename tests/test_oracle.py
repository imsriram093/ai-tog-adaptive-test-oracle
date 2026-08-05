from ai_tog.baseline import load_categories
from ai_tog.embedding.tfidf_embedder import TfidfEmbedder
from ai_tog.oracle import OracleEngine
from ai_tog.profile import ProfileBuilder

DATA_DIR = "data/categories"


def _build(category_id):
    categories = load_categories(DATA_DIR)
    cat = categories[category_id]
    corpus = [ex.text for ex in cat.baseline_examples] + [ex.text for ex in cat.rejected_examples]
    embedder = TfidfEmbedder(corpus)
    profile = ProfileBuilder(embedder).build(cat)
    return cat, embedder, profile


def test_paraphrase_passes_structural_and_invariant_checks():
    cat, embedder, profile = _build("refund_policy_explanation")
    oracle = OracleEngine(embedder)
    candidate = ("Provided you still have your receipt, items returned inside "
                 "the 30-day period qualify for a full refund.")
    signals = oracle.score(profile, candidate)
    assert signals.structural_valid
    assert signals.invariants_satisfied
    assert signals.semantic_similarity > 0


def test_wrong_window_fails_required_term_invariant():
    cat, embedder, profile = _build("refund_policy_explanation")
    oracle = OracleEngine(embedder)
    candidate = "You can return items within 90 days for store credit only, no cash refunds."
    signals = oracle.score(profile, candidate)
    assert not signals.invariants_satisfied
    assert any("30 day" in v for v in signals.violations)


def test_banned_term_fails_even_if_required_term_present():
    cat, embedder, profile = _build("capital_city_fact")
    oracle = OracleEngine(embedder)
    candidate = "Many people mistakenly think it's Paris, but the capital of France is actually Berlin."
    signals = oracle.score(profile, candidate)
    assert not signals.invariants_satisfied
    assert any("berlin" in v.lower() for v in signals.violations)


def test_attempted_out_of_scope_help_is_caught_by_banned_terms():
    cat, embedder, profile = _build("out_of_scope_decline")
    oracle = OracleEngine(embedder)
    candidate = ("Sure! Here's an essay introduction paragraph about climate change: "
                 "Climate change is one of the most pressing issues of our time.")
    signals = oracle.score(profile, candidate)
    assert not signals.invariants_satisfied


def test_too_short_answer_fails_structural_check():
    cat, embedder, profile = _build("out_of_scope_decline")
    oracle = OracleEngine(embedder)
    signals = oracle.score(profile, "Can't help.")
    assert not signals.structural_valid
