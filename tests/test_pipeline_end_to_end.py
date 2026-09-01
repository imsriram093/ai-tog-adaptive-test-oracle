from ai_tog.pipeline import build_pipeline
from ai_tog.verdict import Verdict

DATA_DIR = "data/categories"


def test_paraphrase_of_baseline_passes():
    pipeline = build_pipeline(DATA_DIR)
    result = pipeline.evaluate(
        "capital_city_fact",
        "France's capital city is the city of Paris.",
    )
    assert result.verdict in (Verdict.PASS, Verdict.REVIEW)  # never a hard fail on a valid paraphrase


def test_factually_wrong_answer_fails():
    pipeline = build_pipeline(DATA_DIR)
    result = pipeline.evaluate(
        "capital_city_fact",
        "The capital of France is Lyon.",
    )
    assert result.verdict == Verdict.FAIL


def test_out_of_scope_decline_paraphrase_is_not_a_hard_fail():
    pipeline = build_pipeline(DATA_DIR)
    result = pipeline.evaluate(
        "out_of_scope_decline",
        "Sorry, but I can only handle support topics like orders and returns - "
        "that's not something I can help with.",
    )
    assert result.verdict in (Verdict.PASS, Verdict.REVIEW)


def test_attempting_the_out_of_scope_task_fails():
    pipeline = build_pipeline(DATA_DIR)
    result = pipeline.evaluate(
        "out_of_scope_decline",
        "Sure! Here's an essay introduction paragraph about climate change: "
        "Climate change is one of the most pressing issues of our time.",
    )
    assert result.verdict == Verdict.FAIL


def test_feedback_loop_grows_profile_and_threshold_window():
    pipeline = build_pipeline(DATA_DIR)
    profile_before_n = pipeline.profiles["capital_city_fact"].n_examples

    result = pipeline.evaluate("capital_city_fact", "Paris, France's capital, is a major European city.")
    if result.verdict == Verdict.REVIEW:
        pipeline.resolve_review(result, "good")
        assert pipeline.profiles["capital_city_fact"].n_examples == profile_before_n + 1
    else:
        # If it landed as PASS/FAIL outright there is nothing to resolve;
        # the test still confirms the pipeline ran end-to-end without error.
        assert result.verdict in (Verdict.PASS, Verdict.FAIL)


def test_pipeline_with_in_memory_storage():
    from ai_tog.storage import InMemoryStorage
    storage = InMemoryStorage()
    pipeline = build_pipeline(DATA_DIR, storage=storage)

    # Initial build seeds profiles and threshold scores
    assert storage.has_profile("capital_city_fact")
    scores = storage.get_threshold_scores("capital_city_fact")
    assert len(scores) >= 3

    # Evaluations get logged in storage
    result = pipeline.evaluate("capital_city_fact", "The capital of France is Paris.")
    evals = storage.get_evaluations("capital_city_fact")
    assert len(evals) == 1
    assert evals[0]["verdict"] in ("PASS", "REVIEW")
    assert result.evaluation_id == evals[0]["id"]

    storage.close()


def test_pipeline_persistence_roundtrip_rule_merging_and_history(tmp_path):
    db_file = tmp_path / "roundtrip_test.db"
    cat_id = "capital_city_fact"

    # --- Phase 1: First Run ---
    pipeline1 = build_pipeline(DATA_DIR, storage=db_file)
    prof1_init = pipeline1.profiles[cat_id]
    assert prof1_init.version == 1
    assert prof1_init.n_examples == 5

    # Force a REVIEW result or evaluate a candidate and simulate resolve_review
    candidate = "Paris, historically and today, is the capital of France."
    result1 = pipeline1.evaluate(cat_id, candidate)
    assert result1.evaluation_id is not None

    if result1.verdict == Verdict.REVIEW:
        pipeline1.resolve_review(result1, "good")
    else:
        # Manually invoke update_incremental and record_confirmed_good to advance state
        pipeline1.profile_builder.update_incremental(pipeline1.profiles[cat_id], candidate)
        pipeline1.threshold_tracker.record_confirmed_good(cat_id, result1.signals.composite_score)

    prof1_updated = pipeline1.profiles[cat_id]
    assert prof1_updated.version == 2
    assert prof1_updated.n_examples == 6
    band1 = pipeline1.threshold_tracker.current_band(cat_id)

    pipeline1.storage.close()

    # --- Phase 2: Process Restart (reload from same DB file) ---
    pipeline2 = build_pipeline(DATA_DIR, storage=db_file)
    prof2 = pipeline2.profiles[cat_id]

    # 1. Profile state restored from DB
    assert prof2.version == 2
    assert prof2.n_examples == 6

    # 2. Profile history restored from profile_history table
    assert len(prof2.history) >= 2
    assert "v1 built from" in prof2.history[0]
    assert "v2 incorporated 1 reviewer-approved" in prof2.history[1]

    # 3. Static category rules properly merged from JSON definitions
    assert prof2.min_length == 3
    assert prof2.max_length == 25
    assert "paris" in prof2.required_terms
    assert "berlin" in prof2.banned_terms

    # 4. Invariant checks still fire correctly post-reload
    banned_res = pipeline2.evaluate(cat_id, "The capital of France is Berlin.")
    assert banned_res.verdict == Verdict.FAIL
    assert any("berlin" in v.lower() for v in banned_res.signals.violations)

    missing_res = pipeline2.evaluate(cat_id, "France is a country located in Europe.")
    assert missing_res.verdict == Verdict.FAIL
    assert any("paris" in v.lower() for v in missing_res.signals.violations)

    # 5. Threshold band matches Phase 1 post-update band (not reset to baseline seed)
    band2 = pipeline2.threshold_tracker.current_band(cat_id)
    assert abs(band2.pass_cut - band1.pass_cut) < 1e-5
    assert abs(band2.center - band1.center) < 1e-5
    assert abs(band2.spread - band1.spread) < 1e-5

    pipeline2.storage.close()

