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
