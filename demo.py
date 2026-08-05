"""
Demonstrates the two things the benchmark alone can't show: the threshold
band actually adapting as confirmed-good evidence accumulates, and drift
detection firing when a simulated "model update" shifts scores.

Run: python demo.py
"""
from ai_tog.pipeline import build_pipeline

DATA_DIR = "data/categories"


def run():
    pipeline = build_pipeline(DATA_DIR)
    tracker = pipeline.threshold_tracker
    cat_id = "capital_city_fact"

    print("=== Initial band, seeded from leave-one-out baseline scores ===")
    band = tracker.current_band(cat_id)
    print(f"pass_cut={band.pass_cut:.3f}  review_cut={band.review_cut:.3f}  "
          f"center={band.center:.3f}  spread={band.spread:.3f}  cold_start={band.is_cold_start}")

    print("\n=== Simulating 8 more confirmed-good build runs (steady quality) ===")
    for score in [0.58, 0.61, 0.55, 0.63, 0.59, 0.60, 0.57, 0.62]:
        tracker.record_confirmed_good(cat_id, score)
    band = tracker.current_band(cat_id)
    print(f"pass_cut={band.pass_cut:.3f}  review_cut={band.review_cut:.3f}  "
          f"center={band.center:.3f}  spread={band.spread:.3f}")
    print("drift check:", tracker.detect_drift(cat_id, recent_n=8) or "none")

    print("\n=== Simulating a model update that shifts scores down ===")
    for score in [0.32, 0.29, 0.34, 0.30, 0.33, 0.28, 0.35, 0.31]:
        tracker.record_confirmed_good(cat_id, score)
    band = tracker.current_band(cat_id)
    print(f"pass_cut={band.pass_cut:.3f}  review_cut={band.review_cut:.3f}  "
          f"center={band.center:.3f}  spread={band.spread:.3f}")
    print("drift check:", tracker.detect_drift(cat_id, recent_n=8) or "none")
    print("-> the band has recalibrated to the new normal AND a drift warning fired,")
    print("   so a human gets flagged rather than the oracle silently drifting.")

    print("\n=== Feedback loop: resolving a REVIEW verdict ===")
    profile_before = pipeline.profiles[cat_id].n_examples
    result = pipeline.evaluate(cat_id, "Paris, historically and today, is the capital of France.")
    print(f"verdict={result.verdict.value}  score={result.signals.composite_score:.3f}  "
          f"explanation={result.explanation}")
    if result.verdict.value == "REVIEW":
        pipeline.resolve_review(result, "good")
        print(f"reference profile n_examples: {profile_before} -> {pipeline.profiles[cat_id].n_examples}")
    else:
        print(f"(landed as {result.verdict.value} outright this run - nothing to resolve)")


if __name__ == "__main__":
    run()
