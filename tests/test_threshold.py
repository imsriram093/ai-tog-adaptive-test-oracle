from ai_tog.threshold import AdaptiveThresholdTracker, COLD_START_PASS_CUT


def test_cold_start_band_used_before_min_samples():
    tracker = AdaptiveThresholdTracker(min_samples=5)
    band = tracker.current_band("unseen_category")
    assert band.is_cold_start
    assert band.pass_cut == COLD_START_PASS_CUT


def test_band_becomes_adaptive_once_enough_samples_recorded():
    tracker = AdaptiveThresholdTracker(min_samples=5)
    for score in [0.80, 0.82, 0.79, 0.81, 0.80, 0.83]:
        tracker.record_confirmed_good("cat_a", score)
    band = tracker.current_band("cat_a")
    assert not band.is_cold_start
    assert 0.75 < band.center < 0.85
    assert band.pass_cut < band.center  # pass cut sits below the rolling mean


def test_spread_floor_prevents_degenerate_tight_band():
    tracker = AdaptiveThresholdTracker(min_samples=3, spread_floor=0.05)
    for score in [0.80, 0.80, 0.80, 0.80]:  # zero real variance
        tracker.record_confirmed_good("cat_b", score)
    band = tracker.current_band("cat_b")
    assert band.spread >= 0.05


def test_drift_detection_flags_a_shifted_recent_window():
    tracker = AdaptiveThresholdTracker(min_samples=5, window_size=100)
    # long run of stable high scores
    for _ in range(20):
        tracker.record_confirmed_good("cat_c", 0.85)
    # then a sustained shift downward, simulating a model update
    for _ in range(10):
        tracker.record_confirmed_good("cat_c", 0.55)
    drift = tracker.detect_drift("cat_c", recent_n=10)
    assert drift is not None
    assert "downward" in drift


def test_no_drift_flagged_for_stable_scores():
    tracker = AdaptiveThresholdTracker(min_samples=5, window_size=100)
    for _ in range(30):
        tracker.record_confirmed_good("cat_d", 0.80)
    assert tracker.detect_drift("cat_d", recent_n=10) is None
