from __future__ import annotations
from collections import deque
from dataclasses import dataclass

import numpy as np

from .storage import Storage

# Cold-start fallback band, used only until a category has accumulated
# `min_samples` confirmed-good scores. Deliberately conservative and
# clearly separated from the adaptive logic below, so it is never mistaken
# for a tuned, "real" threshold - it is a bootstrap value only.
COLD_START_PASS_CUT = 0.75
COLD_START_REVIEW_CUT = 0.55


@dataclass
class ThresholdBand:
    pass_cut: float
    review_cut: float
    center: float
    spread: float
    is_cold_start: bool


class AdaptiveThresholdTracker:
    """
    Stage 4: replaces a fixed pass/fail cutoff with boundaries derived from
    a rolling window of scores that are *known* to belong to accepted
    answers (seeded from leave-one-out baseline scores, then grown by
    confirmed-good review resolutions from Stage 5). This is a control-chart
    style approach borrowed from statistical process control: instead of a
    constant limit, the boundary is `center - z * spread`, recomputed as the
    window shifts, so a legitimate change in the module's normal output
    (e.g. after a model upgrade) shifts the boundary with it instead of
    triggering a wave of false failures.

    pass_z / fail_z: how many standard deviations below the rolling mean a
    score can fall before moving from PASS -> REVIEW -> FAIL.
    """

    def __init__(
        self,
        window_size: int = 50,
        pass_z: float = 1.0,
        fail_z: float = 3.0,
        min_samples: int = 5,
        spread_floor: float = 0.02,
        storage: Storage | None = None,
    ):
        self.window_size = window_size
        self.pass_z = pass_z
        self.fail_z = fail_z
        self.min_samples = min_samples
        self.spread_floor = spread_floor
        self.storage = storage
        self._windows: dict[str, deque[float]] = {}

    def seed(self, category_id: str, scores: list[float]) -> None:
        self._windows[category_id] = deque(scores, maxlen=self.window_size)

    def load_from_storage(self, category_id: str) -> bool:
        """
        Rebuild rolling window from persisted threshold_scores on startup.
        Returns True if scores were loaded, False otherwise.
        """
        if self.storage is None:
            return False
        scores = self.storage.get_threshold_scores(category_id, limit=self.window_size)
        if scores:
            self.seed(category_id, scores)
            return True
        return False

    def record_confirmed_good(self, category_id: str, score: float) -> None:
        self._windows.setdefault(category_id, deque(maxlen=self.window_size)).append(score)
        if self.storage is not None:
            self.storage.record_threshold_score(category_id, score)


    def current_band(self, category_id: str) -> ThresholdBand:
        window = self._windows.get(category_id, [])
        if len(window) < self.min_samples:
            return ThresholdBand(
                pass_cut=COLD_START_PASS_CUT,
                review_cut=COLD_START_REVIEW_CUT,
                center=0.0,
                spread=0.0,
                is_cold_start=True,
            )
        arr = np.array(window)
        center = float(arr.mean())
        spread = float(arr.std(ddof=1)) if len(arr) > 1 else self.spread_floor
        spread = max(spread, self.spread_floor)
        return ThresholdBand(
            pass_cut=center - self.pass_z * spread,
            review_cut=center - self.fail_z * spread,
            center=center,
            spread=spread,
            is_cold_start=False,
        )

    def detect_drift(self, category_id: str, recent_n: int = 10, z: float = 2.0) -> str | None:
        """
        Flags when the mean of the most recent `recent_n` confirmed-good
        scores has moved more than `z` standard deviations away from the
        *historical* mean - a signal that the module's normal output has
        shifted (e.g. after a model or prompt change) and the reference
        profile may need a deliberate review, not just silent recalibration.

        Deliberately compares against the historical portion of the window
        only (everything before the most recent `recent_n` samples), not the
        whole window. Folding the recent samples into the baseline stats
        would let a real, sustained drift inflate its own comparison
        baseline and mask itself - the same failure mode this method exists
        to catch.
        """
        window = self._windows.get(category_id)
        if not window or len(window) < recent_n * 2:
            return None
        arr = np.array(window)
        historical, recent = arr[:-recent_n], arr[-recent_n:]
        if len(historical) < 2:
            return None
        hist_mean = float(historical.mean())
        hist_std = max(float(historical.std(ddof=1)), self.spread_floor)
        recent_mean = float(recent.mean())
        if abs(recent_mean - hist_mean) > z * hist_std:
            direction = "downward" if recent_mean < hist_mean else "upward"
            return (
                f"drift detected for '{category_id}': recent mean {recent_mean:.3f} vs "
                f"historical mean {hist_mean:.3f} ({direction}, > {z:.1f}sigma)"
            )
        return None
