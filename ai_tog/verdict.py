from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from .oracle import SignalScores
from .profile import ProfileBuilder, ReferenceProfile
from .threshold import AdaptiveThresholdTracker, ThresholdBand


class Verdict(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


@dataclass
class EvaluationResult:
    category_id: str
    candidate_text: str
    signals: SignalScores
    band: ThresholdBand
    verdict: Verdict
    explanation: str


class VerdictAggregator:
    """
    Stage 5: turns Stage 3's multi-signal scores and Stage 4's current
    threshold band into one of three verdicts - never a forced binary
    decision, because forcing genuinely ambiguous, non-deterministic output
    into pass/fail produces false alarms. A hard invariant violation (a
    banned term, a missing required fact) always fails outright regardless
    of semantic similarity, since no amount of "close enough" phrasing
    should excuse a factually wrong or policy-violating answer.

    resolve_review() is the closed loop: when a human resolves a REVIEW
    verdict as "good", that resolution is folded back into both the
    reference profile (Stage 2) and the adaptive threshold's rolling window
    (Stage 4), so the oracle improves across builds instead of staying static.
    """

    def __init__(self, threshold_tracker: AdaptiveThresholdTracker):
        self.tracker = threshold_tracker

    def decide(self, category_id: str, candidate_text: str, signals: SignalScores) -> EvaluationResult:
        band = self.tracker.current_band(category_id)
        score = signals.composite_score

        if not signals.invariants_satisfied:
            verdict = Verdict.FAIL
            explanation = f"hard invariant violated: {signals.violations}"
        elif score >= band.pass_cut:
            verdict = Verdict.PASS
            explanation = f"score {score:.3f} >= pass cut {band.pass_cut:.3f}"
        elif score >= band.review_cut:
            verdict = Verdict.REVIEW
            explanation = (
                f"score {score:.3f} between review cut {band.review_cut:.3f} "
                f"and pass cut {band.pass_cut:.3f}"
            )
        else:
            verdict = Verdict.FAIL
            explanation = f"score {score:.3f} < review cut {band.review_cut:.3f}"

        return EvaluationResult(category_id, candidate_text, signals, band, verdict, explanation)

    def resolve_review(
        self,
        result: EvaluationResult,
        human_label: str,
        profile_builder: ProfileBuilder,
        profile: ReferenceProfile,
    ) -> None:
        """
        human_label: "good" or "bad".

        "good": the answer is folded into the reference profile centroid
        AND its score is added to the adaptive threshold's rolling window,
        since it is now a confirmed example of what a passing score looks
        like for this category.

        "bad": logged only (via the caller) - deliberately NOT added to the
        profile or the threshold window, so a confirmed-bad answer can never
        drag the "acceptable" region toward itself.
        """
        if result.verdict != Verdict.REVIEW:
            raise ValueError("resolve_review() should only be called on REVIEW verdicts.")
        if human_label not in ("good", "bad"):
            raise ValueError("human_label must be 'good' or 'bad'.")

        if human_label == "good":
            profile_builder.update_incremental(profile, result.candidate_text)
            self.tracker.record_confirmed_good(result.category_id, result.signals.composite_score)
