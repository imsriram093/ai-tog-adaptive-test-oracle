from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from .oracle import SignalScores
from .profile import ProfileBuilder, ReferenceProfile
from .storage import Storage
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
    evaluation_id: int | None = None


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

    def __init__(
        self,
        threshold_tracker: AdaptiveThresholdTracker,
        storage: Storage | None = None,
    ):
        self.tracker = threshold_tracker
        self.storage = storage

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

        evaluation_id: int | None = None
        if self.storage is not None:
            evaluation_id = self.storage.record_evaluation(
                category_id=category_id,
                candidate_text=candidate_text,
                semantic_similarity=signals.semantic_similarity,
                structural_valid=signals.structural_valid,
                invariants_satisfied=signals.invariants_satisfied,
                composite_score=score,
                verdict=verdict.value,
                explanation=explanation,
            )

        return EvaluationResult(
            category_id=category_id,
            candidate_text=candidate_text,
            signals=signals,
            band=band,
            verdict=verdict,
            explanation=explanation,
            evaluation_id=evaluation_id,
        )

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

        "bad": logged only (via the caller / storage) - deliberately NOT added to the
        profile or the threshold window, so a confirmed-bad answer can never
        drag the "acceptable" region toward itself.
        """
        if result.verdict != Verdict.REVIEW:
            raise ValueError("resolve_review() should only be called on REVIEW verdicts.")
        if human_label not in ("good", "bad"):
            raise ValueError("human_label must be 'good' or 'bad'.")

        if self.storage is not None and result.evaluation_id is not None:
            self.storage.record_review_resolution(
                evaluation_id=result.evaluation_id,
                human_label=human_label,
            )

        if human_label == "good":
            profile_builder.update_incremental(profile, result.candidate_text)
            self.tracker.record_confirmed_good(result.category_id, result.signals.composite_score)

