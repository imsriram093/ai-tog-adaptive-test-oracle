from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from .embedding.base import Embedder
from .profile import ReferenceProfile

# Composite-score penalties applied when a candidate fails a non-semantic
# signal. Kept as named constants (not buried magic numbers) so they can be
# tuned and cited directly in the project write-up.
STRUCTURAL_VIOLATION_PENALTY = 0.15
INVARIANT_VIOLATION_PENALTY = 0.25


@dataclass
class SignalScores:
    """
    Stage 3 output: the individual signals behind one verdict, kept
    separate rather than collapsed immediately, so the verdict aggregator
    (Stage 5) and any human reviewer can see *why* a score landed where it
    did instead of trusting one opaque number.
    """
    semantic_similarity: float
    structural_valid: bool
    invariants_satisfied: bool
    violations: list[str] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        score = self.semantic_similarity
        if not self.structural_valid:
            score -= STRUCTURAL_VIOLATION_PENALTY
        if not self.invariants_satisfied:
            score -= INVARIANT_VIOLATION_PENALTY
        return max(0.0, score)


class OracleEngine:
    """
    Stage 3: scores a new candidate output against a category's reference
    profile on three independent signals - semantic similarity, structural
    validity, and domain invariants - rather than a single similarity
    number. A candidate can be semantically close to known-good answers and
    still be wrong (e.g. right topic, wrong required fact), which a
    similarity-only oracle would miss.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def score(self, profile: ReferenceProfile, candidate_text: str) -> SignalScores:
        vec = self.embedder.encode([candidate_text])[0]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        semantic_similarity = float(vec @ profile.centroid)

        violations: list[str] = []
        word_count = len(candidate_text.split())
        structural_valid = True
        if profile.min_length is not None and word_count < profile.min_length:
            structural_valid = False
            violations.append(f"too short: {word_count} words < minimum {profile.min_length}")
        if profile.max_length is not None and word_count > profile.max_length:
            structural_valid = False
            violations.append(f"too long: {word_count} words > maximum {profile.max_length}")

        # Normalize hyphens to spaces before matching so "30-day" and
        # "30 day" are treated as the same phrase - a naive substring check
        # would otherwise miss legitimate answers over a formatting detail
        # that has nothing to do with correctness.
        text_norm = candidate_text.lower().replace("-", " ")
        invariants_satisfied = True
        for term in profile.required_terms:
            if term.lower().replace("-", " ") not in text_norm:
                invariants_satisfied = False
                violations.append(f"missing required term: '{term}'")
        for term in profile.banned_terms:
            if term.lower().replace("-", " ") in text_norm:
                invariants_satisfied = False
                violations.append(f"contains banned term: '{term}'")

        return SignalScores(
            semantic_similarity=semantic_similarity,
            structural_valid=structural_valid,
            invariants_satisfied=invariants_satisfied,
            violations=violations,
        )
