from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .baseline import TestCategory
from .embedding.base import Embedder
from .storage import Storage



@dataclass
class ReferenceProfile:
    """
    Stage 2 artifact: the statistical fingerprint of "acceptable" for one
    test category. Not a single expected answer - a centroid in embedding
    space plus the structural/domain rules carried over from the category
    definition. `n_examples` and `version` track how much the profile has
    grown since its initial build, which matters once the feedback loop
    (Stage 5 -> Stage 2) starts folding in human-approved review resolutions.
    """
    category_id: str
    centroid: np.ndarray
    n_examples: int
    min_length: int | None
    max_length: int | None
    required_terms: list[str]
    banned_terms: list[str]
    created_at: str
    version: int = 1
    history: list[str] = field(default_factory=list)


class ProfileBuilder:
    """
    Turns a curated TestCategory (Stage 1) into a ReferenceProfile (Stage 2),
    and later updates that profile incrementally as human-approved review
    resolutions arrive from the verdict aggregator (Stage 5).
    """

    def __init__(self, embedder: Embedder, storage: Storage | None = None):
        self.embedder = embedder
        self.storage = storage

    def build(self, category: TestCategory) -> ReferenceProfile:
        texts = [ex.text for ex in category.baseline_examples]
        vecs = self.embedder.encode(texts)
        centroid = _normalize(vecs.mean(axis=0))
        return ReferenceProfile(
            category_id=category.category_id,
            centroid=centroid,
            n_examples=len(texts),
            min_length=category.min_length,
            max_length=category.max_length,
            required_terms=list(category.required_terms),
            banned_terms=list(category.banned_terms),
            created_at=_now(),
            version=1,
            history=[f"v1 built from {len(texts)} baseline examples ({_now()})"],
        )

    def update_incremental(self, profile: ReferenceProfile, approved_text: str) -> ReferenceProfile:
        """
        Folds one new human-approved example into the centroid using a
        running-mean update, so the whole embedding matrix does not need to
        be re-encoded and re-averaged on every single review resolution.

            new_centroid = normalize(old_centroid * n + new_vec) / (n + 1)
        """
        new_vec = self.embedder.encode([approved_text])[0]
        n = profile.n_examples
        blended = (profile.centroid * n + new_vec) / (n + 1)
        profile.centroid = _normalize(blended)
        profile.n_examples = n + 1
        profile.version += 1
        note = (
            f"v{profile.version} incorporated 1 reviewer-approved example "
            f"(n={profile.n_examples}, {_now()})"
        )
        profile.history.append(note)
        if self.storage is not None:
            self.storage.save_profile(
                category_id=profile.category_id,
                centroid=profile.centroid,
                n_examples=profile.n_examples,
                version=profile.version,
                updated_at=_now(),
                note=note,
            )
        return profile


    def leave_one_out_seed_scores(self, category: TestCategory) -> list[float]:
        """
        Produces a realistic starting distribution of "what does a genuinely
        good answer score against this profile" for Stage 4 to seed its
        rolling window with, instead of naively scoring each baseline
        example against a centroid that already includes itself (which is
        always optimistic and would seed the adaptive threshold too high).

        For each baseline example i, builds the centroid from every OTHER
        baseline example and scores example i against that centroid - the
        same idea as leave-one-out cross-validation.
        """
        texts = [ex.text for ex in category.baseline_examples]
        if len(texts) < 3:
            raise ValueError("Need at least 3 baseline examples for leave-one-out seeding.")
        vecs = self.embedder.encode(texts)
        scores = []
        for i in range(len(texts)):
            others = np.delete(vecs, i, axis=0)
            loo_centroid = _normalize(others.mean(axis=0))
            scores.append(float(vecs[i] @ loo_centroid))
        return scores


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
