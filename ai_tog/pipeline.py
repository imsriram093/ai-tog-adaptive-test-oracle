from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from .baseline import TestCategory, load_categories
from .embedding.base import Embedder
from .embedding.tfidf_embedder import TfidfEmbedder
from .oracle import OracleEngine
from .profile import ProfileBuilder, ReferenceProfile
from .storage import Storage, SQLiteStorage
from .threshold import AdaptiveThresholdTracker
from .verdict import EvaluationResult, VerdictAggregator


@dataclass
class Pipeline:
    categories: dict[str, TestCategory]
    profiles: dict[str, ReferenceProfile]
    embedder: Embedder
    profile_builder: ProfileBuilder
    oracle: OracleEngine
    threshold_tracker: AdaptiveThresholdTracker
    verdict_aggregator: VerdictAggregator
    storage: Storage | None = None

    def evaluate(self, category_id: str, candidate_text: str) -> EvaluationResult:
        profile = self.profiles[category_id]
        signals = self.oracle.score(profile, candidate_text)
        return self.verdict_aggregator.decide(category_id, candidate_text, signals)

    def resolve_review(self, result: EvaluationResult, human_label: str) -> None:
        profile = self.profiles[result.category_id]
        self.verdict_aggregator.resolve_review(result, human_label, self.profile_builder, profile)


def build_pipeline(
    data_dir: str | Path,
    embedder_backend: str = "tfidf",
    window_size: int = 50,
    pass_z: float = 1.0,
    fail_z: float = 3.0,
    storage: Storage | str | Path | None = None,
) -> Pipeline:
    """
    Loads all Stage-1 categories from `data_dir`, builds Stage-2 reference
    profiles, seeds Stage-4's adaptive threshold from leave-one-out baseline
    scores, and returns a ready-to-use Pipeline.

    embedder_backend: "tfidf" (default, no network access required - used in
    CI/this sandbox) or "semantic" (real sentence-transformers embeddings;
    requires the optional 'semantic' extra and network access to the model
    hub on first run).

    storage: Optional Storage backend (e.g. SQLiteStorage, InMemoryStorage, or a path
    to a .db file). If provided, existing profiles and threshold history are loaded
    from storage across restarts. If None, runs purely in-memory.
    """
    categories = load_categories(data_dir)
    if not categories:
        raise ValueError(f"No category JSON files found in {data_dir}")

    if isinstance(storage, (str, Path)):
        storage = SQLiteStorage(storage)

    if embedder_backend == "tfidf":
        corpus = [ex.text for cat in categories.values() for ex in cat.baseline_examples]
        embedder: Embedder = TfidfEmbedder(corpus)
    elif embedder_backend == "semantic":
        from .embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()
    else:
        raise ValueError(f"Unknown embedder_backend: {embedder_backend!r}")

    profile_builder = ProfileBuilder(embedder, storage=storage)
    oracle = OracleEngine(embedder)
    threshold_tracker = AdaptiveThresholdTracker(
        window_size=window_size,
        pass_z=pass_z,
        fail_z=fail_z,
        storage=storage,
    )
    verdict_aggregator = VerdictAggregator(threshold_tracker, storage=storage)

    profiles = {}
    for category_id, category in categories.items():
        if storage is not None and storage.has_profile(category_id):
            # Category already has persisted history -> ONLY load from storage
            stored = storage.get_profile(category_id)
            if stored is not None:
                profile = ReferenceProfile(
                    category_id=category_id,
                    centroid=stored["centroid"],
                    n_examples=stored["n_examples"],
                    min_length=category.min_length,
                    max_length=category.max_length,
                    required_terms=list(category.required_terms),
                    banned_terms=list(category.banned_terms),
                    created_at=stored["updated_at"],
                    version=stored["version"],
                    history=stored["history"],
                )
                profiles[category_id] = profile
                threshold_tracker.load_from_storage(category_id)
                continue

        # No persisted state yet -> bootstrap from JSON + leave-one-out seeding
        profile = profile_builder.build(category)
        profiles[category_id] = profile
        seed_scores = profile_builder.leave_one_out_seed_scores(category)
        threshold_tracker.seed(category_id, seed_scores)

        if storage is not None:
            storage.save_profile(
                category_id=category_id,
                centroid=profile.centroid,
                n_examples=profile.n_examples,
                version=profile.version,
                updated_at=profile.created_at,
                note=profile.history[0] if profile.history else f"v1 built from {profile.n_examples} baseline examples",
            )
            storage.record_threshold_scores(category_id, seed_scores)

    return Pipeline(
        categories=categories,
        profiles=profiles,
        embedder=embedder,
        profile_builder=profile_builder,
        oracle=oracle,
        threshold_tracker=threshold_tracker,
        verdict_aggregator=verdict_aggregator,
        storage=storage,
    )

