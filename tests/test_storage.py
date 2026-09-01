import pytest
import sqlite3
from collections import deque
import numpy as np

from ai_tog.storage import Storage, SQLiteStorage, InMemoryStorage


def test_storage_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Storage()


def test_in_memory_storage_profile_crud_and_history():
    storage = InMemoryStorage()
    centroid = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    assert not storage.has_profile("cat_1")
    assert storage.get_profile("cat_1") is None

    # Save initial v1 profile
    storage.save_profile(
        category_id="cat_1",
        centroid=centroid,
        n_examples=5,
        version=1,
        updated_at="2026-09-01T10:00:00+00:00",
        note="v1 built from 5 baseline examples",
    )

    assert storage.has_profile("cat_1")
    prof = storage.get_profile("cat_1")
    assert prof is not None
    assert prof["category_id"] == "cat_1"
    assert np.allclose(prof["centroid"], centroid)
    assert prof["n_examples"] == 5
    assert prof["version"] == 1
    assert prof["updated_at"] == "2026-09-01T10:00:00+00:00"
    assert prof["history"] == ["v1 built from 5 baseline examples"]

    # Update to v2
    new_centroid = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    storage.save_profile(
        category_id="cat_1",
        centroid=new_centroid,
        n_examples=6,
        version=2,
        updated_at="2026-09-01T10:05:00+00:00",
        note="v2 incorporated 1 reviewer-approved example",
    )

    prof2 = storage.get_profile("cat_1")
    assert prof2["version"] == 2
    assert prof2["n_examples"] == 6
    assert np.allclose(prof2["centroid"], new_centroid)
    assert len(prof2["history"]) == 2
    assert prof2["history"] == [
        "v1 built from 5 baseline examples",
        "v2 incorporated 1 reviewer-approved example",
    ]

    history = storage.get_profile_history("cat_1")
    assert len(history) == 2
    assert history[0]["version"] == 1
    assert history[0]["note"] == "v1 built from 5 baseline examples"
    assert history[1]["version"] == 2
    assert history[1]["note"] == "v2 incorporated 1 reviewer-approved example"
    storage.close()


def test_threshold_scores_order_limit_and_deque_eviction():
    storage = InMemoryStorage()
    scores = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    storage.record_threshold_scores("cat_test", scores)

    # 1. All scores returned in chronological order
    all_scores = storage.get_threshold_scores("cat_test")
    assert all_scores == scores

    # 2. Most recent limit=5 scores returned in correct chronological order
    recent_5 = storage.get_threshold_scores("cat_test", limit=5)
    assert recent_5 == [0.60, 0.70, 0.80, 0.90, 1.00]

    # 3. Verify rolling window deque eviction order
    window = deque(recent_5, maxlen=5)
    assert list(window) == [0.60, 0.70, 0.80, 0.90, 1.00]

    # Append new score 1.10 -> oldest score (0.60) MUST be evicted
    window.append(1.10)
    assert list(window) == [0.70, 0.80, 0.90, 1.00, 1.10]
    assert 0.60 not in window
    storage.close()


def test_evaluations_audit_log_and_review_resolutions():
    storage = InMemoryStorage()
    eval_id1 = storage.record_evaluation(
        category_id="cat_a",
        candidate_text="Sample text 1",
        semantic_similarity=0.85,
        structural_valid=True,
        invariants_satisfied=True,
        composite_score=0.85,
        verdict="PASS",
        explanation="score >= pass cut",
    )
    assert eval_id1 == 1

    eval_id2 = storage.record_evaluation(
        category_id="cat_a",
        candidate_text="Sample text 2",
        semantic_similarity=0.60,
        structural_valid=True,
        invariants_satisfied=True,
        composite_score=0.60,
        verdict="REVIEW",
        explanation="score between review and pass cut",
    )
    assert eval_id2 == 2

    # Query evaluations
    all_evals = storage.get_evaluations()
    assert len(all_evals) == 2
    assert all_evals[0]["id"] == 1
    assert all_evals[0]["verdict"] == "PASS"
    assert all_evals[0]["structural_valid"] is True
    assert all_evals[1]["id"] == 2
    assert all_evals[1]["verdict"] == "REVIEW"

    # Record review resolution
    res_id = storage.record_review_resolution(evaluation_id=eval_id2, human_label="good")
    assert res_id == 1

    resolutions = storage.get_review_resolutions(evaluation_id=eval_id2)
    assert len(resolutions) == 1
    assert resolutions[0]["evaluation_id"] == eval_id2
    assert resolutions[0]["human_label"] == "good"

    # Test invalid human_label validation
    with pytest.raises(ValueError):
        storage.record_review_resolution(evaluation_id=eval_id2, human_label="invalid_label")

    storage.close()


def test_foreign_key_enforcement_in_memory():
    storage = InMemoryStorage()
    # Inserting review_resolution for nonexistent evaluation_id must raise IntegrityError
    with pytest.raises(sqlite3.IntegrityError):
        storage.record_review_resolution(evaluation_id=999, human_label="good")

    # Inserting profile_history for nonexistent category_id must raise IntegrityError
    with pytest.raises(sqlite3.IntegrityError):
        with storage._conn:
            storage._conn.execute(
                "INSERT INTO profile_history (category_id, version, note, recorded_at) VALUES (?, ?, ?, ?)",
                ("nonexistent_cat", 1, "note", "2026-09-01T10:00:00+00:00"),
            )
    storage.close()


def test_sqlite_file_storage_roundtrip(tmp_path):
    db_file = tmp_path / "test_storage.db"

    # Phase 1: Open storage, write data, close
    storage1 = SQLiteStorage(db_file)
    centroid = np.array([0.5, -0.5, 0.707], dtype=np.float64)
    storage1.save_profile("cat_rt", centroid, 5, 1, note="initial baseline")
    storage1.record_threshold_scores("cat_rt", [0.75, 0.80, 0.85])
    eval_id = storage1.record_evaluation(
        category_id="cat_rt",
        candidate_text="Candidate paraphrase",
        semantic_similarity=0.78,
        structural_valid=True,
        invariants_satisfied=True,
        composite_score=0.78,
        verdict="REVIEW",
        explanation="Review needed",
    )
    storage1.record_review_resolution(eval_id, "good")
    storage1.save_profile("cat_rt", centroid * 1.1, 6, 2, note="updated v2")
    storage1.close()

    # Phase 2: Open a fresh SQLiteStorage instance on the same file
    storage2 = SQLiteStorage(db_file)

    # 1. Profile and history preserved
    prof = storage2.get_profile("cat_rt")
    assert prof is not None
    assert prof["version"] == 2
    assert prof["n_examples"] == 6
    assert np.allclose(prof["centroid"], centroid * 1.1)
    assert prof["history"] == ["initial baseline", "updated v2"]

    # 2. Threshold scores preserved
    scores = storage2.get_threshold_scores("cat_rt")
    assert scores == [0.75, 0.80, 0.85]

    # 3. Evaluations and resolutions preserved
    evals = storage2.get_evaluations("cat_rt")
    assert len(evals) == 1
    assert evals[0]["id"] == eval_id
    assert evals[0]["verdict"] == "REVIEW"

    resolutions = storage2.get_review_resolutions(eval_id)
    assert len(resolutions) == 1
    assert resolutions[0]["human_label"] == "good"

    # 4. Foreign key constraint enforced on reload
    with pytest.raises(sqlite3.IntegrityError):
        storage2.record_review_resolution(evaluation_id=9999, human_label="bad")

    storage2.close()
