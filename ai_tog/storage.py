from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import io
from pathlib import Path
import sqlite3
import numpy as np


def _serialize_array(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, arr)
    return bio.getvalue()


def _deserialize_array(blob: bytes) -> np.ndarray:
    bio = io.BytesIO(blob)
    bio.seek(0)
    return np.load(bio)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage(ABC):
    """
    Abstract persistence interface for AI-TOG. Rest of the pipeline interacts
    only with this interface to persist and restore learned state across
    process lifecycles.
    """

    @abstractmethod
    def save_profile(
        self,
        category_id: str,
        centroid: np.ndarray,
        n_examples: int,
        version: int,
        updated_at: str | None = None,
        note: str | None = None,
    ) -> None:
        """Persist or update reference profile and optionally append a history entry."""
        raise NotImplementedError

    @abstractmethod
    def get_profile(self, category_id: str) -> dict | None:
        """
        Load persisted profile data for a category, including its version history.
        Returns dict with keys: category_id, centroid, n_examples, version, updated_at, history
        or None if not found.
        """
        raise NotImplementedError

    @abstractmethod
    def get_profile_history(self, category_id: str) -> list[dict]:
        """Return full history list for a category ordered chronologically."""
        raise NotImplementedError

    @abstractmethod
    def has_profile(self, category_id: str) -> bool:
        """Check if a reference profile exists for category_id."""
        raise NotImplementedError

    @abstractmethod
    def record_threshold_score(
        self,
        category_id: str,
        score: float,
        recorded_at: str | None = None,
    ) -> None:
        """Append a single confirmed-good score to threshold_scores."""
        raise NotImplementedError

    @abstractmethod
    def record_threshold_scores(
        self,
        category_id: str,
        scores: list[float],
        recorded_at: str | None = None,
    ) -> None:
        """Append multiple scores to threshold_scores (e.g. on initial seed)."""
        raise NotImplementedError

    @abstractmethod
    def get_threshold_scores(
        self,
        category_id: str,
        limit: int | None = None,
    ) -> list[float]:
        """
        Return threshold scores in chronological order (oldest first, newest last).
        If limit is specified, returns the most recent `limit` scores in chronological order.
        """
        raise NotImplementedError

    @abstractmethod
    def record_evaluation(
        self,
        category_id: str,
        candidate_text: str,
        semantic_similarity: float,
        structural_valid: bool,
        invariants_satisfied: bool,
        composite_score: float,
        verdict: str,
        explanation: str,
        evaluated_at: str | None = None,
    ) -> int:
        """Record an evaluation to the audit trail and return generated evaluation ID."""
        raise NotImplementedError

    @abstractmethod
    def get_evaluations(self, category_id: str | None = None) -> list[dict]:
        """Return logged evaluations ordered chronologically."""
        raise NotImplementedError

    @abstractmethod
    def record_review_resolution(
        self,
        evaluation_id: int,
        human_label: str,
        resolved_at: str | None = None,
    ) -> int:
        """Record a human review resolution ('good' or 'bad') for an evaluation."""
        raise NotImplementedError

    @abstractmethod
    def get_review_resolutions(self, evaluation_id: int | None = None) -> list[dict]:
        """Return review resolutions ordered chronologically."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close connection and release resources."""
        raise NotImplementedError


class SQLiteStorage(Storage):
    """
    SQLite persistence backend implementing the Storage interface using
    Python's standard library sqlite3 module.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS reference_profiles (
      category_id TEXT PRIMARY KEY,
      centroid BLOB NOT NULL,          -- serialized numpy array
      n_examples INTEGER NOT NULL,
      version INTEGER NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS profile_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      category_id TEXT NOT NULL REFERENCES reference_profiles(category_id),
      version INTEGER NOT NULL,
      note TEXT NOT NULL,
      recorded_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS threshold_scores (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      category_id TEXT NOT NULL,
      score REAL NOT NULL,
      recorded_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS evaluations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      category_id TEXT NOT NULL,
      candidate_text TEXT NOT NULL,
      semantic_similarity REAL NOT NULL,
      structural_valid INTEGER NOT NULL,
      invariants_satisfied INTEGER NOT NULL,
      composite_score REAL NOT NULL,
      verdict TEXT NOT NULL,
      explanation TEXT NOT NULL,
      evaluated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS review_resolutions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      evaluation_id INTEGER NOT NULL REFERENCES evaluations(id),
      human_label TEXT NOT NULL CHECK (human_label IN ('good','bad')),
      resolved_at TEXT NOT NULL
    );
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.executescript(self.SCHEMA)

    def save_profile(
        self,
        category_id: str,
        centroid: np.ndarray,
        n_examples: int,
        version: int,
        updated_at: str | None = None,
        note: str | None = None,
    ) -> None:
        ts = updated_at or _now()
        blob = _serialize_array(centroid)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO reference_profiles (category_id, centroid, n_examples, version, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(category_id) DO UPDATE SET
                  centroid = excluded.centroid,
                  n_examples = excluded.n_examples,
                  version = excluded.version,
                  updated_at = excluded.updated_at
                """,
                (category_id, blob, n_examples, version, ts),
            )
            if note:
                self._conn.execute(
                    """
                    INSERT INTO profile_history (category_id, version, note, recorded_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (category_id, version, note, ts),
                )

    def get_profile(self, category_id: str) -> dict | None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT category_id, centroid, n_examples, version, updated_at
            FROM reference_profiles
            WHERE category_id = ?
            """,
            (category_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        centroid = _deserialize_array(row["centroid"])
        history_rows = self.get_profile_history(category_id)
        history = [h["note"] for h in history_rows]

        return {
            "category_id": row["category_id"],
            "centroid": centroid,
            "n_examples": row["n_examples"],
            "version": row["version"],
            "updated_at": row["updated_at"],
            "history": history,
        }

    def get_profile_history(self, category_id: str) -> list[dict]:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT id, category_id, version, note, recorded_at
            FROM profile_history
            WHERE category_id = ?
            ORDER BY id ASC
            """,
            (category_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def has_profile(self, category_id: str) -> bool:
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT 1 FROM reference_profiles WHERE category_id = ? LIMIT 1",
            (category_id,),
        )
        return cursor.fetchone() is not None

    def record_threshold_score(
        self,
        category_id: str,
        score: float,
        recorded_at: str | None = None,
    ) -> None:
        ts = recorded_at or _now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO threshold_scores (category_id, score, recorded_at)
                VALUES (?, ?, ?)
                """,
                (category_id, float(score), ts),
            )

    def record_threshold_scores(
        self,
        category_id: str,
        scores: list[float],
        recorded_at: str | None = None,
    ) -> None:
        ts = recorded_at or _now()
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO threshold_scores (category_id, score, recorded_at)
                VALUES (?, ?, ?)
                """,
                [(category_id, float(s), ts) for s in scores],
            )

    def get_threshold_scores(
        self,
        category_id: str,
        limit: int | None = None,
    ) -> list[float]:
        cursor = self._conn.cursor()
        if limit is None:
            cursor.execute(
                """
                SELECT score FROM threshold_scores
                WHERE category_id = ?
                ORDER BY id ASC
                """,
                (category_id,),
            )
        else:
            cursor.execute(
                """
                SELECT score FROM (
                    SELECT id, score FROM threshold_scores
                    WHERE category_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) sub ORDER BY id ASC
                """,
                (category_id, limit),
            )
        return [float(row["score"]) for row in cursor.fetchall()]

    def record_evaluation(
        self,
        category_id: str,
        candidate_text: str,
        semantic_similarity: float,
        structural_valid: bool,
        invariants_satisfied: bool,
        composite_score: float,
        verdict: str,
        explanation: str,
        evaluated_at: str | None = None,
    ) -> int:
        ts = evaluated_at or _now()
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO evaluations (
                    category_id, candidate_text, semantic_similarity,
                    structural_valid, invariants_satisfied, composite_score,
                    verdict, explanation, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    category_id,
                    candidate_text,
                    float(semantic_similarity),
                    1 if structural_valid else 0,
                    1 if invariants_satisfied else 0,
                    float(composite_score),
                    str(verdict),
                    str(explanation),
                    ts,
                ),
            )
            return cursor.lastrowid

    def get_evaluations(self, category_id: str | None = None) -> list[dict]:
        cursor = self._conn.cursor()
        if category_id is None:
            cursor.execute("SELECT * FROM evaluations ORDER BY id ASC")
        else:
            cursor.execute(
                "SELECT * FROM evaluations WHERE category_id = ? ORDER BY id ASC",
                (category_id,),
            )
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            d["structural_valid"] = bool(d["structural_valid"])
            d["invariants_satisfied"] = bool(d["invariants_satisfied"])
            results.append(d)
        return results

    def record_review_resolution(
        self,
        evaluation_id: int,
        human_label: str,
        resolved_at: str | None = None,
    ) -> int:
        if human_label not in ("good", "bad"):
            raise ValueError("human_label must be 'good' or 'bad'.")
        ts = resolved_at or _now()
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO review_resolutions (evaluation_id, human_label, resolved_at)
                VALUES (?, ?, ?)
                """,
                (int(evaluation_id), human_label, ts),
            )
            return cursor.lastrowid

    def get_review_resolutions(self, evaluation_id: int | None = None) -> list[dict]:
        cursor = self._conn.cursor()
        if evaluation_id is None:
            cursor.execute("SELECT * FROM review_resolutions ORDER BY id ASC")
        else:
            cursor.execute(
                "SELECT * FROM review_resolutions WHERE evaluation_id = ? ORDER BY id ASC",
                (evaluation_id,),
            )
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        self._conn.close()


class InMemoryStorage(SQLiteStorage):
    """
    In-memory SQLite storage for fast, isolated unit testing without touching disk.
    Ensures 100% exact SQL and schema fidelity with SQLiteStorage.
    """

    def __init__(self):
        super().__init__(db_path=":memory:")
