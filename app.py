"""
app.py: Local HTTP Server and REST API for AI-TOG Interactive Web Dashboard.
Uses Python standard library ONLY (http.server, json, sqlite3, threading, pathlib).
Binds to 127.0.0.1:8000 with thread-safe SQLite operations.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sqlite3
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_tog.pipeline import build_pipeline, Pipeline
from ai_tog.verdict import Verdict, EvaluationResult
from ai_tog.oracle import STRUCTURAL_VIOLATION_PENALTY, INVARIANT_VIOLATION_PENALTY
import benchmark

DATA_DIR = PROJECT_ROOT / "data" / "categories"
DB_PATH = PROJECT_ROOT / "data" / "ui_demo.db"
WEB_DIR = PROJECT_ROOT / "web"

# Thread lock for thread-safe SQLite & pipeline state mutations
PIPELINE_LOCK = threading.Lock()

# Global pipeline instance
_pipeline: Pipeline | None = None

# Cached benchmark results
_benchmark_cache: dict[str, Any] | None = None


def get_pipeline(rebuild: bool = False) -> Pipeline:
    global _pipeline
    with PIPELINE_LOCK:
        if rebuild or _pipeline is None:
            if _pipeline is not None and _pipeline.storage is not None:
                try:
                    _pipeline.storage.close()
                except Exception:
                    pass
            _pipeline = build_pipeline(DATA_DIR, storage=str(DB_PATH))
        return _pipeline


def run_cached_benchmark() -> dict[str, Any]:
    global _benchmark_cache
    if _benchmark_cache is not None:
        return _benchmark_cache

    with PIPELINE_LOCK:
        # Run benchmark on a clean standalone pipeline instance to not mutate ui_demo.db
        bench_pipe = build_pipeline(DATA_DIR)
        cases_out = []
        naive_correct = ai_tog_correct = 0
        naive_fr = naive_fa = ai_tog_fr = ai_tog_fa = 0
        cat_stats: dict[str, dict[str, int]] = {}

        for case in benchmark.CASES:
            cat = bench_pipe.categories[case.category_id]
            baseline_texts = [ex.text for ex in cat.baseline_examples]
            naive_verdict = benchmark.naive_exact_match_verdict(baseline_texts, case.text)
            res = bench_pipe.evaluate(case.category_id, case.text)
            ai_tog_verdict = "accept" if res.verdict in (Verdict.PASS, Verdict.REVIEW) else "reject"

            naive_ok = (naive_verdict == case.expected)
            ai_tog_ok = (ai_tog_verdict == case.expected)
            naive_correct += int(naive_ok)
            ai_tog_correct += int(ai_tog_ok)

            if case.expected == "accept" and naive_verdict == "reject":
                naive_fr += 1
            if case.expected == "reject" and naive_verdict == "accept":
                naive_fa += 1
            if case.expected == "accept" and ai_tog_verdict == "reject":
                ai_tog_fr += 1
            if case.expected == "reject" and ai_tog_verdict == "accept":
                ai_tog_fa += 1

            s = cat_stats.setdefault(case.category_id, {
                "total": 0, "naive_correct": 0, "ai_tog_correct": 0,
                "naive_fr": 0, "naive_fa": 0, "ai_tog_fr": 0, "ai_tog_fa": 0
            })
            s["total"] += 1
            s["naive_correct"] += int(naive_ok)
            s["ai_tog_correct"] += int(ai_tog_ok)
            if case.expected == "accept" and naive_verdict == "reject":
                s["naive_fr"] += 1
            if case.expected == "reject" and naive_verdict == "accept":
                s["naive_fa"] += 1
            if case.expected == "accept" and ai_tog_verdict == "reject":
                s["ai_tog_fr"] += 1
            if case.expected == "reject" and ai_tog_verdict == "accept":
                s["ai_tog_fa"] += 1

            cases_out.append({
                "category_id": case.category_id,
                "text": case.text,
                "label": case.label,
                "expected": case.expected,
                "naive_verdict": naive_verdict,
                "ai_tog_verdict": ai_tog_verdict,
                "ai_tog_raw_verdict": res.verdict.value,
                "score": round(res.signals.composite_score, 4),
                "semantic": round(res.signals.semantic_similarity, 4),
                "structural_valid": res.signals.structural_valid,
                "invariants_satisfied": res.signals.invariants_satisfied,
                "violations": res.signals.violations,
                "explanation": res.explanation,
                "is_naive_correct": naive_ok,
                "is_ai_tog_correct": ai_tog_ok,
            })

        total = len(benchmark.CASES)
        _benchmark_cache = {
            "total_cases": total,
            "naive": {
                "correct": naive_correct,
                "accuracy": round(naive_correct / total * 100, 1),
                "false_rejects": naive_fr,
                "false_accepts": naive_fa,
            },
            "ai_tog": {
                "correct": ai_tog_correct,
                "accuracy": round(ai_tog_correct / total * 100, 1),
                "false_rejects": ai_tog_fr,
                "false_accepts": ai_tog_fa,
            },
            "category_breakdown": cat_stats,
            "cases": cases_out,
            "known_limitations": [
                {
                    "case": "Versailles relocation ('The capital of France used to be Paris but is now Versailles.')",
                    "category": "capital_city_fact",
                    "observed": "False Accept (PASS, score 0.652)",
                    "why": "Semantic similarity alone cannot distinguish past vs current truth when 'paris' required term is present and 'versailles' is not in static banned list."
                },
                {
                    "case": "Partial refund ('Returns within 30 days get a partial refund of 50%.')",
                    "category": "refund_policy_explanation",
                    "observed": "False Accept (PASS, score 0.366)",
                    "why": "Mentions '30 days' and required keywords; lexical TF-IDF does not capture semantic contradiction of '50%' without invariant rule."
                },
                {
                    "case": "Gift-card restriction ('We offer a 30-day money-back guarantee, but only for items bought with a gift card.')",
                    "category": "refund_policy_explanation",
                    "observed": "False Review (REVIEW, score 0.179)",
                    "why": "Score lands in adaptive review zone; correctly flags for human auditor review instead of silently passing."
                },
                {
                    "case": "Number word normalization ('Within thirty days of purchase...')",
                    "category": "refund_policy_explanation",
                    "observed": "False Reject (FAIL, missing '30 day')",
                    "why": "Invariant rule normalizes punctuation and hyphens ('30-day' -> '30 day') but does not normalize spelled-out numbers ('thirty' != '30'). Documented design boundary."
                }
            ]
        }
        return _benchmark_cache


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server_version = "AITOG-Observatory/1.0"

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        if path == "/api/categories":
            self.handle_get_categories()
        elif path == "/api/audit_logs":
            self.handle_get_audit_logs()
        elif path == "/api/benchmark":
            self.handle_get_benchmark()
        else:
            self.serve_static_file(path)

    def do_POST(self) -> None:
        path = self.path.split("?")[0]

        if path == "/api/evaluate":
            self.handle_post_evaluate()
        elif path == "/api/resolve_review":
            self.handle_post_resolve_review()
        elif path == "/api/reset":
            self.handle_post_reset()
        else:
            self.send_json_response({"error": "Endpoint not found"}, status=404)

    # ---------------- API HANDLERS ----------------

    def handle_get_categories(self) -> None:
        pipe = get_pipeline()
        with PIPELINE_LOCK:
            categories_data = []
            for cat_id, cat in pipe.categories.items():
                prof = pipe.profiles[cat_id]
                band = pipe.threshold_tracker.current_band(cat_id)
                categories_data.append({
                    "category_id": cat_id,
                    "prompt": cat.prompt,
                    "min_length": cat.min_length,
                    "max_length": cat.max_length,
                    "required_terms": cat.required_terms,
                    "banned_terms": cat.banned_terms,
                    "baseline_examples_count": len(cat.baseline_examples),
                    "profile": {
                        "version": prof.version,
                        "n_examples": prof.n_examples,
                        "created_at": prof.created_at,
                        "history": prof.history,
                    },
                    "band": {
                        "pass_cut": round(band.pass_cut, 4),
                        "review_cut": round(band.review_cut, 4),
                        "center": round(band.center, 4),
                        "spread": round(band.spread, 4),
                        "is_cold_start": band.is_cold_start,
                    }
                })

            self.send_json_response({
                "storage_mode": "SQLite Persistent",
                "db_path": str(DB_PATH.relative_to(PROJECT_ROOT)),
                "categories": categories_data,
            })

    def handle_post_evaluate(self) -> None:
        data = self.read_json_body()
        if not data or "category_id" not in data or "candidate_text" not in data:
            self.send_json_response({"error": "Missing category_id or candidate_text"}, status=400)
            return

        cat_id = str(data["category_id"]).strip()
        text = str(data["candidate_text"]).strip()
        if not text:
            self.send_json_response({"error": "Candidate text cannot be empty"}, status=400)
            return

        pipe = get_pipeline()
        with PIPELINE_LOCK:
            if cat_id not in pipe.categories:
                self.send_json_response({"error": f"Unknown category: {cat_id}"}, status=404)
                return

            res: EvaluationResult = pipe.evaluate(cat_id, text)
            prof = pipe.profiles[cat_id]

            payload = {
                "category_id": res.category_id,
                "candidate_text": res.candidate_text,
                "evaluation_id": res.evaluation_id,
                "verdict": res.verdict.value,
                "explanation": res.explanation,
                "signals": {
                    "composite_score": round(res.signals.composite_score, 4),
                    "semantic_similarity": round(res.signals.semantic_similarity, 4),
                    "structural_valid": res.signals.structural_valid,
                    "invariants_satisfied": res.signals.invariants_satisfied,
                    "violations": res.signals.violations,
                },
                "penalties": {
                    "structural_violation_penalty": STRUCTURAL_VIOLATION_PENALTY,
                    "invariant_violation_penalty": INVARIANT_VIOLATION_PENALTY,
                    "applied_structural": not res.signals.structural_valid,
                    "applied_invariant": not res.signals.invariants_satisfied,
                },
                "band": {
                    "pass_cut": round(res.band.pass_cut, 4),
                    "review_cut": round(res.band.review_cut, 4),
                    "center": round(res.band.center, 4),
                    "spread": round(res.band.spread, 4),
                    "is_cold_start": res.band.is_cold_start,
                },
                "profile": {
                    "version": prof.version,
                    "n_examples": prof.n_examples,
                }
            }
            self.send_json_response(payload)

    def handle_post_resolve_review(self) -> None:
        data = self.read_json_body()
        if not data or "evaluation_id" not in data or "human_label" not in data:
            self.send_json_response({"error": "Missing evaluation_id or human_label"}, status=400)
            return

        eval_id = int(data["evaluation_id"])
        human_label = str(data["human_label"]).strip().lower()
        if human_label not in ("good", "bad"):
            self.send_json_response({"error": "human_label must be 'good' or 'bad'"}, status=400)
            return

        pipe = get_pipeline()
        with PIPELINE_LOCK:
            if pipe.storage is None:
                self.send_json_response({"error": "Storage backend not available"}, status=500)
                return

            evals = [e for e in pipe.storage.get_evaluations() if e["id"] == eval_id]
            if not evals:
                self.send_json_response({"error": f"Evaluation id {eval_id} not found"}, status=404)
                return

            eval_record = evals[0]
            cat_id = eval_record["category_id"]
            if eval_record["verdict"] != "REVIEW":
                self.send_json_response({"error": f"Evaluation {eval_id} verdict is '{eval_record['verdict']}', expected 'REVIEW'"}, status=400)
                return

            # Reconstruct evaluation result object for resolve_review
            from ai_tog.oracle import SignalScores
            from ai_tog.threshold import ThresholdBand

            band = pipe.threshold_tracker.current_band(cat_id)
            signals = SignalScores(
                semantic_similarity=eval_record["semantic_similarity"],
                structural_valid=eval_record["structural_valid"],
                invariants_satisfied=eval_record["invariants_satisfied"],
                violations=[],
            )
            eval_res = EvaluationResult(
                category_id=cat_id,
                candidate_text=eval_record["candidate_text"],
                signals=signals,
                band=band,
                verdict=Verdict.REVIEW,
                explanation=eval_record["explanation"],
                evaluation_id=eval_id,
            )

            pipe.resolve_review(eval_res, human_label)
            prof = pipe.profiles[cat_id]
            new_band = pipe.threshold_tracker.current_band(cat_id)

            self.send_json_response({
                "success": True,
                "evaluation_id": eval_id,
                "category_id": cat_id,
                "human_label": human_label,
                "profile": {
                    "version": prof.version,
                    "n_examples": prof.n_examples,
                    "history": prof.history,
                },
                "band": {
                    "pass_cut": round(new_band.pass_cut, 4),
                    "review_cut": round(new_band.review_cut, 4),
                    "center": round(new_band.center, 4),
                    "spread": round(new_band.spread, 4),
                    "is_cold_start": new_band.is_cold_start,
                }
            })

    def handle_get_audit_logs(self) -> None:
        pipe = get_pipeline()
        with PIPELINE_LOCK:
            if pipe.storage is None:
                self.send_json_response({"evaluations": [], "resolutions": [], "history": {}})
                return

            evaluations = pipe.storage.get_evaluations()
            resolutions = pipe.storage.get_review_resolutions()

            history: dict[str, list[dict]] = {}
            for cat_id in pipe.categories.keys():
                history[cat_id] = pipe.storage.get_profile_history(cat_id)

            self.send_json_response({
                "evaluations": evaluations,
                "resolutions": resolutions,
                "history": history,
            })

    def handle_get_benchmark(self) -> None:
        bench_data = run_cached_benchmark()
        self.send_json_response(bench_data)

    def handle_post_reset(self) -> None:
        global _pipeline
        with PIPELINE_LOCK:
            if _pipeline is not None and _pipeline.storage is not None:
                try:
                    _pipeline.storage.close()
                except Exception:
                    pass
            _pipeline = None

            # Remove SQLite database file
            if DB_PATH.exists():
                try:
                    DB_PATH.unlink()
                except Exception as e:
                    self.send_json_response({"error": f"Failed to remove DB: {e}"}, status=500)
                    return

            # Rebuild fresh pipeline
            _pipeline = build_pipeline(DATA_DIR, storage=str(DB_PATH))

        self.send_json_response({"success": True, "message": "Demo state reset successfully to baseline"})

    # ---------------- STATIC FILE SERVING ----------------

    def serve_static_file(self, path: str) -> None:
        if path in ("", "/"):
            path = "/index.html"

        # Sanitize path to prevent directory traversal
        clean_rel = path.lstrip("/").replace("\\", "/")
        target_path = (WEB_DIR / clean_rel).resolve()

        if not str(target_path).startswith(str(WEB_DIR.resolve())) or not target_path.is_file():
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"404 Not Found")
            return

        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
        }
        ct = content_types.get(target_path.suffix.lower(), "application/octet-stream")

        try:
            with open(target_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"500 Internal Error: {e}".encode("utf-8"))

    # ---------------- HELPERS ----------------

    def read_json_body(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return None
            raw = self.wfile if False else self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def send_json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep terminal log clean and compact
        sys.stderr.write(f"[HTTP] {self.address_string()} - {format % args}\n")


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    # Ensure web directory exists
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure data directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    print(f"[AI-TOG] Bootstrapping pipeline with database: {DB_PATH.name}...")
    get_pipeline()
    print("[AI-TOG] Warm-up complete. Pipeline ready.")

    server = HTTPServer((host, port), DashboardRequestHandler)
    print("=" * 68)
    print(f"  AI-TOG Observatory Dashboard Running")
    print(f"  Local URL:  http://{host}:{port}")
    print(f"  Bound To:   {host}:{port} (Single-threaded SQLite-safe)")
    print("  Press Ctrl+C to stop.")
    print("=" * 68)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[AI-TOG] Stopping server...")
    finally:
        server.server_close()
        if _pipeline is not None and _pipeline.storage is not None:
            _pipeline.storage.close()
        print("[AI-TOG] Server stopped cleanly.")


if __name__ == "__main__":
    run_server()
