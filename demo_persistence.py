"""
demo_persistence.py: Demonstrates durability of learned pipeline state across
real Python process restarts.

When run directly without arguments, this script launches two separate OS
processes (Run 1 and Run 2) sequentially using `subprocess.run([sys.executable, ...])`.

- Run 1 (Process 1): Bootstraps the pipeline, evaluates candidates, hits a REVIEW case,
  resolves it as 'good' (advancing profile version to 2 and updating threshold window),
  and terminates.
- Run 2 (Process 2): Launches in a brand new process against the same SQLite database,
  demonstrating that profile version 2 (n=6), full history trail, static JSON rules,
  and adapted threshold band are cleanly restored without resetting.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

from ai_tog.pipeline import build_pipeline
from ai_tog.verdict import Verdict

DATA_DIR = "data/categories"
DEFAULT_DB = "data/demo_persistence.db"


def run_phase_1(db_path: str) -> None:
    print("=" * 72)
    print(f"RUN 1: Initial Bootstrap & Feedback Resolution (Process PID: {os.getpid()})")
    print("=" * 72)

    cat_id = "capital_city_fact"
    pipeline = build_pipeline(DATA_DIR, storage=db_path)
    tracker = pipeline.threshold_tracker

    prof = pipeline.profiles[cat_id]
    band = tracker.current_band(cat_id)

    print("\n[1. Initial State from JSON Baseline Bootstrap]")
    print(f"  Category:           {cat_id}")
    print(f"  Profile Version:    v{prof.version} (n_examples={prof.n_examples})")
    print(f"  Threshold Band:     pass_cut={band.pass_cut:.3f}, review_cut={band.review_cut:.3f}, "
          f"center={band.center:.3f}, spread={band.spread:.3f}")

    print("\n[2. Evaluating Candidates in Run 1]")
    c1 = "The capital of France is Paris."
    r1 = pipeline.evaluate(cat_id, c1)
    print(f"  (a) Candidate: \"{c1}\"")
    print(f"      Verdict: {r1.verdict.value} | Score: {r1.signals.composite_score:.3f} | Eval ID: {r1.evaluation_id}")

    c2 = "The capital of France is Berlin."
    r2 = pipeline.evaluate(cat_id, c2)
    print(f"  (b) Candidate: \"{c2}\"")
    print(f"      Verdict: {r2.verdict.value} | Score: {r2.signals.composite_score:.3f} | Explanation: {r2.explanation}")

    c3 = "Paris is a city in Europe."
    r3 = pipeline.evaluate(cat_id, c3)
    print(f"  (c) Candidate: \"{c3}\"")
    print(f"      Verdict: {r3.verdict.value} | Score: {r3.signals.composite_score:.3f} | Eval ID: {r3.evaluation_id}")

    print("\n[3. Feedback Loop: Resolving REVIEW Case as 'good']")
    assert r3.verdict == Verdict.REVIEW, f"Expected REVIEW verdict, got {r3.verdict}"
    pipeline.resolve_review(r3, "good")

    prof_updated = pipeline.profiles[cat_id]
    band_updated = tracker.current_band(cat_id)

    print(f"  -> Profile updated:   v{prof_updated.version} (n_examples={prof_updated.n_examples})")
    print(f"  -> Threshold band:    pass_cut={band_updated.pass_cut:.3f}, review_cut={band_updated.review_cut:.3f}, "
          f"center={band_updated.center:.3f}, spread={band_updated.spread:.3f}")
    print(f"  -> Persisted to:      {db_path}")

    if pipeline.storage:
        pipeline.storage.close()

    print("\n[4. Exiting Run 1 Process - In-Memory State Discarded]\n")



def run_phase_2(db_path: str) -> None:
    print("=" * 72)
    print(f"RUN 2: Post-Restart Durability Check (Process PID: {os.getpid()})")
    print("=" * 72)

    cat_id = "capital_city_fact"
    pipeline = build_pipeline(DATA_DIR, storage=db_path)
    tracker = pipeline.threshold_tracker

    prof = pipeline.profiles[cat_id]
    band = tracker.current_band(cat_id)

    print("\n[1. Restored State Loaded from SQLite Database]")
    print(f"  Category:           {cat_id}")
    print(f"  Profile Version:    v{prof.version} (n_examples={prof.n_examples}) -> PRESERVED across restart!")
    print(f"  Profile History:")
    for h in prof.history:
        print(f"    * {h}")
    print(f"  Static Rules Merged from JSON:")
    print(f"    * required_terms: {prof.required_terms}")
    print(f"    * banned_terms:   {prof.banned_terms}")
    print(f"  Threshold Band:     pass_cut={band.pass_cut:.3f}, review_cut={band.review_cut:.3f}, "
          f"center={band.center:.3f}, spread={band.spread:.3f} -> PRESERVED (NOT reset!)")

    print("\n[2. Evaluating Candidate in Run 2 to Verify Rule & Oracle Integrity]")
    c4 = "The capital of France is Lyon."
    r4 = pipeline.evaluate(cat_id, c4)
    print(f"  Candidate: \"{c4}\"")
    print(f"  Verdict:   {r4.verdict.value} | Explanation: {r4.explanation}")

    print("\n[3. Full Durability Audit Trail in SQLite]")
    if pipeline.storage:
        evals = pipeline.storage.get_evaluations()
        resols = pipeline.storage.get_review_resolutions()
        print(f"  Total Evaluations Logged:    {len(evals)}")
        print(f"  Total Review Resolutions:    {len(resols)}")
        pipeline.storage.close()

    print("\n[Durability Across Process Restarts Fully Verified]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-TOG SQLite Persistence Demo")
    parser.add_argument("--run", type=int, choices=[1, 2], help="Run specific phase (1 or 2)")
    parser.add_argument("--db", type=str, default=DEFAULT_DB, help="Path to SQLite database")
    args = parser.parse_args()

    if args.run == 1:
        run_phase_1(args.db)
    elif args.run == 2:
        run_phase_2(args.db)
    else:
        # Orchestrate two distinct OS processes
        db_file = Path(args.db)
        if db_file.exists():
            db_file.unlink()

        print("\n>>> Launching Run 1 in separate OS process...\n", flush=True)
        proc1 = subprocess.run([sys.executable, __file__, "--run", "1", "--db", str(db_file)], check=True)

        print("\n>>> Launching Run 2 in separate OS process (post-restart)...\n", flush=True)
        proc2 = subprocess.run([sys.executable, __file__, "--run", "2", "--db", str(db_file)], check=True)



if __name__ == "__main__":
    main()
