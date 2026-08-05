"""
ai_tog: Adaptive Test Oracle Generation for AI-Integrated Software Modules.

A five-stage pipeline for testing software modules that embed non-deterministic
AI/ML components, where a single "golden" expected output does not exist.

Stages
------
1. baseline    - curated known-good example answers per test category
2. profile     - reference profile builder (embedding centroid + structural rules)
3. oracle      - multi-signal scoring engine (semantic + structural + invariant)
4. threshold   - adaptive pass/review/fail boundaries from rolling score history
5. verdict     - verdict aggregation + human-in-the-loop feedback into stage 2
"""

__version__ = "0.1.0"
