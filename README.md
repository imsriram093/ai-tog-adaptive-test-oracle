# AI-TOG — Adaptive Test Oracle Generation for AI-Integrated Software Modules

A working implementation of the five-stage pipeline for testing software
modules that embed non-deterministic AI/ML components, where a single
"golden" expected output doesn't exist.

## Why this exists

Once a software module embeds an LLM or other ML component, the same input
can legitimately produce different but equally correct outputs across runs.
Exact-match testing breaks in this setting: it either gets disabled
(silently losing coverage) or replaced with brittle substring checks (high
false-positive *and* false-negative rates). This project builds an oracle
that scores a *region* of acceptable outputs instead of a single expected
string, and adapts its pass/fail boundary from real execution data instead
of a hand-tuned constant.

## The five stages

| Stage | Module | What it does |
|---|---|---|
| 1. Baseline | `ai_tog/baseline.py` | Curated, QA-approved known-good answers per test category |
| 2. Reference profile | `ai_tog/profile.py` | Embedding centroid + structural/domain rules built from the baseline |
| 3. Oracle engine | `ai_tog/oracle.py` | Multi-signal scoring: semantic similarity + structural validity + invariants |
| 4. Adaptive threshold | `ai_tog/threshold.py` | Rolling-window, control-chart-style pass/review/fail boundaries + drift detection |
| 5. Verdict + feedback | `ai_tog/verdict.py` | Final verdict, plus the loop that folds human-approved reviews back into Stages 2 & 4 |

`ai_tog/pipeline.py` wires all five together.

## Quickstart

```bash
pip install -e .              # sandbox-safe TF-IDF embedding backend, no network needed
pytest tests/ -v               # 22 unit tests across all five stages
python benchmark.py            # adaptive oracle vs naive exact-match, on held-out cases
python demo.py                 # watch the threshold band adapt + drift detection fire
```

### Using real semantic embeddings instead of the TF-IDF fallback

This sandbox and CI environment can't reach the model hub, so everything
above runs on `TfidfEmbedder` - a lexical, dependency-light stand-in that
implements the same `Embedder` interface as the real backend. On your own
machine, with internet access:

```bash
pip install -e ".[semantic]"
```

```python
from ai_tog.pipeline import build_pipeline
pipeline = build_pipeline("data/categories", embedder_backend="semantic")
```

This swaps in `SentenceTransformerEmbedder` (`all-MiniLM-L6-v2` by default) -
no other code changes anywhere in the pipeline, because every stage depends
only on the `Embedder` interface, never a concrete model.

## Real benchmark result (TF-IDF backend, 11 held-out cases across 3 categories)

```
Naive exact-match oracle : 5/11 correct   false-rejects=6  false-accepts=0
AI-TOG adaptive oracle   : 11/11 correct   false-rejects=0  false-accepts=0
```

Every false-reject from the naive oracle is a valid, differently-worded
answer it wrongly failed. Both oracles correctly reject 0 bad answers as
good (0 false-accepts) - AI-TOG isn't just more permissive, it's actually
discriminating.

## Adding a new test category

Drop a new JSON file into `data/categories/`:

```json
{
  "category_id": "your_category",
  "prompt": "The question or task this category covers",
  "min_length": 5,
  "max_length": 60,
  "required_terms": ["must appear"],
  "banned_terms": ["must not appear"],
  "baseline_examples": [
    {"text": "a QA-approved good answer", "approved_by": "your_name"}
  ]
}
```

At least 3 baseline examples are required (5+ recommended) so Stage 2 can
compute a meaningful centroid and spread.

## What's deliberately out of scope (see project report, Section 3.4)

- Retraining or fine-tuning the AI/ML model embedded in the module under test
- General-purpose testing of standalone ML models (a separate, well-studied problem)
- Production-grade, multi-tenant deployment of the oracle as a hosted service

## Project team

K Sriram (24BYB0151), Piyush Patani (24BYB0146) — BCSE301L/P Software Engineering, VIT Vellore
