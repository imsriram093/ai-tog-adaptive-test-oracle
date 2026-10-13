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
| Storage | `ai_tog/storage.py` | Persists reference profiles, threshold score history, evaluation audit logs, and review resolutions across process restarts via SQLite; Storage ABC also has an InMemoryStorage implementation for tests. |

`ai_tog/pipeline.py` wires all stages together.

## Quickstart

```bash
pip install -e .              # sandbox-safe TF-IDF embedding backend, no network needed
pytest tests/ -v               # 30 unit tests across all stages and persistence layer
python benchmark.py            # adaptive oracle vs naive exact-match, on held-out cases
python demo.py                 # watch the threshold band adapt + drift detection fire
python demo_persistence.py     # verify learned state durability across process restarts
```

### Enabling SQLite persistence across restarts

By default, `build_pipeline(data_dir)` runs purely in-memory with `storage=None`, preserving the exact prior behavior with zero disk side effects. To make learned state durable across process lifecycles, pass a database path or storage backend:

```python
from ai_tog.pipeline import build_pipeline
pipeline = build_pipeline("data/categories", storage="ai_tog.db")
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

## Real benchmark result (TF-IDF backend, 24 held-out cases across 3 categories)

```
Naive exact-match oracle : 12/24 correct   false-rejects=12  false-accepts=0
AI-TOG adaptive oracle   : 21/24 correct   false-rejects=0   false-accepts=3
```

Every false-reject from the naive oracle is a valid, differently-worded
answer it wrongly failed. AI-TOG's 3 false-accepts are same-topic, different-
fact answers where embedding similarity alone cannot detect the factual error —
a documented, understood limitation (see `evidence.md`, Section 3).

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

## Interactive Web Dashboard (Viva & Demonstration)

A local, offline-capable dashboard built with Python standard library and vanilla HTML/CSS/JS for viva defense and live demonstration.

### Running the Dashboard

```bash
python app.py
```
Open your browser at **`http://127.0.0.1:8000`**.

### How to Reset State for a Clean Demo

Click the **Reset Demo State** button in the **Audit Log** tab, or run:
```bash
curl -X POST http://127.0.0.1:8000/api/reset
```
This safely deletes `data/ui_demo.db` and rebuilds the reference profiles from Stage 1 JSON baselines.

### Viva Presets & Verified Outputs

| Preset Group | Case Description | Category | Real Verdict | Real Score | Why / Mechanism |
|---|---|---|---|---|---|
| **Valid** | Reordered clauses | `refund_policy_explanation` | **PASS** | 0.539 | Lexical match within policy; score >= pass cut (0.213) |
| **Valid** | Conversational register | `refund_policy_explanation` | **PASS** | 0.397 | Valid paraphrase; score >= pass cut (0.213) |
| **Valid** | Extra detail (em dash) | `capital_city_fact` | **PASS** | 0.798 | True fact with elaboration; score >= pass cut (0.382) |
| **Caught** | Wrong window (90 days) | `refund_policy_explanation` | **FAIL** | 0.016 | Hard invariant: missing '30 day', contains 'store credit only' |
| **Caught** | Attempted essay task | `out_of_scope_decline` | **FAIL** | 0.000 | Hard invariant: contains banned 'here\'s an essay' |
| **Caught** | Marseille (Wrong city) | `capital_city_fact` | **FAIL** | 0.479 | Hard invariant: missing required term 'paris' |
| **Feedback Loop** | Terse paraphrase (9 words) | `refund_policy_explanation` | **REVIEW** | 0.139 | Structural penalty: too short (< 10 words); lands in review zone |
| **Feedback Loop** | Gift-card restriction | `refund_policy_explanation` | **REVIEW** | 0.179 | Score lands between review cut (0.068) and pass cut (0.213) |
| **Feedback Loop** | Paris is in Europe | `capital_city_fact` | **REVIEW** | 0.311 | True geographic detail; score lands between 0.133 and 0.382 |
| **Limitation** | Versailles relocation | `capital_city_fact` | **PASS (False)** | 0.652 | Mentions 'paris'; 'versailles' not in static banned list |
| **Limitation** | Partial refund of 50% | `refund_policy_explanation` | **PASS (False)** | 0.366 | Mentions '30 days'; '50%' not in static banned list |
| **Limitation** | Spelled-out 'thirty days' | `refund_policy_explanation` | **FAIL (False)** | 0.053 | Invariant normalizer does not convert number words to digits |

## What's deliberately out of scope (see project report, Section 3.4)

- Retraining or fine-tuning the AI/ML model embedded in the module under test
- General-purpose testing of standalone ML models (a separate, well-studied problem)
- Production-grade, multi-tenant deployment of the oracle as a hosted service

## Project team

K Sriram (24BYB0151), Piyush Patani (24BYB0146) — BCSE301L/P Software Engineering, VIT Vellore

