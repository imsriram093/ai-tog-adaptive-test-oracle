# 🎓 Learn AI-TOG Step-by-Step — Complete Guide

> This guide teaches you the **entire project** from scratch. Follow it in order — each lesson builds on the previous one. Budget **2–3 hours** total.

---

## How This Guide Works

Each lesson follows the same pattern:
1. **💡 The Concept** — what this piece does and WHY it exists
2. **📄 The Code** — walk through the actual file, line by line
3. **🖥️ Try It** — commands to run so you SEE it working
4. **✅ Checkpoint** — questions to test yourself before moving on

---

# Lesson 1: The Problem This Project Solves

### 💡 The Concept

Traditional testing compares output to **one exact expected answer**:

```
assert chatbot_response == "The capital of France is Paris."
```

This works great for calculators (2+2 is always 4). But AI/LLM outputs are **non-deterministic** — the same question can produce:

- "The capital of France is Paris."
- "Paris is France's capital city."
- "France's capital is Paris, located in the Île-de-France region."

All three are **correct**, but exact-match testing would fail the last two. So developers either:
- ❌ Disable testing entirely (losing coverage)
- ❌ Use brittle substring checks (high false-positive/negative rates)

**AI-TOG's solution:** Instead of matching ONE answer, define a **region of acceptable answers** using:
- Semantic similarity (does it MEAN the same thing?)
- Structural rules (right length?)
- Domain invariants (mentions required facts? avoids banned terms?)

And make the pass/fail boundary **adaptive** — it learns from data.

### ✅ Checkpoint
> - Why does `assert response == "expected"` break for AI outputs?
> - What are the 3 signals AI-TOG uses instead of exact matching?

---

# Lesson 2: Stage 1 — Baseline (`baseline.py`)

> 📄 Open [baseline.py](ai_tog/baseline.py)

### 💡 The Concept

Stage 1 is the **ground truth**. Before the oracle can judge anything, it needs to know what "correct" looks like. This comes from **human QA experts** who curate known-good answers.

Think of it like a teacher preparing an answer key — not just one answer, but **multiple acceptable phrasings**.

### 📄 The Code (line by line)

**`BaselineExample` (lines 7–11):**
```python
@dataclass
class BaselineExample:
    text: str                        # The actual answer text
    approved_by: str = "unspecified" # Who approved this as correct
    notes: str = ""                  # Optional notes
```
This is just a container. A `dataclass` automatically gives you `__init__`, `__repr__`, etc. — it's Python's way of saying "this is a simple data holder."

**`TestCategory` (lines 14–41):**
```python
@dataclass
class TestCategory:
    category_id: str                   # e.g., "capital_city_fact"
    prompt: str                        # The question being tested
    baseline_examples: list[BaselineExample]  # 5 known-good answers
    rejected_examples: list[BaselineExample]  # Known-BAD answers (for reference)
    min_length: int | None = None      # Minimum word count
    max_length: int | None = None      # Maximum word count
    required_terms: list[str]          # Words that MUST appear (e.g., "paris")
    banned_terms: list[str]            # Words that must NOT appear (e.g., "berlin")
```

The `__post_init__` validation is important:
```python
def __post_init__(self):
    if len(self.baseline_examples) < 3:
        raise ValueError(...)  # Need at least 3 to compute a meaningful centroid
```
Why 3? Because with fewer than 3 points, you can't compute a reliable average position in embedding space.

**`load_category()` and `load_categories()` (lines 43–63):**
```python
def load_category(path):
    data = json.loads(Path(path).read_text())  # Read JSON file
    return TestCategory(...)                    # Convert to Python object

def load_categories(directory):
    categories = {}
    for path in sorted(directory.glob("*.json")):  # Find all .json files
        cat = load_category(path)
        categories[cat.category_id] = cat          # Dict: "capital_city_fact" → TestCategory
    return categories
```

### 🖥️ Try It — Open Python and run:
```python
python -c "from ai_tog.baseline import load_categories; cats = load_categories('data/categories'); print(list(cats.keys())); cat = cats['capital_city_fact']; print(f'Prompt: {cat.prompt}'); print(f'Baseline examples: {len(cat.baseline_examples)}'); print(f'Required terms: {cat.required_terms}'); print(f'Banned terms: {cat.banned_terms}')"
```

Now look at the actual JSON file:
> 📄 Open [capital_city.json](data/categories/capital_city.json)

See how the JSON maps directly to the `TestCategory` fields.

### ✅ Checkpoint
> - What does `TestCategory` represent?
> - Why is there a minimum of 3 baseline examples?
> - What's the difference between `required_terms` and `banned_terms`?
> - If you wanted to add a new test category (e.g., "weather forecast"), what would you do?

---

# Lesson 3: The Embedding Layer (`embedding/`)

> 📄 Open [base.py](ai_tog/embedding/base.py) and [tfidf_embedder.py](ai_tog/embedding/tfidf_embedder.py)

### 💡 The Concept

The core problem: how do you measure if two sentences **mean the same thing**?

Computers can't understand language directly. So we convert text into **numbers** (vectors). Sentences with similar meaning get vectors that point in similar directions.

**Analogy:** Imagine every sentence is an arrow in space. "Paris is the capital" and "France's capital is Paris" point in almost the same direction. "The weather is sunny" points somewhere completely different.

**Cosine similarity** measures the angle between two arrows:
- Same direction → similarity = 1.0 (identical meaning)
- Perpendicular → similarity = 0.0 (unrelated)
- Opposite → similarity = -1.0 (opposite meaning)

### 📄 The Code

**The Interface (`base.py`):**
```python
class Embedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) array of L2-normalized embeddings."""

    @property
    @abstractmethod
    def dim(self) -> int: ...    # Vector dimensions

    @property
    @abstractmethod
    def name(self) -> str: ...   # Backend name
```
This is an **abstract interface** — it defines WHAT an embedder must do, not HOW. Any embedder that implements `encode()`, `dim`, and `name` can be plugged in. This is the **Strategy Pattern**.

**TF-IDF Embedder (`tfidf_embedder.py`):**
```python
class TfidfEmbedder(Embedder):
    def __init__(self, corpus: list[str], max_features=512):
        self._vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        self._vectorizer.fit(corpus)  # Learn vocabulary from all baseline examples
```

TF-IDF = **Term Frequency × Inverse Document Frequency**
- **TF**: How often a word appears in THIS text (more = more important)
- **IDF**: How rare the word is across ALL texts (rarer = more important)
- Words like "the" appear everywhere → low IDF → low weight
- Words like "Paris" appear in specific contexts → high IDF → high weight

The `encode()` method converts text → sparse vector → dense numpy array → L2-normalized (unit length).

**Why L2-normalize?** When all vectors have length 1, the dot product `a @ b` equals cosine similarity. This simplifies the math throughout the pipeline.

**`_l2_normalize()` helper:**
```python
def _l2_normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)  # Length of each vector
    norms[norms == 0] = 1.0   # Avoid division by zero
    return vecs / norms        # Divide each vector by its length → unit length
```

### 🖥️ Try It:
```python
python -c "
from ai_tog.embedding.tfidf_embedder import TfidfEmbedder
import numpy as np
corpus = ['The capital of France is Paris', 'Paris is the capital city of France', 'The weather today is sunny']
emb = TfidfEmbedder(corpus)
vecs = emb.encode(corpus)
print(f'Vector shape: {vecs.shape}')
print(f'Vector norms (should be ~1.0): {np.linalg.norm(vecs, axis=1)}')
print(f'Similarity (same topic): {vecs[0] @ vecs[1]:.3f}')
print(f'Similarity (diff topic): {vecs[0] @ vecs[2]:.3f}')
"
```

You should see: same-topic similarity is **much higher** than different-topic.

### ✅ Checkpoint
> - What does "embedding" mean? (Converting text to numbers/vectors)
> - Why L2-normalize all vectors?
> - Why is TF-IDF called a "fallback"? (It's lexical, not truly semantic — it matches WORDS not MEANING)
> - What would `SentenceTransformerEmbedder` do differently? (Uses a neural network to capture actual meaning)

---

# Lesson 4: Stage 2 — Reference Profile (`profile.py`)

> 📄 Open [profile.py](ai_tog/profile.py)

### 💡 The Concept

Stage 1 gives us 5 known-good answers. Stage 2 turns them into a **statistical fingerprint** — a single point in embedding space that represents "what acceptable looks like."

**Analogy:** If you plot 5 correct answers as dots in space, the **centroid** (average position) is the center of the "acceptable zone." New answers close to the center are likely correct. Answers far away are likely wrong.

### 📄 The Code

**`ReferenceProfile` (lines 13–32):**
```python
@dataclass
class ReferenceProfile:
    category_id: str
    centroid: np.ndarray    # The "center point" of all known-good answers
    n_examples: int         # How many examples built this centroid
    min_length: int | None  # Carried from TestCategory
    max_length: int | None
    required_terms: list[str]
    banned_terms: list[str]
    created_at: str
    version: int = 1        # Increments as feedback updates the profile
    history: list[str]      # Audit trail of changes
```

**`ProfileBuilder.build()` (lines 46–61):**
```python
def build(self, category):
    texts = [ex.text for ex in category.baseline_examples]  # Extract 5 answer texts
    vecs = self.embedder.encode(texts)                       # Convert to 5 vectors
    centroid = _normalize(vecs.mean(axis=0))                 # Average → normalize
    return ReferenceProfile(centroid=centroid, ...)
```

Step by step:
1. Take 5 answer texts → encode into 5 vectors (each ~512 dimensions)
2. `vecs.mean(axis=0)` → average across all 5 vectors → one "center" vector
3. `_normalize()` → make it unit length again
4. Package with the structural rules from the category

**`update_incremental()` (lines 63–91) — THE FEEDBACK LOOP:**
```python
def update_incremental(self, profile, approved_text):
    new_vec = self.embedder.encode([approved_text])[0]
    n = profile.n_examples
    blended = (profile.centroid * n + new_vec) / (n + 1)  # Running mean
    profile.centroid = _normalize(blended)
    profile.n_examples = n + 1
    profile.version += 1
```

This is a **running mean** — instead of re-computing the average from scratch with all examples, it blends the new vector into the existing centroid. The formula:

```
new_centroid = (old_centroid × n + new_vector) / (n + 1)
```

Why? Because when a human says "this REVIEW answer is actually good," we want to expand the "acceptable zone" to include it — without re-encoding all previous examples.

**`leave_one_out_seed_scores()` (lines 94–115) — CLEVER TRICK:**

For each baseline example `i`:
1. Build centroid from every OTHER example (excluding `i`)
2. Score example `i` against that centroid
3. This gives a realistic score distribution

Why not just score each example against the full centroid? Because the full centroid **includes** that example — it would be like testing yourself with the answer key in front of you. The scores would be artificially high (optimistic), and the threshold would be set too tight, causing false failures on real candidates.

### 🖥️ Try It:
```python
python -c "
from ai_tog.baseline import load_categories
from ai_tog.embedding.tfidf_embedder import TfidfEmbedder
from ai_tog.profile import ProfileBuilder
import numpy as np
cats = load_categories('data/categories')
cat = cats['capital_city_fact']
corpus = [ex.text for ex in cat.baseline_examples]
emb = TfidfEmbedder(corpus)
builder = ProfileBuilder(emb)
profile = builder.build(cat)
print(f'Centroid shape: {profile.centroid.shape}')
print(f'Centroid norm: {np.linalg.norm(profile.centroid):.6f}')
print(f'n_examples: {profile.n_examples}, version: {profile.version}')
loo = builder.leave_one_out_seed_scores(cat)
print(f'LOO scores: {[f\"{s:.3f}\" for s in loo]}')
print(f'LOO mean: {sum(loo)/len(loo):.3f}')
"
```

### ✅ Checkpoint
> - What is a centroid? (Average vector of all known-good answers)
> - Why does `update_incremental` use a running mean instead of re-computing from scratch?
> - What problem does leave-one-out seeding solve? (Prevents optimistic threshold initialization)

---

# Lesson 5: Stage 3 — Oracle Engine (`oracle.py`)

> 📄 Open [oracle.py](ai_tog/oracle.py)

### 💡 The Concept

The oracle scores a candidate answer on **three independent signals**, not just similarity. A candidate can be semantically close but structurally wrong (too short) or violate a domain rule (mentions a banned term).

### 📄 The Code

**Constants (lines 12–13):**
```python
STRUCTURAL_VIOLATION_PENALTY = 0.15   # Deducted from score if wrong length
INVARIANT_VIOLATION_PENALTY = 0.25    # Deducted from score if missing/banned terms
```

**`SignalScores` (lines 16–36):**
```python
@dataclass
class SignalScores:
    semantic_similarity: float    # 0.0 to 1.0 — how similar the MEANING is
    structural_valid: bool        # True/False — is word count within range?
    invariants_satisfied: bool    # True/False — all rules met?
    violations: list[str]         # Human-readable reasons for failures

    @property
    def composite_score(self):
        score = self.semantic_similarity
        if not self.structural_valid:
            score -= 0.15            # Penalty
        if not self.invariants_satisfied:
            score -= 0.25            # Bigger penalty
        return max(0.0, score)       # Never go below 0
```

**`OracleEngine.score()` (lines 52–89):**

Three checks happen in sequence:

```python
# CHECK 1: Semantic Similarity
vec = self.embedder.encode([candidate_text])[0]   # Embed the candidate
vec = vec / norm                                    # Normalize
semantic_similarity = float(vec @ profile.centroid) # Dot product = cosine sim

# CHECK 2: Structural Validity (word count)
word_count = len(candidate_text.split())
if word_count < profile.min_length:  # Too short?
    structural_valid = False
    violations.append(f"too short: {word_count} words < minimum {profile.min_length}")

# CHECK 3: Domain Invariants (required/banned terms)
text_norm = candidate_text.lower().replace("-", " ")  # "30-day" → "30 day"
for term in profile.required_terms:
    if term not in text_norm:
        invariants_satisfied = False
        violations.append(f"missing required term: '{term}'")
for term in profile.banned_terms:
    if term in text_norm:
        invariants_satisfied = False
        violations.append(f"contains banned term: '{term}'")
```

**The hyphen normalization** (`replace("-", " ")`) is a detail worth noting: "30-day" and "30 day" should match. Without this, valid answers would fail over formatting.

### 🖥️ Try It:
```python
python -c "
from ai_tog.baseline import load_categories
from ai_tog.embedding.tfidf_embedder import TfidfEmbedder
from ai_tog.oracle import OracleEngine
from ai_tog.profile import ProfileBuilder
cats = load_categories('data/categories')
cat = cats['capital_city_fact']
corpus = [ex.text for ex in cat.baseline_examples]
emb = TfidfEmbedder(corpus)
profile = ProfileBuilder(emb).build(cat)
oracle = OracleEngine(emb)

# Test a CORRECT answer
s1 = oracle.score(profile, 'Paris is the capital of France.')
print(f'Correct answer:  sim={s1.semantic_similarity:.3f} struct={s1.structural_valid} inv={s1.invariants_satisfied} composite={s1.composite_score:.3f}')

# Test a WRONG answer (banned term)
s2 = oracle.score(profile, 'The capital of France is Berlin.')
print(f'Wrong (Berlin):  sim={s2.semantic_similarity:.3f} struct={s2.structural_valid} inv={s2.invariants_satisfied} composite={s2.composite_score:.3f} violations={s2.violations}')

# Test a TOO SHORT answer
s3 = oracle.score(profile, 'Paris.')
print(f'Too short:       sim={s3.semantic_similarity:.3f} struct={s3.structural_valid} inv={s3.invariants_satisfied} composite={s3.composite_score:.3f} violations={s3.violations}')
"
```

### ✅ Checkpoint
> - Why THREE signals instead of just similarity? (Similarity alone can't catch same-topic factual errors or structural issues)
> - What's the composite_score formula? (semantic_similarity - penalties, clamped to 0)
> - Why is the invariant penalty (0.25) larger than the structural penalty (0.15)?

---

# Lesson 6: Stage 4 — Adaptive Threshold (`threshold.py`)

> 📄 Open [threshold.py](ai_tog/threshold.py)

### 💡 The Concept

Now we have a composite score for each candidate. But how do we decide what score is "good enough"?

A **fixed threshold** (e.g., "pass if score > 0.7") doesn't work because:
- Different categories have different natural score ranges
- Scores shift when the AI model is updated
- A static cutoff can't adapt to accumulated evidence

AI-TOG uses **Statistical Process Control** (like manufacturing quality control):
- Keep a **rolling window** of scores from confirmed-good answers
- The pass/fail boundary = `mean - z × standard_deviation`
- As more evidence accumulates, the boundary tightens
- If scores suddenly shift, **drift detection** fires an alert

### 📄 The Code

**`ThresholdBand` (lines 17–23):**
```python
@dataclass
class ThresholdBand:
    pass_cut: float       # Score above this → PASS
    review_cut: float     # Score between review_cut and pass_cut → REVIEW
    center: float         # Rolling mean of confirmed-good scores
    spread: float         # Rolling std deviation
    is_cold_start: bool   # True if not enough data yet
```

The three zones:
```
Score:  |------FAIL------|------REVIEW------|------PASS------|
        0           review_cut          pass_cut            1.0
```

**Cold Start (lines 13–14):**
```python
COLD_START_PASS_CUT = 0.75    # Conservative defaults
COLD_START_REVIEW_CUT = 0.55  # Used until enough real scores arrive
```

**`current_band()` (lines 81–101):**
```python
def current_band(self, category_id):
    window = self._windows.get(category_id, [])
    if len(window) < self.min_samples:       # Not enough data?
        return ThresholdBand(is_cold_start=True, ...)  # Use conservative defaults

    arr = np.array(window)
    center = arr.mean()                       # Average of confirmed-good scores
    spread = arr.std(ddof=1)                  # Sample standard deviation
    spread = max(spread, self.spread_floor)   # Never let spread be zero
    return ThresholdBand(
        pass_cut = center - self.pass_z * spread,    # e.g., mean - 1.0σ
        review_cut = center - self.fail_z * spread,  # e.g., mean - 3.0σ
    )
```

**Why `pass_z=1.0` and `fail_z=3.0`?** These come from the "68-95-99.7 rule":
- ~68% of scores fall within 1σ of the mean → above pass_cut → PASS
- ~99.7% fall within 3σ → below review_cut → almost certainly wrong → FAIL
- Between 1σ and 3σ → uncertain → REVIEW (ask a human)

**Drift Detection (lines 103–134):**
```python
def detect_drift(self, category_id, recent_n=10, z=2.0):
    # Split window into historical and recent
    historical, recent = arr[:-recent_n], arr[-recent_n:]
    hist_mean = historical.mean()
    recent_mean = recent.mean()
    # If recent mean moved > 2σ from historical mean → drift!
    if abs(recent_mean - hist_mean) > z * hist_std:
        return f"drift detected: recent {recent_mean:.3f} vs historical {hist_mean:.3f}"
```

This catches when the AI model changes behavior (e.g., after an update). The threshold adapts, but it also **alerts a human** so they can decide if the shift is intentional.

### 🖥️ Try It:
```python
python -c "
from ai_tog.threshold import AdaptiveThresholdTracker

tracker = AdaptiveThresholdTracker(min_samples=5)

# Before any data → cold start
band = tracker.current_band('test')
print(f'Cold start: pass_cut={band.pass_cut} review_cut={band.review_cut} cold={band.is_cold_start}')

# Add confirmed-good scores
for s in [0.80, 0.82, 0.79, 0.81, 0.80, 0.83]:
    tracker.record_confirmed_good('test', s)

band = tracker.current_band('test')
print(f'Adaptive:   pass_cut={band.pass_cut:.3f} review_cut={band.review_cut:.3f} center={band.center:.3f} spread={band.spread:.3f} cold={band.is_cold_start}')
"
```

### ✅ Checkpoint
> - What's the difference between cold-start and adaptive mode?
> - What does `spread_floor` prevent? (If all scores are identical, spread=0 would make pass_cut=review_cut=center, so everything is either PASS or FAIL with no REVIEW zone)
> - Why detect drift instead of just silently adapting?

---

# Lesson 7: Stage 5 — Verdict + Feedback (`verdict.py`)

> 📄 Open [verdict.py](ai_tog/verdict.py)

### 💡 The Concept

Stage 5 is where everything comes together. It takes:
- The multi-signal scores from Stage 3
- The threshold band from Stage 4

And produces a **final verdict**: PASS, REVIEW, or FAIL.

But the most important part is `resolve_review()` — **the closed feedback loop** that makes the whole system adaptive.

### 📄 The Code

**`Verdict` enum (lines 11–14):**
```python
class Verdict(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"
```

**`VerdictAggregator.decide()` (lines 52–93):**
```python
def decide(self, category_id, candidate_text, signals):
    band = self.tracker.current_band(category_id)
    score = signals.composite_score

    # Rule 1: Hard invariant violation → always FAIL (no exceptions)
    if not signals.invariants_satisfied:
        verdict = Verdict.FAIL

    # Rule 2: Score above pass_cut → PASS
    elif score >= band.pass_cut:
        verdict = Verdict.PASS

    # Rule 3: Score between review_cut and pass_cut → REVIEW
    elif score >= band.review_cut:
        verdict = Verdict.REVIEW

    # Rule 4: Score below review_cut → FAIL
    else:
        verdict = Verdict.FAIL
```

**Key design decision:** Invariant violations **always fail**, even if the semantic similarity is 1.0. If the answer contains a banned term like "Berlin" for the capital-of-France question, no amount of "close enough" should excuse it.

**`resolve_review()` — THE CLOSED LOOP (lines 95–128):**
```python
def resolve_review(self, result, human_label, profile_builder, profile):
    if human_label == "good":
        # → Update Stage 2: expand the "acceptable zone" centroid
        profile_builder.update_incremental(profile, result.candidate_text)
        # → Update Stage 4: add this score to the threshold window
        self.tracker.record_confirmed_good(result.category_id, result.signals.composite_score)

    # If "bad": logged only — NEVER added to profile or threshold
    # (a confirmed-bad answer must never drag the acceptable zone toward itself)
```

This is the **entire learning mechanism**:
1. Oracle evaluates a candidate → REVIEW (uncertain)
2. Human says "this is actually good"
3. The candidate's embedding is folded into the centroid (Stage 2 learns)
4. The candidate's score is added to the threshold window (Stage 4 adjusts)
5. Next time a similar answer appears, it's more likely to PASS directly

### ✅ Checkpoint
> - Why do invariant violations always → FAIL regardless of score?
> - What happens when a human labels a REVIEW as "good"? (Two things update: centroid AND threshold window)
> - What happens when a human labels a REVIEW as "bad"? (Nothing updates — only logged. This is deliberate.)

---

# Lesson 8: The Pipeline + Storage (`pipeline.py` + `storage.py`)

> 📄 Open [pipeline.py](ai_tog/pipeline.py) and [storage.py](ai_tog/storage.py)

### 💡 The Concept

**Pipeline:** Wires all 5 stages together into a single, easy-to-use object. You call `pipeline.evaluate(category, text)` and get back a verdict. That's it.

**Storage:** Makes the system's learned state **survive restarts**. Without it, every time you restart the program, the oracle "forgets" everything it learned. With SQLite storage, profiles, threshold scores, evaluations, and review resolutions all persist to disk.

### 📄 The Pipeline Code

**`build_pipeline()` (lines 36–132) — The Factory:**

```python
def build_pipeline(data_dir, embedder_backend="tfidf", storage=None):
    # 1. Load all categories from JSON
    categories = load_categories(data_dir)

    # 2. Create embedder (TF-IDF or Sentence Transformer)
    if embedder_backend == "tfidf":
        corpus = [ex.text for cat in categories.values() for ex in cat.baseline_examples]
        embedder = TfidfEmbedder(corpus)

    # 3. Create all stage components
    profile_builder = ProfileBuilder(embedder, storage=storage)
    oracle = OracleEngine(embedder)
    threshold_tracker = AdaptiveThresholdTracker(storage=storage)
    verdict_aggregator = VerdictAggregator(threshold_tracker, storage=storage)

    # 4. For each category: build profile + seed threshold
    for category_id, category in categories.items():
        if storage and storage.has_profile(category_id):
            # RESTORE from database (previous run learned something)
            ...
        else:
            # BOOTSTRAP from JSON (first run)
            profile = profile_builder.build(category)
            seed_scores = profile_builder.leave_one_out_seed_scores(category)
            threshold_tracker.seed(category_id, seed_scores)

    return Pipeline(categories, profiles, embedder, ...)
```

**`Pipeline.evaluate()` — The One-Liner Users Call:**
```python
def evaluate(self, category_id, candidate_text):
    profile = self.profiles[category_id]                         # Stage 2
    signals = self.oracle.score(profile, candidate_text)         # Stage 3
    return self.verdict_aggregator.decide(category_id, candidate_text, signals)  # Stage 4+5
```

### 📄 The Storage Code

**5 SQLite Tables:**

| Table | What it stores |
|-------|---------------|
| `reference_profiles` | Centroid blob, n_examples, version per category |
| `profile_history` | Audit trail of every profile update |
| `threshold_scores` | Every confirmed-good score (for rebuilding the rolling window) |
| `evaluations` | Full audit log of every verdict ever given |
| `review_resolutions` | Human labels ("good"/"bad") for REVIEW verdicts |

**`InMemoryStorage`** is just `SQLiteStorage(":memory:")` — same SQL, same schema, but in RAM. Used for tests so they're fast and don't create files.

### 🖥️ Try It — The Full Pipeline:
```python
python -c "
from ai_tog.pipeline import build_pipeline
from ai_tog.verdict import Verdict

pipeline = build_pipeline('data/categories')

# Test various answers
tests = [
    ('capital_city_fact', 'Paris is the capital of France.'),
    ('capital_city_fact', 'The capital of France is Berlin.'),
    ('refund_policy_explanation', 'You can return items within 30 days for a full refund with your receipt.'),
    ('out_of_scope_decline', 'Sorry, I can only help with support topics.'),
]

for cat_id, text in tests:
    r = pipeline.evaluate(cat_id, text)
    print(f'{r.verdict.value:6}  score={r.signals.composite_score:.3f}  {text[:60]}...')
"
```

### ✅ Checkpoint
> - What does `build_pipeline()` do? (Wires all stages, loads data, seeds thresholds)
> - Why does the pipeline check `storage.has_profile()` before building? (To restore learned state from a previous run instead of starting fresh)
> - What's the difference between `SQLiteStorage` and `InMemoryStorage`? (Disk vs RAM — same code, same SQL)

---

# Lesson 9: The Tests — How to Verify Everything Works

> 📄 Open any test file in [tests/](tests)

### 💡 The Concept

The test suite is organized by the **testing pyramid**:

```
         /\
        /  \     Acceptance (benchmark.py — 24 held-out cases)
       /    \
      /------\   System (test_pipeline_end_to_end.py — full pipeline)
     /        \
    /----------\ Integration (test_storage.py roundtrip, pipeline+storage)
   /            \
  /--------------\ Unit (test_embedding, test_profile, test_oracle, test_threshold)
```

### 🖥️ Run Everything:
```bash
# Run all 30 tests
python -m pytest tests/ -v

# Run just one test file
python -m pytest tests/test_oracle.py -v

# Run one specific test
python -m pytest tests/test_threshold.py::test_drift_detection_flags_a_shifted_recent_window -v

# Run the benchmark
python benchmark.py

# Run the demos
python demo.py
python demo_persistence.py
```

### ✅ Final Checkpoint
> Can you trace the **complete flow** of a single evaluation?
> 1. JSON file → `load_categories()` → `TestCategory` (Stage 1)
> 2. `ProfileBuilder.build()` → `ReferenceProfile` with centroid (Stage 2)
> 3. `OracleEngine.score()` → `SignalScores` with 3 signals (Stage 3)
> 4. `AdaptiveThresholdTracker.current_band()` → `ThresholdBand` (Stage 4)
> 5. `VerdictAggregator.decide()` → `EvaluationResult` with PASS/REVIEW/FAIL (Stage 5)
> 6. If REVIEW → `resolve_review("good")` → updates centroid + threshold (feedback loop)

---

# 🗺️ Quick Reference Map

| File | Stage | One-Line Summary |
|------|-------|-----------------|
| `baseline.py` | 1 | Load known-good answers from JSON |
| `embedding/base.py` | — | Abstract interface for text→vector |
| `embedding/tfidf_embedder.py` | — | Bag-of-words vector backend |
| `profile.py` | 2 | Build centroid from baseline, incremental updates |
| `oracle.py` | 3 | Score candidate on 3 signals: semantic + structural + invariant |
| `threshold.py` | 4 | Adaptive pass/review/fail boundaries from rolling window |
| `verdict.py` | 5 | Final verdict + feedback loop back to stages 2 & 4 |
| `pipeline.py` | All | Wire everything together, handle persistence |
| `storage.py` | — | SQLite persistence for all learned state |
| `benchmark.py` | — | Compare AI-TOG vs naive exact-match on 24 cases |
| `demo.py` | — | Show threshold adapting + drift detection |
| `demo_persistence.py` | — | Show state surviving across process restarts |
