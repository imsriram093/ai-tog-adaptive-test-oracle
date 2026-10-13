/**
 * app.js: Observatory Frontend Controller for AI-TOG.
 * Vanilla JavaScript (Zero external libraries, zero CDN dependencies).
 * Enforces textContent rendering for 100% XSS safety.
 */

(function () {
  "use strict";

  // State Management
  const state = {
    theme: localStorage.getItem("aitog-theme") || "dark",
    categories: [],
    selectedCategoryId: null,
    activeTab: "playground",
    lastEvaluationResult: null,
    benchmarkData: null,
    auditData: null,
    isReducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  };

  // Presets Mapping (Real ground truth from benchmark.py & pipeline verification)
  const PRESETS = {
    "valid-1": {
      category_id: "refund_policy_explanation",
      text: "So long as the purchase was made within the last 30 days and you kept your receipt, a full refund will be given.",
      note: "Expected: PASS (Paraphrase with reordered clauses)"
    },
    "valid-2": {
      category_id: "refund_policy_explanation",
      text: "If you come back in the 30-day window and show us your receipt, the refund is yours in full.",
      note: "Expected: PASS (Conversational register)"
    },
    "valid-3": {
      category_id: "capital_city_fact",
      text: "Paris, France's capital, is a major European city.",
      note: "Expected: PASS (Paraphrase with extra geographical detail)"
    },
    "caught-1": {
      category_id: "refund_policy_explanation",
      text: "You can return items within 90 days for store credit only, no cash refunds.",
      note: "Expected: FAIL (Hallucinated window & store credit policy violation)"
    },
    "caught-2": {
      category_id: "out_of_scope_decline",
      text: "Sure! Here's an essay introduction paragraph about climate change: Climate change is one of the most pressing issues of our time.",
      note: "Expected: FAIL (Attempted out-of-scope essay task)"
    },
    "caught-3": {
      category_id: "capital_city_fact",
      text: "The capital of France is Marseille.",
      note: "Expected: FAIL (Wrong city, violates required term 'paris')"
    },
    "review-1": {
      category_id: "refund_policy_explanation",
      text: "Full refunds are available for 30-day returns with receipt.",
      note: "Expected: REVIEW (Terse wording triggers structural word-count penalty)"
    },
    "review-2": {
      category_id: "refund_policy_explanation",
      text: "We offer a 30-day money-back guarantee, but only for items bought with a gift card.",
      note: "Expected: REVIEW (Borderline score lands in adaptive review zone)"
    },
    "review-3": {
      category_id: "capital_city_fact",
      text: "Paris is a city in Europe.",
      note: "Expected: REVIEW (True fact with missing specific prompt focus)"
    },
    "limit-1": {
      category_id: "capital_city_fact",
      text: "The capital of France used to be Paris but is now Versailles.",
      note: "Known Limitation: False Accept (PASS, required term 'paris' present, 'versailles' not in banned terms)"
    },
    "limit-2": {
      category_id: "refund_policy_explanation",
      text: "Returns within 30 days get a partial refund of 50%.",
      note: "Known Limitation: False Accept (PASS, '50%' not in static invariant list)"
    },
    "limit-3": {
      category_id: "refund_policy_explanation",
      text: "Within thirty days of purchase, any product may be returned with proof of purchase for a complete refund.",
      note: "Known Limitation: False Reject (FAIL, spelled-out 'thirty' != literal '30 day')"
    }
  };

  // Pipeline 5-Stage Explanations
  const STAGE_EXPLANATIONS = {
    "1": {
      title: "Stage 1: Curated Baseline Examples (Ground Truth)",
      tag: "STAGE 1 SPECIFICATION",
      desc: "Curated, QA-approved known-good answers per category, defining prompt boundaries and baseline examples (at least 3-5 required). Provides the statistical foundation for Stage 2.",
    },
    "2": {
      title: "Stage 2: Reference Profile Builder",
      tag: "STAGE 2 REFERENCE PROFILE",
      desc: "Computes the L2-normalized embedding centroid vector across baseline texts and merges static domain invariants (required terms, banned terms, min/max length). Grown iteratively in Stage 5.",
    },
    "3": {
      title: "Stage 3: Multi-Signal Oracle Scoring Engine",
      tag: "STAGE 3 SIGNAL ENGINE",
      desc: "Evaluates candidate outputs across 3 orthogonal signals: (1) Semantic Cosine Similarity against Centroid, (2) Structural Length Bounds, and (3) Hard Domain Invariants. Penalizes composite score on violations.",
    },
    "4": {
      title: "Stage 4: Adaptive Threshold (Control-Chart Bounds)",
      tag: "STAGE 4 STATISTICAL TRACKER",
      desc: "Maintains a rolling window of confirmed-good scores. Derives dynamic boundaries: Pass Cut = μ - 1.0σ and Review Cut = μ - 3.0σ. Automatically shifts boundaries as model outputs evolve and detects statistical drift.",
    },
    "5": {
      title: "Stage 5: Verdict Aggregation & Closed-Loop Feedback",
      tag: "STAGE 5 CLOSED FEEDBACK LOOP",
      desc: "Resolves candidate into PASS, REVIEW, or FAIL. When a human resolves a REVIEW case as 'good', that candidate is folded into Stage 2 profile (n=n+1, version++) and its score is added to the Stage 4 rolling window.",
    }
  };

  // DOM Elements
  const dom = {
    themeToggle: document.getElementById("themeToggle"),
    categorySelect: document.getElementById("categorySelect"),
    catPromptDisplay: document.getElementById("catPromptDisplay"),
    catLengthDisplay: document.getElementById("catLengthDisplay"),
    catRequiredDisplay: document.getElementById("catRequiredDisplay"),
    catBannedDisplay: document.getElementById("catBannedDisplay"),
    catProfileDisplay: document.getElementById("catProfileDisplay"),
    candidateInput: document.getElementById("candidateInput"),
    evaluateBtn: document.getElementById("evaluateBtn"),
    clearInputBtn: document.getElementById("clearInputBtn"),
    verdictCard3D: document.getElementById("verdictCard3D"),
    verdictBadge: document.getElementById("verdictBadge"),
    verdictEvalId: document.getElementById("verdictEvalId"),
    verdictExplanation: document.getElementById("verdictExplanation"),
    humanReviewWidget: document.getElementById("humanReviewWidget"),
    resolveGoodBtn: document.getElementById("resolveGoodBtn"),
    resolveBadBtn: document.getElementById("resolveBadBtn"),
    gaugeCenterVal: document.getElementById("gaugeCenterVal"),
    gaugeSpreadVal: document.getElementById("gaugeSpreadVal"),
    gaugePassCutVal: document.getElementById("gaugePassCutVal"),
    gaugeReviewCutVal: document.getElementById("gaugeReviewCutVal"),
    coldStartChip: document.getElementById("coldStartChip"),
    scoreMarker: document.getElementById("scoreMarker"),
    markerPill: document.getElementById("markerPill"),
    bladeReview: document.getElementById("bladeReview"),
    bladePass: document.getElementById("bladePass"),
    sigSemanticScore: document.getElementById("sigSemanticScore"),
    sigSemanticBar: document.getElementById("sigSemanticBar"),
    sigStructuralBadge: document.getElementById("sigStructuralBadge"),
    sigStructuralDetail: document.getElementById("sigStructuralDetail"),
    sigStructuralPenalty: document.getElementById("sigStructuralPenalty"),
    sigInvariantBadge: document.getElementById("sigInvariantBadge"),
    sigInvariantDetail: document.getElementById("sigInvariantDetail"),
    sigInvariantPenalty: document.getElementById("sigInvariantPenalty"),
    violationsContainer: document.getElementById("violationsContainer"),
    violationsList: document.getElementById("violationsList"),
    sigCompositeScore: document.getElementById("sigCompositeScore"),
    sigCompositeFormula: document.getElementById("sigCompositeFormula"),
    systemNotification: document.getElementById("systemNotification"),
    notificationMessage: document.getElementById("notificationMessage"),
    closeNotifBtn: document.getElementById("closeNotifBtn"),
    ambientCanvas: document.getElementById("ambientCanvas"),
    stageInspectorTag: document.getElementById("stageInspectorTag"),
    stageInspectorTitle: document.getElementById("stageInspectorTitle"),
    stageInspectorDesc: document.getElementById("stageInspectorDesc"),
    stageJsonCatLabel: document.getElementById("stageJsonCatLabel"),
    stageJsonCode: document.getElementById("stageJsonCode"),
    copyStageJsonBtn: document.getElementById("copyStageJsonBtn"),
    evaluationsTableBody: document.getElementById("evaluationsTableBody"),
    resolutionsTableBody: document.getElementById("resolutionsTableBody"),
    historyTableBody: document.getElementById("historyTableBody"),
    evalCount: document.getElementById("evalCount"),
    resCount: document.getElementById("resCount"),
    refreshAuditBtn: document.getElementById("refreshAuditBtn"),
    resetDemoBtn: document.getElementById("resetDemoBtn"),
    resetModal: document.getElementById("resetModal"),
    cancelResetBtn: document.getElementById("cancelResetBtn"),
    confirmResetBtn: document.getElementById("confirmResetBtn"),
    benchNaiveAccuracy: document.getElementById("benchNaiveAccuracy"),
    benchNaiveCorrect: document.getElementById("benchNaiveCorrect"),
    benchNaiveFR: document.getElementById("benchNaiveFR"),
    benchNaiveFA: document.getElementById("benchNaiveFA"),
    benchAitogAccuracy: document.getElementById("benchAitogAccuracy"),
    benchAitogCorrect: document.getElementById("benchAitogCorrect"),
    benchAitogFR: document.getElementById("benchAitogFR"),
    benchAitogFA: document.getElementById("benchAitogFA"),
    benchCategoryTableBody: document.getElementById("benchCategoryTableBody"),
    limitationsGrid: document.getElementById("limitationsGrid"),
  };

  // ========================================================================
  // INITIALIZATION
  // ========================================================================

  async function init() {
    initTheme();
    initCanvasParticles();
    initTiltCards();
    initNavigation();
    initPresets();
    initEventListeners();

    await loadCategories();
    // Pre-fetch benchmark in background
    fetchBenchmark();
  }

  // ========================================================================
  // THEME & AMBIENT CANVAS
  // ========================================================================

  function initTheme() {
    document.documentElement.setAttribute("data-theme", state.theme);
    dom.themeToggle.addEventListener("click", () => {
      state.theme = state.theme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", state.theme);
      localStorage.setItem("aitog-theme", state.theme);
    });
  }

  function initCanvasParticles() {
    if (state.isReducedMotion || !dom.ambientCanvas) return;
    const canvas = dom.ambientCanvas;
    const ctx = canvas.getContext("2d");
    let width, height;
    let particles = [];
    let mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
    let animationFrameId = null;

    function resize() {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    }

    function createParticles() {
      particles = [];
      const count = Math.min(60, Math.floor(width / 30));
      for (let i = 0; i < count; i++) {
        particles.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.25,
          vy: (Math.random() - 0.5) * 0.25,
          radius: Math.random() * 1.5 + 0.5,
          alpha: Math.random() * 0.4 + 0.15,
        });
      }
    }

    function animate() {
      if (document.hidden) {
        animationFrameId = requestAnimationFrame(animate);
        return;
      }

      ctx.clearRect(0, 0, width, height);

      // Smooth mouse parallax
      mouse.x += (mouse.targetX - mouse.x) * 0.05;
      mouse.y += (mouse.targetY - mouse.y) * 0.05;

      const isLight = document.documentElement.getAttribute("data-theme") === "light";
      const dotColor = isLight ? "rgba(26, 31, 29, " : "rgba(95, 211, 165, ";

      for (let p of particles) {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0) p.x = width;
        if (p.x > width) p.x = 0;
        if (p.y < 0) p.y = height;
        if (p.y > height) p.y = 0;

        const px = p.x + mouse.x * (p.radius * 2);
        const py = p.y + mouse.y * (p.radius * 2);

        ctx.beginPath();
        ctx.arc(px, py, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = dotColor + p.alpha + ")";
        ctx.fill();
      }

      animationFrameId = requestAnimationFrame(animate);
    }

    window.addEventListener("resize", () => {
      resize();
      createParticles();
    });

    window.addEventListener("mousemove", (e) => {
      mouse.targetX = (e.clientX / width - 0.5) * 15;
      mouse.targetY = (e.clientY / height - 0.5) * 15;
    });

    resize();
    createParticles();
    animate();
  }

  // Pointer-driven 3D Card Tilt (~6deg max)
  function initTiltCards() {
    if (state.isReducedMotion) return;
    const cards = document.querySelectorAll("[data-tilt]");

    cards.forEach((card) => {
      card.addEventListener("mousemove", (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        const rotateX = ((y - centerY) / centerY) * -4.5;
        const rotateY = ((x - centerX) / centerX) * 4.5;

        card.style.setProperty("--pointer-x", `${x}px`);
        card.style.setProperty("--pointer-y", `${y}px`);
        card.style.transform = `perspective(1200px) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg) translateY(-2px)`;
      });

      card.addEventListener("mouseleave", () => {
        card.style.transform = `perspective(1200px) rotateX(0deg) rotateY(0deg) translateY(0px)`;
      });
    });
  }

  // ========================================================================
  // NAVIGATION & TABS
  // ========================================================================

  function initNavigation() {
    const tabButtons = document.querySelectorAll(".tab-btn");
    tabButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");
        switchTab(targetTab);
      });
    });
  }

  function switchTab(tabId) {
    state.activeTab = tabId;

    document.querySelectorAll(".tab-btn").forEach((b) => {
      const active = b.getAttribute("data-tab") === tabId;
      b.classList.toggle("active", active);
      b.setAttribute("aria-selected", active ? "true" : "false");
    });

    document.querySelectorAll(".view-panel").forEach((panel) => {
      const active = panel.id === `view-${tabId}`;
      panel.classList.toggle("active", active);
    });

    // Lazy load tab data
    if (tabId === "pipeline") {
      updateStageInspector(5);
    } else if (tabId === "audit") {
      loadAuditLogs();
    } else if (tabId === "benchmark") {
      renderBenchmarkTab();
    }
  }

  // ========================================================================
  // PRESET CLICKS
  // ========================================================================

  function initPresets() {
    const chips = document.querySelectorAll(".preset-chip");
    chips.forEach((chip) => {
      chip.addEventListener("click", () => {
        const key = chip.getAttribute("data-preset");
        const preset = PRESETS[key];
        if (!preset) return;

        // Switch category if needed
        if (dom.categorySelect.value !== preset.category_id) {
          dom.categorySelect.value = preset.category_id;
          handleCategoryChange();
        }

        dom.candidateInput.value = preset.text;
        showNotification(preset.note, "info");
      });
    });
  }

  // ========================================================================
  // API & CATEGORIES
  // ========================================================================

  async function loadCategories() {
    try {
      const res = await fetch("/api/categories");
      if (!res.ok) throw new Error("Failed to load categories");
      const data = await res.json();
      state.categories = data.categories;

      dom.categorySelect.replaceChildren();
      state.categories.forEach((cat) => {
        const opt = document.createElement("option");
        opt.value = cat.category_id;
        opt.textContent = `${cat.category_id} (v${cat.profile.version})`;
        dom.categorySelect.appendChild(opt);
      });

      if (state.categories.length > 0) {
        state.selectedCategoryId = state.categories[0].category_id;
        renderCategoryMeta(state.categories[0]);
        updateDepthGauge(state.categories[0].band, null);
      }
    } catch (err) {
      showNotification(`API Error: ${err.message}`, "error");
    }
  }

  function handleCategoryChange() {
    state.selectedCategoryId = dom.categorySelect.value;
    const cat = state.categories.find((c) => c.category_id === state.selectedCategoryId);
    if (cat) {
      renderCategoryMeta(cat);
      updateDepthGauge(cat.band, null);
      resetVerdictCard();
      if (state.activeTab === "pipeline") {
        updateStageInspector(5);
      }
    }
  }

  function renderCategoryMeta(cat) {
    dom.catPromptDisplay.textContent = cat.prompt;
    dom.catLengthDisplay.textContent = `min: ${cat.min_length ?? "None"} | max: ${cat.max_length ?? "None"} words`;

    // Required terms tags
    dom.catRequiredDisplay.replaceChildren();
    if (cat.required_terms && cat.required_terms.length > 0) {
      cat.required_terms.forEach((t) => {
        const span = document.createElement("span");
        span.className = "term-tag";
        span.textContent = t;
        dom.catRequiredDisplay.appendChild(span);
      });
    } else {
      dom.catRequiredDisplay.textContent = "None";
    }

    // Banned terms tags
    dom.catBannedDisplay.replaceChildren();
    if (cat.banned_terms && cat.banned_terms.length > 0) {
      cat.banned_terms.forEach((t) => {
        const span = document.createElement("span");
        span.className = "term-tag";
        span.textContent = t;
        dom.catBannedDisplay.appendChild(span);
      });
    } else {
      dom.catBannedDisplay.textContent = "None";
    }

    dom.catProfileDisplay.textContent = `v${cat.profile.version} (n=${cat.profile.n_examples} baseline examples)`;
  }

  // ========================================================================
  // EVALUATION & SCORECARD
  // ========================================================================

  async function evaluateCandidate() {
    const text = dom.candidateInput.value.trim();
    if (!text) {
      showNotification("Please enter or select candidate text to evaluate.", "error");
      dom.candidateInput.focus();
      return;
    }

    const btn = dom.evaluateBtn;
    const spinner = btn.querySelector(".btn-spinner");
    const btnText = btn.querySelector(".btn-text");

    btn.disabled = true;
    spinner.classList.remove("hidden");
    btnText.textContent = "Evaluating...";

    try {
      const response = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          category_id: state.selectedCategoryId,
          candidate_text: text,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || "Evaluation failed");
      }

      const result = await response.json();
      state.lastEvaluationResult = result;

      // Render Verdict, Depth Gauge, and Scorecard
      renderEvaluationResult(result);
    } catch (err) {
      showNotification(`Evaluation Error: ${err.message}`, "error");
    } finally {
      btn.disabled = false;
      spinner.classList.add("hidden");
      btnText.textContent = "Evaluate Output";
    }
  }

  function renderEvaluationResult(result) {
    const backFace = dom.verdictCard3D.querySelector(".verdict-back");

    // Remove prior verdict classes
    backFace.classList.remove("verdict-is-pass", "verdict-is-review", "verdict-is-fail");
    dom.verdictCard3D.classList.remove("flipped");

    const vLower = result.verdict.toLowerCase();
    backFace.classList.add(`verdict-is-${vLower}`);

    dom.verdictBadge.textContent = result.verdict;
    dom.verdictEvalId.textContent = `EVAL #${result.evaluation_id ?? "—"}`;
    dom.verdictExplanation.textContent = result.explanation;

    // Show / Hide Stage 5 Human Review widget
    if (result.verdict === "REVIEW") {
      dom.humanReviewWidget.classList.remove("hidden");
    } else {
      dom.humanReviewWidget.classList.add("hidden");
    }

    // Flip card to reveal back side
    setTimeout(() => {
      dom.verdictCard3D.classList.add("flipped");
    }, 50);

    // Update Depth Gauge
    updateDepthGauge(result.band, result.signals.composite_score);

    // Update Stage 3 Scorecard
    renderScorecard(result);
  }

  function renderScorecard(result) {
    const sig = result.signals;
    const pen = result.penalties;

    // 1. Semantic Similarity
    dom.sigSemanticScore.textContent = sig.semantic_similarity.toFixed(4);
    const semPercent = Math.max(0, Math.min(100, sig.semantic_similarity * 100));
    dom.sigSemanticBar.style.width = `${semPercent.toFixed(1)}%`;
    dom.sigSemanticBar.style.backgroundColor = sig.semantic_similarity >= result.band.pass_cut ? "var(--pass)" : "var(--review)";

    // 2. Structural Validity
    dom.sigStructuralBadge.textContent = sig.structural_valid ? "VALID" : "INVALID";
    dom.sigStructuralBadge.className = `status-badge ${sig.structural_valid ? "pass" : "fail"}`;
    const wordCount = result.candidate_text.trim().split(/\s+/).length;
    dom.sigStructuralDetail.textContent = `Length: ${wordCount} words (Bounds checked)`;
    dom.sigStructuralPenalty.textContent = pen.applied_structural ? "Penalty applied: -0.15" : "No penalty (-0.15)";
    dom.sigStructuralPenalty.classList.toggle("applied", pen.applied_structural);

    // 3. Domain Invariants
    dom.sigInvariantBadge.textContent = sig.invariants_satisfied ? "SATISFIED" : "VIOLATED";
    dom.sigInvariantBadge.className = `status-badge ${sig.invariants_satisfied ? "pass" : "fail"}`;
    dom.sigInvariantDetail.textContent = sig.invariants_satisfied ? "All required present, banned absent" : "Invariant rule violation detected";
    dom.sigInvariantPenalty.textContent = pen.applied_invariant ? "Penalty applied: -0.25 (or hard fail)" : "No penalty (-0.25)";
    dom.sigInvariantPenalty.classList.toggle("applied", pen.applied_invariant);

    // Violations list
    dom.violationsList.replaceChildren();
    if (sig.violations && sig.violations.length > 0) {
      dom.violationsContainer.classList.remove("hidden");
      sig.violations.forEach((v) => {
        const li = document.createElement("li");
        li.textContent = v;
        dom.violationsList.appendChild(li);
      });
    } else {
      dom.violationsContainer.classList.add("hidden");
    }

    // Composite Score
    dom.sigCompositeScore.textContent = sig.composite_score.toFixed(4);
    let formula = `semantic (${sig.semantic_similarity.toFixed(3)})`;
    if (pen.applied_structural) formula += ` - 0.15 struct`;
    if (pen.applied_invariant) formula += ` - 0.25 invar`;
    dom.sigCompositeFormula.textContent = formula;
  }

  // ========================================================================
  // STAGE 4 DEPTH GAUGE
  // ========================================================================

  function updateDepthGauge(band, score) {
    dom.gaugeCenterVal.textContent = band.center.toFixed(4);
    dom.gaugeSpreadVal.textContent = band.spread.toFixed(4);
    dom.gaugePassCutVal.textContent = band.pass_cut.toFixed(4);
    dom.gaugeReviewCutVal.textContent = band.review_cut.toFixed(4);

    dom.coldStartChip.textContent = band.is_cold_start ? "Cold Start Fallback" : "Adaptive Window Band";
    dom.coldStartChip.classList.toggle("cold-start", band.is_cold_start);

    // Normalized positioning for blades (scale 0.0 to 1.0)
    // Clamp review cut between 10% and 50%
    const rCutNorm = Math.max(10, Math.min(50, band.review_cut * 100));
    // Clamp pass cut between 45% and 85%
    const pCutNorm = Math.max(rCutNorm + 15, Math.min(85, band.pass_cut * 100));

    dom.bladeReview.style.left = `${rCutNorm.toFixed(1)}%`;
    dom.bladePass.style.left = `${pCutNorm.toFixed(1)}%`;

    // Position score marker
    if (score !== null && score !== undefined) {
      const scorePercent = Math.max(2, Math.min(98, score * 100));
      dom.scoreMarker.style.left = `${scorePercent.toFixed(1)}%`;
      dom.markerPill.textContent = `Score: ${score.toFixed(3)}`;
      dom.scoreMarker.style.display = "flex";
    } else {
      dom.scoreMarker.style.display = "none";
    }
  }

  function resetVerdictCard() {
    dom.verdictCard3D.classList.remove("flipped");
    dom.humanReviewWidget.classList.add("hidden");
  }

  // ========================================================================
  // STAGE 5 REVIEW RESOLUTION
  // ========================================================================

  async function resolveReview(label) {
    if (!state.lastEvaluationResult || !state.lastEvaluationResult.evaluation_id) {
      showNotification("No active evaluation ID to resolve.", "error");
      return;
    }

    const evalId = state.lastEvaluationResult.evaluation_id;
    try {
      const res = await fetch("/api/resolve_review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          evaluation_id: evalId,
          human_label: label,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || "Resolution failed");
      }

      const data = await res.json();
      dom.humanReviewWidget.classList.add("hidden");

      if (label === "good") {
        showNotification(`Approved! Category profile advanced to v${data.profile.version} (n=${data.profile.n_examples}). Adaptive band updated.`, "success");
      } else {
        showNotification(`Resolution recorded as '${label}'. Logged to audit trail without mutating profile.`, "info");
      }

      // Re-fetch categories to reflect updated profile version & band
      await loadCategories();
      // Restore selected category
      dom.categorySelect.value = data.category_id;
      handleCategoryChange();
    } catch (err) {
      showNotification(`Resolution Error: ${err.message}`, "error");
    }
  }

  // ========================================================================
  // PIPELINE 5-STAGE EXPLORER
  // ========================================================================

  function updateStageInspector(stageNumber) {
    const stageStr = String(stageNumber);
    const meta = STAGE_EXPLANATIONS[stageStr];
    if (!meta) return;

    dom.stageInspectorTag.textContent = meta.tag;
    dom.stageInspectorTitle.textContent = meta.title;
    dom.stageInspectorDesc.textContent = meta.desc;

    // Highlight slab in 3D isometric stack
    document.querySelectorAll(".iso-slab").forEach((slab) => {
      const active = slab.getAttribute("data-stage") === stageStr;
      slab.classList.toggle("active", active);
    });

    const cat = state.categories.find((c) => c.category_id === state.selectedCategoryId) || state.categories[0];
    if (!cat) return;

    dom.stageJsonCatLabel.textContent = cat.category_id;

    let jsonState = {};
    if (stageStr === "1") {
      jsonState = {
        category_id: cat.category_id,
        prompt: cat.prompt,
        baseline_examples_count: cat.baseline_examples_count,
        sample_baseline: "France's capital city is the city of Paris.",
      };
    } else if (stageStr === "2") {
      jsonState = {
        category_id: cat.category_id,
        profile_version: cat.profile.version,
        n_examples: cat.profile.n_examples,
        min_length: cat.min_length,
        max_length: cat.max_length,
        required_terms: cat.required_terms,
        banned_terms: cat.banned_terms,
        history_entries: cat.profile.history,
      };
    } else if (stageStr === "3") {
      jsonState = {
        scoring_engine: "OracleEngine",
        embedder: "TfidfEmbedder (max_features=512, ngram=(1,2))",
        penalties: {
          structural_violation_penalty: 0.15,
          invariant_violation_penalty: 0.25,
        },
        formula: "composite_score = max(0, semantic_similarity - structural_pen - invariant_pen)",
      };
    } else if (stageStr === "4") {
      jsonState = {
        adaptive_tracker: "AdaptiveThresholdTracker",
        window_size: 50,
        pass_z: 1.0,
        fail_z: 3.0,
        min_samples: 5,
        spread_floor: 0.02,
        current_band: cat.band,
      };
    } else if (stageStr === "5") {
      jsonState = {
        verdict_aggregator: "VerdictAggregator",
        verdicts: ["PASS", "REVIEW", "FAIL"],
        closed_loop_feedback: {
          on_approved_good: "profile_builder.update_incremental(profile) -> v(n+1); threshold_tracker.record_confirmed_good(score)",
          on_bad: "Logged to audit trail without updating profile/window",
        },
      };
    }

    dom.stageJsonCode.textContent = JSON.stringify(jsonState, null, 2);
  }

  // ========================================================================
  // AUDIT LOGS & PERSISTENCE TAB
  // ========================================================================

  async function loadAuditLogs() {
    try {
      const res = await fetch("/api/audit_logs");
      if (!res.ok) throw new Error("Failed to load audit logs");
      const data = await res.json();
      state.auditData = data;

      // 1. Evaluations Table
      dom.evaluationsTableBody.replaceChildren();
      dom.evalCount.textContent = data.evaluations.length;
      if (data.evaluations.length === 0) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 6;
        td.className = "table-empty";
        td.textContent = "No evaluations recorded in SQLite yet.";
        tr.appendChild(td);
        dom.evaluationsTableBody.appendChild(tr);
      } else {
        data.evaluations.slice().reverse().forEach((ev) => {
          const tr = document.createElement("tr");

          const tdId = document.createElement("td");
          tdId.className = "monospace";
          tdId.textContent = String(ev.id);

          const tdCat = document.createElement("td");
          tdCat.textContent = ev.category_id;

          const tdText = document.createElement("td");
          tdText.textContent = ev.candidate_text.length > 55 ? ev.candidate_text.slice(0, 52) + "..." : ev.candidate_text;
          tdText.title = ev.candidate_text;

          const tdScore = document.createElement("td");
          tdScore.className = "monospace";
          tdScore.textContent = ev.composite_score.toFixed(3);

          const tdVer = document.createElement("td");
          const pill = document.createElement("span");
          pill.className = `status-badge ${ev.verdict.toLowerCase() === "pass" ? "pass" : ev.verdict.toLowerCase() === "review" ? "review" : "fail"}`;
          pill.textContent = ev.verdict;
          tdVer.appendChild(pill);

          const tdTime = document.createElement("td");
          tdTime.className = "monospace";
          tdTime.textContent = ev.evaluated_at ? ev.evaluated_at.split("T")[1]?.slice(0, 8) || ev.evaluated_at : "—";

          tr.append(tdId, tdCat, tdText, tdScore, tdVer, tdTime);
          dom.evaluationsTableBody.appendChild(tr);
        });
      }

      // 2. Resolutions Table
      dom.resolutionsTableBody.replaceChildren();
      dom.resCount.textContent = data.resolutions.length;
      if (data.resolutions.length === 0) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 4;
        td.className = "table-empty";
        td.textContent = "No review resolutions recorded yet.";
        tr.appendChild(td);
        dom.resolutionsTableBody.appendChild(tr);
      } else {
        data.resolutions.slice().reverse().forEach((r) => {
          const tr = document.createElement("tr");

          const tdId = document.createElement("td");
          tdId.className = "monospace";
          tdId.textContent = String(r.id);

          const tdEvalId = document.createElement("td");
          tdEvalId.className = "monospace";
          tdEvalId.textContent = `#${r.evaluation_id}`;

          const tdLabel = document.createElement("td");
          tdLabel.textContent = r.human_label;
          tdLabel.className = r.human_label === "good" ? "color-pass font-bold" : "color-fail font-bold";

          const tdTime = document.createElement("td");
          tdTime.className = "monospace";
          tdTime.textContent = r.resolved_at ? r.resolved_at.split("T")[1]?.slice(0, 8) || r.resolved_at : "—";

          tr.append(tdId, tdEvalId, tdLabel, tdTime);
          dom.resolutionsTableBody.appendChild(tr);
        });
      }

      // 3. Profile Evolution History Table
      dom.historyTableBody.replaceChildren();
      let totalHistory = 0;
      for (const [catId, entries] of Object.entries(data.history)) {
        entries.forEach((h) => {
          totalHistory++;
          const tr = document.createElement("tr");

          const tdCat = document.createElement("td");
          tdCat.textContent = catId;

          const tdVer = document.createElement("td");
          tdVer.className = "monospace";
          tdVer.textContent = `v${h.version}`;

          const tdNote = document.createElement("td");
          tdNote.textContent = h.note;

          const tdTime = document.createElement("td");
          tdTime.className = "monospace";
          tdTime.textContent = h.recorded_at ? h.recorded_at.split("T")[0] + " " + (h.recorded_at.split("T")[1]?.slice(0, 8) || "") : "—";

          tr.append(tdCat, tdVer, tdNote, tdTime);
          dom.historyTableBody.appendChild(tr);
        });
      }

      if (totalHistory === 0) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 4;
        td.className = "table-empty";
        td.textContent = "No history recorded yet.";
        tr.appendChild(td);
        dom.historyTableBody.appendChild(tr);
      }
    } catch (err) {
      showNotification(`Audit Log Error: ${err.message}`, "error");
    }
  }

  async function resetDemoState() {
    try {
      const res = await fetch("/api/reset", { method: "POST" });
      if (!res.ok) throw new Error("Reset failed");
      const data = await res.json();
      dom.resetModal.classList.add("hidden");
      showNotification(data.message, "success");
      await loadCategories();
      await loadAuditLogs();
      resetVerdictCard();
    } catch (err) {
      showNotification(`Reset Error: ${err.message}`, "error");
    }
  }

  // ========================================================================
  // BENCHMARK TAB
  // ========================================================================

  async function fetchBenchmark() {
    if (state.benchmarkData) return state.benchmarkData;
    try {
      const res = await fetch("/api/benchmark");
      if (!res.ok) throw new Error("Failed to load benchmark");
      state.benchmarkData = await res.json();
      return state.benchmarkData;
    } catch (err) {
      console.warn("Benchmark fetch error:", err);
      return null;
    }
  }

  async function renderBenchmarkTab() {
    const data = await fetchBenchmark();
    if (!data) {
      dom.benchCategoryTableBody.replaceChildren();
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 5;
      td.className = "table-empty";
      td.textContent = "Failed to load benchmark summary.";
      tr.appendChild(td);
      dom.benchCategoryTableBody.appendChild(tr);
      return;
    }

    // Top Summary Cards
    dom.benchNaiveAccuracy.textContent = `${data.naive.accuracy}%`;
    dom.benchNaiveCorrect.textContent = `${data.naive.correct} / ${data.total_cases} Correct`;
    dom.benchNaiveFR.textContent = data.naive.false_rejects;
    dom.benchNaiveFA.textContent = data.naive.false_accepts;

    dom.benchAitogAccuracy.textContent = `${data.ai_tog.accuracy}%`;
    dom.benchAitogCorrect.textContent = `${data.ai_tog.correct} / ${data.total_cases} Correct`;
    dom.benchAitogFR.textContent = data.ai_tog.false_rejects;
    dom.benchAitogFA.textContent = data.ai_tog.false_accepts;

    // Category Breakdown Table
    dom.benchCategoryTableBody.replaceChildren();
    for (const [catId, stats] of Object.entries(data.category_breakdown)) {
      const tr = document.createElement("tr");

      const tdCat = document.createElement("td");
      tdCat.textContent = catId;

      const tdNaive = document.createElement("td");
      tdNaive.className = "monospace";
      tdNaive.textContent = `${stats.naive_correct}/${stats.total} (${Math.round((stats.naive_correct / stats.total) * 100)}%)`;

      const tdAitog = document.createElement("td");
      tdAitog.className = "monospace color-pass font-bold";
      tdAitog.textContent = `${stats.ai_tog_correct}/${stats.total} (${Math.round((stats.ai_tog_correct / stats.total) * 100)}%)`;

      const tdNaiveFR = document.createElement("td");
      tdNaiveFR.className = "monospace";
      tdNaiveFR.textContent = `FR: ${stats.naive_fr} | FA: ${stats.naive_fa}`;

      const tdAitogFR = document.createElement("td");
      tdAitogFR.className = "monospace";
      tdAitogFR.textContent = `FR: ${stats.ai_tog_fr} | FA: ${stats.ai_tog_fa}`;

      tr.append(tdCat, tdNaive, tdAitog, tdNaiveFR, tdAitogFR);
      dom.benchCategoryTableBody.appendChild(tr);
    }

    // Known Limitations
    dom.limitationsGrid.replaceChildren();
    data.known_limitations.forEach((item) => {
      const card = document.createElement("div");
      card.className = "limit-item";

      const title = document.createElement("div");
      title.className = "limit-title";
      title.textContent = item.case;

      const badge = document.createElement("span");
      badge.className = "limit-badge";
      badge.textContent = item.observed;

      const why = document.createElement("p");
      why.className = "limit-why";
      why.textContent = item.why;

      card.append(title, badge, why);
      dom.limitationsGrid.appendChild(card);
    });
  }

  // ========================================================================
  // NOTIFICATIONS & HELPERS
  // ========================================================================

  function showNotification(msg, type = "info") {
    dom.notificationMessage.textContent = msg;
    dom.systemNotification.className = `system-notification ${type}`;
    dom.systemNotification.classList.remove("hidden");

    if (type !== "error") {
      setTimeout(() => {
        dom.systemNotification.classList.add("hidden");
      }, 5000);
    }
  }

  // ========================================================================
  // EVENT LISTENERS
  // ========================================================================

  function initEventListeners() {
    dom.categorySelect.addEventListener("change", handleCategoryChange);
    dom.evaluateBtn.addEventListener("click", evaluateCandidate);
    dom.clearInputBtn.addEventListener("click", () => {
      dom.candidateInput.value = "";
      dom.candidateInput.focus();
    });

    // Human Review Resolution
    dom.resolveGoodBtn.addEventListener("click", () => resolveReview("good"));
    dom.resolveBadBtn.addEventListener("click", () => resolveReview("bad"));

    // Stage Slabs in Pipeline 3D view
    document.querySelectorAll(".iso-slab").forEach((slab) => {
      slab.addEventListener("click", () => {
        const st = slab.getAttribute("data-stage");
        updateStageInspector(Number(st));
      });
    });

    // Copy Stage JSON
    dom.copyStageJsonBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(dom.stageJsonCode.textContent).then(() => {
        showNotification("Stage runtime JSON copied to clipboard.", "success");
      });
    });

    // Audit Log buttons
    dom.refreshAuditBtn.addEventListener("click", loadAuditLogs);
    dom.resetDemoBtn.addEventListener("click", () => {
      dom.resetModal.classList.remove("hidden");
    });
    dom.cancelResetBtn.addEventListener("click", () => {
      dom.resetModal.classList.add("hidden");
    });
    dom.confirmResetBtn.addEventListener("click", resetDemoState);

    // Dismiss notification
    dom.closeNotifBtn.addEventListener("click", () => {
      dom.systemNotification.classList.add("hidden");
    });
  }

  // Boot on DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
