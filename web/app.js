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
      desc: "Resolves candidate into PASS, REVIEW, or FAIL. Closed-Loop Feedback: When a human approves a REVIEW case, Stage 5 folds the candidate into the Stage 2 Reference Profile (version++, n=n+1) and appends its score to the Stage 4 rolling threshold window. Rejected cases are logged to SQLite without profile mutation.",
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
    gaugeStage: document.getElementById("gaugeStage"),
    gaugeTrack: document.getElementById("gaugeTrack"),
    slabFail: document.getElementById("slabFail"),
    slabReview: document.getElementById("slabReview"),
    slabPass: document.getElementById("slabPass"),
    labelSlabFail: document.getElementById("labelSlabFail"),
    labelSlabReview: document.getElementById("labelSlabReview"),
    labelSlabPass: document.getElementById("labelSlabPass"),
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
    initSpotlightAndDepthLayers();
    initSmallElementTilt();
    initNavigation();
    initPresets();
    initEventListeners();

    // Listen for reduced motion changes
    window.matchMedia("(prefers-reduced-motion: reduce)").addEventListener("change", (e) => {
      state.isReducedMotion = e.matches;
    });

    await loadCategories();
    // Pre-fetch benchmark in background
    fetchBenchmark();
  }

  // ========================================================================
  // THEME & AMBIENT CANVAS (CYBER NETWORK GRAPH)
  // ========================================================================

  function initTheme() {
    document.documentElement.setAttribute("data-theme", state.theme);
    dom.themeToggle.addEventListener("click", () => {
      state.theme = state.theme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", state.theme);
      localStorage.setItem("aitog-theme", state.theme);
    });
  }

  // PART A #9: Cyber Network Graph Canvas (drifting nodes, connecting lines, slow scanline)
  function initCanvasParticles() {
    if (!dom.ambientCanvas) return;
    const canvas = dom.ambientCanvas;
    const ctx = canvas.getContext("2d");
    let width, height;
    let nodes = [];
    let mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
    let scanlineY = 0;
    let animationFrameId = null;
    let isRunning = false;

    function resize() {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    }

    function createNodes() {
      nodes = [];
      const count = Math.min(42, Math.max(18, Math.floor(width / 38)));
      for (let i = 0; i < count; i++) {
        nodes.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.28,
          vy: (Math.random() - 0.5) * 0.28,
          radius: Math.random() * 1.5 + 1.0,
          color: i % 3 === 0 ? "139, 92, 255" : "0, 240, 255",
          alpha: Math.random() * 0.4 + 0.2,
        });
      }
    }

    function animate() {
      if (document.hidden || state.isReducedMotion) {
        isRunning = false;
        return;
      }
      ctx.clearRect(0, 0, width, height);

      mouse.x += (mouse.targetX - mouse.x) * 0.05;
      mouse.y += (mouse.targetY - mouse.y) * 0.05;

      const isDark = state.theme === "dark";

      // Slow cyber scanline
      scanlineY = (scanlineY + 0.45) % height;
      ctx.fillStyle = isDark ? "rgba(0, 240, 255, 0.025)" : "rgba(105, 51, 211, 0.02)";
      ctx.fillRect(0, scanlineY, width, 2);

      // Update node positions with mouse parallax
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        n.x += n.vx;
        n.y += n.vy;

        if (n.x < -10) n.x = width + 10;
        if (n.x > width + 10) n.x = -10;
        if (n.y < -10) n.y = height + 10;
        if (n.y > height + 10) n.y = -10;

        n.renderX = n.x + mouse.x * (n.radius * 1.2);
        n.renderY = n.y + mouse.y * (n.radius * 1.2);
      }

      // Draw connecting lines between nodes within 95px
      const maxDist = 95;
      const maxDistSq = maxDist * maxDist;
      ctx.lineWidth = 0.75;

      for (let i = 0; i < nodes.length; i++) {
        const ni = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const nj = nodes[j];
          const dx = ni.renderX - nj.renderX;
          const dy = ni.renderY - nj.renderY;
          const distSq = dx * dx + dy * dy;

          if (distSq < maxDistSq) {
            const dist = Math.sqrt(distSq);
            const lineAlpha = (1 - dist / maxDist) * 0.16;
            ctx.strokeStyle = isDark ? `rgba(0, 240, 255, ${lineAlpha})` : `rgba(105, 51, 211, ${lineAlpha})`;
            ctx.beginPath();
            ctx.moveTo(ni.renderX, ni.renderY);
            ctx.lineTo(nj.renderX, nj.renderY);
            ctx.stroke();
          }
        }
      }

      // Draw node dots
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        ctx.beginPath();
        ctx.arc(n.renderX, n.renderY, n.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${n.color}, ${n.alpha * (isDark ? 0.75 : 0.5)})`;
        ctx.fill();
      }

      animationFrameId = requestAnimationFrame(animate);
    }

    function startLoop() {
      if (!isRunning && !document.hidden && !state.isReducedMotion) {
        isRunning = true;
        animationFrameId = requestAnimationFrame(animate);
      }
    }

    function stopLoop() {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
      }
      isRunning = false;
    }

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stopLoop();
      else startLoop();
    });

    window.addEventListener("resize", () => {
      resize();
      createNodes();
    });

    window.addEventListener("mousemove", (e) => {
      mouse.targetX = (e.clientX / width - 0.5) * 18;
      mouse.targetY = (e.clientY / height - 0.5) * 18;
    });

    resize();
    createNodes();
    startLoop();
  }

  // PART A #1, #2, #3: Spotlight Edge & Depth Layers Parallax via single shared rAF loop
  function initSpotlightAndDepthLayers() {
    if (state.isReducedMotion) return;

    const cards = Array.from(document.querySelectorAll(".card"));
    if (cards.length === 0) return;

    let targetMouse = { x: -1000, y: -1000 };
    let currentMouse = { x: -1000, y: -1000 };
    let rafId = null;

    window.addEventListener("pointermove", (e) => {
      targetMouse.x = e.clientX;
      targetMouse.y = e.clientY;
    }, { passive: true });

    function tick() {
      if (document.hidden || state.isReducedMotion) {
        rafId = requestAnimationFrame(tick);
        return;
      }

      // Lerp mouse coordinates (factor ~0.08)
      currentMouse.x += (targetMouse.x - currentMouse.x) * 0.08;
      currentMouse.y += (targetMouse.y - currentMouse.y) * 0.08;

      const viewportH = window.innerHeight;

      for (let i = 0; i < cards.length; i++) {
        const card = cards[i];
        const rect = card.getBoundingClientRect();

        // Skip off-screen cards
        if (rect.bottom < -50 || rect.top > viewportH + 50) continue;

        const localX = currentMouse.x - rect.left;
        const localY = currentMouse.y - rect.top;

        // Update spotlight coordinates
        card.style.setProperty("--mx", `${localX.toFixed(1)}px`);
        card.style.setProperty("--my", `${localY.toFixed(1)}px`);

        // Check if cursor is over or near card
        const isHovered = localX >= 0 && localX <= rect.width && localY >= 0 && localY <= rect.height;
        const targetNormX = isHovered ? ((localX - rect.width / 2) / (rect.width / 2)) : 0;
        const targetNormY = isHovered ? ((localY - rect.height / 2) / (rect.height / 2)) : 0;

        card._normX = (card._normX || 0) + (targetNormX - (card._normX || 0)) * 0.08;
        card._normY = (card._normY || 0) + (targetNormY - (card._normY || 0)) * 0.08;

        const layers = card.querySelectorAll("[data-depth]");
        if (layers.length > 0) {
          if (Math.abs(card._normX) > 0.002 || Math.abs(card._normY) > 0.002) {
            layers.forEach((layer) => {
              const depth = parseFloat(layer.getAttribute("data-depth")) || 1;
              const maxPx = depth === 1 ? 1.5 : depth === 2 ? 3.5 : 6.0;
              const dx = (card._normX * maxPx).toFixed(2);
              const dy = (card._normY * maxPx).toFixed(2);
              layer.style.transform = `translate3d(${dx}px, ${dy}px, 0)`;
            });
            card._hasMoved = true;
          } else if (card._hasMoved) {
            layers.forEach((layer) => {
              layer.style.transform = "";
            });
            card._hasMoved = false;
          }
        }
      }

      rafId = requestAnimationFrame(tick);
    }

    rafId = requestAnimationFrame(tick);
  }

  // PART A #4: Small Element Tilt (Verdict Badge, KPI chips) <= 8deg, 600ms spring return
  function initSmallElementTilt() {
    if (state.isReducedMotion) return;

    const targets = document.querySelectorAll(".verdict-badge-pill, .stat-number, .kpi-card");

    targets.forEach((el) => {
      el.addEventListener("pointermove", (e) => {
        if (state.isReducedMotion) return;
        const rect = el.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width - 0.5; // -0.5 to 0.5
        const y = (e.clientY - rect.top) / rect.height - 0.5;
        const rotX = (-y * 16).toFixed(2); // Max 8 deg
        const rotY = (x * 16).toFixed(2);  // Max 8 deg
        el.style.transition = "transform 0.05s ease-out";
        el.style.transform = `perspective(500px) rotateX(${rotX}deg) rotateY(${rotY}deg)`;
      });

      el.addEventListener("pointerleave", () => {
        if (state.isReducedMotion) return;
        el.style.transition = "transform 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275)";
        el.style.transform = "perspective(500px) rotateX(0deg) rotateY(0deg)";
      });
    });
  }

  // Smooth number count-up animation
  function animateNumber(el, targetVal, decimals = 4, duration = 380) {
    if (!el || targetVal === null || targetVal === undefined) return;
    if (state.isReducedMotion) {
      el.textContent = targetVal.toFixed(decimals);
      return;
    }
    const currentText = el.textContent.trim();
    const startVal = (currentText === "—" || currentText === "") ? targetVal : (parseFloat(currentText) || 0);
    if (Math.abs(startVal - targetVal) < 0.0001) {
      el.textContent = targetVal.toFixed(decimals);
      return;
    }
    const startTime = performance.now();
    function step(now) {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = startVal + (targetVal - startVal) * ease;
      el.textContent = current.toFixed(decimals);
      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        el.textContent = targetVal.toFixed(decimals);
      }
    }
    requestAnimationFrame(step);
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

  // PART A #5: Cyber Decrypt Text Scrambler (Resolves to PASS/REVIEW/FAIL over ~600ms)
  function scrambleDecrypt(targetText, element, duration = 600, onComplete) {
    if (state.isReducedMotion) {
      element.textContent = targetText;
      if (onComplete) onComplete();
      return;
    }

    const chars = "0123456789ABCDEF!#@%&*<>[]:/~";
    const length = targetText.length;
    const startTime = performance.now();

    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      const resolvedCount = Math.floor(progress * length);

      let output = "";
      for (let i = 0; i < length; i++) {
        if (i < resolvedCount) {
          output += targetText[i];
        } else {
          output += chars[Math.floor(Math.random() * chars.length)];
        }
      }

      element.textContent = output;

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        element.textContent = targetText;
        if (onComplete) onComplete();
      }
    }

    requestAnimationFrame(update);
  }

  function renderEvaluationResult(result) {
    const backFace = dom.verdictCard3D.querySelector(".verdict-back");

    // Remove prior verdict classes, glow bursts, and glitch
    backFace.classList.remove("verdict-is-pass", "verdict-is-review", "verdict-is-fail", "glow-burst-pass", "glow-burst-review", "glow-burst-fail");
    dom.verdictCard3D.classList.remove("glitch-active", "glow-burst-pass", "glow-burst-review", "glow-burst-fail");

    // Transition to revealed state (flat layout, no coin-flip rotate)
    dom.verdictCard3D.classList.add("flipped");

    const vLower = result.verdict.toLowerCase();

    // 1. Scan-line sweep across the verdict card
    const sweep = document.getElementById("scanlineSweep");
    if (sweep && !state.isReducedMotion) {
      sweep.classList.remove("sweeping");
      void sweep.offsetWidth; // trigger reflow
      sweep.classList.add("sweeping");
    }

    // 2. Text "decrypt" scramble (random hex/glyphs resolving to PASS/REVIEW/FAIL over ~600ms)
    scrambleDecrypt(result.verdict, dom.verdictBadge, 600, () => {
      // 3. Glow pulse in verdict colour upon resolution
      backFace.classList.add(`verdict-is-${vLower}`, `glow-burst-${vLower}`);
      dom.verdictCard3D.classList.add(`glow-burst-${vLower}`);

      // 4. FAIL adds a brief 150ms RGB-split glitch. PASS/REVIEW do NOT glitch.
      if (result.verdict === "FAIL" && !state.isReducedMotion) {
        dom.verdictCard3D.classList.add("glitch-active");
        setTimeout(() => {
          dom.verdictCard3D.classList.remove("glitch-active");
        }, 150);
      }
    });

    dom.verdictEvalId.textContent = `EVAL #${result.evaluation_id ?? "—"}`;
    dom.verdictExplanation.textContent = result.explanation;

    // Show / Hide Stage 5 Human Review widget
    if (result.verdict === "REVIEW") {
      dom.humanReviewWidget.classList.remove("hidden");
    } else {
      dom.humanReviewWidget.classList.add("hidden");
    }

    // Update Depth Gauge
    updateDepthGauge(result.band, result.signals.composite_score);

    // Update Stage 3 Scorecard
    renderScorecard(result);
  }

  function renderScorecard(result) {
    const sig = result.signals;
    const pen = result.penalties;

    // 1. Semantic Similarity
    animateNumber(dom.sigSemanticScore, sig.semantic_similarity, 4);
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
    animateNumber(dom.sigCompositeScore, sig.composite_score, 4);
    let formula = `semantic (${sig.semantic_similarity.toFixed(3)})`;
    if (pen.applied_structural) formula += ` - 0.15 struct`;
    if (pen.applied_invariant) formula += ` - 0.25 invar`;
    dom.sigCompositeFormula.textContent = formula;
  }

  // ========================================================================
  // STAGE 4 DEPTH GAUGE (Unified Linear Mapping Function)
  // ========================================================================

  function updateDepthGauge(band, score) {
    if (!band) return;

    animateNumber(dom.gaugeCenterVal, band.center, 4);
    animateNumber(dom.gaugeSpreadVal, band.spread, 4);
    animateNumber(dom.gaugePassCutVal, band.pass_cut, 4);
    animateNumber(dom.gaugeReviewCutVal, band.review_cut, 4);

    dom.coldStartChip.textContent = band.is_cold_start ? "Cold Start Fallback" : "Adaptive Window Band";
    dom.coldStartChip.classList.toggle("cold-start", band.is_cold_start);

    // ONE linear mapping function derived from live band
    // min = review_cut - 1.5 * spread, max = 1.0 (with safe guards)
    const spread = Math.max(0.015, band.spread || 0.1);
    const minVal = Math.min(0, band.review_cut - 1.5 * spread);
    const maxVal = 1.0;
    const range = Math.max(0.15, maxVal - minVal);

    const clamp = (v) => Math.max(0, Math.min(1, v));
    const pos = (val) => clamp((val - minVal) / range) * 100;

    const rCutPct = pos(band.review_cut);
    const pCutPct = pos(band.pass_cut);

    // Dynamic zone widths directly computed from the exact same mapping function
    const failWidth = Math.max(1, rCutPct);
    const reviewWidth = Math.max(1, pCutPct - rCutPct);
    const passWidth = Math.max(1, 100 - pCutPct);

    if (dom.slabFail) dom.slabFail.style.width = `${failWidth.toFixed(2)}%`;
    if (dom.slabReview) dom.slabReview.style.width = `${reviewWidth.toFixed(2)}%`;
    if (dom.slabPass) dom.slabPass.style.width = `${passWidth.toFixed(2)}%`;

    // Boundary blades positioned from the exact same cuts
    dom.bladeReview.style.left = `${rCutPct.toFixed(2)}%`;
    dom.bladePass.style.left = `${pCutPct.toFixed(2)}%`;

    // Auto-stagger labels if closer than 60px
    const trackWidth = (dom.gaugeTrack && dom.gaugeTrack.offsetWidth > 0) ? dom.gaugeTrack.offsetWidth : 420;
    const pixelDist = (Math.abs(pCutPct - rCutPct) / 100) * trackWidth;
    const reviewTag = dom.bladeReview.querySelector(".blade-tag");
    const passTag = dom.bladePass.querySelector(".blade-tag");

    if (reviewTag && passTag) {
      if (pixelDist < 60) {
        reviewTag.style.top = "calc(100% + 12px)";
        passTag.style.top = "calc(100% + 34px)";
      } else {
        reviewTag.style.top = "calc(100% + 12px)";
        passTag.style.top = "calc(100% + 12px)";
      }
    }

    // Zone label auto-shorten so text never sits under a blade
    if (dom.labelSlabFail) {
      dom.labelSlabFail.textContent = failWidth < 12 ? "FAIL" : "FAIL ZONE";
    }
    if (dom.labelSlabReview) {
      dom.labelSlabReview.textContent = reviewWidth < 14 ? "REV" : "REVIEW ZONE";
      dom.labelSlabReview.style.display = reviewWidth < 7 ? "none" : "";
    }
    if (dom.labelSlabPass) {
      dom.labelSlabPass.textContent = passWidth < 14 ? "PASS" : "PASS ZONE";
    }

    // Floating score marker positioned using the SAME linear function
    if (score !== null && score !== undefined) {
      const scorePct = pos(score);
      dom.scoreMarker.style.left = `${scorePct.toFixed(2)}%`;
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
        created_at: formatTimestamp(cat.profile.created_at),
        min_length: cat.min_length,
        max_length: cat.max_length,
        required_terms: cat.required_terms,
        banned_terms: cat.banned_terms,
        history_entries: (cat.profile.history || []).map(formatTimestampsInText),
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
  // TIMEZONE FORMATTING HELPER (Asia/Kolkata / IST, UTC+05:30)
  // Backend stores UTC (ISO 8601 with +00:00 / Z, or naive UTC strings).
  // Frontend converts to IST at render time: "21 Oct 2026, 18:23:17 IST"
  // ========================================================================

  const DISPLAY_TZ = "Asia/Kolkata";
  const IST_FORMATTER = new Intl.DateTimeFormat("en-GB", {
    timeZone: DISPLAY_TZ,
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });

  function formatTimestamp(value) {
    if (!value || (typeof value !== "string" && !(value instanceof Date))) return "—";
    try {
      let iso = typeof value === "string" ? value.trim() : value.toISOString();
      if (!iso) return "—";
      // Normalize naive SQL timestamp (e.g. "2026-10-21 12:53:17") to UTC ISO
      if (/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}/.test(iso)) {
        iso = iso.replace(" ", "T") + "Z";
      }
      const d = new Date(iso);
      if (isNaN(d.getTime())) return "—";
      return IST_FORMATTER.format(d) + " IST";
    } catch {
      return "—";
    }
  }

  const ISO_TIMESTAMP_REGEX = /\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?/g;

  function formatTimestampsInText(text) {
    if (!text || typeof text !== "string") return text || "";
    return text.replace(ISO_TIMESTAMP_REGEX, (match) => {
      const formatted = formatTimestamp(match);
      return formatted !== "—" ? formatted : match;
    });
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

      // 1. Evaluations Table (sorted newest-first)
      dom.evaluationsTableBody.replaceChildren();
      const evaluations = (data.evaluations || []).slice().sort((a, b) => (new Date(b.evaluated_at || 0) - new Date(a.evaluated_at || 0)) || (Number(b.id) - Number(a.id)));
      dom.evalCount.textContent = evaluations.length;
      if (evaluations.length === 0) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 6;
        td.className = "table-empty";
        td.textContent = "No evaluations recorded in SQLite yet.";
        tr.appendChild(td);
        dom.evaluationsTableBody.appendChild(tr);
      } else {
        evaluations.forEach((ev) => {
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
          const vLower = (ev.verdict || "").toLowerCase();
          pill.className = `status-badge ${vLower === "pass" ? "pass" : vLower === "review" ? "review" : "fail"}`;
          pill.textContent = ev.verdict;
          tdVer.appendChild(pill);

          const tdTime = document.createElement("td");
          tdTime.className = "monospace cell-timestamp";
          tdTime.textContent = formatTimestamp(ev.evaluated_at);

          tr.append(tdId, tdCat, tdText, tdScore, tdVer, tdTime);
          dom.evaluationsTableBody.appendChild(tr);
        });
      }

      // 2. Resolutions Table (sorted newest-first)
      dom.resolutionsTableBody.replaceChildren();
      const resolutions = (data.resolutions || []).slice().sort((a, b) => (new Date(b.resolved_at || 0) - new Date(a.resolved_at || 0)) || (Number(b.id) - Number(a.id)));
      dom.resCount.textContent = resolutions.length;
      if (resolutions.length === 0) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 4;
        td.className = "table-empty";
        td.textContent = "No review resolutions recorded yet.";
        tr.appendChild(td);
        dom.resolutionsTableBody.appendChild(tr);
      } else {
        resolutions.forEach((r) => {
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
          tdTime.className = "monospace cell-timestamp";
          tdTime.textContent = formatTimestamp(r.resolved_at);

          tr.append(tdId, tdEvalId, tdLabel, tdTime);
          dom.resolutionsTableBody.appendChild(tr);
        });
      }

      // 3. Profile Evolution History Table (sorted ascending by time, v1 before v2)
      dom.historyTableBody.replaceChildren();
      let totalHistory = 0;
      for (const [catId, entries] of Object.entries(data.history || {})) {
        const sortedEntries = (entries || []).slice().sort((a, b) => (new Date(a.recorded_at || 0) - new Date(b.recorded_at || 0)) || (Number(a.version) - Number(b.version)));
        sortedEntries.forEach((h) => {
          totalHistory++;
          const tr = document.createElement("tr");

          const tdCat = document.createElement("td");
          tdCat.textContent = catId;

          const tdVer = document.createElement("td");
          tdVer.className = "monospace";
          tdVer.textContent = `v${h.version}`;

          const tdNote = document.createElement("td");
          tdNote.textContent = formatTimestampsInText(h.note || "");

          const tdTime = document.createElement("td");
          tdTime.className = "monospace cell-timestamp";
          tdTime.textContent = formatTimestamp(h.recorded_at);

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
