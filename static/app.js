/* seq2formula dashboard logic: WS event stream, uPlot live charts,
   KaTeX formula rendering, run controls, history / hall / curiosities. */

"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  ws: null,
  wsOk: false,
  status: "idle",          // idle | running | paused | finished
  puzzle: null,            // current puzzle object or null
  runStartedAt: null,      // client clock, for the elapsed KPI
  genTimestamps: [],       // for the measured generations/s KPI
  buffers: null,           // per-run chart buffers
  lastFormulaText: "",
  charts: {},
  rafPending: false,
  raceLog: [],             // event feed entries, newest first
  lastGenNum: null,
  lastExactSize: Infinity,
  lastLoggedBest: null,
  lastRunSolved: false,
  ops: new Set(["+", "-", "*", "/", "^"]),  // enabled binary operators
  settings: null,          // engine knobs from /api/settings (Settings tab)
  settingDefaults: null,
  lastTopUpdate: 0,        // throttle the top-10 list (100 gen/s is unreadable)
  lastLogGen: null,        // throttle the event feed
  wasShocking: false,      // dedupe shock entries in the race log
  pendingPuzzleRestore: "", // remembered puzzle id, applied once the library loads
  isAutopilot: false,      // is the autopilot queue currently running
  mode: "sequence",        // sequence | curve — what the current run is
  curveStrokes: [],        // target strokes: arrays of {x, y} in the unit square
  curveStroke: [],         // freehand stroke currently being drawn
  curveDrawing: false,
  curveBuf: null,          // per-run curve chart buffers
  curveBestCurves: null,   // latest best curves: [{xs, ys}] per stroke
  curveSolved: false,
  curveRafPending: false,
  lastCurveTexts: "",
  curveLog: [],            // curve race log entries, newest first
  curvePreset: null,       // name of the loaded preset shape (or null)
  curveGrid: 0,            // grid cells per side for snapping (0 = freehand)
  curveErase: false,       // eraser toggle: pointer removes nearby points
  curveErasing: false,     // an erase gesture is in progress
  curveSmooth: false,      // round the corners of freehand strokes
  lastCurveTopUpdate: 0,   // throttle the curve top list
  wasCurveShocking: false, // dedupe shock entries in the curve race log
  lastCurveLogGen: null,   // throttle "new best" entries
  champions: [],           // sequence-mode best-formula history for the evolution playback
  evoPlaying: false,       // evolution playback in progress
  evoTimer: null,          // playback stepper handle
  curvePlayRaf: null,      // curve playback animation handle
};

/* ---------- formatting ---------- */

function fmtNum(v) {
  if (v === null || v === undefined) return "—";
  if (v === 0) return "0";
  if (!isFinite(v)) return "∞";
  const abs = Math.abs(v);
  if (abs >= 1000) return Math.round(v).toLocaleString("ru-RU");
  if (abs >= 1) return String(parseFloat(v.toFixed(3)));
  return String(parseFloat(v.toPrecision(3)));
}

function fmtPercent(v) {
  if (v === null || v === undefined) return "—";
  return fmtNum(v * 100) + "%";
}

function fmtTime(sec) {
  if (sec === null || sec === undefined) return "—";
  if (sec < 60) return fmtNum(sec) + " s";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m} m ${String(s).padStart(2, "0")} s`;
}

function fmtWhen(iso) {
  const d = new Date(iso);
  return d.toLocaleString("ru-RU", {
    day: "2-digit", month: "2-digit",
    hour: "2-digit", minute: "2-digit",
  });
}

/* ---------- AST -> LaTeX ---------- */

function numText(v) {
  if (Number.isInteger(v)) return String(v);
  return String(parseFloat(v.toPrecision(6)));
}

function isSimple(tex) {
  // a bare token that never needs parentheses: n, 2, 0.5
  return /^(n|-?\d+(\.\d+)?)$/.test(tex);
}

function negativeConst(ast) {
  if (ast.t === "c" && ast.v < 0) return ast.v;
  if (ast.t === "~" && ast.x.t === "c") return -ast.x.v;
  return null;
}

function astToLatex(ast) {
  if (ast.t === "c") {
    // special constants survive the JSON round-trip as exact doubles
    if (Math.abs(ast.v - Math.PI) < 1e-12) return "\\pi";
    if (Math.abs(ast.v - Math.E) < 1e-12) return "e";
    const s = numText(ast.v);
    return ast.v < 0 ? `\\left(${s}\\right)` : s;
  }
  if (ast.t === "n") return "n";
  if (ast.t === "~") {
    const inner = astToLatex(ast.x);
    return "-" + (isSimple(inner) ? inner : `\\left(${inner}\\right)`);
  }
  if (ast.t === "f") {
    const inner = astToLatex(ast.x);
    switch (ast.n) {
      case "sqrt": return `\\sqrt{${inner}}`;
      case "cbrt": return `\\sqrt[3]{${inner}}`;
      case "abs": return `\\left|${inner}\\right|`;
      case "floor": return `\\left\\lfloor${inner}\\right\\rfloor`;
      case "fact": {
        // postfix ! binds tightest: parens unless the argument is a bare
        // variable, a non-negative constant, or an already-parenthesized
        // function call (n!! would read ambiguously)
        const bare = ast.x.t === "n"
          || (ast.x.t === "c" && ast.x.v >= 0)
          || (ast.x.t === "f" && ast.x.n !== "fact");
        return `${bare ? inner : `\\left(${inner}\\right)`}!`;
      }
      default:
        return `\\operatorname{${ast.n}}\\left(${inner}\\right)`;
    }
  }
  if (ast.t === "if") {
    // piecewise fork: the cases environment draws its own left brace
    return `\\begin{cases}${astToLatex(ast.a)} & \\text{if } ${astToLatex(ast.c)} \\\\ ${astToLatex(ast.b)} & \\text{otherwise}\\end{cases}`;
  }
  // binary op: fold a negative constant into the sign (n + -3 -> n - 3)
  const neg = ast.t === "+" || ast.t === "-" ? negativeConst(ast.r) : null;
  if (neg !== null) {
    const op = ast.t === "+" ? "-" : "+";
    const right = numText(Math.abs(neg));
    return `${astToLatex(ast.l)} ${op} ${right}`;
  }
  const L = astToLatex(ast.l);
  const R = astToLatex(ast.r);
  switch (ast.t) {
    case "+":
      return `${L} + ${R}`;
    case "-":
      return `${L} - ${/^-| \+ | - /.test(R) ? `\\left(${R}\\right)` : R}`;
    case "*":
      // juxtapose a leading coefficient: 2 * n -> 2n, 0.5 * (n+3) -> 0.5(n+3)
      if (/^\d+(\.\d+)?$/.test(L) && !/^\d/.test(R)) return `${L}${R}`;
      return `${L} \\cdot ${R}`;
    case "/":
      return `\\frac{${L}}{${R}}`;
    case "mod": {
      // mod has the same precedence as +/-: wrap such children so the
      // reading order is unambiguous (a+b mod c reads as a+(b mod c) to
      // most people, but the AST means (a+b) mod c)
      const wrapL = ast.l.t === "+" || ast.l.t === "-" || ast.l.t === "mod";
      const wrapR = ast.r.t === "+" || ast.r.t === "-" || ast.r.t === "mod";
      return `${wrapL ? `\\left(${L}\\right)` : L} \\bmod ${wrapR ? `\\left(${R}\\right)` : R}`;
    }
    case "^":
      return `${isSimple(L) ? L : `\\left(${L}\\right)`}^{${R}}`;
    case "<":
      return `${L} \\lt ${R}`;
    case ">":
      return `${L} \\gt ${R}`;
  }
  return "?";
}

/* Plain reference text ("2^n", "(2n-1)^2") -> LaTeX, best effort. */
function refToLatex(text) {
  let latex = text
    .replace(/\^(-?\d+|[a-z])/g, "^{$1}")
    // Convert written functions into proper math symbols
    .replace(/sqrt\(([^)]+)\)/g, "\\sqrt{$1}")
    .replace(/cbrt\(([^)]+)\)/g, "\\sqrt[3]{$1}")
    .replace(/√\(([^)]+)\)/g, "\\sqrt{$1}")
    .replace(/√([a-zA-Z0-9]+)/g, "\\sqrt{$1}")
    .replace(/sin/g, "\\sin")
    .replace(/cos/g, "\\cos")
    .replace(/tan/g, "\\tan")
    .replace(/ln/g, "\\ln");

  // Convert a simple "A / B" into a fraction for pretty display
  const parts = latex.split("/");
  if (parts.length === 2) {
    return `\\frac{${parts[0].trim()}}{${parts[1].trim()}}`;
  }
  return latex;
}

function renderKatex(el, latex, displayMode) {
  try {
    katex.render(latex, el, { displayMode, throwOnError: true });
  } catch (e) {
    el.textContent = latex; // never blank the panel on a rendering hiccup
  }
}

/* Curve mode: the grammar's variable "n" plays the role of the parameter;
   relabel it to t in the rendered LaTeX. Word boundaries only — \sin, \ln
   and friends must survive untouched. */
function astToLatexT(ast) {
  return astToLatex(ast).replace(/(?<![a-zA-Z\\])n(?![a-zA-Z])/g, "t");
}

/* ---------- charts (uPlot) ---------- */

const CHART_COLORS = {
  accent: "#8b93f8", muted: "#b4b4ba", grid: "#232327",
  faint: "#94949b", amber: "#e2a655", teal: "#4ec9b0", positive: "#3fc07a",
};

function axisOpts(digits = 0) {
  return {
    // stroke colors both the axis line and the tick labels; without it
    // uPlot draws black labels that vanish on the dark theme
    stroke: CHART_COLORS.faint,
    grid: { stroke: CHART_COLORS.grid },
    ticks: { stroke: CHART_COLORS.grid },
    values: (u, vals) => vals.map((v) =>
      Math.abs(v) >= 1000 ? String(Math.round(v)) : String(parseFloat(v.toFixed(digits)))),
  };
}

function baseOpts(height, series, legend) {
  return {
    width: 100,
    height,
    series,
    legend: { show: legend },
    cursor: { drag: { x: false, y: false, setScale: false }, x: false, y: false },
    pxAlign: false,
  };
}

function lineSeries(label, color, dash) {
  return {
    label,
    stroke: color,
    width: label === "best" ? 2 : 1.5,
    dash,
    points: { show: false },
  };
}

function makeCharts() {
  // Axis titles make the charts self-explanatory: without them "1 -> 400"
  // on the size histogram is meaningless to a reader.
  const axisX = { ...axisOpts(), label: "generation", values: (u, vals) => vals.map((v) => String(Math.round(v))) };
  const common = { ...baseOpts(170, [
      { label: "generation" },
      lineSeries("best", CHART_COLORS.accent),
      lineSeries("mean", CHART_COLORS.muted, [4, 4]),
    ], true), scales: { x: { time: false } }, axes: [axisX, { ...axisOpts(3), label: "fitness" }] };

  state.charts.fitness = new uPlot({ ...common }, [[], [], []], $("chartFitness"));

  state.charts.error = new uPlot(
    { ...common,
      series: [
        { label: "generation" },
        { ...lineSeries("best", CHART_COLORS.amber), width: 2 },
        lineSeries("mean", CHART_COLORS.muted, [4, 4]),
      ],
      axes: [axisX, { ...axisOpts(3), label: "error", values: (u, vals) => vals.map((v) => fmtNum(v)) }] },
    [[], [], []], $("chartError"));

  state.charts.diversity = new uPlot(
    { ...baseOpts(140, [
        { label: "generation" },
        { ...lineSeries("diversity", CHART_COLORS.teal), width: 1.5 },
      ], false),
      scales: { x: { time: false } },
      axes: [axisX, { ...axisOpts(2), label: "unique share" }] },
    [[], []], $("chartDiversity"));

  const barsPath = window.uPlot && uPlot.paths && uPlot.paths.bars
    ? uPlot.paths.bars({ align: 1, size: [1, 100] }) : undefined;
  state.charts.sizes = new uPlot(
    { ...baseOpts(140, [
        { label: "size" },
        { label: "trees", stroke: CHART_COLORS.accent, width: 2,
          points: { show: false }, paths: barsPath, fill: "rgba(139,147,248,0.18)" },
      ], false),
      scales: { x: { time: false } },
      axes: [{ ...axisX, label: "formula size" }, { ...axisOpts(), label: "formula count" }] },
    [[], []], $("chartSizes"));

  // hero mini-chart: the formula (line) against the target sequence (dots)
  const dots = (size) => window.uPlot && uPlot.paths && uPlot.paths.points
    ? uPlot.paths.points({ size }) : undefined;
  state.charts.hero = new uPlot(
    { ...baseOpts(110, [
        { label: "n" },
        { label: "target", stroke: CHART_COLORS.faint, width: 0, paths: dots(5),
          points: { show: true } },
        { label: "formula", stroke: CHART_COLORS.accent, width: 2,
          points: { show: true, size: 3 } },
      ], false),
      scales: { x: { time: false } },
      axes: [{ show: false }, { show: false }] },
    [[], [], []], $("heroChart"));

  // curve mode: best/mean error per generation, same look as the
  // sequence error chart
  state.charts.curveError = new uPlot(
    { ...common,
      series: [
        { label: "generation" },
        { ...lineSeries("best", CHART_COLORS.amber), width: 2 },
        lineSeries("mean", CHART_COLORS.muted, [4, 4]),
      ],
      axes: [axisX, { ...axisOpts(3), label: "error",
                      values: (u, vals) => vals.map((v) => fmtNum(v)) }] },
    [[], [], []], $("chartCurveError"));

  // curve mode: full stat parity with Search
  state.charts.curveFitness = new uPlot({ ...common }, [[], [], []],
    $("chartCurveFitness"));

  state.charts.curveDiversity = new uPlot(
    { ...baseOpts(140, [
        { label: "generation" },
        { ...lineSeries("diversity", CHART_COLORS.teal), width: 1.5 },
      ], false),
      scales: { x: { time: false } },
      axes: [axisX, { ...axisOpts(2), label: "unique share" }] },
    [[], []], $("chartCurveDiversity"));

  state.charts.curveSizes = new uPlot(
    { ...baseOpts(140, [
        { label: "size" },
        { label: "pairs", stroke: CHART_COLORS.accent, width: 2,
          points: { show: false }, paths: barsPath, fill: "rgba(139,147,248,0.18)" },
      ], false),
      scales: { x: { time: false } },
      axes: [{ ...axisX, label: "pair size" }, { ...axisOpts(), label: "pair count" }] },
    [[], []], $("chartCurveSizes"));

  resizeCharts();
}

function resizeCharts() {
  const boxes = [
    ["fitness", "chartFitness"], ["error", "chartError"],
    ["diversity", "chartDiversity"], ["sizes", "chartSizes"],
    ["hero", "heroChart"], ["curveError", "chartCurveError"],
    ["curveFitness", "chartCurveFitness"],
    ["curveDiversity", "chartCurveDiversity"], ["curveSizes", "chartCurveSizes"],
  ];
  for (const [key, id] of boxes) {
    const el = $(id);
    const chart = state.charts[key];
    if (el && chart && el.clientWidth > 0) {
      // uPlot's setSize() needs an explicit height: passing width only
      // rebuilds the canvas with height 0 and the chart goes blank
      chart.setSize({ width: el.clientWidth, height: chart.height || 170 });
    }
  }
}

function resetBuffers() {
  state.buffers = {
    gen: [], bestF: [], meanF: [], bestE: [], meanE: [], div: [],
    sizesX: [], sizesY: [],
  };
  state.lastFormulaText = "";
  state.champions = [];
  stopEvolutionPlayback(true);
  updateEvoBtn();
  state.genTimestamps = [];
  state.raceLog = [];
  state.lastExactSize = Infinity;
  state.lastLoggedBest = null;
  state.lastLogGen = null;
  state.lastGenNum = null;
  state.lastTopUpdate = 0;
  state.wasShocking = false;
  if (state.charts.hero) state.charts.hero.setData([[], [], []]);
  renderRaceLog();
}

function scheduleChartUpdate() {
  if (state.rafPending) return;
  state.rafPending = true;
  requestAnimationFrame(() => {
    state.rafPending = false;
    const b = state.buffers;
    if (!b) return;
    state.charts.fitness.setData([b.gen, b.bestF, b.meanF]);
    state.charts.error.setData([b.gen, b.bestE, b.meanE]);
    state.charts.diversity.setData([b.gen, b.div]);
    state.charts.sizes.setData([b.sizesX, b.sizesY]);
    
    // Force the legend to show the latest values by moving the cursor.
    // uPlot's setCursor({ idx }) is flaky if the chart hasn't fully laid
    // out; passing the exact pixel position (left) is more reliable.
    const lastIdx = b.gen.length - 1;
    if (lastIdx >= 0) {
      const f = state.charts.fitness;
      const e = state.charts.error;
      const d = state.charts.diversity;
      const s = state.charts.sizes;
      
      // Only auto-move if the user isn't actively hovering (cursor.event is set on mouseover)
      if (!f.cursor.event) {
        f.setCursor({ left: f.valToPos(b.gen[lastIdx], 'x') });
        e.setCursor({ left: e.valToPos(b.gen[lastIdx], 'x') });
        d.setCursor({ left: d.valToPos(b.gen[lastIdx], 'x') });
        s.setCursor({ left: s.valToPos(b.sizesX[lastIdx], 'x') });
      }
    }
    
    resizeCharts();
  });
}

/* ---------- live updates ---------- */

function setChip(text, cls) {
  const chip = $("liveChip");
  chip.textContent = i18n.t(text);
  chip.className = "chip" + (cls ? " " + cls : "");
  if (cls !== "error") chip.title = "";
}

function setStatus(status) {
  // the async body fills the DOM after awaits — translate after it settles
  setTimeout(() => { if (window.i18n && i18n.lang !== "en") i18n.apply(); }, 500);
  if (status === "running") stopTitleFlash();
  state.status = status;
  const running = status === "running" || status === "paused" || status === "finished";
  $("liveEmpty").hidden = running;
  $("liveGrid").hidden = !running;
  const pauseBtn = $("pauseBtn");
  const stepBtn = $("stepBtn");
  const stopBtn = $("stopBtn");
  const startBtn = $("startBtn");
  pauseBtn.disabled = status !== "running" && status !== "paused";
  stepBtn.disabled = status === "running" || status === "finished";
  stopBtn.disabled = status !== "running" && status !== "paused";
  startBtn.disabled = status === "running" || status === "paused";
  // continue is offered only for a finished run that found an answer
  $("continueBtn").hidden = !(status === "finished" && state.lastRunSolved);
  // operator chips apply to the next run — locked while one is live
  document.querySelectorAll(".op-chip").forEach((chip) => {
    chip.disabled = status === "running" || status === "paused";
  });
  $("pauseLabel").textContent = status === "paused" ? i18n.t("Resume") : i18n.t("Pause");
  if (status === "running") setChip("searching", "run");
  else if (status === "paused") setChip("paused", "paused");
  else if (status === "finished") setChip("finished", "finished");
  else setChip("no run");
  syncCurveUI();
}

function applyGen(msg) {
  if (msg.mode === "curve") {
    applyCurveGen(msg);
    return;
  }
  const b = state.buffers;
  if (!b) return;
  b.gen.push(msg.gen);
  const s = msg.stats;
  b.bestF.push(s.best_fitness);
  b.meanF.push(s.mean_fitness);
  b.bestE.push(s.best_error === null ? null : s.best_error);
  b.meanE.push(s.mean_error === null ? null : s.mean_error);
  b.div.push(s.diversity);
  b.sizesX = s.size_hist.map((_, i) => i + 1);
  b.sizesY = s.size_hist;

  // KPIs (generations are 1-based: a 10000-gen run ends on "10000")
  $("kpiGen").textContent = String(msg.gen + 1);
  const errEl = $("kpiError");
  errEl.textContent = msg.best.error === null ? "∞" : fmtPercent(msg.best.error);
  errEl.classList.toggle("ok", msg.best.error !== null && msg.best.error < 1e-9);
  $("kpiSize").textContent = String(msg.best.size);
  const uniqueCount = Math.round(s.diversity * msg.population);
  $("kpiDiversity").textContent = `${fmtPercent(s.diversity)} (${uniqueCount})`;
  $("kpiEvaluated").textContent =
    ((msg.gen + 1) * (msg.population || 500)).toLocaleString("ru-RU");
  const now = performance.now();
  state.genTimestamps.push(now);
  while (state.genTimestamps.length > 1 &&
         now - state.genTimestamps[0] > 2000) state.genTimestamps.shift();
  const span = (now - state.genTimestamps[0]) / 1000;
  const gps = state.genTimestamps.length > 1 && span > 0.3
    ? fmtNum((state.genTimestamps.length - 1) / span) + " g/s" : "—";
  const elapsed = state.runStartedAt ? fmtTime((now - state.runStartedAt) / 1000) : "—";
  $("kpiSpeedTime").textContent = `${gps} · ${elapsed}`;

  // hero formula: re-render only when the text changes
  if (msg.best.text !== state.lastFormulaText) {
    state.lastFormulaText = msg.best.text;
    renderKatex($("bestFormula"), astToLatex(msg.best.ast), true);
    // remember each distinct champion for the evolution playback
    state.champions.push({
      gen: msg.gen + 1, text: msg.best.text, ast: msg.best.ast,
      values: msg.best.values, error: msg.best.error,
    });
    updateEvoBtn();
  }
  const sub = [];
  sub.push(`fitness ${fmtNum(msg.best.fitness)}`);
  if (msg.best.exact) sub.push(`<span class="ok">${i18n.t("exact match by values")}</span>`);
  $("bestSub").innerHTML = sub.join(" · ");
  $("heroLabel").textContent = i18n.t(msg.solved && !msg.solved_now
    ? "Best formula (shrinking — hunting a shorter one)" : "Best formula");

  // top-10 list: at 100 gen/s it flickers unreadably. Throttle to ~3
  // updates/s; force a redraw on the exact-hit event or during shrink.
  if (now - state.lastTopUpdate > 300 || msg.solved_now || msg.solved) {
    renderTop(msg.top || []);
    state.lastTopUpdate = now;
  }
  updateRaceLog(msg);
  renderTerms(msg.best);
  updateHeroChart(msg.best.values);
  scheduleChartUpdate();
}

/* hero mini-chart: formula vs target dots, redrawn every gen. The y
   window is pinned to the target sequence: a candidate with far-out
   values would stretch the scale and squash everything into bands, so
   out-of-window formula points are hidden (the line breaks there). */
function heroWindow(sequence) {
  const nums = sequence.map(Number);
  const lo = Math.min(...nums), hi = Math.max(...nums);
  const pad = Math.max(0.5, (hi - lo) * 0.15);
  return [lo - pad, hi + pad];
}

function prepPlotData(sequence, values) {
  const [yLo, yHi] = heroWindow(sequence);
  const xs = sequence.map((_, i) => i + 1);
  const got = Array.isArray(values) && values.length === sequence.length
    ? values.map((v) => (v === null || v < yLo || v > yHi ? null : v))
    : xs.map(() => null);
  return [xs, sequence.map(Number), got];
}

function updateHeroChart(values) {
  const seq = state.puzzleTerms;
  const chart = state.charts.hero;
  if (!seq || !seq.length || !chart) return;
  chart.setData(prepPlotData(seq, values));
}

/* Evolution playback: step through every distinct champion formula the
   run has produced, one per step, redrawing the hero chart and formula.
   Long unchanged stretches are already deduped — only text changes are
   recorded. Click again (or start a new run) to stop. */
function updateEvoBtn() {
  const btn = $("evoBtn");
  btn.hidden = state.champions.length < 2;
  btn.textContent = state.evoPlaying ? i18n.t("Stop")
    : i18n.t("Evolution ({0})", state.champions.length);
}

function stopEvolutionPlayback(silent) {
  if (state.evoTimer) {
    clearInterval(state.evoTimer);
    state.evoTimer = null;
  }
  const wasPlaying = state.evoPlaying;
  state.evoPlaying = false;
  if (wasPlaying && !silent) restoreFinalChampion();
  updateEvoBtn();
}

function restoreFinalChampion() {
  const last = state.champions[state.champions.length - 1];
  if (!last) return;
  renderKatex($("bestFormula"), astToLatex(last.ast), true);
  updateHeroChart(last.values);
  $("heroLabel").textContent = i18n.t("Best formula");
}

function playEvolution() {
  if (state.evoPlaying) {
    stopEvolutionPlayback(false);
    return;
  }
  const champs = state.champions;
  if (champs.length < 2) return;
  state.evoPlaying = true;
  updateEvoBtn();
  let i = 0;
  const step = () => {
    if (!state.evoPlaying) return;
    const ch = champs[i];
    renderKatex($("bestFormula"), astToLatex(ch.ast), true);
    updateHeroChart(ch.values);
    $("heroLabel").textContent =
      i18n.t("Evolution {0}/{1} · gen. {2}", i + 1, champs.length, ch.gen)
      + (ch.error === null || ch.error === undefined ? "" : ` · ${fmtPercent(ch.error)}`);
    i += 1;
    if (i >= champs.length) stopEvolutionPlayback(false);
  };
  step();
  state.evoTimer = setInterval(step, 700);
}

/* ---------- hover plot previews (history / hall / curiosities) ---------- */

function showPlotPopover(evt, sequence, values) {
  if (!Array.isArray(sequence) || !sequence.length) return;
  const pop = $("plotPopover");
  pop.hidden = false;
  if (!state.charts.popover) {
    state.charts.popover = new uPlot(
      { ...baseOpts(150, [
          { label: "n" },
          { label: "target", stroke: CHART_COLORS.faint, width: 0,
            paths: window.uPlot && uPlot.paths && uPlot.paths.points
              ? uPlot.paths.points({ size: 4 }) : undefined,
            points: { show: true } },
          { label: "formula", stroke: CHART_COLORS.accent, width: 2,
            points: { show: true, size: 3 } },
        ], false),
        scales: { x: { time: false } },
        axes: [{ ...axisOpts(), values: (u, vals) => vals.map((v) => String(Math.round(v))) },
               { ...axisOpts(3), values: (u, vals) => vals.map((v) => fmtNum(v)) }] },
      [[], [], []], pop);
  }
  const W = 320, H = 190;
  pop.style.left = Math.max(8, Math.min(evt.clientX + 14,
    window.innerWidth - W - 10)) + "px";
  pop.style.top = Math.max(8, Math.min(evt.clientY + 12,
    window.innerHeight - H - 10)) + "px";
  state.charts.popover.setSize({ width: W - 16, height: H - 40 });
  state.charts.popover.setData(prepPlotData(sequence, values));
}

function hidePlotPopover() {
  $("plotPopover").hidden = true;
}

function bindPlotHover(el, getData) {
  el.addEventListener("mouseenter", (e) => {
    const [seq, vals] = getData();
    showPlotPopover(e, seq, vals);
  });
  el.addEventListener("mousemove", (e) => {
    const pop = $("plotPopover");
    if (!pop.hidden) {
      const W = 320, H = 190;
      pop.style.left = Math.max(8, Math.min(e.clientX + 14,
        window.innerWidth - W - 10)) + "px";
      pop.style.top = Math.max(8, Math.min(e.clientY + 12,
        window.innerHeight - H - 10)) + "px";
    }
  });
  el.addEventListener("mouseleave", hidePlotPopover);
}

function renderTop(top) {
  const list = $("raceList");
  list.innerHTML = "";
  top.forEach((item, i) => {
    const li = document.createElement("li");
    li.className = "race-item" + (i === 0 ? " leader" : "");
    
    // Hover preview: plot the formula against the target sequence
    if (state.puzzleTerms && item.values) {
      bindPlotHover(li, () => [state.puzzleTerms, item.values]);
    }
    
    const place = document.createElement("span");
    place.className = "race-place num";
    place.textContent = String(i + 1);
    li.appendChild(place);
    const f = document.createElement("span");
    f.className = "race-formula";
    renderKatex(f, astToLatex(item.ast), false);
    li.appendChild(f);
    
    const meta = document.createElement("span");
    meta.className = "race-meta num";
    meta.textContent = i18n.t("fit {0} · size {1}", fmtNum(item.fitness), item.size);
    meta.title = "Fitness = 1 - error - size_penalty. Closer to 1.0 is better.";
    li.appendChild(meta);
    list.appendChild(li);
  });
}

function pushLogEntry(parts) {
  // parts: {pre, latex, text, cls} — pre/text are plain text, latex is
  // rendered as math between them
  parts.gen = state.lastGenNum;
  state.raceLog.unshift(parts);
  if (state.raceLog.length > 7) state.raceLog.pop();
  renderRaceLog();
}

function renderRaceLog() {
  const list = $("raceLog");
  list.innerHTML = "";
  state.raceLog.forEach((e) => {
    const li = document.createElement("li");
    li.className = "race-log-item" + (e.cls ? " " + e.cls : "");
    const g = document.createElement("span");
    g.className = "race-log-gen num";
    // generations are 1-based everywhere on the dashboard
    g.textContent = e.gen === null || e.gen === undefined
      ? "—" : "gen. " + (e.gen + 1);
    li.appendChild(g);
    if (e.pre) {
      const pre = document.createElement("span");
      pre.textContent = e.pre + " ";
      li.appendChild(pre);
    }
    if (e.latex) {
      const f = document.createElement("span");
      f.className = "race-log-formula";
      renderKatex(f, e.latex, false);
      li.appendChild(f);
    }
    if (e.text) {
      const t = document.createElement("span");
      t.textContent = e.latex ? " " + e.text : e.text;
      li.appendChild(t);
    }
    list.appendChild(li);
  });
}

function updateRaceLog(msg) {
  state.lastGenNum = msg.gen;
  // a stuck run shocks every stagnation window. Instead of hiding them
  // entirely, coalesce: if the newest entry is already a shock, just
  // update its generation. This keeps the feed clean and leaves room
  // for formulas, while still showing that the engine is alive.
  if (msg.shock && !state.wasShocking) {
    const top = state.raceLog[0];
    if (top && top.cls === "log-shock") {
      top.gen = msg.gen;
      renderRaceLog();
    } else {
      pushLogEntry({
        pre: "shock: fresh formulas mixed in",
        cls: "log-shock",
      });
    }
  }
  state.wasShocking = msg.shock;
  if (msg.solved_now) {
    pushLogEntry({ pre: "exact match found", cls: "log-solved" });
  } else if (msg.solved && msg.best.exact && msg.best.size < state.lastExactSize) {
    pushLogEntry({
      pre: i18n.t("shrinking: a shorter formula —"), text: i18n.t("size {0}", msg.best.size),
      cls: "log-shrink",
    });
  }
  state.lastExactSize = msg.best.exact ? msg.best.size : state.lastExactSize;
  if (msg.new_best && !msg.best.exact) {
    // Log the new best, but throttle it to once per 10 generations so
    // a 100 gen/s run doesn't turn the feed into an unreadable blur.
    if (!state.lastLogGen || msg.gen - state.lastLogGen >= 10) {
      pushLogEntry({
        pre: "new best:", latex: astToLatex(msg.best.ast),
        text: i18n.t("error {0}", fmtPercent(msg.best.error)),
      });
      state.lastLogGen = msg.gen;
    }
  } else if (msg.new_best && msg.best.exact && !msg.solved_now) {
    pushLogEntry({ pre: "new best:", latex: astToLatex(msg.best.ast) });
  }
}

function renderTerms(best) {
  const seq = state.puzzleTerms;
  if (!seq) return;
  const head = $("termsHead");
  head.innerHTML = "";
  addCell(head, "th", "n");
  seq.forEach((_, i) => addCell(head, "th", String(i + 1)));

  buildTermsRow($("termsTarget"), "expected", seq.map(fmtNum));
  const got = best.values === null
    ? seq.map(() => "—")
    : best.values.map(fmtNum);
  buildTermsRow($("termsGot"), "got", got);
  const errRow = [];
  if (best.errors === null) {
    seq.forEach(() => errRow.push("—"));
  } else {
    best.errors.forEach((e) => errRow.push(e < 1e-9 ? "0" : fmtPercent(e)));
  }
  const errCells = [];
  buildTermsRow($("termsErr"), "diff", errRow, errCells);
  if (best.errors !== null) {
    best.errors.forEach((e, i) => {
      if (e < 1e-9) errCells[i].classList.add("ok");
    });
  }
  $("termsEmpty").hidden = best.errors !== null;
}

function addCell(row, tag, text, cls) {
  const cell = document.createElement(tag);
  cell.textContent = text;
  if (cls) cell.className = cls;
  row.appendChild(cell);
  return cell;
}

function buildTermsRow(row, label, cells, collect) {
  row.innerHTML = "";
  addCell(row, "td", label);
  cells.forEach((c) => {
    const cell = addCell(row, "td", c, "num");
    if (collect) collect.push(cell);
  });
}

function applySolved(msg) {
  if (state.mode === "curve") {
    applyCurveSolved(msg);
    return;
  }
  state.lastRunSolved = true;
  setStatus("finished");
  setChip("solved", "solved");
  $("resultsDot").hidden = false;
  
  if (state.settings) {
    if (state.settings.notify_sound && state.settings.notify_volume > 0) {
      playSolvedSound(state.settings.notify_volume);
    }
    if (state.settings.notify_flash) {
      startTitleFlash();
    }
  }
  pushLogEntry({
    pre: "solved:", latex: msg.formula ? astToLatex(msg.formula) : null,
    text: i18n.t("{0} generations, {1}", msg.generations, fmtTime(msg.time_s)),
    cls: "log-solved",
  });
  if (msg.formula) {
    renderKatex($("bestFormula"), astToLatex(msg.formula), true);
  }
  if (msg.formula_text) state.lastFormulaText = msg.formula_text;
  const parts = [msg.generations === 1
    ? i18n.t("Solved in 1 generation")
    : i18n.t("Solved in {0} generations", msg.generations)];
  if (msg.time_s !== null) parts.push(fmtTime(msg.time_s));
  const note = $("solvedNote");
  note.classList.add("show");
  note.innerHTML = "";
  const span = document.createElement("span");
  span.textContent = parts.join(" · ") + ". ";
  note.appendChild(span);
  const f = document.createElement("span");
  f.className = "solved-formula";
  if (msg.formula) renderKatex(f, astToLatex(msg.formula), false);
  note.appendChild(f);
  if (msg.shortest_text && msg.formula_text && msg.shortest_text !== msg.formula_text) {
    const short = document.createElement("span");
    short.textContent = " shorter: ";
    note.appendChild(short);
    const s2 = document.createElement("span");
    renderKatex(s2, "\\text{" + msg.shortest_text + "}", false);
    note.appendChild(s2);
  }
  refreshAll();
}

function applyStatusMsg(msg) {
  state.mode = msg.mode || "sequence";
  setStatus(msg.status);
  if ("solved" in msg) state.lastRunSolved = msg.solved === true;
  
  // Detect if autopilot moved to a new sequence and clear the charts
  const seqStr = msg.sequence ? msg.sequence.join(",") : "";
  const oldSeqStr = state.puzzleTerms ? state.puzzleTerms.join(",") : "";
  const seqChanged = msg.sequence && state.puzzleTerms && seqStr !== oldSeqStr;

  if (seqChanged) {
    resetBuffers();
    state.runStartedAt = performance.now();
    $("termsEmpty").hidden = true;
  }

  if (msg.sequence && msg.sequence.length) {
    // Sync the UI inputs when the sequence changes or on first load
    if (seqChanged || (!state.puzzleTerms && msg.puzzle)) {
      $("seqInput").value = msg.sequence.join(" ");
      if (msg.puzzle) {
        $("puzzleSelect").value = msg.puzzle.id;
        const p = PUZZLE_CACHE.find((x) => x.id === msg.puzzle.id);
        if (p) $("refInput").value = p.reference || "";
      } else {
        $("puzzleSelect").value = "";
      }
      updatePuzzleLabel();
      updateOpsHint();
      
      // Keep localStorage in sync so a page reload restores the current task
      localStorage.setItem("seq2formula_seq", $("seqInput").value);
      localStorage.setItem("seq2formula_ref", $("refInput").value);
      localStorage.setItem("seq2formula_puzzle", msg.puzzle ? msg.puzzle.id : "");
    }
    state.puzzleTerms = msg.sequence;
  }
  if (msg.puzzle) {
    state.puzzle = msg.puzzle;
    const p = PUZZLE_CACHE.find((x) => x.id === msg.puzzle.id);
    if (p) markPuzzle(p);
  }
  
  // If we connected mid-run, the server sent the last generation data.
  // Apply it immediately so the UI isn't empty waiting for the next tick.
  if (msg.last_gen && (!state.lastGenNum || msg.last_gen.gen >= state.lastGenNum)) {
    if (msg.mode === "curve") applyCurveGen(msg.last_gen);
    else applyGen(msg.last_gen);
  }

  // Update autopilot state and button
  if ("is_autopilot" in msg) {
    state.isAutopilot = msg.is_autopilot;
    const btn = $("autopilotBtn");
    if (state.isAutopilot) {
      btn.classList.remove("btn-positive");
      btn.classList.add("btn-danger");
      btn.textContent = i18n.t("Stop autopilot");
    } else {
      btn.classList.add("btn-positive");
      btn.classList.remove("btn-danger");
      btn.textContent = i18n.t("Autopilot");
    }
  }

  // Highlight the currently running sequence in the tables
  highlightActiveRow();
  
  // A run started while the pane was hidden leaves charts with 0 width.
  // Force a resize after the browser has laid out the newly unhidden grid.
  if (msg.status === "running") {
    setTimeout(resizeCharts, 10);
  }
}

/* ---------- curve mode ---------- */

/* Mirror of engine/curve.py SOLVE_ERROR: "solved" = mean two-sided
   distance below 2% of the unit square. */
const SOLVE_ERROR_JS = 0.02;
const CURVE_SAMPLES_JS = 128;     // mirror of engine CURVE_SAMPLES
const CURVE_STROKE_POINTS = 64;   // stroke resample target (engine upsamples)
const MAX_CURVE_POINTS = 600;     // server-side cap, mirrored here
const MAX_CURVE_STROKES = 12;     // mirror of engine MAX_STROKES

/* Normalized accessors: gallery entries may hold the new `strokes` list
   or the legacy single-stroke `points` flat cloud; ASTs became per-stroke
   lists in the multi-stroke format. */
function entryStrokes(entry) {
  if (Array.isArray(entry.strokes) && entry.strokes.length) return entry.strokes;
  return [entry.points || []];
}

function astList(ast) {
  return Array.isArray(ast) ? ast : [ast];
}

/* Evenly spaced points along a polyline (proportional to segment
   length) — the building block for the one-stroke preset shapes. */
function alongPath(vertices, n) {
  let len = 0;
  for (let i = 1; i < vertices.length; i++) {
    len += Math.hypot(vertices[i].x - vertices[i - 1].x,
                      vertices[i].y - vertices[i - 1].y);
  }
  const out = [];
  for (let k = 0; k < n; k++) {
    const target = (len * k) / (n - 1);
    let acc = 0;
    for (let i = 1; i < vertices.length; i++) {
      const dx = vertices[i].x - vertices[i - 1].x;
      const dy = vertices[i].y - vertices[i - 1].y;
      const d = Math.hypot(dx, dy);
      if (acc + d >= target || i === vertices.length - 1) {
        const t = d === 0 ? 0 : (target - acc) / d;
        out.push({ x: vertices[i - 1].x + dx * t,
                   y: vertices[i - 1].y + dy * t });
        break;
      }
      acc += d;
    }
  }
  return out;
}

/* One-stroke shape presets, same idea as the puzzle library in
   Results but for curves. Coordinates: unit square, y down. */
const CURVE_PRESETS = [
  { name: "Line", pts: alongPath([{ x: 0.12, y: 0.5 }, { x: 0.88, y: 0.5 }], 30) },
  { name: "Angle", pts: alongPath([{ x: 0.2, y: 0.15 }, { x: 0.2, y: 0.85 },
                                  { x: 0.85, y: 0.85 }], 48) },
  { name: "Circle", pts: Array.from({ length: 48 }, (_, i) => {
      const a = -Math.PI / 2 + (i / 47) * 2 * Math.PI;
      return { x: 0.5 + 0.35 * Math.cos(a), y: 0.5 + 0.35 * Math.sin(a) };
    }) },
  { name: "Wave", pts: Array.from({ length: 60 }, (_, i) => {
      const t = i / 59;
      return { x: 0.1 + 0.8 * t, y: 0.5 + 0.25 * Math.sin(4 * Math.PI * t) };
    }) },
  { name: "Zigzag", pts: alongPath([{ x: 0.1, y: 0.25 }, { x: 0.3, y: 0.75 },
                                    { x: 0.5, y: 0.25 }, { x: 0.7, y: 0.75 },
                                    { x: 0.9, y: 0.25 }], 60) },
  { name: "Spiral", pts: Array.from({ length: 90 }, (_, i) => {
      const t = i / 89;
      const a = -Math.PI / 2 + t * 2.5 * 2 * Math.PI;
      const r = 0.04 + 0.32 * t;
      return { x: 0.5 + r * Math.cos(a), y: 0.5 + r * Math.sin(a) };
    }) },
  { name: "Star", pts: (() => {
      const v = [];
      for (let k = 0; k < 10; k++) {
        const a = -Math.PI / 2 + k * Math.PI / 5;
        const r = k % 2 === 0 ? 0.38 : 0.16;
        v.push({ x: 0.5 + r * Math.cos(a), y: 0.5 + r * Math.sin(a) });
      }
      v.push({ ...v[0] });  // closed outline, still one stroke
      return alongPath(v, 70);
    })() },
  { name: "Heart", pts: Array.from({ length: 60 }, (_, i) => {
      const th = (i / 59) * 2 * Math.PI;
      // classic heart curve, rescaled to the unit square (y down)
      return { x: 0.5 + 0.02 * 16 * Math.sin(th) ** 3,
               y: 0.35 - 0.02 * (13 * Math.cos(th) - 5 * Math.cos(2 * th)
                 - 2 * Math.cos(3 * th) - Math.cos(4 * th)) };
    }) },
];

function resetCurveState() {
  state.curveBuf = {
    gen: [], bestF: [], meanF: [], bestE: [], meanE: [], div: [],
    sizesX: [], sizesY: [],
  };
  state.curveBestCurves = null;
  state.curveSolved = false;
  state.lastCurveTexts = "";
  state.curveLog = [];
  state.lastCurveTopUpdate = 0;
  state.wasCurveShocking = false;
  state.lastCurveLogGen = null;
  state.genTimestamps = [];
  state.lastGenNum = null;
  if (state.charts.curveError) state.charts.curveError.setData([[], [], []]);
  if (state.charts.curveFitness) state.charts.curveFitness.setData([[], [], []]);
  if (state.charts.curveDiversity) state.charts.curveDiversity.setData([[], []]);
  if (state.charts.curveSizes) state.charts.curveSizes.setData([[], []]);
  renderCurveLog();
}

/* Total target points across all strokes (the server caps at 200). */
function totalCurvePoints() {
  return state.curveStrokes.reduce((sum, s) => sum + s.length, 0);
}

/* Enable/disable the curve controls and keep the pane visible exactly
   while a curve run exists (running, paused or finished). */
function syncCurveUI() {
  const running = state.status === "running" || state.status === "paused";
  $("curveStartBtn").disabled = running || !totalCurvePoints();
  $("curveStopBtn").disabled = !running;
  $("curveUndoBtn").disabled = running || !state.curveStrokes.length;
  // the target is baked into the engine at start — no edits mid-run
  $("curveClearBtn").disabled = running;
  $("curveEraseBtn").disabled = running;
  $("curveSmoothBtn").disabled = running;
  $("curveCanvas").classList.toggle("locked", running);
  const chip = $("curveChip");
  chip.textContent = i18n.t(state.status === "running" ? "searching"
    : state.status === "paused" ? "paused"
    : state.status === "finished" ? "finished" : "no run");
  const showLive = state.mode === "curve" && (running || state.status === "finished");
  $("curveEmpty").hidden = showLive;
  $("curveLive").hidden = !showLive;
}

/* Round the corners of a raw freehand stroke: two passes of Chaikin
   corner cutting (75/25) before resampling. Only for freehand mode —
   grid strokes are already straight between their nodes. */
function smoothStroke(raw) {
  if (!raw || raw.length < 3) return raw;
  let pts = raw;
  for (let pass = 0; pass < 2; pass++) {
    const out = [pts[0]];
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      out.push({ x: a.x * 0.75 + b.x * 0.25, y: a.y * 0.75 + b.y * 0.25 },
               { x: a.x * 0.25 + b.x * 0.75, y: a.y * 0.25 + b.y * 0.75 });
    }
    out.push(pts[pts.length - 1]);
    pts = out;
  }
  return pts;
}

/* Drop consecutive duplicate points (the grid snap produces them while
   the pointer lingers inside one cell). */
function dedupeStroke(raw) {
  const out = [];
  for (const p of raw) {
    const last = out[out.length - 1];
    if (!last || last.x !== p.x || last.y !== p.y) out.push(p);
  }
  return out;
}

/* Eraser: drop every target point within the brush radius of the pointer
   (radius is canvas pixels converted to the unit square). */function eraseAt(p) {
  const r = 24 / 800;  // ~24 px brush on the 800px canvas
  let changed = false;
  const kept = [];
  for (const stroke of state.curveStrokes) {
    const rest = stroke.filter((q) => {
      const far = Math.hypot(q.x - p.x, q.y - p.y) > r;
      if (!far) changed = true;
      return far;
    });
    if (rest.length) kept.push(rest);
  }
  if (!changed) return;
  state.curveStrokes = kept;
  state.curvePreset = null;  // hand-edited now, not a preset
  syncCurveUI();
  drawCurveCanvas();
}

/* Resample a freehand stroke to ~64 evenly spaced points (by arc length,
   like the engine samples t); a short stroke collapses to one point,
   which is how a simple click becomes a single target point. */
function resampleStroke(raw) {
  if (!raw || !raw.length) return [];
  if (raw.length === 1) return [raw[0]];
  let len = 0;
  for (let i = 1; i < raw.length; i++) {
    len += Math.hypot(raw[i].x - raw[i - 1].x, raw[i].y - raw[i - 1].y);
  }
  if (len < 0.02) return [raw[raw.length - 1]];
  const n = Math.min(CURVE_STROKE_POINTS, Math.max(2, raw.length));
  const step = len / (n - 1);
  const out = [raw[0]];
  let acc = 0;
  let prev = raw[0];
  for (let i = 1; i < raw.length && out.length < n; i++) {
    let d = Math.hypot(raw[i].x - prev.x, raw[i].y - prev.y);
    while (acc + d >= step && out.length < n) {
      const t = (step - acc) / d;
      const p = { x: prev.x + (raw[i].x - prev.x) * t,
                  y: prev.y + (raw[i].y - prev.y) * t };
      out.push(p);
      d = Math.hypot(raw[i].x - p.x, raw[i].y - p.y);
      prev = p;
      acc = 0;
    }
    acc += d;
    prev = raw[i];
  }
  return out;
}

/* ---- user-saved drawings (server-side, survive any browser) ---- */

async function loadSavedDrawings() {
  const data = await api("/api/drawings");
  const sel = $("curveMySel");
  sel.innerHTML = `<option value="">${i18n.t("— load a shape —")}</option>`;
  for (const d of data.drawings) {
    const opt = document.createElement("option");
    opt.value = String(d.id);
    opt.textContent = i18n.t("{0} ({1} strokes)", d.name, d.strokes.length);
    opt.dataset.strokes = JSON.stringify(d.strokes);
    sel.appendChild(opt);
  }
  $("curveDelBtn").disabled = !data.drawings.length;
  $("curveExportBtn").disabled = !data.drawings.length;
}

function wireSavedDrawings() {
  const sel = $("curveMySel");
  sel.addEventListener("change", () => {
    const opt = sel.selectedOptions[0];
    if (!opt || !opt.value) return;
    state.curveStrokes =
      JSON.parse(opt.dataset.strokes).map((stroke) =>
        stroke.map((p) => Array.isArray(p) ? { x: p[0], y: p[1] } : { ...p }));
    state.curveBestCurves = null;
    state.curveSolved = false;
    state.curvePreset = null;
    drawCurveCanvas();
    syncCurveUI();
  });
  $("curveSaveBtn").addEventListener("click", async () => {
    if (!state.curveStrokes.length) {
      alertErr("nothing to save: the drawing is empty");
      return;
    }
    const name = $("curveSaveName").value.trim();
    if (!name) {
      alertErr("type a shape name in the field left of the button");
      return;
    }
    try {
      const resp = await fetch("/api/drawings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          strokes: state.curveStrokes.map((stroke) =>
            stroke.map((p) => [p.x, p.y])),
        }),
      });
      if (!resp.ok) throw new Error((await resp.json()).detail || "HTTP " + resp.status);
      await loadSavedDrawings();
      // re-select the just-saved name in the reloaded list
      const opt = Array.from(sel.options).find((o) => o.textContent.startsWith(name));
      if (opt) sel.value = opt.value;
      $("curveDelBtn").disabled = false;
      $("curveSaveName").value = "";
    } catch (e) { alertErr(i18n.t("failed to save: ") + e.message); }
  });
  $("curveDelBtn").addEventListener("click", async () => {
    const opt = sel.selectedOptions[0];
    if (!opt || !opt.value) return;
    try {
      const resp = await fetch(`/api/drawings/${opt.value}`, { method: "DELETE" });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      await loadSavedDrawings();
    } catch (e) { alertErr(i18n.t("failed to delete: ") + e.message); }
  });
}

function initCurveCanvas() {
  const canvas = $("curveCanvas");

  const editable = () =>
    state.status !== "running" && state.status !== "paused";

  const toUnit = (e) => {
    const rect = canvas.getBoundingClientRect();
    const p = {
      x: Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)),
    };
    // grid mode: snap to the nearest grid node so strokes come out straight
    if (state.curveGrid > 0) {
      p.x = Math.round(p.x * state.curveGrid) / state.curveGrid;
      p.y = Math.round(p.y * state.curveGrid) / state.curveGrid;
    }
    return p;
  };

  const pushPoint = (p) => {
    const cur = state.curveStroke;
    // skip duplicates: snapped points repeat while the pointer is still
    if (cur.length && cur[cur.length - 1].x === p.x &&
        cur[cur.length - 1].y === p.y) return;
    cur.push(p);
  };

  canvas.addEventListener("pointerdown", (e) => {
    if (!editable()) return;
    if (state.curveErase) {
      canvas.setPointerCapture(e.pointerId);
      state.curveErasing = true;
      eraseAt(toUnit(e));
      return;
    }
    if (state.curveStrokes.length >= MAX_CURVE_STROKES) {
      alertErr(i18n.t("maximum of {0} strokes", MAX_CURVE_STROKES));
      return;
    }
    canvas.setPointerCapture(e.pointerId);
    state.curveDrawing = true;
    state.curveStroke = [toUnit(e)];
    state.curveStrokes.push([]);
    drawCurveCanvas();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!editable()) return;
    if (state.curveErasing) {
      eraseAt(toUnit(e));
      return;
    }
    if (!state.curveDrawing) return;
    pushPoint(toUnit(e));
    drawCurveCanvas();
  });
  const finishStroke = () => {
    if (state.curveErasing) {
      state.curveErasing = false;
      return;
    }
    if (!state.curveDrawing) return;
    state.curveDrawing = false;
    // smoothing rounds the corners of freehand strokes; a snapped grid
    // stroke is already straight between nodes, smoothing it would ruin it
    const raw = (state.curveSmooth && state.curveGrid === 0)
      ? smoothStroke(state.curveStroke) : state.curveStroke;
    // grid strokes keep only their corner nodes — compact and exact;
    // freehand strokes are resampled to ~64 evenly spaced points
    const pts = state.curveGrid > 0
      ? dedupeStroke(raw) : resampleStroke(raw);
    state.curveStroke = [];
    if (totalCurvePoints() + pts.length > MAX_CURVE_POINTS) {
      state.curveStrokes.pop();  // drop the stroke we opened
      alertErr(i18n.t("maximum of {0} target points", MAX_CURVE_POINTS));
    } else {
      state.curveStrokes[state.curveStrokes.length - 1] = pts;
      state.curvePreset = null;  // hand-drawn now, not a preset
    }
    syncCurveUI();
    drawCurveCanvas();
  };
  canvas.addEventListener("pointerup", finishStroke);
  canvas.addEventListener("pointercancel", finishStroke);

  const sel = $("curveGridSel");
  if (sel) {
    const saved = Number(localStorage.getItem("curveGrid") || "0");
    if ([8, 12, 16, 24, 32].includes(saved)) {
      state.curveGrid = saved;
      sel.value = String(saved);
    }
    sel.addEventListener("change", () => {
      state.curveGrid = Number(sel.value) || 0;
      localStorage.setItem("curveGrid", sel.value);
      drawCurveCanvas();
    });
  }

  // eraser + smoothing toggles, persisted like the grid choice
  const eraseBtn = $("curveEraseBtn");
  if (eraseBtn) {
    eraseBtn.addEventListener("click", () => {
      state.curveErase = !state.curveErase;
      eraseBtn.setAttribute("aria-pressed", String(state.curveErase));
    });
  }
  const smoothBtn = $("curveSmoothBtn");
  if (smoothBtn) {
    const savedSmooth = localStorage.getItem("curveSmooth") === "1";
    state.curveSmooth = savedSmooth;
    smoothBtn.setAttribute("aria-pressed", String(savedSmooth));
    smoothBtn.addEventListener("click", () => {
      state.curveSmooth = !state.curveSmooth;
      localStorage.setItem("curveSmooth", state.curveSmooth ? "1" : "0");
      smoothBtn.setAttribute("aria-pressed", String(state.curveSmooth));
    });
  }
}

function drawCurveCanvas(progress = 1) {
  const canvas = $("curveCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // frame of the unit square
  ctx.strokeStyle = CHART_COLORS.grid;
  ctx.lineWidth = 2;
  ctx.strokeRect(1, 1, W - 2, H - 2);

  // snap grid: shows where strokes will land in grid mode
  if (state.curveGrid > 0) {
    ctx.strokeStyle = CHART_COLORS.grid;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.5;
    const N = state.curveGrid;
    ctx.beginPath();
    for (let i = 1; i < N; i++) {
      const v = (i / N) * W;
      ctx.moveTo(v, 0);
      ctx.lineTo(v, H);
      ctx.moveTo(0, v);
      ctx.lineTo(W, v);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // target points — the drawing the user made
  ctx.fillStyle = CHART_COLORS.faint;
  for (const stroke of state.curveStrokes) {
    for (const p of stroke) {
      ctx.beginPath();
      ctx.arc(p.x * W, p.y * H, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // in-progress stroke as feedback
  if (state.curveStroke.length > 1) {
    ctx.strokeStyle = CHART_COLORS.accent;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(state.curveStroke[0].x * W, state.curveStroke[0].y * H);
    for (const p of state.curveStroke) ctx.lineTo(p.x * W, p.y * H);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // best curve per stroke (parametric polylines over 64 samples);
  // progress < 1 draws only the first fraction of each polyline
  const curves = state.curveBestCurves;
  const playBtn = $("curvePlayBtn");
  if (playBtn) playBtn.disabled = !(curves && curves.length);
  if (curves && curves.length) {
    ctx.strokeStyle = state.curveSolved ? CHART_COLORS.positive : CHART_COLORS.accent;
    ctx.lineWidth = 3;
    ctx.lineJoin = "round";
    const nPts = Math.max(2, Math.ceil(curves[0].xs.length * progress));
    for (const c of curves) {
      if (!c || !c.xs || !c.xs.length) continue;
      ctx.beginPath();
      ctx.moveTo(c.xs[0] * W, c.ys[0] * H);
      const n = Math.min(nPts, c.xs.length);
      for (let i = 1; i < n; i++) ctx.lineTo(c.xs[i] * W, c.ys[i] * H);
      ctx.stroke();
    }
  }
}

/* Curve playback: animate the best-fit polylines as if a pen were
   drawing them, ~2.5 s per pass. Any redraw with progress 1 cancels it. */
function playCurveFit() {
  if (state.curvePlayRaf) {
    cancelAnimationFrame(state.curvePlayRaf);
    state.curvePlayRaf = null;
  }
  const dur = 2500;
  const t0 = performance.now();
  const frame = (t) => {
    const p = Math.min(1, (t - t0) / dur);
    drawCurveCanvas(0.5 - 0.5 * Math.cos(Math.PI * p)); // ease in-out
    if (p < 1) {
      state.curvePlayRaf = requestAnimationFrame(frame);
    } else {
      state.curvePlayRaf = null;
    }
  };
  state.curvePlayRaf = requestAnimationFrame(frame);
}

function applyCurveGen(msg) {
  const b = state.curveBuf;
  if (!b) return;
  b.gen.push(msg.gen);
  const s = msg.stats;
  b.bestF.push(s.best_fitness);
  b.meanF.push(s.mean_fitness);
  b.bestE.push(s.best_error === null ? null : s.best_error);
  b.meanE.push(s.mean_error === null ? null : s.mean_error);
  b.div.push(s.diversity);
  b.sizesX = s.size_hist.map((_, i) => i + 1);
  b.sizesY = s.size_hist;

  $("curveGen").textContent = String(msg.gen + 1);
  const errEl = $("curveError");
  errEl.textContent = msg.best.error === null ? "∞" : fmtPercent(msg.best.error);
  errEl.classList.toggle("ok",
    msg.best.error !== null && msg.best.error < SOLVE_ERROR_JS);
  $("curveSize").textContent = String(msg.best.size);
  const uniqueCount = Math.round(s.diversity * msg.population);
  $("curveDiversity").textContent = `${fmtPercent(s.diversity)} (${uniqueCount})`;
  // speed/time KPI — same logic as the sequence pane
  const now = performance.now();
  state.genTimestamps.push(now);
  while (state.genTimestamps.length > 1 &&
         now - state.genTimestamps[0] > 2000) state.genTimestamps.shift();
  const span = (now - state.genTimestamps[0]) / 1000;
  const gps = state.genTimestamps.length > 1 && span > 0.3
    ? fmtNum((state.genTimestamps.length - 1) / span) + " g/s" : "—";
  const elapsed = state.runStartedAt ? fmtTime((now - state.runStartedAt) / 1000) : "—";
  $("curveSpeedTime").textContent = `${gps} · ${elapsed}`;

  // hero pairs: re-render only when the text changes
  const texts = msg.best.x_text.join("\n") + "\n" + msg.best.y_text.join("\n");
  if (texts !== state.lastCurveTexts) {
    state.lastCurveTexts = texts;
    renderCurveHero(msg.best);
  }
  const sub = [`fitness ${fmtNum(msg.best.fitness)}`];
  if (msg.best.solved) sub.push(`<span class="ok">${i18n.t("threshold beaten")}</span>`);
  $("curveSub").innerHTML = sub.join(" · ");

  // top-5 pairs: throttled like the sequence race list
  if (now - state.lastCurveTopUpdate > 300 || msg.solved_now || msg.solved) {
    renderCurveTop(msg.top || []);
    state.lastCurveTopUpdate = now;
  }
  updateCurveRaceLog(msg);
  renderCurveCheck(msg.best);

  state.curveSolved = msg.solved;
  state.curveBestCurves = msg.best.curves;
  scheduleCurveUpdate();
}

/* Hero: one x/y formula block per stroke; the stroke index only shows
   when the drawing actually has several strokes. */
function renderCurveHero(best) {
  const box = $("curveFormulas");
  box.innerHTML = "";
  const multi = best.x_text.length > 1;
  best.x_text.forEach((xt, k) => {
    if (multi) {
      const label = document.createElement("span");
      label.className = "curve-stroke-label";
      label.textContent = i18n.t("Stroke {0}:", k + 1);
      box.appendChild(label);
    }
    const fx = document.createElement("div");
    fx.className = "hero-formula";
    renderKatex(fx, `x(t) = ${astToLatexT(astList(best.x_ast)[k])}`, true);
    box.appendChild(fx);
    const fy = document.createElement("div");
    fy.className = "hero-formula";
    renderKatex(fy, `y(t) = ${astToLatexT(astList(best.y_ast)[k])}`, true);
    box.appendChild(fy);
  });
}

/* Curve-mode top list: each entry is one pair of formulas PER stroke. */
function renderCurveTop(top) {
  const list = $("curveRaceList");
  list.innerHTML = "";
  top.forEach((item, i) => {
    const li = document.createElement("li");
    li.className = "race-item" + (i === 0 ? " leader" : "");
    const place = document.createElement("span");
    place.className = "race-place num";
    place.textContent = String(i + 1);
    li.appendChild(place);
    const xs = astList(item.x_ast);
    const ys = astList(item.y_ast);
    xs.forEach((xa, k) => {
      const f = document.createElement("span");
      f.className = "race-formula curve-race-formula";
      renderKatex(f, `x(t)=${astToLatexT(xa)}`, false);
      li.appendChild(f);
      const f2 = document.createElement("span");
      f2.className = "race-formula curve-race-formula";
      renderKatex(f2, `y(t)=${astToLatexT(ys[k])}`, false);
      li.appendChild(f2);
    });
    const meta = document.createElement("span");
    meta.className = "race-meta num";
    meta.textContent = i18n.t("error {0} · size {1}", fmtPercent(item.error), item.size);
    li.appendChild(meta);
    list.appendChild(li);
  });
}

/* Curve race log: same feed style as Search, curve-specific events. */
function pushCurveLogEntry(parts) {
  parts.gen = state.lastGenNum;
  state.curveLog.unshift(parts);
  if (state.curveLog.length > 7) state.curveLog.pop();
  renderCurveLog();
}

function renderCurveLog() {
  const list = $("curveRaceLog");
  if (!list) return;
  list.innerHTML = "";
  state.curveLog.forEach((e) => {
    const li = document.createElement("li");
    li.className = "race-log-item" + (e.cls ? " " + e.cls : "");
    const g = document.createElement("span");
    g.className = "race-log-gen num";
    g.textContent = e.gen === null || e.gen === undefined
      ? "—" : "gen. " + (e.gen + 1);
    li.appendChild(g);
    if (e.pre) {
      const pre = document.createElement("span");
      pre.textContent = e.pre + " ";
      li.appendChild(pre);
    }
    if (e.latex) {
      const f = document.createElement("span");
      f.className = "race-log-formula";
      renderKatex(f, e.latex, false);
      li.appendChild(f);
    }
    if (e.text) {
      const t = document.createElement("span");
      t.textContent = e.latex ? " " + e.text : e.text;
      li.appendChild(t);
    }
    list.appendChild(li);
  });
}

function updateCurveRaceLog(msg) {
  state.lastGenNum = msg.gen;
  if (msg.shock && !state.wasCurveShocking) {
    const top = state.curveLog[0];
    if (top && top.cls === "log-shock") {
      top.gen = msg.gen;
      renderCurveLog();
    } else {
      pushCurveLogEntry({
        pre: "shock: fresh pairs mixed in",
        cls: "log-shock",
      });
    }
  }
  state.wasCurveShocking = msg.shock;
  if (msg.solved_now) {
    pushCurveLogEntry({ pre: "threshold met (error < 2%)", cls: "log-solved" });
  } else if (msg.new_best) {
    // throttle like Search: a 100 gen/s run must not blur the feed
    if (!state.lastCurveLogGen || msg.gen - state.lastCurveLogGen >= 10) {
      const multi = msg.best.x_text.length > 1;
      pushCurveLogEntry({
        pre: multi ? "new best pairs:" : "new best pair:",
        latex: `x(t)=${astToLatexT(astList(msg.best.x_ast)[0])},\\ y(t)=${astToLatexT(astList(msg.best.y_ast)[0])}`,
        text: i18n.t("error {0}", fmtPercent(msg.best.error))
          + (multi ? i18n.t(" · {0} strokes", msg.best.x_text.length) : ""),
      });
      state.lastCurveLogGen = msg.gen;
    }
  }
}

/* Per-t check table, the curve analog of the formula check table.
   Single stroke: shows x(t), y(t) and the miss; several strokes: the
   miss averaged over the strokes (the formulas sit in the hero). */
const CURVE_CHECK_TS = [0, 0.2, 0.4, 0.6, 0.8, 1];

function renderCurveCheck(best) {
  const head = $("curveTermsHead");
  head.innerHTML = "";
  addCell(head, "th", "t");
  for (const t of CURVE_CHECK_TS) addCell(head, "th", fmtNum(t));

  const idx = (t) => Math.round(t * (CURVE_SAMPLES_JS - 1));
  const curves = best.curves || [];
  const single = curves.length === 1;
  const curve = single ? curves[0] : null;
  const errCells = [];
  const missAt = (t) => {
    if (!curves.length) return null;
    let sum = 0;
    for (let k = 0; k < curves.length; k++) {
      const stroke = state.curveStrokes[k] || [];
      const cx = curves[k].xs[idx(t)], cy = curves[k].ys[idx(t)];
      let d = Infinity;
      for (const p of stroke) d = Math.min(d, Math.hypot(p.x - cx, p.y - cy));
      sum += Math.min(1, d);
    }
    return sum / curves.length;
  };

  if (single) {
    buildTermsRow($("curveTermsX"), "x(t)",
      CURVE_CHECK_TS.map((t) => curve.xs ? fmtNum(curve.xs[idx(t)]) : "—"));
    buildTermsRow($("curveTermsY"), "y(t)",
      curve.ys ? CURVE_CHECK_TS.map((t) => fmtNum(curve.ys[idx(t)]))
               : CURVE_CHECK_TS.map(() => "—"));
  } else {
    $("curveTermsX").innerHTML = "";
    $("curveTermsY").innerHTML = "";
  }
  const errs = CURVE_CHECK_TS.map((t) => {
    const d = missAt(t);
    return d === null ? "—" : fmtPercent(d);
  });
  buildTermsRow($("curveTermsErr"), "miss", errs, errCells);
  errs.forEach((e, i) => {
    if (parseFloat(e.replace("%", "")) < 2) errCells[i].classList.add("ok");
  });
}

function scheduleCurveUpdate() {
  cancelCurvePlayback();
  if (state.curveRafPending) return;
  state.curveRafPending = true;
  requestAnimationFrame(() => {
    state.curveRafPending = false;
    const b = state.curveBuf;
    if (b) {
      if (state.charts.curveError) {
        state.charts.curveError.setData([b.gen, b.bestE, b.meanE]);
      }
      if (state.charts.curveFitness) {
        state.charts.curveFitness.setData([b.gen, b.bestF, b.meanF]);
      }
      if (state.charts.curveDiversity) {
        state.charts.curveDiversity.setData([b.gen, b.div]);
      }
      if (state.charts.curveSizes) {
        state.charts.curveSizes.setData([b.sizesX, b.sizesY]);
      }
    }
    drawCurveCanvas();
  });
}

/* a live generation redraws the canvas — cancel a running playback so
   the two rAF loops do not fight over the same frame */
function cancelCurvePlayback() {
  if (state.curvePlayRaf) {
    cancelAnimationFrame(state.curvePlayRaf);
    state.curvePlayRaf = null;
  }
}

function applyCurveSolved(msg) {
  state.lastRunSolved = true;
  setStatus("finished");
  $("curveChip").textContent = i18n.t("solved");
  $("curveChip").className = "chip solved";
  if (state.settings) {
    if (state.settings.notify_sound && state.settings.notify_volume > 0) {
      playSolvedSound(state.settings.notify_volume);
    }
    if (state.settings.notify_flash) startTitleFlash();
  }
  const note = $("curveNote");
  note.classList.add("show");
  note.textContent = i18n.t("Threshold beaten in {0} generations", msg.generations)
    + (msg.time_s !== null && msg.time_s !== undefined
        ? ` · ${fmtTime(msg.time_s)}` : "");
  refreshAll();
}

async function startCurveRun(event) {
  if (event) event.preventDefault();
  stopTitleFlash();
  if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
  if (!totalCurvePoints()) {
    alertErr("draw a shape");
    return;
  }
  if (state.ops.size === 0) {
    alertErr("no operation is selected");
    return;
  }
  const seedRaw = $("curveSeedInput").value.trim();
  let seed = null;
  if (seedRaw !== "") {
    seed = Number(seedRaw);
    if (!Number.isInteger(seed) || seed < 0) {
      alertErr("the seed must be a non-negative integer");
      return;
    }
  }
  // round to 4 decimals: keeps the payload small, invisible on the canvas
  const strokes = state.curveStrokes.map((stroke) => stroke.map((p) => [
    Math.round(p.x * 10000) / 10000,
    Math.round(p.y * 10000) / 10000,
  ]));
  const body = {
    mode: "curve",
    strokes,
    preset: state.curvePreset,
    speed: Number($("speedRange").value),
    ops: [...state.ops],
    seed,
  };
  const btn = $("curveStartBtn");
  const originalText = btn.innerHTML;
  btn.disabled = true;
  btn.textContent = i18n.t("Starting…");
  try {
    const status = await api("/api/run", body);
    resetCurveState();
    state.lastRunSolved = false;
    state.runStartedAt = performance.now();
    $("curveNote").classList.remove("show");
    $("curveNote").innerHTML = "";
    applyStatusMsg(status);
    scheduleCurveUpdate();
  } catch (e) {
    alertErr(i18n.t("failed to start: ") + e.message);
  } finally {
    btn.innerHTML = originalText;
    syncCurveUI();
  }
}

async function loadCurves() {
  // the async body fills the DOM after awaits — translate after it settles
  setTimeout(() => { if (window.i18n && i18n.lang !== "en") i18n.apply(); }, 500);
  const data = await api("/api/curves");
  const rows = $("curveRows");
  rows.innerHTML = "";
  if (!data.curves.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.className = "list-empty";
    td.textContent = "Gallery is empty: finished runs will appear here.";
    tr.appendChild(td);
    rows.appendChild(tr);
    return;
  }
  for (const c of data.curves) {
    const tr = document.createElement("tr");
    tr.className = "clickable-row";
    tr.title = i18n.t("Put this drawing back on the canvas");
    tr.addEventListener("click", () => {
      const strokes = entryStrokes(c);
      if (!strokes.length || !strokes[0].length) return;
      state.curveStrokes = strokes.map((stroke) =>
        stroke.map(([x, y]) => ({ x, y })));
      state.curveBestCurves = null;
      state.curveSolved = false;
      state.curvePreset = c.preset || null;
      drawCurveCanvas();
      syncCurveUI();
    });

    addCell(tr, "td", fmtWhen(c.ts));

    const f = document.createElement("td");
    const xs = astList(c.x_ast);
    const ys = astList(c.y_ast);
    xs.forEach((xa, k) => {
      const wrap = document.createElement("span");
      wrap.className = "curve-pair";
      renderKatex(wrap, `x(t) = ${astToLatexT(xa)},\\ y(t) = ${astToLatexT(ys[k])}`, false);
      f.appendChild(wrap);
    });
    tr.appendChild(f);

    const err = document.createElement("td");
    err.className = "num " + (c.solved ? "solved-text" : "amber-text");
    err.textContent = fmtPercent(c.error);
    tr.appendChild(err);

    addCell(tr, "td", String(c.generations), "num");
    addCell(tr, "td", fmtTime(c.time_s), "num");
    rows.appendChild(tr);
  }
}

/* ---------- websocket ---------- */

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;
  ws.onopen = () => setConn(true);
  ws.onclose = () => { setConn(false); setTimeout(connectWS, 2000); };
  ws.onerror = () => setConn(false);
  ws.onmessage = (event) => {
    let msg;
    try { msg = JSON.parse(event.data); } catch { return; }
    switch (msg.type) {
      case "status": applyStatusMsg(msg); break;
      case "gen": applyGen(msg); break;
      case "solved": applySolved(msg); break;
      case "finished":
        if (msg.autopilot_next) {
          refreshAll();
        } else {
          setStatus("finished");
          setChip(msg.reason === "autopilot_done" ? "autopilot finished" : "stopped", "finished");
          if (msg.is_autopilot === false) {
            state.isAutopilot = false;
            const btn = $("autopilotBtn");
            btn.classList.add("btn-positive");
            btn.classList.remove("btn-danger");
            btn.textContent = i18n.t("Autopilot");
          }
          refreshAll();
        }
        break;
    }
  };
}

function setConn(ok) {
  state.wsOk = ok;
  $("connDot").className = "dot " + (ok ? "on" : "off");
  $("connText").textContent = ok ? "server connected" : "no connection to the server";
}

/* ---------- REST ---------- */

async function api(path, body) {
  const opts = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    let detail = "";
    try { detail = (await resp.json()).detail || ""; } catch { /* keep empty */ }
    throw new Error(detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

let PUZZLE_CACHE = [];

/* Localized puzzle name: the library data is English canonical
   (engine/sequences.json) with an optional name_ru pair. */
function pzName(p) {
  if (!p) return "";
  return (i18n.lang === "ru" && p.name_ru) ? p.name_ru : p.name;
}

async function loadPuzzles() {
  // the async body fills the DOM after awaits — translate after it settles
  setTimeout(() => { if (window.i18n && i18n.lang !== "en") i18n.apply(); }, 500);
  const data = await api("/api/puzzles");
  PUZZLE_CACHE = data.puzzles;
  const select = $("puzzleSelect");
  const menu = $("puzzleSelectMenu");
  const keep = select.value;
  
  select.innerHTML = "";
  menu.innerHTML = "";
  
  // Re-create the search input and filters because menu.innerHTML = "" wiped them.
  const searchWrap = document.createElement("div");
  searchWrap.className = "puzzle-search-wrap";
  searchWrap.innerHTML = `
    <input type="search" id="puzzleSearch" class="input puzzle-search" placeholder="Search sequences…">
    <div class="puzzle-filters">
      <select id="filterStatus" class="input puzzle-filter-select">
        <option value="all">${i18n.t("All")}</option>
        <option value="unsolved">${i18n.t("Unsolved")}</option>
        <option value="solved">${i18n.t("Solved")}</option>
      </select>
      <select id="filterDiff" class="input puzzle-filter-select">
        <option value="all" style="color: var(--text)">${i18n.t("Any")}</option>
        <option value="easy" style="color: var(--positive)">${i18n.t("Easy")}</option>
        <option value="medium" style="color: var(--amber)">${i18n.t("Medium")}</option>
        <option value="hard" style="color: var(--negative)">${i18n.t("Hard")}</option>
      </select>
    </div>
  `;
  menu.appendChild(searchWrap);

  const own = document.createElement("option");
  own.value = "";
  own.textContent = i18n.t("— custom sequence —");
  select.appendChild(own);
  
  const tiers = { 1: "T1 · linear", 2: "T2 · powers", 3: "T3 · mixed",
                  4: "T4 · new operations", 5: "T5 · very hard" };
                  
  for (const tier of [1, 2, 3, 4, 5]) {
    const group = document.createElement("optgroup");
    group.label = tiers[tier];
    select.appendChild(group);
    
    const menuGroup = document.createElement("div");
    menuGroup.className = "modern-select-group";
    menuGroup.textContent = tiers[tier];
    menu.appendChild(menuGroup);
    
    for (const p of PUZZLE_CACHE.filter((x) => x.tier === tier)) {
      // 1. Hidden native option
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = `${pzName(p)} · ${p.terms.slice(0, 4).join(" ")}…`;
      group.appendChild(opt);
      
      // 2. Custom visual item
      const item = document.createElement("div");
      item.className = "modern-select-item";
      item.dataset.id = p.id;
      // "unseen" means never solved. Anything else means at least one solve.
      item.dataset.solved = (p.difficulty && p.difficulty !== "unseen") ? "true" : "false";
      item.dataset.diff = p.difficulty || "unseen";
      if (p.id === keep) item.classList.add("active");
      
      const info = document.createElement("div");
      const name = document.createElement("div");
      name.className = "modern-select-item-name";
      const dot = document.createElement("span");
      dot.className = `diff-dot ${p.difficulty || "unseen"}`;
      name.appendChild(dot);
      name.appendChild(document.createTextNode(pzName(p)));
      
      const desc = document.createElement("div");
      desc.className = "modern-select-item-desc";
      const seq = document.createElement("span");
      seq.className = "modern-select-item-seq";
      seq.textContent = p.terms.slice(0, 4).join(" ") + (p.terms.length > 4 ? "…" : "");
      desc.appendChild(seq);
      
      if (p.reference) {
        const ref = document.createElement("span");
        ref.className = "modern-select-item-ref";
        renderKatex(ref, refToLatex(p.reference), false);
        desc.appendChild(ref);
      }
      
      info.appendChild(name);
      info.appendChild(desc);
      item.appendChild(info);
      
      item.addEventListener("click", () => {
        select.value = p.id;
        select.dispatchEvent(new Event("change"));
        togglePuzzleMenu(false);
      });
      
      menu.appendChild(item);
    }
  }
  
  // Setup search and filter listeners
  $("puzzleSearch").value = "";
  $("puzzleSearch").addEventListener("input", applyPuzzleFilters);
  $("filterStatus").addEventListener("change", applyPuzzleFilters);
  $("filterDiff").addEventListener("change", applyPuzzleFilters);
  $("puzzleSearch").addEventListener("click", (e) => e.stopPropagation());
  $("filterStatus").addEventListener("click", (e) => e.stopPropagation());
  $("filterDiff").addEventListener("click", (e) => e.stopPropagation());

  select.value = keep;
  // First page load only: apply the remembered puzzle from a previous session.
  if (state.pendingPuzzleRestore && !select.value) {
    const p = PUZZLE_CACHE.find((x) => x.id === state.pendingPuzzleRestore);
    if (p) {
      select.value = state.pendingPuzzleRestore;
      $("seqInput").value = p.terms.join(" ");
      markPuzzle(p);
    }
    state.pendingPuzzleRestore = "";
  }
  updatePuzzleLabel();
  updateOpsHint();
}

function applyPuzzleFilters() {
  const q = $("puzzleSearch").value.toLowerCase().trim();
  const status = $("filterStatus").value;
  const diff = $("filterDiff").value;

  document.querySelectorAll(".modern-select-item").forEach(item => {
    const text = item.textContent.toLowerCase();
    const isSolved = item.dataset.solved === "true";
    const itemDiff = item.dataset.diff;

    let match = text.includes(q);
    if (status === "solved" && !isSolved) match = false;
    if (status === "unsolved" && isSolved) match = false;
    if (diff !== "all" && itemDiff !== diff) match = false;

    item.style.display = match ? "" : "none";
  });

  // Hide tier groups that have no visible items
  document.querySelectorAll(".modern-select-group").forEach(group => {
    let hasVisible = false;
    let next = group.nextElementSibling;
    while (next && !next.classList.contains("modern-select-group")) {
      if (next.style.display !== "none") hasVisible = true;
      next = next.nextElementSibling;
    }
    group.style.display = hasVisible ? "" : "none";
  });
}

function togglePuzzleMenu(show) {
  const menu = $("puzzleSelectMenu");
  const btn = $("puzzleSelectTrigger");
  const isHidden = menu.hidden;
  const shouldShow = show !== undefined ? show : isHidden;
  menu.hidden = !shouldShow;
  btn.setAttribute("aria-expanded", shouldShow);
  if (shouldShow) {
    const active = menu.querySelector(".modern-select-item.active");
    if (active) active.scrollIntoView({ block: "nearest" });
  }
}

function updatePuzzleLabel() {
  const val = $("puzzleSelect").value;
  const label = $("puzzleSelectLabel");
  if (!val) {
    label.textContent = i18n.t("— custom sequence —");
    $("puzzleSelectMenu").querySelectorAll(".modern-select-item").forEach(el => el.classList.remove("active"));
  } else {
    const p = PUZZLE_CACHE.find(x => x.id === val);
    label.textContent = p ? pzName(p) : i18n.t("— custom sequence —");
    $("puzzleSelectMenu").querySelectorAll(".modern-select-item").forEach(el => {
      el.classList.toggle("active", el.dataset.id === val);
    });
  }
}

const TOKEN_LABELS = {
  "+": "+", "-": "−", "*": "×", "/": "÷", "^": "xʸ", mod: "%",
  sqrt: "√", cbrt: "∛", fact: "!", abs: "|x|", floor: "⌊x⌋",
  sin: "sin", cos: "cos", tan: "tan", ln: "ln", pi: "π", e: "e",
  if: "if", "<": "<", ">": ">",
};

/* T4 puzzles are only solvable with the right grammar features on; when
   one is selected, say which chips to flip instead of letting the run
   fail silently. */
function updateOpsHint() {
  const el = $("opsReq");
  const p = PUZZLE_CACHE.find((x) => x.id === $("puzzleSelect").value);
  const missing = p && p.requires ? p.requires.filter((t) => !state.ops.has(t)) : [];
  if (!missing.length) {
    el.hidden = true;
    return;
  }
  el.textContent = i18n.t("\"{0}\" requires: ", pzName(p)) +
    missing.map((t) => TOKEN_LABELS[t] || t).join(", ") +
    i18n.t(" (currently disabled)");
  el.hidden = false;
}

function renderReferenceNote(text, size) {
  const note = $("solvedNote");
  note.classList.add("show");
  note.innerHTML = "";
  const label = document.createElement("span");
  label.textContent = i18n.t("Reference: ");
  note.appendChild(label);
  const ref = document.createElement("span");
  renderKatex(ref, refToLatex(text), false);
  note.appendChild(ref);
  
  if (size !== undefined && size !== null) {
    const meta = document.createElement("span");
    meta.style.marginLeft = "12px";
    meta.style.color = "var(--muted)";
    meta.style.fontSize = "12px";
    meta.title = "The exact formula size (counting sign complexity and number digits).";
    meta.textContent = i18n.t("(size {0})", size);
    note.appendChild(meta);
  }
}

function clearReferenceNote() {
  const note = $("solvedNote");
  note.classList.remove("show");
  note.innerHTML = "";
}

function markPuzzle(p) {
  state.puzzle = p;
  state.puzzleTerms = p ? p.terms.map(Number) : null;
  if (!p) { clearReferenceNote(); return; }
  if (p.reference) renderReferenceNote(p.reference, p.ref_size);
}

async function loadLibrary() {
  // the async body fills the DOM after awaits — translate after it settles
  setTimeout(() => { if (window.i18n && i18n.lang !== "en") i18n.apply(); }, 500);
  const data = await api("/api/history");
  const puzzles = await api("/api/puzzles");
  const history = data.history;
  const rows = $("libraryRows");
  rows.innerHTML = "";

  // Group best attempts by sequence.
  // Priority: solved > unsolved. Among solved: shortest formula wins.
  // Among unsolved: lowest error (or smallest size) wins.
  const best = {};
  for (const h of history) {
    const key = h.sequence.join(",");
    if (!best[key]) {
      best[key] = h;
    } else {
      const current = best[key];
      if (h.solved && !current.solved) {
        best[key] = h;
      } else if (h.solved && current.solved) {
        if (h.size < current.size) best[key] = h;
      } else if (!h.solved && !current.solved) {
        if (h.error < current.error) best[key] = h;
      }
    }
  }

  for (const p of puzzles.puzzles) {
    const tr = document.createElement("tr");
    tr.className = "clickable-row";
    tr.title = "Run a search for this sequence";
    tr.addEventListener("click", () => {
      $("puzzleSelect").value = p.id;
      $("seqInput").value = p.terms.join(" ");
      $("refInput").value = p.reference || "";
      // pre-fill the seed from the best attempt, so replaying this run
      // is one click away; old entries have no seed and stay empty
      $("seedInput").value = h && h.seed !== null && h.seed !== undefined
        ? String(h.seed) : "";
      markPuzzle(p);
      updateOpsHint();
      $("ptabSearch").click();
    });

    const key = p.terms.join(",");
    const h = best[key];
    tr.dataset.seq = key;
    tr.dataset.id = p.id;
    tr.dataset.terms = JSON.stringify(p.terms);
    tr.dataset.ref = p.reference || "";
    tr.dataset.diff = p.difficulty || "unseen";
    tr.dataset.solved = h && h.solved ? "true" : "false";

    // Store plain text for the search filter (KaTeX renders math as spans,
    // making textContent useless for matching "2^n").
    tr.dataset.search = ((p.name + " " + (p.name_ru || "") + " " + p.terms.join(" ") + " " + (p.reference || "") + " " + (h && h.formula ? h.formula : "")).toLowerCase());

    const title = document.createElement("td");
    const dot = document.createElement("span");
    dot.className = `diff-dot ${p.difficulty || "unseen"}`;
    dot.title = p.difficulty === "easy" ? "Easy" : p.difficulty === "medium" ? "Medium" : p.difficulty === "hard" ? "Hard" : "Not solved yet";
    title.appendChild(dot);
    title.innerHTML += `<b>${pzName(p)}</b><br><span style="color:var(--faint);font-size:12px; margin-left: 14px;">${p.terms.join(" ")}</span>`;
    tr.appendChild(title);

    const ref = document.createElement("td");
    renderKatex(ref, refToLatex(p.reference || "—"), false);
    tr.appendChild(ref);

    const result = document.createElement("td");
    if (h) {
      if (h.formula_ast) renderKatex(result, astToLatex(h.formula_ast), false);
      else result.textContent = h.formula;
      result.className = h.solved ? "solved-text" : "amber-text";
      
      const refSize = p.ref_size || 0;
      if (h.solved && h.size < refSize) {
        tr.classList.add("better-than-ref");
        result.title = i18n.t("A shorter formula than the reference was found! (Size {0} vs {1})", h.size, refSize);
        
        const badge = document.createElement("span");
        badge.className = "ref-badge";
        badge.textContent = i18n.t("shorter");
        badge.title = i18n.t("Shorter than the reference ({0} vs {1})", h.size, refSize);
        result.appendChild(badge);
      }
      
      bindPlotHover(tr, () => [h.sequence, h.values]);
    } else {
      result.textContent = "—";
      result.style.color = "var(--faint)";
    }
    tr.appendChild(result);

    const err = document.createElement("td");
    err.className = "num";
    if (h) {
      if (h.solved) err.textContent = "0%";
      else err.textContent = fmtPercent(h.error);
      err.className = h.solved ? "num solved-text" : "num amber-text";
    } else {
      err.textContent = "—";
    }
    tr.appendChild(err);

    const gens = document.createElement("td");
    gens.className = "num";
    if (h) {
      let gensText = String(h.generations);
      // Find previous attempt for delta
      const attempts = history.filter(x => x.sequence.join(",") === key && x.ts !== h.ts);
      if (attempts.length) {
        const prev = attempts[0]; // history is newest first
        const delta = h.generations - prev.generations;
        if (delta < 0) gensText += `<span class="delta-neg">(${delta})</span>`;
        else if (delta > 0) gensText += `<span class="delta-pos">(+${delta})</span>`;
      }
      gens.innerHTML = gensText;
    } else {
      gens.textContent = "—";
    }
    tr.appendChild(gens);

    rows.appendChild(tr);
  }
}

async function loadMySeqs() {
  // the async body fills the DOM after awaits — translate after it settles
  setTimeout(() => { if (window.i18n && i18n.lang !== "en") i18n.apply(); }, 500);
  const data = await api("/api/history");
  const rows = $("mySeqRows");
  rows.innerHTML = "";

  // Keep only custom sequences, group by sequence.
  // Priority: solved > unsolved. Among solved: shortest formula wins.
  const custom = data.history.filter(h => !h.puzzle);
  const best = {};
  for (const h of custom) {
    const key = h.sequence.join(",");
    if (!best[key]) {
      best[key] = h;
    } else {
      const current = best[key];
      if (h.solved && !current.solved) {
        best[key] = h;
      } else if (h.solved && current.solved) {
        if (h.size < current.size) best[key] = h;
      } else if (!h.solved && !current.solved) {
        if (h.error < current.error) best[key] = h;
      }
    }
  }

  const unique = Object.values(best);
  if (!unique.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.className = "list-empty";
    td.textContent = "No custom sequences yet.";
    tr.appendChild(td);
    rows.appendChild(tr);
    return;
  }

  // Sort by timestamp descending
  unique.sort((a, b) => b.ts.localeCompare(a.ts));

  for (const h of unique) {
    const tr = document.createElement("tr");
    tr.className = "clickable-row";
    tr.dataset.seq = h.sequence.join(",");
    tr.dataset.terms = JSON.stringify(h.sequence);
    tr.dataset.ref = h.reference || "";
    tr.dataset.solved = h.solved ? "true" : "false";
    tr.title = "Run the search again";
    tr.addEventListener("click", () => {
      $("seqInput").value = h.sequence.join(" ");
      $("refInput").value = h.reference || "";
      $("seedInput").value = h.seed !== null && h.seed !== undefined
        ? String(h.seed) : "";
      $("puzzleSelect").value = "";
      markPuzzle(null);
      updateOpsHint();
      $("ptabSearch").click();
    });

    tr.dataset.search = (h.sequence.map(fmtNum).join(" ") + " " + (h.reference || "") + " " + h.formula + " " + (h.solved ? "0%" : fmtPercent(h.error))).toLowerCase();

    const seq = document.createElement("td");
    seq.className = "seq";
    seq.textContent = h.sequence.map(fmtNum).join(" ");
    tr.appendChild(seq);

    const ref = document.createElement("td");
    if (h.reference) renderKatex(ref, refToLatex(h.reference), false);
    else { ref.textContent = "—"; ref.style.color = "var(--faint)"; }
    tr.appendChild(ref);

    const result = document.createElement("td");
    if (h.formula_ast) renderKatex(result, astToLatex(h.formula_ast), false);
    else result.textContent = h.formula;
    result.className = h.solved ? "solved-text" : "amber-text";
    tr.appendChild(result);

    const err = document.createElement("td");
    err.className = "num";
    if (h.solved) err.textContent = "0%";
    else err.textContent = fmtPercent(h.error);
    err.className = h.solved ? "num solved-text" : "num amber-text";
    tr.appendChild(err);

    const gens = document.createElement("td");
    gens.className = "num";
    let gensText = String(h.generations);
    // Find previous attempt for delta
    const attempts = custom.filter(x => x.sequence.join(",") === h.sequence.join(",") && x.ts !== h.ts);
    if (attempts.length) {
      const prev = attempts[0]; // history is newest first
      const delta = h.generations - prev.generations;
      if (delta < 0) gensText += `<span class="delta-neg">(${delta})</span>`;
      else if (delta > 0) gensText += `<span class="delta-pos">(+${delta})</span>`;
    }
    gens.innerHTML = gensText;
    tr.appendChild(gens);

    bindPlotHover(tr, () => [h.sequence, h.values]);
    rows.appendChild(tr);
  }
}

async function loadCuriosities() {
  // the async body fills the DOM after awaits — translate after it settles
  setTimeout(() => { if (window.i18n && i18n.lang !== "en") i18n.apply(); }, 500);
  const data = await api("/api/curiosities");
  const rows = $("curioRows");
  rows.innerHTML = "";
  if (!data.curiosities.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.className = "list-empty";
    td.textContent = "No curiosities yet: almost-correct monsters will appear after the first runs.";
    tr.appendChild(td);
    rows.appendChild(tr);
    return;
  }
  for (const c of data.curiosities) {
    const tr = document.createElement("tr");
    tr.className = "clickable-row";
    tr.title = "Run a search for this sequence";
    tr.addEventListener("click", () => {
      $("seqInput").value = c.sequence.join(" ");
      $("refInput").value = "";
      $("puzzleSelect").value = "";
      markPuzzle(null);
      updateOpsHint();
      $("ptabSearch").click();
    });
    tr.dataset.search = (c.sequence.map(fmtNum).join(" ") + " " + c.formula_text + " " + c.size + " " + c.error + " " + (c.gen + 1)).toLowerCase();

    addCell(tr, "td", c.sequence.map(fmtNum).join(" "), "seq");
    const f = addCell(tr, "td", "");
    if (c.formula_ast) renderKatex(f, astToLatex(c.formula_ast), false);
    else f.textContent = c.formula_text;
    addCell(tr, "td", String(c.size), "num");
    addCell(tr, "td", fmtPercent(c.error), "num");
    addCell(tr, "td", String(c.gen + 1), "num");
    bindPlotHover(tr, () => [c.sequence, c.values]);
    rows.appendChild(tr);
  }
}

/* Shape presets table: best curve run per preset (like the library,
   but for the curve tab). Click = load the drawing onto the canvas. */
async function loadShapes() {
  // the async body fills the DOM after awaits — translate after it settles
  setTimeout(() => { if (window.i18n && i18n.lang !== "en") i18n.apply(); }, 500);
  const data = await api("/api/curves");
  const rows = $("shapeRows");
  rows.innerHTML = "";
  const best = {};
  for (const c of data.curves) {
    if (!c.preset) continue;
    const current = best[c.preset];
    if (!current || (c.error !== null && c.error < current.error)) {
      best[c.preset] = c;
    }
  }
  for (const preset of CURVE_PRESETS) {
    const c = best[preset.name];
    const tr = document.createElement("tr");
    tr.className = "clickable-row";
    tr.title = "Open this shape on the Curves tab";
    tr.addEventListener("click", () => {
      const src = c ? entryStrokes(c) : [preset.pts];
      state.curveStrokes = src.map((stroke) =>
        stroke.map((p) => Array.isArray(p) ? { x: p[0], y: p[1] } : { ...p }));
      state.curveBestCurves = null;
      state.curveSolved = false;
      state.curvePreset = preset.name;
      drawCurveCanvas();
      syncCurveUI();
      $("ptabCurves").click();
    });

    const name = document.createElement("td");
    name.textContent = i18n.t(preset.name);
    tr.appendChild(name);

    const result = document.createElement("td");
    if (c) {
      const xs = astList(c.x_ast);
      const ys = astList(c.y_ast);
      renderKatex(result,
        `x(t)=${astToLatexT(xs[0])},\\ y(t)=${astToLatexT(ys[0])}`
        + (xs.length > 1 ? `\\ \\ldots (${xs.length})` : ""), false);
      result.className = c.solved ? "solved-text" : "amber-text";
    } else {
      result.textContent = "—";
      result.style.color = "var(--faint)";
    }
    tr.appendChild(result);

    const err = document.createElement("td");
    if (c) {
      err.textContent = fmtPercent(c.error);
      err.className = "num " + (c.solved ? "solved-text" : "amber-text");
    } else {
      err.textContent = "—";
      err.className = "num";
    }
    tr.appendChild(err);

    addCell(tr, "td", c ? String(c.generations) : "—", "num");
    addCell(tr, "td", c ? fmtTime(c.time_s) : "—", "num");
    rows.appendChild(tr);
  }
}

/* Curve curiosities: finished unsolved runs whose best pair came visibly
   close ("polygon instead of a circle"). Click = load the drawing onto
   the curve canvas, like the shapes table. */
async function loadCurveCuriosities() {
  // the async body fills the DOM after awaits — translate after it settles
  setTimeout(() => { if (window.i18n && i18n.lang !== "en") i18n.apply(); }, 500);
  const data = await api("/api/curve_curiosities");
  if (data.threshold) {
    $("curveCurioCaption").textContent =
      i18n.t("Near-hit gallery: error between ")
      + fmtPercent(data.threshold.min) + " and " + fmtPercent(data.threshold.max);
  }
  const rows = $("curveCurioRows");
  rows.innerHTML = "";
  if (!data.curiosities.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.className = "list-empty";
    td.textContent = "No curve curiosities yet: they will appear after unsolved runs where the curve nearly hit.";
    tr.appendChild(td);
    rows.appendChild(tr);
    return;
  }
  for (const c of data.curiosities) {
    const tr = document.createElement("tr");
    tr.className = "clickable-row";
    tr.title = "Open this drawing on the Curves tab";
    tr.addEventListener("click", () => {
      state.curveStrokes = entryStrokes(c).map((stroke) =>
        stroke.map((p) => Array.isArray(p) ? { x: p[0], y: p[1] } : { ...p }));
      state.curveBestCurves = null;
      state.curveSolved = false;
      state.curvePreset = c.preset || null;
      drawCurveCanvas();
      syncCurveUI();
      $("ptabCurves").click();
    });
    tr.dataset.search =
      ((c.preset || "custom drawing") + " " + (c.x_text || []).join(" ")
       + " " + (c.y_text || []).join(" ") + " " + c.error).toLowerCase();

    addCell(tr, "td", c.preset || "custom drawing");
    const f = addCell(tr, "td", "");
    const xs = astList(c.x_ast);
    const ys = astList(c.y_ast);
    if (xs.length && ys.length) {
      renderKatex(f,
        `x(t)=${astToLatexT(xs[0])},\\ y(t)=${astToLatexT(ys[0])}`
        + (xs.length > 1 ? `\\ \\ldots (${xs.length})` : ""), false);
    } else {
      f.textContent = (c.x_text || []).join(" | ") + " / " +
                      (c.y_text || []).join(" | ");
    }
    addCell(tr, "td", fmtPercent(c.error), "num");
    addCell(tr, "td", String(c.generations), "num");
    rows.appendChild(tr);
  }
}

function highlightActiveRow() {
  document.querySelectorAll("tbody tr").forEach(tr => tr.classList.remove("active-autopilot"));
  if (state.status === "running" && state.mode === "sequence" && state.puzzleTerms) {
    const seqKey = state.puzzleTerms.join(",");
    document.querySelectorAll(`tbody tr[data-seq="${seqKey}"]`).forEach(row => {
      row.classList.add("active-autopilot");
    });
  }
}

function applyTableFilter() {
  const val = $("tableFilter").value;
  const isLibrary = !$("libraryTable").hidden;
  const table = isLibrary ? $("libraryTable") : $("mySeqTable");
  const rows = table.querySelectorAll("tbody tr");
  if (!rows.length || rows[0].children.length === 1) return;

  rows.forEach(tr => {
    // Never hide the row that is currently being processed by the autopilot,
    // so the user can see it even if it just got solved and doesn't match
    // the "Unsolved" filter anymore.
    if (tr.classList.contains("active-autopilot")) {
      tr.style.display = "";
      return;
    }

    const isSolved = tr.dataset.solved === "true";
    const diff = tr.dataset.diff || "unseen";
    let show = true;

    if (val === "unsolved" && isSolved) show = false;
    if (val === "solved" && !isSolved) show = false;
    if (val === "easy" && diff !== "easy") show = false;
    if (val === "medium" && diff !== "medium") show = false;
    if (val === "hard" && diff !== "hard") show = false;

    tr.style.display = show ? "" : "none";
  });
}

function refreshAll() {
  Promise.all([
    loadPuzzles().catch(() => {}),
    loadLibrary().catch(() => {}),
    loadMySeqs().catch(() => {}),
    loadCuriosities().catch(() => {}),
    loadStats().catch(() => {}),
    loadCurves().catch(() => {}),
    loadShapes().catch(() => {}),
    loadCurveCuriosities().catch(() => {})
  ]).then(() => {
    applyTableFilter();
    highlightActiveRow();
  });
}

/* Library summary: solved fraction, per-tier aggregates, records. */
const TIER_LABELS = { 1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "T5" };

async function loadStats() {
  // the async body fills the DOM after awaits — translate after it settles
  setTimeout(() => { if (window.i18n && i18n.lang !== "en") i18n.apply(); }, 500);
  const data = await api("/api/stats");
  const headline = $("statsHeadline");
  const tiers = $("statsTiers");
  headline.textContent = i18n.t("Solved {0} of {1} sequences", data.solved, data.total);
  tiers.innerHTML = "";
  for (const t of data.by_tier) {
    const cell = document.createElement("div");
    cell.className = "stats-tier" + (t.solved === t.total ? " done" : "");
    cell.dataset.tip = t.solved === t.total
      ? "Every sequence of this tier is solved"
      : i18n.t("Solved {0} of {1} sequences at this level", t.solved, t.total);
    const name = document.createElement("span");
    name.className = "stats-tier-name";
    name.textContent = TIER_LABELS[t.tier] || `T${t.tier}`;
    const frac = document.createElement("span");
    frac.className = "num";
    frac.textContent = `${t.solved}/${t.total}`;
    const gens = document.createElement("span");
    gens.className = "stats-tier-gens";
    gens.textContent = t.avg_gens !== null ? i18n.t("⌀ {0} gens", t.avg_gens) : "—";
    cell.append(name, frac, gens);
    tiers.appendChild(cell);
  }
}

/* ---------- controls ---------- */

function parseTerms(text) {
  const parts = text.split(/[\s,;]+/).filter(Boolean);
  if (!parts.length) throw new Error("the sequence is empty");
  const terms = parts.map(Number);
  if (terms.some((t) => !isFinite(t))) throw new Error("not all values are numbers");
  if (terms.length < 2 || terms.length > 32) throw new Error("a sequence must be 2 to 32 members long");
  return terms;
}

async function startRun(event) {
  event.preventDefault();
  $("solvedNote").classList.remove("show");
  state.lastRunSolved = false;
  stopTitleFlash();
  if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
  
  if (state.ops.size === 0) {
    alertErr("no operation is selected");
    return;
  }
  
  // Empty seed = fresh random run on the server; any integer = deterministic.
  // The field is digits-only (see the input listener in bindControls), so
  // this guard only catches a hand-edited DOM value.
  const seedRaw = $("seedInput").value.trim();
  let seed = null;
  if (seedRaw !== "") {
    seed = Number(seedRaw);
    if (!Number.isInteger(seed) || seed < 0) {
      alertErr("the seed must be a non-negative integer");
      return;
    }
  }
  const puzzleId = $("puzzleSelect").value || null;
  let body;
  if (puzzleId) {
    body = { puzzle_id: puzzleId, speed: Number($("speedRange").value),
             ops: [...state.ops], seed };
  } else {
    let terms;
    try { terms = parseTerms($("seqInput").value); }
    catch (e) { alertErr(e.message); return; }
    const customRef = $("refInput").value.trim();
    body = { terms, speed: Number($("speedRange").value), ops: [...state.ops], reference: customRef || null, seed };
  }

  // visual loading state to prevent double-clicks while API is in flight
  const startBtn = $("startBtn");
  const originalText = startBtn.innerHTML;
  startBtn.disabled = true;
  startBtn.innerHTML = '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg> Запуск...';

  try {
    const status = await api("/api/run", body);
    resetBuffers();
    state.runStartedAt = performance.now();
    state.puzzle = status.puzzle;
    state.puzzleTerms = status.sequence;
    if (status.puzzle) {
      const p = PUZZLE_CACHE.find((x) => x.id === status.puzzle.id);
      if (p) markPuzzle(p);
    } else {
      const customRef = $("refInput").value.trim();
      // We don't have the pre-calculated size for custom refs,
      // so we just omit the size parameter.
      if (customRef) renderReferenceNote(customRef, null);
      else clearReferenceNote();
    }
    $("termsEmpty").hidden = true;
    applyStatusMsg(status);
    scheduleChartUpdate();
  } catch (e) {
    alertErr(i18n.t("failed to start: ") + e.message);
  } finally {
    startBtn.innerHTML = originalText;
    startBtn.disabled = state.status === "running" || state.status === "paused";
  }
}

function alertErr(message) {
  setChip("error", "error");
  // inline, dismissible — a hover-only chip title made failures invisible
  for (const id of ["errorNote", "curveErrorNote"]) {
    const el = $(id);
    if (!el) continue;
    el.textContent = message + "  (click to hide)";
    el.hidden = false;
    el.onclick = () => { el.hidden = true; };
  }
  console.error(message);
}

async function control(action) {
  try {
    await api("/api/control", { action });
    if (action === "stop") {
      state.isAutopilot = false;
      const btn = $("autopilotBtn");
      btn.classList.add("btn-positive");
      btn.classList.remove("btn-danger");
      btn.textContent = i18n.t("Autopilot");
    }
    // The server saves the best unsolved formula on pause. Refresh the
    // tables immediately so the user sees the progress without waiting.
    if (action === "pause") {
      refreshAll();
    }
  }
  catch (e) { alertErr(e.message); }
}

async function setSpeed(gps) {
  try { await api("/api/speed", { gps }); }
  catch (e) { /* speed is best-effort */ }
}

async function copyBestFormula() {
  const text = state.lastFormulaText;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {
    // Fallback for non-secure contexts (clipboard API unavailable)
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
  }
  const badge = $("copyBadge");
  badge.hidden = false;
  clearTimeout(copyBestFormula.timer);
  copyBestFormula.timer = setTimeout(() => { badge.hidden = true; }, 1200);
}

const RESULT_META = {
  library: "Every library sequence: the reference and the best result found (green — solved, orange — an approximation so far)",
  shapes: "Preset shapes from the Curves tab: the best result for each (click to put the drawing back)",
  myseqs: "Your own sequences, the best formulas found, and their errors",
  curio: "Almost-correct but bulky monsters — traces of evolution's wandering",
};

/* ---------- settings (Settings tab) ---------- */

const SETTING_FIELDS = [
  ["setPopulation", "population", "number"],
  ["setMaxGenerations", "max_generations", "number"],
  ["setSpeedMax", "speed_max", "number"],
  ["setParsimony", "parsimony", "number"],
  ["setMaxDepth", "max_depth", "number"],
  ["setTournament", "tournament", "number"],
  ["setPCrossover", "p_crossover", "number"],
  ["setPMutate", "p_mutate", "number"],
  ["setElitism", "elitism", "number"],
  ["setShrinkGenerations", "shrink_generations", "number"],
  ["setShrinkParsimony", "shrink_parsimony", "number"],
  ["setAutopilotGens", "autopilot_gens", "number"],
  ["setNotifyFlash", "notify_flash", "checkbox"],
  ["setNotifySound", "notify_sound", "checkbox"],
  ["setNotifyVolume", "notify_volume", "range"],
];

async function loadSettings() {
  const data = await api("/api/settings");
  state.settings = data.settings;
  state.settingDefaults = data.defaults;
  fillSettingsForm();
  applySpeedMax();
}

function fillSettingsForm() {
  if (!state.settings) return;
  for (const [id, key, type] of SETTING_FIELDS) {
    if (type === "checkbox") {
      $(id).checked = Boolean(state.settings[key]);
    } else {
      $(id).value = String(state.settings[key]);
    }
  }
  // update the numeric label next to the volume slider
  $("setNotifyVolumeVal").textContent = state.settings.notify_volume;
}

/* the speed slider's upper bound is a setting; keep the slider honest */
function applySpeedMax() {
  if (!state.settings) return;
  const slider = $("speedRange");
  const max = Math.round(state.settings.speed_max);
  slider.max = String(max);
  if (Number(slider.value) > max) {
    slider.value = String(max);
    $("speedValue").textContent = i18n.t("{0} generations/s", max);
    setSpeed(max);
  }
}

async function saveSettings(event) {
  if (event) event.preventDefault();
  const body = {};
  for (const [id, key, type] of SETTING_FIELDS) {
    if (type === "checkbox") {
      body[key] = $(id).checked;
    } else {
      body[key] = Number($(id).value);
    }
  }
  try {
    const data = await api("/api/settings", body);
    state.settings = data.settings;
    state.settingDefaults = data.defaults;
    fillSettingsForm();
    applySpeedMax();
    flashSettingsMeta("saved — applies to the next runs");
  } catch (e) {
    alertErr(i18n.t("failed to save settings: ") + e.message);
  }
}

async function resetSettings() {
  if (!state.settingDefaults) return;
  try {
    const data = await api("/api/settings", state.settingDefaults);
    state.settings = data.settings;
    state.settingDefaults = data.defaults;
    fillSettingsForm();
    applySpeedMax();
    flashSettingsMeta("settings reset to their defaults");
  } catch (e) {
    alertErr(i18n.t("failed to reset settings: ") + e.message);
  }
}

function flashSettingsMeta(text) {
  const el = $("settingsMeta");
  el.textContent = text;
  el.classList.add("ok");
  clearTimeout(flashSettingsMeta.timer);
  flashSettingsMeta.timer = setTimeout(() => {
    el.textContent = "apply to the next runs, stored locally";
    el.classList.remove("ok");
  }, 2500);
}

function initFromUrlAndStorage() {
  // Restore ops from localStorage, stripping any stray spaces that
  // might have been saved by an older version of the URL parser.
  const savedOps = localStorage.getItem("seq2formula_ops");
  if (savedOps) {
    try {
      const parsed = JSON.parse(savedOps);
      if (Array.isArray(parsed)) {
        const clean = parsed.filter(o => typeof o === 'string' && o.trim() !== '');
        if (clean.length > 0) state.ops = new Set(clean);
      }
    } catch (e) { /* ignore corrupted localStorage */ }
  }

  // Restore the last sequence / reference typed by the user, so a reload
  // doesn't wipe them. URL params (below) still win: an explicit ?seq=
  // link is a stronger intent than the last keystroke.
  const savedSeq = localStorage.getItem("seq2formula_seq");
  if (savedSeq) $("seqInput").value = savedSeq;
  const savedSeed = localStorage.getItem("seq2formula_seed");
  // stored values from before this fix may contain non-digits — sanitize
  if (savedSeed !== null) $("seedInput").value = savedSeed.replace(/\D/g, "");
  const savedRef = localStorage.getItem("seq2formula_ref");
  if (savedRef) $("refInput").value = savedRef;

  // Remembered puzzle selection is applied later: the select is empty
  // until loadPuzzles() fills it. See loadPuzzles() for the restore.
  state.pendingPuzzleRestore = localStorage.getItem("seq2formula_puzzle") || "";

  // Override with URL params if present
  const urlParams = new URLSearchParams(window.location.search);
  const seq = urlParams.get('seq');
  if (seq) {
    $("seqInput").value = seq;
    state.pendingPuzzleRestore = ""; // explicit sequence wins over remembered puzzle
  }
  
  const urlOps = urlParams.get('ops');
  if (urlOps) {
    state.ops.clear();
    if (urlOps === 'all') {
      Object.keys(TOKEN_LABELS).forEach(k => state.ops.add(k));
    } else {
      // URLSearchParams decodes '+' as ' ', so we fix it back.
      const fixedOps = urlOps.replace(/ /g, '+');
      fixedOps.split(',').forEach(o => {
        const clean = o.trim();
        if (clean) state.ops.add(clean);
      });
    }
  }

  // Sync chips UI
  document.querySelectorAll(".op-chip").forEach(chip => {
    chip.setAttribute("aria-pressed", String(state.ops.has(chip.dataset.op)));
  });
}

/* ---------- notifications (sound + title flash) ---------- */

let audioCtx = null;
function playSolvedSound(volume) {
  if (!window.AudioContext && !window.webkitAudioContext) return;
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") audioCtx.resume();

  // The user liked the "rising" chime, so we bring it back but with a very
  // soft envelope and a non-linear gain curve. 1.0 is ~0.3 max gain (quiet),
  // 0.1 is ~0.003 (barely audible whisper).
  const maxGain = Math.pow(volume, 2) * 0.3;
  if (maxGain < 0.0001) return;

  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = "sine";
  osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5
  osc.frequency.exponentialRampToValueAtTime(1318.51, audioCtx.currentTime + 0.15); // E6

  gain.gain.setValueAtTime(0, audioCtx.currentTime);
  gain.gain.linearRampToValueAtTime(maxGain, audioCtx.currentTime + 0.05);
  gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.6);

  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + 0.6);
}

let titleFlashTimer = null;
function startTitleFlash() {
  if (titleFlashTimer) return;
  const original = document.title;
  let flash = false;
  titleFlashTimer = setInterval(() => {
    document.title = flash ? original : "Solved! — seq2formula";
    flash = !flash;
  }, 1000);
}

function stopTitleFlash() {
  if (titleFlashTimer) {
    clearInterval(titleFlashTimer);
    titleFlashTimer = null;
    document.title = "seq2formula — formula search";
  }
}

/* Keyboard shortcuts for the live controls. Skipped when focus is in an
   editable element or a button, so typing and Space-activating a button
   both keep working. */
function bindHotkeys() {
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const tag = (document.activeElement || {}).tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT"
        || tag === "BUTTON") return;

    if (e.code === "Space") {
      if (state.status !== "running" && state.status !== "paused") return;
      e.preventDefault();
      control(state.status === "paused" ? "resume" : "pause");
    } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      if (state.status === "finished") return;
      e.preventDefault();
      control("step");
    } else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      const slider = $("speedRange");
      const step = e.key === "ArrowUp" ? 5 : -5;
      const next = Math.max(Number(slider.min),
                            Math.min(Number(slider.max), Number(slider.value) + step));
      slider.value = String(next);
      $("speedValue").textContent = next + " generations/s";
      setSpeed(next);
      e.preventDefault();
    } else if (e.key === "Escape") {
      if (state.status !== "running" && state.status !== "paused") return;
      e.preventDefault();
      control("stop");
    }
  });
}

function bindControls() {
  initFromUrlAndStorage();
  
  $("runForm").addEventListener("submit", startRun);
  $("seqInput").addEventListener("input", () => {
    $("puzzleSelect").value = "";
    updatePuzzleLabel();
    localStorage.setItem("seq2formula_puzzle", "");
    markPuzzle(null);
    localStorage.setItem("seq2formula_seq", $("seqInput").value);
  });
  
  $("puzzleSelectTrigger").addEventListener("click", (e) => {
    e.stopPropagation();
    togglePuzzleMenu();
  });
  
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".field-puzzle")) {
      togglePuzzleMenu(false);
    }
  });
  $("refInput").addEventListener("input", () => {
    localStorage.setItem("seq2formula_ref", $("refInput").value);
  });
  // digits-only: strip anything else on every input (typing, paste, IME).
  // This is what stops letters from becoming an empty (="random") value.
  $("seedInput").addEventListener("input", (e) => {
    const cleaned = e.target.value.replace(/\D/g, "");
    if (cleaned !== e.target.value) e.target.value = cleaned;
    localStorage.setItem("seq2formula_seed", cleaned);
  });
  $("puzzleSelect").addEventListener("change", (e) => {
    const p = PUZZLE_CACHE.find((x) => x.id === e.target.value);
    // Switching tasks discards the current run: a finished run resets
    // to a fresh view immediately (the server's stop is a no-op there),
    // a running/paused one is stopped as if the user pressed Stop.
    if (state.status !== "idle") {
      control("stop");
      if (state.status === "finished") setStatus("idle");
    }
    if (p) {
      $("seqInput").value = p.terms.join(" ");
      $("refInput").value = p.reference || "";
      localStorage.setItem("seq2formula_seq", $("seqInput").value);
      localStorage.setItem("seq2formula_ref", $("refInput").value);
    }
    localStorage.setItem("seq2formula_puzzle", e.target.value);
    markPuzzle(p || null);
    updatePuzzleLabel();
    updateOpsHint();
  });
  $("pauseBtn").addEventListener("click", () => {
    control(state.status === "paused" ? "resume" : "pause");
  });
  $("stepBtn").addEventListener("click", () => control("step"));
  $("stopBtn").addEventListener("click", () => control("stop"));
  $("continueBtn").addEventListener("click", () => control("continue"));
  // curve mode controls: start/stop/clear + digits-only seed
  $("curveStartBtn").addEventListener("click", startCurveRun);
  $("curveStopBtn").addEventListener("click", () => control("stop"));
  $("curveClearBtn").addEventListener("click", () => {
    state.curveStrokes = [];
    state.curveStroke = [];
    state.curveBestCurves = null;
    state.curvePreset = null;
    drawCurveCanvas();
    syncCurveUI();
  });
  $("curveUndoBtn").addEventListener("click", () => {
    if (state.status === "running" || state.status === "paused") return;
    state.curveStrokes.pop();
    state.curvePreset = null;
    drawCurveCanvas();
    syncCurveUI();
  });
  $("curvePlayBtn").addEventListener("click", () => {
    if (!state.curveBestCurves || !state.curveBestCurves.length) return;
    playCurveFit();
  });
  $("evoBtn").addEventListener("click", playEvolution);
  $("curveExportBtn").addEventListener("click", async () => {
    try {
      const data = await api("/api/drawings");
      if (!data.drawings.length) {
        alertErr("no saved drawings yet");
        return;
      }
      const blob = new Blob([JSON.stringify(data.drawings, null, 2)],
        { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `seq2formula_drawings_${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      alertErr(i18n.t("failed to download drawings: ") + err.message);
    }
  });
  $("curveSeedInput").addEventListener("input", (e) => {
    const cleaned = e.target.value.replace(/\D/g, "");
    if (cleaned !== e.target.value) e.target.value = cleaned;
  });
  // one-stroke shape presets: click = the drawing appears on the canvas
  const presetRow = $("presetRow");
  for (const preset of CURVE_PRESETS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn preset-btn";
    btn.textContent = i18n.t(preset.name);
    btn.addEventListener("click", () => {
      if (state.status === "running" || state.status === "paused") return;
      state.curveStrokes = [preset.pts.map((p) => ({ ...p }))];
      state.curveBestCurves = null;
      state.curvePreset = preset.name;
      drawCurveCanvas();
      syncCurveUI();
    });
    presetRow.appendChild(btn);
  }
  $("bestFormula").addEventListener("click", copyBestFormula);
  $("bestFormula").addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      copyBestFormula();
    }
  });
  let speedTimer = null;
  $("speedRange").addEventListener("input", (e) => {
    $("speedValue").textContent = e.target.value + " generations/s";
    clearTimeout(speedTimer);
    speedTimer = setTimeout(() => setSpeed(Number(e.target.value)), 200);
  });

  const tabs = [["tabLibrary", "libraryTable", "library"],
                ["tabShapes", "shapeTable", "shapes"],
                ["tabMySeqs", "mySeqTable", "myseqs"],
                ["tabCurio", "curioTable", "curio"]];
  for (const [tabId, bodyId, metaKey] of tabs) {
    $(tabId).addEventListener("click", () => {
      for (const [t, b] of tabs) {
        $(t).setAttribute("aria-selected", String(t === tabId));
        $(b).hidden = t !== tabId;
      }
      // the curiosities tab hosts two tables: sequence + curve curiosities
      $("curveCurioTable").hidden = tabId !== "tabCurio";
      $("resultsMeta").textContent = RESULT_META[metaKey];
      // reset the text search, but keep the dropdown filter
      $("tableSearch").value = "";
      // re-apply filter to the newly visible table
      applyTableFilter();
      // Ensure text search is cleared visually
      const activeTable = tabs.find(([t]) => t === tabId)[1];
      document.querySelectorAll(`#${activeTable} tbody tr`).forEach(tr => {
        if (tr.dataset.search) tr.style.display = ""; 
      });
    });
  }
  $("resultsMeta").textContent = RESULT_META.library;

  // top-level page tabs: Search / Curves / Results / Settings
  const panes = [["ptabSearch", "paneSearch"], ["ptabCurves", "paneCurves"],
                 ["ptabResults", "paneResults"],
                 ["ptabSettings", "paneSettings"]];
  for (const [tabId, paneId] of panes) {
    $(tabId).addEventListener("click", () => {
      for (const [t, p] of panes) {
        $(t).setAttribute("aria-selected", String(t === tabId));
        $(p).hidden = t !== tabId;
      }
      if (tabId === "ptabResults") {
        $("resultsDot").hidden = true;
      }
      if (tabId === "ptabResults" || tabId === "ptabCurves") {
        // delay helps uPlot get the right container width
        setTimeout(resizeCharts, 50);
      }
    });
  }

  // settings form: save / reset, slider upper bound follows speed_max
  $("settingsForm").addEventListener("submit", saveSettings);
  $("settingsReset").addEventListener("click", resetSettings);
  
  // Auto-save notification toggles so the user doesn't have to click "Save"
  $("setNotifyFlash").addEventListener("change", saveSettings);
  $("setNotifySound").addEventListener("change", saveSettings);
  // Volume slider: update label live, save on release
  $("setNotifyVolume").addEventListener("input", (e) => {
    $("setNotifyVolumeVal").textContent = e.target.value;
  });
  $("setNotifyVolume").addEventListener("change", saveSettings);

  // operator picker: toggle chips, block a run without any operator
  const syncOpsDisabled = () => {
    const busy = state.status === "running" || state.status === "paused";
    document.querySelectorAll(".op-chip").forEach((chip) => {
      chip.disabled = busy;
    });
  };
  const syncOpsUI = () => {
    document.querySelectorAll(".op-chip").forEach(chip => {
      chip.setAttribute("aria-pressed", String(state.ops.has(chip.dataset.op)));
    });
    localStorage.setItem("seq2formula_ops", JSON.stringify([...state.ops]));
    updateOpsHint();
  };

  document.querySelectorAll(".clickable-group").forEach(label => {
    label.addEventListener("click", () => {
      if (state.status === "running" || state.status === "paused") return;
      const group = label.closest(".ops-group");
      const chips = Array.from(group.querySelectorAll(".op-chip"));
      const allEnabled = chips.every(c => state.ops.has(c.dataset.op));
      
      chips.forEach(chip => {
        const op = chip.dataset.op;
        if (allEnabled) {
          // If we are disabling all, ensure at least one op remains globally
          if (state.ops.size === 1 && state.ops.has(op)) return;
          state.ops.delete(op);
        } else {
          state.ops.add(op);
        }
      });
      syncOpsUI();
    });
  });

  document.querySelectorAll(".op-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const op = chip.dataset.op;
      if (state.ops.has(op) && state.ops.size === 1) {
        $("opsWarn").hidden = false;
        setTimeout(() => { $("opsWarn").hidden = true; }, 2500);
        return;
      }
      const on = !state.ops.has(op);
      state.ops[on ? "add" : "delete"](op);
      syncOpsUI();
    });
  });
  syncOpsDisabled();

  // table search: hide rows that don't match the query
  $("tableSearch").addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase().trim();
    document.querySelectorAll(".tab-body table:not([hidden]) tbody tr").forEach(tr => {
      // Prefer the raw text we stored in data-search (KaTeX breaks textContent)
      const text = tr.dataset.search || tr.textContent.toLowerCase();
      tr.style.display = text.includes(q) ? "" : "none";
    });
  });

  $("tableFilter").addEventListener("change", applyTableFilter);

  // autopilot trigger
  $("autopilotBtn").addEventListener("click", async () => {
    if (state.isAutopilot) {
      // Act as a stop button
      control("stop");
      return;
    }

    // The shapes table has no autopilot rows; with it open the sequence
    // tables are both hidden and would silently drive the queue.
    if (!$("shapeTable").hidden) return;

    const isLibrary = !$("libraryTable").hidden;
    const table = isLibrary ? $("libraryTable") : $("mySeqTable");
    
    // Grab only the VISIBLE rows, in their current DOM order
    const rows = Array.from(table.querySelectorAll("tbody tr")).filter(tr => tr.style.display !== "none");
    
    if (!rows.length || rows[0].children.length === 1) return;

    const items = rows.map(tr => {
      const isLib = !!tr.dataset.id;
      return {
        puzzle_id: isLib ? tr.dataset.id : null,
        terms: tr.dataset.terms ? JSON.parse(tr.dataset.terms) : null,
        reference: tr.dataset.ref || null
      };
    });

    const gens = state.settings.autopilot_gens;
    
    try {
      await api("/api/autopilot", { items, gens, ops: [...state.ops] });
    } catch (err) {
      alertErr(i18n.t("failed to start autopilot: ") + err.message);
    }
  });

  // export history as JSON
  $("exportBtn").addEventListener("click", async () => {
    try {
      const data = await api("/api/history");
      const blob = new Blob([JSON.stringify(data.history, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `seq2formula_history_${new Date().toISOString().slice(0,10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      alertErr(i18n.t("failed to download history: ") + err.message);
    }
  });

  // column sorting: reorders the DOM rows based on text or numeric value
  document.querySelectorAll(".tab-body table th").forEach((th) => {
    th.addEventListener("click", () => {
      const table = th.closest("table");
      const tbody = table.querySelector("tbody");
      const rows = Array.from(tbody.querySelectorAll("tr"));
      if (rows.length <= 1 || rows[0].children.length === 1) return;

      // cellIndex is the column position relative to the current table,
      // unlike the cumulative index across all tables
      const index = th.cellIndex;
      const isAsc = th.dataset.sort !== "asc";
      
      // reset other headers in this table
      table.querySelectorAll("th").forEach(other => {
        if (other !== th) delete other.dataset.sort;
      });
      th.dataset.sort = isAsc ? "asc" : "desc";

      rows.sort((a, b) => {
        const aCell = a.children[index];
        const bCell = b.children[index];
        if (!aCell || !bCell) return 0;

        // Sort by raw dataset text if available, otherwise fallback to textContent
        let aText, bText;
        if (th.classList.contains("num")) {
          aText = aCell.textContent.trim();
          bText = bCell.textContent.trim();
        } else {
          // For non-numeric columns, localeCompare handles Cyrillic fine
          aText = aCell.textContent.trim();
          bText = bCell.textContent.trim();
        }

        const aNum = parseFloat(aText.replace(/[^\d.-]/g, ''));
        const bNum = parseFloat(bText.replace(/[^\d.-]/g, ''));
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
          return isAsc ? aNum - bNum : bNum - aNum;
        }
        return isAsc ? aText.localeCompare(bText) : bText.localeCompare(aText);
      });

      // Hide tbody to skip layout recalculation during DOM manipulation.
      // Re-rendering heavy KaTeX DOM trees on every row move freezes the UI.
      tbody.style.display = "none";
      tbody.replaceChildren(...rows);
      tbody.style.display = "";
    });
  });

  bindHotkeys();

  // re-fit charts when any chart container changes size (viewport
  // changes and pane switches); window resize alone misses both
  const ro = new ResizeObserver(resizeCharts);
  for (const id of ["chartFitness", "chartError", "chartDiversity", "chartSizes",
                    "heroChart", "chartCurveError", "chartCurveFitness",
                    "chartCurveDiversity", "chartCurveSizes"]) {
    ro.observe($(id));
  }

  window.addEventListener("resize", resizeCharts);
}

/* ---------- boot ---------- */

resetBuffers();
makeCharts();
initCurveCanvas();
wireSavedDrawings();
loadSavedDrawings().catch(() => {});
resetCurveState();
bindControls();
refreshAll();
loadSettings().catch(() => {});
connectWS();
setStatus("idle");
i18n.ready.then(() => {
  if (i18n.lang !== "en") i18n.apply();
  // settings -> interface language select (one entry per locale folder)
  const sel = document.getElementById("setLang");
  if (sel) {
    const names = { en: "English", ru: "Русский" };
    for (const code of i18n.locales) {
      const opt = document.createElement("option");
      opt.value = code;
      opt.textContent = names[code] || code;
      sel.appendChild(opt);
    }
    sel.value = i18n.lang;
    sel.onchange = () => i18n.set(sel.value);
  }
});
