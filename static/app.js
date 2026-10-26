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
  settings: null,          // engine knobs from /api/settings (Настройки tab)
  settingDefaults: null,
  lastTopUpdate: 0,        // throttle the top-10 list (100 gen/s is unreadable)
  lastLogGen: null,        // throttle the event feed
  wasShocking: false,      // dedupe shock entries in the race log
  pendingPuzzleRestore: "", // remembered puzzle id, applied once the library loads
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
  if (sec < 60) return fmtNum(sec) + " с";
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${m} м ${String(s).padStart(2, "0")} с`;
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
    width: label === "лучшая" ? 2 : 1.5,
    dash,
    points: { show: false },
  };
}

function makeCharts() {
  // Axis titles make the charts self-explanatory: without them "1 -> 400"
  // on the size histogram is meaningless to a reader.
  const axisX = { ...axisOpts(), label: "поколение", values: (u, vals) => vals.map((v) => String(Math.round(v))) };
  const common = { ...baseOpts(170, [
      { label: "поколение" },
      lineSeries("лучшая", CHART_COLORS.accent),
      lineSeries("средняя", CHART_COLORS.muted, [4, 4]),
    ], true), scales: { x: { time: false } }, axes: [axisX, { ...axisOpts(3), label: "fitness" }] };

  state.charts.fitness = new uPlot({ ...common }, [[], [], []], $("chartFitness"));

  state.charts.error = new uPlot(
    { ...common,
      series: [
        { label: "поколение" },
        { ...lineSeries("лучшая", CHART_COLORS.amber), width: 2 },
        lineSeries("средняя", CHART_COLORS.muted, [4, 4]),
      ],
      axes: [axisX, { ...axisOpts(3), label: "ошибка", values: (u, vals) => vals.map((v) => fmtNum(v)) }] },
    [[], [], []], $("chartError"));

  state.charts.diversity = new uPlot(
    { ...baseOpts(140, [
        { label: "поколение" },
        { ...lineSeries("разнообразие", CHART_COLORS.teal), width: 1.5 },
      ], false),
      scales: { x: { time: false } },
      axes: [axisX, { ...axisOpts(2), label: "доля уникальных" }] },
    [[], []], $("chartDiversity"));

  const barsPath = window.uPlot && uPlot.paths && uPlot.paths.bars
    ? uPlot.paths.bars({ align: 1, size: [1, 100] }) : undefined;
  state.charts.sizes = new uPlot(
    { ...baseOpts(140, [
        { label: "размер" },
        { label: "деревьев", stroke: CHART_COLORS.accent, width: 2,
          points: { show: false }, paths: barsPath, fill: "rgba(139,147,248,0.18)" },
      ], false),
      scales: { x: { time: false } },
      axes: [{ ...axisX, label: "размер формулы" }, { ...axisOpts(), label: "кол-во формул" }] },
    [[], []], $("chartSizes"));

  // hero mini-chart: the formula (line) against the target sequence (dots)
  const dots = (size) => window.uPlot && uPlot.paths && uPlot.paths.points
    ? uPlot.paths.points({ size }) : undefined;
  state.charts.hero = new uPlot(
    { ...baseOpts(110, [
        { label: "n" },
        { label: "цель", stroke: CHART_COLORS.faint, width: 0, paths: dots(5),
          points: { show: true } },
        { label: "формула", stroke: CHART_COLORS.accent, width: 2,
          points: { show: true, size: 3 } },
      ], false),
      scales: { x: { time: false } },
      axes: [{ show: false }, { show: false }] },
    [[], [], []], $("heroChart"));

  resizeCharts();
}

function resizeCharts() {
  const boxes = [
    ["fitness", "chartFitness"], ["error", "chartError"],
    ["diversity", "chartDiversity"], ["sizes", "chartSizes"],
    ["hero", "heroChart"],
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
  chip.textContent = text;
  chip.className = "chip" + (cls ? " " + cls : "");
  if (cls !== "error") chip.title = "";
}

function setStatus(status) {
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
  $("pauseLabel").textContent = status === "paused" ? "Продолжить" : "Пауза";
  if (status === "running") setChip("поиск идёт", "run");
  else if (status === "paused") setChip("пауза", "paused");
  else if (status === "finished") setChip("завершено", "finished");
  else setChip("нет прогона");
}

function applyGen(msg) {
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
    ? fmtNum((state.genTimestamps.length - 1) / span) + " пок/с" : "—";
  const elapsed = state.runStartedAt ? fmtTime((now - state.runStartedAt) / 1000) : "—";
  $("kpiSpeedTime").textContent = `${gps} · ${elapsed}`;

  // hero formula: re-render only when the text changes
  if (msg.best.text !== state.lastFormulaText) {
    state.lastFormulaText = msg.best.text;
    renderKatex($("bestFormula"), astToLatex(msg.best.ast), true);
  }
  const sub = [];
  sub.push(`fitness ${fmtNum(msg.best.fitness)}`);
  if (msg.best.exact) sub.push('<span class="ok">точное совпадение по значениям</span>');
  $("bestSub").innerHTML = sub.join(" · ");
  $("heroLabel").textContent = msg.solved && !msg.solved_now
    ? "Лучшая формула (усадка — ищем короче)" : "Лучшая формула";

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

/* ---------- hover plot previews (history / hall / curiosities) ---------- */

function showPlotPopover(evt, sequence, values) {
  if (!Array.isArray(sequence) || !sequence.length) return;
  const pop = $("plotPopover");
  pop.hidden = false;
  if (!state.charts.popover) {
    state.charts.popover = new uPlot(
      { ...baseOpts(150, [
          { label: "n" },
          { label: "цель", stroke: CHART_COLORS.faint, width: 0,
            paths: window.uPlot && uPlot.paths && uPlot.paths.points
              ? uPlot.paths.points({ size: 4 }) : undefined,
            points: { show: true } },
          { label: "формула", stroke: CHART_COLORS.accent, width: 2,
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
    meta.textContent = `fit ${fmtNum(item.fitness)} · размер ${item.size}`;
    meta.title = "Fitness = 1 - ошибка - штраф_за_размер. Ближе к 1.0 — лучше.";
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
      ? "—" : "пок. " + (e.gen + 1);
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
  if (msg.shock && !state.wasShocking) {
    pushLogEntry({
      pre: "встряска: подмешаны свежие формулы",
      cls: "log-shock",
    });
  }
  state.wasShocking = msg.shock;
  if (msg.solved_now) {
    pushLogEntry({ pre: "точное совпадение найдено", cls: "log-solved" });
  } else if (msg.solved && msg.best.exact && msg.best.size < state.lastExactSize) {
    pushLogEntry({
      pre: "усадка: формула короче —", text: `размер ${msg.best.size}`,
      cls: "log-shrink",
    });
  }
  state.lastExactSize = msg.best.exact ? msg.best.size : state.lastExactSize;
  if (msg.new_best && !msg.best.exact) {
    // Log the new best, but throttle it to once per 10 generations so
    // a 100 gen/s run doesn't turn the feed into an unreadable blur.
    if (!state.lastLogGen || msg.gen - state.lastLogGen >= 10) {
      pushLogEntry({
        pre: "новая лучшая:", latex: astToLatex(msg.best.ast),
        text: `ошибка ${fmtPercent(msg.best.error)}`,
      });
      state.lastLogGen = msg.gen;
    }
  } else if (msg.new_best && msg.best.exact && !msg.solved_now) {
    pushLogEntry({ pre: "новая лучшая:", latex: astToLatex(msg.best.ast) });
  }
}

function renderTerms(best) {
  const seq = state.puzzleTerms;
  if (!seq) return;
  const head = $("termsHead");
  head.innerHTML = "";
  addCell(head, "th", "n");
  seq.forEach((_, i) => addCell(head, "th", String(i + 1)));

  buildTermsRow($("termsTarget"), "должно быть", seq.map(fmtNum));
  const got = best.values === null
    ? seq.map(() => "—")
    : best.values.map(fmtNum);
  buildTermsRow($("termsGot"), "получается", got);
  const errRow = [];
  if (best.errors === null) {
    seq.forEach(() => errRow.push("—"));
  } else {
    best.errors.forEach((e) => errRow.push(e < 1e-9 ? "0" : fmtPercent(e)));
  }
  const errCells = [];
  buildTermsRow($("termsErr"), "расхождение", errRow, errCells);
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
  state.lastRunSolved = true;
  setStatus("finished");
  setChip("решено", "solved");
  $("resultsDot").hidden = false;
  pushLogEntry({
    pre: "решено:", latex: msg.formula ? astToLatex(msg.formula) : null,
    text: `${msg.generations} поколений, ${fmtTime(msg.time_s)}`,
    cls: "log-solved",
  });
  if (msg.formula) {
    renderKatex($("bestFormula"), astToLatex(msg.formula), true);
  }
  if (msg.formula_text) state.lastFormulaText = msg.formula_text;
  const parts = [`Решено за ${msg.generations} поколений`];
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
    short.textContent = " короче: ";
    note.appendChild(short);
    const s2 = document.createElement("span");
    renderKatex(s2, "\\text{" + msg.shortest_text + "}", false);
    note.appendChild(s2);
  }
  refreshAll();
}

function applyStatusMsg(msg) {
  setStatus(msg.status);
  if ("solved" in msg) state.lastRunSolved = msg.solved === true;
  if (msg.sequence && msg.sequence.length) {
    state.puzzleTerms = msg.sequence;
  }
  if (msg.puzzle) {
    state.puzzle = msg.puzzle;
    const p = PUZZLE_CACHE.find((x) => x.id === msg.puzzle.id);
    if (p) markPuzzle(p);
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
        setStatus("finished");
        setChip(msg.reason === "max_generations" ? "лимит поколений" : "остановлено", "finished");
        refreshAll();
        break;
    }
  };
}

function setConn(ok) {
  state.wsOk = ok;
  $("connDot").className = "dot " + (ok ? "on" : "off");
  $("connText").textContent = ok ? "сервер на связи" : "нет связи с сервером";
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

async function loadPuzzles() {
  const data = await api("/api/puzzles");
  PUZZLE_CACHE = data.puzzles;
  const select = $("puzzleSelect");
  const keep = select.value;  // refreshAll() rebuilds the list after every
  select.innerHTML = "";      // solve — the selection must survive it
  const own = document.createElement("option");
  own.value = "";
  own.textContent = "— своя последовательность —";
  select.appendChild(own);
  const tiers = { 1: "T1 · линейные", 2: "T2 · степени", 3: "T3 · смешанные",
                  4: "T4 · новые операции", 5: "T5 · очень сложные" };
  for (const tier of [1, 2, 3, 4, 5]) {
    const group = document.createElement("optgroup");
    group.label = tiers[tier];
    for (const p of PUZZLE_CACHE.filter((x) => x.tier === tier)) {
      const opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = `${p.name} · ${p.terms.slice(0, 4).join(" ")}…`;
      group.appendChild(opt);
    }
    select.appendChild(group);
  }
  select.value = keep;
  // First page load only: apply the remembered puzzle from a previous
  // session. Later calls preserve the current choice through `keep`.
  if (state.pendingPuzzleRestore && !select.value) {
    const p = PUZZLE_CACHE.find((x) => x.id === state.pendingPuzzleRestore);
    if (p) {
      select.value = state.pendingPuzzleRestore;
      $("seqInput").value = p.terms.join(" ");
      markPuzzle(p);
    }
    state.pendingPuzzleRestore = "";
  }
  updateOpsHint();
}

const TOKEN_LABELS = {
  "+": "+", "-": "−", "*": "×", "/": "÷", "^": "xʸ", mod: "%",
  sqrt: "√", cbrt: "∛", fact: "!", abs: "|x|",
  sin: "sin", cos: "cos", tan: "tan", ln: "ln", pi: "π", e: "e",
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
  el.textContent = `для загадки «${p.name}» требуются: ` +
    missing.map((t) => TOKEN_LABELS[t] || t).join(", ") +
    " (сейчас отключены)";
  el.hidden = false;
}

function calcWeightedSize(text) {
  // Insert explicit '*' for implicit multiplication (e.g. 2n -> 2 * n)
  const formatted = text.toLowerCase()
    .replace(/√/g, " sqrt ").replace(/∛/g, " cbrt ")
    .replace(/π/g, " pi ").replace(/!/g, " ! ")
    .replace(/(\d)([a-z(])/g, "$1 * $2")
    .replace(/(\))([a-z0-9(])/g, "$1 * $2");

  const tokens = formatted.match(/(sqrt|cbrt|fact|abs|sin|cos|tan|ln|pi|e|\d+(\.\d+)?|n|[-+*/^%()!])/g) || [];
  
  let size = 0;
  for (const t of tokens) {
    if (t === "(" || t === ")") continue;
    if (["sqrt", "cbrt", "fact", "abs", "sin", "cos", "tan", "ln", "!", "%", "mod"].includes(t)) {
      size += 2;
    } else if (["+", "-", "*", "/", "^"].includes(t)) {
      size += 1;
    } else if (["n", "pi", "e"].includes(t)) {
      size += 1;
    } else if (/^\d+(\.\d+)?$/.test(t)) {
      // 1000 -> 4, 10 -> 2, 0.5 -> 1 (primitives stay 1)
      const num = parseFloat(t);
      if (num < 1) size += 1;
      else size += String(Math.floor(Math.abs(num))).length;
    }
  }
  return size;
}

function renderReferenceNote(text) {
  const note = $("solvedNote");
  note.classList.add("show");
  note.innerHTML = "";
  const label = document.createElement("span");
  label.textContent = "Эталон: ";
  note.appendChild(label);
  const ref = document.createElement("span");
  renderKatex(ref, refToLatex(text), false);
  note.appendChild(ref);
  
  const size = calcWeightedSize(text);
  const meta = document.createElement("span");
  meta.style.marginLeft = "12px";
  meta.style.color = "var(--muted)";
  meta.style.fontSize = "12px";
  meta.title = "Точный размер формулы (с учётом сложности знаков и разрядности чисел).";
  meta.textContent = `(размер ${size})`;
  note.appendChild(meta);
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
  if (p.reference) renderReferenceNote(p.reference);
}

async function loadLibrary() {
  const data = await api("/api/history");
  const puzzles = await api("/api/puzzles");
  const history = data.history;
  const rows = $("libraryRows");
  rows.innerHTML = "";

  // Group best attempts by sequence
  const best = {};
  for (const h of history) {
    const key = h.sequence.join(",");
    if (!best[key] || h.solved || (!best[key].solved && h.size < best[key].size)) {
      best[key] = h;
    }
  }

  for (const p of puzzles.puzzles) {
    const tr = document.createElement("tr");
    tr.className = "clickable-row";
    tr.title = "Запустить поиск этой загадки";
    tr.addEventListener("click", () => {
      $("puzzleSelect").value = p.id;
      $("seqInput").value = p.terms.join(" ");
      $("refInput").value = "";
      markPuzzle(p);
      updateOpsHint();
      $("ptabSearch").click();
    });

    const key = p.terms.join(",");
    const h = best[key];

    const title = document.createElement("td");
    title.innerHTML = `<b>${p.name}</b><br><span style="color:var(--faint);font-size:12px">${p.terms.join(" ")}</span>`;
    tr.appendChild(title);

    const ref = document.createElement("td");
    renderKatex(ref, refToLatex(p.reference || "—"), false);
    tr.appendChild(ref);

    const result = document.createElement("td");
    if (h) {
      if (h.formula_ast) renderKatex(result, astToLatex(h.formula_ast), false);
      else result.textContent = h.formula;
      result.className = h.solved ? "solved-text" : "amber-text";
      
      const refSize = calcWeightedSize(p.reference || "");
      if (h.solved && h.size < refSize) {
        tr.classList.add("better-than-ref");
        result.title = `Найдена формула короче эталона! (Размер ${h.size} против ${refSize})`;
      }
      
      bindPlotHover(tr, () => [h.sequence, h.values]);
    } else {
      result.textContent = "—";
      result.style.color = "var(--faint)";
    }
    tr.appendChild(result);

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
  const data = await api("/api/history");
  const rows = $("mySeqRows");
  rows.innerHTML = "";

  // Keep only custom sequences, group by sequence
  const custom = data.history.filter(h => !h.puzzle);
  const best = {};
  for (const h of custom) {
    const key = h.sequence.join(",");
    if (!best[key] || h.solved || (!best[key].solved && h.size < best[key].size)) {
      best[key] = h;
    }
  }

  const unique = Object.values(best);
  if (!unique.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.className = "list-empty";
    td.textContent = "Пользовательские последовательности пока отсутствуют.";
    tr.appendChild(td);
    rows.appendChild(tr);
    return;
  }

  // Sort by timestamp descending
  unique.sort((a, b) => b.ts.localeCompare(a.ts));

  for (const h of unique) {
    const tr = document.createElement("tr");
    tr.className = "clickable-row";
    tr.title = "Запустить поиск заново";
    tr.addEventListener("click", () => {
      $("seqInput").value = h.sequence.join(" ");
      $("refInput").value = h.reference || "";
      $("puzzleSelect").value = "";
      markPuzzle(null);
      updateOpsHint();
      $("ptabSearch").click();
    });

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
  const data = await api("/api/curiosities");
  const rows = $("curioRows");
  rows.innerHTML = "";
  if (!data.curiosities.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.className = "list-empty";
    td.textContent = "Курьёзов пока нет: почти-правильные монстры появятся после первых прогонов.";
    tr.appendChild(td);
    rows.appendChild(tr);
    return;
  }
  for (const c of data.curiosities) {
    const tr = document.createElement("tr");
    tr.className = "clickable-row";
    tr.title = "Запустить поиск этой последовательности";
    tr.addEventListener("click", () => {
      $("seqInput").value = c.sequence.join(" ");
      $("refInput").value = "";
      $("puzzleSelect").value = "";
      markPuzzle(null);
      updateOpsHint();
      $("ptabSearch").click();
    });
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

function refreshAll() {
  loadPuzzles().catch(() => {});
  loadLibrary().catch(() => {});
  loadMySeqs().catch(() => {});
  loadCuriosities().catch(() => {});
  loadStats().catch(() => {});
}

/* Library summary: solved fraction, per-tier aggregates, records. */
const TIER_LABELS = { 1: "T1", 2: "T2", 3: "T3", 4: "T4", 5: "T5" };

async function loadStats() {
  const data = await api("/api/stats");
  const headline = $("statsHeadline");
  const tiers = $("statsTiers");
  const records = $("statsRecords");
  headline.textContent = `Решено ${data.solved} из ${data.total} загадок`;
  tiers.innerHTML = "";
  for (const t of data.by_tier) {
    const cell = document.createElement("div");
    cell.className = "stats-tier" + (t.solved === t.total ? " done" : "");
    cell.dataset.tip = t.solved === t.total
      ? "Все загадки уровня решены"
      : `Решено ${t.solved} из ${t.total} загадок этого уровня`;
    const name = document.createElement("span");
    name.className = "stats-tier-name";
    name.textContent = TIER_LABELS[t.tier] || `T${t.tier}`;
    const frac = document.createElement("span");
    frac.className = "num";
    frac.textContent = `${t.solved}/${t.total}`;
    const gens = document.createElement("span");
    gens.className = "stats-tier-gens";
    gens.textContent = t.avg_gens !== null ? `⌀ ${t.avg_gens} пок.` : "—";
    cell.append(name, frac, gens);
    tiers.appendChild(cell);
  }
  records.innerHTML = "";
  for (const r of data.records) {
    const row = document.createElement("div");
    row.className = "stats-record";
    const who = document.createElement("span");
    who.textContent = r.attempts > 1
      ? `${r.puzzle} · ${r.attempts} попыток`
      : r.puzzle;
    const val = document.createElement("span");
    val.className = "num";
    val.textContent = `${r.generations} пок. · ${fmtTime(r.time_s)}`;
    row.append(who, val);
    records.appendChild(row);
  }
}

/* ---------- controls ---------- */

function parseTerms(text) {
  const parts = text.split(/[\s,;]+/).filter(Boolean);
  if (!parts.length) throw new Error("последовательность не заполнена");
  const terms = parts.map(Number);
  if (terms.some((t) => !isFinite(t))) throw new Error("не все значения являются числами");
  if (terms.length < 2 || terms.length > 32) throw new Error("допустимая длина последовательности — от 2 до 32 членов");
  return terms;
}

async function startRun(event) {
  event.preventDefault();
  $("solvedNote").classList.remove("show");
  state.lastRunSolved = false;
  
  if (state.ops.size === 0) {
    alertErr("не выбрана ни одна операция");
    return;
  }
  
  const puzzleId = $("puzzleSelect").value || null;
  let body;
  if (puzzleId) {
    body = { puzzle_id: puzzleId, speed: Number($("speedRange").value),
             ops: [...state.ops] };
  } else {
    let terms;
    try { terms = parseTerms($("seqInput").value); }
    catch (e) { alertErr(e.message); return; }
    const customRef = $("refInput").value.trim();
    body = { terms, speed: Number($("speedRange").value), ops: [...state.ops], reference: customRef || null };
  }
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
      if (customRef) renderReferenceNote(customRef);
      else clearReferenceNote();
    }
    $("termsEmpty").hidden = true;
    applyStatusMsg(status);
    scheduleChartUpdate();
  } catch (e) {
    alertErr("не удалось запустить: " + e.message);
  }
}

function alertErr(message) {
  setChip("ошибка", "error");
  $("liveChip").title = message;
  console.error(message);
}

async function control(action) {
  try { await api("/api/control", { action }); }
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
  library: "все загадки из библиотеки: эталон и лучший найденный результат (зелёный — решено, оранжевый — пока только приближение)",
  myseqs: "введённые вами последовательности, лучшие найденные формулы и ошибки",
  curio: "почти правильные, но громоздкие монстры — следы блуждания эволюции",
};

/* ---------- settings (Настройки tab) ---------- */

const SETTING_FIELDS = [
  ["setPopulation", "population"],
  ["setMaxGenerations", "max_generations"],
  ["setSpeedMax", "speed_max"],
  ["setParsimony", "parsimony"],
  ["setMaxDepth", "max_depth"],
  ["setTournament", "tournament"],
  ["setPCrossover", "p_crossover"],
  ["setPMutate", "p_mutate"],
  ["setElitism", "elitism"],
  ["setShrinkGenerations", "shrink_generations"],
  ["setShrinkParsimony", "shrink_parsimony"],
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
  for (const [id, key] of SETTING_FIELDS) {
    $(id).value = String(state.settings[key]);
  }
}

/* the speed slider's upper bound is a setting; keep the slider honest */
function applySpeedMax() {
  if (!state.settings) return;
  const slider = $("speedRange");
  const max = Math.round(state.settings.speed_max);
  slider.max = String(max);
  if (Number(slider.value) > max) {
    slider.value = String(max);
    $("speedValue").textContent = `${max} поколений/с`;
    setSpeed(max);
  }
}

async function saveSettings(event) {
  if (event) event.preventDefault();
  const body = {};
  for (const [id, key] of SETTING_FIELDS) body[key] = Number($(id).value);
  try {
    const data = await api("/api/settings", body);
    state.settings = data.settings;
    state.settingDefaults = data.defaults;
    fillSettingsForm();
    applySpeedMax();
    flashSettingsMeta("сохранено — применяется к следующим запускам");
  } catch (e) {
    alertErr("не удалось сохранить настройки: " + e.message);
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
    flashSettingsMeta("настройки сброшены к значениям по умолчанию");
  } catch (e) {
    alertErr("не удалось сбросить настройки: " + e.message);
  }
}

function flashSettingsMeta(text) {
  const el = $("settingsMeta");
  el.textContent = text;
  el.classList.add("ok");
  clearTimeout(flashSettingsMeta.timer);
  flashSettingsMeta.timer = setTimeout(() => {
    el.textContent = "применяются к следующим запускам, хранятся локально";
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

function bindControls() {
  initFromUrlAndStorage();
  
  $("runForm").addEventListener("submit", startRun);
  $("seqInput").addEventListener("input", () => {
    $("puzzleSelect").value = "";
    localStorage.setItem("seq2formula_puzzle", "");
    markPuzzle(null);
    localStorage.setItem("seq2formula_seq", $("seqInput").value);
  });
  $("refInput").addEventListener("input", () => {
    localStorage.setItem("seq2formula_ref", $("refInput").value);
  });
  $("puzzleSelect").addEventListener("change", (e) => {
    const p = PUZZLE_CACHE.find((x) => x.id === e.target.value);
    if (p) {
      $("seqInput").value = p.terms.join(" ");
      $("refInput").value = p.reference || "";  // populate the reference field
      // keep the persisted state in sync with what's on screen now
      localStorage.setItem("seq2formula_seq", $("seqInput").value);
      localStorage.setItem("seq2formula_ref", $("refInput").value);
    }
    localStorage.setItem("seq2formula_puzzle", e.target.value);
    markPuzzle(p || null);
    updateOpsHint();
  });
  $("pauseBtn").addEventListener("click", () => {
    control(state.status === "paused" ? "resume" : "pause");
  });
  $("stepBtn").addEventListener("click", () => control("step"));
  $("stopBtn").addEventListener("click", () => control("stop"));
  $("continueBtn").addEventListener("click", () => control("continue"));
  $("bestFormula").addEventListener("click", copyBestFormula);
  $("bestFormula").addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      copyBestFormula();
    }
  });
  let speedTimer = null;
  $("speedRange").addEventListener("input", (e) => {
    $("speedValue").textContent = e.target.value + " поколений/с";
    clearTimeout(speedTimer);
    speedTimer = setTimeout(() => setSpeed(Number(e.target.value)), 200);
  });

  const tabs = [["tabLibrary", "libraryTable", "library"],
                ["tabMySeqs", "mySeqTable", "myseqs"],
                ["tabCurio", "curioTable", "curio"]];
  for (const [tabId, bodyId, metaKey] of tabs) {
    $(tabId).addEventListener("click", () => {
      for (const [t, b] of tabs) {
        $(t).setAttribute("aria-selected", String(t === tabId));
        $(b).hidden = t !== tabId;
      }
      $("resultsMeta").textContent = RESULT_META[metaKey];
    });
  }
  $("resultsMeta").textContent = RESULT_META.library;

  // top-level page tabs: Поиск / Результаты / Настройки
  const panes = [["ptabSearch", "paneSearch"], ["ptabResults", "paneResults"],
                 ["ptabSettings", "paneSettings"]];
  for (const [tabId, paneId] of panes) {
    $(tabId).addEventListener("click", () => {
      for (const [t, p] of panes) {
        $(t).setAttribute("aria-selected", String(t === tabId));
        $(p).hidden = t !== tabId;
      }
      if (tabId === "ptabResults") {
        $("resultsDot").hidden = true;
        resizeCharts();
      }
    });
  }

  // settings form: save / reset, slider upper bound follows speed_max
  $("settingsForm").addEventListener("submit", saveSettings);
  $("settingsReset").addEventListener("click", resetSettings);

  // operator picker: toggle chips, block a run without any operator
  const syncOpsDisabled = () => {
    const busy = state.status === "running" || state.status === "paused";
    document.querySelectorAll(".op-chip").forEach((chip) => {
      chip.disabled = busy;
    });
  };
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
      chip.setAttribute("aria-pressed", String(on));
      localStorage.setItem("seq2formula_ops", JSON.stringify([...state.ops]));
      updateOpsHint();
    });
  });
  syncOpsDisabled();

  // re-fit charts when any chart container changes size (viewport
  // changes and pane switches); window resize alone misses both
  const ro = new ResizeObserver(resizeCharts);
  for (const id of ["chartFitness", "chartError", "chartDiversity", "chartSizes",
                    "heroChart"]) {
    ro.observe($(id));
  }

  window.addEventListener("resize", resizeCharts);
}

/* ---------- boot ---------- */

resetBuffers();
makeCharts();
bindControls();
refreshAll();
loadSettings().catch(() => {});
connectWS();
setStatus("idle");
