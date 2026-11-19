/* Neuro engine training visualizer — standalone page (mode "neuro").
   Connects to the same /ws stream as the main dashboard and draws the
   learning curves: reward, best error, policy entropy, valid fraction,
   plus the root token distribution ("what the network is thinking"). */

"use strict";

const $ = (id) => document.getElementById(id);
const MAX_POINTS = 4000;            // charts keep the whole run, capped
const state = {
  status: "idle",
  mode: null,
  gens: [],        // x axis: generation numbers
  reward: [],      // mean batch reward
  bestReward: [],  // best reward in batch (shaping included)
  error: [],       // best_ever error at that generation
  entropy: [],
  valid: [],
  shockFlags: [],
  gps: null,
  lastGenAt: 0,
};

/* ---------- LaTeX (subset mirrored from app.js) ---------- */

function numText(v) { return String(parseFloat(v.toPrecision(6))); }
function isSimple(t) { return /^(n|-?\d+(\.\d+)?)$/.test(t); }
function negativeConst(ast) {
  if (ast.t === "c" && ast.v < 0) return ast.v;
  if (ast.t === "~" && ast.x.t === "c") return -ast.x.v;
  return null;
}
function astToLatex(ast) {
  if (ast.t === "c") {
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
    if (ast.n === "sqrt") return `\\sqrt{${inner}}`;
    if (ast.n === "cbrt") return `\\sqrt[3]{${inner}}`;
    if (ast.n === "abs") return `\\left|${inner}\\right|`;
    if (ast.n === "floor") return `\\left\\lfloor${inner}\\right\\rfloor`;
    if (ast.n === "fact") {
      const bare = ast.x.t === "n" || (ast.x.t === "c" && ast.x.v >= 0);
      return `${bare ? inner : `\\left(${inner}\\right)`}!`;
    }
    return `\\operatorname{${ast.n}}\\left(${inner}\\right)`;
  }
  const neg = ast.t === "+" || ast.t === "-" ? negativeConst(ast.r) : null;
  if (neg !== null) {
    const op = ast.t === "+" ? "-" : "+";
    return `${astToLatex(ast.l)} ${op} ${numText(Math.abs(neg))}`;
  }
  if (ast.t === "if") {
    // piecewise fork: the cases environment draws its own left brace
    return `\\begin{cases}${astToLatex(ast.a)} & \\text{if } ${astToLatex(ast.c)} \\\\ ${astToLatex(ast.b)} & \\text{otherwise}\\end{cases}`;
  }
  const L = astToLatex(ast.l);
  const R = astToLatex(ast.r);
  switch (ast.t) {
    case "+": return `${L} + ${R}`;
    case "-": return `${L} - ${/^-| \+ | - /.test(R) ? `\\left(${R}\\right)` : R}`;
    case "*":
      if (/^\d+(\.\d+)?$/.test(L) && !/^\d/.test(R)) return `${L}${R}`;
      return `${L} \\cdot ${R}`;
    case "/": return `\\frac{${L}}{${R}}`;
    case "mod": return `\\left(${L}\\right) \\bmod \\left(${R}\\right)`;
    case "^": return `${isSimple(L) ? L : `\\left(${L}\\right)`}^{${R}}`;
    case "<": return `${L} \\lt ${R}`;
    case ">": return `${L} \\gt ${R}`;
  }
  return "?";
}
function renderKatex(el, latex) {
  try { katex.render(latex, el, { displayMode: true, throwOnError: true }); }
  catch (e) { el.textContent = latex; }
}

/* ---------- charts ---------- */

const commonOpts = {
  width: 760, height: 190,
  legend: { show: false },
  cursor: { show: false },
  scales: { x: { time: false } },
  axes: [
    { stroke: "#94949b", grid: { stroke: "#232327" }, ticks: { show: false } },
    { stroke: "#94949b", grid: { stroke: "#232327" }, ticks: { show: false } },
  ],
};

function mkLine(series, opts = {}) {
  const el = opts.el;
  const data = [state.gens, ...series];
  return new uPlot({
    ...commonOpts,
    width: el.clientWidth || 700,   // fit the card, no fixed overflow
    ...opts.uopts,
    series: [
      {},
      ...series.map((s) => ({
        stroke: s.color, width: s.width || 1.5,
        fill: s.fill || undefined,
        points: { show: false },
        spanGaps: true,
      })),
    ],
  }, data, el);
}

let chReward, chError, chEntropy;

function push(gen, g) {
  const s = g.stats || {};
  state.gens.push(gen);
  state.reward.push(s.reward ?? null);
  state.bestReward.push(s.best_reward ?? null);
  state.entropy.push(s.entropy ?? null);
  state.valid.push(s.valid_frac ?? null);
  state.shockFlags.push(!!g.shock);
  // best_ever error progression: the engine reports it in every event
  const prevErr = state.error.length ? state.error[state.error.length - 1] : null;
  state.error.push(s.best_error ?? prevErr);
  if (state.gens.length > MAX_POINTS) {
    for (const arr of [state.gens, state.reward, state.bestReward,
                       state.entropy, state.valid, state.error,
                       state.shockFlags]) arr.shift();
  }
}

function drawCharts() {
  chReward.setData([state.gens, state.reward, state.bestReward]);
  chError.setData([state.gens, state.error]);
  chEntropy.setData([state.gens, state.entropy, state.valid]);
}

function sizeCharts() {
  const w = $("chartReward").clientWidth || 760;
  [chReward, chError, chEntropy].forEach((c) => {
    if (!c) return;
    // vendored uPlot exposes setSize, newer builds resize
    if (typeof c.resize === "function") c.resize(w, 190);
    else if (typeof c.setSize === "function") c.setSize({ width: w, height: 190 });
  });
}

/* ---------- KPIs, formula, tokens ---------- */

function tokLabel(tok) {
  if (tok === "var") return "n";
  if (tok === "neg") return "-()";
  if (tok === "^nk") return "n^k";
  if (tok === "^kn") return "k^n";
  if (tok === "^en") return "(…)^n";
  if (tok === "^nn") return "n^n";
  // const tokens are exactly "c" + a number: "c2", "c-3", "c0.5";
  // anything else that merely starts with c (cos, cbrt) is a function
  if (/^c[-0-9.]/.test(tok)) return tok.slice(1);
  return tok;
}

function renderKpis(g) {
  const s = (g && g.stats) || {};
  $("kpiGen").textContent = g ? g.gen : 0;
  // progressive grammar stage: 0 = base alphabet, 1 = +functions,
  // 2 = +piecewise/specials (the full enabled grammar)
  $("kpiStage").textContent =
    s.stage === 1 ? "base + functions"
    : s.stage === 2 ? "full" : "base";
  $("kpiErr").textContent =
    (s.best_error != null)
      ? (s.best_error * 100).toFixed(2) + "%" : "—";
  $("kpiReward").textContent = s.reward != null ? s.reward.toFixed(3) : "—";
  $("kpiH").textContent = s.entropy != null ? s.entropy.toFixed(2) : "—";
  $("kpiValid").textContent =
    s.valid_frac != null ? Math.round(s.valid_frac * 100) + "%" : "—";
  $("kpiGps").textContent = state.gps != null ? state.gps.toFixed(1) + "/s" : "—";
}

function renderBest(g) {
  if (!g || !g.best || !g.best.ast) {
    $("bestFormula").textContent = "—";
    $("bestMeta").textContent = "";
    return;
  }
  renderKatex($("bestFormula"), astToLatex(g.best.ast));
  const bits = [i18n.t(`error ${(g.best.error * 100).toFixed(2)}%`),
                i18n.t(`size ${g.best.size}`)];
  if (g.best.exact) bits.push("exact hit");
  $("bestMeta").textContent = bits.join(" · ");
}

function renderTokens(g) {
  const probs = g && g.stats && g.stats.root_probs;
  const el = $("tokBars");
  if (!probs || !probs.length) {
    el.innerHTML = '<div style="color:var(--faint);font-size:12px">no data</div>';
    return;
  }
  const max = probs[0][1] || 1;
  el.innerHTML = probs.map(([tok, p]) => `
    <div class="tokbar-row">
      <span class="tokbar-name">${tokLabel(tok)}</span>
      <div class="tokbar-track"><div class="tokbar-fill"
        style="width:${(p / max * 100).toFixed(1)}%"></div></div>
      <span class="tokbar-val">${(p * 100).toFixed(0)}%</span>
    </div>`).join("");
}

function renderTop5(g) {
  if (window.i18n && i18n.lang !== "en") i18n.apply();
  const tb = $("top5Table").querySelector("tbody");
  tb.innerHTML = (g && g.top ? g.top : []).map((row) => `
    <tr><td class="num">${(row.error != null ? (row.error * 100).toFixed(2) : "—")}%</td>
        <td>${row.text}</td></tr>`).join("");
}

function renderStatus(snap) {
  const badge = $("statusBadge");
  const running = snap.status === "running" || snap.status === "paused";
  badge.hidden = !(running || snap.solved);
  if (snap.solved) {
    badge.textContent = "solved";
    badge.className = "badge-solved";
  } else if (snap.status === "running") {
    badge.textContent = "training";
    badge.className = "badge-run";
  } else {
    badge.textContent = "paused";
    badge.className = "badge-shock";  // amber: paused matches the main chip
  }
}

/* ---------- websocket ---------- */

let ws;
function setConn(ok) {
  const el = $("conn");
  if (el) el.classList.toggle("bad", !ok);
  $("connText").textContent = ok ? "connected" : "no connection to the server";
}
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => setConn(true);
  ws.onclose = () => { setConn(false); setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "status") { applySnapshot(msg); return; }
    if (msg.type === "solved" || msg.type === "finished") {
      $("statusBadge").hidden = false;
      $("statusBadge").className = "badge-solved";
      $("statusBadge").textContent = msg.type === "solved" ? "solved" : "finished";
      if (msg.type === "solved") flashSolved();
      return;
    }
    if (msg.mode !== "neuro") return;   // another engine's run
    state.lastGenAt = performance.now();
    push(msg.gen, msg);
    renderKpis(msg); renderBest(msg); renderTokens(msg);
    renderTop5(msg); drawCharts();
    $("shockBadge").hidden = !msg.shock;
    if (msg.solved_now) flashSolved();
  };
}

function applySnapshot(snap) {
  state.status = snap.status;
  state.mode = snap.mode;
  if (snap.mode === "neuro" && snap.last_gen && !state.gens.length) {
    // restore after reload: replay the snapshot's last state only
    push(snap.last_gen.gen, snap.last_gen);
    renderKpis(snap.last_gen); renderBest(snap.last_gen);
    renderTokens(snap.last_gen); renderTop5(snap.last_gen);
    drawCharts();
  }
  renderStatus(snap);
  $("runInfo").innerHTML = snap.puzzle
    ? `<div class="stat-line"><span>${i18n.t("puzzle")}</span><b>${snap.puzzle.name}</b></div>
       <div class="stat-line"><span>Seed</span><b class="num">${snap.seed ?? "—"}</b></div>`
    : "";
}

let flashTimer;
function flashSolved() {
  document.title = "Solved — seq2formula";
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { document.title = "seq2formula — Neuro"; }, 4000);
}

/* ---------- controls ---------- */

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || res.statusText);
  }
  return res.json();
}

async function startRun() {
  const raw = $("termsInput").value.trim().replace(/,/g, " ");
  const terms = raw.split(/\s+/).filter(Boolean).map(Number);
  if (terms.length < 2 || terms.some((v) => !Number.isFinite(v))) {
    alert("Enter at least two numbers separated by spaces");
    return;
  }
  resetData();
  const seedRaw = $("seedInput").value.trim();
  try {
    await post("/api/run", {
      mode: "neuro",
      terms,
      seed: seedRaw === "" ? undefined : Number(seedRaw),
      speed: Number($("speedSlider").value),
    });
  } catch (e) { alert("failed to start: " + e.message); }
}

function resetData() {
  for (const arr of [state.gens, state.reward, state.bestReward,
                     state.error, state.entropy, state.valid,
                     state.shockFlags]) arr.length = 0;
  drawCharts();
}

function wireControls() {
  $("btnStart").onclick = startRun;
  $("btnPause").onclick = () => post("/api/control",
    { action: state.status === "paused" ? "resume" : "pause" });
  $("btnStep").onclick = () => post("/api/control", { action: "step" });
  $("btnStop").onclick = () => post("/api/control", { action: "stop" });
  $("speedSlider").oninput = (e) => {
    $("speedLabel").textContent = `${e.target.value}/s`;
  };
  $("speedSlider").onchange = (e) =>
    post("/api/speed", { gps: Number(e.target.value) }).catch(() => {});
}

async function loadPuzzles() {
  const data = await fetch("/api/puzzles").then((r) => r.json());
  const sel = $("puzzleSelect");
  const puzzles = data.puzzles || data;
  for (const p of puzzles.slice(0, 40)) {
    const opt = document.createElement("option");
    opt.value = (p.terms || []).join(" ");
    opt.textContent = `T${p.tier} · ${p.name} · ${(p.terms || []).slice(0, 5).join(" ")}…`;
    sel.appendChild(opt);
  }
  sel.onchange = () => { $("termsInput").value = sel.value; };
}

/* ---------- speed / gps sampler ---------- */

setInterval(() => {
  const now = performance.now();
  const dt = (now - state.lastGenAt) / 1000;
  if (state.status === "running" && dt > 0 && dt < 3 && state.lastGenAt) {
    state.gps = 1 / dt;
    $("kpiGps").textContent = state.gps.toFixed(1) + "/s";
  } else if (state.status !== "running") {
    state.gps = null;
    $("kpiGps").textContent = "—";
  }
}, 500);

/* ---------- boot ---------- */

chReward = mkLine(
  [{ color: "#4ec9b0" }, { color: "#8b93f8", width: 2 }],
  { el: $("chartReward") });
chError = mkLine(
  [{ color: "#e2a655", width: 2, fill: "#2a221433" }],
  { el: $("chartError") });
chEntropy = mkLine(
  [{ color: "#8b93f8" }, { color: "#3fc07a" }],
  { el: $("chartEntropy") });

window.addEventListener("resize", sizeCharts);
sizeCharts();
wireControls();
connect();
loadPuzzles();

