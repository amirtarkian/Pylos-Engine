/**
 * main.js -- dice cube solver UI: editor, manual turns, solving, playback,
 * charts and training status.
 *
 * The logical state is `values` (54 pip counts, URFDLB facelet order). Every
 * turn updates it with the server-provided move permutation and, in step,
 * animates the 3D cube.
 */

import { initCube, setState, animateMove, updateHighlights, setHoverFacelet,
         setOnFaceletClick } from "./cube3d.js";
import { pipDataUrl } from "./pips.js";
import { lineChart, barChart } from "./charts.js";

const FACES = "URFDLB";
const NET_OFFSET = { U: [0, 3], R: [3, 6], F: [3, 3], D: [6, 3], L: [3, 0], B: [3, 9] }; // [row, col]
const METHOD_INFO = {
  mcts: { name: "AlphaZero-style MCTS", blurb: "network policy + value guide a tree search" },
  astar: { name: "Weighted A*", blurb: "network value as the heuristic (DeepCubeA)" },
  kociemba: { name: "Kociemba two-phase", blurb: "classical solver over many goal states" },
};
const C1 = "#3987e5", C2 = "#d95926", C3 = "#199e70";

let geom = null;
let values = [];
let selected = null;
let queue = Promise.resolve();
let solution = null;      // { method, result, start, htm:[{move, actions}], idx }
let lastSolve = null;
let playing = false;

// ── Helpers ──────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

function applyPerm(vals, action) {
  const perm = geom.move_perms[action];
  return perm.map((src) => vals[src]);
}

function stateString(vals) {
  return FACES.split("").map((_, f) => vals.slice(f * 9, f * 9 + 9).join("")).join(" ");
}

function parseState(text) {
  const d = (text.match(/\d/g) || []).map(Number);
  return d.length === 54 ? d : null;
}

function inverse(a) { return (a + 6) % 12; }

function notationToActions(m) {
  const f = FACES.indexOf(m[0]);
  if (m.endsWith("2")) return [f, f];
  return [m.endsWith("'") ? f + 6 : f];
}

async function api(path, body) {
  const res = await fetch(path, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : undefined);
  return res.json();
}

// ── State updates ────────────────────────────────────────────────

function enqueue(fn) {
  queue = queue.then(fn).catch((e) => console.error(e));
  return queue;
}

/** Turn a layer: animate, then update the logical state. */
function turn(action, ms = 260) {
  return enqueue(async () => {
    await animateMove(action, ms);
    values = applyPerm(values, action);
    refresh();
  });
}

function loadValues(vals, { keepSolution = false } = {}) {
  values = vals.slice();
  setState(values);
  if (!keepSolution) clearSolution();
  refresh();
}

let validateTimer = null;
function refresh() {
  updateHighlights(values);
  renderNet();
  const { complete, missing } = faceStats(values);
  $("st-faces").textContent = complete;
  $("st-missing").textContent = missing;
  $("st-solved").innerHTML = complete === 6 ? '<span class="solved-badge">✓ Solved</span>' : "";
  $("state-text").value = stateString(values);
  clearTimeout(validateTimer);
  validateTimer = setTimeout(validate, 120);
}

function faceStats(vals) {
  let complete = 0, missing = 0;
  for (let f = 0; f < 6; f++) {
    const s = new Set(vals.slice(f * 9, f * 9 + 9));
    if (s.size === 9) complete++;
    missing += 9 - s.size;
  }
  return { complete, missing };
}

async function validate() {
  const r = await api("/api/validate", { state: stateString(values) });
  $("problems").innerHTML = (r.problems || []).map((p) => `⚠ ${p}`).join("<br>");
  $("st-net").textContent = r.net_cost !== undefined && r.net_cost !== null ? r.net_cost.toFixed(1) : "–";
}

// ── Net editor ───────────────────────────────────────────────────

function buildNet() {
  const net = $("net");
  net.innerHTML = "";
  const cells = {};
  for (let f = 0; f < 6; f++) {
    const [r0, c0] = NET_OFFSET[FACES[f]];
    for (let j = 0; j < 9; j++) {
      const idx = f * 9 + j;
      const cell = document.createElement("div");
      cell.className = "cell" + (j === 4 ? " center" : "");
      cell.style.gridRow = r0 + Math.floor(j / 3) + 1;
      cell.style.gridColumn = c0 + (j % 3) + 1;
      cell.title = `${FACES[f]}${j + 1}`;
      cell.appendChild(document.createElement("img"));
      if (j === 4) {
        const tag = document.createElement("div");
        tag.textContent = FACES[f];
        tag.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font:700 11px system-ui;color:#5ab0ff;text-shadow:0 0 3px #000";
        cell.appendChild(tag);
      } else {
        cell.addEventListener("click", () => select(idx));
      }
      cell.addEventListener("mouseenter", () => setHoverFacelet(idx));
      cell.addEventListener("mouseleave", () => setHoverFacelet(selected));
      net.appendChild(cell);
      cells[idx] = cell;
    }
  }
  net._cells = cells;
}

function renderNet() {
  const cells = $("net")._cells;
  const status = [];
  for (let f = 0; f < 6; f++) {
    const face = values.slice(f * 9, f * 9 + 9);
    const counts = {};
    face.forEach((v) => { counts[v] = (counts[v] || 0) + 1; });
    const done = Object.keys(counts).length === 9;
    const missing = [1, 2, 3, 4, 5, 6, 7, 8, 9].filter((v) => !counts[v]);
    status.push(`<span>${FACES[f]}: ${done ? '<span class="ok">✓ 1–9</span>'
      : `<span class="bad">missing ${missing.join(",")}</span>`}</span>`);
    for (let j = 0; j < 9; j++) {
      const idx = f * 9 + j;
      const cell = cells[idx];
      const img = cell.querySelector("img");
      const pip = done ? "#7ee2a8" : counts[values[idx]] > 1 ? "#ff8a5c" : "#f4f4f0";
      img.src = pipDataUrl(values[idx], { pip, center: j === 4 });
      cell.classList.toggle("sel", idx === selected);
      cell.classList.toggle("dup", !done && counts[values[idx]] > 1);
    }
  }
  $("face-status").innerHTML = status.join("");
}

function select(idx) {
  selected = idx;
  setHoverFacelet(idx);
  renderNet();
}

function moveSelection(dr, dc) {
  if (selected === null) return select(0);
  const f = Math.floor(selected / 9), j = selected % 9;
  const [r0, c0] = NET_OFFSET[FACES[f]];
  let r = r0 + Math.floor(j / 3) + dr, c = c0 + (j % 3) + dc;
  for (let g = 0; g < 6; g++) {
    const [gr, gc] = NET_OFFSET[FACES[g]];
    if (r >= gr && r < gr + 3 && c >= gc && c < gc + 3) {
      const k = g * 9 + (r - gr) * 3 + (c - gc);
      if (k % 9 !== 4) select(k);
      else { r += dr; c += dc; continue; }
      return;
    }
  }
}

function setSelectedValue(v) {
  if (selected === null || selected % 9 === 4) return;
  values[selected] = v;
  setState(values);
  clearSolution();
  // advance to next non-center sticker on the same face, then next face
  let next = selected + 1;
  if (next % 9 === 4) next++;
  if (next >= 54) next = 0;
  selected = next;
  setHoverFacelet(selected);
  refresh();
}

// ── Move pad & keyboard ──────────────────────────────────────────

function buildMovePad() {
  const pad = $("move-pad");
  geom.actions.forEach((name, a) => {
    const b = document.createElement("button");
    b.textContent = name;
    b.title = a < 6 ? `${name}: turn the ${faceName(a)} face clockwise` : `${name}: counter-clockwise`;
    b.addEventListener("click", () => { stopPlaying(); clearSolution(); turn(a); });
    pad.appendChild(b);
  });
}

function faceName(a) {
  return { U: "top", R: "right", F: "front", D: "bottom", L: "left", B: "back" }[FACES[a % 6]];
}

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
  if (/^[1-9]$/.test(e.key)) { setSelectedValue(Number(e.key)); e.preventDefault(); return; }
  const arrows = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
  if (arrows[e.key]) { moveSelection(...arrows[e.key]); e.preventDefault(); return; }
  if (e.key === "Escape") { selected = null; setHoverFacelet(null); renderNet(); return; }
  const f = FACES.indexOf(e.key.toUpperCase());
  if (f >= 0 && !e.metaKey && !e.ctrlKey) {
    stopPlaying();
    clearSolution();
    turn(e.shiftKey ? f + 6 : f);
  }
});

// ── Solving ──────────────────────────────────────────────────────

async function solve() {
  stopPlaying();
  $("btn-solve").disabled = true;
  const t = Number($("time-limit").value) || 8;
  $("solve-busy").textContent = `searching… (up to ~${Math.round(t * 2 + 5)} s)`;
  $("methods").innerHTML = "";
  try {
    await queue;
    const start = values.slice();
    const r = await api("/api/solve", { state: stateString(start), time_limit: t });
    lastSolve = { result: r, start };
    renderMethods();
    if (r.problems && r.problems.length) {
      $("methods").innerHTML = `<div class="hint" style="color:var(--warn)">${r.problems.join("<br>")}</div>`;
    } else if (r.already_solved) {
      $("methods").innerHTML = '<div class="hint">Already solved — every face holds 1–9.</div>';
    } else if (r.best) {
      pickMethod(r.results.mcts && r.results.mcts.found ? "mcts" : r.best);
    }
    renderVisits();
  } finally {
    $("btn-solve").disabled = false;
    $("solve-busy").textContent = "";
  }
}

function renderMethods() {
  const r = lastSolve.result;
  const el = $("methods");
  el.innerHTML = "";
  for (const key of ["mcts", "astar", "kociemba"]) {
    const res = r.results[key];
    const info = METHOD_INFO[key];
    const div = document.createElement("div");
    const ok = res && res.found && res.solved;
    div.className = "method" + (ok ? "" : " fail") + (solution && solution.method === key ? " active" : "");
    let meta = info.blurb;
    if (res) {
      if (res.nodes !== undefined) meta += ` · ${res.nodes.toLocaleString()} nodes`;
      if (res.targets_tried !== undefined) meta += ` · ${res.targets_tried} goal states tried`;
      if (res.seconds !== undefined) meta += ` · ${res.seconds}s`;
    } else {
      meta += key === "kociemba" ? " · not installed (pip install kociemba)" : " · no trained model";
    }
    div.innerHTML = `<span class="name">${info.name}${r.best === key ? '<span class="tag">shortest</span>' : ""}</span>
      <span class="len">${ok ? `${res.length_htm} <span style="font-size:11px;color:var(--muted)">moves</span>` : res ? "not found" : "—"}</span>
      <span class="meta">${meta}</span>`;
    if (ok) div.addEventListener("click", () => pickMethod(key));
    el.appendChild(div);
  }
}

function pickMethod(key) {
  stopPlaying();
  const res = lastSolve.result.results[key];
  const htm = res.moves.map((m) => ({ move: m, actions: notationToActions(m) }));
  // cost / missing traces are per quarter turn -> sample at HTM boundaries
  const qIdx = [0];
  htm.forEach((h) => qIdx.push(qIdx[qIdx.length - 1] + h.actions.length));
  solution = {
    method: key, result: res, start: lastSolve.start.slice(), htm, idx: 0,
    cost: res.cost_trace ? qIdx.map((q) => res.cost_trace[q]) : null,
    missing: qIdx.map((q) => res.missing_trace[q]),
  };
  // rewind the cube to the state that was solved
  loadValues(solution.start, { keepSolution: true });
  $("solution").style.display = "block";
  renderMethods();
  renderSolution();
}

function clearSolution() {
  if (!solution) return;
  stopPlaying();
  solution = null;
  $("solution").style.display = "none";
  if (lastSolve) renderMethods();
}

function renderSolution() {
  if (!solution) return;
  const s = solution;
  const mv = $("moves");
  mv.innerHTML = "";
  s.htm.forEach((h, i) => {
    const b = document.createElement("span");
    b.className = "mv" + (i < s.idx ? " done" : i === s.idx ? " cur" : "");
    b.textContent = h.move;
    b.title = `Move ${i + 1}: ${h.move}`;
    b.addEventListener("click", () => jumpTo(i + 1));
    mv.appendChild(b);
  });
  $("pl-count").textContent = `${s.idx} / ${s.htm.length}`;
  const labels = ["start", ...s.htm.map((h, i) => `after ${i + 1}: ${h.move}`)];
  const series = [];
  if (s.cost) series.push({ name: "Network estimate", color: C1, values: s.cost, labels });
  series.push({ name: "Actual moves left", color: C3, dash: true, labels,
                values: s.htm.map((_, i) => s.htm.slice(i).reduce((n, h) => n + h.actions.length, 0)).concat([0]) });
  lineChart($("chart-cost"), { title: "Distance to solved (quarter turns)", series, current: s.idx,
                               xName: "move", onPick: (i) => jumpTo(i) });
  lineChart($("chart-missing"), {
    title: "Values still missing across all faces",
    series: [{ name: "Missing values", color: C2, values: s.missing, labels }],
    current: s.idx, xName: "move", format: (v) => String(v), onPick: (i) => jumpTo(i),
  });
}

function step(dir) {
  if (!solution) return Promise.resolve();
  const s = solution;
  if (dir > 0 && s.idx < s.htm.length) {
    const acts = s.htm[s.idx].actions;
    s.idx++;
    acts.forEach((a) => turn(a, speedMs()));
  } else if (dir < 0 && s.idx > 0) {
    s.idx--;
    const acts = s.htm[s.idx].actions.slice().reverse().map(inverse);
    acts.forEach((a) => turn(a, speedMs()));
  }
  renderSolution();
  return queue;
}

function jumpTo(target) {
  if (!solution) return;
  stopPlaying();
  const s = solution;
  while (s.idx < target) step(1);
  while (s.idx > target) step(-1);
}

function speedMs() { return playing ? 380 : 260; }

async function play() {
  if (!solution) return;
  if (playing) return stopPlaying();
  if (solution.idx >= solution.htm.length) jumpToStart();
  playing = true;
  $("pl-play").textContent = "❚❚ Pause";
  while (playing && solution && solution.idx < solution.htm.length) {
    await step(1);
    await new Promise((r) => setTimeout(r, 180));
  }
  stopPlaying();
}

function stopPlaying() {
  playing = false;
  $("pl-play").textContent = "▶ Play";
}

function jumpToStart() {
  if (!solution) return;
  enqueue(async () => {
    solution.idx = 0;
    loadValues(solution.start, { keepSolution: true });
    renderSolution();
  });
}

function renderVisits() {
  const r = lastSolve && lastSolve.result.results.mcts;
  if (!r || !r.root_moves || !r.root_moves.length) {
    $("mcts-section").style.display = "none";
    return;
  }
  $("mcts-section").style.display = "block";
  const rm = r.root_moves;
  const first = r.found && r.actions.length ? r.actions[0] : null;
  barChart($("chart-visits"), {
    title: `Visit share per first move${first !== null ? ` · chosen: ${geom.actions[first]}` : ""}`,
    labels: rm.map((m) => m.move),
    values: rm.map((m) => m.share),
    color: C1,
    highlight: first,
    format: (v) => `${Math.round(v * 100)}%`,
    tip: (i) => `<b>${rm[i].move}</b><br>visits ${rm[i].visits} (${Math.round(rm[i].share * 100)}%)<br>
      policy prior ${(rm[i].prior * 100).toFixed(1)}%<br>best est. moves to go ${(-rm[i].value).toFixed(1)}`,
  });
}

// ── Training panel ───────────────────────────────────────────────

async function refreshTraining() {
  let t;
  try { t = await api("/api/training"); } catch { return; }
  const p = t.progress;
  const ck = t.manifest.checkpoints;
  const last = ck.length ? ck[ck.length - 1] : null;
  if (p) {
    const eta = p.eta_seconds ? ` · ETA ${Math.round(p.eta_seconds / 60)} min` : "";
    $("train-kv").innerHTML = `Status <b>${p.status}</b> · step <b>${p.step.toLocaleString()}</b> / ${p.total_steps.toLocaleString()}
      · scramble depth <b>${p.max_depth}</b>${p.status === "training" ? eta : ""}<br>
      The network learns cost-to-go by value iteration on scrambles of random solved cubes;
      depth grows each time the target network is refreshed.`;
  }
  if (t.history.length) {
    lineChart($("chart-loss"), {
      title: "Value loss (MSE, log scale)", logY: true, xName: "log point",
      series: [{ name: "Value loss", color: C1, values: t.history.map((h) => h.value_loss),
                 labels: t.history.map((h) => `step ${h.step} · depth ${h.depth}`) }],
      format: (v) => v.toFixed(4),
    });
  }
  if (last && last.greedy_solve_rate) {
    const depths = Object.keys(last.greedy_solve_rate);
    barChart($("chart-rate"), {
      title: `Greedy solve rate by scramble depth (step ${last.step}, no search)`,
      labels: depths, values: depths.map((d) => last.greedy_solve_rate[d]), color: C3, yMax: 1,
      format: (v) => `${Math.round(v * 100)}%`,
      tip: (i) => `${depths[i]}-move scrambles: <b>${Math.round(last.greedy_solve_rate[depths[i]] * 100)}%</b> solved by following the network greedily`,
    });
  }
}

// ── Boot ─────────────────────────────────────────────────────────

async function scrambleNew() {
  stopPlaying();
  const n = Number($("scramble-n").value) || 20;
  const r = await api(`/api/scramble?n=${n}`);
  await queue;
  loadValues(parseState(r.solved_state));
  r.scramble.forEach((m) => notationToActions(m).forEach((a) => turn(a, 70)));
}

async function randomSolved() {
  stopPlaying();
  const r = await api("/api/scramble?n=0");
  await queue;
  loadValues(parseState(r.state));
}

async function boot() {
  geom = await api("/api/geometry");
  initCube($("canvas-container"), geom);
  setOnFaceletClick((i) => { if (i % 9 !== 4) select(i); });
  buildNet();
  buildMovePad();

  $("btn-scramble").addEventListener("click", scrambleNew);
  $("btn-solved").addEventListener("click", randomSolved);
  $("btn-solve").addEventListener("click", solve);
  $("btn-load-text").addEventListener("click", () => {
    const v = parseState($("state-text").value);
    if (v) loadValues(v);
    else $("problems").textContent = "⚠ Need exactly 54 digits (URFDLB order).";
  });
  $("pl-next").addEventListener("click", () => { stopPlaying(); step(1); });
  $("pl-prev").addEventListener("click", () => { stopPlaying(); step(-1); });
  $("pl-start").addEventListener("click", () => { stopPlaying(); jumpToStart(); });
  $("pl-end").addEventListener("click", () => solution && jumpTo(solution.htm.length));
  $("pl-play").addEventListener("click", play);

  const r = await api("/api/scramble?n=20");
  loadValues(parseState(r.state));
  refreshTraining();
  setInterval(refreshTraining, 15000);
}

boot();
