/**
 * charts.js -- tiny dependency-free SVG charts with hover tooltips.
 *
 *   lineChart(el, { title, series:[{name,color,values,dash}], current, yLabel })
 *   barChart(el,  { title, labels, values, color, tip(i), yMax, format })
 */

const NS = "http://www.w3.org/2000/svg";
const W = 400;

function svgEl(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

function frame(el, title, legend, height) {
  el.innerHTML = "";
  const head = document.createElement("div");
  head.className = "chart-title";
  head.innerHTML = `<span>${title}</span>`;
  if (legend && legend.length > 1) {
    const lg = document.createElement("span");
    lg.className = "chart-legend";
    lg.innerHTML = legend.map((s) =>
      `<span><i style="background:${s.color}${s.dash ? ";opacity:.85" : ""}"></i>${s.name}</span>`).join("");
    head.appendChild(lg);
  }
  el.appendChild(head);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${height}`, role: "img", "aria-label": title });
  el.appendChild(svg);
  const tip = document.createElement("div");
  tip.className = "tip";
  el.appendChild(tip);
  return { svg, tip };
}

function showTip(el, tip, html, xFrac, yPx) {
  tip.innerHTML = html;
  tip.style.display = "block";
  const box = el.getBoundingClientRect();
  const x = xFrac * box.width;
  tip.style.left = `${Math.min(Math.max(0, x - tip.offsetWidth / 2), box.width - tip.offsetWidth)}px`;
  tip.style.top = `${yPx - tip.offsetHeight - 6}px`;
}

export function lineChart(el, { title, series, current = null, height = 120, xName = "step",
                                format = (v) => v.toFixed(1), logY = false, onPick = null }) {
  const { svg, tip } = frame(el, title, series, height);
  const m = { l: 30, r: 8, t: 8, b: 18 };
  const n = Math.max(...series.map((s) => s.values.length));
  if (n === 0) return;
  const all = series.flatMap((s) => s.values).filter((v) => v !== null && Number.isFinite(v));
  const tx = (v) => (logY ? Math.log10(Math.max(v, 1e-4)) : v);
  let yMin = logY ? Math.floor(Math.min(...all.map(tx))) : 0;
  let yMax = logY ? Math.ceil(Math.max(...all.map(tx))) : niceMax(Math.max(...all));
  if (yMax === yMin) yMax = yMin + 1;
  const xs = (i) => m.l + (n === 1 ? 0 : (i / (n - 1)) * (W - m.l - m.r));
  const ys = (v) => m.t + (1 - (tx(v) - yMin) / (yMax - yMin)) * (height - m.t - m.b);

  const ticks = logY ? Array.from({ length: yMax - yMin + 1 }, (_, i) => yMin + i)
                     : [0, 0.5, 1].map((f) => yMin + f * (yMax - yMin));
  const axis = svgEl("g", { class: "axis" }, svg);
  for (const t of ticks) {
    const y = logY ? m.t + (1 - (t - yMin) / (yMax - yMin)) * (height - m.t - m.b) : ys(t);
    svgEl("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "gridline" }, axis);
    const lbl = svgEl("text", { x: m.l - 4, y: y + 3, "text-anchor": "end" }, axis);
    lbl.textContent = logY ? (t >= 0 ? `1e${t}` : `1e${t}`) : (Number.isInteger(t) ? t : t.toFixed(1));
  }
  const xl = svgEl("text", { x: W - m.r, y: height - 3, "text-anchor": "end" }, axis);
  xl.textContent = `${xName} → ${n - 1}`;

  if (current !== null) {
    svgEl("line", { x1: xs(current), x2: xs(current), y1: m.t, y2: height - m.b,
                    stroke: "#5ab0ff", "stroke-width": 1, "stroke-dasharray": "3 3" }, svg);
  }

  for (const s of series) {
    const pts = s.values.map((v, i) => (v === null ? null : `${xs(i)},${ys(v)}`)).filter(Boolean);
    svgEl("polyline", { points: pts.join(" "), fill: "none", stroke: s.color, "stroke-width": 2,
                        "stroke-linejoin": "round", "stroke-linecap": "round",
                        ...(s.dash ? { "stroke-dasharray": "5 4" } : {}) }, svg);
    if (current !== null && s.values[current] !== undefined && s.values[current] !== null) {
      svgEl("circle", { cx: xs(current), cy: ys(s.values[current]), r: 4, fill: s.color,
                        stroke: "#14142b", "stroke-width": 2 }, svg);
    }
  }

  // Hover crosshair
  const cross = svgEl("line", { y1: m.t, y2: height - m.b, stroke: "rgba(255,255,255,.35)",
                                "stroke-width": 1, visibility: "hidden" }, svg);
  const hit = svgEl("rect", { x: m.l, y: 0, width: W - m.l - m.r, height, fill: "transparent",
                              style: onPick ? "cursor:pointer" : "" }, svg);
  const idxAt = (evt) => {
    const box = svg.getBoundingClientRect();
    const x = ((evt.clientX - box.left) / box.width) * W;
    return Math.max(0, Math.min(n - 1, Math.round(((x - m.l) / (W - m.l - m.r)) * (n - 1))));
  };
  hit.addEventListener("mousemove", (evt) => {
    const i = idxAt(evt);
    cross.setAttribute("x1", xs(i));
    cross.setAttribute("x2", xs(i));
    cross.setAttribute("visibility", "visible");
    const rows = series.filter((s) => s.values[i] !== undefined && s.values[i] !== null)
      .map((s) => `<div><span style="color:${s.color}">●</span> ${s.name}: <b>${format(s.values[i])}</b></div>`);
    const label = series[0].labels ? series[0].labels[i] : `${xName} ${i}`;
    showTip(el, tip, `<div style="color:#a9a9c0">${label}</div>${rows.join("")}`,
            xs(i) / W, el.querySelector(".chart-title").offsetHeight + m.t);
  });
  hit.addEventListener("mouseleave", () => {
    cross.setAttribute("visibility", "hidden");
    tip.style.display = "none";
  });
  if (onPick) hit.addEventListener("click", (evt) => onPick(idxAt(evt)));
}

export function barChart(el, { title, labels, values, color, tip: tipFn, yMax = null, height = 110,
                               format = (v) => v.toFixed(2), highlight = null }) {
  const { svg, tip } = frame(el, title, null, height);
  const m = { l: 30, r: 8, t: 8, b: 18 };
  const top = yMax ?? niceMax(Math.max(...values, 1e-9));
  const n = values.length;
  const slot = (W - m.l - m.r) / n;
  const bw = Math.max(4, Math.min(26, slot - 2));   // 2px gap between bars
  const ys = (v) => m.t + (1 - v / top) * (height - m.t - m.b);
  const axis = svgEl("g", { class: "axis" }, svg);
  for (const f of [0, 0.5, 1]) {
    const y = ys(f * top);
    svgEl("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: "gridline" }, axis);
    const t = svgEl("text", { x: m.l - 4, y: y + 3, "text-anchor": "end" }, axis);
    t.textContent = format(f * top);
  }
  values.forEach((v, i) => {
    const x = m.l + i * slot + (slot - bw) / 2;
    const y = ys(v);
    const h = Math.max(0, height - m.b - y);
    const r = Math.min(4, bw / 2, h);
    // Rounded top, square baseline
    const d = h <= 0 ? "" :
      `M${x},${height - m.b} V${y + r} Q${x},${y} ${x + r},${y} H${x + bw - r} Q${x + bw},${y} ${x + bw},${y + r} V${height - m.b} Z`;
    const isHi = highlight !== null && highlight === i;
    if (d) svgEl("path", { d, fill: color, opacity: highlight === null || isHi ? 1 : 0.55 }, svg);
    const lbl = svgEl("text", { x: x + bw / 2, y: height - 5, "text-anchor": "middle" }, axis);
    lbl.textContent = labels[i];
    const hitbox = svgEl("rect", { x: m.l + i * slot, y: m.t, width: slot, height: height - m.t - m.b,
                                   fill: "transparent" }, svg);
    hitbox.addEventListener("mousemove", () => {
      showTip(el, tip, tipFn ? tipFn(i) : `${labels[i]}: <b>${format(v)}</b>`,
              (x + bw / 2) / W, el.querySelector(".chart-title").offsetHeight + y * (el.querySelector("svg").getBoundingClientRect().height / height));
    });
    hitbox.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  });
}
