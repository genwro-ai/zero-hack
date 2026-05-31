window.ZHCharts = (function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";

  function s(tag, attrs, kids) {
    const n = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    if (kids) (Array.isArray(kids) ? kids : [kids]).forEach((c) =>
      n.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return n;
  }
  function h(tag, attrs, kids) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") n.className = attrs[k];
      else if (k === "html") n.innerHTML = attrs[k];
      else if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    }
    if (kids != null) (Array.isArray(kids) ? kids : [kids]).forEach((c) =>
      n.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return n;
  }
  const fmt = (x, d = 3) => (x == null ? "—" : Number(x).toFixed(d));
  const pct = (x) => (x == null ? "—" : (x * 100).toFixed(1) + "%");
  function hash(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
    return (h >>> 0) / 4294967296;
  }
  const color = (kind) => (kind === "baseline" ? "var(--baseline)" : "var(--accent)");

  let tip;
  function tooltip() { if (!tip) { tip = h("div", { class: "zh-tip" }); document.body.appendChild(tip); } return tip; }
  function showTip(html, ev) { const t = tooltip(); t.innerHTML = html; t.style.opacity = "1"; moveTip(ev); }
  function moveTip(ev) {
    const t = tooltip(), pad = 14;
    let x = ev.clientX + pad, y = ev.clientY + pad;
    const r = t.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - pad;
    t.style.left = x + "px"; t.style.top = y + "px";
  }
  function hideTip() { if (tip) tip.style.opacity = "0"; }
  function bindTip(node, html) {
    node.addEventListener("mouseenter", (e) => showTip(html, e));
    node.addEventListener("mousemove", moveTip);
    node.addEventListener("mouseleave", hideTip);
  }

  function barChart(container, rows, opts) {
    opts = opts || {};
    const fmtv = opts.format || pct;
    const domain = opts.domain || [0, 1];
    const rowH = opts.rowH || 42, padT = 10, padB = 26;
    const W = container.clientWidth || 560;
    const mLeft = opts.labelWidth || 168, mRight = opts.valueWidth || 64;
    const H = padT + padB + rows.length * rowH;
    const innerW = Math.max(40, W - mLeft - mRight);
    const x = (v) => mLeft + ((v - domain[0]) / (domain[1] - domain[0])) * innerW;
    const svg = s("svg", { class: "zh-svg", viewBox: `0 0 ${W} ${H}`, width: "100%", height: H });

    const ticks = opts.ticks || [domain[0], (domain[0] + domain[1]) / 2, domain[1]];
    ticks.forEach((tv) => {
      svg.appendChild(s("line", { x1: x(tv), x2: x(tv), y1: padT, y2: H - padB, class: "zh-grid" }));
      svg.appendChild(s("text", { x: x(tv), y: H - 8, class: "zh-axis", "text-anchor": "middle" }, fmtv(tv)));
    });
    if (opts.reference != null) {
      svg.appendChild(s("line", { x1: x(opts.reference), x2: x(opts.reference), y1: padT - 2, y2: H - padB + 2, class: "zh-ref" }));
      svg.appendChild(s("text", { x: x(opts.reference), y: padT - 1, class: "zh-ref-label", "text-anchor": "middle" }, opts.referenceLabel || "ref"));
    }

    rows.forEach((r, i) => {
      const y0 = padT + i * rowH, barH = Math.min(22, rowH - 16), by = y0 + (rowH - barH) / 2;
      const c = color(r.kind);
      const g = s("g", { class: "zh-bar" });
      g.appendChild(s("text", { x: mLeft - 12, y: by + barH / 2 + 4, class: "zh-rowlabel", "text-anchor": "end" }, r.label));
      g.appendChild(s("rect", { x: mLeft, y: by, width: innerW, height: barH, rx: 2, class: "zh-track" }));
      const bw = r.value == null ? 0 : Math.max(0, x(r.value) - mLeft);
      const rect = s("rect", { x: mLeft, y: by, width: bw, height: barH, rx: 2, fill: c, class: "zh-fill" + (r.best ? " best" : "") });
      rect.style.transform = "scaleX(0)";
      rect.style.transitionDelay = (i * 70) + "ms";
      g.appendChild(rect);
      if (typeof requestAnimationFrame !== "undefined") requestAnimationFrame(() => requestAnimationFrame(() => { rect.style.transform = "scaleX(1)"; }));
      else rect.style.transform = "scaleX(1)";
      g.appendChild(s("text", { x: r.value == null ? mLeft + 6 : x(r.value) + 8, y: by + barH / 2 + 4,
        class: "zh-val" + (r.value == null ? " na" : ""), "text-anchor": "start" }, fmtv(r.value)));
      bindTip(g, `<b>${r.label}</b><br>${opts.metricName || "value"}: <b>${fmtv(r.value)}</b>` + (r.sub ? `<br><span class="muted">${r.sub}</span>` : ""));
      svg.appendChild(g);
    });
    container.appendChild(svg);
  }

  function waferMap(container, opts) {
    opts = opts || {};
    const ring = opts.ring || 12, gap = opts.gap || 3, size = opts.size || 300;
    const die = (size - (ring + 1) * gap) / ring;
    const cx = size / 2, cy = size / 2, R = size / 2 - 2;
    const svg = s("svg", { class: "zh-svg zh-wafer", viewBox: `0 0 ${size} ${size + 8}`, width: "100%", height: size + 8 });
    svg.appendChild(s("circle", { cx: cx, cy: cy, r: R, fill: "var(--wafer-bg)", stroke: "var(--rule)", "stroke-width": 1 }));
    let total = 0, flagged = 0;
    for (let r = 0; r < ring; r++) for (let c = 0; c < ring; c++) {
      const x = gap + c * (die + gap), y = gap + r * (die + gap);
      const dcx = x + die / 2, dcy = y + die / 2;
      if (Math.hypot(dcx - cx, dcy - cy) > R - die * 0.45) continue;
      total++;
      const bad = hash(`${opts.seed || "w"}:${r}:${c}`) < (opts.rate || 0.1);
      if (bad) flagged++;
      const g = s("g");
      g.appendChild(s("rect", { x: x, y: y, width: die, height: die, rx: 1.5,
        fill: bad ? "rgba(var(--oxide-rgb),.92)" : "rgba(100,116,139,.16)",
        stroke: bad ? "var(--oxide)" : "rgba(100,116,139,.4)", "stroke-width": bad ? 1 : 0.6,
        class: "zh-die" + (bad ? " bad" : "") }));
      bindTip(g, bad ? `<b>die ${r}·${c}</b><br>flagged: rule violation` : `die ${r}·${c}<br>valid`);
      svg.appendChild(g);
    }
    svg.appendChild(s("line", { x1: cx - 16, y1: size + 2, x2: cx + 16, y2: size + 2, stroke: "var(--ink)", "stroke-width": 2 }));
    container.appendChild(svg);
    return { total, flagged };
  }

  const FLOW_COLORS = {
    clean: "#475569", thermal: "#c2410c", litho: "#0b5cad", etch: "#6d28d9",
    doping: "#a16207", deposition: "#0f766e", planarize: "#334155", via: "#0e7490",
    metal: "#1e5a8a", passivation: "#4d7c0f", test: "#15803d", logistics: "#86198f", other: "#475569",
  };
  function classifyStep(label) {
    const s = (label || "").toUpperCase();
    if (s.includes("LITHO") || s.includes("PHOTORESIST") || s.includes("DEVELOP") || s.includes("BAKE")) return "litho";
    if (s.includes("ETCH") || s.includes("STRIP") || s.includes("OPEN")) return "etch";
    if (s.includes("IMPLANT") || s.includes("DOP")) return "doping";
    if (s.includes("ANNEAL") || s.includes("OXIDATION") || s.includes("OXIDE GROWTH") || s.includes("DIFFUSION")) return "thermal";
    if (s.includes("DEPOSIT") || s.includes("SPUTTER") || s.includes("GROWTH") || s.includes("POLY")) return "deposition";
    if (s.includes("CMP") || s.includes("PLANAR") || s.includes("GRIND")) return "planarize";
    if (s.includes("VIA") || s.includes("CONTACT")) return "via";
    if (s.includes("METAL")) return "metal";
    if (s.includes("PASSIVATION")) return "passivation";
    if (s.includes("TEST") || s.includes("MEASURE") || s.includes("INSPECT")) return "test";
    if (s.includes("SHIP") || s.includes("LOT") || s.includes("RELEASE")) return "logistics";
    if (s.includes("CLEAN") || s.includes("RCA")) return "clean";
    return "other";
  }
  function processFlow(container, steps, opts) {
    opts = opts || {};
    const wrap = h("div", { class: "zh-flow" + (opts.compact ? " compact" : "") });
    steps.forEach((st, i) => {
      const type = st.type || classifyStep(st.label || st);
      const label = st.label || st;
      const node = h("div", { class: "zh-flow-step" + (st.flag ? " flag" : "") });
      node.style.setProperty("--fc", FLOW_COLORS[type] || FLOW_COLORS.other);
      node.style.animationDelay = (i * 38) + "ms";
      node.appendChild(h("span", { class: "zh-flow-i" }, String(i + 1).padStart(2, "0")));
      node.appendChild(h("span", { class: "zh-flow-l" }, label));
      if (st.flag) node.appendChild(h("span", { class: "zh-flow-x" }, "⚑"));
      wrap.appendChild(node);
      if (i < steps.length - 1) wrap.appendChild(h("span", { class: "zh-flow-arrow" }, "›"));
    });
    container.appendChild(wrap);
  }

  return { barChart, waferMap, processFlow, classifyStep, h, s, fmt, pct, color };
})();
