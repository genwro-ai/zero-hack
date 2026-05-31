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
  const RED = "#e41717", RED_WARM = "#e35052", ID = "#d7dde2", MUTED = "#807d76", OK = "#4ea36a";
  const kindColor = (kind) =>
    kind === "chosen" ? RED : kind === "neural" ? ID : MUTED;

  /* ---- tooltip ---- */
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

  /* ---- single-metric horizontal bar chart ---- */
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
      const c = r.color || kindColor(r.kind);
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

  /* ---- grouped bar: each row carries several named values (e.g. ID/OOD) ---- */
  function groupedBar(container, rows, opts) {
    opts = opts || {};
    const series = opts.series; // [{key,label,color}]
    const domain = opts.domain || [0, 1];
    const W = container.clientWidth || 560;
    const mLeft = opts.labelWidth || 150, mRight = 52;
    const groupGap = 22, barH = 13, barGap = 4, padT = 8, padB = 26;
    const groupH = series.length * barH + (series.length - 1) * barGap;
    const rowH = groupH + groupGap;
    const H = padT + padB + rows.length * rowH;
    const innerW = Math.max(40, W - mLeft - mRight);
    const x = (v) => mLeft + ((v - domain[0]) / (domain[1] - domain[0])) * innerW;
    const fmtv = opts.format || pct;
    const svg = s("svg", { class: "zh-svg", viewBox: `0 0 ${W} ${H}`, width: "100%", height: H });

    (opts.ticks || [domain[0], (domain[0] + domain[1]) / 2, domain[1]]).forEach((tv) => {
      svg.appendChild(s("line", { x1: x(tv), x2: x(tv), y1: padT, y2: H - padB, class: "zh-grid" }));
      svg.appendChild(s("text", { x: x(tv), y: H - 8, class: "zh-axis", "text-anchor": "middle" }, fmtv(tv)));
    });

    rows.forEach((r, i) => {
      const gy = padT + i * rowH;
      const g = s("g");
      g.appendChild(s("text", { x: mLeft - 12, y: gy + groupH / 2 + 4, class: "zh-rowlabel" + (r.chosen ? "" : ""), "text-anchor": "end",
        fill: r.chosen ? RED_WARM : "var(--ink-soft)" }, (r.chosen ? "▸ " : "") + r.label));
      series.forEach((se, j) => {
        const by = gy + j * (barH + barGap);
        g.appendChild(s("rect", { x: mLeft, y: by, width: innerW, height: barH, rx: 2, class: "zh-track" }));
        const v = r[se.key];
        const bw = v == null ? 0 : Math.max(0, x(v) - mLeft);
        const rect = s("rect", { x: mLeft, y: by, width: bw, height: barH, rx: 2, fill: se.color, class: "zh-fill" + (r.chosen && se.highlight ? " best" : "") });
        rect.style.opacity = r.chosen ? 1 : (se.dim ? .55 : .82);
        rect.style.transform = "scaleX(0)"; rect.style.transitionDelay = (i * 60 + j * 40) + "ms";
        g.appendChild(rect);
        requestAnimationFrame(() => requestAnimationFrame(() => { rect.style.transform = "scaleX(1)"; }));
        g.appendChild(s("text", { x: x(v) + 7, y: by + barH - 2, class: "zh-val", "font-size": "10.5", "text-anchor": "start" }, fmtv(v)));
        const cell = s("rect", { x: mLeft, y: by, width: innerW, height: barH, fill: "transparent" });
        bindTip(cell, `<b>${r.label}</b> · ${se.label}<br>${fmtv(v)}` + (r.sub ? `<br><span class="muted">${r.sub}</span>` : ""));
        g.appendChild(cell);
      });
      svg.appendChild(g);
    });
    container.appendChild(svg);
  }

  /* ---- ID -> OOD gap track: light marker (ID) connected to red marker (OOD) ---- */
  function gapBar(container, d, opts) {
    opts = opts || {};
    const domain = opts.domain || [0, 1];
    const W = container.clientWidth || 480, H = 46;
    const mLeft = 4, mRight = 4, cy = 22;
    const innerW = W - mLeft - mRight;
    const x = (v) => mLeft + ((v - domain[0]) / (domain[1] - domain[0])) * innerW;
    const svg = s("svg", { class: "zh-svg", viewBox: `0 0 ${W} ${H}`, width: "100%", height: H });
    // baseline track
    svg.appendChild(s("line", { x1: mLeft, x2: W - mRight, y1: cy, y2: cy, stroke: "rgba(255,255,255,.06)", "stroke-width": 6, "stroke-linecap": "round" }));
    const xi = x(d.id), xo = x(d.ood);
    // drop connector
    if (!d.flat) {
      const lo = Math.min(xi, xo), hi = Math.max(xi, xo);
      const drop = s("line", { x1: lo, x2: lo, y1: cy, y2: cy, stroke: RED, "stroke-width": 6, "stroke-linecap": "round", opacity: .55 });
      svg.appendChild(drop);
      requestAnimationFrame(() => requestAnimationFrame(() => { drop.setAttribute("x2", hi); drop.style.transition = "all .7s var(--ease)"; }));
    }
    // ID marker (light)
    const gi = s("g");
    gi.appendChild(s("circle", { cx: xi, cy, r: 7, fill: ID, stroke: "var(--bg)", "stroke-width": 2 }));
    gi.appendChild(s("text", { x: xi, y: cy - 13, class: "zh-val", "text-anchor": "middle", "font-size": "11", fill: ID }, pct(d.id)));
    gi.appendChild(s("text", { x: xi, y: cy + 19, class: "zh-axis", "text-anchor": "middle", "font-size": "8.5" }, "ID"));
    bindTip(gi, `<b>in-distribution</b><br>${pct(d.id)}`);
    // OOD marker (red)
    const go = s("g");
    go.appendChild(s("circle", { cx: xo, cy, r: 7, fill: d.flat ? ID : RED, stroke: "var(--bg)", "stroke-width": 2, filter: d.flat ? null : "drop-shadow(0 0 5px rgba(228,23,23,.7))" }));
    go.appendChild(s("text", { x: xo, y: cy - 13, class: "zh-val", "text-anchor": "middle", "font-size": "11", fill: d.flat ? ID : RED_WARM }, pct(d.ood)));
    go.appendChild(s("text", { x: xo, y: cy + 19, class: "zh-axis", "text-anchor": "middle", "font-size": "8.5" }, "OOD"));
    bindTip(go, `<b>out-of-distribution</b><br>${pct(d.ood)}` + (d.flat ? "" : `<br><span class="muted">drop ${((d.id - d.ood) * 100).toFixed(1)} pts</span>`));
    svg.appendChild(gi); svg.appendChild(go);
    container.appendChild(svg);
  }

  /* ---- line chart (learning curves) ---- */
  function lineChart(container, series, opts) {
    opts = opts || {};
    const domain = opts.domain || [0, 1], xdom = opts.xdomain || [1, 10];
    const W = container.clientWidth || 520, H = opts.height || 230;
    const mL = 44, mR = 14, mT = 12, mB = 30;
    const iw = W - mL - mR, ih = H - mT - mB;
    const X = (v) => mL + ((v - xdom[0]) / (xdom[1] - xdom[0])) * iw;
    const Y = (v) => mT + ih - ((v - domain[0]) / (domain[1] - domain[0])) * ih;
    const fmtv = opts.format || ((x) => x.toFixed(2));
    const svg = s("svg", { class: "zh-svg", viewBox: `0 0 ${W} ${H}`, width: "100%", height: H });
    (opts.yticks || [domain[0], (domain[0] + domain[1]) / 2, domain[1]]).forEach((tv) => {
      svg.appendChild(s("line", { x1: mL, x2: W - mR, y1: Y(tv), y2: Y(tv), class: "zh-grid" }));
      svg.appendChild(s("text", { x: mL - 8, y: Y(tv) + 3.5, class: "zh-axis", "text-anchor": "end" }, fmtv(tv)));
    });
    (opts.xticks || [xdom[0], (xdom[0] + xdom[1]) / 2, xdom[1]]).forEach((tv) => {
      svg.appendChild(s("text", { x: X(tv), y: H - 9, class: "zh-axis", "text-anchor": "middle" }, String(tv)));
    });
    if (opts.xlabel) svg.appendChild(s("text", { x: mL + iw / 2, y: H - 9, class: "zh-axis", "text-anchor": "middle", opacity: 0 }, opts.xlabel));
    series.forEach((se, si) => {
      const pts = se.points.filter((p) => p.y != null);
      if (!pts.length) return;
      const dPath = pts.map((p, i) => (i ? "L" : "M") + X(p.x) + " " + Y(p.y)).join(" ");
      if (se.area) {
        const area = `M${X(pts[0].x)} ${Y(domain[0])} ` + pts.map((p) => "L" + X(p.x) + " " + Y(p.y)).join(" ") + ` L${X(pts[pts.length - 1].x)} ${Y(domain[0])} Z`;
        svg.appendChild(s("path", { d: area, fill: se.color, class: "zh-area" }));
      }
      const path = s("path", { d: dPath, class: "zh-line", stroke: se.color, "stroke-dasharray": se.dash || null });
      const len = (pts.length) * 60;
      path.style.strokeDasharray = se.dash ? se.dash : len; path.style.strokeDashoffset = se.dash ? 0 : len;
      if (!se.dash) { path.style.transition = "stroke-dashoffset 1.1s var(--ease)"; requestAnimationFrame(() => requestAnimationFrame(() => { path.style.strokeDashoffset = 0; })); }
      svg.appendChild(path);
      pts.forEach((p) => {
        const dot = s("circle", { cx: X(p.x), cy: Y(p.y), r: 3, fill: se.color, class: "zh-dot" });
        bindTip(dot, `<b>${se.label}</b><br>epoch ${p.x}: <b>${fmtv(p.y)}</b>`);
        svg.appendChild(dot);
      });
    });
    container.appendChild(svg);
  }

  /* ---- wafer map ---- */
  function waferMap(container, opts) {
    opts = opts || {};
    const ring = opts.ring || 12, gap = opts.gap || 3, size = opts.size || 300;
    const die = (size - (ring + 1) * gap) / ring;
    const cx = size / 2, cy = size / 2, R = size / 2 - 2;
    const svg = s("svg", { class: "zh-svg zh-wafer", viewBox: `0 0 ${size} ${size + 8}`, width: "100%", height: size + 8 });
    svg.appendChild(s("circle", { cx, cy, r: R, fill: "rgba(255,255,255,.022)", stroke: "var(--line)", "stroke-width": 1 }));
    let total = 0, flagged = 0;
    for (let r = 0; r < ring; r++) for (let c = 0; c < ring; c++) {
      const x = gap + c * (die + gap), y = gap + r * (die + gap);
      const dcx = x + die / 2, dcy = y + die / 2;
      if (Math.hypot(dcx - cx, dcy - cy) > R - die * 0.45) continue;
      total++;
      const bad = hash(`${opts.seed || "w"}:${r}:${c}`) < (opts.rate || 0.1);
      if (bad) flagged++;
      const g = s("g");
      g.appendChild(s("rect", { x, y, width: die, height: die, rx: 1.5,
        fill: bad ? "rgba(228,23,23,.92)" : "rgba(215,221,226,.10)",
        stroke: bad ? RED : "rgba(215,221,226,.22)", "stroke-width": bad ? 1 : 0.6,
        class: "zh-die" + (bad ? " bad" : "") }));
      if (bad) g.querySelector("rect").style.filter = "drop-shadow(0 0 3px rgba(228,23,23,.6))";
      bindTip(g, bad ? `<b>die ${r}·${c}</b><br>flagged · rule violation` : `die ${r}·${c}<br>valid`);
      svg.appendChild(g);
    }
    svg.appendChild(s("line", { x1: cx - 16, y1: size + 2, x2: cx + 16, y2: size + 2, stroke: "var(--muted)", "stroke-width": 2 }));
    container.appendChild(svg);
    return { total, flagged };
  }

  /* ---- process flow ribbon ---- */
  const FLOW_COLORS = {
    clean: "#7a8893", thermal: "#d98a3a", litho: "#5b9bd5", etch: "#9b7bd4",
    doping: "#c9a23a", deposition: "#3f9b8e", planarize: "#8a93a0", via: "#4aa3b8",
    metal: "#6f9bc4", passivation: "#7fae4a", test: "#4ea36a", logistics: "#c46aa8", other: "#7a8893",
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

  return { barChart, groupedBar, gapBar, lineChart, waferMap, processFlow, classifyStep, h, s, fmt, pct, kindColor,
    colors: { RED, RED_WARM, ID, MUTED, OK } };
})();
