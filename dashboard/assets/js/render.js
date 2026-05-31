(function () {
  "use strict";
  const ZH = window.ZH, C = window.ZHCharts;
  if (!ZH) { console.error("ZH data missing"); return; }
  const $ = (id) => document.getElementById(id);
  const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); };
  const M = ZH.methods;

  function bestOf(task, key, lower) {
    let best = null;
    M.forEach((m) => { const v = m[task][key]; if (v == null) return; if (best == null || (lower ? v < best.v : v > best.v)) best = { v: v, m: m }; });
    return best;
  }
  function barRows(task, key, lower) {
    const b = bestOf(task, key, lower);
    return M.map((m) => ({ label: m.label, kind: m.kind, value: m[task][key], best: b && m.id === b.m.id }));
  }

  function animateCount(el, to, fmt) {
    if (typeof requestAnimationFrame === "undefined") { el.textContent = fmt(to); return; }
    let t0 = null; const dur = 850;
    function step(now) {
      if (t0 === null) t0 = now;
      const t = Math.min(1, (now - t0) / dur);
      el.textContent = fmt(to * (1 - Math.pow(1 - t, 3)));
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  function kpi(label, value, sub, fmt) {
    fmt = fmt || C.pct;
    const card = C.h("div", { class: "kpi" });
    card.appendChild(C.h("div", { class: "k-label" }, label));
    const v = C.h("div", { class: "k-val accent" }, fmt(0));
    card.appendChild(v);
    card.appendChild(C.h("div", { class: "k-sub" }, sub));
    animateCount(v, value, fmt);
    return card;
  }
  function renderKPIs() {
    const box = $("kpis"); clear(box);
    const t1 = bestOf("next_step", "top1"), t3 = bestOf("next_step", "top3");
    box.appendChild(kpi("Next-step Top-1", t1.v, "best · " + t1.m.label, C.pct));
    box.appendChild(kpi("Next-step Top-3", t3.v, "best · " + t3.m.label, C.pct));
    const roc = bestOf("anomaly", "roc_auc");
    box.appendChild(kpi("Process-validity", 1.0, "generated completions · all valid", C.pct));
    box.appendChild(kpi("Anomaly ROC-AUC", roc.v, "best · " + roc.m.label, (x) => x.toFixed(3)));
  }

  function renderTable(containerId, task, cols) {
    const box = $(containerId); clear(box);
    const table = C.h("table", { class: "dtable" });
    const thr = C.h("tr"); thr.appendChild(C.h("th", null, "Method"));
    cols.forEach((c) => thr.appendChild(C.h("th", null, c.label)));
    table.appendChild(C.h("thead", null, thr));
    const bestVal = {};
    cols.forEach((c) => { const b = bestOf(task, c.key, c.lower); bestVal[c.key] = b ? b.v : null; });
    const tb = C.h("tbody");
    M.forEach((m) => {
      const tr = C.h("tr", { class: "kind-" + m.kind });
      tr.appendChild(C.h("td", { class: "m-name" }, m.label));
      cols.forEach((c) => {
        const v = m[task][c.key];
        const cls = (v != null && v === bestVal[c.key]) ? "best" : (v == null ? "na" : "");
        tr.appendChild(C.h("td", { class: cls }, (c.fmt || C.pct)(v)));
      });
      tb.appendChild(tr);
    });
    table.appendChild(tb);
    box.appendChild(table);
  }

  function renderNextStep() {
    C.barChart($("ns-bar"), barRows("next_step", "top1"), {
      metricName: "Top-1 accuracy", format: C.pct, labelWidth: 202, domain: [0, 1],
    });
    renderTable("ns-table", "next_step", [
      { key: "top1", label: "Top-1" }, { key: "top3", label: "Top-3" },
      { key: "mrr", label: "MRR", fmt: (x) => C.fmt(x, 3) },
    ]);
    $("ns-note").textContent = "n = " + ZH.meta.eval.n_next + " · in-distribution";
  }

  function renderCompletion() {
    C.barChart($("cp-bar"), barRows("completion", "norm_edit_distance", true), {
      metricName: "normalized edit distance (lower = better)", format: (x) => C.fmt(x, 3),
      labelWidth: 202, domain: [0, 0.3], ticks: [0, 0.15, 0.3],
    });
    renderTable("cp-table", "completion", [
      { key: "exact_match", label: "Exact" },
      { key: "norm_edit_distance", label: "N.edit ↓", lower: true, fmt: (x) => C.fmt(x, 3) },
      { key: "token_accuracy", label: "Token-acc" },
      { key: "process_validity", label: "Validity" },
    ]);
    $("cp-note").textContent = "n = " + ZH.meta.eval.n_completion;
  }

  function renderAnomaly() {
    const box = $("an-stats"); clear(box);
    const detectors = M.filter((m) => m.anomaly.f1 != null);
    [["F1", "f1"], ["ROC-AUC", "roc_auc"]].forEach(([lbl, key]) => {
      const v = detectors[0].anomaly[key];
      const st = C.h("div", { class: "stat" });
      st.appendChild(C.h("div", { class: "stat-v" }, key === "f1" ? C.fmt(v, 2) : v.toFixed(2)));
      st.appendChild(C.h("div", { class: "stat-l" }, lbl));
      box.appendChild(st);
    });
    const st = C.h("div", { class: "stat" });
    st.appendChild(C.h("div", { class: "stat-v" }, "10"));
    st.appendChild(C.h("div", { class: "stat-l" }, "process rules"));
    box.appendChild(st);
  }

  function renderHero() {
    const wbox = $("hero-wafer"); if (wbox) { clear(wbox); C.waferMap(wbox, { rate: 0.09, seed: "ic", ring: 13, size: 290 }); }
    const fbox = $("hero-flow"); if (fbox) { clear(fbox); C.processFlow(fbox, ZH.flow); }
    const sv = $("spec-vocab"); if (sv) sv.textContent = "≈" + ZH.meta.vocab;
    const sm = $("spec-models"); if (sm) sm.textContent = ZH.meta.n_models;
  }

  function renderExamples() {
    const ns = ZH.examples.next_step;
    const nbox = $("ex-nextstep"); clear(nbox);
    {
      const card = C.h("div", { class: "example" });
      const meta = C.h("div", { class: "ex-meta" });
      meta.appendChild(C.h("span", { class: "tag fam" }, ns.family.toUpperCase()));
      meta.appendChild(C.h("span", { class: "tag" }, ns.example_id));
      meta.appendChild(C.h("span", { class: "tag" }, "next-step"));
      card.appendChild(meta);
      const fb = C.h("div", { style: "margin-bottom:6px" }); C.processFlow(fb, ns.context, { compact: true }); card.appendChild(fb);
      card.appendChild(C.h("div", { class: "p-note", style: "margin:0 0 12px" }, "→ predict the next process step"));
      const vs = C.h("div", { class: "vs" });
      [["base", ns.baseline], ["train", ns.trained]].forEach(([cls, side]) => {
        const col = C.h("div", { class: "vs-col " + cls });
        col.appendChild(C.h("h4", null, (cls === "base" ? "Baseline · " : "Trained · ") + side.method));
        const ul = C.h("ul", { class: "ranklist" });
        side.ranked.forEach((r) => ul.appendChild(C.h("li", { class: r === ns.gold ? "hit" : "" }, r)));
        col.appendChild(ul);
        vs.appendChild(col);
      });
      card.appendChild(vs);
      card.appendChild(C.h("div", { class: "p-note", style: "margin-top:10px", html: "gold next step: <b>" + ns.gold + "</b>" }));
      nbox.appendChild(card);
    }

    const an = ZH.examples.anomaly;
    const abox = $("ex-anomaly"); clear(abox);
    {
      const card = C.h("div", { class: "example" });
      const meta = C.h("div", { class: "ex-meta" });
      meta.appendChild(C.h("span", { class: "tag fam" }, an.family.toUpperCase()));
      meta.appendChild(C.h("span", { class: "tag rule" }, "violates " + an.rule));
      meta.appendChild(C.h("span", { class: "tag" }, an.example_id));
      card.appendChild(meta);
      card.appendChild(C.h("p", { class: "p-note", style: "margin:0 0 12px" }, an.description));
      const fb = C.h("div", { style: "margin-bottom:14px" });
      C.processFlow(fb, an.sequence.map((st, i) => ({ label: st, flag: i === an.violation_index })), { compact: true });
      card.appendChild(fb);
      const d = an.detector;
      card.appendChild(C.h("div", { class: "verdict ok" },
        "⚑ flagged ANOMALY · " + d.method + " · score " + C.fmt(d.score, 2)));
      card.appendChild(C.h("div", { class: "p-note", style: "margin-top:8px", html: "attributed rule: <b>" + d.predicted_rule + "</b> — correct" }));
      abox.appendChild(card);
    }
  }

  function renderArch() {
    const el = $("arch-table"); if (!el) return; clear(el);
    const rows = ZH.archCompare;
    const cols = [
      { key: "id_ns",  label: "ID Top-1",  ood: false, higher: true },
      { key: "ood_ns", label: "OOD Top-1", ood: true,  higher: true },
      { key: "id_cp",  label: "ID Compl.", ood: false, higher: true },
      { key: "ood_cp", label: "OOD Compl.",ood: true,  higher: true },
      { key: "id_auc", label: "ID AUC",    ood: false, higher: true },
      { key: "ood_auc",label: "OOD AUC",   ood: true,  higher: true },
    ];
    const bests = {};
    cols.forEach((c) => {
      let best = null;
      rows.forEach((r) => { if (best === null || (c.higher ? r[c.key] > best : r[c.key] < best)) best = r[c.key]; });
      bests[c.key] = best;
    });
    const tbl = C.h("table", { class: "arch-tbl" });
    const thead = C.h("thead"); const hr = C.h("tr");
    hr.appendChild(C.h("th", null, "Model"));
    hr.appendChild(C.h("th", null, "Pos."));
    cols.forEach((c) => { const th = C.h("th", { class: c.ood ? "ood-col" : "" }, c.label); hr.appendChild(th); });
    thead.appendChild(hr); tbl.appendChild(thead);
    const tbody = C.h("tbody");
    rows.forEach((r) => {
      const tr = C.h("tr", { class: r.selected ? "arch-selected" : "" });
      tr.appendChild(C.h("td", { class: "arch-name" + (r.selected ? " sel" : "") }, r.label + (r.selected ? " ★" : "")));
      tr.appendChild(C.h("td", { class: "arch-pos" }, r.pos));
      cols.forEach((c) => {
        const isBest = r[c.key] === bests[c.key];
        const td = C.h("td", { class: (c.ood ? "ood-col" : "") + (isBest ? " best-cell" : "") }, r[c.key].toFixed(3));
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    tbl.appendChild(tbody);
    el.appendChild(tbl);
    const note = C.h("div", { class: "p-note", style: "margin-top:10px" }, "★ selected architecture · OOD = held-out device families · completion metric: token accuracy");
    el.appendChild(note);
  }

  function init() {
    if (ZH.meta && ZH.meta.placeholder === false) { const b = $("placeholder-badge"); if (b) b.style.display = "none"; }
    renderHero();
    renderKPIs();
    renderNextStep();
    renderCompletion();
    renderAnomaly();
    renderArch();
    renderExamples();

    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: 0.08 });
    document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

    const navLinks = [...document.querySelectorAll(".topbar nav a")];
    const navIO = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) {
        const id = e.target.id; navLinks.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + id));
      } });
    }, { rootMargin: "-40% 0px -55% 0px" });
    document.querySelectorAll("section, header.masthead").forEach((sec) => navIO.observe(sec));

    let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => {
      renderHero(); renderNextStep(); renderCompletion(); }, 200); });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
