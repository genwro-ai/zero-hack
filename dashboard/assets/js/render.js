(function () {
  "use strict";
  const ZH = window.ZH, C = window.ZHCharts;
  if (!ZH) { console.error("ZH data missing"); return; }
  const $ = (id) => document.getElementById(id);
  const clear = (el) => { while (el && el.firstChild) el.removeChild(el.firstChild); };
  const COL = C.colors;

  /* ---------- KPI strip ---------- */
  function animateCount(el, to, fmt) {
    if (typeof requestAnimationFrame === "undefined") { el.textContent = fmt(to); return; }
    let t0 = null; const dur = 900;
    function step(now) {
      if (t0 === null) t0 = now;
      const t = Math.min(1, (now - t0) / dur);
      el.textContent = fmt(to * (1 - Math.pow(1 - t, 3)));
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  function kpi(label, value, sub, fmt, isId) {
    const card = C.h("div", { class: "kpi" + (isId ? " id" : "") });
    card.appendChild(C.h("div", { class: "k-label" }, label));
    const v = C.h("div", { class: "k-val" }, fmt(0));
    card.appendChild(v);
    card.appendChild(C.h("div", { class: "k-sub" }, sub));
    animateCount(v, value, fmt);
    return card;
  }
  function renderKPIs() {
    const box = $("kpis"); clear(box);
    const auc2 = (x) => x.toFixed(3);
    box.appendChild(kpi("ID anomaly ROC-AUC", 0.999, "GPT-ALiBi + DPO · near-perfect", auc2, true));
    box.appendChild(kpi("OOD anomaly ROC-AUC", 0.765, "held-out family · the gap", auc2, false));
    box.appendChild(kpi("Process-validity", 1.0, "decoded completions · all rule-valid", C.pct, true));
    box.appendChild(kpi("Next-step Top-1", 0.687, "GPT-ALiBi · in-distribution", C.pct, true));
  }

  /* ---------- 01 · architecture sweep ---------- */
  function renderArchitecture() {
    const rows = ZH.architectures.map((a) => ({
      label: a.label, chosen: a.chosen,
      ood_compl: a.ood.compl, ood_auc: a.ood.auc,
      sub: a.pos + " · ID compl " + C.pct(a.id.compl),
    }));
    C.groupedBar($("arch-bar"), rows, {
      series: [
        { key: "ood_compl", label: "OOD completion", color: COL.ID, highlight: true },
        { key: "ood_auc", label: "OOD anomaly AUC", color: COL.RED, dim: false },
      ],
      domain: [0, 1], labelWidth: 150, format: (x) => C.fmt(x, 3),
    });

    const box = $("arch-table"); clear(box);
    const t = C.h("table", { class: "dtable" });
    const head = C.h("tr");
    ["Architecture", "ID next", "OOD next", "ID compl", "OOD compl", "ID AUC", "OOD AUC"].forEach((c, i) =>
      head.appendChild(C.h("th", null, c)));
    t.appendChild(C.h("thead", null, head));
    const tb = C.h("tbody");
    ZH.architectures.forEach((a) => {
      const tr = C.h("tr", { class: a.chosen ? "kind-chosen" : "kind-neural" });
      tr.appendChild(C.h("td", { class: "m-name" }, a.label));
      [[a.id.next], [a.ood.next], [a.id.compl], [a.ood.compl], [a.id.auc], [a.ood.auc]].forEach(([v], i) => {
        const best = a.chosen && (i === 3); // OOD completion is the win
        tr.appendChild(C.h("td", { class: best ? "best" : "" }, C.fmt(v, 3)));
      });
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    box.appendChild(t);
  }

  /* ---------- 02 · the generalization gap ---------- */
  function renderGap() {
    const box = $("gap-rows"); clear(box);
    const order = [
      ["next", ZH.gap.next, [0.4, 1]],
      ["compl", ZH.gap.compl, [0.4, 1]],
      ["anom_auc", ZH.gap.anom_auc, [0.4, 1]],
      ["anom_f1", ZH.gap.anom_f1, [0.4, 1]],
    ];
    order.forEach(([k, d, dom]) => {
      const row = C.h("div", { class: "gap-row" });
      const name = C.h("div", { class: "gap-name" }, d.label);
      name.appendChild(C.h("span", { class: "gn-note" }, d.note));
      row.appendChild(name);
      const track = C.h("div", { class: "gap-track" });
      row.appendChild(track);
      box.appendChild(row);
      C.gapBar(track, { id: d.id, ood: d.ood, flat: d.flat }, { domain: dom });
    });
  }

  /* ---------- 03 · next-step ---------- */
  function renderNextStep() {
    const rows = ZH.models.map((m) => ({
      label: m.label, kind: m.kind, value: m.id_view.top1,
      best: m.id === "gpt", sub: m.tag,
    }));
    C.barChart($("ns-bar"), rows, {
      metricName: "Top-1 (ID)", format: C.pct, labelWidth: 150, domain: [0.5, 0.75],
      ticks: [0.5, 0.625, 0.75],
    });
    $("ns-note").textContent = "leave-one-family-out · mean of 3 holdouts";

    const box = $("ns-table"); clear(box);
    const t = C.h("table", { class: "dtable" });
    const head = C.h("tr");
    ["Model", "View", "Top-1", "Top-3", "MRR", "Exact"].forEach((c) => head.appendChild(C.h("th", null, c)));
    t.appendChild(C.h("thead", null, head));
    const tb = C.h("tbody");
    ZH.models.forEach((m, mi) => {
      [["ID", m.id_view], ["OOD", m.ood_view]].forEach(([view, v], vi) => {
        const tr = C.h("tr", { class: "kind-" + m.kind + (view === "OOD" && mi === 0 ? "" : "") });
        if (view === "OOD" && mi === 0) tr.className += " view-sep";
        tr.appendChild(C.h("td", { class: "m-name" }, view === "ID" ? m.label : ""));
        tr.appendChild(C.h("td", { class: "tag-cell" }, view));
        [["top1", C.pct], ["top3", C.pct], ["mrr", (x) => C.fmt(x, 3)], ["exact", (x) => C.fmt(x, 3)]].forEach(([key, f]) => {
          const val = v[key];
          tr.appendChild(C.h("td", { class: val == null ? "na" : "" }, f(val)));
        });
        tb.appendChild(tr);
      });
    });
    t.appendChild(tb);
    box.appendChild(t);
  }

  /* ---------- 04 · completion ---------- */
  function renderCompletion() {
    const box = $("cp-stats"); clear(box);
    const stats = [["100%", "process-validity", true], ["0.4%", "exact match", false], ["0.224", "norm. edit dist ↓", false]];
    stats.forEach(([v, l, id]) => {
      const st = C.h("div");
      st.appendChild(C.h("div", { class: "stat-v" + (id ? " id" : "") }, v));
      st.appendChild(C.h("div", { class: "stat-l" }, l));
      box.appendChild(st);
    });

    const tbox = $("cp-table"); clear(tbox);
    const data = [
      { label: "Teacher forcing", top1: 0.8100, exact: 0.0033, ned: 0.2240, tok: 0.3919, val: 1.0, best: false },
      { label: "Scheduled sampling", top1: 0.8087, exact: 0.0017, ned: 0.2199, tok: 0.3711, val: 1.0, best: true },
    ];
    const t = C.h("table", { class: "dtable" });
    const head = C.h("tr");
    ["LSTM variant", "Top-1", "Exact", "N.edit ↓", "Token-acc", "Validity"].forEach((c) => head.appendChild(C.h("th", null, c)));
    t.appendChild(C.h("thead", null, head));
    const tb = C.h("tbody");
    data.forEach((d) => {
      const tr = C.h("tr", { class: "kind-neural" });
      tr.appendChild(C.h("td", { class: "m-name" }, d.label));
      tr.appendChild(C.h("td", null, C.pct(d.top1)));
      tr.appendChild(C.h("td", null, C.fmt(d.exact, 3)));
      tr.appendChild(C.h("td", { class: d.best ? "best" : "" }, C.fmt(d.ned, 3)));
      tr.appendChild(C.h("td", null, C.pct(d.tok)));
      tr.appendChild(C.h("td", { class: "best" }, C.pct(d.val)));
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    tbox.appendChild(t);
  }

  /* ---------- 05 · DPO ---------- */
  function renderDPO() {
    const d = ZH.dpo;
    C.barChart($("dpo-bar"), [
      { label: "SFT · ROC-AUC", value: d.id.sft.auc, color: COL.MUTED },
      { label: "+DPO · ROC-AUC", value: d.id.dpo.auc, color: COL.RED, best: true },
      { label: "SFT · F1", value: d.id.sft.f1, color: COL.MUTED },
      { label: "+DPO · F1", value: d.id.dpo.f1, color: COL.RED, best: true },
    ], { metricName: "in-distribution", format: (x) => C.fmt(x, 3), labelWidth: 132, domain: [0.9, 1.0], ticks: [0.9, 0.95, 1.0] });

    const box = $("dpo-table"); clear(box);
    const t = C.h("table", { class: "dtable" });
    const head = C.h("tr");
    ["Policy", "View", "Top-1", "F1", "ROC-AUC", "Token-acc"].forEach((c) => head.appendChild(C.h("th", null, c)));
    t.appendChild(C.h("thead", null, head));
    const tb = C.h("tbody");
    [["ID", "id"], ["OOD", "ood"]].forEach(([view, key], vi) => {
      [["SFT (augmented)", "sft", "kind-neural"], ["+ DPO", "dpo", "kind-chosen"]].forEach(([lbl, pk, cls], pi) => {
        const v = d[key][pk];
        const tr = C.h("tr", { class: cls + (vi === 1 && pi === 0 ? " view-sep" : "") });
        tr.appendChild(C.h("td", { class: "m-name" }, lbl));
        tr.appendChild(C.h("td", { class: "tag-cell" }, view));
        const isBest = view === "ID" && pk === "dpo";
        tr.appendChild(C.h("td", null, C.pct(v.top1)));
        tr.appendChild(C.h("td", { class: isBest ? "best" : "" }, C.fmt(v.f1, 3)));
        tr.appendChild(C.h("td", { class: isBest ? "best" : "" }, C.fmt(v.auc, 3)));
        tr.appendChild(C.h("td", null, C.pct(v.tok)));
        tb.appendChild(tr);
      });
    });
    t.appendChild(tb);
    box.appendChild(t);
  }

  /* ---------- 06 · learning curves ---------- */
  function renderCurves() {
    const tf = ZH.curves.teacher_forcing, ss = ZH.curves.scheduled_sampling;
    C.lineChart($("curve-top1"), [
      { label: "teacher forcing", color: COL.RED, points: tf.map((p) => ({ x: p.e, y: p.top1 })) },
      { label: "scheduled sampling", color: COL.ID, dash: "5 4", points: ss.map((p) => ({ x: p.e, y: p.top1 })) },
    ], { domain: [0.80, 0.83], yticks: [0.80, 0.81, 0.82, 0.83], xdomain: [1, 10], xticks: [1, 5, 10], height: 230, format: (x) => x.toFixed(3) });

    C.lineChart($("curve-loss"), [
      { label: "train loss", color: COL.RED, area: true, points: tf.map((p) => ({ x: p.e, y: p.train })) },
      { label: "validation loss", color: COL.ID, dash: "5 4", points: tf.map((p) => ({ x: p.e, y: p.valid })) },
    ], { domain: [0.30, 0.50], yticks: [0.30, 0.40, 0.50], xdomain: [1, 10], xticks: [1, 5, 10], height: 230, format: (x) => x.toFixed(2) });
  }

  /* ---------- hero ---------- */
  function renderHero() {
    const wbox = $("hero-wafer"); if (wbox) { clear(wbox); C.waferMap(wbox, { rate: 0.10, seed: "ic", ring: 13, size: 300 }); }
    const fbox = $("hero-flow"); if (fbox) { clear(fbox); C.processFlow(fbox, ZH.flow); }
    const sv = $("spec-vocab"); if (sv) sv.textContent = "≈" + ZH.meta.vocab;
    const sp = $("spec-params"); if (sp) sp.textContent = ZH.meta.params;
    const team = $("team"); if (team) { clear(team); ZH.meta.team.forEach((t) => team.appendChild(C.h("span", null, t))); }
  }

  /* ---------- 08 · examples ---------- */
  function renderExamples() {
    // next-step
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
      card.appendChild(C.h("div", { class: "p-note", style: "margin-top:10px", html: "gold next step: <b style='color:var(--red-warm)'>" + ns.gold + "</b>" }));
      nbox.appendChild(card);
    }

    // completion
    const cp = ZH.examples.completion;
    const cbox = $("ex-completion"); clear(cbox);
    {
      const card = C.h("div", { class: "example" });
      const meta = C.h("div", { class: "ex-meta" });
      meta.appendChild(C.h("span", { class: "tag fam" }, cp.family.toUpperCase()));
      meta.appendChild(C.h("span", { class: "tag" }, cp.example_id));
      meta.appendChild(C.h("span", { class: "tag ok" }, "✓ process-valid"));
      meta.appendChild(C.h("span", { class: "tag" }, "exact-match 0"));
      card.appendChild(meta);
      const fb = C.h("div", { style: "margin-bottom:10px" }); C.processFlow(fb, cp.context, { compact: true }); card.appendChild(fb);
      const vs = C.h("div", { class: "vs" });
      [["base", "Reference (gold)", cp.gold], ["train", "Decoder · GPT-ALiBi", cp.trained]].forEach(([cls, title, seq]) => {
        const col = C.h("div", { class: "vs-col " + cls });
        col.appendChild(C.h("h4", null, title));
        const sd = C.h("div", { class: "seq" });
        seq.forEach((step, i) => {
          const mismatch = cls === "train" && cp.gold[i] !== step;
          sd.appendChild(C.h("span", { class: "step" + (mismatch ? " bad" : "") }, step));
        });
        col.appendChild(sd);
        vs.appendChild(col);
      });
      card.appendChild(vs);
      card.appendChild(C.h("div", { class: "p-note", style: "margin-top:12px;line-height:1.6" }, cp.note));
      cbox.appendChild(card);
    }

    // anomaly
    const an = ZH.examples.anomaly;
    const abox = $("ex-anomaly"); clear(abox);
    {
      const card = C.h("div", { class: "example" });
      const meta = C.h("div", { class: "ex-meta" });
      meta.appendChild(C.h("span", { class: "tag fam" }, an.family.toUpperCase()));
      meta.appendChild(C.h("span", { class: "tag rule" }, "violates " + an.rule));
      meta.appendChild(C.h("span", { class: "tag" }, an.example_id));
      card.appendChild(meta);
      card.appendChild(C.h("p", { class: "p-note", style: "margin:0 0 12px;line-height:1.6" }, an.description));
      const fb = C.h("div", { style: "margin-bottom:14px" });
      C.processFlow(fb, an.sequence.map((st, i) => ({ label: st, flag: i === an.violation_index })), { compact: true });
      card.appendChild(fb);
      const d = an.detector;
      card.appendChild(C.h("div", { class: "verdict flag" },
        "⚑ flagged ANOMALY · " + d.method + " · score " + C.fmt(d.score, 2) + " (below threshold)"));
      card.appendChild(C.h("div", { class: "p-note", style: "margin-top:8px", html: "attributed rule: <b style='color:var(--red-warm)'>" + d.predicted_rule + "</b> — correct" }));
      abox.appendChild(card);
    }
  }

  /* ---------- init ---------- */
  function init() {
    renderHero();
    renderKPIs();
    renderArchitecture();
    renderGap();
    renderNextStep();
    renderCompletion();
    renderDPO();
    renderCurves();
    renderExamples();

    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { threshold: 0.06 });
    document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

    const navLinks = [...document.querySelectorAll(".topbar nav a")];
    const navIO = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) {
        const id = e.target.id; navLinks.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + id));
      } });
    }, { rootMargin: "-40% 0px -55% 0px" });
    document.querySelectorAll("section[id], header.masthead").forEach((sec) => navIO.observe(sec));

    let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => {
      renderHero(); renderArchitecture(); renderGap(); renderNextStep(); renderDPO(); renderCurves();
    }, 220); });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
