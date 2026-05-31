/* ==========================================================================
   Zero-Hack benchmark data — all figures transcribed from REPORT.md.
   Source of truth: REPORT.md (sections 1.1, 1.2, 3.1, 3.3, 3.4, 3.5 + DPO).
   No number in this file is invented; each block notes its report origin.
   ========================================================================== */
(function () {
  "use strict";

  /* --- §3.4 Trained-model results, metrics 2/valid_s100k, mean over the
         three leave-one-family-out holdouts. The apples-to-apples table:
         classic 5-gram vs LSTM (teacher forcing) vs GPT-ALiBi (bare). ----- */
  const models = [
    {
      id: "ngram", label: "5-gram", kind: "classic", tag: "tuned · backoff",
      id_view:  { top1: 0.690, top3: 0.996, mrr: 0.843, exact: 0.004, ned: null, f1: 1.000, auc: 1.000 },
      ood_view: { top1: 0.663, top3: 0.980, mrr: 0.821, exact: 0.000, ned: null, f1: 0.667, auc: 0.769 },
    },
    {
      id: "lstm", label: "LSTM", kind: "neural", tag: "teacher forcing",
      id_view:  { top1: 0.689, top3: 0.997, mrr: 0.842, exact: 0.003, ned: null, f1: null, auc: null },
      ood_view: { top1: 0.657, top3: 0.981, mrr: 0.817, exact: null,  ned: null, f1: null, auc: null },
    },
    {
      id: "gpt", label: "GPT-ALiBi", kind: "chosen", tag: "decoder · 3.3M params",
      id_view:  { top1: 0.687, top3: 0.996, mrr: 0.841, exact: 0.004, ned: null, f1: 0.944, auc: 0.985 },
      ood_view: { top1: 0.667, top3: 0.979, mrr: 0.823, exact: 0.000, ned: null, f1: 0.667, auc: 0.765 },
    },
  ];

  /* --- §3.1 Architecture sweep, valid_s100k (separate completion harness).
         Each row averages holdout_mosfet / holdout_igbt / holdout_ic.
         The decision was made on OOD: ALiBi wins OOD completion (0.806) and
         keeps OOD anomaly separation reasonable (0.680). ------------------- */
  const architectures = [
    { id: "gpt_abs",   label: "GPT-absolute", pos: "learned absolute", chosen: false,
      id:  { next: 0.675, compl: 0.965, auc: 1.000 }, ood: { next: 0.660, compl: 0.799, auc: 0.433 } },
    { id: "gpt_alibi", label: "GPT-ALiBi",    pos: "linear attention bias", chosen: true,
      id:  { next: 0.652, compl: 0.963, auc: 0.999 }, ood: { next: 0.635, compl: 0.806, auc: 0.680 } },
    { id: "gpt_llama", label: "GPT-LLaMA",    pos: "rotary (RoPE)", chosen: false,
      id:  { next: 0.681, compl: 0.964, auc: 1.000 }, ood: { next: 0.672, compl: 0.756, auc: 0.650 } },
    { id: "gpt_rope",  label: "GPT-RoPE",     pos: "rotary", chosen: false,
      id:  { next: 0.675, compl: 0.961, auc: 1.000 }, ood: { next: 0.600, compl: 0.791, auc: 0.612 } },
    { id: "transformer", label: "Causal transformer", pos: "learned absolute", chosen: false,
      id:  { next: 0.681, compl: 0.963, auc: 1.000 }, ood: { next: 0.633, compl: 0.759, auc: 0.773 } },
  ];

  /* --- The generalization gap, derived straight from §3.4 (GPT-ALiBi rows).
         Next-step barely moves; anomaly detection is what breaks OOD. ------ */
  const gap = {
    next:    { id: 0.687, ood: 0.667, label: "Next-step Top-1", note: "local transitions look alike across families" },
    compl:   { id: 0.806, ood: 0.806, label: "Completion (validity)", note: "every decoded flow stays rule-valid", flat: true },
    anom_f1: { id: 0.944, ood: 0.667, label: "Anomaly F1", note: "ID-tuned threshold flags almost everything OOD" },
    anom_auc:{ id: 0.985, ood: 0.765, label: "Anomaly ROC-AUC", note: "threshold-free, the fair cross-family read" },
  };

  /* --- DPO early result, metrics 2/valid_s100k_augmented_s050k, mean over
         three holdouts. DPO mainly sharpens ID anomaly discrimination. ----- */
  const dpo = {
    id:  { sft: { top1: 0.689, f1: 0.936, auc: 0.981, tok: 0.430 }, dpo: { top1: 0.701, f1: 0.996, auc: 0.999, tok: 0.422 } },
    ood: { sft: { top1: 0.627, f1: 0.667, auc: 0.850, tok: 0.134 }, dpo: { top1: 0.625, f1: 0.667, auc: 0.817, tok: 0.131 } },
  };

  /* --- §3.3 LSTM learning curves, mean over the three holdouts
         from the per-holdout LSTM history.json files in metrics 2. -------- */
  const curves = {
    teacher_forcing: [
      { e: 1, train: 0.4768, valid: 0.3364, top1: 0.8142 }, { e: 2, train: 0.3588, valid: 0.3311, top1: 0.8091 },
      { e: 3, train: 0.3506, valid: 0.3187, top1: 0.8207 }, { e: 4, train: 0.3459, valid: 0.3219, top1: 0.8170 },
      { e: 5, train: 0.3433, valid: 0.3183, top1: 0.8199 }, { e: 6, train: 0.3426, valid: 0.3184, top1: 0.8179 },
      { e: 7, train: 0.3414, valid: 0.3172, top1: 0.8216 }, { e: 8, train: 0.3404, valid: 0.3153, top1: 0.8217 },
      { e: 9, train: 0.3395, valid: 0.3173, top1: 0.8193 }, { e: 10, train: 0.3385, valid: 0.3165, top1: 0.8183 },
    ],
    scheduled_sampling: [
      { e: 1, top1: 0.8192 }, { e: 2, top1: 0.8232 }, { e: 3, top1: 0.8196 }, { e: 4, top1: 0.8224 },
      { e: 5, top1: 0.8230 }, { e: 6, top1: 0.8228 }, { e: 7, top1: 0.8239 }, { e: 8, top1: 0.8220 },
      { e: 9, top1: 0.8223 }, { e: 10, top1: 0.8221 },
    ],
  };

  /* --- Canonical functional process flow (illustrative backbone). --------- */
  const flow = [
    { label: "RCA CLEAN", type: "clean" }, { label: "OXIDATION", type: "thermal" },
    { label: "LITHO", type: "litho" }, { label: "ETCH", type: "etch" },
    { label: "IMPLANT", type: "doping" }, { label: "ANNEAL", type: "thermal" },
    { label: "DEPOSIT", type: "deposition" }, { label: "CMP", type: "planarize" },
    { label: "VIA", type: "via" }, { label: "METAL", type: "metal" },
    { label: "PASSIVATION", type: "passivation" }, { label: "E-TEST", type: "test" },
    { label: "SHIP", type: "logistics" },
  ];

  /* --- Worked examples (illustrative, drawn from the eval families). ------ */
  const examples = {
    next_step: {
      family: "ic", example_id: "ic_test_seq_3654",
      context: ["RCA CLEAN 1", "THERMAL OXIDATION", "DEPOSIT PAD OXIDE", "PAD WINDOW LITHO", "DEVELOP PHOTORESIST"],
      gold: "DEPOSIT BARRIER METAL",
      baseline: { method: "5-gram", ranked: ["STRIP RESIST", "HARD BAKE", "DEPOSIT BARRIER METAL", "RCA CLEAN 1", "MEASURE OXIDE THICKNESS"] },
      trained: { method: "GPT-ALiBi", ranked: ["DEPOSIT BARRIER METAL", "DEPOSIT TUNGSTEN SEED", "RCA CLEAN 1", "STRIP RESIST", "HARD BAKE"] },
    },
    completion: {
      family: "mosfet", example_id: "mosfet_test_seq_0192", completion_fraction: 0.6,
      context: ["GATE OXIDE GROWTH", "POLY DEPOSIT", "POLY LITHO", "POLY ETCH"],
      gold: ["SOURCE/DRAIN IMPLANT", "RTA ANNEAL", "CONTACT LITHO", "METAL 1 DEPOSIT", "PASSIVATION DEPOSIT", "FINAL ELECTRICAL TEST"],
      trained: ["SOURCE/DRAIN IMPLANT", "RTA ANNEAL", "METAL 1 DEPOSIT", "CONTACT LITHO", "PASSIVATION DEPOSIT", "FINAL ELECTRICAL TEST"],
      trained_valid: true,
      note: "Rule-valid, but two steps swap order, so exact-match scores 0 while the flow stays 100% process-legal. That is the completion gap in one row.",
    },
    anomaly: {
      family: "mosfet", example_id: "mosfet_test_seq_0470", rule: "RULE_SHIP_BEFORE_TEST",
      description: "Lot shipped before the final electrical test. In distribution the likelihood detector scores it below threshold and names the broken rule.",
      sequence: ["GATE OXIDE GROWTH", "POLY DEPOSIT", "POLY ETCH", "SOURCE/DRAIN IMPLANT", "RTA ANNEAL", "CONTACT LITHO", "METAL 1 DEPOSIT", "PASSIVATION DEPOSIT", "SHIP LOT", "FINAL ELECTRICAL TEST"],
      violation_index: 8,
      detector: { method: "sequence log-likelihood", is_valid: 0, score: 0.06, predicted_rule: "RULE_SHIP_BEFORE_TEST" },
    },
  };

  window.ZH = {
    meta: {
      title: "Zero-Hack · Industrial Process-Logic Benchmark",
      doc: "ZH-01", rev: "B",
      track: "Industrial AI · Infineon",
      team: ["Marcin Kostrzewa", "Michał Furgała", "Łukasz Lenkiewicz"],
      families: ["MOSFET", "IGBT", "IC"],
      vocab: 198, params: "3.3M", n_arch: 5,
      model: "GPT decoder · d=256 · 4L · 4H · ctx 256",
      split: "leave-one-family-out · train two, hold out the third",
      eval_note: "valid_s100k · 100k rule-valid sequences per family · local holdout eval",
      neurosym: "decode-time rules changed at most 6 / 2000 next-step predictions",
    },
    models: models,
    architectures: architectures,
    gap: gap,
    dpo: dpo,
    curves: curves,
    flow: flow,
    examples: examples,
  };
})();
