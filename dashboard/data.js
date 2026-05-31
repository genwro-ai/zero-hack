(function () {
  "use strict";

  const methods = [
    {
      id: "ngram", label: "5-gram (tuned)", kind: "baseline",
      next_step: { top1: 0.6903, top3: 0.9958, mrr: 0.8428 },
      completion: { exact_match: 0.0042, norm_edit_distance: 0.2272, token_accuracy: 0.4308, process_validity: null },
      anomaly: { f1: 0.9997, roc_auc: 0.9999 },
    },
    {
      id: "gpt_alibi", label: "GPT-ALiBi · DPO", kind: "neural",
      next_step: { top1: 0.7014, top3: 0.9969, mrr: 0.8484 },
      completion: { exact_match: 0.0029, norm_edit_distance: 0.2240, token_accuracy: 0.4219, process_validity: null },
      anomaly: { f1: 0.9963, roc_auc: 0.9995 },
    },
  ];

  const flow = [
    { label: "RCA CLEAN", type: "clean" },
    { label: "OXIDATION", type: "thermal" },
    { label: "LITHO", type: "litho" },
    { label: "ETCH", type: "etch" },
    { label: "IMPLANT", type: "doping" },
    { label: "ANNEAL", type: "thermal" },
    { label: "DEPOSIT", type: "deposition" },
    { label: "CMP", type: "planarize" },
    { label: "VIA", type: "via" },
    { label: "METAL", type: "metal" },
    { label: "PASSIVATION", type: "passivation" },
    { label: "E-TEST", type: "test" },
    { label: "SHIP", type: "logistics" },
  ];

  const examples = {
    next_step: {
      family: "ic",
      example_id: "ic_test_seq_3654",
      context: ["RCA CLEAN 1", "THERMAL OXIDATION", "DEPOSIT PAD OXIDE", "PAD WINDOW LITHO", "DEVELOP PHOTORESIST"],
      gold: "DEPOSIT BARRIER METAL",
      baseline: { method: "5-gram (tuned)", ranked: ["STRIP RESIST", "HARD BAKE", "DEPOSIT BARRIER METAL", "RCA CLEAN 1", "MEASURE OXIDE THICKNESS"] },
      trained: { method: "GPT-ALiBi · DPO", ranked: ["DEPOSIT BARRIER METAL", "DEPOSIT TUNGSTEN SEED", "RCA CLEAN 1", "STRIP RESIST", "HARD BAKE"] },
    },
    completion: {
      family: "mosfet",
      example_id: "mosfet_test_seq_0192",
      completion_fraction: 0.6,
      context: ["GATE OXIDE GROWTH", "POLY DEPOSIT", "POLY LITHO", "POLY ETCH"],
      gold: ["SOURCE/DRAIN IMPLANT", "RTA ANNEAL", "CONTACT LITHO", "METAL 1 DEPOSIT", "PASSIVATION DEPOSIT", "FINAL ELECTRICAL TEST"],
      trained: ["SOURCE/DRAIN IMPLANT", "RTA ANNEAL", "METAL 1 DEPOSIT", "CONTACT LITHO", "PASSIVATION DEPOSIT", "FINAL ELECTRICAL TEST"],
      trained_valid: true,
      note: "Rule-valid, but two steps are swapped → exact-match 0 while staying 100% process-legal.",
    },
    anomaly: {
      family: "mosfet",
      example_id: "mosfet_test_seq_0470",
      rule: "RULE_SHIP_BEFORE_TEST",
      description: "Lot shipped before the final electrical test — the detector scores it below threshold and names the rule.",
      sequence: ["GATE OXIDE GROWTH", "POLY DEPOSIT", "POLY ETCH", "SOURCE/DRAIN IMPLANT", "RTA ANNEAL", "CONTACT LITHO", "METAL 1 DEPOSIT", "PASSIVATION DEPOSIT", "SHIP LOT", "FINAL ELECTRICAL TEST"],
      violation_index: 8,
      detector: { method: "5-gram log-likelihood", is_valid: 0, score: 0.06, predicted_rule: "RULE_SHIP_BEFORE_TEST" },
    },
  };

  const archCompare = [
    { id: "gpt_abs",   label: "GPT-absolute",    pos: "absolute",  selected: false, id_ns: 0.675, ood_ns: 0.660, id_cp: 0.965, ood_cp: 0.799, id_auc: 1.000, ood_auc: 0.433 },
    { id: "gpt_alibi", label: "GPT-ALiBi",        pos: "ALiBi",     selected: true,  id_ns: 0.652, ood_ns: 0.635, id_cp: 0.963, ood_cp: 0.806, id_auc: 0.999, ood_auc: 0.680 },
    { id: "gpt_llama", label: "GPT-LLaMA-style",  pos: "RoPE",      selected: false, id_ns: 0.681, ood_ns: 0.672, id_cp: 0.964, ood_cp: 0.756, id_auc: 1.000, ood_auc: 0.650 },
    { id: "gpt_rope",  label: "GPT-RoPE",         pos: "RoPE",      selected: false, id_ns: 0.675, ood_ns: 0.600, id_cp: 0.961, ood_cp: 0.791, id_auc: 1.000, ood_auc: 0.612 },
    { id: "causal",    label: "Causal transformer",pos: "absolute",  selected: false, id_ns: 0.681, ood_ns: 0.633, id_cp: 0.963, ood_cp: 0.759, id_auc: 1.000, ood_auc: 0.773 },
  ];

  window.ZH = {
    meta: {
      title: "Zero-Hack · Industrial Process-Logic Benchmark",
      doc: "ZH-01",
      rev: "A",
      track: "Industrial-AI / Process-Logic · Infineon",
      families: ["IC", "MOSFET", "IGBT"],
      vocab: 198,
      n_models: 2,
      placeholder: false,
      eval: {
        condition: "in-distribution · all 3 families · 5k sequences / family",
        n_next: 600, n_completion: 600, n_anomaly: 300,
        note: "Local holdout eval — official organizer scorer pending.",
      },
    },
    methods: methods,
    flow: flow,
    examples: examples,
    archCompare: archCompare,
  };
})();
