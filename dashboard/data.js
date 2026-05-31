(function () {
  "use strict";

  const methods = [
    {
      id: "most_frequent", label: "Most-frequent", kind: "baseline",
      next_step: { top1: 0.7067, top3: 0.9950, mrr: 0.8501 },
      completion: { exact_match: 0.0017, norm_edit_distance: 0.2458, token_accuracy: null, process_validity: null },
      anomaly: { f1: 1.000, roc_auc: 1.000 },
    },
    {
      id: "ngram", label: "5-gram (backoff)", kind: "baseline",
      next_step: { top1: 0.7133, top3: 0.9967, mrr: 0.8536 },
      completion: { exact_match: 0.0050, norm_edit_distance: 0.2243, token_accuracy: null, process_validity: null },
      anomaly: { f1: 1.000, roc_auc: 1.000 },
    },
    {
      id: "lstm_tf", label: "LSTM · teacher-forcing", kind: "neural",
      next_step: { top1: 0.8100, top3: 0.9954, mrr: null },
      completion: { exact_match: 0.0033, norm_edit_distance: 0.2240, token_accuracy: 0.3919, process_validity: 1.000 },
      anomaly: { f1: null, roc_auc: null },
    },
    {
      id: "lstm_ss", label: "LSTM · scheduled-sampling", kind: "neural",
      next_step: { top1: 0.8087, top3: 0.9949, mrr: null },
      completion: { exact_match: 0.0017, norm_edit_distance: 0.2199, token_accuracy: 0.3711, process_validity: 1.000 },
      anomaly: { f1: null, roc_auc: null },
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
      baseline: { method: "5-gram (backoff)", ranked: ["STRIP RESIST", "HARD BAKE", "DEPOSIT BARRIER METAL", "RCA CLEAN 1", "MEASURE OXIDE THICKNESS"] },
      trained: { method: "LSTM · teacher-forcing", ranked: ["DEPOSIT BARRIER METAL", "DEPOSIT TUNGSTEN SEED", "RCA CLEAN 1", "STRIP RESIST", "HARD BAKE"] },
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

  window.ZH = {
    meta: {
      title: "Zero-Hack · Industrial Process-Logic Benchmark",
      doc: "ZH-01",
      rev: "A",
      track: "Industrial-AI / Process-Logic · Infineon",
      families: ["IC", "MOSFET", "IGBT"],
      vocab: 198,
      n_models: 4,
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
  };
})();
