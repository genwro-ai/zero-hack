# Experiment report -- genwro — Industrial AI (Infineon)

## Team

- **Marcin Kostrzewa** — AI Engineer
- **Michal Furgala** — AI Engineer
- **Lukasz Lenkiewicz** — AI Engineer

**Track:** Industrial AI (Infineon) — learning process logic from semiconductor fabrication sequences.

---

## TL;DR

{2–3 sentences: what we built, on which tasks (next-step / completion / anomaly / OOD), and the headline result vs. baseline.}

---

## Problem

We model the process grammar behind MOSFET, IGBT, and IC fabrication flows (≈198-step vocabulary, ~115–150 steps/sequence, shared backbone + family-specific steps). Three scored tasks plus an OOD differentiator:

- **Task 1 — Next step:** Top-1/3/5 accuracy, MRR.
- **Task 2 — Completion:** exact match, normalized edit distance, token & block accuracy, **process-validity rate** of generated continuations.
- **Task 3 — Anomaly:** accuracy, precision/recall/F1, ROC-AUC, rule-attribution accuracy.
- **Task 4 — OOD:** leave-one-family-out — train on two families, evaluate on the unseen third.

---

## Approach

Our solution is built in three steps, each with a distinct goal.

- **Grammar generation.** We are given only three grammar sets (MOSFET, IGBT, IC), but the solution should generalize to new, previously unseen grammars — not just memorize the three we have. That is why a robust grammar-generation procedure is crucial: by synthesizing additional rule-valid process flows (and genuinely novel, family-less ones for OOD evaluation) we expose the model to far greater structural variety than the starter data alone.
- **Neural network architecture.** Neural networks are state-of-the-art across most ML areas, so they were the natural choice here. We settled on a transformer-based architecture specifically because of its efficiency and strength in sequence modelling, which matches the discrete, long-range structure of fabrication sequences.
- **Fine-tuning.** Plain supervised (teacher-forced) learning may not be enough to capture the complex, long-range patterns in these sequences. We therefore aim to improve the trained model with a carefully designed loss / reward function — using the process-rule validator as a verifiable signal — so the model is optimized for valid, coherent completions rather than only next-token likelihood.

---

## 1. Classic & LSTM baselines

Lower bounds and the main non-neural references, all evaluated on the same split artifacts (`scripts/create_dataset_splits.py`) and eval sets.

### 1.1 Classic baselines

In-distribution (all three families trained), test sets of 600 (next-step / completion) and 300 (anomaly) examples.

| Model | View | Task 1 Top-1 | Top-3 | MRR | Task 2 ExactMatch | Norm. edit dist | Task 3 F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Most-frequent | ID | 0.7067 | 0.9950 | 0.8501 | 0.0017 | 0.2458 | 1.000 | 1.000 |
| N-gram (5, backoff) | ID | 0.7133 | 0.9967 | 0.8536 | 0.0050 | 0.2243 | 1.000 | 1.000 |

### 1.2 LSTM baseline

As a first neural baseline we trained an LSTM next-step model on a small 5k-samples-per-family subset (all three families, no holdout) to gauge what early results to expect before scaling up data and training. The model is trained with standard teacher forcing i.e. at every step it is fed the ground-truth previous token and learns to predict the next one.

Teacher forcing, however, can struggle at generation time. If the model predicts a single token incorrectly, that wrong token is fed back as context for the next step, and the error propagates and compounds through the rest of the sequence. Scheduled sampling is designed to address exactly this: during training it occasionally injects the model's own (possibly incorrect) predictions instead of the ground-truth token, so the model is exposed to its own mistakes and can hopefully learn to recover from them.

To test whether this helps, we trained both variants and measured free-running completion directly (high teacher-forced next-step accuracy does not by itself imply good completion). Task-1 numbers are per-position next-step accuracy on the test set; Task-2 numbers are free-running completion from 0.6/0.8 prefix cuts over 600 held-out sequences.

**Conclusion.** In practice, scheduled sampling did *not* help at this scale. It is slightly worse than plain teacher forcing on exact match, token accuracy, and block accuracy, and only marginally better on normalized edit distance, while next-step accuracy is essentially unchanged. Both variants decode 100% process-valid completions, so the remaining gap is token-level fidelity rather than rule validity — the error-recovery benefit we hoped for did not materialize in this small-scale run.

| Variant | Task 1 Top-1 | Top-3 | MRR | Task 2 ExactMatch | Norm. edit dist | Token acc | **Process-validity rate** |
|---|---|---|---|---|---|---|---|
| Teacher forcing | 0.8100 | 0.9954 | — | 0.0033 | 0.2240 | 0.3919 | **1.000** |
| Scheduled sampling | 0.8087 | 0.9949 | — | 0.0017 | 0.2199 | 0.3711 | **1.000** |

---

## 2. Synthetic data preparation


| Attempt | Generator | Purpose | Volume | Raw validity | OOV | Dups / family collisions | Used for | Outcome |
|---|---|---|---|---|---|---|---|---|
| {Block recombination} | `scripts/generate_pseudo_families.py` | {…} | | | | | | {leaks held-out blocks — eval risk} |
| **Novel families (leak-free)** | `scripts/generate_novel_families.py` | OOD eval, no family backbone | | ~1.000 | 0 | 0 / 0 | eval-only | {TODO} |
| {Hard-negative near-misses} | {…} | Task 3 training | | | | | anomaly head | {TODO} |

**Notes:**
- Novel-family flows are composed from atomic role-typed ops sampled independently with rule preconditions satisfied by construction (not block recombination), so they can't leak a held-out family's block grammar. Output: `data/eval/novel_families/<profile>/` — **never enters training**.
- {What worked / what didn't in coverage of the 11 variation axes.}

---

## 3. Model training

### 3.1 Architectures compared

| Model | Params | Depth × width | Context | Pos. encoding | Notes |
|---|---|---|---|---|---|
| LSTM | | | | — | baseline |
| {Transformer decoder} | | | | {RoPE/ALiBi/abs} | {…} |
| {variant} | | | | | |

### 3.2 Training setup

- **Objective:** {next-step teacher-forced CE / + scheduled sampling}.
- **Data / splits:** {dataset, holdout family, # sequences}.
- **Optimizer / schedule:** {…}. **Hardware:** {Leonardo, N×A100, wall-clock}.
- **Slurm:** `{slurm/train_lstm.sbatch / train_models.sbatch / run_holdout_gpt_alibi*.sbatch}`.

### 3.3 Learning curves

{TODO: embed train/val loss and Top-k-vs-step plots — `extras/` or `reports/`.}

```
[ train / val loss curve ]            [ Top-3 accuracy vs. step ]
   placeholder                            placeholder
```

### 3.4 Trained-model results (baseline vs. trained, same inputs)

| Model | View | Task 1 Top-1 | Top-3 | MRR | Task 2 ExactMatch | Process-validity | Task 3 F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Best classic (ref) | ID | | | | | | | |
| LSTM | ID | | | | | | | |
| {Transformer} | ID | | | | | | | |
| {Transformer} | OOD | | | | | | | |

> ID→OOD drop (the Task-4 story): {TODO}.

---

## 4. RL fine-tuning

GRPO (RLVR-style) fine-tuning of a pretrained next-step policy, using the process-rule `validate_sequence` as a **verifiable reward** for the completion objective (`scripts/grpo_finetune.py` → `zero_hack.models.grpo`).

### 4.1 Approach

- **Base policy:** {LSTM / transformer checkpoint}.
- **Reward:** {validity / shaped reward — describe `RewardConfig`}.
- **Prompts:** sampled from **training families only** (`--prompt-families` defaults to `train_families`) — the holdout family stays unseen to preserve the OOD protocol.
- **Config:** {group size, KL coeff, LR, steps — `GRPOConfig`}.
- **Slurm:** `slurm/grpo_finetune.sbatch` (matrix: `scripts/run_grpo_completion_matrix.sh`).

### 4.2 Results — before vs. after fine-tuning

| Policy | View | Task 2 ExactMatch | Norm. edit dist | Token acc | **Process-validity rate** | Reward |
|---|---|---|---|---|---|---|
| Pretrained (base) | ID | | | | | |
| + GRPO | ID | | | | | |
| Pretrained (base) | OOD | | | | | |
| + GRPO | OOD | | | | | |

> **Finding:** {did the verifiable reward raise validity / exact-match without collapsing diversity? TODO}.

---

## How to run it

```bash
uv sync

# Splits + eval sets
uv run python scripts/create_dataset_splits.py
uv run python scripts/make_all_eval_sets.py

# Classic baselines
uv run python -m zero_hack.models.ngram.train --holdout-family ic

# LSTM (teacher forcing vs scheduled sampling)
sbatch slurm/train_lstm.sbatch
sbatch slurm/eval_lstm_completion.sbatch

# RL fine-tuning
sbatch slurm/grpo_finetune.sbatch
```

See [README.md](README.md) for the full data/eval layout.

---

## What worked / What didn't

- **Worked:** {…}
- **Didn't:** {…}

## What we'd do with another 36 hours

- {Next step 1} · {Next step 2}

## Credits & dependencies

- **Libraries / frameworks:** {PyTorch, …}
- **Data:** Infineon Industrial AI starter data + our synthetic generators.
- **AI coding tools used:** {…}
