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

We used a small pipeline:

- Generate more valid process flows from the provided grammar, plus separate
  novel family eval sets for OOD checks. This gives the model more structure
  than the three starter families alone.
- Train a GPT decoder with ALiBi attention bias for next-step prediction. We use
  the same model for completion and likelihood anomaly scoring.
- Use the process-rule validator during fine tuning where useful, so generated
  continuations are scored on process validity and token likelihood.

---

## 1. Classic & LSTM baselines

We started with simple baselines on the shared split files from
`scripts/create_dataset_splits.py`. They set the floor for next-step ranking,
completion, and anomaly detection.

### 1.1 Classic baselines

These runs train on all three families and evaluate on ID test sets: 600
examples for next-step and completion, 300 for anomaly.

| Model | View | Task 1 Top-1 | Top-3 | MRR | Task 2 ExactMatch | Norm. edit dist | Task 3 F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Most-frequent | ID | 0.7067 | 0.9950 | 0.8501 | 0.0017 | 0.2458 | 1.000 | 1.000 |
| N-gram (5, backoff) | ID | 0.7133 | 0.9967 | 0.8536 | 0.0050 | 0.2243 | 1.000 | 1.000 |

### 1.2 LSTM baseline

We then trained an LSTM on the 5k-samples-per-family subset, using all three
families and no holdout. This was a quick check before spending more compute on
larger transformer runs.

We compared standard teacher forcing with scheduled sampling. Teacher forcing
always feeds the true previous token during training. Scheduled sampling
sometimes feeds the model's own previous prediction, which should make
free-running completion less brittle.

It did not help here. Next-step accuracy stayed almost unchanged, and scheduled
sampling was slightly worse on exact match, token accuracy, and block accuracy.
Both variants produced valid process flows, so the remaining problem was matching
the exact held-out sequence, not satisfying the validator.

| Variant | Task 1 Top-1 | Top-3 | MRR | Task 2 ExactMatch | Norm. edit dist | Token acc | Process-validity rate |
|---|---|---|---|---|---|---|---|
| Teacher forcing | 0.8100 | 0.9954 | n/a | 0.0033 | 0.2240 | 0.3919 | 1.000 |
| Scheduled sampling | 0.8087 | 0.9949 | n/a | 0.0017 | 0.2199 | 0.3711 | 1.000 |

---

## 2. Synthetic data preparation

We tested three ways to create extra process flows.

| Method | Where | How it works | Output | Audit | Decision |
|---|---|---|---|---|---|
| Planner-based unseen data | `main:src/zero_hack/data/synth/` and `scripts/generate_unseen_data.py` | Builds valid flows from phase-level process units. The planner injects required cleans, lithography, etch, CMP, and test ordering before a rule can be violated. Family labels are partly decoupled from content through synthetic labels and `UNK` dropout. | `data/generated/<dataset>/raw.csv` | Validator backstop, phase monotonicity tests, vocabulary coverage tests. | Useful for broad augmentation. We kept it separate from the stricter OOD story because labels are synthetic training labels, not real unseen families. |
| Pseudo families | `data/eval/pseudo_families/` | Recombines whole blocks from MOSFET, IGBT, and IC generators. For example, one profile can use MOSFET prep, IC cycles, MOSFET via, and IC metal. | 3 profiles, 5k sequences each | 1.000 raw validity, 0 OOV steps, 5k unique sequences per profile. | Useful stress test, but not clean OOD. It can reuse held-out family block grammar, so we treat it as eval-only and do not use it as final OOD evidence. |
| Novel families | `scripts/generate_novel_families.py` and `data/eval/novel_families/` | Builds flows from atomic role-typed operation pools instead of family block generators. The script samples cleans, depositions, etches, implants, via steps, metal steps, passivation, backside steps, and tests independently while satisfying validator rules by construction. | `novel_sparse`, `novel_mixed`, `novel_dense`, 5k sequences each | 1.000 raw validity, 0 OOV steps, 5k unique sequences per profile, 0 exact collisions with generated family references. | Use for leak-free OOD evaluation. These flows have process logic but no official family backbone. |

The novel-family generator also reports n-gram distance from real families. The
dense profile has the most new local structure: 0.578 mean novel 3-gram fraction
and 0.767 mean novel 5-gram fraction. The sparse profile is closer to the real
families, with 0.391 novel 3-grams and 0.636 novel 5-grams. That gives us a
small difficulty sweep without changing the validator or introducing unknown
tokens.

---

## 3. Model training

### 3.1 Architectures compared

We compared five transformer variants on `valid_s100k`. Each run trained on two
families and evaluated on both ID samples and the unseen third family. The table
averages `holdout_mosfet`, `holdout_igbt`, and `holdout_ic` from
`architecture_metrics/valid_s100k/` and `arch_compare.json`.

![Architecture comparison on valid_s100k](reports/architecture_comparison_valid_s100k.png)

| Architecture | Position scheme | ID next-step | OOD next-step | ID completion | OOD completion | ID anomaly AUC | OOD anomaly AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-absolute | learned absolute | 0.675 | 0.660 | 0.965 | 0.799 | 1.000 | 0.433 |
| GPT-ALiBi | linear attention bias | 0.652 | 0.635 | 0.963 | 0.806 | 0.999 | 0.680 |
| GPT-LLaMA-style | rotary | 0.681 | 0.672 | 0.964 | 0.756 | 1.000 | 0.650 |
| GPT-RoPE | rotary | 0.675 | 0.600 | 0.961 | 0.791 | 1.000 | 0.612 |
| Causal transformer | learned absolute | 0.681 | 0.633 | 0.963 | 0.759 | 1.000 | 0.773 |

We chose GPT-ALiBi for the final transformer architecture. The ID scores do not
separate the models much: completion is around 0.96 block accuracy and anomaly
AUC is almost perfect for every variant. The useful signal is OOD. GPT-ALiBi has
the best OOD completion score at 0.806, which matters most for generating the
remaining process flow from a prefix. It also keeps anomaly separation reasonable
at 0.680 OOD AUC, far above GPT-absolute and RoPE, though below the causal
transformer.

That tradeoff matched the submission goal. We needed one model that can rank the
next step, complete sequences, and produce likelihood scores for anomaly
detection. ALiBi gives the decoder a simple distance bias instead of relying only
on learned absolute positions, so it handles prefixes of different lengths
cleanly and keeps generation stable on the held-out family. The causal
transformer is better if we optimize only for OOD anomaly AUC, but GPT-ALiBi is
the stronger all-task choice for our completion-focused pipeline.

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
