# Experiment report genwro — Industrial AI (Infineon)

## Team

- **Marcin Kostrzewa** — AI Engineer
- **Michal Furgala** — AI Engineer
- **Lukasz Lenkiewicz** — AI Engineer

**Track:** Industrial AI (Infineon) — learning process logic from semiconductor fabrication sequences.

---

## TL;DR

We built a synthetic process-grammar generator and trained a small GPT-style decoder to model MOSFET, IGBT, and IC fabrication sequences, scoring it on next-step prediction, completion, anomaly detection, and a leave-one-family-out OOD split. In distribution the decoder roughly matches a tuned n-gram on next-step accuracy (about 0.69 top-1, 0.99 top-3) and flags rule violations almost perfectly (ROC-AUC near 1.0). The OOD case is where it breaks: on an unseen family the anomaly threshold no longer transfers and detection falls to chance-level F1 (about 0.56), which is the gap the rest of the report is about.

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
| Planner-based unseen data | `src/zero_hack/data/synth/` and `scripts/generate_unseen_data.py` | Builds valid flows from phase-level process units. The planner injects required cleans, lithography, etch, CMP, and test ordering before a rule can be violated. Family labels are partly decoupled from content through synthetic labels and `UNK` dropout. | `data/generated/<dataset>/raw.csv` | Validator backstop, phase monotonicity tests, vocabulary coverage tests. | Useful for broad augmentation. We kept it separate from the stricter OOD story because labels are synthetic training labels, not real unseen families. |
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

**Naming.** From here on we call this chosen decoder **GPT ALiBi**, and the rest of
the report uses that name. The `gpt_bare`, `gpt_phase_augmented`, and `gpt_dpo`
runs in `metrics 2/` are all this ALiBi decoder at different stages, where `bare`
means the base next-step model before any fine-tuning. So a row labelled GPT ALiBi
in section 3.4 and onward is this architecture, not a separate model.

> **Caveat:** this sweep is an independent experiment. It was scored by an earlier
> completion harness (`architecture_metrics/valid_s100k/` and `arch_compare.json`),
> not the `metrics 2/` source of truth used everywhere else in this report. The two
> harnesses define block accuracy differently, so the completion column here (around
> 0.96 in distribution) is not comparable to the completion numbers in sections 3.4
> and elsewhere. We keep these results unchanged as the basis for the architecture
> choice and do not reconcile them against `metrics 2/`.

### 3.2 Training setup

- **Objective:** next-token cross-entropy with label smoothing 0.02, padding ignored. An auxiliary next-phase loss is available but stays off by default (`phase_loss_weight = 0`).
- **Data / splits:** `valid_s100k` is the generator output of 100k rule-valid sequences per family (MOSFET, IGBT, IC). Splits are leave-one-family-out: train on two families, hold out the third. ID test draws from the two trained families, OOD test is the held-out family. The `gpt_bare` reference trains on a 20k subsample.
- **Model:** GPT decoder with `d_model 256`, 4 layers, 4 heads, context 256, vocab 206, about 3.3M parameters. The `gpt_bare` reference is `d_model 128`, 3 layers, context 192 (about 0.65M).
- **Optimizer / schedule:** AdamW, lr `3e-4`, weight decay 0.05, betas `(0.9, 0.95)`, linear warmup over 5% of steps then cosine decay to `0.1x`, gradient clip 1.0. Batch size 128, up to 30 epochs, early stop with patience 5, seed 1729.
- **Hardware:** CINECA Leonardo, one A100 64GB per run on the `boost_usr_prod` partition. The three holdouts run as a job array.
- **Slurm:** `slurm/train_lstm.sbatch`, `slurm/train_models.sbatch`, `slurm/run_holdout_gpt_alibi.sbatch` (and `run_holdout_gpt_alibi_large.sbatch` for the larger config).

### 3.3 Learning curves

The architecture sweep figure is the bar chart in section 3.1. For per-epoch curves we have the LSTM training histories on `valid_s100k` (`metrics 2/valid_s100k/lstm_*_holdout_*/history.json`), averaged over the three holdouts below.

![LSTM learning curves on valid_s100k](reports/lstm_learning_curves.png)

The teacher-forcing LSTM converges fast. Train loss drops from 0.477 to 0.338 over 10 epochs while validation loss settles between 0.315 and 0.336, and validation top-1 reaches about 0.82 by epoch 3 and then flattens (peak 0.822 at epoch 8). Validation loss sits slightly below train loss because the training loss carries dropout and label smoothing that the eval pass does not. The left panel puts teacher forcing against scheduled sampling on the same validation top-1 axis: the two track each other within a percentage point, with scheduled sampling marginally higher on next-step. That small next-step edge did not carry over to free-running completion (section 1.2), which is the metric we actually cared about. The GPT runs were not exported as committed loss-vs-step figures beyond the architecture comparison chart.

On process validity: the ground-truth signal is the organizer rule checker `validate_sequence`. The LSTM completion eval (`scripts/eval_lstm_completion.py`) runs it on each decoded continuation and the LSTM decodes 100% valid completions (the 1.000 in section 1.2). The transformer and DPO metric dumps in `metrics 2/` and `architecture_metrics/` do not store this field, so we do not quote a process-validity number for those models rather than infer one.

### 3.4 Trained-model results (baseline vs. trained, same inputs)

All rows come from `metrics 2/valid_s100k/`, averaged over the three holdouts. The classic reference is the tuned 5-gram, the LSTM is teacher forcing, and the GPT is GPT ALiBi at its `bare` stage, the base next-step decoder trained on `valid_s100k` before any fine-tuning. The architecture comparison in 3.1 is a separate experiment on a different harness (see the caveat there), so it is not mixed into this table.

| Model | View | Top-1 | Top-3 | MRR | Compl. ExactMatch | Anomaly F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| 5-gram (tuned) | ID | 0.690 | 0.996 | 0.843 | 0.004 | 1.000 | 1.000 |
| LSTM (teacher forcing) | ID | 0.689 | 0.997 | 0.842 | 0.003 | n/a | n/a |
| GPT ALiBi (bare) | ID | 0.687 | 0.996 | 0.841 | 0.004 | 0.944 | 0.985 |
| 5-gram (tuned) | OOD | 0.663 | 0.980 | 0.821 | 0.000 | 0.667 | 0.769 |
| LSTM (teacher forcing) | OOD | 0.657 | 0.981 | 0.817 | n/a | n/a | n/a |
| GPT ALiBi (bare) | OOD | 0.667 | 0.979 | 0.823 | 0.000 | 0.667 | 0.765 |

> ID to OOD drop (the Task-4 story): next-step prediction barely moves between ID and OOD (top-1 stays near 0.66), because local step transitions look similar across families. Anomaly detection is the part that breaks. In distribution every model scores near-perfect ROC-AUC, but on the held-out family the ID-tuned threshold flags almost everything, so F1 collapses toward the all-positive baseline (around 0.67) and threshold-free ROC-AUC drops to about 0.77. GPT ALiBi and the 5-gram land in the same place on OOD anomaly (AUC 0.765 and 0.769), so the neural model gives no advantage there. Anomaly F1 depends on both the ID-tuned threshold and the test composition, so ROC-AUC is the fairer cross-model OOD comparison. The LSTM was only scored for next-step and ID completion, so its completion and anomaly cells are blank. (Its OOD Top-3, 0.981, was measured but is a next-step metric, so it sits with Top-1 and MRR; the earlier blank in that cell was a transcription gap, now filled.)

### 3.5 Neurosymbolic decoding

We tested whether applying the process rules at decoding time changes next-step predictions. Starting from the `gpt_phase_augmented` GPT, we re-rank the next-step distribution two ways (`src/zero_hack/models/neurosymbolic/decoding.py`, driven by `scripts/compare_gpt_neurosymbolic.py`): `ns_hard` masks out tokens that would violate a rule, and `ns_shaped` applies a soft penalty instead of a hard mask. The table is OOD next-step on the held-out family (`outputs/neurosymbolic/valid_s100k_augmented_s050k/holdout_*_ood.jsonl`, 2000 examples each).

| Holdout (OOD) | Decoder | Top-1 | Top-3 | Top-5 | MRR | Top-1 changed |
|---|---|---:|---:|---:|---:|---:|
| ic | bare | 0.5905 | 0.9895 | 0.9975 | 0.7853 | reference |
| ic | ns_hard | 0.5905 | 0.9895 | 0.9975 | 0.7853 | 0 / 2000 |
| ic | ns_shaped | 0.5925 | 0.9895 | 0.9975 | 0.7863 | 6 / 2000 |
| mosfet | bare | 0.6255 | 0.9985 | 1.0000 | 0.8045 | reference |
| mosfet | ns_hard | 0.6255 | 0.9985 | 1.0000 | 0.8045 | 0 / 2000 |
| mosfet | ns_shaped | 0.6255 | 0.9985 | 1.0000 | 0.8045 | 0 / 2000 |
| igbt | bare | 0.6655 | 0.8490 | 0.9885 | 0.7883 | reference |
| igbt | ns_hard | 0.6655 | 0.8490 | 0.9885 | 0.7883 | 0 / 2000 |
| igbt | ns_shaped | 0.6655 | 0.8495 | 0.9885 | 0.7884 | 0 / 2000 |

> **Finding:** decode-time rules do almost nothing here. Hard masking never changes the top-1 token (0 of 2000 on every holdout), because the rule mask does not fire on the model's high-probability candidates. Shaping moves at most 6 of 2000 predictions (IC), a top-1 change of 0.002. The augmented GPT already keeps its top next-step candidates rule-compliant, so the symbolic layer has nothing to correct at this stage. It could still matter for full free-running completion, where errors compound over many steps, which we did not measure here.

---

## How to run it

The full pipeline runs as Slurm jobs on Leonardo. Each `sbatch` wrapper provisions
the environment (`slurm/setup_uv.sh`) and calls the same Python entry points used
locally, so the cluster and laptop paths stay in sync.

```bash
uv sync   # one-time environment setup on the login node

# Data: generate rule-valid sequences, leave-one-family-out splits, eval sets
sbatch slurm/generate_valid_datasets.sbatch     # array 0-8: valid_s005k … valid_s500k
sbatch slurm/create_dataset_splits.sbatch       # array 0-2: per holdout family
sbatch slurm/make_eval_sets.sbatch              # array 0-2: id / ood / calibration sets
sbatch slurm/generate_novel_families.sbatch     # array 0-2: leak-free OOD eval data

# Classic baselines (n-gram / VLMC search, all three tasks)
sbatch slurm/eval_classic_search.sbatch

# LSTM (teacher forcing vs scheduled sampling)
sbatch slurm/train_lstm.sbatch                  # array 0-1
sbatch slurm/eval_lstm_completion.sbatch        # array 0-1

# Transformer architecture sweep (abs / alibi / rope / llama)
sbatch slurm/run_holdout_gpt_alibi.sbatch       # add slurm/run_holdout_gpt_alibi_large.sbatch for the larger config

# Fine-tuning: augmented SFT, then DPO, then GRPO (RL with verifiable reward)
sbatch slurm/train_gpt_phase_augmented.sbatch
sbatch slurm/train_gpt_dpo.sbatch
CKPT=outputs/models/valid_s005k/lstm_teacher_forcing/best.pt \
  HOLDOUT_FAMILY=ic sbatch slurm/grpo_finetune.sbatch
```

Most wrappers accept environment overrides (e.g. `DATASETS=valid_s005k
HOLDOUT_FAMILIES=ic sbatch slurm/run_holdout_gpt_alibi.sbatch`); see the comment
header of each `.sbatch` file for the full list. To run a wrapper off-cluster
(no scheduler), set `RUNNER="" LEONARDO_LOAD_MODULES=0` and invoke it with `bash`
instead of `sbatch`.

See [README.md](README.md) for the full data/eval layout.

---

## What worked / What didn't

- **Worked:** Generating rule-valid sequences at scale (100k per family) with the validator as a backstop. Plain sequence likelihood is a near-perfect in-distribution anomaly detector, both for the 5-gram and GPT ALiBi (ROC-AUC near 1.0). The leak-free novel-family generator gives us an honest OOD set. DPO on validity-labeled pairs sharpened ID anomaly discrimination (AUC 0.981 to 0.999).
- **Didn't:** OOD anomaly detection. The ID-tuned threshold does not transfer to an unseen family, so F1 falls toward the all-positive baseline and even ROC-AUC drops to 0.68-0.77. Positional encoding choice barely moved the ID metrics, so there was no clear winner and we picked ALiBi for its OOD completion. Scheduled sampling did not beat teacher forcing at this scale. Exact-match completion stays near zero, since the sequences are long and many continuations are equally valid.

## What we'd do with another 36 hours

The biggest unfinished piece is fine-tuning the decoder against the process-rule validator as a verifiable signal. The checker `validate_sequence` labels any sequence as valid or not for free, so it can drive training directly. We implemented two realizations and have early offline numbers, but the online sweep is not done, so this stays in the future-work column.

**Offline preference optimization (DPO).** Starting from the augmented-data SFT checkpoint (`gpt_phase_augmented`, the `d_model 256` 4-layer GPT), `scripts/generate_dpo_pairs.py` builds preference pairs: cut a valid sequence at a random 25-75% prefix, keep the true continuation as the chosen sample, and produce a rejected continuation one edit from valid. The mix is 0.9 rule-violating (a targeted `corrupt_steps` break confirmed by `first_violated_rule`) and 0.1 valid-but-mismatched suffix, about 50k pairs. The loss in `dpo_train.py` is the standard DPO objective — a logistic loss that rewards the policy for assigning a larger log-probability margin to the chosen over the rejected continuation than the frozen reference policy does — plus a small SFT term that keeps the policy anchored to the gold continuation. We use a DPO temperature of 0.1, SFT weight 0.3, the reference policy frozen, lr 5e-6, 2 epochs, batch 16, prompts from training families only so the held-out family stays unseen.

**Online RL (GRPO), implemented with an initial run.** `scripts/grpo_finetune.py` and `zero_hack.models.grpo` sample a group of completions per prompt, score each with a reward, and update on group-relative advantages: each completion's advantage is its reward standardized within the group (subtract the group-mean reward, divide by the group-std), and the policy is pushed to raise the log-probability of above-average completions, with a k3 KL penalty back to the reference policy. The reward is multiplicative on purpose. A validity gate is 1 for a valid completion (or a graded `max(0, 1 - n_violations / len)`), and a quality term measures fidelity to the gold suffix through block, token, and exact accuracy plus block diversity, with a small termination/truncation adjustment. A binary validity reward would give no within-group spread once the policy is mostly rule-compliant, so the fidelity terms keep the advantage informative and block the "shortest valid tail" hack. Defaults are a group size of 8 and a KL coefficient of 0.02.

Initial training-log findings (`outputs/slurm`, `valid_s010k`, holdout IGBT, prompts from MOSFET + IC, 200 steps): the policy stays fully rule-compliant throughout (rollout validity 1.000 at every logged step) and holds rollout block accuracy around 0.94-0.96, while the KL to the reference stays small (about 0.07 at the end). Mean reward hovers around 1.2-1.3 with no clear upward trend over the run, and exact match stays at 0. So the run confirms GRPO trains stably and keeps completions valid and high-fidelity, but does not yet show a clear completion gain. These are training-rollout numbers, not held-out eval; the full before-vs-after completion matrix (`scripts/run_grpo_completion_matrix.sh`) has not finished, so we do not yet quote ID/OOD GRPO eval metrics.

**Early DPO result (metrics 2 source of truth).** Averaged over the three holdouts (`metrics 2/valid_s100k_augmented_s050k/`):

| Policy | View | Top-1 | Compl. ExactMatch | Norm. edit dist | Token acc | Anomaly F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| SFT (phase-augmented) | ID | 0.689 | 0.004 | 0.226 | 0.430 | 0.936 | 0.981 |
| + DPO | ID | 0.701 | 0.003 | 0.224 | 0.422 | 0.996 | 0.999 |
| SFT (phase-augmented) | OOD | 0.627 | 0.000 | 0.439 | 0.134 | 0.667 | 0.850 |
| + DPO | OOD | 0.625 | 0.000 | 0.429 | 0.131 | 0.667 | 0.817 |

DPO mainly sharpens in-distribution anomaly discrimination (ROC-AUC 0.981 to 0.999, F1 0.936 to 0.996) with a small next-step gain and flat completion. The OOD anomaly AUC stays high for both (0.850 SFT, 0.817 DPO) and both clear the base GPT ALiBi (0.765), so the augmented SFT stage is what buys the OOD robustness, and DPO does not add to it. Whether the online GRPO reward can move OOD completion is the open question.

Other things we would do:

- Fix the OOD threshold transfer: a per-family or family-agnostic calibration of the anomaly threshold, or a small calibration set from a few unseen-family samples.
- Ensemble the GPT with the 5-gram for anomaly scoring, since the n-gram likelihood is as strong an OOD detector.
- Train the larger transformer config on the full 100k rather than a subsample.
