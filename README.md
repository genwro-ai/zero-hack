# zero-hack

Our participant repository for **Zero One Hack_01**, Industrial AI track by Infineon: learning and benchmarking process logic from semiconductor fabrication sequences.

The goal is to build a reproducible workflow for sequence modeling on the provided MOSFET, IGBT, and IC process-flow data. The expected hackathon deliverable is a working artifact with baseline-vs-trained evaluation for next-step prediction, sequence completion, and anomaly detection.

## Repository layout

```text
.
├── data/industrial/          # Provided Industrial AI training data and grammar docs
├── configs/                  # Experiment and training configuration files
├── docs/track/               # Industrial track briefings, English and German
├── docs/submission/          # Hackathon submission instructions and report template
├── notebooks/                # Exploratory work
├── outputs/                  # Generated predictions, logs, and plots
├── reports/                  # Analysis notes and exported artifacts
├── scripts/                  # Reproducible local entry points and utilities
├── slurm/                    # Leonardo/cluster Slurm job scripts
└── src/zero_hack/            # Python package code
```

The repo tracks the relevant Industrial data, track instructions, and submission instructions in the organized locations above.

## Setup

This repo uses `uv`, Python 3.12, and a package-style Python layout.

```bash
uv sync
uv run pre-commit install
uv run zero-hack-info
```

Useful checks:

```bash
uv run python data/industrial/generate_sequences.py --family mosfet --estimate-only
uv run python data/industrial/generate_sequences.py --validate data/industrial/MOSFET_variants.csv
```

Run all configured checks before committing:

```bash
uv run pre-commit run --all-files
uv run ruff check src
uv run ruff format --check src
```

## Data

The Industrial starter data is in `data/industrial/`.

Key files:

- `MOSFET_variants.csv`, `IGBT_variants.csv`, `IC_variants.csv`: 1,000 valid process sequences per family in long format.
- `synthetic_mosfet.csv`, `syntheticIGBT.csv`, `syntheticIC.csv`: canonical reference sequences.
- `*_Longdescr.csv` and `*_longdescription_parameters.csv`: enriched step descriptions and process parameters.
- `generation_rules.md`: grammar, forbidden patterns, variation axes, and evaluation protocol.
- `generate_sequences.py`: original generator and validator copied from the track material; kept with the data because it defines and validates the dataset grammar.

Package helpers are available from `zero_hack.data`:

```python
from zero_hack.data import load_family_sequences

mosfet_sequences = load_family_sequences("mosfet")
```

## Baselines

Symbolic next-step baselines live under `src/zero_hack/models/`. They read the
train/valid/test CSVs produced by `scripts/create_dataset_splits.py`, so every
model is compared on the same split artifacts.

- **Most frequent** (`models/most_frequent/`): counting baseline conditioned on
  `(family, position bucket, previous step)` with stupid-backoff to coarser
  contexts. The sanity-check lower bound.
- **N-gram** (`models/ngram/`): family-conditioned counting model (default
  5-gram) with stupid-backoff to shorter contexts. The main symbolic baseline.

Both expose `predict_topk` (next step), greedy autoregressive completion, and
`score_sequence` (log-likelihood, used as a soft anomaly score). Fit and
evaluate next-step accuracy directly:

```bash
uv run python -m zero_hack.models.most_frequent.train
uv run python -m zero_hack.models.ngram.train
```

Use `--splits-dir data/generated/valid_s100k/splits` to select a dataset size,
`--holdout-family ic` for the two-families-train / third-family-test setup, and
`--limit-per-family N` for a fast smoke run.

## Evaluation

`zero_hack.eval` implements the shared eval protocol
([generation_rules.md §5](data/industrial/generation_rules.md)) for all three
tasks, with no external dependencies. It reuses the canonical 10-rule
`validate_sequence` from `generate_sequences.py` rather than re-implementing it.

| Task | Metrics |
|---|---|
| 1 — Next step | Top-1/3/5 accuracy, MRR |
| 2 — Completion | Exact match, normalized edit distance, token accuracy, block-level accuracy |
| 3 — Anomaly | Accuracy, precision/recall/F1, confusion matrix, ROC-AUC, rule-attribution accuracy |

Three scripts make the pipeline runnable end-to-end. Until the organizers
distribute the real eval files, `make_eval_set.py` synthesises an equivalent
held-out eval set (Tasks 1 & 2 from prefix cuts; Task 3 from validator-flagged
perturbations) so metrics and the baseline-vs-trained comparison run today:

```bash
# 1. Build eval inputs + ground truth from the held-out test split
uv run python scripts/make_eval_set.py

# 2. Produce the three submission files from a baseline
uv run python scripts/baseline_predict.py --model ngram

# 3. Score a submission against ground truth (mirrors the organizer CLI)
uv run python scripts/eval_metrics.py --task next_step \
  --ground-truth outputs/eval/nextstep_truth.csv \
  --predictions  outputs/preds/ngram/nextstep.csv \
  --eval-input   outputs/eval/eval_input_valid.csv
```

`scripts/eval_metrics.py` mirrors the organizer `eval_metrics.py` interface
(`--task {next_step,completion,anomaly} --ground-truth --predictions`); prefer
the official scorer once it is available. Block-level accuracy is our
interpretation (LCS over functional-block runs) pending the official definition.

## Track Instructions

Industrial AI materials copied from the hackathon source:

- [Industrial track README](docs/track/README.md)
- [English track briefing](docs/track/industrial_en.md)
- [German track briefing](docs/track/industrial_de.md)
- [Training data guide](data/industrial/README.md)
- [Generation rules and eval protocol](data/industrial/generation_rules.md)

## Submission Notes

Submission is through the hackathon Tally form by Sunday 10:00. The required public repository fields are team name, repository URL, slides PDF, and a max 2-minute demo video.

Industrial track-specific repository deliverables:

- `nextstep.csv` for Task 1.
- `completion.csv` for Task 2.
- `anomaly.csv` for Task 3.

  These three submission files are generated by `scripts/baseline_predict.py`
  (see [Evaluation](#evaluation)).
- Training artifacts such as checkpoints, logs, and loss curves.
- Scores from the organizer `eval_metrics.py` script when it is available; until
  then, self-score locally with `scripts/eval_metrics.py`.
- A baseline-vs-trained comparison on identical inputs (the shared splits and
  eval set make this direct).

Full copied instructions are in [docs/submission/SUBMISSION.md](docs/submission/SUBMISSION.md). A starter report is at [REPORT.md](REPORT.md).
