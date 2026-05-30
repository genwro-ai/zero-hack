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
from zero_hack.data import load_industrial_family_records

mosfet_records = load_industrial_family_records("data/industrial", "mosfet")
```

## Baselines

Classic next-step baselines live under `src/zero_hack/models/`. They read the
train/valid/test CSVs produced by `scripts/create_dataset_splits.py`, so every
model is compared on the same split artifacts.

- **Most frequent** (`models/most_frequent/`): position-frequency baseline
  conditioned on `(family, position bucket)`, with fallback to family/global
  frequencies. The sanity-check lower bound.
- **N-gram** (`models/ngram/`): family-conditioned counting model (default
  5-gram) with stupid-backoff to shorter contexts. The main classic baseline.

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

The local and Slurm helper scripts run the same Python entry points for the
three data stages:

```bash
# Local
local_scripts/generate_valid_datasets.sh
local_scripts/generate_augmented_datasets.sh
local_scripts/create_mixed_dataset_splits.sh --force
local_scripts/make_eval_sets.sh

# Slurm
sbatch slurm/generate_valid_datasets.sbatch
sbatch slurm/generate_augmented_datasets.sbatch
sbatch slurm/create_mixed_dataset_splits.sbatch
sbatch slurm/create_dataset_splits.sbatch
sbatch slurm/make_eval_sets.sbatch
```

For richer training, use the mixed split pipeline. It builds
`mixed_s<size>_v40_a60` datasets with 40% vanilla `generate_sequences.py`
records and 60% augmented records in `train.csv`, `valid.csv`, and the
compatibility `test.csv`. It also writes `test_standard.csv` from vanilla-only
records and `test_diverse.csv` from augmented-only records. Eval generation for
mixed datasets creates both
`data/eval/<dataset>/holdout_<family>/standard/{id,ood}/` and
`data/eval/<dataset>/holdout_<family>/diverse/{id,ood}/`.

Until the organizers distribute the real eval files, `make_eval_set.py`
synthesises local held-out eval sets under
`data/eval/<dataset>/holdout_<family>/{id,ood}/` for legacy datasets, or under
the `standard/` and `diverse/` suites for mixed datasets, using prefix cuts for
Tasks 1 & 2 and validator-flagged perturbations for Task 3:

```bash
# Build all dataset-size x holdout-family eval sets
uv run python scripts/make_all_eval_sets.py

# Train and evaluate classic baselines on the holdout setup
uv run python scripts/run_holdout_experiments.py \
  --datasets mixed_s005k_v40_a60 \
  --holdout-families ic \
  --models most_frequent ngram \
  --views standard/id standard/ood diverse/id diverse/ood

# Score a submission against ground truth (mirrors the organizer CLI)
uv run python scripts/eval_metrics.py --task next_step \
  --ground-truth data/eval/valid_s005k/holdout_ic/ood/nextstep_truth.csv \
  --predictions  outputs/preds/valid_s005k/holdout_ic/ood/ngram/nextstep.csv \
  --eval-input   data/eval/valid_s005k/holdout_ic/ood/eval_input_valid.csv
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

  The holdout workflow writes these prediction files under
  `outputs/preds/<dataset>/holdout_<family>/{id,ood}/<method>/` (see
  [Evaluation](#evaluation)).
- Training artifacts such as checkpoints, logs, and loss curves.
- Scores from the organizer `eval_metrics.py` script when it is available; until
  then, self-score locally with `scripts/eval_metrics.py`.
- A baseline-vs-trained comparison on identical inputs (the shared splits and
  eval set make this direct).

Full copied instructions are in [docs/submission/SUBMISSION.md](docs/submission/SUBMISSION.md). A starter report is at [REPORT.md](REPORT.md).
