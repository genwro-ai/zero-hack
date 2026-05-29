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
- Training artifacts such as checkpoints, logs, and loss curves.
- Scores from the organizer `eval_metrics.py` script when it is available.
- A baseline-vs-trained comparison on identical inputs.

Full copied instructions are in [docs/submission/SUBMISSION.md](docs/submission/SUBMISSION.md). A starter report is at [REPORT.md](REPORT.md).
