# Report

## TL;DR

We are participating in Zero One Hack_01 on the Industrial AI track. This report will document our reproducible workflow for semiconductor process-sequence modeling, including baselines, trained models, evaluation results, and submission artifacts.

## Problem

Industrial process workflows are long ordered sequences where correctness depends on process logic, not only local token frequency. The track asks whether models can learn this logic for MOSFET, IGBT, and IC fabrication sequences.

## Approach

- Start from the provided generator, grammar, and validated training sequences.
- Establish simple baselines for next-step prediction, completion, and anomaly detection.
- Train or fine-tune sequence models and compare them against the baselines.
- Report metrics by task and family, including failure cases.

## How to Run

```bash
uv sync
uv run zero-hack-info
uv run python data/industrial/generate_sequences.py --family mosfet --estimate-only
```

## Results

TBD.

## What Worked / What Did Not

TBD.

## Next Steps

TBD.

## Credits & Dependencies

This repository includes Industrial AI track material from Zero One Hack_01. Dependencies are tracked through `pyproject.toml` and `uv.lock`.
