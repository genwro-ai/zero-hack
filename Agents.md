# Agent Guidance

This repository is a Zero One Hack_01 participant repo for the Industrial AI track.

Keep changes focused on reproducible sequence-modeling work:

- Use `uv` for environment and command execution.
- Keep source code under `src/zero_hack/`.
- Keep experiment config under `configs/`.
- Keep reusable local utilities under `scripts/`.
- Keep cluster job scripts under `slurm/`.
- Keep provided Industrial track data under `data/industrial/`.
- Keep hackathon and submission instructions under `docs/`.
- Do not commit secrets, tokens, cluster credentials, or private data.
- Preserve submission artifacts and evaluation outputs in clear, named locations.
- Generated training datasets live under `data/generated/<dataset>/`.
- Holdout evaluation datasets live under `data/eval/<dataset>/holdout_<family>/{id,ood}/`.
- Method prediction CSVs live under `outputs/preds/<dataset>/holdout_<family>/{id,ood}/<method>/`.
- Metric JSON summaries live under `outputs/metrics/<dataset>/holdout_<family>/{id,ood}/<method>/`.
- Keep deep-learning training outputs/checkpoints separate from classic-baseline predictions and metrics.

Before changing data-generation or evaluation logic, read `data/industrial/generation_rules.md`.
