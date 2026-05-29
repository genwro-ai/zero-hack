# Claude Guidance

This repository supports our Zero One Hack_01 Industrial AI submission.

Default workflow:

- Use `uv sync` and `uv run ...` for Python commands.
- Prefer small, reproducible scripts over notebook-only logic.
- Put reusable local scripts in `scripts/` and cluster job scripts in `slurm/`.
- Put experiment or training configuration in `configs/`.
- Keep model outputs, prediction CSVs, plots, and logs under `outputs/` or `reports/`.
- Reference copied track instructions in `docs/track/` and `data/industrial/generation_rules.md`.
- Keep `README.md` and `REPORT.md` honest about what runs and what is still pending.

Do not add API keys, credentials, or generated private cluster files to the repository.
