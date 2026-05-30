# Local Workflow Helpers

These shell wrappers run the same Python entry points used by the Slurm jobs.
Keep workflow logic in `scripts/`; keep these files as local convenience
wrappers.

## Pipeline

```bash
# 1. Generate raw valid sequence datasets.
local_scripts/generate_valid_datasets.sh

# 2. Create train/valid/test splits from those raw datasets.
local_scripts/create_all_dataset_splits.sh --force

# 3. Create evaluation datasets for each dataset size and holdout family.
local_scripts/make_eval_sets.sh
```

The matching Slurm jobs are:

```bash
sbatch slurm/generate_valid_datasets.sbatch
sbatch slurm/create_dataset_splits.sbatch
sbatch slurm/make_eval_sets.sbatch
```

The shared defaults are:

- generated raw data and splits: `data/generated/<dataset>/...`
- eval datasets: `data/eval/<dataset>/holdout_<family>/{id,ood}/...`
- dataset sizes: `valid_s005k`, `valid_s010k`, `valid_s020k`,
  `valid_s100k`
- holdout families: `mosfet`, `igbt`, `ic`
- default eval sizing matches the hackathon protocol per three families:
  600 Task 1/2 rows from 100 sequences/family x 2 cuts, and 987 Task 3
  anomaly rows from 200 valid + 129 invalid sequences/family.

## Sync To Leonardo

```bash
local_scripts/rsync_to_scratch.sh <leonardo-user>
```

By default this syncs source/config/docs/scripts but excludes `.venv`, caches,
`data/generated`, and Slurm logs.
