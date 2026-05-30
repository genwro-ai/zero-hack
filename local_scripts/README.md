# Local Workflow Helpers

These shell wrappers run the same Python entry points used by the Slurm jobs.
Keep workflow logic in `scripts/`; keep these files as local convenience
wrappers.

## Pipeline

```bash
# 1. Generate raw vanilla and augmented sequence datasets.
local_scripts/generate_valid_datasets.sh
local_scripts/generate_augmented_datasets.sh

# 2. Create 40/60 vanilla/augmented mixed splits.
local_scripts/create_mixed_dataset_splits.sh --force

# 3. Create standard and diverse evaluation datasets.
local_scripts/make_eval_sets.sh
```

The matching Slurm jobs are:

```bash
sbatch slurm/generate_valid_datasets.sbatch
sbatch slurm/generate_augmented_datasets.sbatch
sbatch slurm/create_mixed_dataset_splits.sbatch
sbatch slurm/make_eval_sets.sbatch
```

The shared defaults are:

- generated raw data and splits: `data/generated/<dataset>/...`
- mixed datasets: `mixed_s005k_v40_a60`, `mixed_s010k_v40_a60`,
  `mixed_s020k_v40_a60`, `mixed_s100k_v40_a60`, `mixed_s500k_v40_a60`,
  `mixed_s1000k_v40_a60`
- eval datasets:
  `data/eval/<dataset>/holdout_<family>/{standard,diverse}/{id,ood}/...`
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
