# Leonardo Notes

This repo is intended to run on LEONARDO with Slurm.

The hackathon reservation used by the Slurm scripts is:

```text
s_tra_ncc
```

## Storage

Use the repository checkout for code and `$SCRATCH` for generated datasets.

Default generated-data location in the Slurm scripts:

```text
$SCRATCH/zero-hack/generated
```

The generated CSVs are large and are ignored by git.

## Submit Dataset Generation

Set the project account once per shell:

```bash
export LEONARDO_ACCOUNT=EUHPC_1234
```

Replace `EUHPC_1234` with the actual account.

Submit all configured size/family generation jobs:

```bash
sbatch slurm/generate_valid_datasets.sbatch
```

Generation and splitting run inside Slurm jobs, not on the login node.

The job scripts request one Leonardo Booster node with:

```text
partition: boost_usr_prod
reservation: s_tra_ncc
nodes: 1
ntasks-per-node: 1
gpus-per-task: 1
mem: 120GB
cpus-per-task: 8
time: 0:30:00 by default, override with `--time` when needed
```

This submits an array for:

```text
5k, 10k, 20k, 100k, 500k, 1000k
mosfet, igbt, ic
```

Useful overrides:

```bash
# Store generated data somewhere else.
sbatch --export=ALL,OUTPUT_ROOT=$WORK/zero-hack/generated slurm/generate_valid_datasets.sbatch

# Regenerate existing files.
sbatch --export=ALL,FORCE=1 slurm/generate_valid_datasets.sbatch

# Run an extra validation pass after writing.
sbatch --export=ALL,VALIDATE_AFTER=1 slurm/generate_valid_datasets.sbatch

# Use one array task for a short test.
sbatch --array=0-0 slurm/generate_valid_datasets.sbatch

# Override walltime when needed.
sbatch --time=24:00:00 slurm/generate_valid_datasets.sbatch
```

## Submit Dataset Splits

After raw generation finishes:

```bash
sbatch slurm/create_dataset_splits.sbatch
```

Useful overrides:

```bash
# Split data stored under $WORK.
sbatch --export=ALL,OUTPUT_ROOT=$WORK/zero-hack/generated slurm/create_dataset_splits.sbatch

# Split only one family, useful for testing.
sbatch --array=0-0 --export=ALL,FAMILIES=mosfet slurm/create_dataset_splits.sbatch

# Overwrite existing splits.
sbatch --export=ALL,FORCE=1 slurm/create_dataset_splits.sbatch
```

## Monitor

```bash
squeue --me
```

Cancel a job:

```bash
scancel <job_id>
```

Logs are written under:

```text
outputs/slurm/
```

## Software

The Slurm scripts try to load:

```bash
module load python/3.11.7
module load profile/deeplrn
```

The project itself uses `uv` and Python 3.12. If the module setup conflicts with the environment, disable module loading:

```bash
sbatch --export=ALL,LEONARDO_LOAD_MODULES=0 slurm/generate_valid_datasets.sbatch
```
