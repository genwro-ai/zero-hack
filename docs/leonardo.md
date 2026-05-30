# Leonardo Notes

This repo is intended to run on LEONARDO with Slurm.

The hackathon reservation used by the Slurm scripts is:

```text
s_tra_ncc
```

## Storage

Keep the repository checkout under `$SCRATCH` on Leonardo. The Slurm scripts write generated datasets inside the repo by default.

Default generated-data location in the Slurm scripts:

```text
data/generated
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
account: euhpc_d30_031
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
5k, 10k, 20k, 100k
mosfet, igbt, ic
```

Useful overrides:

```bash
# Store generated data somewhere else, if needed.
sbatch --export=ALL,OUTPUT_ROOT=$SCRATCH/zero-hack-data/generated slurm/generate_valid_datasets.sbatch

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
# Split data stored outside the repo.
sbatch --export=ALL,OUTPUT_ROOT=$SCRATCH/zero-hack-data/generated slurm/create_dataset_splits.sbatch

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

The Slurm jobs source:

```bash
slurm/setup_uv.sh
```

This script:

- installs `uv` under `$SCRATCH/.local/bin` if `uv` is not available
- uses `$SCRATCH/.cache/uv` as the uv cache
- runs `uv sync --python 3.12`

The first run may need network access to install `uv` and Python dependencies.
Leonardo compute nodes do not have direct internet access. If `uv` is not already
installed under `$SCRATCH/.local/bin`, either install it once beforehand or submit
with proxy variables exported:

```bash
sbatch \
  --export=ALL,HTTP_PROXY=...,HTTPS_PROXY=...,http_proxy=...,https_proxy=... \
  slurm/generate_valid_datasets.sbatch
```

Do not commit proxy credentials to the repository.

You can run setup once from a login node:

```bash
./slurm/setup_uv.sh
```

Or run setup as a Slurm job:

```bash
sbatch slurm/setup_uv.sbatch
```

For Slurm-based setup, pass proxy variables if `uv` is not already installed:

```bash
sbatch \
  --export=ALL,HTTP_PROXY=...,HTTPS_PROXY=...,http_proxy=...,https_proxy=... \
  slurm/setup_uv.sbatch
```

The Slurm scripts try to load:

```bash
module load python/3.11.7
module load profile/deeplrn
```

The project itself uses `uv` and Python 3.12. If the module setup conflicts with the environment, disable module loading:

```bash
sbatch --export=ALL,LEONARDO_LOAD_MODULES=0 slurm/generate_valid_datasets.sbatch
```
