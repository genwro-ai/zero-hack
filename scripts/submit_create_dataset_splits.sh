#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/slurm
sbatch "$@" slurm/create_dataset_splits.sbatch
