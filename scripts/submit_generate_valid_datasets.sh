#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/slurm
sbatch "$@" slurm/generate_valid_datasets.sbatch
