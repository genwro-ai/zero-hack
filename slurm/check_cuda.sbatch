#!/bin/bash
#SBATCH --job-name=zero-hack-cuda-check
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of `srun` tasks executed per node
#SBATCH --gpus-per-task=1 # Number of GPUs (up to 4 on Leonardo)
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:10:00 # Short diagnostic job
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err

set -euo pipefail

# Sanity-check that the installed PyTorch can actually use CUDA on a GPU node.
#
# Run this after changing the torch version / CUDA wheel index in pyproject.toml
# to confirm the new build matches the driver before launching real training.
#
#   sbatch slurm/check_cuda.sbatch
#
# It verifies, on an allocated GPU:
#   - the NVIDIA driver / CUDA runtime visible via nvidia-smi
#   - torch.__version__ and the CUDA toolkit torch was built against
#   - torch.cuda.is_available() and the detected device(s)
#   - a real matmul on the GPU (forces a CUDA kernel launch + sync)
# The job exits non-zero if CUDA is not usable, so failure is easy to spot.
#
# Optional environment overrides:
#   RUNNER=srun                  # set RUNNER= to run without srun
#   LEONARDO_LOAD_MODULES=0      # skip `module load` on non-Leonardo hosts
#   UV_CACHE_DIR=$SCRATCH/.cache/uv

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi

if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

mkdir -p outputs/slurm

source slurm/setup_uv.sh

RUNNER="${RUNNER:-srun}"

echo "===================== nvidia-smi ====================="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found on this node"
fi

echo "===================== torch CUDA check ====================="
time ${RUNNER:+$RUNNER} uv run python - <<'PY'
import sys
import torch

print(f"torch version            : {torch.__version__}")
print(f"compiled CUDA version    : {torch.version.cuda}")
print(f"cuDNN version            : {torch.backends.cudnn.version()}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available to PyTorch on this node.", file=sys.stderr)
    raise SystemExit(1)

n = torch.cuda.device_count()
print(f"device count             : {n}")
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    print(
        f"  device[{i}]: {props.name} "
        f"(capability {props.major}.{props.minor}, "
        f"{props.total_memory / 1024**3:.1f} GiB)"
    )

# Force a real kernel launch + device synchronisation on the default device.
device = torch.device("cuda")
a = torch.randn(2048, 2048, device=device)
b = torch.randn(2048, 2048, device=device)
c = a @ b
torch.cuda.synchronize()
print(f"matmul on {device}: result sum = {c.sum().item():.4f} (OK)")
print("CUDA is working correctly.")
PY

echo "===================== done ====================="
