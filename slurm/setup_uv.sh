#!/bin/bash
set -euo pipefail

# Bootstrap uv for Leonardo jobs.
#
# Installs uv under $SCRATCH/.local/bin when uv is not already available.
# Network access is required the first time this runs on a fresh account.

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
  export UV_INSTALL_DIR="${UV_INSTALL_DIR:-$SCRATCH/.local/bin}"
else
  export UV_INSTALL_DIR="${UV_INSTALL_DIR:-$HOME/.local/bin}"
fi

export PATH="$UV_INSTALL_DIR:$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  mkdir -p "$UV_INSTALL_DIR"
  printf "uv not found; installing to %s\n" "$UV_INSTALL_DIR"
  if [[ -n "${SLURM_JOB_ID:-}" && -z "${HTTPS_PROXY:-${https_proxy:-}}" ]]; then
    cat >&2 <<'EOF'
uv is not installed and no HTTPS proxy is configured.

Leonardo compute nodes do not have direct internet access. Either:
  1. Install uv once from a login node:
       ./slurm/setup_uv.sh
     Then submit jobs normally, or
  2. Submit with proxy variables exported, for example:
       sbatch --export=ALL,HTTP_PROXY=...,HTTPS_PROXY=...,http_proxy=...,https_proxy=... slurm/generate_valid_datasets.sbatch

Do not commit proxy credentials to the repository.
EOF
    exit 2
  fi
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" sh
  export PATH="$UV_INSTALL_DIR:$PATH"
fi

uv --version

# Where the dependency download happens matters: Leonardo compute nodes have no
# outbound internet, so a `uv sync` that needs to fetch a wheel times out there.
# Inside a Slurm job (compute node) we install from the local cache only and
# skip the dev group (pytest/ruff/pre-commit are not needed to train). On a
# login node we allow the network so the cache can be populated.
#
# One-time prep on a LOGIN node before submitting jobs:
#   cd "$repo" && bash slurm/setup_uv.sh
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export UV_OFFLINE=1
  uv sync --python 3.12 --frozen --no-dev
else
  uv sync --python 3.12
fi
