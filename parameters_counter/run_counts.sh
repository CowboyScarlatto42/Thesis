#!/usr/bin/env bash
set -euo pipefail

# Wrapper that uses conda to create/use an environment and run the counter.
# Usage: ./run_counts.sh [--conf path/to/conf.conf]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEUS_ROOT="$REPO_ROOT/NeuS"
PY_SCRIPT="$REPO_ROOT/count_neus.py"
ENV_NAME="${CONDA_ENV_NAME:-neus-torch}"

if [ ! -f "$PY_SCRIPT" ]; then
  echo "ERROR: non trovo $PY_SCRIPT" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda non trovato. Installa Miniforge/Miniconda e riprova." >&2
  exit 2
fi

## Quiet mode: suppress all non-essential output so only the counts appear on stdout

# Create env if missing (suppress output)
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  :
else
  conda create -y -n "$ENV_NAME" python=3.9 >/dev/null 2>&1 || true
fi

# Install packages quietly
conda install -y -n "$ENV_NAME" pytorch torchvision torchaudio -c pytorch -c conda-forge >/dev/null 2>&1 || true
conda run -n "$ENV_NAME" --no-capture-output pip install -U pip >/dev/null 2>&1 || true
conda run -n "$ENV_NAME" --no-capture-output pip install pyhocon >/dev/null 2>&1 || true

# Run the Python script inside the conda env, setting PYTHONPATH so NeuS is importable
# Redirect stderr to /dev/null to avoid warnings and other messages; stdout will contain only counts
conda run -n "$ENV_NAME" bash -lc "export PYTHONPATH=\"$NEUS_ROOT\":\"\$PYTHONPATH\"; PYTHONWARNINGS=ignore python3 \"$PY_SCRIPT\" \"\$@\"" 2>/dev/null

