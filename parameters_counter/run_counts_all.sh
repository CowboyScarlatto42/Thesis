#!/usr/bin/env bash
set -euo pipefail

# Run the counting script for all HOCON config files in a directory.
# Usage: ./run_counts_all.sh [confs_dir]
# Default confs_dir: ./NeuS/confs

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFS_DIR="${1:-$REPO_ROOT/NeuS/confs}"
RUN_SINGLE="$REPO_ROOT/run_counts.sh"

if [ ! -x "$RUN_SINGLE" ]; then
  echo "ERROR: run_counts.sh non trovato o non eseguibile: $RUN_SINGLE" >&2
  echo "Rendi eseguibile con: chmod +x $RUN_SINGLE" >&2
  exit 2
fi

if [ ! -d "$CONFS_DIR" ]; then
  echo "Directory confs non trovata: $CONFS_DIR" >&2
  exit 3
fi

shopt -s nullglob
confs=("$CONFS_DIR"/*.conf)
if [ ${#confs[@]} -eq 0 ]; then
  echo "Nessun file .conf in $CONFS_DIR" >&2
  exit 4
fi

for conf in "${confs[@]}"; do
  echo "===== $conf ====="
  # call the single-run wrapper; it will be quiet and print only counts
  "$RUN_SINGLE" --conf "$conf"
  echo
done
