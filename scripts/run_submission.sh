#!/usr/bin/env bash
# =============================================================================
# run_submission.sh
# Generate the official submission CSV for the ranking / official split.
# This is the configuration used to produce submission_ranking2.csv
# (mode = official, tracker = samurai_large).
#
# Usage:
#   bash scripts/run_submission.sh [val_dir] [tracker]
# Example:
#   bash scripts/run_submission.sh "$HOME/Nikhil/dataset/validation/Ranking" samurai_large
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(dirname "${SCRIPT_DIR}")"

# ---- Arguments ----
VAL_DIR="${1:-$HOME/Nikhil/dataset/validation/Ranking}"   # official / ranking split root
TRACKER="${2:-samurai_large}"                              # tracker used by this submission

# ---- Paths (edit / export to match your machine) ----
OUTPUT="${OUTPUT:-${PKG_ROOT}/results/submission_ranking2.csv}"

# Project root used by val.py to resolve dataset paths.
export HOTC_BASE_DIR="${HOTC_BASE_DIR:-${PKG_ROOT}/src}"
export HOTC_VAL_DIR="${VAL_DIR}"
# Local SAMURAI / SAM2 repository (track.py imports the sam2 package from here).
export SAMURAI_REPO_DIR="${SAMURAI_REPO_DIR:-$HOME/Nikhil/samurai_project/hotc2026_run/samurai/sam2}"

mkdir -p "$(dirname "${OUTPUT}")"

echo "Mode     : official"
echo "Tracker  : ${TRACKER}"
echo "Val dir  : ${VAL_DIR}"
echo "Output   : ${OUTPUT}"

python "${PKG_ROOT}/src/val.py" \
  --mode official \
  --tracker "${TRACKER}" \
  --val-dir "${VAL_DIR}" \
  --output "${OUTPUT}" \
  --workers 1 \
  --decimals 6
