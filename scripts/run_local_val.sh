#!/usr/bin/env bash
# =============================================================================
# run_local_val.sh
# Local fold-0 evaluation (mode = local). Reports AUC / DP20 / IoU / Failure
# on the held-out fold. Requires the local training data and 2026training.csv
# under HOTC_BASE_DIR, and the scikit-learn dependency (GroupKFold splitting).
#
# Usage:
#   bash scripts/run_local_val.sh [tracker]
# Example:
#   bash scripts/run_local_val.sh samurai_large
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(dirname "${SCRIPT_DIR}")"

# ---- Arguments ----
TRACKER="${1:-samurai_large}"   # tracker used by this submission

# Project root used by val.py to resolve dataset/training/CSV paths.
export HOTC_BASE_DIR="${HOTC_BASE_DIR:-${PKG_ROOT}/src}"
# Local SAMURAI / SAM2 repository (track.py imports the sam2 package from here).
export SAMURAI_REPO_DIR="${SAMURAI_REPO_DIR:-$HOME/Nikhil/samurai_project/hotc2026_run/samurai/sam2}"

echo "Mode     : local (fold-0)"
echo "Tracker  : ${TRACKER}"
echo "Base dir : ${HOTC_BASE_DIR}"

python "${PKG_ROOT}/src/val.py" \
  --mode local \
  --tracker "${TRACKER}" \
  --workers 1 \
  --decimals 6
