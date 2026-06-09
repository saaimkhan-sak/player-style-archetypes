#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
SEASON_ARG=()
if [[ "${1:-}" != "" ]]; then
  SEASON_ARG=(--season_label "$1")
fi

# Build/update the latest season app artifacts from fresh NHL + MoneyPuck data.
"$PYTHON_BIN" scripts/refresh_latest_data.py "${SEASON_ARG[@]}"

# Auto-commit + push updates (make sure TRACK_PATHS includes generated artifacts)
./scripts/autopush.sh
