#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Activate venv
source "$REPO_DIR/.venv/bin/activate"

# 1) Build/update the app tables you serve in Streamlit
python pipelines/08_build_app_tables.py --season_label 20252026

# (Optional) run other pipeline steps here if you want:
# python pipelines/XX_something.py ...

# 2) Auto-commit + push updates (make sure TRACK_PATHS includes data/app)
./scripts/autopush.sh
