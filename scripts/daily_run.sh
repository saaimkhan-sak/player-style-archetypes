#!/usr/bin/env bash
set -euo pipefail

REPO="/Users/saaimkhan/Library/CloudStorage/OneDrive-HarvardUniversity/Documents/projects/passion-projects/nhl-analytics/player-style-archetypes"
cd "$REPO"

# Activate venv
source "$REPO/.venv/bin/activate"

# 1) Build/update the app tables you serve in Streamlit
python pipelines/08_build_app_tables.py --season_label 20252026

# (Optional) run other pipeline steps here if you want:
# python pipelines/XX_something.py ...

# 2) Auto-commit + push updates (make sure TRACK_PATHS includes data/app)
./scripts/autopush.sh

