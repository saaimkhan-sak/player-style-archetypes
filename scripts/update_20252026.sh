#!/usr/bin/env bash
set -euo pipefail

SEASON="20252026"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# 0) authoritative schedule + download missing games
python pipelines/00_reconcile_season_schedule.py --season_label "$SEASON" --download_missing

# sanity: schedule exists and has rows
python - << PY
import pandas as pd
df = pd.read_parquet("data/processed/schedule_20252026.parquet")
print("schedule rows:", len(df))
assert len(df) > 0
PY

# 1) rebuild downstream artifacts that depend on new games
python pipelines/03_build_player_season_features_boxscore.py \
  --schedule_parquet "data/processed/schedule_${SEASON}.parquet" \
  --season_label "$SEASON"

python pipelines/06_build_player_directory.py --season_label "$SEASON"
python pipelines/04_build_model_matrices.py --season_label "$SEASON"
python pipelines/05_fit_nmf_gmm.py --season_label "$SEASON"
python pipelines/07_make_archetype_cards.py --season_label "$SEASON"
python pipelines/08_build_app_tables.py --season_label "$SEASON"

echo "✅ Update complete for $SEASON"
