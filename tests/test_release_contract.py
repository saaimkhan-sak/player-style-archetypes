import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (ROOT / "data/processed/schedule_20252026.parquet").exists(),
    reason="release data is not present in a lightweight checkout",
)
def test_2025_26_game_universe_is_closed():
    schedule = pd.read_parquet(ROOT / "data/processed/schedule_20252026.parquet")
    assert len(schedule[schedule.game_type == 2]) == 1312
    assert len(schedule[schedule.game_type == 3]) == 82
    playoff_ids = set(schedule.loc[schedule.game_type == 3, "game_id"].astype(int))
    assert set(range(2025030411, 2025030417)).issubset(playoff_ids)


@pytest.mark.skipif(
    not (ROOT / "web/data/playoffs.json").exists(),
    reason="web bundle is not present in a lightweight checkout",
)
def test_public_playoff_rows_have_real_samples():
    rows = json.loads((ROOT / "web/data/playoffs.json").read_text())
    latest = [row for row in rows if row["season"] == "20252026"]
    assert latest
    assert all(float(row["playoffGames"]) >= 5 for row in latest)
    assert all(row["sampleReliability"] in {"high", "medium"} for row in latest)


@pytest.mark.skipif(
    not (ROOT / "web/data/manifest.json").exists(),
    reason="release manifest is not present in a lightweight checkout",
)
def test_release_manifest_is_source_closed():
    manifest = json.loads((ROOT / "web/data/manifest.json").read_text())
    assert manifest["snapshotId"].startswith("style-lab-")
    assert manifest["qualityGateStatus"] == "passed"
    assert manifest["playoffDataThrough"] == "2026-06-14"
