#!/usr/bin/env python3
"""Validate permitted MoneyPuck game-by-game downloads against the NHL universe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"playerId", "gameId", "season", "situation", "icetime", "gameDate"}


def read_inputs(season: str) -> list[Path]:
    year = season[:4]
    paths = [
        Path("data/raw/moneypuck") / f"game-by-game-player-data-{year}.csv",
        Path("data/raw/moneypuck") / f"game-by-game-player-data-{int(year) - 1}_to_{int(year) - 1}.csv",
        Path("data/raw/moneypuck") / f"game-by-game-player-data-{year}_to_{int(year) - 1}.csv",
    ]
    return [path for path in paths if path.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season_label", required=True)
    parser.add_argument("--schedule_parquet", default=None)
    parser.add_argument("--as_of_date", required=True)
    parser.add_argument("--require_playoff_games", type=int, default=None)
    args = parser.parse_args()

    schedule_path = Path(args.schedule_parquet or f"data/processed/schedule_{args.season_label}.parquet")
    schedule = pd.read_parquet(schedule_path)
    official = set(schedule.loc[schedule["game_type"].isin([2, 3]), "game_id"].astype(int))
    official_playoffs = set(schedule.loc[schedule["game_type"].eq(3), "game_id"].astype(int))
    paths = read_inputs(args.season_label)
    if not paths:
        raise SystemExit("No permitted MoneyPuck player game-by-game download found.")

    frames = []
    manifest = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frames.append(frame)
        manifest.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rowCount": len(frame),
            "maxGameDate": str(frame.get("gameDate", pd.Series(dtype=str)).max()),
        })
    data = pd.concat(frames, ignore_index=True).drop_duplicates()
    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise SystemExit(f"MoneyPuck input is missing required columns: {missing_columns}")
    data = data[data["season"].astype(str).str.contains(args.season_label[:4], na=False)]
    game_ids = set(pd.to_numeric(data["gameId"], errors="coerce").dropna().astype(int))
    missing = sorted(official - game_ids)
    extra = sorted(game_ids - official)
    duplicates = int(data.duplicated(subset=["playerId", "gameId", "situation"]).sum())
    playoff_coverage = len(official_playoffs & game_ids)
    result = {
        "season": args.season_label,
        "snapshotAsOf": args.as_of_date,
        "sourceFiles": manifest,
        "moneyPuckGameIds": len(game_ids),
        "officialGameIds": len(official),
        "missingGameIds": missing,
        "extraGameIds": extra,
        "duplicatePlayerGameSituationRows": duplicates,
        "playoffGamesCovered": playoff_coverage,
    }
    print(json.dumps(result, indent=2))
    if missing or duplicates:
        raise SystemExit("MoneyPuck coverage/uniqueness gate failed.")
    if args.require_playoff_games is not None and playoff_coverage != args.require_playoff_games:
        raise SystemExit(f"Expected {args.require_playoff_games} playoff games, found {playoff_coverage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
