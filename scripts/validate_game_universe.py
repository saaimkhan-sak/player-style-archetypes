#!/usr/bin/env python3
"""Validate that one release has a complete official/local game universe."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd


def _ids(frame: pd.DataFrame, game_type: int) -> set[int]:
    return set(frame.loc[frame["game_type"].eq(game_type), "game_id"].astype(int))


def _has_endpoint(root: Path, game_id: int, name: str) -> bool:
    path = root / str(game_id) / name
    return path.exists() and path.stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season_label", required=True)
    parser.add_argument("--as_of_date", required=True, help="ISO date for the immutable snapshot.")
    parser.add_argument("--schedule_parquet", help="Optional authoritative schedule parquet.")
    parser.add_argument("--require_regular_games", type=int)
    parser.add_argument("--require_playoff_games", type=int)
    parser.add_argument("--require_final", action="store_true")
    parser.add_argument("--allow_missing_pbp", action="store_true")
    args = parser.parse_args()

    schedule_path = Path(args.schedule_parquet or f"data/processed/schedule_{args.season_label}.parquet")
    schedule = pd.read_parquet(schedule_path)
    required = {"game_id", "game_type", "game_date"}
    missing_columns = required - set(schedule.columns)
    if missing_columns:
        raise SystemExit(f"Schedule is missing required columns: {sorted(missing_columns)}")

    schedule["game_id"] = pd.to_numeric(schedule["game_id"], errors="coerce").astype("Int64")
    schedule = schedule.dropna(subset=["game_id"]).copy()
    if schedule["game_id"].duplicated().any():
        raise SystemExit("Schedule contains duplicate game IDs.")

    snapshot_date = date.fromisoformat(args.as_of_date)
    past = schedule[pd.to_datetime(schedule["game_date"], errors="coerce").dt.date <= snapshot_date]
    if args.require_final and "game_state" in schedule.columns:
        non_final = past[past["game_state"].astype(str).str.upper().isin({"FUT", "PRE"})]
        if not non_final.empty:
            raise SystemExit(f"Past snapshot includes non-final games: {non_final['game_id'].astype(int).tolist()[:10]}")

    root = Path("data/raw") / "season" / args.season_label / "games"
    missing_boxscore = [int(gid) for gid in past["game_id"] if not _has_endpoint(root, int(gid), "boxscore.json")]
    missing_pbp = [int(gid) for gid in past["game_id"] if not _has_endpoint(root, int(gid), "play_by_play.json")]
    if missing_boxscore or (missing_pbp and not args.allow_missing_pbp):
        raise SystemExit(
            f"Endpoint coverage failed: missing boxscores={len(missing_boxscore)}, "
            f"missing play-by-play={len(missing_pbp)}"
        )

    regular = _ids(schedule, 2)
    playoffs = _ids(schedule, 3)
    print(json.dumps({
        "season": args.season_label,
        "snapshotAsOf": args.as_of_date,
        "regularGames": len(regular),
        "playoffGames": len(playoffs),
        "pastGames": len(past),
        "missingBoxscores": missing_boxscore,
        "missingPlayByPlay": missing_pbp,
        "cupFinalIdsPresent": all(gid in playoffs for gid in range(2025030411, 2025030417)),
    }, indent=2))

    if args.require_regular_games is not None and len(regular) != args.require_regular_games:
        raise SystemExit(f"Expected {args.require_regular_games} regular games, found {len(regular)}")
    if args.require_playoff_games is not None and len(playoffs) != args.require_playoff_games:
        raise SystemExit(f"Expected {args.require_playoff_games} playoff games, found {len(playoffs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
