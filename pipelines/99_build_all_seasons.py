from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


def season_label(start_year: int) -> str:
    return f"{start_year}{start_year+1}"


def run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def schedule_path(season: str) -> Path:
    return Path("data/processed") / f"schedule_{season}.parquet"


def schedule_has_games(season: str) -> bool:
    p = schedule_path(season)
    if not p.exists():
        return False
    df = pd.read_parquet(p)
    return len(df) > 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build multi-season database WITHOUT hardcoded date cutoffs.")
    ap.add_argument("--start_year", type=int, required=True, help="e.g. 2000 for 20002001")
    ap.add_argument("--end_year", type=int, required=True, help="e.g. 2025 for 20252026")
    ap.add_argument("--download_missing", action="store_true", help="Download missing games during reconciliation.")
    args = ap.parse_args(argv)

    root = Path(".")
    if not (root / "pipelines").exists():
        raise SystemExit("Run this from the project root (player-style-archetypes).")

    for y in range(args.start_year, args.end_year + 1):
        season = season_label(y)

        # Cancelled lockout season: no games
        if season == "20042005":
            print(f"\nSkipping cancelled lockout season {season}")
            continue

        print("\n==============================")
        print(f"Building season {season}")
        print("==============================")

        # 0) Authoritative schedule (no dates) + optionally download missing games
        cmd = ["python", "pipelines/00_reconcile_season_schedule.py", "--season_label", season]
        if args.download_missing:
            cmd.append("--download_missing")
        run(cmd)

        # If cancelled/empty season, skip everything else
        if not schedule_has_games(season):
            print(f"Season {season}: schedule empty → skipping.")
            continue

        sp = str(schedule_path(season))

        # 1) Build season features from the authoritative schedule
        run(["python", "pipelines/03_build_player_season_features_boxscore.py", "--schedule_parquet", sp, "--season_label", season])

        # 2) Directory + teams played (season-specific directory created by your updated 06)
        run(["python", "pipelines/06_build_player_directory.py", "--season_label", season])

        # 3) Matrices + models + reports + app tables
        run(["python", "pipelines/04_build_model_matrices.py", "--season_label", season])
        run(["python", "pipelines/05_fit_nmf_gmm.py", "--season_label", season])
        run(["python", "pipelines/07_make_archetype_cards.py", "--season_label", season])
        run(["python", "pipelines/08_build_app_tables.py", "--season_label", season])

        print(f"\n✅ Finished season {season}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
