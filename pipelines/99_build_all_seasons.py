from __future__ import annotations

import argparse
import subprocess
from datetime import date
from pathlib import Path
from typing import List, Tuple


def season_label(start_year: int) -> str:
    return f"{start_year}{start_year+1}"


def season_date_window(start_year: int) -> Tuple[str, str]:
    # NHL seasons generally start in early Oct and can run into late June; give a safe window.
    start = date(start_year, 7, 1).isoformat()
    end = date(start_year + 1, 7, 1).isoformat()
    return start, end


def run(cmd: List[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build multi-season NHL archetype database end-to-end.")
    ap.add_argument("--start_year", type=int, required=True, help="First season start year, e.g. 2010 for 20102011")
    ap.add_argument("--end_year", type=int, required=True, help="Last season start year, e.g. 2024 for 20242025")
    ap.add_argument("--game_types", nargs="*", type=int, default=[2, 3], help="Game types to include (2 reg, 3 playoffs).")
    ap.add_argument("--force_schedule", action="store_true", help="Re-pull schedule JSON even if cached.")
    ap.add_argument("--force_games", action="store_true", help="Re-pull per-game JSON even if cached.")
    args = ap.parse_args()

    root = Path(".")
    if not (root / "pipelines").exists():
        raise SystemExit("Run this from the project root (player-style-archetypes).")

    for y in range(args.start_year, args.end_year + 1):
        lab = season_label(y)
        start, end = season_date_window(y)

        schedule_out = f"data/processed/schedule_{lab}_{start}_{end}.parquet"

        # 01 schedule
        cmd = [
            "python", "pipelines/01_pull_schedule.py",
            "--season_label", lab,
            "--start", start,
            "--end", end,
            "--game_types",
        ] + [str(gt) for gt in args.game_types]
        if args.force_schedule:
            cmd.append("--force")
        run(cmd)

        # 02 game json
        cmd = [
            "python", "pipelines/02_pull_game_data.py",
            "--schedule_parquet", schedule_out,
            "--season_label", lab,
        ]
        if args.force_games:
            cmd.append("--force")
        run(cmd)

        # 03 features
        run([
            "python", "pipelines/03_build_player_season_features_boxscore.py",
            "--schedule_parquet", schedule_out,
            "--season_label", lab,
        ])

        # 06 directory + teams_played (from player_game_boxscore)
        run(["python", "pipelines/06_build_player_directory.py", "--season_label", lab])

        # 04 matrices
        run(["python", "pipelines/04_build_model_matrices.py", "--season_label", lab])

        # 05 nmf+gmm
        run(["python", "pipelines/05_fit_nmf_gmm.py", "--season_label", lab])

        # 07 traits/cards
        run(["python", "pipelines/07_make_archetype_cards.py", "--season_label", lab])

        # 08 app tables
        run(["python", "pipelines/08_build_app_tables.py", "--season_label", lab])

        print(f"\n✅ Finished season {lab}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
