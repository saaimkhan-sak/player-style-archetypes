from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
REFRESH_LOG = REPO_ROOT / "reports" / "data_refresh_log.md"


def current_nhl_season_label(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(ZoneInfo("America/New_York"))
    start_year = now.year if now.month >= 10 else now.year - 1
    return f"{start_year}{start_year + 1}"


def upcoming_nhl_season_label(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(ZoneInfo("America/New_York"))
    start_year = now.year if now.month >= 7 else now.year - 1
    return f"{start_year}{start_year + 1}"


def run(args: list[str]) -> None:
    print("\n$ " + " ".join(args))
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def command_output(args: list[str]) -> str:
    result = subprocess.run(args, cwd=REPO_ROOT, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def available_schedule_path(season: str) -> Path:
    path = REPO_ROOT / "data" / "processed" / f"schedule_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    schedule = pd.read_parquet(path)
    raw_root = REPO_ROOT / "data" / "raw" / "season" / season / "games"
    schedule = schedule[
        schedule["game_type"].isin([2, 3])
    ]
    schedule = schedule[
        schedule["game_id"].map(lambda gid: (raw_root / str(int(gid)) / "boxscore.json").exists())
        & schedule["game_id"].map(lambda gid: (raw_root / str(int(gid)) / "play_by_play.json").exists())
    ].copy()

    out_path = REPO_ROOT / "data" / "processed" / f"schedule_{season}_available.parquet"
    schedule.to_parquet(out_path, index=False)
    print(f"Available completed games: {len(schedule):,} -> {out_path.relative_to(REPO_ROOT)}")
    return out_path


def verify_app_outputs(season: str) -> dict[str, int]:
    required = [
        REPO_ROOT / "data" / "app" / f"players_forwards_{season}.parquet",
        REPO_ROOT / "data" / "app" / f"players_defense_{season}.parquet",
        REPO_ROOT / "reports" / f"archetype_cards_forwards_{season}.csv",
        REPO_ROOT / "reports" / f"archetype_cards_defense_{season}.csv",
        REPO_ROOT / "reports" / f"archetype_traits_forwards_{season}.csv",
        REPO_ROOT / "reports" / f"archetype_traits_defense_{season}.csv",
    ]
    missing = [p.relative_to(REPO_ROOT).as_posix() for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Refresh did not produce required app outputs: " + ", ".join(missing))

    row_counts: dict[str, int] = {}
    for group in ("forwards", "defense"):
        path = REPO_ROOT / "data" / "app" / f"players_{group}_{season}.parquet"
        df = pd.read_parquet(path)
        if df.empty:
            raise RuntimeError(f"{path.relative_to(REPO_ROOT)} is empty.")
        row_counts[group] = len(df)
        print(f"Verified {group}: {len(df):,} app rows")
    return row_counts


def artifact_changes() -> list[str]:
    output = command_output(["git", "status", "--short", "--", "data/app", "reports", "models"])
    changes = []
    for line in output.splitlines():
        line = line.strip()
        if line and not line.endswith("reports/data_refresh_log.md"):
            changes.append(line)
    return changes


def prepend_refresh_log(season: str, available_games: int, row_counts: dict[str, int]) -> None:
    REFRESH_LOG.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ZoneInfo("America/New_York"))
    changes = artifact_changes()

    entry = [
        f"## {now:%Y-%m-%d %H:%M:%S %Z}",
        "",
        f"- Season refreshed: `{season}`",
        f"- Completed games included: `{available_games:,}`",
        f"- Forward rows: `{row_counts.get('forwards', 0):,}`",
        f"- Defense rows: `{row_counts.get('defense', 0):,}`",
        f"- Artifact changes before logging: `{len(changes)}`",
    ]
    if changes:
        entry.append("- Changed artifacts:")
        entry.extend(f"  - `{change}`" for change in changes[:50])
        if len(changes) > 50:
            entry.append(f"  - `... {len(changes) - 50} more`")
    else:
        entry.append("- Changed artifacts: `none`")
    entry.append("")

    existing = REFRESH_LOG.read_text(encoding="utf-8") if REFRESH_LOG.exists() else ""
    title = "# Data Refresh Log\n\n"
    body = existing[len(title):] if existing.startswith(title) else existing
    body = re.sub(
        rf"## {now:%Y-%m-%d} [\s\S]*?(?=\n## \d{{4}}-\d{{2}}-\d{{2}} |\Z)",
        "",
        body,
        count=1,
    ).lstrip()
    REFRESH_LOG.write_text(title + "\n".join(entry) + "\n" + body, encoding="utf-8")
    print(f"Updated refresh log -> {REFRESH_LOG.relative_to(REPO_ROOT)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh latest NHL archetype data and app artifacts.")
    parser.add_argument("--season_label", help="Override season label, e.g. 20252026.")
    parser.add_argument("--skip_moneypuck_download", action="store_true")
    parser.add_argument("--as_of_date", help="Snapshot date passed to schedule reconciliation.")
    args = parser.parse_args(argv)

    season = args.season_label or current_nhl_season_label()
    print(f"Refreshing season {season}")

    py = sys.executable
    if not args.skip_moneypuck_download:
        run([py, "scripts/download_moneypuck_game_by_game.py", "--season_label", season])

    schedule_args = [py, "pipelines/00_reconcile_season_schedule.py", "--season_label", season, "--download_missing"]
    if args.as_of_date:
        schedule_args.extend(["--as_of_date", args.as_of_date])
    run(schedule_args)
    schedule_path = available_schedule_path(season)
    available_games = len(pd.read_parquet(schedule_path))
    if available_games == 0:
        print(f"Season {season} has no downloaded games yet; nothing else to refresh.")
        return 0

    run([py, "pipelines/03b_build_moneypuck_player_season_features.py", "--season_label", season])
    run([py, "pipelines/03_build_player_season_features_boxscore.py", "--schedule_parquet", str(schedule_path.relative_to(REPO_ROOT)), "--season_label", season])
    run([py, "pipelines/06_build_player_directory.py", "--season_label", season])
    run([py, "pipelines/04_build_model_matrices.py", "--season_label", season])
    run([py, "pipelines/05_fit_nmf_gmm.py", "--season_label", season])
    run([py, "pipelines/07_make_archetype_cards.py", "--season_label", season])
    run([py, "pipelines/08_build_app_tables.py", "--season_label", season])
    run([py, "pipelines/09_project_playoff_archetypes.py", "--season_label", season])
    run([py, "pipelines/10_build_line_combinations.py", "--season_label", season])

    row_counts = verify_app_outputs(season)
    prepend_refresh_log(season, available_games, row_counts)
    print(f"Refresh complete for {season}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
