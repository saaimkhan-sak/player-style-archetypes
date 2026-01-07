from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

# --- Make repo root importable (so "from src..." works when running scripts directly)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nhl_api import NHLClient  # noqa: E402


def parse_iso_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pull NHL schedule by date range (cached) and save a master game_id list."
    )
    parser.add_argument("--season_label", required=True, help="Label for outputs, e.g., 20232024")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--game_types",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of gameType ints to keep (e.g., 2 for regular season). If omitted, keep all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached files exist.",
    )

    args = parser.parse_args(argv)

    season_label: str = args.season_label
    start: date = parse_iso_date(args.start)
    end: date = parse_iso_date(args.end)
    keep_game_types = set(args.game_types) if args.game_types else None

    client = NHLClient()

    rows = []
    for d in daterange(start, end):
        endpoint = f"/schedule/{d.isoformat()}"
        cache_rel = f"season/{season_label}/schedule/{d.isoformat()}.json"
        js = client.get_json(endpoint, cache_relpath=cache_rel, force=args.force)

        # schedule payload shape may vary; gameWeek is common in api-web.nhle.com/v1 responses
        for day in js.get("gameWeek", []):
            for g in day.get("games", []):
                gt = g.get("gameType")
                if keep_game_types is not None and gt not in keep_game_types:
                    continue

                rows.append(
                    {
                        "game_id": g.get("id"),
                        "game_date": g.get("gameDate"),
                        "game_type": gt,
                        "home_abbrev": (g.get("homeTeam") or {}).get("abbrev"),
                        "away_abbrev": (g.get("awayTeam") or {}).get("abbrev"),
                        "season": season_label,
                    }
                )

    df = pd.DataFrame(rows).dropna(subset=["game_id"]).drop_duplicates("game_id")
    outdir = Path("data/processed")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"schedule_{season_label}_{start.isoformat()}_{end.isoformat()}.parquet"
    df.to_parquet(outpath, index=False)

    print(f"Saved {len(df):,} games to {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
