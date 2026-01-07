from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from tqdm import tqdm

# ensure repo root import path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nhl_api import NHLClient  # noqa: E402


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Pull per-game boxscore + play-by-play JSON for game_ids in a schedule parquet.")
    p.add_argument("--schedule_parquet", required=True, help="Path to schedule parquet produced by 01_pull_schedule.py")
    p.add_argument("--season_label", required=True, help="Season label used for caching paths, e.g., 20242025")
    p.add_argument("--limit", type=int, default=None, help="Optional limit for quick tests")
    p.add_argument("--force", action="store_true", help="Force re-download even if cached")
    args = p.parse_args(argv)

    schedule_path = Path(args.schedule_parquet)
    df = pd.read_parquet(schedule_path)

    # stable ordering
    df = df.sort_values(["game_date", "game_id"])
    if args.limit:
        df = df.head(args.limit)

    client = NHLClient()

    out_rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Pulling games"):
        game_id = int(r["game_id"])

        # --- API endpoints (api-web.nhle.com/v1)
        # These are commonly used with this base; if an endpoint changes, you only edit here.
        pbp_ep = f"/gamecenter/{game_id}/play-by-play"
        box_ep = f"/gamecenter/{game_id}/boxscore"

        pbp_cache = f"season/{args.season_label}/games/{game_id}/play_by_play.json"
        box_cache = f"season/{args.season_label}/games/{game_id}/boxscore.json"

        ok = True
        err = ""
        try:
            client.get_json(pbp_ep, cache_relpath=pbp_cache, force=args.force)
            client.get_json(box_ep, cache_relpath=box_cache, force=args.force)
        except Exception as e:
            ok = False
            err = str(e)

        out_rows.append(
            {
                "game_id": game_id,
                "game_date": r.get("game_date"),
                "game_type": r.get("game_type"),
                "home_abbrev": r.get("home_abbrev"),
                "away_abbrev": r.get("away_abbrev"),
                "pulled_ok": ok,
                "error": err,
            }
        )

    manifest = pd.DataFrame(out_rows)
    outdir = Path("data/processed")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"pulled_manifest_{args.season_label}.parquet"
    manifest.to_parquet(outpath, index=False)

    n_ok = int(manifest["pulled_ok"].sum())
    n_bad = len(manifest) - n_ok
    print(f"Done. OK={n_ok:,}  Failed={n_bad:,}. Manifest: {outpath}")
    if n_bad:
        print("Tip: rerun with --force to retry, or filter failed ids from the manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
