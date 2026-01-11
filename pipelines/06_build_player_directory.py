from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import requests
from tqdm import tqdm

BASE = "https://api-web.nhle.com/v1"

def load_landing(player_id: int, cache_path: Path) -> Dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    url = f"{BASE}/player/{player_id}/landing"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.10)
    return data

def name_piece(x: Any) -> str:
    if isinstance(x, dict):
        return (x.get("default") or x.get("en") or "").strip()
    if isinstance(x, str):
        return x.strip()
    return ""

def full_name_from_landing(j: Dict[str, Any]) -> str:
    # Landing usually has firstName/lastName dicts or fullName
    first = name_piece(j.get("firstName"))
    last = name_piece(j.get("lastName"))
    if first and last:
        return f"{first} {last}".strip()
    full = name_piece(j.get("fullName"))
    if full:
        return full
    # sometimes name.default
    nm = name_piece(j.get("name"))
    return nm

def is_initial_style(name: str) -> bool:
    return bool(re.match(r"^[A-Z]\.\s", str(name).strip()))

def mode_nonnull(s: pd.Series):
    s = s.dropna().astype(str)
    s = s[s.str.strip() != ""]
    if len(s) == 0:
        return None
    return s.mode().iloc[0]

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build season-specific player directory with full names via /player/{id}/landing.")
    ap.add_argument("--season_label", required=True)
    args = ap.parse_args(argv)
    season = args.season_label

    pg_path = Path(f"data/features/player_game_boxscore_{season}.parquet")
    if not pg_path.exists():
        raise FileNotFoundError(f"Missing {pg_path}. Run pipelines/03_build_player_season_features_boxscore.py first.")

    pg = pd.read_parquet(pg_path)

    # baseline from boxscore (may be abbreviated)
    base_dir = (
        pg.groupby("player_id", as_index=False)
          .agg({
              "full_name": mode_nonnull,
              "position": mode_nonnull,
          })
    )
    base_dir["full_name"] = base_dir["full_name"].fillna("")

    # enrich names via landing endpoint
    cache_root = Path("data/raw") / "player_landing"
    rows = []
    for r in tqdm(base_dir.itertuples(index=False), total=len(base_dir), desc=f"Enriching names {season}"):
        pid = int(r.player_id)
        name = str(r.full_name)

        # Only call API if name is missing or initial-style
        if (not name.strip()) or is_initial_style(name):
            cache_path = cache_root / season / f"{pid}.json"
            try:
                j = load_landing(pid, cache_path)
                full = full_name_from_landing(j)
                if full:
                    name = full
            except Exception:
                # if API fails, keep whatever we had
                pass

        rows.append({"player_id": pid, "full_name": name, "position": r.position})

    directory = pd.DataFrame(rows)

    outdir = Path("data/processed")
    outdir.mkdir(parents=True, exist_ok=True)

    # season-specific directory (prevents seasons overwriting each other)
    dir_path = outdir / f"player_directory_{season}.parquet"
    directory.to_parquet(dir_path, index=False)

    # teams played (same as before)
    team_agg = (
        pg.groupby(["season", "player_id", "team"], as_index=False)["toi_s"]
          .sum()
          .sort_values(["season", "player_id", "toi_s"], ascending=[True, True, False])
    )

    def teams_join(g: pd.DataFrame) -> str:
        teams = g["team"].tolist()
        seen = []
        for t in teams:
            if t and t not in seen:
                seen.append(t)
        return "/".join(seen)

    teams = (
        team_agg.groupby(["season", "player_id"], as_index=False)
                .apply(lambda g: pd.Series({"teams_played": teams_join(g)}), include_groups=False)
                .reset_index(drop=True)
    )

    idx = team_agg.groupby(["season", "player_id"])["toi_s"].idxmax()
    primary = team_agg.loc[idx, ["season", "player_id", "team"]].rename(columns={"team": "primary_team"})
    pt = primary.merge(teams, on=["season", "player_id"], how="left")

    pt_path = outdir / f"player_season_teams_{season}.parquet"
    pt.to_parquet(pt_path, index=False)

    print(f"Saved:\n- {dir_path}\n- {pt_path}")
    print(f"Players in directory: {len(directory):,}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
