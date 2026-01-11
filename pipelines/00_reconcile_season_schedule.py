from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

API_WEB = "https://api-web.nhle.com/v1"
STATSAPI = "https://statsapi.web.nhl.com/api/v1"

# Fallback list (adds defunct codes that matter for older seasons)
FALLBACK_TEAMS = sorted({t.lower() for t in [
    # current
    "ANA","ARI","BOS","BUF","CAR","CBJ","CGY","CHI","COL","DAL","DET","EDM","FLA","LAK","MIN","MTL","NJD","NSH","NYI","NYR","OTT","PHI","PIT","SEA","SJS","STL","TBL","TOR","VAN","VGK","WPG","WSH",
    # historical/defunct/old abbreviations
    "ATL","PHX"
]})

def get_team_codes() -> List[str]:
    """
    Try to get all NHL team abbreviations from the legacy stats API.
    If unavailable, fall back to a curated list.
    """
    try:
        r = requests.get(f"{STATSAPI}/teams", timeout=20)
        r.raise_for_status()
        j = r.json()
        teams = []
        for t in j.get("teams", []):
            ab = (t.get("abbreviation") or "").strip()
            if len(ab) == 3:
                teams.append(ab.lower())
        teams = sorted(set(teams))
        return teams if teams else FALLBACK_TEAMS
    except Exception:
        return FALLBACK_TEAMS

def game_type_from_id(game_id: str) -> int:
    # Game ID is 10 digits: first 4 = season start year; next 2 = type (01 preseason, 02 reg, 03 playoffs, etc). :contentReference[oaicite:2]{index=2}
    s = str(game_id)
    if len(s) != 10:
        return -1
    try:
        return int(s[4:6])
    except Exception:
        return -1

def season_start_year_from_label(season_label: str) -> str:
    return str(season_label)[:4]

def fetch_team_season_schedule(team: str, season_label: str, cache_dir: Path) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{team}_{season_label}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    # /v1/club-schedule-season/{team}/{season} is documented by multiple public references. :contentReference[oaicite:3]{index=3}
    url = f"{API_WEB}/club-schedule-season/{team}/{season_label}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.05)
    return data

def walk_games(obj: Any, out: List[Dict[str, Any]]) -> None:
    """
    Recursively find dicts that look like a game record containing a 10-digit id.
    """
    if isinstance(obj, dict):
        # Common structures: {"games":[{...}]} or nested game objects.
        if "id" in obj and str(obj["id"]).isdigit() and len(str(obj["id"])) == 10:
            out.append(obj)
        for v in obj.values():
            walk_games(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk_games(v, out)

def extract_schedule_rows(season_label: str, team_codes: List[str]) -> pd.DataFrame:
    start_year = season_start_year_from_label(season_label)
    cache_dir = Path("data/raw/season_schedules") / season_label

    rows = []
    seen = set()

    for team in team_codes:
        try:
            data = fetch_team_season_schedule(team, season_label, cache_dir)
        except Exception:
            continue

        games: List[Dict[str, Any]] = []
        walk_games(data, games)

        for g in games:
            gid = str(g.get("id", ""))
            if not (gid.isdigit() and len(gid) == 10):
                continue

            # Keep only gameIds that belong to this season (first 4 digits = season start year)
            if not gid.startswith(start_year):
                continue

            if gid in seen:
                continue
            seen.add(gid)

            # Prefer gameDate; fallback to startTimeUTC date portion if present
            gd = g.get("gameDate")
            if not gd:
                st = g.get("startTimeUTC") or g.get("startTimeUtc")
                if isinstance(st, str) and len(st) >= 10:
                    gd = st[:10]

            rows.append({
                "season": season_label,
                "game_id": int(gid),
                "game_type": game_type_from_id(gid),
                "game_date": gd,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort by date if present
    if "game_date" in df.columns:
        df = df.sort_values(["game_date", "game_id"], na_position="last")
    else:
        df = df.sort_values(["game_id"])
    return df.reset_index(drop=True)

def local_game_ids(season_label: str) -> set[int]:
    root = Path("data/raw/season") / season_label / "games"
    if not root.exists():
        return set()
    ids = set()
    for p in root.glob("*"):
        if p.is_dir() and p.name.isdigit():
            ids.add(int(p.name))
    return ids

def write_parquets(df_full: pd.DataFrame, season_label: str) -> Tuple[Path, Path, List[int]]:
    outdir = Path("data/processed")
    outdir.mkdir(parents=True, exist_ok=True)

    full_path = outdir / f"schedule_{season_label}.parquet"
    df_full.to_parquet(full_path, index=False)

    have = local_game_ids(season_label)
    expected = set(df_full["game_id"].astype(int).tolist()) if not df_full.empty else set()
    missing = sorted(list(expected - have))

    df_missing = df_full[df_full["game_id"].isin(missing)].copy()
    miss_path = outdir / f"schedule_{season_label}_missing.parquet"
    df_missing.to_parquet(miss_path, index=False)

    return full_path, miss_path, missing

def run_download_missing(miss_path: Path, season_label: str) -> None:
    # Use your existing downloader; it should skip already-present games unless --force is used.
    import subprocess
    subprocess.run([
        "python", "pipelines/02_pull_game_data.py",
        "--schedule_parquet", str(miss_path),
        "--season_label", season_label
    ], check=True)

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build authoritative season schedule (no dates) and download only missing game JSON.")
    ap.add_argument("--season_label", required=True, help="Season in YYYYYYYY format (e.g., 20192020).")
    ap.add_argument("--download_missing", action="store_true", help="If set, downloads only missing games using pipelines/02_pull_game_data.py")
    args = ap.parse_args(argv)

    teams = get_team_codes()
    df = extract_schedule_rows(args.season_label, teams)

    full_path, miss_path, missing = write_parquets(df, args.season_label)

    print(f"\nSeason {args.season_label}:")
    print(f"- schedule written: {full_path}")
    print(f"- missing schedule: {miss_path}")
    print(f"- expected games: {len(df):,}")
    print(f"- missing games:  {len(missing):,}")

    if args.download_missing and missing:
        print("\nDownloading missing games...")
        run_download_missing(miss_path, args.season_label)
    elif args.download_missing:
        print("\nNo missing games to download.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
