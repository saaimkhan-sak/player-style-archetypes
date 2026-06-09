from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd


BASE_URL = "https://moneypuck.com/moneypuck/playerData/teamPlayerGameByGame"
KINDS = {
    "skaters": "game-by-game-player-data-{start_year}.csv",
    "lines": "game-by-game-line-data-{start_year}.csv",
}
SEASON_TYPES = ("regular", "playoffs")
USER_AGENT = "player-style-archetypes-data-refresh/1.0"


def season_start_year(season_label: str) -> int:
    season_label = str(season_label).strip()
    if not re.fullmatch(r"\d{8}", season_label):
        raise ValueError(f"Expected season label like 20252026, got {season_label!r}")
    return int(season_label[:4])


def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def csv_links(index_url: str) -> list[str]:
    html = fetch_text(index_url)
    names = sorted(set(re.findall(r'href="([^"]+\.csv)"', html, flags=re.IGNORECASE)))
    return [urljoin(index_url, name) for name in names]


def read_csv_url(url: str, retries: int, sleep_s: float) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        req = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(req, timeout=120) as response:
                return pd.read_csv(response)
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt >= retries:
                raise
            wait_s = max(sleep_s, 1.0) * (2 ** attempt) * 10
            print(f"Rate limited by MoneyPuck; waiting {wait_s:.0f}s before retrying {url}")
            time.sleep(wait_s)
        except URLError as exc:
            last_error = exc
            if attempt >= retries:
                raise
            wait_s = max(sleep_s, 1.0) * (2 ** attempt) * 5
            print(f"Network error from MoneyPuck; waiting {wait_s:.0f}s before retrying {url}: {exc}")
            time.sleep(wait_s)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def download_kind(start_year: int, kind: str, out_dir: Path, sleep_s: float, retries: int) -> Path:
    pieces: list[pd.DataFrame] = []

    for season_type in SEASON_TYPES:
        index_url = f"{BASE_URL}/{start_year}/{season_type}/{kind}/"
        try:
            links = csv_links(index_url)
        except HTTPError as exc:
            if exc.code == 404:
                print(f"Skipping missing MoneyPuck directory: {index_url}")
                continue
            raise
        except URLError as exc:
            raise RuntimeError(f"Could not read MoneyPuck directory {index_url}: {exc}") from exc

        if not links:
            print(f"No MoneyPuck CSVs found at {index_url}")
            continue

        print(f"Downloading {len(links)} MoneyPuck {season_type} {kind} files...")
        for url in links:
            df = read_csv_url(url, retries=retries, sleep_s=sleep_s)
            if not df.empty:
                pieces.append(df)
            time.sleep(sleep_s)

    if not pieces:
        raise RuntimeError(f"No MoneyPuck {kind} data found for {start_year}.")

    out = pd.concat(pieces, ignore_index=True)
    if "season" not in out.columns:
        out.insert(3 if "gameId" in out.columns else 0, "season", start_year)
    if "gameId" in out.columns:
        dedupe_cols = [c for c in ("gameId", "playerId", "lineId", "situation", "playerTeam") if c in out.columns]
        if dedupe_cols:
            out = out.drop_duplicates(dedupe_cols)
    out = out.sort_values([c for c in ("season", "gameId", "playerTeam", "playerId", "lineId", "situation") if c in out.columns])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / KINDS[kind].format(start_year=start_year)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out):,} rows -> {out_path}")
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download current-season MoneyPuck team player game-by-game CSVs and combine them."
    )
    parser.add_argument("--season_label", required=True, help="Season label like 20252026.")
    parser.add_argument("--out_dir", default="data/raw/moneypuck")
    parser.add_argument("--kinds", nargs="*", choices=sorted(KINDS), default=sorted(KINDS))
    parser.add_argument("--sleep_s", type=float, default=1.5, help="Pause between MoneyPuck CSV downloads.")
    parser.add_argument("--retries", type=int, default=4, help="Retries for rate-limited or transient downloads.")
    args = parser.parse_args(argv)

    start_year = season_start_year(args.season_label)
    out_dir = Path(args.out_dir)
    for kind in args.kinds:
        download_kind(start_year, kind, out_dir, sleep_s=args.sleep_s, retries=args.retries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
