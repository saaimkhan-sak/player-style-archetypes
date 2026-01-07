from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nhl_api import NHLClient  # noqa: E402


def extract_full_name(js: dict) -> str:
    # landing endpoint usually uses dicts with "default"
    first = js.get("firstName")
    last = js.get("lastName")

    if isinstance(first, dict):
        first = first.get("default") or first.get("en")
    if isinstance(last, dict):
        last = last.get("default") or last.get("en")

    if isinstance(first, str) and isinstance(last, str) and first.strip() and last.strip():
        return f"{first.strip()} {last.strip()}"

    # fallback
    full = js.get("fullName")
    if isinstance(full, str) and full.strip():
        return full.strip()

    if isinstance(js.get("name"), dict):
        v = js["name"].get("default") or js["name"].get("en")
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Enrich player_directory.parquet with full names via NHL player landing endpoint.")
    ap.add_argument("--season_label", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    directory_path = Path("data/processed/player_directory.parquet")
    df = pd.read_parquet(directory_path)

    client = NHLClient()

    updated = 0
    names = []
    for pid in df["player_id"].astype(int).tolist():
        endpoint = f"/player/{pid}/landing"
        cache_rel = f"players/{pid}/landing.json"
        js = client.get_json(endpoint, cache_relpath=cache_rel, force=args.force)
        nm = extract_full_name(js)
        names.append(nm)
        if nm:
            updated += 1

    # only overwrite if we got something
    df["full_name"] = [n if isinstance(n, str) and n.strip() else old for n, old in zip(names, df.get("full_name", [""]*len(df)).tolist())]
    df.to_parquet(directory_path, index=False)

    print(f"Updated names for {updated:,} / {len(df):,} players")
    print(f"Wrote -> {directory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
