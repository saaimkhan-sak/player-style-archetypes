from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def norm_pos(p: str) -> str:
    p = (p or "").upper().strip()
    if p == "L": return "LW"
    if p == "R": return "RW"
    return p


REQ_REG = ["reg_games","reg_avg_toi_min","reg_goals","reg_assists","reg_points","reg_shots","reg_plus_minus","reg_pim"]
REQ_PO  = ["po_games","po_avg_toi_min","po_goals","po_assists","po_points","po_shots","po_plus_minus","po_pim"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build joined tables for Streamlit app (REG+PO stats, teams_played, avg TOI, PIM).")
    ap.add_argument("--season_label", required=True)
    args = ap.parse_args(argv)
    season = args.season_label

    # Use season-specific directory (full names) if available
    dir_path = Path(f"data/processed/player_directory_{season}.parquet")
    if not dir_path.exists():
        raise FileNotFoundError(f"Missing {dir_path}. Run pipelines/06_build_player_directory.py --season_label {season} first.")
    directory = pd.read_parquet(dir_path)

    teams = pd.read_parquet(f"data/processed/player_season_teams_{season}.parquet")
    stats = pd.read_parquet(f"data/features/player_season_boxscore_{season}.parquet")

    outdir = Path("data/app")
    outdir.mkdir(parents=True, exist_ok=True)

    for group in ["forwards", "defense"]:
        arch = pd.read_parquet(f"data/processed/archetypes_{group}_{season}.parquet")

        df = (
            arch.merge(directory, on="player_id", how="left")
                .merge(teams, on=["season","player_id"], how="left")
                .merge(stats, on=["season","player_id"], how="left", suffixes=("", "_stats"))
        )

        df["position"] = df["position"].astype(str).map(norm_pos)
        df["full_name"] = df["full_name"].fillna(df["player_id"].astype(str))
        df["teams_played"] = df["teams_played"].fillna(df.get("primary_team", "NA"))

        # Ensure expected REG/PO columns exist even if no playoffs yet
        for c in REQ_REG + REQ_PO:
            if c not in df.columns:
                df[c] = 0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        pcols = [c for c in df.columns if c.startswith("p") and c[1:].isdigit()]
        probs = df[pcols].to_numpy(dtype=float)
        df["top_cluster"] = probs.argmax(axis=1)
        df["confidence"] = probs.max(axis=1)

        keep = [
            "season","player_id","full_name","teams_played","position",
            *REQ_REG, *REQ_PO,
            "top_cluster","confidence",
        ] + pcols

        out = df[keep].copy()
        outpath = outdir / f"players_{group}_{season}.parquet"
        out.to_parquet(outpath, index=False)
        print(f"Saved {group}: {len(out):,} rows -> {outpath}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
