from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def norm_pos(p: str) -> str:
    p = (p or "").upper().strip()
    if p == "L": return "LW"
    if p == "R": return "RW"
    return p


FORWARD_POS = {"C","LW","RW","W","F"}  # include wings
DEFENSE_POS = {"D","LD","RD"}


def robust_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    params = {"median": {}, "iqr": {}}
    out = df.copy()
    for c in out.columns:
        med = float(np.nanmedian(out[c].values))
        q1 = float(np.nanpercentile(out[c].values, 25))
        q3 = float(np.nanpercentile(out[c].values, 75))
        iqr = q3 - q1
        if iqr == 0:
            iqr = 1.0
        params["median"][c] = med
        params["iqr"][c] = iqr
        out[c] = (out[c] - med) / iqr
    return out, params


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build forward/defense modeling matrices from REGULAR SEASON player-season features.")
    p.add_argument("--season_label", required=True)
    args = p.parse_args(argv)

    df = pd.read_parquet(f"data/features/player_season_boxscore_{args.season_label}.parquet")
    df["position"] = df["position"].astype(str).map(norm_pos)

    # Use REG season only for archetype learning
    df["reg_games"] = pd.to_numeric(df.get("reg_games", 0), errors="coerce").fillna(0).astype(int)
    df["reg_toi_s"] = pd.to_numeric(df.get("reg_toi_s", 0), errors="coerce").fillna(0.0)

    # sample filters (tweak later)
    df = df[(df["reg_games"] >= 5) & (df["reg_toi_s"] >= 60*60)].copy()

    # Feature blocks from REG per60 + usage shares
    blocks = {
        "shooting_scoring": ["reg_shots_per60","reg_goals_per60","reg_assists_per60","reg_points_per60"],
        "physical_disruption": ["reg_hits_per60","reg_blocked_shots_per60","reg_takeaways_per60","reg_giveaways_per60"],
        "discipline": ["reg_pim_per60"],
        "special_teams_usage": ["reg_pp_share","reg_pk_share"],
        "faceoffs": ["reg_fo_pct","reg_fo_taken_per_game"],
    }

    for b, cols in blocks.items():
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Split groups
    fwd = df[df["position"].isin(FORWARD_POS)].copy()
    dfd = df[df["position"].isin(DEFENSE_POS)].copy()

    outdir = Path("data/features")
    outdir.mkdir(parents=True, exist_ok=True)
    schemas = {}

    def make_matrix(sub: pd.DataFrame, name: str):
        all_cols = sum([blocks[b] for b in blocks], [])
        X = sub[all_cols].copy().fillna(0.0)

        Xs, scaler = robust_scale(X)

        out = pd.concat(
            [
                sub[["season","player_id","position","reg_games","reg_toi_s"]].reset_index(drop=True),
                Xs.reset_index(drop=True),
            ],
            axis=1,
        )

        out_path = outdir / f"X_{name}_{args.season_label}.parquet"
        out.to_parquet(out_path, index=False)

        schemas[name] = {
            "blocks": blocks,
            "all_features": all_cols,
            "scaler": scaler,
            "filters": {"min_reg_games": 5, "min_reg_toi_s": 3600},
            "rows": len(out),
            "path": str(out_path),
        }
        print(f"Saved {name}: {len(out):,} rows -> {out_path}")

    make_matrix(fwd, "forwards")
    make_matrix(dfd, "defense")

    schema_path = outdir / f"feature_schema_{args.season_label}.json"
    with schema_path.open("w", encoding="utf-8") as f:
        json.dump(schemas, f, indent=2)
    print(f"Schema saved -> {schema_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
