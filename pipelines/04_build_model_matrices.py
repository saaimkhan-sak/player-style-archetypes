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
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    for c in out.columns:
        med = float(np.nanmedian(out[c].values))
        out[c] = out[c].fillna(med)
        q1 = float(np.nanpercentile(out[c].values, 25))
        q3 = float(np.nanpercentile(out[c].values, 75))
        iqr = q3 - q1
        if iqr == 0:
            iqr = 1.0
        params["median"][c] = med
        params["iqr"][c] = iqr
        out[c] = (out[c] - med) / iqr
        out[c] = out[c].clip(-10.0, 10.0)
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

    # Keep the modeling population stable enough that short call-ups do not
    # define or distort season-level archetypes.
    min_reg_games = 15
    min_reg_toi_s = 60 * 60
    df = df[(df["reg_games"] >= min_reg_games) & (df["reg_toi_s"] >= min_reg_toi_s)].copy()

    # v2 style blocks. Role/deployment (PP/PK, faceoffs), outcomes (goals,
    # points), and duplicate NHL/MoneyPuck event copies are intentionally kept
    # out of archetype learning and remain display/context fields.
    blocks = {
        "shot_creation": [
            "mp_reg_5on5_I_F_shotAttempts_per60",
            "reg_shots_per60",
            "mp_reg_5on5_I_F_xGoalsPerAttempt",
        ],
        "interior_access": [
            "mp_reg_5on5_I_F_highDangerShots_per60",
            "mp_reg_5on5_I_F_highDangerShotShare",
            "mp_reg_5on5_I_F_rebounds_per60",
            "mp_reg_5on5_I_F_reboundxGoals_per60",
        ],
        "possession_continuation": [
            "mp_reg_5on5_I_F_playContinuedInZone_per60",
            "mp_reg_5on5_I_F_playContinuedOutsideZone_per60",
        ],
        "contextual_creation": [
            "mp_reg_5on5_OnIce_F_xGoals_per60",
            "mp_reg_5on5_OnIce_F_shotAttempts_per60",
            "mp_reg_5on5_OnIce_A_xGoals_per60",
        ],
        "disruption": [
            "reg_blocked_shots_per60",
            "reg_hits_per60","reg_takeaways_per60","reg_giveaways_per60",
            "mp_reg_5on5_penaltiesDrawn_per60",
        ],
    }

    for b, cols in blocks.items():
        for c in cols:
            if c not in df.columns:
                raise RuntimeError(f"Required style feature is absent from source snapshot: {c}")
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Split groups
    fwd = df[df["position"].isin(FORWARD_POS)].copy()
    dfd = df[df["position"].isin(DEFENSE_POS)].copy()

    outdir = Path("data/features")
    outdir.mkdir(parents=True, exist_ok=True)
    schemas = {}

    def make_matrix(sub: pd.DataFrame, name: str):
        all_cols = list(dict.fromkeys(sum([blocks[b] for b in blocks], [])))
        X = sub[all_cols].copy()

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
            "filters": {"min_reg_games": min_reg_games, "min_reg_toi_s": min_reg_toi_s},
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
