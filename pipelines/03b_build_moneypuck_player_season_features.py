from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


RAW_FILES = [
    Path("data/raw/moneypuck/game-by-game-player-data-2008_to_2024.csv"),
    Path("data/raw/moneypuck/game-by-game-player-data-2025.csv"),
]

SITUATIONS = ["all", "5on5", "5on4", "4on5"]

IDENTITY_COLS = [
    "playerId",
    "name",
    "gameId",
    "season",
    "playerTeam",
    "position",
    "situation",
]

NUMERIC_COLS = [
    "icetime",
    "shifts",
    "gameScore",
    "onIce_xGoalsPercentage",
    "onIce_corsiPercentage",
    "onIce_fenwickPercentage",
    "I_F_xOnGoal",
    "I_F_xGoals",
    "I_F_xRebounds",
    "I_F_primaryAssists",
    "I_F_secondaryAssists",
    "I_F_shotsOnGoal",
    "I_F_missedShots",
    "I_F_blockedShotAttempts",
    "I_F_shotAttempts",
    "I_F_goals",
    "I_F_rebounds",
    "I_F_reboundGoals",
    "I_F_playContinuedInZone",
    "I_F_playContinuedOutsideZone",
    "I_F_savedShotsOnGoal",
    "I_F_savedUnblockedShotAttempts",
    "penalties",
    "I_F_faceOffsWon",
    "I_F_hits",
    "I_F_takeaways",
    "I_F_giveaways",
    "I_F_lowDangerShots",
    "I_F_mediumDangerShots",
    "I_F_highDangerShots",
    "I_F_lowDangerxGoals",
    "I_F_mediumDangerxGoals",
    "I_F_highDangerxGoals",
    "I_F_scoreAdjustedShotsAttempts",
    "I_F_unblockedShotAttempts",
    "I_F_dZoneGiveaways",
    "I_F_xGoalsFromActualReboundsOfShots",
    "I_F_reboundxGoals",
    "I_F_xGoals_with_earned_rebounds",
    "I_F_oZoneShiftStarts",
    "I_F_dZoneShiftStarts",
    "I_F_neutralZoneShiftStarts",
    "I_F_flyShiftStarts",
    "faceoffsWon",
    "faceoffsLost",
    "penalityMinutes",
    "penalityMinutesDrawn",
    "penaltiesDrawn",
    "shotsBlockedByPlayer",
    "OnIce_F_xOnGoal",
    "OnIce_F_xGoals",
    "OnIce_F_shotsOnGoal",
    "OnIce_F_shotAttempts",
    "OnIce_F_goals",
    "OnIce_F_rebounds",
    "OnIce_F_highDangerShots",
    "OnIce_F_highDangerxGoals",
    "OnIce_A_xOnGoal",
    "OnIce_A_xGoals",
    "OnIce_A_shotsOnGoal",
    "OnIce_A_shotAttempts",
    "OnIce_A_goals",
    "OnIce_A_rebounds",
    "OnIce_A_highDangerShots",
    "OnIce_A_highDangerxGoals",
    "OffIce_F_xGoals",
    "OffIce_A_xGoals",
    "OffIce_F_shotAttempts",
    "OffIce_A_shotAttempts",
    "xGoalsForAfterShifts",
    "xGoalsAgainstAfterShifts",
    "corsiForAfterShifts",
    "corsiAgainstAfterShifts",
]

RATE_COLS = [
    "I_F_xOnGoal",
    "I_F_xGoals",
    "I_F_shotsOnGoal",
    "I_F_shotAttempts",
    "I_F_goals",
    "I_F_rebounds",
    "I_F_reboundGoals",
    "I_F_playContinuedInZone",
    "I_F_playContinuedOutsideZone",
    "I_F_hits",
    "I_F_takeaways",
    "I_F_giveaways",
    "I_F_lowDangerShots",
    "I_F_mediumDangerShots",
    "I_F_highDangerShots",
    "I_F_lowDangerxGoals",
    "I_F_mediumDangerxGoals",
    "I_F_highDangerxGoals",
    "I_F_reboundxGoals",
    "I_F_xGoals_with_earned_rebounds",
    "penaltiesDrawn",
    "shotsBlockedByPlayer",
    "OnIce_F_xGoals",
    "OnIce_F_shotAttempts",
    "OnIce_A_xGoals",
    "OnIce_A_shotAttempts",
    "xGoalsForAfterShifts",
    "xGoalsAgainstAfterShifts",
]


def season_label_from_start_year(start_year: int | str) -> str:
    start = int(start_year)
    return f"{start}{start + 1}"


def norm_pos(p: object) -> str:
    p = ("" if p is None else str(p)).upper().strip()
    if p == "L":
        return "LW"
    if p == "R":
        return "RW"
    return p if p else "UNK"


def mode_or_first(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    if s.empty:
        return ""
    mode = s.mode()
    return mode.iloc[0] if not mode.empty else s.iloc[0]


def add_per60(df: pd.DataFrame, col: str, toi_col: str) -> pd.Series:
    hours = (df[toi_col] / 3600.0).replace({0: np.nan})
    return df[col] / hours


def load_moneypuck_rows(season_label: str) -> pd.DataFrame:
    start_year = int(str(season_label)[:4])
    pieces = []
    usecols = IDENTITY_COLS + NUMERIC_COLS

    for path in RAW_FILES:
        if not path.exists():
            continue
        part = pd.read_csv(path, usecols=lambda c: c in usecols)
        part = part[part["season"].astype(int) == start_year].copy()
        if not part.empty:
            pieces.append(part)

    if not pieces:
        return pd.DataFrame(columns=usecols)

    df = pd.concat(pieces, ignore_index=True)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["game_type"] = df["gameId"].astype(str).str[4:6].astype(int)
    df["season"] = season_label_from_start_year(start_year)
    df["player_id"] = pd.to_numeric(df["playerId"], errors="coerce").astype("Int64")
    df["position"] = df["position"].map(norm_pos)
    return df.dropna(subset=["player_id"]).copy()


def aggregate_split(df: pd.DataFrame, game_type: int, prefix: str) -> pd.DataFrame:
    sub = df[(df["game_type"] == game_type) & (df["situation"].isin(SITUATIONS))].copy()
    if sub.empty:
        return pd.DataFrame(columns=["season", "player_id"])

    ids = (
        sub.groupby(["season", "player_id"], as_index=False)
        .agg(
            mp_name=("name", mode_or_first),
            mp_position=("position", mode_or_first),
            mp_teams=("playerTeam", lambda x: ",".join(sorted(set(x.dropna().astype(str))))),
        )
    )

    wide_parts = [ids]
    sum_cols = [c for c in NUMERIC_COLS if c in sub.columns]
    mean_cols = [c for c in ["gameScore", "onIce_xGoalsPercentage", "onIce_corsiPercentage", "onIce_fenwickPercentage"] if c in sub.columns]

    for situation in SITUATIONS:
        sit = sub[sub["situation"] == situation].copy()
        if sit.empty:
            continue

        grouped = sit.groupby(["season", "player_id"], as_index=False)[sum_cols].sum()
        games = sit.groupby(["season", "player_id"])["gameId"].nunique().reset_index(name=f"mp_{prefix}{situation}_games")
        grouped = grouped.merge(games, on=["season", "player_id"], how="left")

        for c in mean_cols:
            means = sit.groupby(["season", "player_id"])[c].mean().reset_index(name=f"{c}_avg")
            grouped = grouped.merge(means, on=["season", "player_id"], how="left")

        toi_col = "icetime"
        for c in RATE_COLS:
            if c in grouped.columns:
                grouped[f"{c}_per60"] = add_per60(grouped, c, toi_col)

        if {"I_F_highDangerShots", "I_F_shotAttempts"}.issubset(grouped.columns):
            grouped["I_F_highDangerShotShare"] = grouped["I_F_highDangerShots"] / grouped["I_F_shotAttempts"].replace({0: np.nan})
        if {"I_F_xGoals", "I_F_shotAttempts"}.issubset(grouped.columns):
            grouped["I_F_xGoalsPerAttempt"] = grouped["I_F_xGoals"] / grouped["I_F_shotAttempts"].replace({0: np.nan})
        if {"I_F_reboundxGoals", "I_F_xGoals"}.issubset(grouped.columns):
            grouped["I_F_reboundxGoalsShare"] = grouped["I_F_reboundxGoals"] / grouped["I_F_xGoals"].replace({0: np.nan})
        if {"OnIce_F_xGoals", "OnIce_A_xGoals"}.issubset(grouped.columns):
            total_xg = grouped["OnIce_F_xGoals"] + grouped["OnIce_A_xGoals"]
            grouped["OnIce_xGoalsPercentage_calc"] = grouped["OnIce_F_xGoals"] / total_xg.replace({0: np.nan})
        if {"faceoffsWon", "faceoffsLost"}.issubset(grouped.columns):
            total_fo = grouped["faceoffsWon"] + grouped["faceoffsLost"]
            grouped["faceoffPct"] = grouped["faceoffsWon"] / total_fo.replace({0: np.nan}) * 100.0

        rename = {
            c: f"mp_{prefix}{situation}_{c}"
            for c in grouped.columns
            if c not in {"season", "player_id"}
        }
        grouped = grouped.rename(columns=rename).fillna(0.0)
        wide_parts.append(grouped)

    out = wide_parts[0]
    for part in wide_parts[1:]:
        out = out.merge(part, on=["season", "player_id"], how="outer")
    return out.fillna(0.0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate MoneyPuck game-by-game player rows into player-season advanced features.")
    ap.add_argument("--season_label", required=True, help="Season in YYYYYYYY format, e.g. 20252026.")
    args = ap.parse_args(argv)

    df = load_moneypuck_rows(args.season_label)
    if df.empty:
        raise RuntimeError(f"No MoneyPuck rows found for {args.season_label}.")

    reg = aggregate_split(df, 2, "reg_")
    po = aggregate_split(df, 3, "po_")
    out = reg.merge(po, on=["season", "player_id"], how="outer", suffixes=("", "_po_meta")).fillna(0.0)

    for c in ["mp_name_po_meta", "mp_position_po_meta", "mp_teams_po_meta"]:
        if c in out.columns:
            out = out.drop(columns=c)

    outdir = Path("data/features")
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"player_season_moneypuck_{args.season_label}.parquet"
    out.to_parquet(out_path, index=False)

    print(f"Saved MoneyPuck player-season features -> {out_path}")
    print(f"Rows: {len(out):,}")
    print(f"Columns: {len(out.columns):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
