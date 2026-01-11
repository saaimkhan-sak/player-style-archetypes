# pipelines/07_make_archetype_cards.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPORTS_DIR = Path("reports")
PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")


def _find_player_id_col(df: pd.DataFrame) -> str:
    for c in ["player_id", "playerId", "id"]:
        if c in df.columns:
            return c
    raise KeyError("Could not find a player id column (expected one of: player_id, playerId, id).")


def _prob_cols(df: pd.DataFrame) -> List[str]:
    p = [c for c in df.columns if isinstance(c, str) and re.fullmatch(r"p\d+", c)]
    p.sort(key=lambda x: int(x[1:]))
    return p


def _ensure_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    if "season" not in df.columns:
        df = df.copy()
        df["season"] = season
    return df


def _read_first_existing(paths: List[Path]) -> Optional[pd.DataFrame]:
    for p in paths:
        if p.exists():
            return pd.read_parquet(p)
    return None


def _load_directory(season: str) -> pd.DataFrame:
    """
    Returns a df with at least: player_id, full_name, position (if available).
    """
    df = _read_first_existing([
        PROCESSED_DIR / f"player_directory_{season}.parquet",
        PROCESSED_DIR / "player_directory.parquet",
    ])
    if df is None:
        return pd.DataFrame(columns=["player_id", "full_name", "position"])

    pid = _find_player_id_col(df)
    out = df.rename(columns={pid: "player_id"}).copy()
    if "full_name" not in out.columns:
        out["full_name"] = ""
    if "position" not in out.columns:
        out["position"] = ""
    return out[["player_id", "full_name", "position"]].drop_duplicates("player_id")


def _load_player_season_teams(season: str) -> pd.DataFrame:
    """
    Returns df with: season, player_id, teams_played, primary_team (if available)
    """
    df = _read_first_existing([
        PROCESSED_DIR / f"player_season_teams_{season}.parquet",
        PROCESSED_DIR / "player_season_teams.parquet",
    ])
    if df is None:
        return pd.DataFrame(columns=["season", "player_id", "teams_played", "primary_team"])

    pid = _find_player_id_col(df)
    out = df.rename(columns={pid: "player_id"}).copy()
    out = _ensure_season(out, season)

    # Some files might contain multiple seasons; filter if so
    if "season" in out.columns:
        out = out[out["season"].astype(str) == str(season)].copy()

    if "teams_played" not in out.columns:
        out["teams_played"] = ""
    if "primary_team" not in out.columns:
        out["primary_team"] = ""

    return out[["season", "player_id", "teams_played", "primary_team"]].drop_duplicates(["season", "player_id"])


def _load_archetypes(group: str, season: str) -> pd.DataFrame:
    p = PROCESSED_DIR / f"archetypes_{group}_{season}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing archetypes file: {p}")
    df = pd.read_parquet(p)
    pid = _find_player_id_col(df)
    df = df.rename(columns={pid: "player_id"})
    df = _ensure_season(df, season)

    pcols = _prob_cols(df)
    if not pcols:
        raise ValueError(f"No probability columns p0..pK found in {p}")

    if "top_cluster" not in df.columns:
        df = df.copy()
        df["top_cluster"] = df[pcols].idxmax(axis=1).str[1:].astype(int)

    if "confidence" not in df.columns:
        df = df.copy()
        df["confidence"] = df[pcols].max(axis=1).astype(float)

    return df


def _load_features(group: str, season: str) -> pd.DataFrame:
    """
    Loads the feature matrix produced by pipeline 04 (preferred),
    otherwise tries to fall back to archetypes parquet if it already contains features.
    """
    p = FEATURES_DIR / f"X_{group}_{season}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        pid = _find_player_id_col(df)
        df = df.rename(columns={pid: "player_id"})
        df = _ensure_season(df, season)
        return df

    # Fallback: if no X file, we still try to proceed with empty features
    return pd.DataFrame(columns=["season", "player_id"])


def _feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Pick numeric feature columns, excluding obvious non-features.
    """
    exclude = {
        "season", "player_id", "full_name", "position", "teams_played", "primary_team",
        "top_cluster", "confidence",
    }
    # exclude pK columns
    for c in df.columns:
        if isinstance(c, str) and re.fullmatch(r"p\d+", c):
            exclude.add(c)

    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _weighted_trait_zscores(
    merged: pd.DataFrame,
    pcol: str,
    feat_cols: List[str],
    eps: float = 1e-9,
) -> pd.Series:
    """
    Probability-weighted mean per feature for archetype k,
    converted into z-score vs overall population.
    """
    X = merged[feat_cols].astype(float)

    # overall mean/std
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0).replace(0.0, np.nan)

    w = merged[pcol].astype(float).to_numpy()
    wsum = float(np.nansum(w))

    if not np.isfinite(wsum) or wsum <= eps:
        # fallback: unweighted
        wm = X.mean(axis=0)
    else:
        wm = (X.mul(w, axis=0).sum(axis=0) / wsum)

    z = (wm - mu) / (sd + eps)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z


def _fmt_traits(z: pd.Series, n: int, reverse: bool) -> str:
    s = z.sort_values(ascending=not reverse).head(n)
    parts = [f"{idx}({val:+.2f})" for idx, val in s.items()]
    return ",".join(parts)


def _format_prototypes(
    merged_meta: pd.DataFrame,
    pcol: str,
    n: int = 8,
) -> str:
    """
    merged_meta should already include: full_name, primary_team/teams_played
    """
    tmp = merged_meta.copy()
    tmp[pcol] = tmp[pcol].astype(float)

    tmp = tmp.sort_values(pcol, ascending=False).head(n)

    out = []
    for r in tmp.itertuples(index=False):
        name = getattr(r, "full_name", "") or ""
        pid = getattr(r, "player_id")
        team = getattr(r, "primary_team", "") or getattr(r, "teams_played", "") or ""
        prob = getattr(r, pcol)

        label = name.strip() if str(name).strip() else str(pid)
        if str(team).strip():
            label = f"{label} ({team})"
        out.append(f"{label} p={float(prob):.3f}")
    return " | ".join(out)


def build_cards_for_group(group: str, season: str, top_n_traits: int, low_n_traits: int, n_prototypes: int) -> pd.DataFrame:
    arch = _load_archetypes(group, season)
    feats = _load_features(group, season)

    pcols = _prob_cols(arch)
    K = len(pcols)

    # Merge for traits computation
    feat_cols = _feature_cols(feats)
    merged = arch[["season", "player_id"] + pcols].merge(
        feats[["season", "player_id"] + feat_cols],
        on=["season", "player_id"],
        how="inner",
    )

    # Merge for prototype display (names + teams)
    directory = _load_directory(season)
    teams = _load_player_season_teams(season)

    merged_meta = arch[["season", "player_id"] + pcols].merge(directory, on="player_id", how="left")
    merged_meta = merged_meta.merge(teams, on=["season", "player_id"], how="left")

    # Fill safe defaults
    for c in ["full_name", "position", "teams_played", "primary_team"]:
        if c not in merged_meta.columns:
            merged_meta[c] = ""
        merged_meta[c] = merged_meta[c].fillna("")

    rows = []
    for k in range(K):
        pcol = f"p{k}"
        if pcol not in merged.columns:
            continue

        if feat_cols:
            z = _weighted_trait_zscores(merged, pcol, feat_cols)
            top_traits = _fmt_traits(z, top_n_traits, reverse=True)
            low_traits = _fmt_traits(z, low_n_traits, reverse=False)
        else:
            top_traits = ""
            low_traits = ""

        prototypes = _format_prototypes(merged_meta, pcol, n=n_prototypes)

        rows.append({
            "cluster": k,
            "top_traits": top_traits,
            "low_traits": low_traits,
            "prototype_players": prototypes,
        })

    out = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Create season-specific archetype trait cards for the Streamlit app.")
    ap.add_argument("--season_label", required=True, help="Season key like 20252026")
    ap.add_argument("--groups", nargs="*", default=["forwards", "defense"], choices=["forwards", "defense"])
    ap.add_argument("--top_n_traits", type=int, default=6)
    ap.add_argument("--low_n_traits", type=int, default=4)
    ap.add_argument("--n_prototypes", type=int, default=8)
    args = ap.parse_args(argv)

    season = str(args.season_label)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for group in args.groups:
        print(f"\n[07_make_archetype_cards] Building traits for {group} {season}...")
        try:
            cards = build_cards_for_group(
                group=group,
                season=season,
                top_n_traits=args.top_n_traits,
                low_n_traits=args.low_n_traits,
                n_prototypes=args.n_prototypes,
            )
        except Exception as e:
            print(f"  ⚠️  Skipping {group} {season}: {e}")
            continue

        out_path = REPORTS_DIR / f"archetype_traits_{group}_{season}.csv"
        cards.to_csv(out_path, index=False)
        print(f"  ✅ Wrote {out_path} ({len(cards)} archetypes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
