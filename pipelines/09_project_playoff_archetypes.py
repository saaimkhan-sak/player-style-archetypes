from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd


FORWARD_POS = {"C", "LW", "RW", "W", "F"}
DEFENSE_POS = {"D", "LD", "RD"}


def norm_pos(p: object) -> str:
    p = ("" if p is None else str(p)).upper().strip()
    if p == "L":
        return "LW"
    if p == "R":
        return "RW"
    return p if p else "UNK"


def load_schema(season: str) -> dict:
    path = Path("data/features") / f"feature_schema_{season}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def regular_feature_to_playoff_feature(feature: str) -> str:
    if feature.startswith("reg_"):
        return "po_" + feature[len("reg_") :]
    if feature.startswith("mp_reg_"):
        return "mp_po_" + feature[len("mp_reg_") :]
    raise ValueError(f"Expected regular-season feature name, got {feature!r}")


def numeric_column(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def scaled_playoff_matrix(stats: pd.DataFrame, schema: dict, group: str) -> pd.DataFrame:
    all_features = schema[group]["all_features"]
    scaler = schema[group]["scaler"]

    out = pd.DataFrame(index=stats.index)
    for reg_col in all_features:
        po_col = regular_feature_to_playoff_feature(reg_col)
        values = numeric_column(stats, po_col)
        med = float(scaler["median"].get(reg_col, 0.0))
        iqr = float(scaler["iqr"].get(reg_col, 1.0)) or 1.0
        out[reg_col] = (values - med) / iqr
        out[reg_col] = out[reg_col].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0)
    return out


def project_latent(X_scaled: pd.DataFrame, nmf_artifact: dict, schema: dict, group: str) -> np.ndarray:
    parts = []
    blocks = schema[group]["blocks"]
    for block_name, cols in blocks.items():
        if block_name not in nmf_artifact["nmf_models"]:
            continue
        info = nmf_artifact["nmf_models"][block_name]
        model = info["model"]
        block_cols = [c for c in info["cols"] if c in cols and c in X_scaled.columns]
        if not block_cols:
            continue
        X_block = X_scaled[block_cols].to_numpy(dtype=float)
        X_block = np.maximum(X_block + float(info.get("shift", 0.0)), 0.0)
        if not np.any(getattr(model, "components_", np.array([]))):
            parts.append(np.zeros((len(X_block), int(model.n_components)), dtype=float))
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")
            projected = model.transform(X_block)
        if not np.isfinite(projected).all():
            raise RuntimeError(f"Playoff NMF projection produced NaN or infinite values for {group}.")
        parts.append(projected)

    if not parts:
        raise RuntimeError(f"No playoff latent blocks produced for {group}.")
    return np.concatenate(parts, axis=1)


def build_group_projection(season: str, group: str, min_po_games: int) -> pd.DataFrame:
    stats_path = Path("data/features") / f"player_season_boxscore_{season}.parquet"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {stats_path}")

    schema = load_schema(season)
    stats = pd.read_parquet(stats_path)
    stats["position"] = stats["position"].map(norm_pos)
    stats["po_games"] = numeric_column(stats, "po_games").astype(int)

    positions = FORWARD_POS if group == "forwards" else DEFENSE_POS
    stats = stats[(stats["position"].isin(positions)) & (stats["po_games"] >= min_po_games)].copy()
    if stats.empty:
        return pd.DataFrame()

    X_scaled = scaled_playoff_matrix(stats, schema, group)

    model_dir = Path("models") / season
    nmf_artifact = joblib.load(model_dir / f"nmf_{group}.joblib")
    gmm_artifact = joblib.load(model_dir / f"gmm_{group}.joblib")
    gmm = gmm_artifact["gmm"]

    Z = project_latent(X_scaled, nmf_artifact, schema, group)
    po_probs = gmm.predict_proba(Z)
    po_top = po_probs.argmax(axis=1)
    po_conf = po_probs.max(axis=1)

    reg_path = Path("data/processed") / f"archetypes_{group}_{season}.parquet"
    if reg_path.exists():
        reg = pd.read_parquet(reg_path)
    else:
        reg = pd.read_parquet(Path("data/app") / f"players_{group}_{season}.parquet")
        reg = reg.rename(columns={"top_cluster": "cluster"})
    reg_pcols = [c for c in reg.columns if c.startswith("p") and c[1:].isdigit()]
    reg_keep = ["season", "player_id", "cluster", *reg_pcols]
    reg = reg[reg_keep].rename(columns={"cluster": "reg_top_cluster", **{c: f"reg_{c}" for c in reg_pcols}})

    out = stats[["season", "player_id", "position", "reg_games", "po_games"]].copy()
    out = out.merge(reg, on=["season", "player_id"], how="left")
    for k in range(po_probs.shape[1]):
        out[f"po_p{k}"] = po_probs[:, k]

    out["po_top_cluster"] = po_top
    out["po_confidence"] = po_conf
    out["reg_confidence"] = out[[f"reg_p{k}" for k in range(po_probs.shape[1]) if f"reg_p{k}" in out.columns]].max(axis=1)
    out["archetype_changed"] = out["reg_top_cluster"].astype("Int64") != out["po_top_cluster"].astype("Int64")

    prob_deltas = []
    for k in range(po_probs.shape[1]):
        reg_col = f"reg_p{k}"
        po_col = f"po_p{k}"
        if reg_col in out.columns:
            prob_deltas.append((out[po_col] - out[reg_col]).pow(2))
    out["probability_distance"] = np.sqrt(sum(prob_deltas)) if prob_deltas else np.nan
    return out


def available_feature_seasons() -> list[str]:
    paths = Path("data/features").glob("player_season_boxscore_*.parquet")
    seasons = [p.stem.replace("player_season_boxscore_", "") for p in paths]
    return sorted(s for s in seasons if s[:4].isdigit() and int(s[:4]) >= 2008)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Project playoff feature vectors through regular-season NMF/GMM models."
    )
    parser.add_argument("--season_label")
    parser.add_argument("--all", action="store_true", help="Project every season with feature/model artifacts.")
    parser.add_argument("--min_po_games", type=int, default=1)
    args = parser.parse_args(argv)

    if not args.all and not args.season_label:
        parser.error("Provide --season_label or --all.")

    seasons = available_feature_seasons() if args.all else [args.season_label]

    processed_outdir = Path("data/processed")
    app_outdir = Path("data/app")
    processed_outdir.mkdir(parents=True, exist_ok=True)
    app_outdir.mkdir(parents=True, exist_ok=True)

    wrote = 0
    for season in seasons:
        model_dir = Path("models") / season
        if not model_dir.exists():
            print(f"Skipping {season}: missing {model_dir}")
            continue

        for group in ("forwards", "defense"):
            if not (model_dir / f"nmf_{group}.joblib").exists() or not (model_dir / f"gmm_{group}.joblib").exists():
                print(f"Skipping {season} {group}: missing model artifact")
                continue
            projected = build_group_projection(season, group, args.min_po_games)
            for outdir in (processed_outdir, app_outdir):
                outpath = outdir / f"playoff_archetype_projection_{group}_{season}.parquet"
                projected.to_parquet(outpath, index=False)
            wrote += len(projected)
            print(f"Saved {season} {group}: {len(projected):,} rows")

    print(f"Projected playoff archetypes for {wrote:,} player-seasons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
