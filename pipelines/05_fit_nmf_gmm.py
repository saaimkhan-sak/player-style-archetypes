from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.mixture import GaussianMixture


def load_schema(season_label: str) -> dict:
    p = Path("data/features") / f"feature_schema_{season_label}.json"
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_nonnegative(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """NMF requires nonnegative input; shift whole block if needed."""
    min_val = float(np.min(X))
    shift = -min_val if min_val < 0 else 0.0
    return X + shift, shift


def fit_nmf_block(X_block: np.ndarray, n_components: int, random_state: int) -> Tuple[np.ndarray, NMF]:
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=random_state,
        max_iter=2000,
    )
    W = model.fit_transform(X_block)
    return W, model


def gmm_grid_search(
    Z: np.ndarray,
    k_list: List[int],
    cov_types: List[str],
    random_state: int
) -> Tuple[GaussianMixture, dict]:
    best = None
    best_info = None
    for k in k_list:
        for cov in cov_types:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                random_state=random_state,
                reg_covar=1e-6,
                n_init=5,
                max_iter=2000,
            )
            gmm.fit(Z)
            bic = float(gmm.bic(Z))
            info = {"k": k, "covariance_type": cov, "bic": bic}
            if best is None or bic < best_info["bic"]:
                best = gmm
                best_info = info
    return best, best_info


def build_latent_matrix(
    X_df: pd.DataFrame,
    blocks: Dict[str, List[str]],
    nmf_components: Dict[str, int],
    random_state: int,
) -> Tuple[np.ndarray, dict, List[str]]:
    nmf_models = {}
    latent_parts = []
    latent_names = []

    for block_name, cols in blocks.items():
        cols = [c for c in cols if c in X_df.columns]
        if not cols:
            continue

        X_block = X_df[cols].to_numpy(dtype=float)
        X_block, shift = ensure_nonnegative(X_block)

        k = int(nmf_components.get(block_name, 2))
        W, model = fit_nmf_block(X_block, n_components=k, random_state=random_state)

        nmf_models[block_name] = {
            "cols": cols,
            "shift": shift,
            "n_components": k,
            "model": model,
        }
        latent_parts.append(W)
        latent_names.extend([f"{block_name}_nmf{j}" for j in range(k)])

    if not latent_parts:
        raise RuntimeError("No latent parts created — check blocks vs X_df columns.")
    Z = np.concatenate(latent_parts, axis=1)
    return Z, nmf_models, latent_names


def infer_id_cols(X: pd.DataFrame) -> List[str]:
    """
    Your new matrices store reg_games/reg_toi_s, not games/toi_s.
    We'll keep whatever exists, in a stable order.
    """
    candidates = ["season", "player_id", "position", "reg_games", "reg_toi_s", "games", "toi_s"]
    return [c for c in candidates if c in X.columns]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fit NMF-per-block then GMM (BIC) for forwards/defense.")
    p.add_argument("--season_label", required=True)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--k_min", type=int, default=4)
    p.add_argument("--k_max", type=int, default=14)
    p.add_argument("--cov_types", nargs="*", default=["full", "diag"])
    args = p.parse_args(argv)

    season = args.season_label
    schema = load_schema(season)

    # NMF components per block (tweak later)
    nmf_components = {
        "shooting_scoring": 3,
        "physical_disruption": 3,
        "discipline": 1,
        "special_teams_usage": 2,
        "faceoffs": 2,
    }

    k_list = list(range(args.k_min, args.k_max + 1))
    outdir = Path("models") / season
    outdir.mkdir(parents=True, exist_ok=True)

    for group in ["forwards", "defense"]:
        X_path = Path("data/features") / f"X_{group}_{season}.parquet"
        X = pd.read_parquet(X_path)

        id_cols = infer_id_cols(X)
        if "season" not in id_cols or "player_id" not in id_cols:
            raise RuntimeError(f"Missing required id cols in {X_path}. Found: {list(X.columns)}")

        ids = X[id_cols].copy()
        X_feat = X.drop(columns=id_cols)

        blocks = schema[group]["blocks"]

        Z, nmf_models, latent_names = build_latent_matrix(
            X_feat, blocks=blocks, nmf_components=nmf_components, random_state=args.random_state
        )

        gmm, best_info = gmm_grid_search(Z, k_list=k_list, cov_types=args.cov_types, random_state=args.random_state)

        probs = gmm.predict_proba(Z)
        hard = probs.argmax(axis=1)

        # Save artifacts
        joblib.dump({"nmf_models": nmf_models, "latent_names": latent_names}, outdir / f"nmf_{group}.joblib")
        joblib.dump({"gmm": gmm, "best_info": best_info}, outdir / f"gmm_{group}.joblib")

        # Save outputs
        out = ids.copy()
        out["cluster"] = hard
        for k in range(probs.shape[1]):
            out[f"p{k}"] = probs[:, k]

        out_path = Path("data/processed") / f"archetypes_{group}_{season}.parquet"
        out.to_parquet(out_path, index=False)

        print(f"\n[{group.upper()}]")
        print(f"Best GMM: k={best_info['k']} cov={best_info['covariance_type']} BIC={best_info['bic']:.1f}")
        print(f"Saved archetypes -> {out_path}")
        print(f"Saved models     -> {outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
