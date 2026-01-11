from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# Interpret archetypes using REGULAR SEASON features (more stable, higher sample size)
FEATURES = [
    "reg_shots_per60", "reg_goals_per60", "reg_assists_per60", "reg_points_per60",
    "reg_hits_per60", "reg_blocked_shots_per60", "reg_takeaways_per60", "reg_giveaways_per60",
    "reg_pim_per60",
    "reg_pp_share", "reg_pk_share",
    "reg_fo_pct", "reg_fo_taken_per_game",
]


def zscore_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mu = out[c].mean()
        sd = out[c].std(ddof=0)
        if sd == 0 or np.isnan(sd):
            out[c + "_z"] = 0.0
        else:
            out[c + "_z"] = (out[c] - mu) / sd
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Make archetype cards/traits from GMM outputs + REG player-season features.")
    ap.add_argument("--season_label", required=True)
    args = ap.parse_args(argv)
    season = args.season_label

    directory = pd.read_parquet("data/processed/player_directory.parquet")
    teams = pd.read_parquet(f"data/processed/player_season_teams_{season}.parquet")
    base = pd.read_parquet(f"data/features/player_season_boxscore_{season}.parquet")

    # Ensure features exist
    for c in FEATURES:
        if c not in base.columns:
            base[c] = 0.0
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)

    outdir = Path("reports")
    outdir.mkdir(parents=True, exist_ok=True)

    for group in ["forwards", "defense"]:
        arch = pd.read_parquet(f"data/processed/archetypes_{group}_{season}.parquet")

        # Merge by (season, player_id) to avoid position-code mismatch dropping rows
        merged = (
            arch.merge(directory, on="player_id", how="left")
                .merge(teams, on=["season", "player_id"], how="left")
                .merge(base[["season", "player_id"] + FEATURES], on=["season", "player_id"], how="left")
        )

        merged["full_name"] = merged["full_name"].fillna(merged["player_id"].astype(str))
        merged["teams_played"] = merged["teams_played"].fillna(merged.get("primary_team", "NA"))

        pcols = [c for c in merged.columns if c.startswith("p") and c[1:].isdigit()]
        if not pcols:
            raise RuntimeError("No probability columns p0..pK found in archetypes output.")

        K = len(pcols)

        cards = []
        for k in range(K):
            w = pd.to_numeric(merged[f"p{k}"], errors="coerce").fillna(0.0).to_numpy()
            if w.sum() <= 0:
                w = np.ones_like(w)

            means = {feat: float(np.average(merged[feat].to_numpy(dtype=float), weights=w)) for feat in FEATURES}

            prot = merged.sort_values(f"p{k}", ascending=False).head(12)
            prot_list = []
            for r in prot.itertuples(index=False):
                name = getattr(r, "full_name", None) or str(getattr(r, "player_id"))
                tp = getattr(r, "teams_played", None) or "NA"
                pk = getattr(r, f"p{k}")
                prot_list.append(f"{name} ({tp}) p={pk:.2f}")

            cards.append({
                "cluster": k,
                "soft_size": float(pd.Series(w).sum()),
                **means,
                "prototype_players": " | ".join(prot_list),
            })

        cards_df = pd.DataFrame(cards)
        zdf = zscore_cols(cards_df, FEATURES)

        trait_rows = []
        for r in zdf.itertuples(index=False):
            zcols = [(c, getattr(r, c + "_z")) for c in FEATURES]
            top = sorted(zcols, key=lambda x: x[1], reverse=True)[:4]
            bot = sorted(zcols, key=lambda x: x[1])[:3]
            trait_rows.append({
                "cluster": int(r.cluster),
                "soft_size": float(r.soft_size),
                "top_traits": ", ".join([f"{c}({z:+.2f})" for c, z in top]),
                "low_traits": ", ".join([f"{c}({z:+.2f})" for c, z in bot]),
                "prototype_players": r.prototype_players,
            })

        traits = pd.DataFrame(trait_rows).sort_values("cluster")

        cards_path = outdir / f"archetype_cards_{group}_{season}.csv"
        traits_path = outdir / f"archetype_traits_{group}_{season}.csv"
        cards_df.to_csv(cards_path, index=False)
        traits.to_csv(traits_path, index=False)

        print(f"\n[{group.upper()}] saved:")
        print(f"- {cards_path}")
        print(f"- {traits_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
