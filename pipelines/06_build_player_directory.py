from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def mode_nonnull(s: pd.Series):
    s = s.dropna().astype(str)
    s = s[s.str.strip() != ""]
    if len(s) == 0:
        return None
    return s.mode().iloc[0]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build player directory + teams_played per player-season from player_game_boxscore."
    )
    ap.add_argument("--season_label", required=True)
    args = ap.parse_args(argv)

    season = args.season_label
    pg_path = Path(f"data/features/player_game_boxscore_{season}.parquet")
    if not pg_path.exists():
        raise FileNotFoundError(f"Missing {pg_path}. Run pipelines/03_build_player_season_features_boxscore.py first.")

    pg = pd.read_parquet(pg_path)

    outdir = Path("data/processed")
    outdir.mkdir(parents=True, exist_ok=True)

    # Directory: player_id -> best full_name + mode position
    directory = (
        pg.groupby("player_id", as_index=False)
          .agg({
              "full_name": mode_nonnull,
              "position": mode_nonnull,
          })
    )
    dir_path = outdir / "player_directory.parquet"
    directory.to_parquet(dir_path, index=False)

    # Teams played ordered by TOI
    team_agg = (
        pg.groupby(["season", "player_id", "team"], as_index=False)["toi_s"]
          .sum()
          .sort_values(["season", "player_id", "toi_s"], ascending=[True, True, False])
    )

    def teams_join(g: pd.DataFrame) -> str:
        teams = g["team"].tolist()
        seen = []
        for t in teams:
            if t and t not in seen:
                seen.append(t)
        return "/".join(seen)

    teams = (
        team_agg.groupby(["season", "player_id"], as_index=False)
                .apply(lambda g: pd.Series({"teams_played": teams_join(g)}), include_groups=False)
                .reset_index(drop=True)
    )

    idx = team_agg.groupby(["season", "player_id"])["toi_s"].idxmax()
    primary = team_agg.loc[idx, ["season", "player_id", "team"]].rename(columns={"team": "primary_team"})
    pt = primary.merge(teams, on=["season", "player_id"], how="left")

    pt_path = outdir / f"player_season_teams_{season}.parquet"
    pt.to_parquet(pt_path, index=False)

    print(f"Saved:\n- {dir_path}\n- {pt_path}")
    print(f"Players in directory: {len(directory):,}")
    print(f"Player-seasons teams: {len(pt):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
