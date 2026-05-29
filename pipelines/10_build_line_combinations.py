from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_FILES = [
    Path("data/raw/moneypuck/game-by-game-line-data-2008_to_2024.csv"),
    Path("data/raw/moneypuck/game-by-game-line-data-2025.csv"),
]
USECOLS = [
    "season",
    "playerTeam",
    "position",
    "name",
    "lineId",
    "gameId",
    "icetime",
    "xGoalsFor",
    "xGoalsAgainst",
    "goalsFor",
    "goalsAgainst",
    "shotAttemptsFor",
    "shotAttemptsAgainst",
]


def season_key(start_year: int) -> str:
    return f"{start_year}{start_year + 1}"


def main() -> int:
    parts = []
    for path in RAW_FILES:
        if not path.exists():
            continue
        reader = pd.read_csv(path, usecols=USECOLS, chunksize=250_000)
        for chunk in reader:
            chunk = chunk[chunk["position"].isin(["line", "pairing"])].copy()
            chunk["icetime"] = pd.to_numeric(chunk["icetime"], errors="coerce").fillna(0.0)
            for col in ["xGoalsFor", "xGoalsAgainst", "goalsFor", "goalsAgainst", "shotAttemptsFor", "shotAttemptsAgainst"]:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0)
            chunk["season"] = pd.to_numeric(chunk["season"], errors="coerce").astype("Int64")
            chunk = chunk.dropna(subset=["season", "playerTeam", "position", "name", "lineId"])
            chunk["season_key"] = chunk["season"].astype(int).map(season_key)
            parts.append(
                chunk.groupby(["season_key", "playerTeam", "position", "name", "lineId"], as_index=False)
                .agg(
                    icetime=("icetime", "sum"),
                    games=("gameId", "nunique"),
                    xGoalsFor=("xGoalsFor", "sum"),
                    xGoalsAgainst=("xGoalsAgainst", "sum"),
                    goalsFor=("goalsFor", "sum"),
                    goalsAgainst=("goalsAgainst", "sum"),
                    shotAttemptsFor=("shotAttemptsFor", "sum"),
                    shotAttemptsAgainst=("shotAttemptsAgainst", "sum"),
                )
            )

    if not parts:
        raise FileNotFoundError("No MoneyPuck line-combination CSVs found in data/raw/moneypuck/.")

    out = pd.concat(parts, ignore_index=True)
    out = (
        out.groupby(["season_key", "playerTeam", "position", "name", "lineId"], as_index=False)
        .sum(numeric_only=True)
    )
    out["toi_min"] = out["icetime"] / 60.0
    out["xg_pct"] = out["xGoalsFor"] / (out["xGoalsFor"] + out["xGoalsAgainst"]).replace({0: pd.NA})
    out["goal_pct"] = out["goalsFor"] / (out["goalsFor"] + out["goalsAgainst"]).replace({0: pd.NA})
    out["corsi_pct"] = out["shotAttemptsFor"] / (out["shotAttemptsFor"] + out["shotAttemptsAgainst"]).replace({0: pd.NA})
    out = out.sort_values(["season_key", "playerTeam", "position", "toi_min"], ascending=[False, True, True, False])

    out_path = Path("data/app/line_combinations.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Saved {len(out):,} line/pairing rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
