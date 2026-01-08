from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List
from tqdm import tqdm


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def toi_str_to_seconds(toi: Any) -> float:
    if toi is None:
        return 0.0
    s = str(toi)
    if ":" not in s:
        return 0.0
    try:
        mm, ss = s.split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return 0.0


def norm_pos(p: Any) -> str:
    p = ("" if p is None else str(p)).upper().strip()
    if p == "L":
        return "LW"
    if p == "R":
        return "RW"
    return p if p else "UNK"


def _name_piece(x: Any) -> str:
    if isinstance(x, dict):
        return (x.get("default") or x.get("en") or "").strip()
    if isinstance(x, str):
        return x.strip()
    return ""


def get_name(p: Dict[str, Any]) -> str:
    """
    Strong preference order:
    1) firstName + lastName (dict or str)
    2) fullName (if present)
    3) name.default / name.en
    """
    first = _name_piece(p.get("firstName"))
    last = _name_piece(p.get("lastName"))
    if first and last:
        return f"{first} {last}".strip()

    full = _name_piece(p.get("fullName"))
    if full:
        return full

    nm = ""
    if isinstance(p.get("name"), dict):
        nm = _name_piece(p.get("name"))
    if nm:
        return nm

    return ""


def extract_players_from_boxscore(box: Dict[str, Any], game_id: int, season: str, game_type: int) -> List[dict]:
    out: List[dict] = []
    pbg = box.get("playerByGameStats")
    if not isinstance(pbg, dict):
        return out

    home = (box.get("homeTeam") or {}).get("abbrev") or (box.get("homeTeam") or {}).get("abbreviation")
    away = (box.get("awayTeam") or {}).get("abbrev") or (box.get("awayTeam") or {}).get("abbreviation")

    for side, team in (("homeTeam", home), ("awayTeam", away)):
        team_block = pbg.get(side, {})
        if not isinstance(team_block, dict):
            continue

        for group in ("forwards", "defense", "goalies"):
            plist = team_block.get(group, [])
            if not isinstance(plist, list):
                continue

            for p in plist:
                if not isinstance(p, dict):
                    continue

                pid = p.get("playerId") or p.get("id")
                if pid is None:
                    continue

                pos = norm_pos(p.get("position") or p.get("positionCode") or p.get("pos") or ("G" if group == "goalies" else "UNK"))

                sog = p.get("sog")
                if sog is None:
                    sog = p.get("shotsOnGoal")
                if sog is None:
                    sog = p.get("shots")

                plus_minus = p.get("plusMinus")
                if plus_minus is None:
                    plus_minus = p.get("+/-")

                out.append({
                    "season": season,
                    "game_id": game_id,
                    "game_type": int(game_type),
                    "team": team,
                    "player_id": int(pid),
                    "full_name": get_name(p),
                    "position": pos,

                    "toi_s": toi_str_to_seconds(p.get("toi")),
                    "pp_toi_s": toi_str_to_seconds(p.get("powerPlayToi") or p.get("ppToi")),
                    "pk_toi_s": toi_str_to_seconds(p.get("shortHandedToi") or p.get("shToi")),

                    "goals": int(p.get("goals") or 0),
                    "assists": int(p.get("assists") or 0),
                    "points": int(p.get("points") or (p.get("goals") or 0) + (p.get("assists") or 0)),
                    "shots": int(sog or 0),

                    "hits": int(p.get("hits") or 0),
                    "blocked_shots": int(p.get("blockedShots") or p.get("blocks") or 0),
                    "pim": int(p.get("pim") or p.get("penaltyMinutes") or 0),
                    "takeaways": int(p.get("takeaways") or 0),
                    "giveaways": int(p.get("giveaways") or 0),
                    "plus_minus": int(plus_minus or 0),

                    "faceoff_wins": int(p.get("faceoffWins") or 0),
                    "faceoff_taken": int(p.get("faceoffTaken") or 0),
                })

    return out


def add_per60(df: pd.DataFrame, num_col: str, toi_col: str) -> pd.Series:
    denom = (df[toi_col] / 3600.0).replace({0: np.nan})
    return df[num_col] / denom


def mode_pos(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    s = s[s != "UNK"]
    if len(s) == 0:
        return "UNK"
    return s.mode().iloc[0]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build player-game + player-season boxscore features (REG+PO; ignore preseason).")
    ap.add_argument("--schedule_parquet", required=True)
    ap.add_argument("--season_label", required=True)
    args = ap.parse_args(argv)

    schedule = pd.read_parquet(args.schedule_parquet).sort_values(["game_date", "game_id"])
    schedule = schedule[schedule["game_type"].isin([2, 3])].copy()

    game_type_map = dict(zip(schedule["game_id"].astype(int), schedule["game_type"].astype(int)))

    raw_root = Path("data/raw") / "season" / args.season_label / "games"
    rows: List[dict] = []
    for gid in tqdm(schedule["game_id"].astype(int).tolist(), desc="Parsing boxscores"):
        box_path = raw_root / str(gid) / "boxscore.json"
        box = load_json(box_path)
        gt = game_type_map[int(gid)]
        rows.extend(extract_players_from_boxscore(box, int(gid), args.season_label, gt))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Parsed 0 rows. Boxscore structure mismatch.")

    outdir = Path("data/features")
    outdir.mkdir(parents=True, exist_ok=True)

    pg_path = outdir / f"player_game_boxscore_{args.season_label}.parquet"
    df.to_parquet(pg_path, index=False)

    agg_cols = [
        "toi_s","pp_toi_s","pk_toi_s",
        "goals","assists","points","shots",
        "hits","blocked_shots","pim","takeaways","giveaways","plus_minus",
        "faceoff_wins","faceoff_taken",
    ]
    for c in agg_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    pos_df = df.groupby(["season","player_id"], as_index=False)["position"].agg(position=mode_pos)

    def agg_for_type(game_type: int, prefix: str) -> pd.DataFrame:
        sub = df[df["game_type"] == game_type].copy()
        if sub.empty:
            return pd.DataFrame(columns=["season","player_id"])

        g = sub.groupby(["season","player_id"], as_index=False)[agg_cols].sum()
        g[f"{prefix}games"] = sub.groupby(["season","player_id"])["game_id"].nunique().values

        g[f"{prefix}avg_toi_min"] = (g["toi_s"] / 60.0) / g[f"{prefix}games"].replace({0: np.nan})
        g[f"{prefix}pp_share"] = g["pp_toi_s"] / g["toi_s"].replace({0: np.nan})
        g[f"{prefix}pk_share"] = g["pk_toi_s"] / g["toi_s"].replace({0: np.nan})
        g[f"{prefix}fo_pct"] = (g["faceoff_wins"] / g["faceoff_taken"].replace({0: np.nan})) * 100.0
        g[f"{prefix}fo_taken_per_game"] = g["faceoff_taken"] / g[f"{prefix}games"].replace({0: np.nan})

        for c in ["goals","assists","points","shots","hits","blocked_shots","pim","takeaways","giveaways","plus_minus"]:
            g[f"{prefix}{c}_per60"] = add_per60(g, c, "toi_s")

        g = g.rename(columns={c: f"{prefix}{c}" for c in agg_cols})
        return g

    reg = agg_for_type(2, "reg_")
    po  = agg_for_type(3, "po_")

    ps = pd.merge(reg, po, on=["season","player_id"], how="outer").fillna(0)
    ps = ps.merge(pos_df, on=["season","player_id"], how="left")

    ps_path = outdir / f"player_season_boxscore_{args.season_label}.parquet"
    ps.to_parquet(ps_path, index=False)

    print(f"Saved:\n- {pg_path}\n- {ps_path}")
    print(f"Player-games:   {len(df):,}")
    print(f"Player-seasons: {len(ps):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
