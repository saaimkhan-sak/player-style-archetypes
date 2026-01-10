from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path("data/app")
REPORTS_DIR = Path("reports")

def season_key_to_label(k: str) -> str:
    k = str(k).strip()
    return f"{k[:4]}-{k[4:]}" if (len(k) == 8 and k.isdigit()) else k

def available_seasons() -> list[str]:
    app_dir = Path("data/app")
    if not app_dir.exists():
        return []
    fwd = {f.stem.replace("players_forwards_", "") for f in app_dir.glob("players_forwards_*.parquet")}
    dfd = {f.stem.replace("players_defense_", "") for f in app_dir.glob("players_defense_*.parquet")}
    return sorted(list(fwd & dfd), reverse=True)

@st.cache_data
def load_group(group: str, season: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"players_{group}_{season}.parquet")

@st.cache_data
def load_all_seasons_group(group: str) -> pd.DataFrame:
    frames = []
    for sk in available_seasons():
        path = DATA_DIR / f"players_{group}_{sk}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["season"] = sk
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

@st.cache_data
def load_archetype_name_map_for_season(group: str, season_key: str) -> dict[int, str]:
    """
    Returns {cluster_id -> archetype_name} for a given season & group.
    IMPORTANT: This uses traits (top/low traits) to create a descriptive name,
    NOT example players.
    """
    p = REPORTS_DIR / f"archetype_traits_{group}_{season_key}.csv"
    if not p.exists():
        return {}

    traits_df = pd.read_csv(p)

    def parse_trait_string(s: str):
        if not isinstance(s, str):
            return []
        out = []
        for part in s.split(","):
            part = part.strip()
            m = re.match(r"^([A-Za-z0-9_]+)\(([+-]?\d+\.?\d*)\)$", part)
            if m:
                out.append((m.group(1), float(m.group(2))))
        return out

    def build_name(high_tokens, low_tokens, k: int) -> str:
        high_feats = {f for f, _ in high_tokens}
        low_feats  = {f for f, _ in low_tokens}

        offense_hi = any(f in high_feats for f in ["reg_points_per60","reg_goals_per60","reg_assists_per60","reg_shots_per60"])
        playmaking_hi = "reg_assists_per60" in high_feats
        shooting_hi = "reg_shots_per60" in high_feats
        blocks_hi = "reg_blocked_shots_per60" in high_feats
        hits_hi = "reg_hits_per60" in high_feats
        pim_hi = "reg_pim_per60" in high_feats
        takeaways_hi = "reg_takeaways_per60" in high_feats
        giveaways_lo = "reg_giveaways_per60" in low_feats
        fo_hi = ("reg_fo_taken_per_game" in high_feats) or ("reg_fo_pct" in high_feats)
        pk_hi = "reg_pk_share" in high_feats
        pp_hi = "reg_pp_share" in high_feats

        if pim_hi and hits_hi:
            return "Agitating Heavy-Contact Forward"
        if blocks_hi and hits_hi:
            return "Shot-Blocking Contact Specialist"
        if offense_hi and playmaking_hi and shooting_hi:
            return "High-Volume Playmaking Scorer"
        if takeaways_hi and giveaways_lo:
            return "Puck-Pressure Two-Way Creator"
        if fo_hi and not offense_hi:
            return "Deployment / Faceoff Specialist"
        if pk_hi and not pp_hi:
            return "PK-Leaning Defensive Role"
        if pp_hi and not pk_hi:
            return "PP-Leaning Offensive Role"

        return f"Mixed Profile Archetype {k}"

    m: dict[int, str] = {}
    for _, tr in traits_df.iterrows():
        kk = int(tr["cluster"])
        high = parse_trait_string(tr.get("top_traits", ""))
        low  = parse_trait_string(tr.get("low_traits", ""))
        m[kk] = build_name(high, low, kk)

    return m

def archetype_math_explainer():
    st.expander("What is a 'Player Archetype' and How is it Calculated?", expanded=False).__enter__()
    st.markdown(
        """
This page describes player “styles” (archetypes) learned from boxscore + usage features.
Archetypes are learned per season, so **A0 in one season is not necessarily the same as A0 in another**.
"""
    )
    st.markdown("I normalize for ice time, create per-60 rates, then cluster players into archetypes using a mixture model.")
    st.markdown("This section is shared across pages.")
    st.expander("", expanded=False).__exit__(None, None, None)
