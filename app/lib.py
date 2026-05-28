from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path("data/app")
REPORTS_DIR = Path("reports")
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.archetype_labels import (
    PROFILE_COLOR_MAP,
    PROFILE_ORDER,
    build_archetype_name_summary,
    canonical_profile_name,
    parse_trait_string,
    profile_colors,
)

ARCHETYPE_LABEL_CACHE_KEY = "profile-colors-v2"

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
def load_archetype_name_map_for_season(
    group: str,
    season_key: str,
    label_version: str = ARCHETYPE_LABEL_CACHE_KEY,
) -> dict[int, str]:
    """
    Returns {cluster_id -> archetype_name} for a given season & group.
    IMPORTANT: This uses traits (top/low traits) to create a descriptive name,
    NOT example players.
    """
    p = REPORTS_DIR / f"archetype_traits_{group}_{season_key}.csv"
    if not p.exists():
        return {}

    traits_df = pd.read_csv(p)

    m: dict[int, str] = {}
    for _, tr in traits_df.iterrows():
        kk = int(tr["cluster"])
        high = parse_trait_string(tr.get("top_traits", ""))
        low  = parse_trait_string(tr.get("low_traits", ""))
        m[kk], _ = build_archetype_name_summary(kk, high, low)

    return m

def archetype_math_explainer():
    st.expander("What is a 'Player Archetype' and How is it Calculated?", expanded=False).__enter__()
    st.markdown(
        """
This page describes player “styles” (archetypes) learned from boxscore + usage features.
Archetypes are learned per season, then translated into descriptive titles with consistent colors across seasons.
"""
    )
    st.markdown("I normalize for ice time, create per-60 rates, then cluster players into archetypes using a mixture model.")
    st.markdown("This section is shared across pages.")
    st.expander("", expanded=False).__exit__(None, None, None)
