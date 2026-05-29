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

from src.archetype_labels import build_archetype_name_summary, parse_trait_string
try:
    from src.archetype_labels import (
        ARCHETYPE_LABEL_VERSION,
        PROFILE_COLOR_MAP,
        PROFILE_ORDER,
        canonical_profile_name,
        profile_colors,
        readable_trait_label,
    )
except ImportError:
    ARCHETYPE_LABEL_VERSION = "role-names-v2"
    PROFILE_COLOR_MAP = {
        "High-Volume Playmaking Scorer": ("#2563EB", "#FFFFFF"),
        "Perimeter Skill Scorer": ("#FED7AA", "#7C2D12"),
        "Shot-Creation Scorer": ("#FDBA74", "#7C2D12"),
        "Two-Way Skill Scorer": ("#BBF7D0", "#14532D"),
        "High-Touch Risk/Reward Scorer": ("#F0ABFC", "#701A75"),
        "Shot-Blocking Contact Specialist": ("#0891B2", "#FFFFFF"),
        "Agitating Heavy-Contact Forward": ("#DC2626", "#FFFFFF"),
        "Puck-Pressure Two-Way Creator": ("#16A34A", "#FFFFFF"),
        "Deployment / Role Specialist": ("#64748B", "#FFFFFF"),
        "PP-Leaning Offensive Role": ("#D97706", "#111827"),
        "PK-Leaning Defensive Role": ("#4F46E5", "#FFFFFF"),
        "High-Touch Risk/Reward Playmaker": ("#9333EA", "#FFFFFF"),
        "Checking-Line Disruptor": ("#BE123C", "#FFFFFF"),
        "Physical Shutdown Defenseman": ("#DC2626", "#FFFFFF"),
        "Shot-Blocking Defensive Defenseman": ("#0891B2", "#FFFFFF"),
        "Offensive Puck-Moving Defenseman": ("#2563EB", "#FFFFFF"),
        "Low-Event Puck-Moving Defenseman": ("#F97316", "#111827"),
        "Point-Usage Power-Play Defenseman": ("#D97706", "#111827"),
        "Penalty-Kill Defensive Defenseman": ("#4F46E5", "#FFFFFF"),
        "Transition Risk/Reward Defenseman": ("#9333EA", "#FFFFFF"),
        "Defensive Role Defenseman": ("#64748B", "#FFFFFF"),
        "Puck-Pressure Transition Defenseman": ("#16A34A", "#FFFFFF"),
        "High-Event Physical Defenseman": ("#BE123C", "#FFFFFF"),
    }
    PROFILE_ORDER = list(PROFILE_COLOR_MAP.keys())

    def canonical_profile_name(name: str) -> str:
        return str(name)

    def profile_colors(name: str) -> tuple[str, str]:
        return PROFILE_COLOR_MAP.get(canonical_profile_name(name), ("#E5E7EB", "#111827"))

    def readable_trait_label(feature: str) -> str:
        labels = {
            "reg_points_per60": "Points per 60",
            "reg_goals_per60": "Goals per 60",
            "reg_assists_per60": "Assists per 60",
            "reg_shots_per60": "Shots on goal per 60",
            "reg_hits_per60": "Hits per 60",
            "reg_blocked_shots_per60": "Blocked shots per 60",
            "reg_takeaways_per60": "Takeaways per 60",
            "reg_giveaways_per60": "Giveaways per 60",
            "reg_pim_per60": "Penalty minutes per 60",
            "reg_pp_share": "Power-play usage",
            "reg_pk_share": "Penalty-kill usage",
            "reg_fo_pct": "Faceoff win rate",
            "reg_fo_taken_per_game": "Faceoffs per game",
            "mp_reg_5on5_I_F_highDangerShots_per60": "High-danger shots per 60",
            "mp_reg_5on5_I_F_highDangerShotShare": "Share of shots from high-danger areas",
            "mp_reg_5on5_I_F_xGoals_per60": "Individual expected goals per 60",
            "mp_reg_5on5_I_F_xGoalsPerAttempt": "Expected goals per shot attempt",
            "mp_reg_5on5_I_F_rebounds_per60": "Rebound chances per 60",
            "mp_reg_5on5_I_F_reboundxGoals_per60": "Expected goals from rebounds per 60",
            "mp_reg_5on5_I_F_playContinuedInZone_per60": "Offensive-zone possessions extended per 60",
            "mp_reg_5on5_I_F_playContinuedOutsideZone_per60": "Rush possessions extended per 60",
            "mp_reg_5on5_OnIce_xGoalsPercentage_calc": "On-ice expected-goal share",
            "mp_reg_5on5_OnIce_F_xGoals_per60": "Team expected goals while on ice per 60",
            "mp_reg_5on5_OnIce_A_xGoals_per60": "Expected goals against while on ice per 60",
            "mp_reg_5on5_OnIce_A_shotAttempts_per60": "Shot attempts against while on ice per 60",
            "mp_reg_4on5_OnIce_A_xGoals_per60": "Penalty-kill expected goals against per 60",
            "mp_reg_5on5_shotsBlockedByPlayer_per60": "Shot blocks per 60",
            "mp_reg_5on5_penaltiesDrawn_per60": "Penalties drawn per 60",
            "mp_reg_5on4_I_F_xGoals_per60": "Power-play expected goals per 60",
        }
        if feature in labels:
            return labels[feature]
        text = str(feature)
        for old, new in {
            "mp_reg_": "",
            "reg_": "",
            "5on5_": "",
            "5on4_": "power_play_",
            "4on5_": "penalty_kill_",
            "OnIce_": "on_ice_",
            "I_F_": "individual_",
            "_per60": "_per_60",
            "_calc": "",
        }.items():
            text = text.replace(old, new)
        return text.replace("_", " ").title()

ARCHETYPE_LABEL_CACHE_KEY = ARCHETYPE_LABEL_VERSION
MIN_ADVANCED_SEASON_START = 2008

def season_key_to_label(k: str) -> str:
    k = str(k).strip()
    return f"{k[:4]}-{k[4:]}" if (len(k) == 8 and k.isdigit()) else k

def available_seasons() -> list[str]:
    app_dir = Path("data/app")
    if not app_dir.exists():
        return []
    fwd = {f.stem.replace("players_forwards_", "") for f in app_dir.glob("players_forwards_*.parquet")}
    dfd = {f.stem.replace("players_defense_", "") for f in app_dir.glob("players_defense_*.parquet")}
    seasons = [s for s in (fwd & dfd) if s[:4].isdigit() and int(s[:4]) >= MIN_ADVANCED_SEASON_START]
    return sorted(seasons, reverse=True)

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
        try:
            m[kk], _ = build_archetype_name_summary(kk, high, low, group=group)
        except TypeError:
            m[kk], _ = build_archetype_name_summary(kk, high, low)

    return m

def archetype_math_explainer():
    st.expander("What is a 'Player Archetype' and How is it Calculated?", expanded=False).__enter__()
    st.markdown(
        """
This page describes player “styles” (archetypes) learned from NHL boxscore/usage data plus MoneyPuck advanced player metrics.
Archetypes are learned per season, then translated into descriptive titles with consistent colors across seasons.
"""
    )
    st.markdown("I normalize for ice time, create per-60 rates, then cluster players into archetypes using a mixture model.")
    st.markdown("This section is shared across pages.")
    st.expander("", expanded=False).__exit__(None, None, None)
