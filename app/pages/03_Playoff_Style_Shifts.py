from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from lib import (  # noqa: E402
    available_seasons,
    load_all_seasons_group,
    load_archetype_name_map_for_season,
    season_key_to_label,
)


st.set_page_config(page_title="Playoff Style Shifts", layout="wide")
st.title("Playoff Style Shifts")


def safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    den = pd.to_numeric(den, errors="coerce").replace({0: np.nan})
    return pd.to_numeric(num, errors="coerce").fillna(0) / den


def min_to_mmss(minutes: float) -> str:
    if pd.isna(minutes):
        return "00:00"
    total_s = int(round(float(minutes) * 60))
    return f"{total_s // 60:02d}:{total_s % 60:02d}"


@st.cache_data
def load_playoff_shift_data() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group in ("forwards", "defense"):
        df = load_all_seasons_group(group)
        if df.empty:
            continue
        df = df.copy()
        df["group"] = group
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for col in [
        "reg_games",
        "po_games",
        "reg_points",
        "po_points",
        "reg_goals",
        "po_goals",
        "reg_assists",
        "po_assists",
        "reg_shots",
        "po_shots",
        "reg_pim",
        "po_pim",
        "reg_plus_minus",
        "po_plus_minus",
        "reg_avg_toi_min",
        "po_avg_toi_min",
        "confidence",
        "top_cluster",
    ]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Season"] = df["season"].apply(season_key_to_label)
    df["REG P/GP"] = safe_rate(df["reg_points"], df["reg_games"])
    df["PO P/GP"] = safe_rate(df["po_points"], df["po_games"])
    df["REG G/GP"] = safe_rate(df["reg_goals"], df["reg_games"])
    df["PO G/GP"] = safe_rate(df["po_goals"], df["po_games"])
    df["REG SOG/GP"] = safe_rate(df["reg_shots"], df["reg_games"])
    df["PO SOG/GP"] = safe_rate(df["po_shots"], df["po_games"])
    df["REG PIM/GP"] = safe_rate(df["reg_pim"], df["reg_games"])
    df["PO PIM/GP"] = safe_rate(df["po_pim"], df["po_games"])
    df["REG +/-/GP"] = safe_rate(df["reg_plus_minus"], df["reg_games"])
    df["PO +/-/GP"] = safe_rate(df["po_plus_minus"], df["po_games"])

    df["P/GP change"] = df["PO P/GP"] - df["REG P/GP"]
    df["G/GP change"] = df["PO G/GP"] - df["REG G/GP"]
    df["SOG/GP change"] = df["PO SOG/GP"] - df["REG SOG/GP"]
    df["TOI change"] = df["po_avg_toi_min"] - df["reg_avg_toi_min"]
    df["PIM/GP change"] = df["PO PIM/GP"] - df["REG PIM/GP"]
    df["+/-/GP change"] = df["PO +/-/GP"] - df["REG +/-/GP"]

    # The app does not currently ship full playoff NMF/GMM projections for all
    # seasons, so this score measures role/stat profile movement in tracked
    # playoff columns while retaining the regular-season archetype context.
    metric_cols = ["P/GP change", "SOG/GP change", "TOI change", "PIM/GP change", "+/-/GP change"]
    for col in metric_cols:
        centered = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        spread = centered.std(ddof=0)
        df[col + " z"] = 0.0 if spread == 0 or pd.isna(spread) else (centered - centered.mean()) / spread

    df["playoff_shift_score"] = np.sqrt(sum(df[col + " z"] ** 2 for col in metric_cols))
    df["playoff_shift_score"] = df["playoff_shift_score"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["changed_bucket"] = pd.cut(
        df["playoff_shift_score"],
        bins=[-0.01, 2.0, 3.5, 20],
        labels=["Held steady", "Moderate shift", "Major shift"],
    ).astype(str)

    names: list[str] = []
    for row in df.itertuples(index=False):
        mapping = load_archetype_name_map_for_season(row.group, row.season)
        k = int(getattr(row, "top_cluster", 0))
        names.append(mapping.get(k, f"Archetype {k}"))
    df["regular_archetype"] = names
    df["archetype_label"] = "A" + df["top_cluster"].astype(int).astype(str) + " - " + df["regular_archetype"]
    return df


def filtered_playoff_rows(df: pd.DataFrame, group: str, min_reg_gp: int, min_po_gp: int) -> pd.DataFrame:
    out = df[(df["group"] == group) & (df["reg_games"] >= min_reg_gp) & (df["po_games"] >= min_po_gp)].copy()
    return out.sort_values(["season", "playoff_shift_score"], ascending=[False, False])


def compact_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        [
            "Season",
            "full_name",
            "teams_played",
            "position",
            "archetype_label",
            "reg_games",
            "po_games",
            "REG P/GP",
            "PO P/GP",
            "P/GP change",
            "reg_avg_toi_min",
            "po_avg_toi_min",
            "TOI change",
            "playoff_shift_score",
            "changed_bucket",
        ]
    ].copy()
    out["REG ATOI"] = out["reg_avg_toi_min"].apply(min_to_mmss)
    out["PO ATOI"] = out["po_avg_toi_min"].apply(min_to_mmss)
    out = out.drop(columns=["reg_avg_toi_min", "po_avg_toi_min"])
    out = out.rename(
        columns={
            "full_name": "Player",
            "teams_played": "Team(s)",
            "position": "Pos",
            "archetype_label": "REG archetype",
            "reg_games": "REG GP",
            "po_games": "PO GP",
            "playoff_shift_score": "Shift score",
            "changed_bucket": "Shift band",
        }
    )
    return out


data = load_playoff_shift_data()
if data.empty:
    st.warning("No app data found.")
    st.stop()

st.markdown(
    """
This page compares a player's regular-season profile with their playoff profile. The archetype shown is still the regular-season model assignment; the shift score measures how much their tracked playoff production, shot volume, ice time, penalty rate, and plus-minus move relative to regular season.
"""
)

with st.sidebar:
    group = st.selectbox("Group", ["forwards", "defense"], index=0)
    seasons_with_playoffs = sorted(data.loc[data["po_games"] > 0, "season"].unique(), reverse=True)
    season = st.selectbox("Season", seasons_with_playoffs, index=0, format_func=season_key_to_label)
    min_reg_gp = st.slider("Min regular-season games", 0, 82, 20, step=5)
    min_po_gp = st.slider("Min playoff games", 1, 28, 4, step=1)

base = filtered_playoff_rows(data, group, min_reg_gp, min_po_gp)
season_df = base[base["season"] == season].copy()

if season_df.empty:
    st.info("No players match the current filters.")
    st.stop()

tab_season, tab_archetypes, tab_player = st.tabs(["Season View", "Archetypes", "Player Career"])

with tab_season:
    st.subheader(f"{season_key_to_label(season)} Playoff Shifts")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players", f"{len(season_df):,}")
    c2.metric("Median shift score", f"{season_df['playoff_shift_score'].median():.2f}")
    c3.metric("Major shifts", f"{int((season_df['changed_bucket'] == 'Major shift').sum()):,}")
    c4.metric("Median P/GP change", f"{season_df['P/GP change'].median():+.2f}")

    scatter = (
        alt.Chart(season_df)
        .mark_circle(size=90, opacity=0.78)
        .encode(
            x=alt.X("P/GP change:Q", title="Playoff P/GP minus regular-season P/GP"),
            y=alt.Y("TOI change:Q", title="Playoff ATOI minus regular-season ATOI"),
            color=alt.Color("changed_bucket:N", title="Shift band"),
            size=alt.Size("po_games:Q", title="PO GP", scale=alt.Scale(range=[40, 240])),
            tooltip=[
                alt.Tooltip("full_name:N", title="Player"),
                alt.Tooltip("teams_played:N", title="Team"),
                alt.Tooltip("archetype_label:N", title="REG archetype"),
                alt.Tooltip("reg_games:Q", title="REG GP"),
                alt.Tooltip("po_games:Q", title="PO GP"),
                alt.Tooltip("P/GP change:Q", format="+.2f"),
                alt.Tooltip("TOI change:Q", format="+.1f"),
                alt.Tooltip("playoff_shift_score:Q", title="Shift score", format=".2f"),
            ],
        )
        .properties(height=420)
    )
    zero_x = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#9CA3AF").encode(x="x:Q")
    zero_y = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9CA3AF").encode(y="y:Q")
    st.altair_chart(scatter + zero_x + zero_y, use_container_width=True)

    st.markdown("#### Biggest Playoff Profile Changes")
    st.dataframe(
        compact_table(season_df).sort_values("Shift score", ascending=False).head(30),
        use_container_width=True,
        hide_index=True,
    )

with tab_archetypes:
    st.subheader("Regular-Season Archetypes Under Playoff Pressure")
    arch = (
        base.groupby(["season", "Season", "archetype_label"], as_index=False)
        .agg(
            players=("player_id", "nunique"),
            median_shift=("playoff_shift_score", "median"),
            median_pgp_change=("P/GP change", "median"),
            median_toi_change=("TOI change", "median"),
        )
    )
    arch = arch[arch["players"] >= 3].copy()

    heat = (
        alt.Chart(arch)
        .mark_rect()
        .encode(
            x=alt.X("Season:O", title="Season", sort=[season_key_to_label(s) for s in sorted(base["season"].unique())]),
            y=alt.Y("archetype_label:N", title="REG archetype", sort="-x"),
            color=alt.Color("median_shift:Q", title="Median shift", scale=alt.Scale(scheme="redyellowgreen", reverse=True)),
            tooltip=[
                alt.Tooltip("Season:O"),
                alt.Tooltip("archetype_label:N", title="REG archetype"),
                alt.Tooltip("players:Q", title="Players"),
                alt.Tooltip("median_shift:Q", title="Median shift", format=".2f"),
                alt.Tooltip("median_pgp_change:Q", title="Median P/GP change", format="+.2f"),
                alt.Tooltip("median_toi_change:Q", title="Median TOI change", format="+.1f"),
            ],
        )
        .properties(height=max(360, 28 * arch["archetype_label"].nunique()))
    )
    st.altair_chart(heat, use_container_width=True)

    st.dataframe(
        arch.sort_values(["season", "median_shift"], ascending=[False, False]).rename(
            columns={
                "Season": "Season",
                "archetype_label": "REG archetype",
                "players": "Players",
                "median_shift": "Median shift",
                "median_pgp_change": "Median P/GP change",
                "median_toi_change": "Median TOI change",
            }
        )[["Season", "REG archetype", "Players", "Median shift", "Median P/GP change", "Median TOI change"]],
        use_container_width=True,
        hide_index=True,
    )

with tab_player:
    st.subheader("Player Career Playoff Pattern")
    latest = (
        base.sort_values("season")
        .groupby("player_id", as_index=False)
        .tail(1)
        .sort_values("full_name")
        .copy()
    )
    query = st.text_input("Search player", value="")
    if query.strip():
        latest = latest[latest["full_name"].str.contains(query, case=False, na=False)].copy()

    if latest.empty:
        st.info("No matching playoff players under the current filters.")
    else:
        latest["display"] = latest["full_name"] + " - " + latest["position"].astype(str)
        choice = st.selectbox("Player", latest["display"].tolist())
        player_id = int(latest.set_index("display").loc[choice, "player_id"])
        hist = base[base["player_id"] == player_id].sort_values("season").copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Playoff seasons", f"{len(hist):,}")
        c2.metric("Median shift score", f"{hist['playoff_shift_score'].median():.2f}")
        c3.metric("Career PO GP", f"{int(hist['po_games'].sum()):,}")
        c4.metric("Career P/GP change", f"{hist['P/GP change'].mean():+.2f}")

        long = hist.melt(
            id_vars=["Season", "season"],
            value_vars=["REG P/GP", "PO P/GP", "reg_avg_toi_min", "po_avg_toi_min"],
            var_name="Metric",
            value_name="Value",
        )
        long["Metric"] = long["Metric"].replace(
            {
                "reg_avg_toi_min": "REG ATOI",
                "po_avg_toi_min": "PO ATOI",
            }
        )
        chart = (
            alt.Chart(long)
            .mark_line(point=True)
            .encode(
                x=alt.X("Season:O", sort=[season_key_to_label(s) for s in sorted(hist["season"].unique())]),
                y=alt.Y("Value:Q", title="Value"),
                color=alt.Color("Metric:N"),
                tooltip=["Season", "Metric", alt.Tooltip("Value:Q", format=".2f")],
            )
            .properties(height=340)
        )
        st.altair_chart(chart, use_container_width=True)

        st.dataframe(compact_table(hist), use_container_width=True, hide_index=True)
