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
    PROFILE_COLOR_MAP,
    available_seasons,
    load_all_seasons_group,
    load_archetype_name_map_for_season,
    season_key_to_label,
)


st.set_page_config(page_title="How Does Play Style Change in the Playoffs?", layout="wide")
st.markdown(
    """<style>
section[data-testid="stSidebar"] [data-testid="stPageLink"] a {
  white-space: normal !important;
  line-height: 1.2 !important;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a p,
section[data-testid="stSidebar"] [data-testid="stPageLink"] a span,
section[data-testid="stSidebar"] [data-testid="stPageLink"] a div {
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
}
.profile-table-wrap {overflow-x:auto;}
.profile-table {width:100%; min-width:1500px; border-collapse:collapse; border:1px solid #E5E7EB; border-radius:8px; overflow:hidden;}
.profile-table th,.profile-table td {padding:9px 10px; border-bottom:1px solid #E5E7EB; text-align:left; vertical-align:middle; white-space:nowrap;}
.profile-table th {background:#F9FAFB; color:#6B7280; font-weight:750;}
.profile-table .arch-col {min-width:245px;}
.profile-pill {display:inline-block; padding:4px 10px; border-radius:999px; font-weight:750; white-space:normal; line-height:1.2;}
</style>""",
    unsafe_allow_html=True,
)
st.title("How Does Play Style Change in the Playoffs?")
ARCHETYPE_COLOR_DOMAIN = list(PROFILE_COLOR_MAP.keys())
ARCHETYPE_COLOR_RANGE = [PROFILE_COLOR_MAP[name][0] for name in ARCHETYPE_COLOR_DOMAIN]


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
    df["archetype_label"] = df["regular_archetype"]

    projections: list[pd.DataFrame] = []
    for group in ("forwards", "defense"):
        for season in available_seasons():
            p = Path("data/app") / f"playoff_archetype_projection_{group}_{season}.parquet"
            if not p.exists():
                continue
            proj = pd.read_parquet(p)
            if proj.empty:
                continue
            proj["group"] = group
            projections.append(proj)

    if projections:
        proj = pd.concat(projections, ignore_index=True)
        keep_cols = [
            "season",
            "player_id",
            "group",
            "reg_top_cluster",
            "po_top_cluster",
            "reg_confidence",
            "po_confidence",
            "archetype_changed",
            "probability_distance",
        ]
        proj = proj[[c for c in keep_cols if c in proj.columns]].copy()
        df = df.merge(proj, on=["season", "player_id", "group"], how="left")
    else:
        df["probability_distance"] = np.nan

    df["model_shift_score"] = pd.to_numeric(df.get("probability_distance", np.nan), errors="coerce")
    df["model_shift_band"] = pd.cut(
        df["model_shift_score"],
        bins=[-0.01, 0.25, 0.75, 2.0],
        labels=["Held steady", "Moderate shift", "Major shift"],
    ).astype(str)
    df.loc[df["model_shift_score"].isna(), "model_shift_band"] = "Not projected"

    po_names: list[str] = []
    for row in df.itertuples(index=False):
        po_k = getattr(row, "po_top_cluster", np.nan)
        if pd.isna(po_k):
            po_names.append("Not projected")
            continue
        mapping = load_archetype_name_map_for_season(row.group, row.season)
        po_k_int = int(po_k)
        po_names.append(mapping.get(po_k_int, f"Archetype {po_k_int}"))
    df["playoff_archetype_label"] = po_names
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
            "playoff_archetype_label",
            "reg_games",
            "po_games",
            "REG P/GP",
            "PO P/GP",
            "P/GP change",
            "reg_avg_toi_min",
            "po_avg_toi_min",
            "TOI change",
            "playoff_shift_score",
            "model_shift_score",
            "model_shift_band",
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
            "playoff_archetype_label": "Projected PO archetype",
            "reg_games": "REG GP",
            "po_games": "PO GP",
            "playoff_shift_score": "Shift score",
            "model_shift_score": "Model shift",
            "model_shift_band": "Model shift band",
            "changed_bucket": "Shift band",
        }
    )
    return out


def profile_pill(name: object) -> str:
    text = "" if pd.isna(name) else str(name)
    bg, fg = PROFILE_COLOR_MAP.get(text, ("#E5E7EB", "#111827"))
    return f'<span class="profile-pill" style="background:{bg};color:{fg};">{text}</span>'


def render_profile_changes_table(df: pd.DataFrame) -> str:
    table = compact_table(df).sort_values("Shift score", ascending=False).head(30)
    cols = list(table.columns)
    header = "".join(
        f'<th class="arch-col">{c}</th>' if c in {"REG archetype", "Projected PO archetype"} else f"<th>{c}</th>"
        for c in cols
    )
    rows = []
    for r in table.to_dict("records"):
        cells = []
        for c in cols:
            val = r[c]
            if c in {"REG archetype", "Projected PO archetype"}:
                cells.append(f'<td class="arch-col">{profile_pill(val)}</td>')
            elif isinstance(val, float):
                cells.append(f"<td>{val:.3g}</td>")
            else:
                cells.append(f"<td>{val}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="profile-table-wrap"><table class="profile-table"><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


data = load_playoff_shift_data()
if data.empty:
    st.warning("No app data found.")
    st.stop()

st.markdown(
    """
This page compares a player's regular-season profile with their playoff profile. The archetype shown is still the regular-season model assignment; the shift score measures how much their tracked playoff production, shot volume, ice time, penalty rate, and plus-minus move relative to regular season.
"""
)
st.info(
    """
**Shift score:** this is a "how different did this player look in the playoffs?" score. It compares playoff scoring rate, shot rate, ice time, penalty rate, and plus-minus rate against that same player's regular-season baseline, then combines those changes into one number. A **larger score** means the playoff version of the player looked more different from his regular-season version; a **smaller score** means his role and results stayed steadier. Directionally, a **positive change** in a column like P/GP or TOI means it went up in the playoffs, while a **negative change** means it went down.
"""
)
if data["model_shift_score"].notna().any():
    st.caption("For seasons with projection files, the playoff archetype is calculated by running playoff feature vectors through that season's regular-season NMF/GMM model.")

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
    if season_df["model_shift_score"].notna().any():
        c3.metric("Projected archetype changes", f"{int(season_df['archetype_changed'].fillna(False).sum()):,}")
        c4.metric("Median model shift", f"{season_df['model_shift_score'].median():.2f}")
    else:
        c3.metric("Major shifts", f"{int((season_df['changed_bucket'] == 'Major shift').sum()):,}")
        c4.metric("Median P/GP change", f"{season_df['P/GP change'].median():+.2f}")

    scatter = (
        alt.Chart(season_df)
        .mark_circle(size=90, opacity=0.78)
        .encode(
            x=alt.X("P/GP change:Q", title="Playoff P/GP minus regular-season P/GP"),
            y=alt.Y("TOI change:Q", title="Playoff ATOI minus regular-season ATOI"),
            color=alt.Color(
                "archetype_label:N",
                title="REG archetype",
                scale=alt.Scale(domain=ARCHETYPE_COLOR_DOMAIN, range=ARCHETYPE_COLOR_RANGE),
                legend=alt.Legend(labelLimit=420, orient="right"),
            ),
            shape=alt.Shape("model_shift_band:N", title="Model shift band"),
            size=alt.Size("po_games:Q", title="PO GP", scale=alt.Scale(range=[40, 240])),
            tooltip=[
                alt.Tooltip("full_name:N", title="Player"),
                alt.Tooltip("teams_played:N", title="Team"),
                alt.Tooltip("archetype_label:N", title="REG archetype"),
                alt.Tooltip("playoff_archetype_label:N", title="Projected PO archetype"),
                alt.Tooltip("reg_games:Q", title="REG GP"),
                alt.Tooltip("po_games:Q", title="PO GP"),
                alt.Tooltip("P/GP change:Q", format="+.2f"),
                alt.Tooltip("TOI change:Q", format="+.1f"),
                alt.Tooltip("model_shift_score:Q", title="Model shift", format=".2f"),
                alt.Tooltip("playoff_shift_score:Q", title="Shift score", format=".2f"),
            ],
        )
        .properties(width=820, height=420)
    )
    zero_x = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#9CA3AF").encode(x="x:Q")
    zero_y = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9CA3AF").encode(y="y:Q")
    st.altair_chart(scatter + zero_x + zero_y, use_container_width=False)

    st.markdown("#### Biggest Playoff Profile Changes")
    st.markdown(render_profile_changes_table(season_df), unsafe_allow_html=True)

with tab_archetypes:
    st.subheader("Regular-Season Archetypes Under Playoff Pressure")
    arch = (
        base.groupby(["season", "Season", "archetype_label"], as_index=False)
        .agg(
            players=("player_id", "nunique"),
            median_shift=("playoff_shift_score", "median"),
            median_model_shift=("model_shift_score", "median"),
            archetype_change_rate=("archetype_changed", "mean"),
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
            color=alt.Color("median_model_shift:Q", title="Median model shift", scale=alt.Scale(scheme="redyellowgreen", reverse=True)),
            tooltip=[
                alt.Tooltip("Season:O"),
                alt.Tooltip("archetype_label:N", title="REG archetype"),
                alt.Tooltip("players:Q", title="Players"),
                alt.Tooltip("median_model_shift:Q", title="Median model shift", format=".2f"),
                alt.Tooltip("archetype_change_rate:Q", title="Archetype change rate", format=".0%"),
                alt.Tooltip("median_shift:Q", title="Median stat shift", format=".2f"),
                alt.Tooltip("median_pgp_change:Q", title="Median P/GP change", format="+.2f"),
                alt.Tooltip("median_toi_change:Q", title="Median TOI change", format="+.1f"),
            ],
        )
        .properties(height=max(360, 28 * arch["archetype_label"].nunique()))
    )
    st.altair_chart(heat, use_container_width=True)

    st.dataframe(
        arch.sort_values(["season", "median_model_shift"], ascending=[False, False]).rename(
            columns={
                "Season": "Season",
                "archetype_label": "REG archetype",
                "players": "Players",
                "median_model_shift": "Median model shift",
                "archetype_change_rate": "Archetype change rate",
                "median_shift": "Median stat shift",
                "median_pgp_change": "Median P/GP change",
                "median_toi_change": "Median TOI change",
            }
        )[["Season", "REG archetype", "Players", "Median model shift", "Archetype change rate", "Median stat shift", "Median P/GP change", "Median TOI change"]],
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

        season_sort = [season_key_to_label(s) for s in sorted(hist["season"].unique())]
        profile = hist[[
            "Season", "REG P/GP", "PO P/GP", "reg_avg_toi_min", "po_avg_toi_min",
            "P/GP change", "TOI change", "playoff_shift_score", "archetype_label",
            "playoff_archetype_label", "po_games",
        ]].copy()
        profile["REG ATOI"] = profile["reg_avg_toi_min"]
        profile["PO ATOI"] = profile["po_avg_toi_min"]
        profile["Profile"] = profile["archetype_label"] + " -> " + profile["playoff_archetype_label"].fillna("not projected")

        def paired_delta_chart(reg_col: str, po_col: str, title: str, fmt: str) -> alt.Chart:
            long = profile.melt(
                id_vars=["Season", "Profile", "po_games"],
                value_vars=[reg_col, po_col],
                var_name="Split",
                value_name="Value",
            )
            split_labels = {reg_col: "Regular Season", po_col: "Playoffs"}
            long["Split"] = long["Split"].map(split_labels)
            lines = alt.Chart(long).mark_line(strokeWidth=3, opacity=0.55).encode(
                x=alt.X("Value:Q", title=title, titlePadding=14),
                y=alt.Y("Season:O", sort=season_sort, title=None),
                detail="Season:N",
                color=alt.Color("Split:N", scale=alt.Scale(range=["#94A3B8", "#EF4444"])),
            )
            points = alt.Chart(long).mark_circle(size=110).encode(
                x="Value:Q",
                y=alt.Y("Season:O", sort=season_sort, title=None),
                color=alt.Color("Split:N", title=None, scale=alt.Scale(range=["#94A3B8", "#EF4444"])),
                tooltip=["Season", "Split", alt.Tooltip("Value:Q", format=fmt), "Profile", alt.Tooltip("po_games:Q", title="PO GP")],
            )
            return (lines + points).properties(
                height=max(220, 46 * len(profile)),
                title=title,
                padding={"left": 5, "right": 5, "top": 8, "bottom": 36},
            )

        col_a, col_b = st.columns(2)
        with col_a:
            st.altair_chart(paired_delta_chart("REG P/GP", "PO P/GP", "Scoring Rate: Regular Season vs Playoffs", ".2f"), use_container_width=True)
        with col_b:
            st.altair_chart(paired_delta_chart("REG ATOI", "PO ATOI", "Usage: Regular Season vs Playoffs", ".1f"), use_container_width=True)

        profile_long = profile.melt(
            id_vars=["Season", "playoff_shift_score", "P/GP change", "TOI change", "po_games"],
            value_vars=["archetype_label", "playoff_archetype_label"],
            var_name="Split",
            value_name="Archetype",
        )
        profile_long["Split"] = profile_long["Split"].replace({"archetype_label": "Regular Season", "playoff_archetype_label": "Playoffs"})
        transition = (
            alt.Chart(profile_long)
            .mark_rect(cornerRadius=4)
            .encode(
                x=alt.X("Split:N", title=None, sort=["Regular Season", "Playoffs"]),
                y=alt.Y("Season:O", sort=season_sort, title=None),
                color=alt.Color("Archetype:N", scale=alt.Scale(domain=ARCHETYPE_COLOR_DOMAIN, range=ARCHETYPE_COLOR_RANGE), legend=None),
                tooltip=["Season", "Split", "Archetype", alt.Tooltip("playoff_shift_score:Q", title="Shift score", format=".2f")],
            )
            .properties(height=max(180, 38 * len(profile)), title="Archetype Translation")
        )
        shift = (
            alt.Chart(profile)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("playoff_shift_score:Q", title="Shift score"),
                y=alt.Y("Season:O", sort=season_sort, title=None),
                color=alt.Color("P/GP change:Q", title="P/GP change", scale=alt.Scale(scheme="redblue", domainMid=0)),
                tooltip=[
                    "Season",
                    alt.Tooltip("playoff_shift_score:Q", title="Shift score", format=".2f"),
                    alt.Tooltip("P/GP change:Q", format="+.2f"),
                    alt.Tooltip("TOI change:Q", format="+.1f"),
                ],
            )
            .properties(height=max(180, 38 * len(profile)), title="How Much the Playoff Profile Moved")
        )
        st.altair_chart(alt.hconcat(transition, shift, spacing=28), use_container_width=True)

        st.dataframe(compact_table(hist), use_container_width=True, hide_index=True)
