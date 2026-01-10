import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# Make app/ importable
APP_DIR = Path(__file__).resolve().parent  # .../app
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from lib import available_seasons, season_key_to_label, load_all_seasons_group

st.set_page_config(page_title="NHL Player Archetypes", layout="wide")
st.title("NHL Player Archetypes")

st.markdown("""
A data-driven project that learns **player style archetypes** from publicly available NHL game data, then visualizes:
- **Season-level trends** in player types (forwards vs defense)
- **Team roster fit** / roster gaps by archetype
- **Player-by-player evolution** of style across a career

Use the left navigation to explore:
- **Season Level Trends** (league + team views)
- **Player Evolution** (career timeline)
""")

st.markdown("---")

# ----------------------------
# Sources & inspiration (rendered as a concise “bibliography” section)
# ----------------------------
with st.expander("Methods & key inspirations", expanded=False):
    st.markdown("""
### High-level method
1) Pull NHL game data (schedule + gamecenter JSON) and aggregate to season-level stats (REG vs PO split).  
2) Convert counting stats to comparable rates and usage indicators.  
3) Learn “style fingerprints” via **NMF**, then soft-cluster into archetypes via **Gaussian Mixture Models (GMM)**.  
4) Use the resulting archetype probabilities for season/team/player views.

### Primary API / data sources
- Unofficial NHL API reference (api-web.nhle.com endpoints): Zmalski NHL-API-Reference (GitHub)
- Gamecenter endpoints: boxscore + play-by-play
- Local caching of JSON + parquet feature tables for reproducibility

### Related research (closest matches)
- Ice hockey playing-style clustering with GMM / fuzzy methods (player role classification)
- NMF + GMM pipelines for player style/role identification in hockey
- Player-role identification using expected goals + possession/physical metrics
- NHL location-transition models / NMF-based player similarity work
""")

st.markdown("---")

# ----------------------------
# Auto-generated “key findings” (computed from your local dataset)
# ----------------------------
st.header("Key findings from the dataset (auto-generated)")

seasons = available_seasons()
if not seasons:
    st.info("No seasons found in data/app yet. Build seasons first.")
    st.stop()

# Load both groups across all seasons
all_f = load_all_seasons_group("forwards")
all_d = load_all_seasons_group("defense")

def quick_summary(df: pd.DataFrame, label: str):
    if df.empty:
        st.warning(f"No data found for {label}.")
        return
    st.subheader(label)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Seasons", len(df["season"].unique()))
    with c2:
        st.metric("Players", df["player_id"].nunique())
    with c3:
        st.metric("Rows (player-seasons)", len(df))

quick_summary(all_f, "Forwards")
quick_summary(all_d, "Defense")

st.markdown("### Confidence distribution by season (stability of archetype assignment)")
def conf_by_season(df: pd.DataFrame):
    tmp = df.copy()
    tmp["conf_pct"] = (tmp["confidence"].astype(float) * 100.0)
    out = tmp.groupby("season", as_index=False)["conf_pct"].mean()
    out["Season"] = out["season"].apply(season_key_to_label)
    return out.sort_values("season")

cf = conf_by_season(all_f)
cd = conf_by_season(all_d)

chart = alt.Chart(pd.concat([
    cf.assign(group="forwards"),
    cd.assign(group="defense")
], ignore_index=True)).mark_line(point=True).encode(
    x=alt.X("Season:O", axis=alt.Axis(labelAngle=0), title="Season"),
    y=alt.Y("conf_pct:Q", title="Avg top-archetype confidence (%)", scale=alt.Scale(domain=[0, 100])),
    color="group:N",
    tooltip=["Season", "group", alt.Tooltip("conf_pct:Q", format=".1f")]
).properties(height=260)

st.altair_chart(chart, use_container_width=True)

st.markdown("### How often do players change top archetype year-to-year?")
def archetype_switch_rate(df: pd.DataFrame):
    tmp = df.copy()
    tmp["top"] = tmp["top_cluster"].astype(int)
    tmp = tmp.sort_values(["player_id", "season"])
    tmp["prev_top"] = tmp.groupby("player_id")["top"].shift(1)
    tmp["changed"] = (tmp["top"] != tmp["prev_top"]) & tmp["prev_top"].notna()
    # per-player switch rate
    per = tmp.groupby("player_id", as_index=False).agg(
        seasons=("season", "nunique"),
        switches=("changed", "sum")
    )
    per = per[per["seasons"] >= 3].copy()
    per["switch_rate"] = per["switches"] / (per["seasons"] - 1)
    return per["switch_rate"].dropna()

sr_f = archetype_switch_rate(all_f)
sr_d = archetype_switch_rate(all_d)

c1, c2 = st.columns(2)
with c1:
    st.metric("Forwards median switch rate", f"{sr_f.median():.2f}")
with c2:
    st.metric("Defense median switch rate", f"{sr_d.median():.2f}")

st.markdown("---")
st.caption("This page updates automatically as you add/rebuild seasons in data/app/ and reports/.")
