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
        "reg_games", "po_games", "reg_points", "po_points", "reg_goals", "po_goals",
        "reg_assists", "po_assists", "reg_shots", "po_shots", "reg_pim", "po_pim",
        "reg_plus_minus", "po_plus_minus", "reg_avg_toi_min", "po_avg_toi_min",
        "confidence", "top_cluster",
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

    metric_cols = ["P/GP change", "SOG/GP change", "TOI change", "PIM/GP change", "+/-/GP change"]
    for col in metric_cols:
        centered = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        spread = centered.std(ddof=0)
        df[col + " z"] = 0.0 if spread == 0 or pd.isna(spread) else (centered - centered.mean()) / spread

    df["stat_shift_score"] = np.sqrt(sum(df[col + " z"] ** 2 for col in metric_cols))
    df["stat_shift_score"] = df["stat_shift_score"].replace([np.inf, -np.inf], np.nan).fillna(0)
    # Keep legacy name for backward compatibility with existing chart code
    df["playoff_shift_score"] = df["stat_shift_score"]
    df["changed_bucket"] = pd.cut(
        df["stat_shift_score"],
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
            "season", "player_id", "group",
            "reg_top_cluster", "po_top_cluster",
            "reg_confidence", "po_confidence",
            "archetype_changed", "probability_distance",
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
    return out.sort_values(["season", "model_shift_score"], ascending=[False, False])


def compact_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[
        "Season", "full_name", "teams_played", "position",
        "archetype_label", "playoff_archetype_label",
        "reg_games", "po_games",
        "REG P/GP", "PO P/GP", "P/GP change",
        "reg_avg_toi_min", "po_avg_toi_min", "TOI change",
        "model_shift_score", "model_shift_band",
        "stat_shift_score", "changed_bucket",
    ]].copy()
    out["REG ATOI"] = out["reg_avg_toi_min"].apply(min_to_mmss)
    out["PO ATOI"] = out["po_avg_toi_min"].apply(min_to_mmss)
    out = out.drop(columns=["reg_avg_toi_min", "po_avg_toi_min"])
    out = out.rename(columns={
        "full_name": "Player",
        "teams_played": "Team(s)",
        "position": "Pos",
        "archetype_label": "REG archetype",
        "playoff_archetype_label": "Projected PO archetype",
        "reg_games": "REG GP",
        "po_games": "PO GP",
        "model_shift_score": "Model shift ↑",
        "model_shift_band": "Model shift band",
        "stat_shift_score": "Stat shift",
        "changed_bucket": "Stat shift band",
    })
    return out.sort_values("Model shift ↑", ascending=False)


def profile_pill(name: object) -> str:
    text = "" if pd.isna(name) else str(name)
    bg, fg = PROFILE_COLOR_MAP.get(text, ("#E5E7EB", "#111827"))
    return f'<span class="profile-pill" style="background:{bg};color:{fg};">{text}</span>'


_SHIFT_BAND_COLORS = {
    "Held steady":    ("#DCFCE7", "#166534"),
    "Moderate shift": ("#FEF9C3", "#854D0E"),
    "Major shift":    ("#FEE2E2", "#991B1B"),
    "Not projected":  ("#F1F5F9", "#64748B"),
}


def shift_band_pill(band: object) -> str:
    text = "Not projected" if (band is None or (isinstance(band, float) and pd.isna(band))) else str(band)
    bg, fg = _SHIFT_BAND_COLORS.get(text, ("#F1F5F9", "#64748B"))
    return f'<span class="profile-pill" style="background:{bg};color:{fg};font-size:0.82em;">{text}</span>'


def model_score_pill(score: object) -> str:
    try:
        v = float(score)
    except (TypeError, ValueError):
        return '<span class="profile-pill" style="background:#F1F5F9;color:#64748B;font-size:0.82em;">—</span>'
    if pd.isna(v):
        return '<span class="profile-pill" style="background:#F1F5F9;color:#64748B;font-size:0.82em;">—</span>'
    if v >= 0.75:
        bg, fg = "#FEE2E2", "#991B1B"
    elif v >= 0.25:
        bg, fg = "#FEF9C3", "#854D0E"
    else:
        bg, fg = "#DCFCE7", "#166534"
    return f'<span class="profile-pill" style="background:{bg};color:{fg};font-size:0.82em;">{v:.3f}</span>'


_ARCH_COLS = {"REG archetype", "Projected PO archetype"}
_BAND_COLS = {"Model shift band", "Stat shift band"}
_SCORE_COLS = {"Model shift ↑"}


def _render_table_html(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    header = "".join(
        f'<th class="arch-col">{c}</th>' if c in _ARCH_COLS else f"<th>{c}</th>"
        for c in cols
    )
    rows = []
    for r in table.to_dict("records"):
        cells = []
        for c in cols:
            val = r[c]
            if c in _ARCH_COLS:
                cells.append(f'<td class="arch-col">{profile_pill(val)}</td>')
            elif c in _BAND_COLS:
                cells.append(f"<td>{shift_band_pill(val)}</td>")
            elif c in _SCORE_COLS:
                cells.append(f"<td>{model_score_pill(val)}</td>")
            elif isinstance(val, float):
                cells.append(f"<td>{val:.3g}</td>")
            else:
                cells.append(f"<td>{val}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="profile-table-wrap"><table class="profile-table"><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def render_profile_changes_table(df: pd.DataFrame) -> str:
    return _render_table_html(compact_table(df).head(30))


# ── Data load ──────────────────────────────────────────────────────────────────
data = load_playoff_shift_data()
if data.empty:
    st.warning("No app data found.")
    st.stop()

# ── Intro & explainer ──────────────────────────────────────────────────────────
st.markdown("""
We all know that the playoffs feel different — tighter systems, better goaltending, and higher stakes.
But how much does a player's *actual style* change when the intensity ramps up?
This page answers that question using the same archetype model that classifies regular-season play, now applied to playoff data.
""")

with st.expander("📊 How is the model shift score calculated? (click to expand)", expanded=False):
    st.markdown("""
### The short version
I took each player's playoff statistics, ran them through the exact same machine-learning model used to assign regular-season archetypes, and measured how far the player's playoff "style fingerprint" is from their regular-season one. A bigger number = a bigger identity shift.

---

### Step 1 — Where the data comes from

**Regular season:** The archetype model was trained on [MoneyPuck](https://moneypuck.com) player-level advanced metrics — a well-regarded public data source that tracks things like expected goals (xGoals), shot quality, and on-ice possession at the individual player level, game by game.

**Playoffs:** MoneyPuck publishes playoff statistics on the same site, but only as a season summary (not game by game). I saved the playoff statistics pages for all 18 seasons from 2008-09 through 2025-26 and extracted the data directly from each page's HTML. To capture special-teams context, I collected four separate views for each season:
- **All situations combined**
- **5-on-5 (even strength)** — the most important slice, where most of the game is played
- **5-on-4 (power play)** — when a team has the man advantage
- **4-on-5 (penalty kill)** — when a team is shorthanded

---

### Step 2 — What I calculated from the playoff data

For each player and each situation, I computed the same types of rate statistics the regular-season model uses:

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Expected Goals per 60 min (5v5)** | How many goals a player's shots *should* produce per hour of ice time, based on shot location and type | Separates lucky goal-scorers from genuine shot-quality creators |
| **Shot attempts per 60 min (5v5)** | How frequently a player gets involved in shooting plays | Captures offensive pressure regardless of whether shots go in |
| **High-danger shot share** | What fraction of a player's shots come from the most dangerous areas (in tight, directly in front) | Identifies net-front finishers vs perimeter shooters |
| **On-ice xGoals For/Against per 60 (5v5)** | How good/bad the team was at creating and allowing expected goals *while this player was on the ice* | Measures two-way impact and deployment quality |
| **Rebounds created per 60 (5v5)** | How often a player's shots lead to rebound opportunities | Distinguishes power-play net-front threats from perimeter options |
| **xGoals from rebounds per 60 (5v5)** | Expected goal value generated specifically from rebound shots | Captures a specific scoring style |
| **Shot blocking per 60 (5v5 + 4v5)** | How often a player blocks opposing shots | Key defensive archetype signal |
| **Hits, takeaways, giveaways per 60 (5v5)** | Physical and puck-battle contributions | Separates checking-line forwards from skill players |
| **Penalties drawn per 60 (5v5)** | How often a player draws penalties | Identifies cycle forwards who generate PP time |
| **Zone start distribution** | What share of a player's shifts start in the offensive, neutral, or defensive zone | Captures deployment role — sheltered offensive player vs defensive specialist |
| **Faceoff win % (5v5)** | How often a center wins faceoffs | Key center archetype signal |
| **Power play xGoals (5v4)** | Expected goals on the power play | Identifies PP specialists |
| **Penalty kill opponent xGoals (4v5)** | Expected goals allowed while shorthanded | Identifies PK specialists |

For situations where MoneyPuck's summary view doesn't publish a metric (specifically: play-continuation rates and after-shift xGoals in non-5v5 situations), I substituted the league-average value from the regular-season model — meaning those signals are neutral rather than actively misleading.

---

### Step 3 — Running it through the model

With those playoff rate statistics in hand, I did two things:

**NMF compression:** Non-negative Matrix Factorization squashes all those metrics into a compact "style fingerprint" — a short list of numbers that describe *how* a player plays rather than *how much* they produce. Think of it as distilling a player's full stat line into a few key style dimensions.

**GMM classification:** A Gaussian Mixture Model then takes that fingerprint and outputs a *probability distribution* across archetypes. For example, a player might be classified as:
- Regular season: 72% Playmaking Scorer, 18% Two-Way Creator, 10% Other
- Playoffs: 41% Playmaking Scorer, 44% Two-Way Creator, 15% Other

The regular-season model was not re-trained on playoff data — I used the same fitted model to project each player into archetype space based on their playoff numbers.

---

### Step 4 — The model shift score

The **model shift score** is the Euclidean distance between those two probability distributions:

> *How much did the probability mass move across archetypes from regular season to playoffs?*

- **Score near 0** → The model sees essentially the same player in both contexts. The style fingerprint barely changed.
- **Score around 0.25–0.75** → Moderate shift. The player looks meaningfully different — perhaps leaning into a different role or responding to matchup adjustments.
- **Score above 0.75** → Major shift. The playoff version of this player would likely be classified into a different archetype than the regular-season version.

---

### What about the "stat shift score"?

The table also shows a simpler **stat shift score** that was used before the advanced data was available. It compares raw boxscore metrics (points per game, shots per game, ice time, penalty minutes, plus/minus) between regular season and playoffs using z-scores, then combines them. It is less informative than the model shift score because:
1. Scoring rates *universally* decline in the playoffs due to tighter play and better goaltending — so a drop in P/GP doesn't necessarily mean a player changed their style
2. It doesn't capture shot quality, on-ice possession, or zone-start context

The model shift score addresses both of these problems. Use the stat shift as a sanity check, but trust the model shift as the primary signal.

---

### Limitations
- Playoff sample sizes are smaller than regular-season totals, adding noise — especially for players eliminated in round one
- The model was trained on regular-season distributions, which are slightly wider than playoff distributions (extreme performers are more common in the regular season). This means the model is working slightly "out of sample" when applied to playoffs
- A handful of metrics (play-continuation rates, after-shift xGoals) couldn't be recovered from the summary data and are imputed as league average
    """)

st.info(
    "**Primary signal: Model shift score** — measures how far a player's playoff style fingerprint moves in archetype space, "
    "based on xGoals, shot quality, on-ice possession, and zone starts at even strength, power play, and penalty kill. "
    "**Higher = bigger identity shift.** The scatter plot shape encodes shift band; point size encodes playoff games played."
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

# ── Season View ────────────────────────────────────────────────────────────────
with tab_season:
    st.subheader(f"{season_key_to_label(season)} Playoff Shifts")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players", f"{len(season_df):,}")
    has_model = season_df["model_shift_score"].notna().any()
    if has_model:
        c2.metric("Median model shift", f"{season_df['model_shift_score'].median():.2f}")
        c3.metric("Archetype changes", f"{int(season_df['archetype_changed'].fillna(False).sum()):,}")
        c4.metric("% changed archetype", f"{season_df['archetype_changed'].fillna(False).mean():.0%}")
    else:
        c2.metric("Median stat shift", f"{season_df['stat_shift_score'].median():.2f}")
        c3.metric("Major shifts", f"{int((season_df['changed_bucket'] == 'Major shift').sum()):,}")
        c4.metric("Median P/GP change", f"{season_df['P/GP change'].median():+.2f}")

    st.markdown(
        "**Scatter plot:** Each dot is one player. "
        "X-axis = scoring rate change (playoff P/GP minus regular-season P/GP). "
        "Y-axis = ice-time change (playoff ATOI minus regular-season ATOI). "
        "**Dot shape** encodes the model shift band (how much the archetype fingerprint moved); "
        "**dot size** encodes playoff games played; **color** encodes regular-season archetype."
    )

    scatter = (
        alt.Chart(season_df)
        .mark_point(opacity=0.82, filled=True, size=90)
        .encode(
            x=alt.X("P/GP change:Q", title="Playoff P/GP − Regular-season P/GP"),
            y=alt.Y("TOI change:Q", title="Playoff ATOI − Regular-season ATOI (min)"),
            color=alt.Color(
                "archetype_label:N",
                title="REG archetype",
                scale=alt.Scale(domain=ARCHETYPE_COLOR_DOMAIN, range=ARCHETYPE_COLOR_RANGE),
                legend=alt.Legend(labelLimit=420, orient="right"),
            ),
            shape=alt.Shape(
                "model_shift_band:N",
                title="Model shift band",
                scale=alt.Scale(
                    domain=["Held steady", "Moderate shift", "Major shift", "Not projected"],
                    range=["circle", "square", "triangle-up", "cross"],
                ),
            ),
            size=alt.Size("po_games:Q", title="PO GP", scale=alt.Scale(range=[40, 280])),
            tooltip=[
                alt.Tooltip("full_name:N", title="Player"),
                alt.Tooltip("teams_played:N", title="Team"),
                alt.Tooltip("archetype_label:N", title="REG archetype"),
                alt.Tooltip("playoff_archetype_label:N", title="Projected PO archetype"),
                alt.Tooltip("reg_games:Q", title="REG GP"),
                alt.Tooltip("po_games:Q", title="PO GP"),
                alt.Tooltip("model_shift_score:Q", title="Model shift score", format=".3f"),
                alt.Tooltip("model_shift_band:N", title="Model shift band"),
                alt.Tooltip("P/GP change:Q", title="P/GP change", format="+.2f"),
                alt.Tooltip("TOI change:Q", title="ATOI change (min)", format="+.1f"),
                alt.Tooltip("stat_shift_score:Q", title="Stat shift score", format=".2f"),
            ],
        )
        .properties(width=820, height=420)
    )
    zero_x = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#9CA3AF", strokeDash=[4, 3]).encode(x="x:Q")
    zero_y = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#9CA3AF", strokeDash=[4, 3]).encode(y="y:Q")
    st.altair_chart(scatter + zero_x + zero_y, use_container_width=False)

    st.markdown("#### Biggest Playoff Profile Changes (sorted by model shift score)")
    st.markdown(render_profile_changes_table(season_df), unsafe_allow_html=True)

# ── Archetypes tab ─────────────────────────────────────────────────────────────
with tab_archetypes:
    st.subheader("Regular-Season Archetypes Under Playoff Pressure")
    st.markdown(
        "For each regular-season archetype and season, the chart shows how much that group's "
        "play style shifted in the playoffs — and how many players actually got re-classified "
        "into a different archetype. Hover over any circle for full detail."
    )
    arch = (
        base.groupby(["season", "Season", "archetype_label"], as_index=False)
        .agg(
            players=("player_id", "nunique"),
            median_model_shift=("model_shift_score", "median"),
            archetype_change_rate=("archetype_changed", "mean"),
            median_stat_shift=("stat_shift_score", "median"),
            median_pgp_change=("P/GP change", "median"),
            median_toi_change=("TOI change", "median"),
        )
    )
    arch = arch[arch["players"] >= 3].copy()

    season_sort = [season_key_to_label(s) for s in sorted(base["season"].unique())]
    arch["change_rate_pct"] = (arch["archetype_change_rate"] * 100).round(1)

    # Background grid so empty cells are visible
    bg = (
        alt.Chart(arch)
        .mark_rect(color="#F9FAFB", stroke="#E5E7EB", strokeWidth=0.5)
        .encode(
            x=alt.X("Season:O", title="Season", sort=season_sort),
            y=alt.Y("archetype_label:N", title="REG archetype", sort="-x",
                    axis=alt.Axis(labelLimit=320, titlePadding=18, labelPadding=8)),
        )
    )
    # Circles: color = model shift severity, size = archetype change rate
    dots = (
        alt.Chart(arch)
        .mark_circle(stroke="#FFFFFF", strokeWidth=1)
        .encode(
            x=alt.X("Season:O", title="Season", sort=season_sort),
            y=alt.Y("archetype_label:N", title="REG archetype", sort="-x",
                    axis=alt.Axis(labelLimit=320, titlePadding=18, labelPadding=8)),
            color=alt.Color(
                "median_model_shift:Q",
                title="Median model shift",
                scale=alt.Scale(scheme="orangered", domain=[0, 1.0]),
                legend=alt.Legend(gradientLength=120),
            ),
            size=alt.Size(
                "change_rate_pct:Q",
                title="Archetype change rate (%)",
                scale=alt.Scale(range=[30, 700]),
                legend=alt.Legend(values=[0, 25, 50, 75, 100]),
            ),
            tooltip=[
                alt.Tooltip("Season:O"),
                alt.Tooltip("archetype_label:N", title="REG archetype"),
                alt.Tooltip("players:Q", title="Players"),
                alt.Tooltip("median_model_shift:Q", title="Median model shift", format=".3f"),
                alt.Tooltip("change_rate_pct:Q", title="Archetype change rate (%)", format=".0f"),
                alt.Tooltip("median_stat_shift:Q", title="Median stat shift", format=".2f"),
                alt.Tooltip("median_pgp_change:Q", title="Median P/GP change", format="+.2f"),
                alt.Tooltip("median_toi_change:Q", title="Median TOI change (min)", format="+.1f"),
            ],
        )
    )
    heat = (bg + dots).properties(
        height=max(420, 48 * arch["archetype_label"].nunique()),
        padding={"right": 20, "top": 10, "bottom": 10},
    )
    st.markdown(
        "**How to read this:** Each circle is one archetype × season cell. "
        "🔴 **Color** (light→dark orange-red) = median model shift score — darker means players looked more different in the playoffs. "
        "⚫ **Size** = archetype change rate — bigger dot means a higher share of players got classified into a *different* archetype in the playoffs vs regular season."
    )
    st.altair_chart(heat, use_container_width=True)

    st.dataframe(
        arch.sort_values(["season", "median_model_shift"], ascending=[False, False]).rename(columns={
            "archetype_label": "REG archetype",
            "players": "Players",
            "median_model_shift": "Median model shift",
            "archetype_change_rate": "Archetype change rate",
            "median_stat_shift": "Median stat shift",
            "median_pgp_change": "Median P/GP change",
            "median_toi_change": "Median TOI change",
        })[["Season", "REG archetype", "Players", "Median model shift", "Archetype change rate",
            "Median stat shift", "Median P/GP change", "Median TOI change"]],
        use_container_width=True,
        hide_index=True,
    )

# ── Player Career tab ──────────────────────────────────────────────────────────
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
        valid_model = hist["model_shift_score"].dropna()
        c2.metric(
            "Median model shift",
            f"{valid_model.median():.2f}" if not valid_model.empty else "—",
        )
        c3.metric("Career PO GP", f"{int(hist['po_games'].sum()):,}")
        c4.metric("Career P/GP change", f"{hist['P/GP change'].mean():+.2f}")

        season_sort = [season_key_to_label(s) for s in sorted(hist["season"].unique())]
        profile = hist[[
            "Season", "REG P/GP", "PO P/GP", "reg_avg_toi_min", "po_avg_toi_min",
            "P/GP change", "TOI change", "stat_shift_score", "model_shift_score",
            "archetype_label", "playoff_archetype_label", "po_games",
        ]].copy()
        profile["REG ATOI"] = profile["reg_avg_toi_min"]
        profile["PO ATOI"] = profile["po_avg_toi_min"]
        profile["Profile"] = profile["archetype_label"] + " → " + profile["playoff_archetype_label"].fillna("not projected")
        profile["model_shift_score"] = pd.to_numeric(profile["model_shift_score"], errors="coerce").fillna(0.0)

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
                x=alt.X("Value:Q", title=title, axis=alt.Axis(titlePadding=14)),
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
                padding={"bottom": 36},
            )

        col_a, col_b = st.columns(2)
        with col_a:
            st.altair_chart(paired_delta_chart("REG P/GP", "PO P/GP", "Scoring Rate: Regular Season vs Playoffs", ".2f"), use_container_width=True)
        with col_b:
            st.altair_chart(paired_delta_chart("REG ATOI", "PO ATOI", "Ice Time: Regular Season vs Playoffs (min)", ".1f"), use_container_width=True)

        profile_long = profile.melt(
            id_vars=["Season", "stat_shift_score", "model_shift_score", "P/GP change", "TOI change", "po_games"],
            value_vars=["archetype_label", "playoff_archetype_label"],
            var_name="Split",
            value_name="Archetype",
        )
        profile_long["Split"] = profile_long["Split"].replace({
            "archetype_label": "Regular Season",
            "playoff_archetype_label": "Playoffs",
        })

        transition = (
            alt.Chart(profile_long)
            .mark_rect(cornerRadius=4)
            .encode(
                x=alt.X("Split:N", title=None, sort=["Regular Season", "Playoffs"]),
                y=alt.Y("Season:O", sort=season_sort, title=None),
                color=alt.Color(
                    "Archetype:N",
                    scale=alt.Scale(domain=ARCHETYPE_COLOR_DOMAIN, range=ARCHETYPE_COLOR_RANGE),
                    legend=None,
                ),
                tooltip=["Season", "Split", "Archetype",
                         alt.Tooltip("model_shift_score:Q", title="Model shift score", format=".3f"),
                         alt.Tooltip("stat_shift_score:Q", title="Stat shift score", format=".2f")],
            )
            .properties(height=max(180, 38 * len(profile)), title="Archetype Translation")
        )

        shift = (
            alt.Chart(profile)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("model_shift_score:Q", title="Model shift score"),
                y=alt.Y("Season:O", sort=season_sort, title=None),
                color=alt.Color(
                    "model_shift_score:Q",
                    title="Model shift",
                    scale=alt.Scale(scheme="orangered", domain=[0, 1.0]),
                ),
                tooltip=[
                    "Season",
                    alt.Tooltip("model_shift_score:Q", title="Model shift score", format=".3f"),
                    alt.Tooltip("stat_shift_score:Q", title="Stat shift score", format=".2f"),
                    alt.Tooltip("P/GP change:Q", format="+.2f"),
                    alt.Tooltip("TOI change:Q", title="ATOI change (min)", format="+.1f"),
                    alt.Tooltip("po_games:Q", title="PO GP"),
                ],
            )
            .properties(height=max(180, 38 * len(profile)), title="How Much the Playoff Profile Moved")
        )
        st.altair_chart(alt.hconcat(transition, shift, spacing=28), use_container_width=True)

        st.markdown(_render_table_html(compact_table(hist)), unsafe_allow_html=True)
