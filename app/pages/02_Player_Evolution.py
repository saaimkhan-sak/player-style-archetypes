import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode


# --- Make app/ importable when Streamlit runs pages/ as scripts ---
APP_DIR = Path(__file__).resolve().parents[1]  # .../app
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from lib import (
    available_seasons,
    season_key_to_label,
    load_all_seasons_group,
)

REPORTS_DIR = Path("reports")

st.set_page_config(page_title="Player Archetype Evolution", layout="wide")
st.title("Player Archetype Evolution")

# ----------------------------
# Helpers
# ----------------------------
ERA_ORDER = ["2000–2004", "2005–2009", "2010–2014", "2015–2019", "2020–2025"]

def season_start_year(season_key: str) -> int:
    try:
        return int(str(season_key)[:4])
    except Exception:
        return 0

def era5(season_key: str) -> str:
    y = season_start_year(season_key)
    if y < 2000:
        return "pre-2000"
    if y <= 2004:
        return "2000–2004"
    if y <= 2009:
        return "2005–2009"
    if y <= 2014:
        return "2010–2014"
    if y <= 2019:
        return "2015–2019"
    return "2020–2025"

def parse_trait_string(s: str):
    # "reg_shots_per60(+1.23),reg_points_per60(+0.77)"
    if not isinstance(s, str):
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        m = re.match(r"^([A-Za-z0-9_]+)\(([+-]?\d+\.?\d*)\)$", part)
        if m:
            out.append((m.group(1), float(m.group(2))))
    return out

def prettify_traits_lines(s: str, max_items: int = 4) -> str:
    """
    Convert a traits string like:
      reg_shots_per60(+1.23),reg_points_per60(+0.77)
    into multi-line readable text (one trait per line).
    """
    tokens = parse_trait_string(s)
    lines = []
    for feat, z in tokens[:max_items]:
        arrow = "↑" if z >= 0 else "↓"
        lines.append(f"{arrow} {feat} ({z:+.1f}σ)")
    return "\n".join(lines)


def build_unique_name_summary(cluster: int, high_tokens, low_tokens):
    """
    Same naming logic as your main app: generates a descriptive archetype name.
    """
    high_feats = {f for f, _ in high_tokens}
    low_feats = {f for f, _ in low_tokens}

    offense_hi = any(f in high_feats for f in ["reg_points_per60","reg_goals_per60","reg_assists_per60","reg_shots_per60"])
    playmaking_hi = "reg_assists_per60" in high_feats
    shooting_hi = "reg_shots_per60" in high_feats
    blocks_hi = "reg_blocked_shots_per60" in high_feats
    hits_hi = "reg_hits_per60" in high_feats
    pim_hi = "reg_pim_per60" in high_feats
    takeaways_hi = "reg_takeaways_per60" in high_feats
    giveaways_lo = "reg_giveaways_per60" in low_feats
    pk_hi = "reg_pk_share" in high_feats
    pp_hi = "reg_pp_share" in high_feats
    fo_hi = ("reg_fo_taken_per_game" in high_feats) or ("reg_fo_pct" in high_feats)

    if pim_hi and hits_hi:
        return "Agitating Heavy-Contact Forward", ""
    if blocks_hi and hits_hi:
        return "Shot-Blocking Contact Specialist", ""
    if offense_hi and playmaking_hi and shooting_hi:
        return "High-Volume Playmaking Scorer", ""
    if takeaways_hi and giveaways_lo:
        return "Puck-Pressure Two-Way Creator", ""
    if fo_hi and not offense_hi:
        return "Deployment / Faceoff Specialist", ""
    if pk_hi and not pp_hi:
        return "PK-Leaning Defensive Role", ""
    if pp_hi and not pk_hi:
        return "PP-Leaning Offensive Role", ""

    return f"Mixed Profile Archetype {cluster}", ""

def archetype_description_from_traits(name: str, top_traits: str, low_traits: str) -> str:
    """
    Convert the model name + traits into a human description.
    Keep this short and intuitive (1–2 sentences).
    """
    n = str(name)
    top = str(top_traits).lower()
    low = str(low_traits).lower()

    if "agitating" in n.lower() or ("reg_pim_per60" in top and "reg_hits_per60" in top):
        return "Physical, high-edge profile: plays a heavy game and tends to take more penalties. Often lower offensive creation than scoring archetypes."
    if "shot-blocking" in n.lower() or "reg_blocked_shots_per60" in top:
        return "Defense-tilted profile: blocks shots and plays physical minutes. Typically contributes more via defense/role usage than raw scoring."
    if "playmaking" in n.lower() or ("reg_assists_per60" in top and "reg_shots_per60" in top):
        return "Offense-driving profile: generates shots and assists at high rates. Tends to produce strong points/60."
    if "two-way" in n.lower() or ("reg_takeaways_per60" in top and "reg_giveaways_per60" in low):
        return "Pressure-and-recover profile: creates takeaways while limiting giveaways. Contributes on both sides of the puck."
    if "faceoff" in n.lower() or "reg_fo_taken_per_game" in top or "reg_fo_pct" in top:
        return "Deployment-driven profile: used in draws and role minutes. The style reflects how coaches deploy the player."
    if "pk-leaning" in n.lower() or "reg_pk_share" in top:
        return "Penalty-kill leaning profile: more value comes from shorthanded usage and defensive role."
    if "pp-leaning" in n.lower() or "reg_pp_share" in top:
        return "Power-play leaning profile: production is driven by scoring-role deployment and PP usage."

    return "Mixed profile: blends multiple trait patterns rather than cleanly matching one extreme archetype."


@st.cache_data
def load_traits_csv(group: str, season_key: str) -> pd.DataFrame:
    p = REPORTS_DIR / f"archetype_traits_{group}_{season_key}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["season"] = season_key
    return df

@st.cache_data
def build_season_cluster_to_name(group: str) -> dict[tuple[str,int], str]:
    """
    Returns mapping (season_key, cluster_id) -> descriptive archetype name.
    """
    mapping: dict[tuple[str,int], str] = {}
    for sk in available_seasons():
        t = load_traits_csv(group, sk)
        if t.empty:
            continue
        for _, tr in t.iterrows():
            kk = int(tr["cluster"])
            ht = parse_trait_string(tr.get("top_traits", ""))
            lt = parse_trait_string(tr.get("low_traits", ""))
            nm, _ = build_unique_name_summary(kk, ht, lt)
            mapping[(sk, kk)] = nm
    return mapping

def min_to_mmss(minutes):
    m = float(minutes) if pd.notna(minutes) else 0.0
    total_s = int(round(m * 60))
    mm = total_s // 60
    ss = total_s % 60
    return f"{mm:02d}:{ss:02d}"

def confidence_chip(avg_conf_pct: float) -> tuple[str, str, str]:
    """
    Returns (text, bg_color, fg_color)
    """
    v = float(avg_conf_pct)
    if v >= 95:
        return "Very stable", "#DCFCE7", "#166534"   # green
    if v >= 90:
        return "Stable", "#BBF7D0", "#166534"
    if v >= 80:
        return "Moderate", "#FEF9C3", "#854D0E"      # yellow
    if v >= 70:
        return "Mixed", "#FFEDD5", "#9A3412"         # orange
    return "Highly mixed", "#FEE2E2", "#991B1B"      # red


def mixedness_chip(avg_mixed: float) -> tuple[str, str, str]:
    """
    Mixedness = 1 - top_prob, so lower is better (more archetype certainty).
    """
    v = float(avg_mixed)
    if v <= 0.05:
        return "Very consistent", "#DCFCE7", "#166534"
    if v <= 0.10:
        return "Consistent", "#BBF7D0", "#166534"
    if v <= 0.20:
        return "Some variety", "#FEF9C3", "#854D0E"
    if v <= 0.30:
        return "Role-shifting", "#FFEDD5", "#9A3412"
    return "Blend of styles", "#FEE2E2", "#991B1B"


def render_chip(text: str, bg: str, fg: str) -> str:
    return f"""
<span style="
    display:inline-flex;
    align-items:center;
    justify-content:center;
    padding:2px 10px;
    border-radius:999px;
    background:{bg};
    color:{fg};
    font-weight:700;
    font-size:0.85rem;
    border:1px solid rgba(0,0,0,0.08);
    white-space:nowrap;">
    {text}
</span>
"""


    
PRELINE_CELL = JsCode("""
function(params) {
  return {
    whiteSpace: 'pre-line',
    lineHeight: '1.25',
    display: 'flex',
    alignItems: 'center'
  };
}
""")

def show_multiline_table(df: pd.DataFrame, height: int = 420):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=True, sortable=True, filter=True, wrapText=True, autoHeight=True)
    gb.configure_grid_options(domLayout="autoHeight", suppressSizeToFit=True, alwaysShowHorizontalScroll=True)

    # Apply multi-line style to *every* column
    for c in df.columns:
        gb.configure_column(c, cellStyle=PRELINE_CELL)

    AgGrid(
        df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.NO_UPDATE,
        theme="streamlit",
        height=height,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
    )


# ----------------------------
# Explainer
# ----------------------------
with st.expander("Understanding the Evolution of a Player's Archetype", expanded=False):
    st.markdown("""
### How to Use This Tool
When you search and select a player, the dropdown shows their **name**, **position**, and the **most recent season in the dataset** for that player (i.e. 2024–2025).  

### What Does the Evolution Really Mean?
- **Stable top archetype + high confidence** → consistent role/style across years  
- **Shifts in top archetype** → role changes, team/system changes, aging, or deployment changes  
- **Lower confidence** → “mixed profile” seasons where the player blends multiple archetype patterns

### What is Mixedness?
In this table, you will see a value for each player called "mixedness". What is that? 

I define **Mixedness** as:
""")
    st.latex(r"\text{Mixedness} = 1 - \max_k(p_{ik})")
    st.markdown("where **maxₖ(pᵢₖ)** is the probability of the player’s **top archetype** that season.")
    st.markdown("""
- Mixedness near **0.00** → the model is very confident the player fits a single archetype  
- Mixedness >= **0.40** → the player blends multiple archetypes (probability mass is spread out)
""")

# ----------------------------
# GLOBAL Archetype Glossary (before group selector)
# ----------------------------
st.markdown("## Archetype Glossary")

st.markdown("""
To make it simple, I aggregated all the archetypes the model generated across all the seasons analyzed and placed them in the below table.

""")

# Load multi-season data for both groups
all_f = load_all_seasons_group("forwards")
all_d = load_all_seasons_group("defense")
all_combined = pd.concat([all_f, all_d], ignore_index=True)

map_f = build_season_cluster_to_name("forwards")
map_d = build_season_cluster_to_name("defense")

def traits_registry(group: str, mapping: dict[tuple[str,int], str]) -> dict[str, dict[str, str]]:
    """
    Returns: archetype_name -> {"high": "...", "low": "..."} aggregated from all seasons.
    We keep the most common (mode) high/low trait strings across seasons for that archetype name.
    """
    reg = {}
    counts = {}

    for sk in available_seasons():
        t = load_traits_csv(group, sk)
        if t.empty:
            continue
        for _, tr in t.iterrows():
            kk = int(tr["cluster"])
            name = mapping.get((sk, kk))
            if not name:
                continue

            hi = str(tr.get("top_traits", "")).strip()
            lo = str(tr.get("low_traits", "")).strip()
            if not hi and not lo:
                continue

            if name not in counts:
                counts[name] = {"hi": {}, "lo": {}}
            counts[name]["hi"][hi] = counts[name]["hi"].get(hi, 0) + 1
            counts[name]["lo"][lo] = counts[name]["lo"].get(lo, 0) + 1

    for name, c in counts.items():
        hi = max(c["hi"].items(), key=lambda x: x[1])[0] if c["hi"] else ""
        lo = max(c["lo"].items(), key=lambda x: x[1])[0] if c["lo"] else ""
        reg[name] = {
            "high": prettify_traits_lines(hi, max_items=4),
            "low": prettify_traits_lines(lo, max_items=3),
            "desc": archetype_description_from_traits(name, hi, lo),
        }
    return reg

def build_glossary(group: str, all_df: pd.DataFrame, mapping: dict[tuple[str,int], str], traits_map: dict[str, dict[str,str]]) -> pd.DataFrame:
    if all_df.empty:
        return pd.DataFrame()

    # Determine max K for that group dataset
    pcols = [c for c in all_df.columns if isinstance(c, str) and c.startswith("p") and c[1:].isdigit()]
    if not pcols:
        return pd.DataFrame()

    # Build long rows: player-season membership for every k
    parts = []
    for pc in pcols:
        k = int(pc[1:])
        tmp = all_df[["season","player_id","full_name","position", pc]].copy()
        tmp = tmp.rename(columns={pc: "prob"})
        tmp["k"] = k
        tmp["archetype_name"] = tmp.apply(lambda r: mapping.get((r["season"], int(r["k"])), None), axis=1)
        tmp = tmp.dropna(subset=["archetype_name"])
        parts.append(tmp)

    long = pd.concat(parts, ignore_index=True)
    long["era"] = long["season"].apply(era5)
    long = long[long["era"].isin(ERA_ORDER)].copy()

    # filter: avoid tiny samples if reg_games exists
    if "reg_games" in all_df.columns:
        rg = all_df[["season","player_id","reg_games"]].copy()
        long = long.merge(rg, on=["season","player_id"], how="left")
        long = long[long["reg_games"].fillna(0) >= 15].copy()

    # For each archetype_name, pick 1 exemplar per era (no duplicates)
    rows = []
    for name, sub in long.groupby("archetype_name"):
        # --- exemplars: 5 most recent, 100% confidence (i.e., prob == 1.0), no season text ---
        sub = sub.copy()
        sub["start_year"] = sub["season"].astype(str).str[:4].astype(int)

        # Only keep rows where the player is a "pure" member of this archetype for that season
        # Use >= 0.999 to be robust to float storage
        pure = sub[sub["prob"] >= 0.999].copy()

        # Sort by most recent season, then highest prob
        pure = pure.sort_values(["start_year", "prob"], ascending=[False, False])

        # Pick unique players (no duplicates) up to 5
        used = set()
        exemplars = []
        for _, r in pure.iterrows():
            pid = int(r["player_id"])
            if pid in used:
                continue
            used.add(pid)
            exemplars.append(str(r["full_name"]))
            if len(exemplars) == 5:
                break

        # If fewer than 5 pure members exist, optionally fall back to next-most-confident recent players
        # (comment this block out if you want STRICTLY 100% only)
        if len(exemplars) < 5:
            fallback = sub.sort_values(["start_year", "prob"], ascending=[False, False])
            for _, r in fallback.iterrows():
                pid = int(r["player_id"])
                if pid in used:
                    continue
                used.add(pid)
                exemplars.append(str(r["full_name"]))
                if len(exemplars) == 5:
                    break

        exemplar_lines = exemplars  # already most recent first

        rows.append({
            "Archetype name": name,
            "Description": traits_map.get(name, {}).get("desc", ""),
            "High traits": traits_map.get(name, {}).get("high", ""),
            "Low traits": traits_map.get(name, {}).get("low", ""),
            "Exemplars": "\n".join(exemplar_lines),
        })

    out = pd.DataFrame(rows).sort_values("Archetype name")
    return out

traits_f = traits_registry("forwards", map_f)
traits_d = traits_registry("defense", map_d)

glossary_f = build_glossary("forwards", all_f, map_f, traits_f)
glossary_d = build_glossary("defense", all_d, map_d, traits_d)

map_any = build_season_cluster_to_name("forwards")
traits_any = traits_registry("forwards", map_any)  # if you’re still adding traits/desc
glossary = build_glossary("combined", all_combined, map_any, traits_any)

st.markdown("")
if glossary.empty:
    st.info("No glossary available yet.")
else:
    show_multiline_table(glossary, height=650)  # or st.dataframe if you prefer




# ----------------------------
# Player evolution controls
# ----------------------------
st.markdown("## Player Evolution")

group = st.selectbox("Group", ["forwards", "defense"], index=0)

all_df = all_f if group == "forwards" else all_d
mapping = map_f if group == "forwards" else map_d

if all_df.empty:
    st.warning("No multi-season data found for this group in data/app/.")
    st.stop()

latest = (all_df.sort_values("season")
              .groupby("player_id", as_index=False)
              .tail(1)
              .copy())

latest["Most recent season played"] = latest["season"].apply(season_key_to_label)

# No team in dropdown label (per request)
latest["display"] = (
    latest["full_name"].astype(str)
    + " — " + latest["position"].fillna("UNK").astype(str)
    + " — " + latest["Most recent season played"].astype(str)
)

st.markdown("### Select a player")
query = st.text_input("Search player name", value="")

opts_df = latest.copy()
if query.strip():
    opts_df = opts_df[opts_df["full_name"].str.contains(query, case=False, na=False)].copy()

opts_df = opts_df.sort_values(["season", "full_name"], ascending=[False, True])
options = opts_df["display"].tolist()

if not options:
    st.info("No matches. Try a different search.")
    st.stop()

choice = st.selectbox("Matches", options, index=0, key="player_match_select")
player_id = int(opts_df.set_index("display").loc[choice, "player_id"])
st.caption(f"Selected: {choice}")

# ----------------------------
# Player history
# ----------------------------
hist = all_df[all_df["player_id"] == player_id].copy()
hist = hist.sort_values("season")
hist["Season"] = hist["season"].apply(season_key_to_label)
hist["Confidence (%)"] = (hist["confidence"].astype(float) * 100).round(1)
hist["Mixedness"] = (1.0 - hist["confidence"].astype(float)).round(3)

def top_label(row) -> str:
    k = int(row.get("top_cluster", 0))
    nm = mapping.get((row["season"], k), f"Archetype {k}")
    return f"A{k} — {nm}"

hist["Top archetype (season-specific)"] = hist.apply(top_label, axis=1)

# Summary metrics
c1, c2, c2chip, c3, c3chip = st.columns([1.1, 0.55, 0.95, 0.55, 0.95])

avg_conf = float(hist["Confidence (%)"].mean())
avg_mix = float(hist["Mixedness"].mean())

conf_txt, conf_bg, conf_fg = confidence_chip(avg_conf)
mix_txt, mix_bg, mix_fg = mixedness_chip(avg_mix)

with c1:
    st.metric("Seasons in dataset", len(hist))

with c2:
    st.metric("Avg confidence", f"{avg_conf:.1f}%")

with c2chip:
    # vertical spacing so chip sits centered next to the big number
    st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
    st.markdown(render_chip(conf_txt, conf_bg, conf_fg), unsafe_allow_html=True)

with c3:
    st.metric("Avg mixedness", f"{avg_mix:.3f}")

with c3chip:
    st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
    st.markdown(render_chip(mix_txt, mix_bg, mix_fg), unsafe_allow_html=True)

hist = hist.copy()
hist["changed"] = hist["Top archetype (season-specific)"].ne(hist["Top archetype (season-specific)"].shift(1))

# 1) Clean line chart: confidence over time (no text labels)
st.markdown("### Style timeline")

st.markdown("""
* Hover over data points to see the full details.
* The circled points indicate years where there was a change in player archetype from the previous year.
""")

st.divider()

base = alt.Chart(hist).encode(
    x=alt.X("Season:O", axis=alt.Axis(labelAngle=0), title="Season")
)

line = base.mark_line(point=True).encode(
    y=alt.Y("Confidence (%):Q", title="Top-archetype confidence (%)", scale=alt.Scale(domain=[0, 100])),
    tooltip=[
        alt.Tooltip("Season:O", title="Season"),
        alt.Tooltip("Top archetype (season-specific):N", title="Top archetype"),
        alt.Tooltip("Confidence (%):Q", title="Confidence", format=".1f"),
        alt.Tooltip("Mixedness:Q", title="Mixedness", format=".3f"),
        alt.Tooltip("teams_played:N", title="Teams"),
        alt.Tooltip("position:N", title="Pos"),
    ],
).properties(height=280)

# Highlight only the seasons where archetype changed vs previous season
change_pts = alt.Chart(hist[hist["changed"]]).mark_point(size=160).encode(
    x="Season:O",
    y="Confidence (%):Q",
    tooltip=[
        alt.Tooltip("Season:O", title="Season"),
        alt.Tooltip("Top archetype (season-specific):N", title="New archetype"),
        alt.Tooltip("Confidence (%):Q", title="Confidence", format=".1f"),
        alt.Tooltip("Mixedness:Q", title="Mixedness", format=".3f"),
        alt.Tooltip("teams_played:N", title="Teams"),
        alt.Tooltip("position:N", title="Pos"),
    ],
)

st.altair_chart((line + change_pts).properties(height=340), use_container_width=True)


# 2) A readable per-season “archetype strip” underneath (no overlap)
st.markdown("**Archetype and Career Stats by Season**")

strip = hist[["Season", "Top archetype (season-specific)", "Confidence (%)"]].copy()
strip["Top archetype (season-specific)"] = strip["Top archetype (season-specific)"].astype(str)

# Optional: shorten display just a bit (keeps readability)
# e.g. "A2 — High-Volume Playmaking Scorer" stays fine, but you can truncate if you want:
# strip["Top archetype (season-specific)"] = strip["Top archetype (season-specific)"].str.slice(0, 42)

# Show as a clean table (no scroll, very readable)
#st.markdown("**Top archetype by season + key stats**")

# Build a clean, readable season-by-season table
cols = [
    "Season",
    "teams_played",
    "position",
    "Top archetype (season-specific)",
    "Confidence (%)",
    "Mixedness",

    # Regular season
    "reg_games",
    "reg_avg_toi_min",
    "reg_points",
    "reg_goals",
    "reg_assists",
    "reg_shots",
    "reg_plus_minus",
    "reg_pim",

    # Playoffs (may be 0 for seasons with no playoffs)
    "po_games",
    "po_avg_toi_min",
    "po_points",
    "po_goals",
    "po_assists",
    "po_shots",
    "po_plus_minus",
    "po_pim",
]

# keep only columns that exist in your hist dataframe
cols = [c for c in cols if c in hist.columns]

strip = hist[cols].copy()

# Format TOI columns if present
if "reg_avg_toi_min" in strip.columns:
    strip["REG ATOI"] = strip["reg_avg_toi_min"].apply(min_to_mmss)
    strip = strip.drop(columns=["reg_avg_toi_min"])
if "po_avg_toi_min" in strip.columns:
    strip["PO ATOI"] = strip["po_avg_toi_min"].apply(min_to_mmss)
    strip = strip.drop(columns=["po_avg_toi_min"])

# Rename columns for readability
rename = {
    "teams_played": "Team(s)",
    "position": "Pos",
    "reg_games": "REG GP",
    "reg_points": "REG P",
    "reg_goals": "REG G",
    "reg_assists": "REG A",
    "reg_shots": "REG SOG",
    "reg_plus_minus": "REG +/-",
    "reg_pim": "REG PIM",
    "po_games": "PO GP",
    "po_points": "PO P",
    "po_goals": "PO G",
    "po_assists": "PO A",
    "po_shots": "PO SOG",
    "po_plus_minus": "PO +/-",
    "po_pim": "PO PIM",
}
strip = strip.rename(columns=rename)

# Put columns in a sensible final order (after renaming)
final_cols = [
    "Season", "Team(s)", "Pos",
    "Top archetype (season-specific)", "Confidence (%)", "Mixedness",
    "REG GP", "REG ATOI", "REG P", "REG G", "REG A", "REG SOG", "REG +/-", "REG PIM",
    "PO GP", "PO ATOI", "PO P", "PO G", "PO A", "PO SOG", "PO +/-", "PO PIM",
]
final_cols = [c for c in final_cols if c in strip.columns]
strip = strip[final_cols]

st.dataframe(strip, use_container_width=True, hide_index=True)

