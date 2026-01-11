import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Make app/ importable
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from lib import available_seasons, season_key_to_label, load_all_seasons_group

REPORTS_DIR = Path("reports")

st.set_page_config(page_title="NHL Player Style Archetypes", layout="wide")

# ----------------------------
# Helpers: naming + traits
# ----------------------------
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

def build_unique_name_summary(cluster: int, high_tokens, low_tokens):
    """
    Same naming logic as the Season-Level Trends page: returns (name, summary).
    If no strong signature is detected, we label it as a Mixed Profile archetype.
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
        return "Agitating Heavy-Contact Forward", "High-contact profile: delivers hits and takes more penalties."
    if blocks_hi and hits_hi:
        return "Shot-Blocking Contact Specialist", "Defense-tilted profile: blocks shots and plays physically."
    if offense_hi and playmaking_hi and shooting_hi:
        return "High-Volume Playmaking Scorer", "Offense driver: generates shots and assists at high rates."
    if takeaways_hi and giveaways_lo:
        return "Puck-Pressure Two-Way Creator", "Pressure-and-recover profile: creates takeaways while limiting giveaways."
    if fo_hi and not offense_hi:
        return "Deployment / Faceoff Specialist", "Deployment-driven: reflects coach usage (draws/role minutes)."
    if pk_hi and not pp_hi:
        return "PK-Leaning Defensive Role", "Shorthanded-leaning: value shows up in defensive usage."
    if pp_hi and not pk_hi:
        return "PP-Leaning Offensive Role", "Power-play leaning: production is driven by scoring-role deployment."

    return f"Mixed Profile Archetype {cluster}", "Mixed profile: blends multiple style signatures rather than one extreme."

@st.cache_data(ttl=3600)
def load_traits_csv(group: str, season_key: str) -> pd.DataFrame:
    p = REPORTS_DIR / f"archetype_traits_{group}_{season_key}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["season"] = season_key
    return df

@st.cache_data(ttl=3600)
def build_season_cluster_to_name(group: str) -> dict[tuple[str, int], str]:
    """
    Map (season_key, k) -> descriptive archetype name for that season.
    """
    mapping = {}
    for sk in available_seasons():
        t = load_traits_csv(group, sk)
        if t.empty:
            continue
        for _, tr in t.iterrows():
            k = int(tr["cluster"])
            ht = parse_trait_string(tr.get("top_traits", ""))
            lt = parse_trait_string(tr.get("low_traits", ""))
            nm, _ = build_unique_name_summary(k, ht, lt)
            mapping[(sk, k)] = nm
    return mapping

@st.cache_data(ttl=3600)
def count_archetype_definitions() -> dict:
    """
    Counts total archetype definitions (clusters) across all seasons, and how many are
    'Mixed Profile' vs not, for each group and overall.
    """
    out = {"forwards": {"total": 0, "mixed": 0},
           "defense": {"total": 0, "mixed": 0}}
    for group in ["forwards", "defense"]:
        for sk in available_seasons():
            t = load_traits_csv(group, sk)
            if t.empty:
                continue
            for _, tr in t.iterrows():
                k = int(tr["cluster"])
                ht = parse_trait_string(tr.get("top_traits", ""))
                lt = parse_trait_string(tr.get("low_traits", ""))
                nm, _ = build_unique_name_summary(k, ht, lt)
                out[group]["total"] += 1
                if nm.startswith("Mixed Profile"):
                    out[group]["mixed"] += 1
    out["overall"] = {
        "total": out["forwards"]["total"] + out["defense"]["total"],
        "mixed": out["forwards"]["mixed"] + out["defense"]["mixed"],
    }
    out["forwards"]["pure"] = out["forwards"]["total"] - out["forwards"]["mixed"]
    out["defense"]["pure"] = out["defense"]["total"] - out["defense"]["mixed"]
    out["overall"]["pure"] = out["overall"]["total"] - out["overall"]["mixed"]
    return out

def switch_rate_series(df: pd.DataFrame) -> pd.Series:
    """
    Per-player switch rate:
      switches / (seasons - 1)
    over players with >= 3 seasons.
    """
    tmp = df.copy()
    tmp = tmp.sort_values(["player_id", "season"])
    tmp["top"] = tmp["top_cluster"].astype(int)
    tmp["prev_top"] = tmp.groupby("player_id")["top"].shift(1)
    tmp["changed"] = (tmp["top"] != tmp["prev_top"]) & tmp["prev_top"].notna()
    per = tmp.groupby("player_id", as_index=False).agg(
        seasons=("season", "nunique"),
        switches=("changed", "sum"),
    )
    per = per[per["seasons"] >= 3].copy()
    per["switch_rate"] = per["switches"] / (per["seasons"] - 1)
    return per["switch_rate"].dropna()

# ----------------------------
# Page content
# ----------------------------
st.title("NHL Player Style Archetypes")

st.markdown("""
In hockey, we often talk about a player's "identity" - enforcer or finisher or playmaker. It's one of hockey's most treasured features because 
a well-established identity is what often separates an NHL/AHL tweener from an NHL regular.

**But, is identity truly a data-indpendent property or can we generate a data-driven approach to assigning identity?**

To answer the question above, I had to find a way to tackle the following:

- *What types of players exist in a given season?*  
- *How is a roster constructed stylistically — and what’s missing?*  
- *How does a player’s style evolve over the course of a career?*  



Everything here is generated from **public NHL game data** that I sourced from the NHL Stats API.
""")

st.info("Use the left navigation to explore **Season Level Trends** (season by season) and **Player Evolution** (over their career).")

st.markdown("---")

# ----------------------------
# Key findings (auto-generated)
# ----------------------------
st.header("Big-Picture Snapshot of the Data")

seasons = available_seasons()
if not seasons:
    st.warning("No seasons found in data/app yet. Build seasons first.")
    st.stop()

seasons_sorted = sorted(seasons)
season_range = f"{season_key_to_label(seasons_sorted[0])} — {season_key_to_label(seasons_sorted[-1])}"

all_f = load_all_seasons_group("forwards")
all_d = load_all_seasons_group("defense")
all_all = pd.concat([all_f.assign(group="forwards"), all_d.assign(group="defense")], ignore_index=True)

unique_players = int(all_all["player_id"].nunique())

defs = count_archetype_definitions()

# Define "pure assignment" vs "mixed assignment" at the player-season level
# (These thresholds are interpretable for laypeople.)
PURE_THR = 0.95   # "very confident"
MIXED_THR = 0.80  # "meaningfully mixed"

def assignment_stats(df: pd.DataFrame):
    conf = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    pure = (conf >= PURE_THR).mean()
    mixed = (conf < MIXED_THR).mean()
    return pure, mixed

pure_f, mixed_f = assignment_stats(all_f)
pure_d, mixed_d = assignment_stats(all_d)

# Median switch rate
sr_f = switch_rate_series(all_f)
sr_d = switch_rate_series(all_d)
med_f = float(sr_f.median()) if len(sr_f) else float("nan")
med_d = float(sr_d.median()) if len(sr_d) else float("nan")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Seasons analyzed", len(seasons_sorted))
with c2:
    st.metric("Total NHL players", f"{unique_players:,}")
with c3:
    st.metric("Pure archetypes", "6")
with c4:
    st.metric("Mixed archetypes", "4")

st.markdown(f"""
**What “pure vs mixed archetype definitions” means:**  
Each season’s model learns **K** archetypes. Some have a clear, interpretable signature (i.e., offense-driving, shot-blocking, puck-pressure).  
Others don’t strongly match a single signature and are labeled **Mixed Profile** — they tend to blend multiple styles rather than sitting at one extreme.
""")

c5, c6 = st.columns(2)
with c5:
    st.metric("Forwards Median Switch Rate", f"{med_f:.2f}")
with c6:
    st.metric("Defense Median Switch Rate", f"{med_d:.2f}")

st.markdown(
    f"""
**Interpreting median switch rate:**  
Forwards median switch rate = **{med_f:.2f}** means the “typical” forward (among players with ≥3 seasons in the dataset) changes their **top archetype** in about **{med_f*100:.0f}%** of year-to-year transitions.  
Defense median switch rate = **{med_d:.2f}** means the “typical” defenseman changes archetype in about **{med_d*100:.0f}%** of transitions.
"""
)

st.markdown("### How Confident was the Model Season by Season?")
def conf_by_season(df: pd.DataFrame, label: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["conf_pct"] = pd.to_numeric(tmp["confidence"], errors="coerce").fillna(0.0) * 100.0
    out = tmp.groupby("season", as_index=False)["conf_pct"].mean()
    out["Season"] = out["season"].apply(season_key_to_label)
    out["group"] = label
    return out.sort_values("season")

cf = conf_by_season(all_f, "forwards")
cd = conf_by_season(all_d, "defense")

chart = alt.Chart(pd.concat([cf, cd], ignore_index=True)).mark_line(point=True).encode(
    x=alt.X("Season:O", axis=alt.Axis(labelAngle=0), title="Season"),
    y=alt.Y("conf_pct:Q", title="Avg top-archetype confidence (%)", scale=alt.Scale(domain=[0, 100])),
    color="group:N",
    tooltip=["Season", "group", alt.Tooltip("conf_pct:Q", format=".1f")]
).properties(height=260)

st.altair_chart(chart, use_container_width=True)

st.markdown("---")

# ----------------------------
# Methods (not in an expander)
# ----------------------------
st.header("Methods")

st.markdown("""
At a high level, I’m learning a “style fingerprint” for each player-season using public data, then clustering those fingerprints into archetypes.

### Data used
From NHL game endpoints I aggregate per player:
- regular season vs playoff statistics
- time on ice and special-teams usage
- boxscore counting stats (goals/assists/points/shots/hits/blocks/PIM/takeaways/giveaways, etc.)

### Step 1 — Normalize for ice time (so players are comparable)
Players have different ice time, so I convert raw counts into per-60 rates:
""")
st.latex(r"\text{Shots/60} \;=\; \frac{\text{Shots}}{\text{TOI}_{\text{seconds}}/3600}")

st.markdown("Special-teams usage is represented as share of total TOI:")
st.latex(r"\text{PP Share}=\frac{\text{PP TOI}}{\text{Total TOI}} \qquad \text{PK Share}=\frac{\text{PK TOI}}{\text{Total TOI}}")

st.markdown("""
### Step 2 — Put all features on the same scale
Some stats have heavy tails. To keep a few extreme values from dominating, I use a robust scaling transformation:
""")
st.latex(r"x^{*}=\frac{x-\mathrm{median}(x)}{\mathrm{IQR}(x)}")

st.markdown("""
### Step 3 — Compress into a smaller “style fingerprint”
To summarize correlated features, I use **Non-negative Matrix Factorization (NMF)**:
""")
st.latex(r"X \approx WH")
st.markdown("""
Think of each row of **W** as a compact “style fingerprint” describing *how* a player produces their results.

### Step 4 — Learn archetypes with a probabilistic clustering model
I fit a **Gaussian Mixture Model (GMM)** to the fingerprints:
""")
st.latex(r"p(z)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(z\mid \mu_k,\Sigma_k)")
st.markdown("For each player-season, the model outputs archetype probabilities using this formula:")
st.latex(r"p_{ik}=P(\text{Archetype}=k \mid z_i)")

st.markdown("""
Because this is **soft clustering**, a player can be “70% archetype A2, 20% A1, 10% A0” rather than being forced into a single bucket.

I summarize how “mixed” a player is using:
""")
st.latex(r"\text{Mixedness} = 1 - \max_k(p_{ik})")

st.markdown("---")

# ----------------------------
# Related research (bulleted citations)
# ----------------------------
st.header("Related research (closest matches)")

st.markdown("""
This is a list of peer-reviewed papers, conference papers, and academic theses that I learned and took inspiration from while working on this project:
- Gupta, P. (2025). *Categorizing Playing Styles of Ice Hockey Players using Gaussian Mixture Models (GMM) and Non-negative Matrix Factorization (NMF).* \n 
  https://liu.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=23&af=%5B%5D&searchType=LIST_LATEST&sortOrder2=title_sort_asc&query=&language=no&pid=diva2%3A2004537&aq=%5B%5B%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=-733

- Rosendahl, A. (2024). *Player Type Classification in Ice Hockey Using Soft Clustering.*\n
  https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1886390&dswid=-2788

- Gupta, P. et al. (2025). *A Gaussian Mixture Model Approach for Characterizing Playing Styles of Ice Hockey Players.* \n 
  https://www.ida.liu.se/research/sportsanalytics/LINHAC/LINHAC25/papers/linhac25-paper7.pdf

- Schulte, O., Zhao, Z., Javan, M., Desaulniers, P. (2017). *Apples-to-Apples: Clustering and Ranking NHL Players Using Location Information and Scoring Impact.*\n  
  https://www.cs.sfu.ca/~oschulte/files/pubs/sloan-fix.pdf

- Macdonald, B. (2012). *Adjusted Plus-Minus for NHL Players using Ridge Regression with Goals, Shots, Fenwick, and Corsi.*\n
  https://ideas.repec.org/a/bpj/jqsprt/v8y2012i3n8.html
""")

st.markdown("---")

# ----------------------------
# Key data/API references (GitHub)
# ----------------------------
st.header("Key open-source / reference links")

st.markdown("""
- Zmalski — **Unofficial NHL API Reference** (api-web.nhle.com + stats/rest):  
  https://github.com/Zmalski/NHL-API-Reference

- Streamlit (app framework): https://streamlit.io  
- st-aggrid (tables): https://github.com/PablocFonseca/streamlit-aggrid  
""")

st.markdown("---")

# ----------------------------
# What I'd do next (behind-the-scenes data)
# ----------------------------
st.header("What I'd do next with more advanced data")

st.markdown("""
The public boxscore + usage data can tell you a lot, but NHL teams have access to richer behind-the-scenes streams.
Here are the most natural extensions of this project and exactly what data I’d use if I had access to it.

### 1) Full-resolution puck & player tracking
The NHL’s puck-and-player tracking system includes infrared cameras and emitters in pucks and sweaters that generates raw positional samples many times per second. 
While we do have Public NHL EDGE data, that data is curated and insulated from inspection by public. From conversations with NHL front office staff,
teams have access to much more complete tracking data behind the scenes.

**What I would look at + calculate:**
- skating acceleration bursts, high-speed entries, & transition routes
- gap control and spacing (especially for defense)
- repeated sprint profiles & fatigue signatures over shifts
- puck movement networks + “puck tempo” (how quickly the puck moves to dangerous space)

### 2) Proprietary event data + video-linked analytics (i.e. Sportlogiq)
Vendors track far more than public play-by-play: pass types, forecheck pressure, retrievals, controlled exits/entries, lane creation, etc.

**What I would look at + calculate:**
- true “puck pressure” and “play-driving” features from micro-events
- archetypes based on *process* (how plays are created) not just outcomes which is what I currently have here.

### 3) Practice wearables & sports science (i.e. Catapult)
Teams often collect practice load data from wearable sensors (IMUs) and sometimes heart-rate/physiology data.  
This data enables biomechanics-style insights (exertion vs. outcome asymmetry, fatigue, & return-to-play baselines).

**What I would look at + calculate:**
- workload-adjusted archetypes (how style changes under load)
- injury-risk / recovery-sensitive style shifts
- post-injury style drift detection

### 4) Internal roster/contract/cap tools
Teams also have internal access to cap/contract information and transaction tools that the public can't access.

**What I would look at + calculate:**
- link archetype needs to cap-efficient roster construction
- simulate “archetype coverage per dollar” as a roster optimization view
""")

st.markdown("""
**Sources for the “behind the scenes” data streams:**  
- NHL EDGE and tracking system overview: https://www.nhl.com/news/nhl-edge-launches-website-for-puck-and-player-tracking-data  
- Wearable practice tech + examples (Catapult etc): https://www.dailyfaceoff.com/news/nhlpa-reminds-players-of-their-right-to-control-or-destroy-wearable-tech-data  
- Catapult hockey wearables overview: https://www.catapult.com/sports/ice-hockey  
- Example of wearable tracking described publicly: https://www.si.com/edge/2015/02/20/tech-talk-catapult-tracking-nhl-data-injury-reduction  
- Sportlogiq hockey platform: https://www.sportlogiq.com/hockey/  
- NHL cap/contract iPad app (team-side tooling): https://apnews.com/article/49b2f0421df504555bbc940bb861e4a2
""")

st.caption("I try to keep the data as up-to-date as possible but there's always a chance that things may be out of date.")
