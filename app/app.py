import re
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

st.set_page_config(page_title="Quantifying Player Style Archetypes", layout="wide")

import datetime, hashlib
try:
    _app_hash = hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:10]
except Exception:
    _app_hash = "nohash"

DATA_DIR = Path("data/app")
REPORTS_DIR = Path("reports")

def available_seasons() -> list[str]:
    # seasons are inferred from built app parquet files
    app_dir = Path("data/app")
    if not app_dir.exists():
        return []
    seasons = set()
    for f in app_dir.glob("players_forwards_*.parquet"):
        seasons.add(f.stem.replace("players_forwards_", ""))
    # keep only seasons that also have defense files (optional)
    seasons2 = set()
    for f in app_dir.glob("players_defense_*.parquet"):
        seasons2.add(f.stem.replace("players_defense_", ""))
    seasons = sorted(list(seasons & seasons2), reverse=True)
    return seasons

def season_key_to_label(k: str) -> str:
    k = str(k).strip()
    return f"{k[:4]}-{k[4:]}" if (len(k) == 8 and k.isdigit()) else k



# -------------------------
# Helpers
# -------------------------
def safe_int(x):
    try:
        if pd.isna(x):
            return 0
        return int(float(x))
    except Exception:
        return 0

def safe_float(x):
    try:
        if pd.isna(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def round1(x):
    return round(safe_float(x), 1)

def min_to_mmss(minutes):
    m = safe_float(minutes)
    total_s = int(round(m * 60))
    mm = total_s // 60
    ss = total_s % 60
    return f"{mm:02d}:{ss:02d}"

def prob_cols(df):
    return [c for c in df.columns if c.startswith("p") and c[1:].isdigit()]

def nice_axis():
    return alt.Axis(
        labelAngle=0,
        labelFontSize=12,
        titleFontSize=13,
        labelColor="#111827",
        titleColor="#111827",
        gridColor="#E5E7EB",
        domainColor="#9CA3AF",
        tickColor="#9CA3AF",
    )

def col_width(df, col, min_w=90, max_w=240, char_px=8, pad=26, sample_n=400):
    vals = [str(col)] + df[col].astype(str).head(sample_n).tolist()
    max_len = max(len(v) for v in vals)
    return int(min(max(min_w, max_len * char_px + pad), max_w))

# -------------------------
# Badge styles (Archetype + Confidence)
# -------------------------
ARCH_BADGE_JS = """
function(params) {
  const v = params.value || "";
  const map = {
    "A0": ["#CFFAFE", "#155E75"],  // light red / dark red
    "A1": ["#FFEDD5", "#9A3412"],  // light amber / dark amber
    "A2": ["#DBEAFE", "#1D4ED8"],  // light blue / dark blue
    "A3": ["#EDE9FE", "#6D28D9"],  // light purple / dark purple
  };
  const c = map[v] || ["#E5E7EB", "#111827"];
  return {
    backgroundColor: c[0],
    color: c[1],
    border: "1px solid rgba(0,0,0,0.10)",
    borderRadius: "999px",
    padding: "3px 10px",
    fontWeight: "800",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    textAlign: "center"
  };
}
"""

def conf_js_fixed_thresholds():
    # Player Explorer: >90 green, 80-90 yellow, <80 red
    return """
function(params) {
  const s = params.value || "";
  const v = parseFloat(String(s).replace("%",""));
  let bg = "#FEE2E2";
  let fg = "#991B1B";
  if (v > 90) {
    bg = "#DCFCE7";
    fg = "#166534";
  } else if (v >= 80) {
    bg = "#FEF9C3";
    fg = "#854D0E";
  }
  return {
    backgroundColor: bg,
    color: fg,
    border: "1px solid rgba(0,0,0,0.10)",
    borderRadius: "999px",
    padding: "3px 10px",
    fontWeight: "800",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    textAlign: "center"
  };
}
"""

def conf_js_relative(q33: float, q67: float):
    # Other tables: relative thresholds, same look/feel
    return f"""
function(params) {{
  const s = params.value || "";
  const v = parseFloat(String(s).replace("%",""));
  const q33 = {q33:.3f};
  const q67 = {q67:.3f};

  let bg = "#FEE2E2";
  let fg = "#991B1B";
  if (v >= q67) {{
    bg = "#DCFCE7";
    fg = "#166534";
  }} else if (v >= q33) {{
    bg = "#FEF9C3";
    fg = "#854D0E";
  }}
  return {{
    backgroundColor: bg,
    color: fg,
    border: "1px solid rgba(0,0,0,0.10)",
    borderRadius: "999px",
    padding: "3px 10px",
    fontWeight: "800",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    textAlign: "center"
  }};
}}
"""

def similarity_js_fixed_bins():
    # 95-100 -> green2; 90-95 -> green1; 80-90 -> yellow2; 70-80 -> yellow1; 55-70 -> red2; <55 -> red1
    return """
function(params) {
  const v = parseFloat(params.value);
  if (isNaN(v)) return {};

  let bg = "#FEE2E2"; // red1
  let fg = "#991B1B";

  if (v >= 95) {
    bg = "#22C55E";   // green2
    fg = "#FFFFFF";
  } else if (v >= 90) {
    bg = "#DCFCE7";   // green1
    fg = "#166534";
  } else if (v >= 80) {
    bg = "#F59E0B";   // yellow2
    fg = "#111827";
  } else if (v >= 70) {
    bg = "#FEF9C3";   // yellow1
    fg = "#854D0E";
  } else if (v >= 55) {
    bg = "#EF4444";   // red2
    fg = "#FFFFFF";
  } else {
    bg = "#FEE2E2";   // red1
    fg = "#991B1B";
  }

  return {
    backgroundColor: bg,
    color: fg,
    border: "1px solid rgba(0,0,0,0.10)",
    borderRadius: "999px",
    padding: "3px 10px",
    fontWeight: "800",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    textAlign: "center"
  };
}
"""

def make_badge_grid(df, height=560, pin_cols=("Player","Teams"), player_width_offset_px=0, player_min_px=280, player_max_px=700,
                    archetype_col="Archetype", confidence_col="Confidence",
                    conf_mode="fixed", conf_q33=80.0, conf_q67=90.0,
                    extra_styles=None, key_suffix=""):
    extra_styles = extra_styles or {}
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(sortable=True, filter=True, resizable=True, minWidth=80, flex=0, suppressSizeToFit=True)
    gb.configure_grid_options(domLayout="normal", alwaysShowHorizontalScroll=True, alwaysShowVerticalScroll=True)

    # widths (special-case Player to avoid truncation)
    for c in df.columns:
        if c == "Player":
            sample = [str(c)] + df[c].astype(str).head(2000).tolist()
            max_len = max(len(v) for v in sample)
            width = int(min(max(player_min_px, max_len * 8 + 40 + player_width_offset_px), player_max_px))
        else:
            width = col_width(df, c, min_w=85, max_w=260)

        if c == "Teams":
            width = max(width, 160)
        if c in [archetype_col, confidence_col]:
            width = max(width, 130)

        gb.configure_column(c, width=width)

    for c in pin_cols:
        if c in df.columns:
            gb.configure_column(c, pinned="left")

    if archetype_col in df.columns:
        gb.configure_column(archetype_col, cellStyle=JsCode(ARCH_BADGE_JS), width=max(130, col_width(df, archetype_col, 110, 170)))
    if confidence_col in df.columns:
        js = conf_js_fixed_thresholds() if conf_mode == "fixed" else conf_js_relative(conf_q33, conf_q67)
        gb.configure_column(confidence_col, cellStyle=JsCode(js), width=max(150, col_width(df, confidence_col, 120, 200)))

    for col, jscode in extra_styles.items():
        if col in df.columns:
            gb.configure_column(col, cellStyle=JsCode(jscode))

    grid_key = f"grid-{_app_hash}-" + hashlib.md5(
        (str(list(df.columns)) + str(len(df)) + str(player_min_px) + str(player_max_px) + str(height) + str(key_suffix)).encode("utf-8")
    ).hexdigest()[:8]


    AgGrid(
        df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.NO_UPDATE,
        theme="streamlit",
        height=height,
        fit_columns_on_grid_load=False,
        key=grid_key,
        allow_unsafe_jscode=True
    )

# -------------------------
# Legend helpers
# -------------------------
TRAIT_LABELS = {
    "reg_points_per60": "Points / 60",
    "reg_goals_per60": "Goals / 60",
    "reg_assists_per60": "Assists / 60",
    "reg_shots_per60": "Shots / 60",
    "reg_hits_per60": "Hits / 60",
    "reg_blocked_shots_per60": "Blocks / 60",
    "reg_takeaways_per60": "Takeaways / 60",
    "reg_giveaways_per60": "Giveaways / 60",
    "reg_pim_per60": "PIMs / 60",
    "reg_pp_share": "Power-play usage share",
    "reg_pk_share": "Penalty-kill usage share",
    "reg_fo_pct": "Faceoff %",
    "reg_fo_taken_per_game": "Faceoffs / game",
}

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

def format_traits_multiline(tokens, max_items=5):
    lines = []
    for feat, z in tokens[:max_items]:
        label = TRAIT_LABELS.get(feat, feat.replace("reg_", "").replace("_", " ").title())
        arrow = "↑" if z >= 0 else "↓"
        lines.append(f"{arrow} {label} ({z:+.1f}σ)")
    return "\n".join(lines)

def format_examples_multiline(s: str, max_players=7):
    if not isinstance(s, str) or not s.strip():
        return ""
    items = [x.strip() for x in s.split("|") if x.strip()]
    cleaned = []
    for it in items[:max_players]:
        cleaned.append(it.split(" p=")[0].strip())
    return "\n".join(cleaned)

def build_unique_name_summary(cluster: int, high_tokens, low_tokens):
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
        name = "Agitating Heavy-Contact Forward"
        summary = "High-contact profile that plays on the edge: delivers hits and takes more penalties, with lower scoring involvement."
    elif blocks_hi and hits_hi:
        name = "Shot-Blocking Contact Specialist"
        summary = "Defense-tilted role profile: blocks shots and plays physically; offensive creation is typically lower."
    elif offense_hi and playmaking_hi and shooting_hi:
        name = "High-Volume Playmaking Scorer"
        summary = "Offense driver: creates shots and playmaking at high rates, producing strong points/60."
    elif takeaways_hi and giveaways_lo:
        name = "Puck-Pressure Two-Way Creator"
        summary = "Pressure-and-recover style: generates takeaways while keeping giveaways relatively low, and still contributes offensively."
    elif fo_hi and not offense_hi:
        name = "Deployment / Faceoff Specialist"
        summary = "Usage-driven profile: takes many draws and plays role minutes; style reflects deployment more than pure scoring."
    elif pk_hi and not pp_hi:
        name = "PK-Leaning Defensive Role"
        summary = "Shorthanded-usage anchored profile: more value shows up in role/defensive usage than scoring."
    elif pp_hi and not pk_hi:
        name = "PP-Leaning Offensive Role"
        summary = "Power-play usage anchored profile: production is driven by scoring-role deployment."
    else:
        name = f"Mixed Profile Archetype {cluster}"
        summary = "Distinct statistical footprint relative to peers (see the trait deltas for what stands out)."

    if high_tokens and low_tokens:
        hi1 = TRAIT_LABELS.get(high_tokens[0][0], high_tokens[0][0])
        lo1 = TRAIT_LABELS.get(low_tokens[0][0], low_tokens[0][0])
        summary += f" Key signature: higher {hi1}, lower {lo1}."
    return name, summary

PRELINE_CENTER = {
    "whiteSpace": "pre-line",
    "lineHeight": "1.25",
    "display": "flex",
    "alignItems": "center",
}
CENTER = {"display": "flex", "alignItems": "center", "justifyContent": "center"}

def make_legend_grid(df: pd.DataFrame):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(sortable=False, filter=False, resizable=True)
    gb.configure_grid_options(domLayout="autoHeight", suppressSizeToFit=True, alwaysShowHorizontalScroll=True)

    # Summary fixed at 250px (per request)
    gb.configure_column("Archetype", width=150, pinned="left", cellStyle=CENTER)
    gb.configure_column("Name", width=220, pinned="left", cellStyle=PRELINE_CENTER)
    gb.configure_column("Summary", width=420, wrapText=True, autoHeight=True, cellStyle=PRELINE_CENTER)
    gb.configure_column("Traits that tend to be higher", width=360, wrapText=True, autoHeight=True, cellStyle=PRELINE_CENTER)
    gb.configure_column("Traits that tend to be lower", width=360, wrapText=True, autoHeight=True, cellStyle=PRELINE_CENTER)
    gb.configure_column("Example players", width=320, wrapText=True, autoHeight=True, cellStyle=PRELINE_CENTER)

    AgGrid(
        df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.NO_UPDATE,
        theme="streamlit",
        height=340,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=False
    )

def wrap_label(s: str, width: int = 16) -> str:
    words = str(s).split()
    lines, cur = [], []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + (1 if cur else 0) > width:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += len(w) + (1 if cur_len else 0)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


# -------------------------
# Data loading
# -------------------------
@st.cache_data
def load_group(group: str, season: str):
    return pd.read_parquet(DATA_DIR / f"players_{group}_{season}.parquet")

@st.cache_data
def load_traits(group: str, season: str):
    p = REPORTS_DIR / f"archetype_traits_{group}_{season}.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

# -------------------------
# Page
# -------------------------
st.title("Quantifying Player Style Archetypes")


season_keys = available_seasons()  # e.g. ["20242025","20232024",...], newest first
if season_keys:
    season = st.sidebar.selectbox(
        "Season",
        season_keys,
        index=0,
        format_func=season_key_to_label,  # display "YYYY-YYYY"
        key="season_select",
    )
else:
    season = st.sidebar.text_input("Season label", value="20242025", key="season_text")

group = st.sidebar.selectbox("Group", ["forwards", "defense"], key="group_select")

df = load_group(group, season)
traits = load_traits(group, season)

pcols = prob_cols(df)
K = len(pcols)

archetype_name_map = {k: f"A{k}" for k in range(K)}
if traits is not None:
    for _, tr in traits.iterrows():
        kk = int(tr["cluster"])
        ht = parse_trait_string(tr.get("top_traits", ""))
        lt = parse_trait_string(tr.get("low_traits", ""))
        nm, _ = build_unique_name_summary(kk, ht, lt)
        archetype_name_map[kk] = nm

# Relative confidence thresholds for non-Player-Explorer tables
_conf = (df["confidence"].astype(float) * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
conf_q33 = float(_conf.quantile(0.33)) if len(_conf) else 80.0
conf_q67 = float(_conf.quantile(0.67)) if len(_conf) else 90.0

# -------------------------
# Full intro section you referenced (restored)
# -------------------------
with st.expander("What is a 'Player Archetype' and How is it Calculated?", expanded=True):

    st.markdown(
        """
This page answers:

**“If I ignore player names and only look at what each player *does* on the ice, what *styles* of players exist — and which style does each player most resemble?”**
"""
    )

    st.subheader("Data used")
    st.markdown(
        """
I pulled **public game-by-game NHL boxscore and time-on-ice data** from the NHL Gamecenter endpoints (cached locally), then aggregated it into **regular season vs playoff** splits.

Each data point contributes to “style” like this:
- **Scoring/creation:** shots, goals, assists, points → turned into per-60 rates (e.g., Shots/60)
- **Physical/defensive involvement:** hits, blocks → per-60 rates
- **Puck pressure vs risk:** takeaways vs giveaways → per-60 rates
- **Discipline/edge:** penalty minutes → per-60 rate
- **Role/usage:** PP TOI share and PK TOI share (how a coach deploys the player)
- **Deployment signals:** faceoffs per game and faceoff percentage
"""
    )

    st.subheader("Step 1 — Normalize for ice time")
    st.markdown("Counting stats scale with ice time, so I convert them to *per-60* rates.")
    st.latex(r"\text{Shots/60} \;=\; \frac{\text{Shots}}{\text{TOI}_{\text{seconds}}/3600}")

    st.markdown("I also compute special-teams usage share:")
    st.latex(r"\text{PP Share}=\frac{\text{PP TOI}}{\text{Total TOI}} \qquad \text{PK Share}=\frac{\text{PK TOI}}{\text{Total TOI}}")

    st.subheader("Step 2 — Put every feature on the same scale (robust scaling)")
    st.markdown("To keep extreme values from dominating the model, I robust-scale each feature:")
    st.latex(r"x^{*}=\frac{x-\mathrm{median}(x)}{\mathrm{IQR}(x)}")

    st.subheader("Step 3 — Compress the stats into a smaller “style fingerprint” (NMF)")
    st.markdown("I reduce each skill block using Non-negative Matrix Factorization (NMF):")
    st.latex(r"X \approx WH")
    st.markdown("You can think of each row of **W** as a compact *style fingerprint* for that player.")

    st.subheader("Step 4 — Learn archetypes and assign probabilities (GMM)")
    st.markdown("I fit a Gaussian Mixture Model (GMM) to those fingerprints:")
    st.latex(r"p(z)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(z\mid \mu_k,\Sigma_k)")
    st.markdown("For each player \(i\), the model outputs a probability for each archetype:")
    st.latex(r"p_{ik}=P(\text{Archetype}=k \mid z_i)")

    st.markdown(
    """
**Why do “Mixed Profile” archetypes exist?**  
The model is probabilistic: instead of forcing every player into exactly one bucket, it assigns a probability over archetypes.  
Some players genuinely combine traits that sit between multiple clusters (e.g., moderate scoring + moderate physical play), so they appear as “mixed” because their probability mass is spread across archetypes rather than concentrated in one.
"""
    )
    st.markdown(
    f"""
**Interpretation:** if a player’s probabilities are (0.1, 87.3, 6.4, 6.3)% then the player is **mostly Archetype 1** with **87.3% confidence**.
"""
    )

    
st.markdown(
    f"## For the {season_key_to_label(season)} season, the model found **K = {K}** archetypes (A0–A{K-1})."
)
st.markdown(
    "Every season, the model comes up with different archetype definitions. So an A0 archetype in one season might be a Puck Pressure Two-Way Creator but in another season, A0 might be an Agitating Heavy-Contact Forward. The table below explains what each archetype “means” for this season’s model."
)

st.markdown(f"### Archetype definitions — {season_key_to_label(season)}")

if traits is not None:
    legend_rows = []
    for r in traits.itertuples(index=False):
        k = int(r.cluster)
        high_tokens = parse_trait_string(getattr(r, "top_traits", ""))
        low_tokens  = parse_trait_string(getattr(r, "low_traits", ""))
        name, summary = build_unique_name_summary(k, high_tokens, low_tokens)
        legend_rows.append({
            "Archetype": f"A{k}",
            "Name": name,
            "Summary": summary,
            "Traits that tend to be higher": format_traits_multiline(high_tokens, max_items=5),
            "Traits that tend to be lower": format_traits_multiline(low_tokens, max_items=4),
            "Example players": format_examples_multiline(getattr(r, "prototype_players", ""), max_players=7),
        })
    make_legend_grid(pd.DataFrame(legend_rows).sort_values("Archetype"))

tabs = st.tabs(["Player Explorer", "Team Roster Fit", "Need Finder"])

# -------------------------
# Player Explorer
# -------------------------
with tabs[0]:
    st.subheader("Player Explorer")
    st.markdown("""
**What you’re looking at**
- A scrollable table of players in the selected group for the chosen season.
- Regular-season and playoff totals + average time on ice (ATOI).
- Each player’s top archetype (A0–A3) and a confidence score.

**What you can do**
- Use the search box to quickly filter by player name.
- Jump to [**Detailed view**](#detailed-view) and [**Closest comps**](#closest-comps).
""")

    q = st.text_input("Search player name")
    view = df.copy()
    if q.strip():
        view = view[view["full_name"].str.contains(q, case=False, na=False)]

    disp = view.copy()
    disp["Archetype"] = disp["top_cluster"].apply(lambda x: f"A{safe_int(x)}")
    disp["Confidence"] = (disp["confidence"].astype(float) * 100).round(1).astype(str) + "%"
    disp["REG ATOI"] = disp["reg_avg_toi_min"].apply(min_to_mmss)
    disp["PO ATOI"] = disp["po_avg_toi_min"].apply(min_to_mmss)

    main_tbl = pd.DataFrame({
        "Player": disp["full_name"],
        "Teams": disp["teams_played"],
        "Archetype": disp["Archetype"],
        "Confidence": disp["Confidence"],
        "Pos": disp["position"],
        "REG GP": disp["reg_games"],
        "REG ATOI": disp["REG ATOI"],
        "REG P": disp["reg_points"],
        "REG G": disp["reg_goals"],
        "REG A": disp["reg_assists"],
        "REG SOG": disp["reg_shots"],
        "REG +/-": disp["reg_plus_minus"],
        "REG PIM": disp.get("reg_pim", 0),
        "PO GP": disp["po_games"],
        "PO ATOI": disp["PO ATOI"],
        "PO P": disp["po_points"],
        "PO G": disp["po_goals"],
        "PO A": disp["po_assists"],
        "PO SOG": disp["po_shots"],
        "PO +/-": disp["po_plus_minus"],
        "PO PIM": disp.get("po_pim", 0),
    }).sort_values(["REG P","REG GP"], ascending=False)

    # Player Explorer uses FIXED confidence thresholds
    make_badge_grid(main_tbl, height=560, pin_cols=("Player","Teams"), conf_mode="fixed", conf_q33=conf_q33, conf_q67=conf_q67, player_min_px=150, player_max_px=150)

    st.markdown('<div id="detailed-view"></div>', unsafe_allow_html=True)
    st.markdown("### Detailed view")

    names = view["full_name"].dropna().unique().tolist()
    if names:
        sel = st.selectbox("Select a player", names)
        row = view[view["full_name"] == sel].iloc[0]

        # Closest comps
        P = view[pcols].to_numpy(dtype=float)
        v = np.array([safe_float(row[c]) for c in pcols], dtype=float)
        v_norm = np.linalg.norm(v) + 1e-9
        P_norm = np.linalg.norm(P, axis=1) + 1e-9
        sim = (P @ v) / (P_norm * v_norm)

        comps = view.copy()
        comps["Similarity (%)"] = (sim * 100).round(1)
        comps = comps[comps["full_name"] != row["full_name"]].sort_values("Similarity (%)", ascending=False).head(30)

        st.markdown('<div id="closest-comps"></div>', unsafe_allow_html=True)
        st.markdown("### Closest comps (by archetype mix)")

        comps_disp = comps.copy()
        comps_disp["Archetype"] = comps_disp["top_cluster"].apply(lambda x: f"A{safe_int(x)}")
        comps_disp["Confidence"] = (comps_disp["confidence"].astype(float) * 100).round(1).astype(str) + "%"
        comps_disp["REG ATOI"] = comps_disp["reg_avg_toi_min"].apply(min_to_mmss)
        comps_disp["PO ATOI"] = comps_disp["po_avg_toi_min"].apply(min_to_mmss)

        comps_tbl = pd.DataFrame({
            "Player": comps_disp["full_name"],
            "Teams": comps_disp["teams_played"],
            "Archetype": comps_disp["Archetype"],
            "Confidence": comps_disp["Confidence"],
            "Similarity (%)": comps_disp["Similarity (%)"],
            "Pos": comps_disp["position"],
            "REG GP": comps_disp["reg_games"],
            "REG ATOI": comps_disp["REG ATOI"],
            "REG P": comps_disp["reg_points"],
            "REG G": comps_disp["reg_goals"],
            "REG A": comps_disp["reg_assists"],
            "REG SOG": comps_disp["reg_shots"],
            "REG +/-": comps_disp["reg_plus_minus"],
            "REG PIM": comps_disp.get("reg_pim", 0),
            "PO GP": comps_disp["po_games"],
            "PO ATOI": comps_disp["PO ATOI"],
            "PO P": comps_disp["po_points"],
            "PO G": comps_disp["po_goals"],
            "PO A": comps_disp["po_assists"],
            "PO SOG": comps_disp["po_shots"],
            "PO +/-": comps_disp["po_plus_minus"],
            "PO PIM": comps_disp.get("po_pim", 0),
        })

        make_badge_grid(
            comps_tbl,
            height=560,
            pin_cols=("Player","Teams"),
            conf_mode="fixed",
            conf_q33=conf_q33,
            conf_q67=conf_q67,
            player_min_px=120, player_max_px=150,
            extra_styles={"Similarity (%)": similarity_js_fixed_bins()}
        )

# -------------------------
# Team Roster Fit
# -------------------------
with tabs[1]:
    st.subheader("Team Roster Fit")
    st.markdown("""
**What you’re looking at**
- A team-level style breakdown: each bar is the team’s share of each archetype, weighted by regular-season minutes.
- A roster list showing who is driving each archetype on that team.

**What you can learn**
- Which archetypes the team leans on most.
- Which archetypes are underrepresented (potential roster “gaps”).
- Which players contribute to those archetypes (and with what confidence).
""")

    all_teams = sorted({t for s in df["teams_played"].dropna().unique() for t in str(s).split("/")})
    team = st.selectbox("Team", all_teams)
    team_df = df[df["teams_played"].fillna("").str.contains(team)].copy()

    if team_df.empty:
        st.warning("No players found for that team.")
    else:
        # --- Team archetype share bar chart (restored) ---
        w = (team_df["reg_avg_toi_min"].to_numpy(dtype=float) * team_df["reg_games"].to_numpy(dtype=float)) + 1e-9
        shares = []
        for k in range(K):
            shares.append(float(np.average(team_df[pcols[k]].to_numpy(dtype=float), weights=w)))

        comp_df = pd.DataFrame({
            "A": [f"A{k}" for k in range(K)],
            "Name": [wrap_label(archetype_name_map[k], width=16) for k in range(K)],  # archetype_name_map[k] should be just the NAME now
            "Share": shares,
        })
        top_k = int(comp_df["Share"].to_numpy().argmax())
        comp_df["is_top"] = comp_df["A"] == f"A{top_k}"
        comp_df["Share (%)"] = (comp_df["Share"] * 100).round(1)

        bars = (
            alt.Chart(comp_df)
            .mark_bar(size=55, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("A:O", axis=alt.Axis(labelAngle=0, labelFontSize=16, title=None)),
                y=alt.Y("Share:Q", axis=nice_axis(), title="REG-TOI-weighted share", scale=alt.Scale(domain=[0, 1])),
                color=alt.condition(alt.datum.is_top, alt.value("#16A34A"), alt.value("#BBF7D0")),
                tooltip=[
                    alt.Tooltip("A:O", title="Archetype"),
                    alt.Tooltip("Name:N", title="Archetype name"),
                    alt.Tooltip("Share (%):Q", format=".1f", title="Share (%)"),
                ]

            )
            .properties(height=260)
        )

        name_labels = (
            alt.Chart(comp_df)
            .mark_text(
                dy=18,               # pushes text below axis baseline
                fontSize=14,
                color="#111827"
            )
            .encode(
                x=alt.X("A:O"),
                y=alt.value(0),      # anchor at bottom
                text=alt.Text("Name:N"),
                tooltip=["A", "Name"]
            )
        )



        st.altair_chart((bars + name_labels).properties(height=320), use_container_width=True)

        # --- Underrepresented archetypes (roster gaps) table (restored) ---
        st.markdown("### Underrepresented archetypes (Roster Gaps)")
        st.markdown(
            "I flag gaps by comparing this team’s archetype mix to the league distribution, and checking whether the archetype has "
            "strong minutes coverage (players with high membership) or is concentrated in only 1–2 players."
        )

        def _team_metrics(team_abbrev: str):
            tdf = df[df["teams_played"].fillna("").str.contains(team_abbrev)].copy()
            if tdf.empty:
                return None
            wt = (tdf["reg_avg_toi_min"].to_numpy(dtype=float) * tdf["reg_games"].to_numpy(dtype=float)) + 1e-9

            shares_, coverage_, concentration_ = [], [], []
            for kk in range(K):
                pk = tdf[pcols[kk]].to_numpy(dtype=float)
                shares_.append(float(np.average(pk, weights=wt)))

                strong = pk >= 0.60
                cov = float(wt[strong].sum() / wt.sum()) if wt.sum() > 0 else 0.0
                coverage_.append(cov)

                contrib = wt * pk
                total = contrib.sum()
                if total <= 0:
                    concentration_.append(1.0)
                else:
                    top2 = np.sort(contrib)[-2:].sum()
                    concentration_.append(float(top2 / total))

            return shares_, coverage_, concentration_

        rows = []
        for t in all_teams:
            out = _team_metrics(t)
            if out is None:
                continue
            sh, cov, conc = out
            for kk in range(K):
                rows.append({"team": t, "k": kk, "share": sh[kk], "coverage": cov[kk], "concentration": conc[kk]})

        ts = pd.DataFrame(rows)
        if not ts.empty:
            base = ts.groupby("k", as_index=False).agg(mean_share=("share", "mean"), std_share=("share", "std"))
            ts["cov_rank"] = ts.groupby("k")["coverage"].rank(pct=True)
            ts["conc_rank"] = ts.groupby("k")["concentration"].rank(pct=True)

            me = ts[ts["team"] == team].merge(base, on="k", how="left")
            me["z"] = (me["share"] - me["mean_share"]) / me["std_share"].replace({0: np.nan})
            me["z"] = me["z"].fillna(0.0)

            me["risk"] = (-me["z"]) + np.maximum(0, 0.35 - me["cov_rank"]) * 2.0 + np.maximum(0, me["conc_rank"] - 0.75) * 1.5
            me = me.sort_values("risk", ascending=False)

            me["Archetype"] = me["k"].apply(lambda x: f"A{int(x)}")
            me["Archetype name"] = me["k"].apply(lambda x: archetype_name_map[int(x)])
            me["Team share (%)"] = (me["share"] * 100).round(1)
            me["League avg (%)"] = (me["mean_share"] * 100).round(1)
            me["Strong coverage (%)"] = (me["coverage"] * 100).round(1)
            me["Reliance on top 2 (%)"] = (me["concentration"] * 100).round(1)
            me["Z-score"] = me["z"].round(2)

            me["Note"] = ""
            me.loc[(me["z"] < -0.75) | ((me["z"] < -0.5) & (me["cov_rank"] < 0.35)), "Note"] = "Underrepresented"
            me.loc[(me["Note"] == "") & (me["conc_rank"] > 0.75) & (me["cov_rank"] < 0.5), "Note"] = "Thin coverage"

            show = me[[
                "Archetype","Archetype name","Team share (%)","League avg (%)","Z-score","Strong coverage (%)","Reliance on top 2 (%)","Note"
            ]].reset_index(drop=True)
            st.dataframe(show.reset_index(drop=True), use_container_width=True, hide_index=True)


        # --- Roster list (restored; includes reg/po stats, order like Player Explorer) ---
        st.markdown("### Roster list")
        r = team_df.copy()
        r["Archetype"] = r["top_cluster"].apply(lambda x: f"A{safe_int(x)}")
        r["Confidence"] = (r["confidence"].astype(float) * 100).round(1).astype(str) + "%"
        r["REG ATOI"] = r["reg_avg_toi_min"].apply(min_to_mmss)
        r["PO ATOI"] = r["po_avg_toi_min"].apply(min_to_mmss)

        roster_tbl = pd.DataFrame({
            "Player": r["full_name"],
            "Teams": r["teams_played"],
            "Archetype": r["Archetype"],
            "Confidence": r["Confidence"],
            "Pos": r["position"],
            "REG GP": r["reg_games"],
            "REG ATOI": r["REG ATOI"],
            "REG P": r["reg_points"],
            "REG G": r["reg_goals"],
            "REG A": r["reg_assists"],
            "REG SOG": r["reg_shots"],
            "REG +/-": r["reg_plus_minus"],
            "REG PIM": r.get("reg_pim", 0),
            "PO GP": r["po_games"],
            "PO ATOI": r["PO ATOI"],
            "PO P": r["po_points"],
            "PO G": r["po_goals"],
            "PO A": r["po_assists"],
            "PO SOG": r["po_shots"],
            "PO +/-": r["po_plus_minus"],
            "PO PIM": r.get("po_pim", 0),
        }).sort_values(["REG P","REG GP"], ascending=False)

        make_badge_grid(roster_tbl, height=600, pin_cols=("Player","Teams"), conf_mode="fixed", conf_q33=conf_q33, conf_q67=conf_q67, player_min_px=120, player_max_px=150)

# -------------------------
# Need Finder
# -------------------------
with tabs[2]:
    st.subheader("Need Finder (find players who match a target archetype)")
    st.markdown("""
**What you’re looking at**
- A ranked list of players who best match a selected archetype (A0–A3).

**How to use it**
- Pick the archetype you want to add to a roster.
- Optionally exclude your own team.
- Increase minimum regular-season games to avoid tiny samples.
- “Target similarity (%)” is the model’s estimated probability that the player belongs to that archetype.
""")

    all_teams = sorted({t for s in df["teams_played"].dropna().unique() for t in str(s).split("/")})
    exclude_team = st.selectbox("Exclude team (optional)", ["(none)"] + all_teams)
    target_options = [f"A{k} - {archetype_name_map.get(k, 'Unknown')}" for k in range(K)]
    target_choice = st.selectbox("Target archetype", target_options, key=f"target_archetype_{season}_{group}")

    # convert the selected label back to the integer k
    target = int(target_choice.split(" - ")[0].replace("A", ""))


    min_reg_games = st.slider("Min REG games", 0, 82, 20, step=5)

    view = df.copy()
    if exclude_team != "(none)":
        view = view[~view["teams_played"].fillna("").str.contains(exclude_team)]
    view = view[view.get("reg_games",0) >= min_reg_games].copy()

    view["Target similarity (%)"] = (view[f"p{target}"] * 100).round(1)
    out = view.sort_values(["Target similarity (%)","reg_points"], ascending=False).head(80).copy()

    o = out.copy()
    o["Archetype"] = o["top_cluster"].apply(lambda x: f"A{safe_int(x)}")
    o["Confidence"] = (o["confidence"].astype(float) * 100).round(1).astype(str) + "%"
    o["REG ATOI"] = o["reg_avg_toi_min"].apply(min_to_mmss)
    o["PO ATOI"] = o["po_avg_toi_min"].apply(min_to_mmss)

    need_tbl = pd.DataFrame({
        "Player": o["full_name"],
        "Teams": o["teams_played"],
        "Archetype": o["Archetype"],
        "Confidence": o["Confidence"],
        "Target similarity (%)": o["Target similarity (%)"],
        "Pos": o["position"],
        "REG GP": o["reg_games"],
        "REG ATOI": o["REG ATOI"],
        "REG P": o["reg_points"],
        "REG G": o["reg_goals"],
        "REG A": o["reg_assists"],
        "REG SOG": o["reg_shots"],
        "REG +/-": o["reg_plus_minus"],
        "REG PIM": o.get("reg_pim", 0),
        "PO GP": o["po_games"],
        "PO ATOI": o["PO ATOI"],
        "PO P": o["po_points"],
        "PO G": o["po_goals"],
        "PO A": o["po_assists"],
        "PO SOG": o["po_shots"],
        "PO +/-": o["po_plus_minus"],
        "PO PIM": o.get("po_pim", 0),
    })

    make_badge_grid(
        need_tbl,
        height=650,
        pin_cols=("Player","Teams"),
        conf_mode="fixed",
        conf_q33=conf_q33,
        conf_q67=conf_q67,
        player_min_px=190, player_max_px=500,
        extra_styles={"Target similarity (%)": similarity_js_fixed_bins()},
        key_suffix=f"needfinder_target_{target}"
    )
