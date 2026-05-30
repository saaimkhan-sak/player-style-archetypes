import sys
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
import datetime
import json
import html
import subprocess


from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

st.set_page_config(page_title="What Are the Season Level Trends in Play Style?", layout="wide")

st.markdown(
    """
<style>
.ag-tooltip {
  white-space: pre-line !important;
  max-width: 420px !important;
  line-height: 1.35 !important;
  padding: 12px 14px !important;
  border-radius: 8px !important;
}
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
</style>
""",
    unsafe_allow_html=True,
)

import datetime, hashlib
try:
    _app_hash = hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:10]
except Exception:
    _app_hash = "nohash"

DATA_DIR = Path("data/app")
REPORTS_DIR = Path("reports")
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.archetype_labels import build_archetype_name_summary, parse_trait_string
from app.lib import readable_trait_label
try:
    from src.archetype_labels import PROFILE_COLOR_MAP
except ImportError:
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

ARCHETYPE_LABEL_CACHE_KEY = "profile-colors-v2"

def normalize_profile_name(name: str) -> str:
    replacements = {
        "Low-Contact Scoring Profile": "Perimeter Skill Scorer",
        "Shooting / Scoring Profile": "Shot-Creation Scorer",
        "Shot-Creating Playmaker": "High-Volume Playmaking Scorer",
        "Setup Playmaker": "High-Volume Playmaking Scorer",
        "Low-Contact Scorer": "Perimeter Skill Scorer",
        "Shot-Volume Scorer": "Shot-Creation Scorer",
        "Volume Shooter": "Shot-Creation Scorer",
        "Finisher": "Perimeter Skill Scorer",
        "Defense-First Shot Suppressor": "Shot-Blocking Contact Specialist",
        "Shot Suppressor": "Shot-Blocking Contact Specialist",
        "Puck Hunter": "Puck-Pressure Two-Way Creator",
        "Puck-Pressure Scorer": "Puck-Pressure Two-Way Creator",
        "Physical Disruptor": "Checking-Line Disruptor",
        "Checking Forward": "Checking-Line Disruptor",
        "Penalty-Drawn Edge Player": "Agitating Heavy-Contact Forward",
        "Power-Play Specialist": "PP-Leaning Offensive Role",
        "Penalty-Kill Specialist": "PK-Leaning Defensive Role",
        "Role-Center Specialist": "Deployment / Role Specialist",
    }
    return replacements.get(str(name), str(name))

def group_archetype_name_summary(cluster: int, high_tokens: list, low_tokens: list, group: str) -> tuple[str, str]:
    try:
        return build_archetype_name_summary(cluster, high_tokens, low_tokens, group=group)
    except TypeError:
        name, summary = build_archetype_name_summary(cluster, high_tokens, low_tokens)
        if group != "defense":
            return normalize_profile_name(name), summary
        defense_names = {
            "Agitating Heavy-Contact Forward": ("Physical Shutdown Defenseman", "Defense profile built around contact, crease-area resistance, and a higher-penalty edge."),
            "High-Volume Playmaking Scorer": ("Offensive Puck-Moving Defenseman", "Blue-line offense driver: creates through point shots, exits, and puck movement."),
            "Perimeter Skill Scorer": ("Low-Event Puck-Moving Defenseman", "Puck-moving defense profile with offense showing up without a heavy-contact footprint."),
            "Shot-Blocking Contact Specialist": ("Shot-Blocking Defensive Defenseman", "Defense-first profile: blocks shots, plays through contact, and absorbs hard minutes."),
            "Deployment / Role Specialist": ("Defensive Role Defenseman", "Role-driven defense profile whose statistical lean is moderate rather than extreme."),
            "Puck-Pressure Two-Way Creator": ("Puck-Pressure Transition Defenseman", "Transition defender: pressures puck carriers and turns recoveries into clean exits."),
            "High-Touch Risk/Reward Playmaker": ("Transition Risk/Reward Defenseman", "High-touch defense profile: moves the puck often, with turnover risk attached."),
        }
        return defense_names.get(normalize_profile_name(name), (normalize_profile_name(name), summary))



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
    seasons = [
        s for s in (seasons & seasons2)
        if s[:4].isdigit() and int(s[:4]) >= 2008
    ]
    seasons = sorted(seasons, reverse=True)
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
  const map = __PROFILE_COLOR_MAP__;
    const c = map[v] || ["#E5E7EB", "#111827"];
  return {
    backgroundColor: c[0],
    color: c[1],
    border: "1px solid rgba(0,0,0,0.08)",
    borderRadius: "999px",
    padding: "3px 10px",
    fontWeight: "700",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    textAlign: "center"
  };
}
""".replace("__PROFILE_COLOR_MAP__", json.dumps(PROFILE_COLOR_MAP))

ARCHETYPE_COLOR_DOMAIN = list(PROFILE_COLOR_MAP.keys())
ARCHETYPE_COLOR_RANGE = [PROFILE_COLOR_MAP[name][0] for name in ARCHETYPE_COLOR_DOMAIN]

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
                    extra_styles=None, key_suffix="", archetype_tooltip_col="Archetype details"):
    extra_styles = extra_styles or {}
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(sortable=True, filter=True, resizable=True, minWidth=80, flex=0, suppressSizeToFit=True)
    gb.configure_grid_options(
        domLayout="normal",
        alwaysShowHorizontalScroll=True,
        alwaysShowVerticalScroll=True,
        tooltipShowDelay=150,
        tooltipHideDelay=12000,
        enableBrowserTooltips=True,
    )

    # widths (special-case Player to avoid truncation)
    for c in df.columns:
        if c == "Player":
            sample = [str(c)] + df[c].astype(str).head(2000).tolist()
            max_len = max(len(v) for v in sample)
            width = int(min(max(player_min_px, max_len * 8 + 40 + player_width_offset_px), player_max_px))
        else:
            width = col_width(df, c, min_w=85, max_w=260)

        if c == archetype_tooltip_col:
            gb.configure_column(c, hide=True)
            continue

        if c == "Teams":
            width = max(width, 160)
        if c in [archetype_col, confidence_col]:
            width = max(width, 130)

        gb.configure_column(c, width=width)

    for c in pin_cols:
        if c in df.columns:
            gb.configure_column(c, pinned="left")

    if archetype_col in df.columns:
        archetype_opts = {
            "cellStyle": JsCode(ARCH_BADGE_JS),
            "width": max(340, col_width(df, archetype_col, 260, 430)),
            "minWidth": 320,
            "wrapText": True,
            "autoHeight": True,
        }
        if archetype_tooltip_col in df.columns:
            archetype_opts["tooltipValueGetter"] = JsCode(
                f"""
function(params) {{
  const row = params.data || {{}};
  return row["{archetype_tooltip_col}"] || params.value || "";
}}
"""
            )
        gb.configure_column(archetype_col, **archetype_opts)
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
def format_traits_multiline(tokens, max_items=5):
    lines = []
    for feat, z in tokens[:max_items]:
        label = readable_trait_label(feat)
        arrow = "↑" if z >= 0 else "↓"
        lines.append(f"{arrow} {label} ({z:+.1f}σ)")
    return "\n".join(lines)

def format_traits_inline(tokens, max_items=5):
    pieces = []
    for feat, z in tokens[:max_items]:
        label = readable_trait_label(feat)
        arrow = "higher" if z >= 0 else "lower"
        pieces.append(f"{label} ({arrow}, {z:+.1f}σ)")
    return "; ".join(pieces)

def format_examples_multiline(s: str, max_players=7):
    if not isinstance(s, str) or not s.strip():
        return ""
    items = [x.strip() for x in s.split("|") if x.strip()]
    cleaned = []
    for it in items[:max_players]:
        cleaned.append(it.split(" p=")[0].strip())
    return "\n".join(cleaned)

def build_archetype_detail_map(traits_df: pd.DataFrame | None) -> dict[int, str]:
    if traits_df is None or traits_df.empty:
        return {}

    details: dict[int, str] = {}
    for r in traits_df.itertuples(index=False):
        k = int(r.cluster)
        high_tokens = parse_trait_string(getattr(r, "top_traits", ""))
        low_tokens = parse_trait_string(getattr(r, "low_traits", ""))
        name, summary = group_archetype_name_summary(k, high_tokens, low_tokens, group)
        name = normalize_profile_name(name)
        higher = format_traits_inline(high_tokens, max_items=5) or "None"
        lower = format_traits_inline(low_tokens, max_items=4) or "None"
        details[k] = (
            f"{name}\n"
            f"{summary}\n"
            f"Higher traits: {higher}\n"
            f"Lower traits: {lower}"
        )
    return details

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

    gb.configure_column("Archetype", width=380, minWidth=340, pinned="left", wrapText=True, autoHeight=True, cellStyle=JsCode(ARCH_BADGE_JS))
    gb.configure_column("Summary", width=252, wrapText=True, autoHeight=True, cellStyle=PRELINE_CENTER)
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
        allow_unsafe_jscode=True
    )

def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = str(color).strip().lstrip("#")
    if len(color) != 6:
        return (229, 231, 235)
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

def _rgba(color: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(color)
    return f"rgba({r}, {g}, {b}, {alpha})"

def render_snapshot_metric(label: str, value: str, detail: str, color: str = "#E5E7EB") -> str:
    return f"""
    <div style="
        border: 1px solid rgba(15,23,42,0.08);
        border-left: 5px solid {color};
        border-radius: 8px;
        padding: 13px 14px;
        background: linear-gradient(180deg, {_rgba(color, 0.16)}, rgba(255,255,255,0.94));
        min-height: 104px;
    ">
        <div style="font-size: 0.8rem; color: #64748B; font-weight: 750; text-transform: uppercase;">{html.escape(label)}</div>
        <div style="font-size: 1.65rem; color: #111827; font-weight: 850; line-height: 1.05; margin-top: 7px;">{html.escape(value)}</div>
        <div style="font-size: 0.88rem; color: #475569; margin-top: 6px; line-height: 1.25;">{html.escape(detail)}</div>
    </div>
    """

def percentage_circle_chart(summary_df: pd.DataFrame, max_items: int = 8) -> alt.Chart:
    chart_data = summary_df.head(max_items).copy()
    charts = []
    for _, row in chart_data.iterrows():
        share = float(row["Share"])
        data = pd.DataFrame(
            {
                "Segment": ["Share", "Remainder"],
                "Value": [share, max(0.0, 100.0 - share)],
                "Color": [row["Color"], "#E5E7EB"],
                "Archetype": [row["Archetype"], row["Archetype"]],
                "Players": [row["Players"], row["Players"]],
                "AvgConfidence": [row["AvgConfidence"], row["AvgConfidence"]],
            }
        )
        arc = alt.Chart(data).mark_arc(innerRadius=34, outerRadius=50, stroke="#FFFFFF", strokeWidth=1).encode(
            theta=alt.Theta("Value:Q", stack=True),
            color=alt.Color("Color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("Archetype:N", title="Archetype"),
                alt.Tooltip("Players:Q", title="Players"),
                alt.Tooltip("AvgConfidence:Q", title="Avg confidence", format=".1f"),
            ],
        )
        label = alt.Chart(pd.DataFrame({"ShareLabel": [f"{share:.0f}%"]})).mark_text(
            fontSize=18, fontWeight="bold", color="#111827"
        ).encode(text="ShareLabel:N")
        title = alt.TitleParams(wrap_label(row["Archetype"], width=20), fontSize=11, dy=8, anchor="middle")
        charts.append((arc + label).properties(width=128, height=128, title=title))
    rows = [alt.hconcat(*charts[i:i + 3], spacing=8) for i in range(0, len(charts), 3)]
    return alt.vconcat(*rows, spacing=4).configure_view(stroke=None)

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

def last_name(name: str) -> str:
    parts = str(name).replace(".", "").replace("'", "").split()
    return parts[-1].lower() if parts else ""

def xg_chip(value: float) -> tuple[str, str, str]:
    if pd.isna(value):
        return "n/a", "#E5E7EB", "#374151"
    pct = float(value) * 100
    if pct >= 55:
        return f"{pct:.0f}% xG", "#DCFCE7", "#166534"
    if pct >= 48:
        return f"{pct:.0f}% xG", "#FEF9C3", "#854D0E"
    return f"{pct:.0f}% xG", "#FEE2E2", "#991B1B"

def pct_cell(value: float) -> str:
    value = float(value)
    if value >= 45:
        bg, fg = "#DCFCE7", "#166534"
    elif value >= 25:
        bg, fg = "#FEF9C3", "#854D0E"
    else:
        bg, fg = "#FEE2E2", "#991B1B"
    return f'<td style="background:{bg};color:{fg};font-weight:800;text-align:right;">{value:.1f}</td>'


# -------------------------
# Data loading
# -------------------------
@st.cache_data(ttl=3600)
def load_parquet(path: str, mtime: float):
    return pd.read_parquet(path)

@st.cache_data(ttl=3600)
def read_parquet_fresh(path: str):
    mtime = Path(path).stat().st_mtime
    return load_parquet(path, mtime)

@st.cache_data(ttl=3600)
def load_group(group: str, season: str):
    return pd.read_parquet(DATA_DIR / f"players_{group}_{season}.parquet")

@st.cache_data(ttl=3600)
def load_line_combinations() -> pd.DataFrame:
    p = DATA_DIR / "line_combinations.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)

@st.cache_data(ttl=3600)
def load_traits(group: str, season: str):
    p = REPORTS_DIR / f"archetype_traits_{group}_{season}.csv"
    if p.exists():
        return pd.read_csv(p)
    return None

@st.cache_data(ttl=3600)
def load_all_seasons_group(group: str) -> pd.DataFrame:
    """Load and concatenate app parquets for a group across all built seasons."""
    frames = []
    for sk in available_seasons():  # keys like "20242025"
        path = DATA_DIR / f"players_{group}_{sk}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["season"] = sk
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out

@st.cache_data(ttl=3600)
def load_archetype_name_map_for_season(
    group: str,
    season_key: str,
    label_version: str = ARCHETYPE_LABEL_CACHE_KEY,
) -> dict[int, str]:
    """
    Returns {cluster_id -> archetype_name} for a given season & group,
    derived from that season's traits CSV (so names are season-specific).
    """
    p = REPORTS_DIR / f"archetype_traits_{group}_{season_key}.csv"
    if not p.exists():
        return {}
    traits_df = pd.read_csv(p)
    m = {}
    for _, tr in traits_df.iterrows():
        kk = int(tr["cluster"])
        ht = parse_trait_string(tr.get("top_traits", ""))
        lt = parse_trait_string(tr.get("low_traits", ""))
        nm, _ = group_archetype_name_summary(kk, ht, lt, group)
        m[kk] = nm
    return m

# -------------------------
# Page
# -------------------------
st.title("What Are the Season Level Trends in Play Style?")


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

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ET_TZ = ZoneInfo("America/New_York")

def get_last_updated_ts(season_key: str) -> datetime | None:
    repo_root = Path(__file__).resolve().parents[2]  # app/pages -> app -> repo root
    candidates = [
        repo_root / f"data/app/players_forwards_{season_key}.parquet",
        repo_root / f"data/app/players_defense_{season_key}.parquet",
        repo_root / f"reports/archetype_traits_forwards_{season_key}.csv",
        repo_root / f"reports/archetype_traits_defense_{season_key}.csv",
        repo_root / f"data/processed/schedule_{season_key}.parquet",
    ]
    rel_paths = [str(p.relative_to(repo_root)) for p in candidates if p.exists()]
    if rel_paths:
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%cI", "--", *rel_paths],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            stamp = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
            if stamp:
                return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except Exception:
            pass
    mtimes = [p.stat().st_mtime for p in candidates if p.exists()]
    return datetime.fromtimestamp(max(mtimes), tz=timezone.utc) if mtimes else None

def fmt_updated_et(ts: datetime | None) -> str:
    if ts is None:
        return "unknown"
    return ts.astimezone(ET_TZ).strftime("%Y-%m-%d %H:%M %Z")

season_key = str(season).replace("-", "")  # works for "20252026" and "2025-2026"
last_updated_ts = get_last_updated_ts(season_key)
st.sidebar.caption(f"Data refreshed: {fmt_updated_et(last_updated_ts)}")


df = load_group(group, season)
traits = load_traits(group, season)

pcols = prob_cols(df)
K = len(pcols)

archetype_name_map = {k: f"Archetype {k}" for k in range(K)}
if traits is not None:
    for _, tr in traits.iterrows():
        kk = int(tr["cluster"])
        ht = parse_trait_string(tr.get("top_traits", ""))
        lt = parse_trait_string(tr.get("low_traits", ""))
        nm, _ = group_archetype_name_summary(kk, ht, lt, group)
        archetype_name_map[kk] = normalize_profile_name(nm)
archetype_detail_map = build_archetype_detail_map(traits)

# Relative confidence thresholds for non-Player-Explorer tables
_conf = (df["confidence"].astype(float) * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
conf_q33 = float(_conf.quantile(0.33)) if len(_conf) else 80.0
conf_q67 = float(_conf.quantile(0.67)) if len(_conf) else 90.0

# -------------------------
# Full intro section you referenced (restored)
# -------------------------
with st.expander("A Quick Review of How Player Archetype is Calculated", expanded=False):

    st.subheader("Data used")
    st.markdown(
        """
I pulled **public game-by-game NHL boxscore and time-on-ice data** from the NHL Gamecenter endpoints, then aggregated it into **regular season vs playoff** splits.

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

    st.subheader("Step 2 — Put every feature on the same scale")
    st.markdown("To keep extreme values from dominating the model, I robust-scale each feature:")
    st.latex(r"x^{*}=\frac{x-\mathrm{median}(x)}{\mathrm{IQR}(x)}")

    st.subheader("Step 3 — Compress the stats into a smaller “style fingerprint”")
    st.markdown("I reduce each skill block using Non-negative Matrix Factorization (NMF):")
    st.latex(r"X \approx WH")
    st.markdown("You can think of each row of **W** as a compact *style fingerprint* for that player.")

    st.subheader("Step 4 — Learn archetypes and assign probabilities")
    st.markdown("I fit a Gaussian Mixture Model (GMM) to those fingerprints:")
    st.latex(r"p(z)=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(z\mid \mu_k,\Sigma_k)")
    st.markdown(r"For each player \(i\), the model outputs a probability for each archetype:")
    st.latex(r"p_{ik}=P(\text{Archetype}=k \mid z_i)")

    st.markdown(
    """
**Why do blended style profiles exist?**  
The model is probabilistic: instead of forcing every player into exactly one bucket, it assigns a probability over archetypes.  
Some players genuinely combine traits that sit between multiple clusters (e.g., moderate scoring + moderate physical play), so their profile names describe the strongest trait combination rather than pretending every cluster is one clean role.
"""
    )
    st.markdown(
    f"""
**Interpretation:** if a player’s probabilities are (0.1, 87.3, 6.4, 6.3)% then the player is mostly aligned to one profile with **87.3% confidence**.
"""
    )

    
st.markdown(
    f"## For the {season_key_to_label(season)} season, the model found **{K}** style profiles."
)
st.markdown(
    "Profile colors are consistent across seasons, so the same archetype title keeps the same visual identity wherever it appears."
)

st.markdown(f"### Archetype definitions — {season_key_to_label(season)}")

if traits is not None:
    legend_rows = []
    for r in traits.itertuples(index=False):
        k = int(r.cluster)
        high_tokens = parse_trait_string(getattr(r, "top_traits", ""))
        low_tokens  = parse_trait_string(getattr(r, "low_traits", ""))
        name, summary = group_archetype_name_summary(k, high_tokens, low_tokens, group)
        name = normalize_profile_name(name)
        legend_rows.append({
            "Archetype": name,
            "Summary": summary,
            "Traits that tend to be higher": format_traits_multiline(high_tokens, max_items=5),
            "Traits that tend to be lower": format_traits_multiline(low_tokens, max_items=4),
            "Example players": format_examples_multiline(getattr(r, "prototype_players", ""), max_players=7),
        })
    legend_df = pd.DataFrame(legend_rows)
    if not legend_df.empty:
        legend_df = (
            legend_df.groupby("Archetype", as_index=False)
            .agg({
                "Summary": "first",
                "Traits that tend to be higher": "first",
                "Traits that tend to be lower": "first",
                "Example players": "first",
            })
            .sort_values("Archetype")
        )
    make_legend_grid(legend_df)

tabs = st.tabs(["Archetype Snapshot", "Player Explorer", "Team Roster Construction", "Need Finder"])

# -------------------------
# Archetype Snapshot
# -------------------------
with tabs[0]:
    st.subheader("Archetype Snapshot")

    snap = df.copy()
    snap["Archetype"] = snap["top_cluster"].apply(lambda x: archetype_name_map.get(safe_int(x), f"Archetype {safe_int(x)}"))
    snap["ConfidencePct"] = pd.to_numeric(snap["confidence"], errors="coerce").fillna(0.0) * 100.0
    snap["REG TOI"] = (
        pd.to_numeric(snap.get("reg_avg_toi_min", 0), errors="coerce").fillna(0.0)
        * pd.to_numeric(snap.get("reg_games", 0), errors="coerce").fillna(0.0)
    )

    total_players = max(len(snap), 1)
    total_toi = float(snap["REG TOI"].sum())
    mix = (
        snap.groupby("Archetype", as_index=False)
        .agg(
            Players=("player_id", "nunique"),
            AvgConfidence=("ConfidencePct", "mean"),
            MedianConfidence=("ConfidencePct", "median"),
            TotalTOI=("REG TOI", "sum"),
            AvgPoints=("reg_points", "mean"),
            AvgGames=("reg_games", "mean"),
        )
    )
    mix["Share"] = mix["Players"] / total_players * 100.0
    mix["TOI Share"] = np.where(total_toi > 0, mix["TotalTOI"] / total_toi * 100.0, 0.0)
    mix["Color"] = mix["Archetype"].apply(lambda x: PROFILE_COLOR_MAP.get(x, ("#E5E7EB", "#111827"))[0])
    mix["TextColor"] = mix["Archetype"].apply(lambda x: PROFILE_COLOR_MAP.get(x, ("#E5E7EB", "#111827"))[1])
    mix = mix.sort_values(["Share", "AvgConfidence"], ascending=[False, False]).reset_index(drop=True)

    dominant = mix.iloc[0] if not mix.empty else None
    top3_share = float(mix.head(3)["Share"].sum()) if not mix.empty else 0.0
    entropy = 0.0
    if not mix.empty:
        shares = (mix["Share"] / 100.0).replace(0, np.nan).dropna()
        if len(shares) > 1:
            entropy = float(-(shares * np.log(shares)).sum() / np.log(len(shares)))
    balance_label = "Balanced spread"
    if top3_share >= 75:
        balance_label = "Highly concentrated"
    elif top3_share >= 58:
        balance_label = "Moderately concentrated"

    avg_conf_pct = float(snap["ConfidencePct"].mean()) if len(snap) else 0.0
    mixed_count = int((snap["ConfidencePct"] < 80).sum())
    dominant_color = str(dominant["Color"]) if dominant is not None else "#E5E7EB"
    dominant_name = str(dominant["Archetype"]) if dominant is not None else "None"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(render_snapshot_metric("Dominant archetype", dominant_name, f"{dominant['Share']:.1f}% of {group}" if dominant is not None else "No player data", dominant_color), unsafe_allow_html=True)
    with m2:
        st.markdown(render_snapshot_metric("Top-three concentration", f"{top3_share:.1f}%", balance_label, "#BAE6FD"), unsafe_allow_html=True)
    with m3:
        st.markdown(render_snapshot_metric("Average confidence", f"{avg_conf_pct:.1f}%", "How cleanly players fit their top profile", "#BBF7D0"), unsafe_allow_html=True)
    with m4:
        st.markdown(render_snapshot_metric("Mixed profiles", f"{mixed_count}", "Players below 80% top-profile confidence", "#FDE68A"), unsafe_allow_html=True)

    st.markdown("### Top archetype shares")
    st.altair_chart(percentage_circle_chart(mix, max_items=8), use_container_width=False)

    chart_data = mix.copy()
    chart_data["ArchetypeLabel"] = chart_data["Archetype"].apply(lambda x: wrap_label(x, width=22))

    share_chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, opacity=0.82)
        .encode(
            x=alt.X("Share:Q", title="Players (%)"),
            y=alt.Y("ArchetypeLabel:N", title=None, sort="-x"),
            color=alt.Color("Archetype:N", scale=alt.Scale(domain=ARCHETYPE_COLOR_DOMAIN, range=ARCHETYPE_COLOR_RANGE), legend=None),
            tooltip=[
                alt.Tooltip("Archetype:N", title="Archetype"),
                alt.Tooltip("Players:Q", title="Players"),
                alt.Tooltip("Share:Q", title="Player share", format=".1f"),
                alt.Tooltip("TOI Share:Q", title="TOI share", format=".1f"),
                alt.Tooltip("AvgConfidence:Q", title="Avg confidence", format=".1f"),
            ],
        )
        .properties(height=max(280, 34 * len(chart_data)))
    )
    st.altair_chart(share_chart, use_container_width=True)

    c_left, c_right = st.columns([1.05, 0.95])
    with c_left:
        st.markdown("### High-level trends")
        trend_rows = []
        if dominant is not None:
            trend_rows.append(f"**{dominant_name}** is the largest profile in this selected season/group.")
        if len(mix) >= 2:
            runner = mix.iloc[1]
            gap = float(mix.iloc[0]["Share"] - runner["Share"])
            trend_rows.append(f"The gap between the top two archetypes is **{gap:.1f} percentage points**, so the group is **{balance_label.lower()}**.")
        trend_rows.append(f"The top three archetypes cover **{top3_share:.1f}%** of players.")
        trend_rows.append(f"The distribution balance score is **{entropy:.2f}** on a 0-1 scale, where 1 means a more even archetype spread.")
        trend_rows.append(f"Average top-archetype confidence is **{avg_conf_pct:.1f}%**, with **{mixed_count}** mixed-profile players under 80%.")
        st.markdown("\n".join(f"- {row}" for row in trend_rows))

    with c_right:
        st.markdown("### Archetype table")
        table_df = mix[["Archetype", "Players", "Share", "TOI Share", "AvgConfidence", "AvgPoints", "AvgGames"]].copy()
        table_df = table_df.rename(columns={
            "Share": "Player %",
            "TOI Share": "TOI %",
            "AvgConfidence": "Avg confidence %",
            "AvgPoints": "Avg points",
            "AvgGames": "Avg games",
        })
        for c in ["Player %", "TOI %", "Avg confidence %", "Avg points", "Avg games"]:
            table_df[c] = table_df[c].round(1)
        st.dataframe(table_df, use_container_width=True, hide_index=True)

# -------------------------
# Player Explorer
# -------------------------
with tabs[1]:
    st.subheader("Player Explorer")
    st.markdown("""
**What you’re looking at**
- A scrollable table of players in the selected group for the chosen season.
- Regular-season and playoff totals + average time on ice (ATOI).
- Each player’s top style profile and a confidence score.

**What you can do**
- Use the search box to quickly filter by player name.
- Jump to [**Detailed view**](#detailed-view) and [**Closest comps**](#closest-comps).
""")

    q = st.text_input("Search player name")
    view = df.copy()
    if q.strip():
        view = view[view["full_name"].str.contains(q, case=False, na=False)]

    disp = view.copy()
    disp["Archetype"] = disp["top_cluster"].apply(lambda x: archetype_name_map.get(safe_int(x), f"Archetype {safe_int(x)}"))
    disp["Archetype details"] = disp["top_cluster"].apply(
        lambda x: archetype_detail_map.get(safe_int(x), archetype_name_map.get(safe_int(x), f"Archetype {safe_int(x)}"))
    )
    disp["Confidence"] = (disp["confidence"].astype(float) * 100).round(1).astype(str) + "%"
    disp["REG ATOI"] = disp["reg_avg_toi_min"].apply(min_to_mmss)
    disp["PO ATOI"] = disp["po_avg_toi_min"].apply(min_to_mmss)

    main_tbl = pd.DataFrame({
        "Player": disp["full_name"],
        "Teams": disp["teams_played"],
        "Archetype": disp["Archetype"],
        "Archetype details": disp["Archetype details"],
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
        P = np.nan_to_num(view[pcols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(np.array([safe_float(row[c]) for c in pcols], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        P = np.clip(P, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        v_norm = np.linalg.norm(v)
        P_norm = np.linalg.norm(P, axis=1)
        denom = P_norm * v_norm
        numer = np.sum(P * v, axis=1)
        sim = np.divide(numer, denom, out=np.zeros(len(P), dtype=float), where=denom > 0)

        comps = view.copy()
        comps["Similarity (%)"] = (sim * 100).round(1)
        comps = comps[comps["full_name"] != row["full_name"]].sort_values("Similarity (%)", ascending=False).head(30)

        st.markdown('<div id="closest-comps"></div>', unsafe_allow_html=True)
        st.markdown("### Closest comps (by archetype mix)")

        comps_disp = comps.copy()
        comps_disp["Archetype"] = comps_disp["top_cluster"].apply(lambda x: archetype_name_map.get(safe_int(x), f"Archetype {safe_int(x)}"))
        comps_disp["Archetype details"] = comps_disp["top_cluster"].apply(
            lambda x: archetype_detail_map.get(safe_int(x), archetype_name_map.get(safe_int(x), f"Archetype {safe_int(x)}"))
        )
        comps_disp["Confidence"] = (comps_disp["confidence"].astype(float) * 100).round(1).astype(str) + "%"
        comps_disp["REG ATOI"] = comps_disp["reg_avg_toi_min"].apply(min_to_mmss)
        comps_disp["PO ATOI"] = comps_disp["po_avg_toi_min"].apply(min_to_mmss)

        comps_tbl = pd.DataFrame({
            "Player": comps_disp["full_name"],
            "Teams": comps_disp["teams_played"],
            "Archetype": comps_disp["Archetype"],
            "Archetype details": comps_disp["Archetype details"],
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
with tabs[2]:
    st.subheader("Team Roster Construction")
    st.markdown("""
**What you’re looking at**
- A depth-chart view of the selected team using the 12 forwards or 8 defensemen with the most regular-season ice time.
- Style concentration rings showing the dominant archetype overall and how much of it lives in the top half vs bottom half of the roster.

**What you can learn**
- Whether the team identity is concentrated in stars or spread through depth.
- Which lines/pairs carry each profile.
- Where the roster construction is balanced or thin.
""")

    all_teams = sorted({t for s in df["teams_played"].dropna().unique() for t in str(s).split("/")})
    team = st.selectbox("Team", all_teams)
    team_df = df[df["teams_played"].fillna("").str.contains(team)].copy()

    if team_df.empty:
        st.warning("No players found for that team.")
    else:
        slot_size = 3 if group == "forwards" else 2
        roster_n = 12 if group == "forwards" else 8
        top_n = 6 if group == "forwards" else 4
        slot_label = "Line" if group == "forwards" else "Pair"
        top_label = "Top 6" if group == "forwards" else "Top 4"
        bottom_label = "Bottom 6" if group == "forwards" else "Bottom 4"

        base_roster = team_df.copy()
        base_roster["reg_toi_total"] = pd.to_numeric(base_roster["reg_avg_toi_min"], errors="coerce").fillna(0) * pd.to_numeric(base_roster["reg_games"], errors="coerce").fillna(0)
        base_roster = base_roster.sort_values(["reg_toi_total", "confidence"], ascending=False).reset_index(drop=True)
        base_roster["last_key"] = base_roster["full_name"].map(last_name)
        line_data = load_line_combinations()
        combo_position = "line" if group == "forwards" else "pairing"
        combo_df = pd.DataFrame()
        if not line_data.empty:
            combo_df = line_data[
                (line_data["season_key"].astype(str) == str(season))
                & (line_data["playerTeam"].astype(str) == str(team))
                & (line_data["position"].astype(str) == combo_position)
            ].sort_values("toi_min", ascending=False).copy()

        used_last: set[str] = set()
        unit_cards: list[dict] = []
        roster_rows = []
        if not combo_df.empty:
            for combo in combo_df.itertuples(index=False):
                names = [p.strip() for p in str(combo.name).split("-") if p.strip()]
                if len(names) != slot_size:
                    continue
                keys = [last_name(n) for n in names]
                if any(k in used_last for k in keys):
                    continue
                matched = []
                for key, display_name in zip(keys, names):
                    candidates = base_roster[(base_roster["last_key"] == key) & (~base_roster["last_key"].isin(used_last))]
                    if candidates.empty:
                        break
                    matched.append(candidates.iloc[0].copy())
                if len(matched) != slot_size:
                    continue
                unit = len(unit_cards) + 1
                for m in matched:
                    m["Unit"] = unit
                    roster_rows.append(m)
                    used_last.add(m["last_key"])
                unit_cards.append({"unit": unit, "combo": combo, "players": matched})
                if len(unit_cards) == roster_n // slot_size:
                    break

        if len(roster_rows) < roster_n:
            for _, row in base_roster[~base_roster["last_key"].isin(used_last)].iterrows():
                row = row.copy()
                row["Unit"] = len(roster_rows) // slot_size + 1
                roster_rows.append(row)
                used_last.add(row["last_key"])
                if len(roster_rows) == roster_n:
                    break

        roster = pd.DataFrame(roster_rows).head(roster_n).reset_index(drop=True)
        roster["Depth"] = roster.index + 1
        roster["Unit"] = roster["Unit"].astype(int)
        roster["Archetype"] = roster["top_cluster"].apply(lambda x: archetype_name_map.get(safe_int(x), f"Archetype {safe_int(x)}"))
        roster["Confidence (%)"] = (roster["confidence"].astype(float) * 100).round(1)

        weights = roster["reg_toi_total"].to_numpy(dtype=float) + 1e-9
        shares = []
        for k in range(K):
            shares.append(float(np.average(roster[pcols[k]].to_numpy(dtype=float), weights=weights)))
        top_k = int(np.argmax(shares))
        dominant = archetype_name_map.get(top_k, f"Archetype {top_k}")
        dom_color, dom_fg = PROFILE_COLOR_MAP.get(dominant, ("#64748B", "#FFFFFF"))

        top_half = roster.head(top_n)
        bottom_half = roster.iloc[top_n:roster_n]

        def weighted_share(sub: pd.DataFrame, k: int) -> float:
            if sub.empty:
                return 0.0
            wt = sub["reg_toi_total"].to_numpy(dtype=float) + 1e-9
            return float(np.average(sub[f"p{k}"].to_numpy(dtype=float), weights=wt))

        overall_pct = shares[top_k] * 100
        top_pct = weighted_share(top_half, top_k) * 100
        bottom_pct = weighted_share(bottom_half, top_k) * 100
        spread_gap = abs(top_pct - bottom_pct)
        def ring(label: str, pct: float, sub: str) -> str:
            pct = max(0, min(100, float(pct)))
            return f"""
<div class="identity-ring">
  <div class="ring" style="background:conic-gradient({dom_color} {pct:.1f}%, #E5E7EB 0);">
    <div class="ring-core"><div class="ring-pct">{pct:.0f}%</div><div class="ring-label">{html.escape(label)}</div></div>
  </div>
  <div class="ring-sub">{html.escape(sub)}</div>
</div>
"""

        st.markdown(
            f"""
<style>
.construction-wrap {{display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:16px; margin: 8px 0 18px;}}
.identity-ring {{border:1px solid #E5E7EB; border-radius:8px; padding:18px; background:#FFFFFF; text-align:center;}}
.ring {{width:154px; height:154px; border-radius:50%; margin:0 auto 10px; display:grid; place-items:center;}}
.ring-core {{width:108px; height:108px; border-radius:50%; background:#FFFFFF; display:grid; place-items:center; align-content:center; box-shadow: inset 0 0 0 1px #E5E7EB;}}
.ring-pct {{font-size:30px; line-height:1; font-weight:800; color:#111827;}}
.ring-label {{font-size:12px; color:#4B5563; margin-top:4px;}}
.ring-sub {{font-size:13px; color:#374151;}}
.line-grid {{display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:12px;}}
.unit-card {{border:1px solid #E5E7EB; border-left:6px solid #CBD5E1; border-radius:8px; background:#FFFFFF; overflow:hidden;}}
.unit-head {{padding:10px 12px; font-weight:800; background:#F9FAFB; border-bottom:1px solid #E5E7EB; display:flex; justify-content:space-between; gap:8px; align-items:center;}}
.unit-meta {{font-size:12px; font-weight:700; color:#64748B;}}
.player-row {{padding:10px 12px; border-bottom:1px solid #F3F4F6;}}
.player-row:last-child {{border-bottom:0;}}
.player-name {{font-weight:750; color:#111827;}}
.player-meta {{font-size:12px; color:#6B7280; margin-top:2px;}}
.profile-chip {{display:inline-flex; margin-top:7px; padding:3px 8px; border-radius:999px; font-size:12px; font-weight:750;}}
.xg-chip {{display:inline-flex; padding:2px 7px; border-radius:999px; font-size:12px; font-weight:800; white-space:nowrap;}}
.mix-table {{width:100%; border-collapse:collapse; border:1px solid #E5E7EB; border-radius:8px; overflow:hidden;}}
.mix-table th,.mix-table td {{padding:9px 10px; border-bottom:1px solid #E5E7EB;}}
.mix-table th {{background:#F9FAFB; color:#4B5563; text-align:left;}}
.mix-table tr:last-child td {{border-bottom:0;}}
@media (max-width: 900px) {{.construction-wrap,.line-grid {{grid-template-columns:1fr;}}}}
</style>
<div class="construction-wrap">
{ring("Overall", overall_pct, dominant)}
{ring(top_label, top_pct, f"{dominant} concentration")}
{ring(bottom_label, bottom_pct, f"{dominant} concentration")}
</div>
""",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([2.4, 1])
        c1.metric("Dominant profile", dominant)
        c2.metric("Top/bottom gap", f"{spread_gap:.0f} pts")

        st.markdown(f"### {team} {slot_label.lower()} construction")
        source_msg = "MoneyPuck 5v5 line/pairing minutes" if unit_cards else "regular-season player TOI fallback"
        st.caption(f"Units are selected from {source_msg}.")

        unit_html = ['<div class="line-grid">']
        for unit, sub in roster.groupby("Unit", sort=True):
            unit_arch = sub["Archetype"].mode().iloc[0] if not sub.empty else dominant
            unit_color, _ = PROFILE_COLOR_MAP.get(unit_arch, ("#CBD5E1", "#111827"))
            combo_match = next((u for u in unit_cards if u["unit"] == int(unit)), None)
            if combo_match:
                combo = combo_match["combo"]
                chip_text, chip_bg, chip_fg = xg_chip(getattr(combo, "xg_pct", pd.NA))
                meta = f'{getattr(combo, "toi_min", 0):.0f} min · <span class="xg-chip" style="background:{chip_bg};color:{chip_fg};">{chip_text}</span>'
            else:
                meta = f'{float(sub["reg_toi_total"].sum()):.0f} min · xG n/a*'
            unit_html.append(
                f'<div class="unit-card" style="border-left-color:{unit_color};"><div class="unit-head"><span>{slot_label} {int(unit)}</span><span class="unit-meta">{meta}</span></div>'
            )
            for _, r in sub.iterrows():
                arch = r["Archetype"]
                bg, fg = PROFILE_COLOR_MAP.get(arch, ("#E5E7EB", "#111827"))
                conf_val = float(r["Confidence (%)"])
                if conf_val >= 90:
                    conf_bg, conf_fg = "#DCFCE7", "#166534"
                elif conf_val >= 80:
                    conf_bg, conf_fg = "#FEF9C3", "#854D0E"
                else:
                    conf_bg, conf_fg = "#FEE2E2", "#991B1B"
                conf_chip = f'<span style="display:inline-flex;padding:1px 6px;border-radius:999px;font-size:11px;font-weight:800;background:{conf_bg};color:{conf_fg};margin-left:5px;">{conf_val:.1f}%</span>'
                g = int(r.get("reg_goals", 0))
                a = int(r.get("reg_assists", 0))
                p = int(r.get("reg_points", 0))
                unit_html.append(
                    f"""<div class="player-row">
  <div class="player-name">{html.escape(str(r["full_name"]))}{conf_chip}</div>
  <div class="player-meta">{html.escape(str(r["position"]))} · {int(r.get("reg_games", 0))} GP · {min_to_mmss(r.get("reg_avg_toi_min", 0))} ATOI · {g}G, {a}A, {p}P</div>
  <span class="profile-chip" style="background:{bg};color:{fg};">{html.escape(arch)}</span>
</div>"""
                )
            unit_html.append("</div>")
        unit_html.append("</div>")
        st.markdown("".join(unit_html), unsafe_allow_html=True)
        if not unit_cards:
            st.caption("* xG data not available for these units; units use regular-season player TOI.")

        st.markdown("### Roster profile mix")
        mix_rows = []
        for k, share in enumerate(shares):
            name = archetype_name_map.get(k, f"Archetype {k}")
            mix_rows.append({
                "Archetype": name,
                "Overall (%)": round(share * 100, 1),
                f"{top_label} (%)": round(weighted_share(top_half, k) * 100, 1),
                f"{bottom_label} (%)": round(weighted_share(bottom_half, k) * 100, 1),
            })
        mix_df = (
            pd.DataFrame(mix_rows)
            .groupby("Archetype", as_index=False)
            .sum(numeric_only=True)
            .sort_values("Overall (%)", ascending=False)
        )
        header = f"<tr><th>Archetype</th><th>Overall (%)</th><th>{top_label} (%)</th><th>{bottom_label} (%)</th></tr>"
        rows_html = []
        for r in mix_df.to_dict("records"):
            bg, fg = PROFILE_COLOR_MAP.get(r["Archetype"], ("#E5E7EB", "#111827"))
            rows_html.append(
                "<tr>"
                f'<td><span class="profile-chip" style="background:{bg};color:{fg};margin-top:0;">{html.escape(r["Archetype"])}</span></td>'
                + pct_cell(r["Overall (%)"])
                + pct_cell(r[f"{top_label} (%)"])
                + pct_cell(r[f"{bottom_label} (%)"])
                + "</tr>"
            )
        st.markdown(f'<table class="mix-table">{header}{"".join(rows_html)}</table>', unsafe_allow_html=True)

        st.markdown("### Depth chart table")
        show_roster = roster[["Unit", "Depth", "full_name", "position", "Archetype", "Confidence (%)", "reg_games", "reg_avg_toi_min", "reg_points", "reg_goals", "reg_assists"]].copy()
        show_roster = show_roster.rename(columns={
            "Unit": slot_label,
            "full_name": "Player",
            "position": "Pos",
            "reg_games": "REG GP",
            "reg_avg_toi_min": "REG ATOI",
            "reg_points": "REG P",
            "reg_goals": "REG G",
            "reg_assists": "REG A",
        })
        show_roster["REG ATOI"] = show_roster["REG ATOI"].apply(min_to_mmss)
        show_roster["Confidence"] = show_roster["Confidence (%)"].apply(lambda x: f"{x:.1f}%")
        show_roster = show_roster.drop(columns=["Confidence (%)"])
        make_badge_grid(
            show_roster,
            height=420,
            pin_cols=("Player",),
            conf_mode="fixed",
            conf_q33=80.0,
            conf_q67=90.0,
            player_min_px=160,
            player_max_px=280,
            key_suffix=f"depth_{team}_{season}_{group}",
        )

        st.divider()
        st.markdown("### League-context roster gaps")
        st.caption("Gap score compares this team's profile mix to the rest of the league using the same selected group.")

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

            me["Archetype"] = me["k"].apply(lambda x: archetype_name_map.get(int(x), f"Archetype {int(x)}"))
            me["Team share (%)"] = (me["share"] * 100).round(1)
            me["League avg (%)"] = (me["mean_share"] * 100).round(1)
            me["Strong coverage (%)"] = (me["coverage"] * 100).round(1)
            me["Reliance on top 2 (%)"] = (me["concentration"] * 100).round(1)
            me["Z-score"] = me["z"].round(2)

            me["Note"] = ""
            me.loc[(me["z"] < -0.75) | ((me["z"] < -0.5) & (me["cov_rank"] < 0.35)), "Note"] = "Underrepresented"
            me.loc[(me["Note"] == "") & (me["conc_rank"] > 0.75) & (me["cov_rank"] < 0.5), "Note"] = "Thin coverage"

            show = me[[
                "Archetype","Team share (%)","League avg (%)","Z-score","Strong coverage (%)","Reliance on top 2 (%)","Note"
            ]].reset_index(drop=True)
            make_badge_grid(
                show,
                height=max(300, 42 * len(show) + 60),
                pin_cols=("Archetype",),
                confidence_col="__none__",
                player_min_px=260,
                player_max_px=400,
                key_suffix=f"gaps_{team}_{season}_{group}",
            )

# -------------------------
# Need Finder
# -------------------------
with tabs[3]:
    st.subheader("Need Finder (find players who match a target archetype)")
    st.markdown("""
**What you’re looking at**
- A ranked list of players who best match a selected style profile.

**How to use it**
- Pick the archetype you want to add to a roster.
- Optionally exclude your own team.
- Increase minimum regular-season games to avoid tiny samples.
- “Target similarity (%)” is the model’s estimated probability that the player belongs to that archetype.
""")

    all_teams = sorted({t for s in df["teams_played"].dropna().unique() for t in str(s).split("/")})
    exclude_team = st.selectbox("Exclude team (optional)", ["(none)"] + all_teams)
    target_options = {archetype_name_map.get(k, f"Archetype {k}"): k for k in range(K)}
    target_choice = st.selectbox("Target archetype", list(target_options.keys()), key=f"target_archetype_{season}_{group}")

    # convert the selected label back to the integer k
    target = target_options[target_choice]


    min_reg_games = st.slider("Min REG games", 0, 82, 20, step=5)

    view = df.copy()
    if exclude_team != "(none)":
        view = view[~view["teams_played"].fillna("").str.contains(exclude_team)]
    view = view[view.get("reg_games",0) >= min_reg_games].copy()

    view["Target similarity (%)"] = (view[f"p{target}"] * 100).round(1)
    out = view.sort_values(["Target similarity (%)","reg_points"], ascending=False).head(80).copy()

    o = out.copy()
    o["Archetype"] = o["top_cluster"].apply(lambda x: archetype_name_map.get(safe_int(x), f"Archetype {safe_int(x)}"))
    o["Archetype details"] = o["top_cluster"].apply(
        lambda x: archetype_detail_map.get(safe_int(x), archetype_name_map.get(safe_int(x), f"Archetype {safe_int(x)}"))
    )
    o["Confidence"] = (o["confidence"].astype(float) * 100).round(1).astype(str) + "%"
    o["REG ATOI"] = o["reg_avg_toi_min"].apply(min_to_mmss)
    o["PO ATOI"] = o["po_avg_toi_min"].apply(min_to_mmss)

    need_tbl = pd.DataFrame({
        "Player": o["full_name"],
        "Teams": o["teams_played"],
        "Archetype": o["Archetype"],
        "Archetype details": o["Archetype details"],
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
