import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from lib import (  # noqa: E402
    ARCHETYPE_LABEL_CACHE_KEY,
    PROFILE_COLOR_MAP,
    available_seasons,
    build_archetype_name_summary,
    load_all_seasons_group,
    parse_trait_string,
    readable_trait_label,
)


REPORTS_DIR = Path("reports")
ERA_ORDER = ["2008-2009", "2010-2014", "2015-2019", "2020-2025"]

ARCHETYPE_CELL_JS = JsCode(
    """
function(params) {
  const map = __PROFILE_COLOR_MAP__;
  const c = map[params.value] || ["#E5E7EB", "#111827"];
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
)

PRELINE_CELL = JsCode(
    """
function(params) {
  return {
    whiteSpace: 'pre-line',
    lineHeight: '1.25',
    display: 'flex',
    alignItems: 'center'
  };
}
"""
)


st.set_page_config(page_title="What Are All the Player Styles According to the Model?", layout="wide")
st.markdown(
    """<style>
section[data-testid="stSidebar"] [data-testid="stPageLink"] a {
  white-space: normal !important;
  line-height: 1.2 !important;
}
</style>""",
    unsafe_allow_html=True,
)
st.title("What Are All the Player Styles According to the Model?")


def season_start_year(season_key: str) -> int:
    try:
        return int(str(season_key)[:4])
    except Exception:
        return 0


def era5(season_key: str) -> str:
    y = season_start_year(season_key)
    if y <= 2009:
        return "2008-2009"
    if y <= 2014:
        return "2010-2014"
    if y <= 2019:
        return "2015-2019"
    return "2020-2025"


def prettify_traits_lines(s: str, max_items: int = 4) -> str:
    tokens = parse_trait_string(s)
    lines = []
    for feat, z in tokens[:max_items]:
        arrow = "↑" if z >= 0 else "↓"
        lines.append(f"{arrow} {readable_trait_label(feat)} ({z:+.1f}σ)")
    return "\n".join(lines)


def normalize_legacy_archetype_name(name: str) -> str:
    replacements = {
        "Low-Contact Scoring Profile": "Perimeter Skill Scorer",
        "Low-Contact Scorer": "Perimeter Skill Scorer",
        "Shooting / Scoring Profile": "Shot-Creation Scorer",
        "Shot-Volume Scorer": "Shot-Creation Scorer",
        "Volume Shooter": "Shot-Creation Scorer",
        "Finisher": "Perimeter Skill Scorer",
        "Checking-Line Contact Profile": "Checking-Line Disruptor",
        "Puck-Pressure Scoring Profile": "Puck-Pressure Scorer",
    }
    if name is None or pd.isna(name):
        return None
    return replacements.get(str(name), str(name))


def archetype_description_from_traits(name: str, top_traits: str, low_traits: str) -> str:
    n = str(name).lower()
    top = str(top_traits).lower()
    low = str(low_traits).lower()

    if "risk/reward" in n or "high-touch" in n:
        return "Puck-dominant creator: handles the puck often and drives plays, but the role comes with more turnover risk."
    if "low-contact scorer" in n:
        return "Skill-first scorer: produces offense while staying lighter on hits, blocks, and penalty-driven physical play."
    if "agitating" in n or ("reg_pim_per60" in top and "reg_hits_per60" in top):
        return "Physical, high-edge profile: plays a heavy game and tends to take more penalties. Often lower offensive creation than scoring archetypes."
    if "shot-blocking" in n or "reg_blocked_shots_per60" in top:
        return "Defense-tilted profile: blocks shots and plays physical minutes. Typically contributes more via defense/role usage than raw scoring."
    if "playmaking" in n or ("reg_assists_per60" in top and "reg_shots_per60" in top):
        return "Offense-driving profile: generates shots and assists at high rates. Tends to produce strong points/60."
    if "two-way" in n or ("reg_takeaways_per60" in top and "reg_giveaways_per60" in low):
        return "Pressure-and-recover profile: creates takeaways while limiting giveaways. Contributes on both sides of the puck."
    if "pk-leaning" in n or "reg_pk_share" in top:
        return "Penalty-kill leaning profile: more value comes from shorthanded usage and defensive role."
    if "pp-leaning" in n or "reg_pp_share" in top:
        return "Power-play leaning profile: production is driven by scoring-role deployment and PP usage."

    top_tokens = parse_trait_string(top_traits)
    low_tokens = parse_trait_string(low_traits)
    top_labels = [readable_trait_label(feat).lower() for feat, _ in top_tokens[:2]]
    low_labels = [readable_trait_label(feat).lower() for feat, _ in low_tokens[:2]]
    if top_labels:
        desc = f"Blended profile whose strongest signals are {', '.join(top_labels)}"
        if low_labels:
            desc += f", with less emphasis on {', '.join(low_labels)}"
        return desc + "."
    return "Blended role profile defined more by usage and secondary contributions than by one extreme scoring or defensive marker."


@st.cache_data
def load_traits_csv(group: str, season_key: str) -> pd.DataFrame:
    p = REPORTS_DIR / f"archetype_traits_{group}_{season_key}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["season"] = season_key
    return df


@st.cache_data
def build_season_cluster_to_name(
    group: str,
    label_version: str = ARCHETYPE_LABEL_CACHE_KEY,
) -> dict[tuple[str, int], str]:
    mapping: dict[tuple[str, int], str] = {}
    for sk in available_seasons():
        t = load_traits_csv(group, sk)
        if t.empty:
            continue
        for _, tr in t.iterrows():
            kk = int(tr["cluster"])
            ht = parse_trait_string(tr.get("top_traits", ""))
            lt = parse_trait_string(tr.get("low_traits", ""))
            nm, _ = build_archetype_name_summary(kk, ht, lt, group=group)
            mapping[(sk, kk)] = normalize_legacy_archetype_name(nm)
    return mapping


def traits_registry(group: str, mapping: dict[tuple[str, int], str]) -> dict[str, dict[str, str]]:
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for sk in available_seasons():
        t = load_traits_csv(group, sk)
        if t.empty:
            continue
        for _, tr in t.iterrows():
            kk = int(tr["cluster"])
            name = normalize_legacy_archetype_name(mapping.get((sk, kk)))
            if not name:
                continue
            hi = str(tr.get("top_traits", "")).strip()
            lo = str(tr.get("low_traits", "")).strip()
            if not hi and not lo:
                continue
            counts.setdefault(name, {"hi": {}, "lo": {}})
            counts[name]["hi"][hi] = counts[name]["hi"].get(hi, 0) + 1
            counts[name]["lo"][lo] = counts[name]["lo"].get(lo, 0) + 1

    registry = {}
    for name, c in counts.items():
        hi = max(c["hi"].items(), key=lambda x: x[1])[0] if c["hi"] else ""
        lo = max(c["lo"].items(), key=lambda x: x[1])[0] if c["lo"] else ""
        registry[name] = {
            "high": prettify_traits_lines(hi, max_items=4),
            "low": prettify_traits_lines(lo, max_items=3),
            "desc": archetype_description_from_traits(name, hi, lo),
        }
    return registry


def build_glossary(
    all_df: pd.DataFrame,
    mapping: dict[tuple[str, int], str],
    traits_map: dict[str, dict[str, str]],
) -> pd.DataFrame:
    if all_df.empty:
        return pd.DataFrame()
    pcols = [c for c in all_df.columns if isinstance(c, str) and c.startswith("p") and c[1:].isdigit()]
    if not pcols:
        return pd.DataFrame()

    parts = []
    for pc in pcols:
        k = int(pc[1:])
        tmp = all_df[["season", "player_id", "full_name", pc]].copy()
        tmp = tmp.rename(columns={pc: "prob"})
        tmp["k"] = k
        tmp["archetype_name"] = tmp.apply(
            lambda r: normalize_legacy_archetype_name(mapping.get((r["season"], int(r["k"])), None)),
            axis=1,
        )
        tmp = tmp.dropna(subset=["archetype_name"])
        parts.append(tmp)

    long = pd.concat(parts, ignore_index=True)
    long["era"] = long["season"].apply(era5)
    long = long[long["era"].isin(ERA_ORDER)].copy()

    if "reg_games" in all_df.columns:
        long = long.merge(all_df[["season", "player_id", "reg_games"]], on=["season", "player_id"], how="left")
        long = long[long["reg_games"].fillna(0) >= 15].copy()

    rows = []
    for name, sub in long.groupby("archetype_name"):
        sub = sub.copy()
        sub["start_year"] = sub["season"].astype(str).str[:4].astype(int)
        ranked = sub.sort_values(["start_year", "prob"], ascending=[False, False])
        used = set()
        exemplars = []
        for _, r in ranked.iterrows():
            pid = int(r["player_id"])
            if pid in used:
                continue
            used.add(pid)
            exemplars.append(str(r["full_name"]))
            if len(exemplars) == 5:
                break

        rows.append({
            "Archetype name": name,
            "Description": traits_map.get(name, {}).get("desc", ""),
            "High traits": traits_map.get(name, {}).get("high", ""),
            "Low traits": traits_map.get(name, {}).get("low", ""),
            "Exemplars": "\n".join(exemplars),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.groupby("Archetype name", as_index=False)
        .agg({
            "Description": "first",
            "High traits": "first",
            "Low traits": "first",
            "Exemplars": "first",
        })
        .sort_values("Archetype name")
    )


def show_multiline_table(df: pd.DataFrame, height: int = 650):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(resizable=True, sortable=True, filter=True, wrapText=True, autoHeight=True)
    gb.configure_grid_options(domLayout="autoHeight", suppressSizeToFit=True, alwaysShowHorizontalScroll=True)
    for c in df.columns:
        gb.configure_column(c, cellStyle=PRELINE_CELL)
    if "Archetype name" in df.columns:
        gb.configure_column("Archetype name", width=380, minWidth=340, wrapText=True, autoHeight=True, cellStyle=ARCHETYPE_CELL_JS)

    AgGrid(
        df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.NO_UPDATE,
        theme="streamlit",
        height=height,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
    )


st.markdown(
    "This glossary aggregates the named archetypes learned across all available seasons. "
    "Each row shows the clearest statistical signature and a few recent example players."
)

all_f = load_all_seasons_group("forwards")
all_d = load_all_seasons_group("defense")
map_f = build_season_cluster_to_name("forwards")
map_d = build_season_cluster_to_name("defense")
traits_f = traits_registry("forwards", map_f)
traits_d = traits_registry("defense", map_d)

glossary_group = st.radio("Glossary group", ["forwards", "defense"], horizontal=True)
glossary = build_glossary(
    all_f if glossary_group == "forwards" else all_d,
    map_f if glossary_group == "forwards" else map_d,
    traits_f if glossary_group == "forwards" else traits_d,
)

if glossary.empty:
    st.info("No glossary available yet.")
else:
    show_multiline_table(glossary)
