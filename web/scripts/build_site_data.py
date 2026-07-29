#!/usr/bin/env python3
"""Build compact browser data for the standalone Vercel app.

This script only reads the existing Streamlit data products. It writes a
separate JSON bundle under web/data and never changes the Python app.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"
DATA_DIR = ROOT / "data" / "app"
REPORTS_DIR = ROOT / "reports"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.archetype_labels import (  # noqa: E402
    build_archetype_name_summary,
    canonical_profile_name,
    parse_trait_string,
    readable_trait_label,
)


def clean(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, str, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, digits)
    if hasattr(value, "item"):
        return clean(value.item(), digits=digits)
    return value


def one_decimal(value: float) -> str:
    rounded = Decimal(str(value)).quantize(
        Decimal("0.1"),
        rounding=ROUND_HALF_UP,
    )
    return f"{rounded:.1f}"


def season_label(key: str) -> str:
    key = str(key)
    return f"{key[:4]}–{key[6:]}" if len(key) == 8 else key


def available_seasons() -> list[str]:
    forwards = {
        p.stem.removeprefix("players_forwards_")
        for p in DATA_DIR.glob("players_forwards_*.parquet")
    }
    defense = {
        p.stem.removeprefix("players_defense_")
        for p in DATA_DIR.glob("players_defense_*.parquet")
    }
    return sorted(
        [
            s
            for s in forwards & defense
            if len(s) == 8 and s.isdigit() and int(s[:4]) >= 2008
        ],
        reverse=True,
    )


def profile_maps(seasons: list[str]) -> dict[str, dict[str, dict[int, str]]]:
    maps: dict[str, dict[str, dict[int, str]]] = {
        "forwards": {},
        "defense": {},
    }
    for group in maps:
        for season in seasons:
            path = REPORTS_DIR / f"archetype_traits_{group}_{season}.csv"
            if not path.exists():
                maps[group][season] = {}
                continue
            traits = pd.read_csv(path)
            season_map: dict[int, str] = {}
            for _, row in traits.iterrows():
                cluster = int(row["cluster"])
                high = parse_trait_string(row.get("top_traits", ""))
                low = parse_trait_string(row.get("low_traits", ""))
                name, _ = build_archetype_name_summary(
                    cluster,
                    high,
                    low,
                    group=group,
                )
                season_map[cluster] = canonical_profile_name(name)
            maps[group][season] = season_map
    return maps


def describe_profile(name: str, high: list[tuple[str, float]], low: list[tuple[str, float]]) -> str:
    lowered = name.lower()
    if "risk/reward" in lowered:
        return "Puck-dominant creation with more turnover exposure."
    if "playmaking" in lowered or "play-driving" in lowered:
        return "Creates offense through puck movement, shots, and sustained possession."
    if "two-way" in lowered:
        return "Adds offense while recovering pucks and limiting chances against."
    if "shot-blocking" in lowered:
        return "Protects the middle through blocks, contact, and defensive usage."
    if "shutdown" in lowered or "defensive role" in lowered:
        return "Defense-first deployment built around suppression and difficult minutes."
    if "finisher" in lowered or "scorer" in lowered:
        return "Turns touches into shots and scoring chances at a high rate."
    if "workload" in lowered or "specialist" in lowered:
        return "A role-driven profile shaped by usage and situational minutes."

    signals = [readable_trait_label(feature).lower() for feature, _ in high[:2]]
    if signals:
        return f"Defined by {', '.join(signals)}."
    return "A blended profile without one dominant statistical signal."


STYLE_READS: dict[str, tuple[str, str]] = {
    "Agitating Heavy-Contact Forward": (
        "Edge and Contact",
        "Forecheck pressure, confrontation and disruption are carrying as "
        "much roster value as clean possession.",
    ),
    "Balanced Net-Front Contributor": (
        "Net-Front Balance",
        "The premium is on players who can win interior ice and keep a play "
        "alive without becoming one-dimensional.",
    ),
    "Checking-Line Disruptor": (
        "Checking-Line Pressure",
        "The hockey value sits in pursuit, contact and nuisance work—the "
        "shifts that bend a game before they reach the scoring line.",
    ),
    "Cycle Pressure Play-Driver": (
        "Cycle Pressure",
        "This is an offensive-zone profile: retrievals, possession and "
        "second efforts are doing more of the separating than one-and-done "
        "rushes.",
    ),
    "Defensive Role Defenseman": (
        "Defensive Structure",
        "The blue-line economy is built on structure and assignment "
        "discipline, with fewer defenders asked to freelance their way into "
        "value.",
    ),
    "High-Event Physical Defenseman": (
        "Physical Event Hockey",
        "The back end is being defined by confrontation and event volume, a "
        "trade-off that can tilt territory but also raises the temperature.",
    ),
    "High-Touch Puck Mover": (
        "Puck Movement",
        "The puck is flowing through players trusted to handle it often and "
        "extend sequences, not simply finish them.",
    ),
    "High-Touch Risk/Reward Scorer": (
        "Puck-Dominant Scoring",
        "Creation is being driven by frequent touches and scoring ambition, "
        "with some turnover risk accepted as the cost of doing business.",
    ),
    "High-Volume Playmaking Scorer": (
        "Volume Playmaking",
        "The separating skill is dual-threat creation: shooting often enough "
        "to hold defenders while still moving the puck to the next option.",
    ),
    "Interior Net-Front Finisher": (
        "Inside Finishing",
        "The premium is on getting to the hard areas and ending plays around "
        "the crease rather than living on low-contact volume.",
    ),
    "Low-Event Puck-Moving Defenseman": (
        "Low-Noise Puck Movement",
        "The back end is favoring controlled puck movement—advancing play "
        "without turning every touch into a high-variance event.",
    ),
    "Offensive Puck-Moving Defenseman": (
        "Blue-Line Offense",
        "The defense corps is being asked to add offense through exits, point "
        "touches and distribution, not merely survive its minutes.",
    ),
    "Perimeter Skill Scorer": (
        "Perimeter Skill",
        "Skill and shooting remain central, but much of the offense is "
        "arriving from space rather than constant net-front pressure.",
    ),
    "Physical Shutdown Defenseman": (
        "Physical Shutdown Hockey",
        "The blue-line value is direct: close space, protect the middle and "
        "make difficult minutes physically expensive.",
    ),
    "Play-Driving Puck-Moving Defenseman": (
        "Puck-Moving Control",
        "Possession is being tilted from the back end, with defenders expected "
        "to start exits and keep attacks alive.",
    ),
    "Point-Usage Power-Play Defenseman": (
        "Power-Play Point Usage",
        "Power-play touches and point usage are carrying unusual weight, "
        "making deployment part of the style rather than a footnote.",
    ),
    "Puck-Pressure Transition Defenseman": (
        "Pressure Into Transition",
        "Pressure is feeding transition: recoveries matter because the next "
        "pass can turn defense into clean offense.",
    ),
    "Puck-Pressure Two-Way Creator": (
        "Two-Way Puck Pressure",
        "Puck pursuit is feeding offense, tying recoveries, support and "
        "creation into the same job description.",
    ),
    "Rush / Transition Chance Creator": (
        "Rush Creation",
        "The season tilts toward speed through the neutral zone and offense "
        "created before the defense can get set.",
    ),
    "Shot-Blocking Contact Specialist": (
        "Defensive Detail",
        "Contact, shot blocking and a willingness to absorb difficult minutes "
        "hold real roster value in this mix.",
    ),
    "Shot-Blocking Defensive Defenseman": (
        "Shot-Blocking Defense",
        "The model's center of gravity is defensive detail—inside positioning, "
        "blocks and the willingness to take hard minutes.",
    ),
    "Shot-Creation Scorer": (
        "Shot Creation",
        "The model is rewarding players who manufacture attempts and force "
        "volume rather than live on selective finishing.",
    ),
    "Suppression Workload Forward": (
        "Suppression and Workload",
        "The roster value lives in difficult minutes, defensive responsibility "
        "and taking air out of opposing attacks.",
    ),
    "Transition Risk/Reward Defenseman": (
        "Aggressive Transition",
        "The back end is trading some security for advancement, asking "
        "defenders to move the puck aggressively and live with the occasional "
        "mistake.",
    ),
    "Two-Way Shot-Share Driver": (
        "Two-Way Possession",
        "The most valuable forwards are doing more than producing: they are "
        "helping own the ice and turn possession into repeatable offense.",
    ),
    "Two-Way Skill Scorer": (
        "Two-Way Skill",
        "Two-way utility is at the center of roster construction, with "
        "creation and responsibility reinforcing each other.",
    ),
}

SEASON_EDITORIALS: dict[str, dict[str, tuple[str, str]]] = {
    "20082009": {
        "forwards": (
            "Transition and playmaking control the map",
            "The two creation-first lanes—{dominant_name} and "
            "{runner_up_name}—combine for {top_two_share}% of the forward "
            "pool. With only {tail_share}% outside the top three, the model "
            "resolves a clear hierarchy rather than six equal-sized roles.",
        ),
        "defense": (
            "A defensive lead with an offensive counterweight",
            "{dominant_name} leads, but {runner_up_name} still takes "
            "{runner_up_share}% of the defense pool. The top three absorb "
            "{top_three_share}% across {profile_count} learned styles, leaving "
            "the last two as edge cases rather than co-equal lanes.",
        ),
    },
    "20092010": {
        "forwards": (
            "Two creation lanes split the forward pool",
            "Only {dominant_gap} points separate {dominant_name} from "
            "{runner_up_name}, and together they account for "
            "{top_two_share}% of forwards. This is less a runaway than a "
            "two-lane division of high-creation players.",
        ),
        "defense": (
            "The blue line lands in a near dead heat",
            "The leading defense profiles sit just {dominant_gap} points "
            "apart. With {after_top_two_share}% of defenders outside that "
            "pair, the model sees a genuine contest at the top and a "
            "meaningful supporting tier behind it.",
        ),
    },
    "20102011": {
        "forwards": (
            "Net-front finishing swallows the forward map",
            "{dominant_name} contains {dominant_share}% of the forward pool "
            "and clears the runner-up by {dominant_gap} points. Even with "
            "{profile_count} learned styles, this season’s feature space is "
            "organized around one overwhelming interior-scoring identity.",
        ),
        "defense": (
            "Defensive detail leads without monopolizing",
            "{dominant_name} holds the largest share, but "
            "{after_dominant_share}% of defenders land elsewhere. The "
            "{tail_share}% outside the top three keeps the bottom of the map "
            "relevant instead of reducing the season to one defensive type.",
        ),
    },
    "20112012": {
        "forwards": (
            "Rush offense provides the anchor",
            "Nearly one in two forwards land in {dominant_name}. The remaining "
            "{after_dominant_share}% is spread across five other styles, so "
            "the model shows one strong anchor above a diversified support "
            "layer.",
        ),
        "defense": (
            "No single defense identity breaks away",
            "{dominant_name} leads at {dominant_share}%, but "
            "{runner_up_name} remains close at {runner_up_share}%. Six "
            "learned styles and a {tail_share}% share outside the top three "
            "make this a comparatively distributed within-season map.",
        ),
    },
    "20122013": {
        "forwards": (
            "Perimeter skill becomes the central lane",
            "{dominant_name} and {runner_up_name} combine for "
            "{top_two_share}% of the forward pool. Once the third profile is "
            "included, only {tail_share}% remains for the other two styles—a "
            "sharply concentrated snapshot.",
        ),
        "defense": (
            "Transition pressure creates separation",
            "{dominant_name} clears the next defense style by "
            "{dominant_gap} points. The bottom three profiles share just "
            "{tail_share}% of the pool, so most defenders sit inside a clear "
            "three-tier hierarchy.",
        ),
    },
    "20132014": {
        "forwards": (
            "Cycle pressure gives the season its shape",
            "The leading pair accounts for {top_two_share}% of forwards, with "
            "{dominant_name} holding the larger lane. The final three styles "
            "combine for only {tail_share}%, leaving cycle pressure and "
            "two-way shot-share play as the defining split.",
        ),
        "defense": (
            "Puck movement and shot blocking share the stage",
            "The gap between {dominant_name} and {runner_up_name} is "
            "{dominant_gap} points, while {tail_share}% sits outside the top "
            "three. The model reads the blue line as a contest between "
            "advancement and defensive detail, not a one-style season.",
        ),
    },
    "20142015": {
        "forwards": (
            "Contrasting identities define the top",
            "{dominant_name} and {runner_up_name} together cover "
            "{top_two_share}% of forwards. Their labels describe opposite "
            "routes to a roster role—transition creation versus contact and "
            "defensive involvement—while the other three styles share only "
            "{tail_share}%.",
        ),
        "defense": (
            "The blue line is effectively a three-style map",
            "The top three defense profiles account for {top_three_share}% of "
            "the pool across four learned styles. The meaningful split is "
            "between {dominant_name} at {dominant_share}% and "
            "{runner_up_name} at {runner_up_share}%; the fourth profile is "
            "barely present.",
        ),
    },
    "20152016": {
        "forwards": (
            "High-touch scoring leads without closing the field",
            "{dominant_name} owns the plurality at {dominant_share}%, but "
            "{tail_share}% of forwards remain outside the top three. That "
            "leaves a clear first identity alongside a real specialist tier, "
            "rather than a closed three-style market.",
        ),
        "defense": (
            "Puck-moving control carries the plurality",
            "{dominant_name} leads a four-style defense map without reaching "
            "a majority. The top two take {top_two_share}%, leaving "
            "{after_top_two_share}% split between the remaining profiles.",
        ),
    },
    "20162017": {
        "forwards": (
            "Risk/reward scoring crosses the halfway mark",
            "{dominant_name} reaches {dominant_share}% and leads the second "
            "profile by {dominant_gap} points. The bottom three still account "
            "for {tail_share}%, enough to preserve a smaller but visible "
            "specialist layer.",
        ),
        "defense": (
            "Structure and transition finish level",
            "{dominant_name} and {runner_up_name} are separated by only "
            "{dominant_gap} points and combine for {top_two_share}% of the "
            "defense pool. A third style takes most of what remains, leaving "
            "just {tail_share}% for the bottom three.",
        ),
    },
    "20172018": {
        "forwards": (
            "Two-way puck pressure makes an outlier map",
            "{dominant_name} contains {dominant_share}% of forwards and leads "
            "the runner-up by {dominant_gap} points. The other four learned "
            "styles divide only {after_dominant_share}%, making this a nearly "
            "single-center classification.",
        ),
        "defense": (
            "Defense-first profiles dominate the split",
            "{dominant_name} and {runner_up_name} combine for "
            "{top_two_share}% of defenders. The top three reach "
            "{top_three_share}% across four styles, so almost the entire map "
            "sits inside a defense-first hierarchy.",
        ),
    },
    "20182019": {
        "forwards": (
            "The forward field opens up",
            "The leading profile reaches only {dominant_share}%, and "
            "{tail_share}% of forwards sit outside the top three. Across six "
            "learned styles, the model shows several viable identities rather "
            "than one dominant lane.",
        ),
        "defense": (
            "Puck-moving control leads a tiered blue line",
            "{dominant_name} leads at {dominant_share}%, followed by "
            "{runner_up_name} at {runner_up_share}%. The last two profiles "
            "share {tail_share}%, creating a clear top tier with a small tail.",
        ),
    },
    "20192020": {
        "forwards": (
            "Transition and inside offense split the top",
            "{dominant_name} leads, but {runner_up_name} and the third profile "
            "keep the map from becoming a runaway. Those three styles account "
            "for {top_three_share}%, leaving {tail_share}% across the final "
            "three identities.",
        ),
        "defense": (
            "Two profiles nearly absorb the defense pool",
            "{dominant_name} and {runner_up_name} combine for "
            "{top_two_share}% of defenders. With the top three at "
            "{top_three_share}% across four learned styles, the fourth profile "
            "is statistically peripheral.",
        ),
    },
    "20202021": {
        "forwards": (
            "Rush creation leads a two-lane attack",
            "{dominant_name} and {runner_up_name} account for "
            "{top_two_share}% of the forward pool. The final three profiles "
            "combine for {tail_share}%, so the model’s main divide sits "
            "between transition offense and perimeter skill.",
        ),
        "defense": (
            "Transition pressure clears space",
            "{dominant_name} leads the next defense style by "
            "{dominant_gap} points in a four-profile map. The lone style "
            "outside the top three still takes {tail_share}%, so the season "
            "has a clear leader without erasing its fourth lane.",
        ),
    },
    "20212022": {
        "forwards": (
            "Shot creation becomes the primary identity",
            "{dominant_name} reaches {dominant_share}% and pairs with "
            "{runner_up_name} for {top_two_share}% of forwards. The bottom "
            "three styles share only {tail_share}%, making shot volume and "
            "two-way scoring the model’s central divide.",
        ),
        "defense": (
            "Three profiles—and one overwhelming leader",
            "Because the model learned exactly three defense profiles, the "
            "top-three share is mechanically 100%. The meaningful result is "
            "the imbalance inside that set: {dominant_name} holds "
            "{dominant_share}%, compared with {runner_up_share}% for the "
            "runner-up.",
        ),
    },
    "20222023": {
        "forwards": (
            "Two-way possession leads a layered mix",
            "{dominant_name} holds the largest share at {dominant_share}%, "
            "while {runner_up_name} takes {runner_up_share}%. The bottom three "
            "profiles still combine for {tail_share}%, leaving more texture "
            "than the headline ranking alone suggests.",
        ),
        "defense": (
            "Shot-blocking defense takes majority position",
            "{dominant_name} contains {dominant_share}% of defenders, and the "
            "top two styles combine for {top_two_share}%. The top three reach "
            "{top_three_share}% in a four-profile model, leaving the final "
            "lane almost empty.",
        ),
    },
    "20232024": {
        "forwards": (
            "Puck pressure and defensive contact set the frame",
            "{dominant_name} and {runner_up_name} combine for "
            "{top_two_share}% of forwards. Once the third style is included, "
            "only {tail_share}% remains for the final two profiles, producing "
            "a tightly ordered five-style map.",
        ),
        "defense": (
            "The blue line becomes a two-profile story",
            "{dominant_name} and {runner_up_name} absorb {top_two_share}% of "
            "the defense pool. The third profile lifts the concentration to "
            "{top_three_share}% across four styles, so nearly all of the "
            "season sits inside two main identities.",
        ),
    },
    "20242025": {
        "forwards": (
            "The forward map has no runaway leader",
            "The largest style reaches {dominant_share}%, while "
            "{tail_share}% of forwards sit outside the top three. Six learned "
            "profiles retain meaningful representation, making this a broad "
            "within-season distribution rather than a top-heavy one.",
        ),
        "defense": (
            "Five defense identities remain in play",
            "{dominant_name} leads at {dominant_share}%, but the bottom two "
            "profiles still account for {tail_share}% of defenders. The "
            "distribution has a clear ordering without collapsing into a "
            "two- or three-style map.",
        ),
    },
    "20252026": {
        "forwards": (
            "Puck-dominant scoring pulls away",
            "{dominant_name} and {runner_up_name} together account for "
            "{top_two_share}% of forwards, with the leading style alone above "
            "half. The bottom three share {tail_share}%, so specialist roles "
            "remain visible without setting the season’s center of gravity.",
        ),
        "defense": (
            "The defense pool stays genuinely plural",
            "No defense style reaches 40%, and {tail_share}% of defenders sit "
            "outside the top three. Across five learned profiles, the model "
            "shows a broad distribution with a leader but no controlling "
            "majority.",
        ),
    },
}


def build_season_read(
    season: str,
    group: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    profiles = payload["profiles"]
    if len(profiles) < 2:
        raise RuntimeError(
            f"{season} {group} needs at least two profiles for a season read"
        )

    dominant = profiles[0]
    runner_up = profiles[1]
    player_count = len(payload["players"])
    dominant_share = 100 * int(dominant["count"]) / player_count
    runner_up_share = 100 * int(runner_up["count"]) / player_count
    third_share = (
        100 * int(profiles[2]["count"]) / player_count
        if len(profiles) > 2
        else 0.0
    )
    dominant_gap = dominant_share - runner_up_share
    top_two_share = dominant_share + runner_up_share
    top_three_share = top_two_share + third_share
    after_dominant_share = 100 - dominant_share
    after_top_two_share = 100 - top_two_share
    tail_share = 100 - top_three_share
    profile_count = len(profiles)
    confidence_pct = float(payload["averageConfidence"]) * 100
    mixed_count = int(payload["mixedCount"])
    mixed_share = 100 * mixed_count / player_count if player_count else 0.0
    player_noun = "forwards" if group == "forwards" else "defenders"
    short_style = STYLE_READS.get(
        str(dominant["name"]),
        (
            str(dominant["name"]),
            "The leading profile sets the season's tactical center of gravity.",
        ),
    )[0]

    editorial_headline, editorial_template = SEASON_EDITORIALS.get(
        season,
        {},
    ).get(
        group,
        (
            f"{short_style} leads the season",
            "{dominant_name} holds {dominant_share}% of the group, while "
            "{tail_share}% remains outside the three most common learned "
            "styles.",
        ),
    )
    editorial_context = {
        "dominant_name": str(dominant["name"]),
        "runner_up_name": str(runner_up["name"]),
        "dominant_share": one_decimal(dominant_share),
        "runner_up_share": one_decimal(runner_up_share),
        "third_share": one_decimal(third_share),
        "dominant_gap": one_decimal(dominant_gap),
        "top_two_share": one_decimal(top_two_share),
        "top_three_share": one_decimal(top_three_share),
        "after_dominant_share": one_decimal(after_dominant_share),
        "after_top_two_share": one_decimal(after_top_two_share),
        "tail_share": one_decimal(tail_share),
        "profile_count": profile_count,
    }
    style_group = "forward" if group == "forwards" else "defense"
    factual_paragraph = (
        f"The {season_label(season)} model learned {profile_count} "
        f"{style_group} styles. {dominant['name']} led with "
        f"{int(dominant['count']):,} of {player_count:,} {player_noun} "
        f"({one_decimal(dominant_share)}%), followed by "
        f"{runner_up['name']} at {one_decimal(runner_up_share)}%."
    )
    headline = f"{season_label(season)}: {editorial_headline}"
    editorial_paragraph = editorial_template.format(**editorial_context)

    return {
        "headline": headline,
        "paragraphs": [
            factual_paragraph,
            editorial_paragraph,
        ],
        "facts": [
            {
                "label": "Lead over No. 2",
                "value": f"{one_decimal(dominant_gap)} pts",
            },
            {
                "label": "Average confidence",
                "value": f"{one_decimal(confidence_pct)}%",
            },
            {
                "label": "Mixed profiles",
                "value": f"{mixed_count:,}",
            },
        ],
        "comparison": None,
        "metrics": {
            "dominantGap": float(one_decimal(dominant_gap)),
            "topThreeShare": float(one_decimal(top_three_share)),
            "confidencePct": float(one_decimal(confidence_pct)),
            "mixedShare": float(one_decimal(mixed_share)),
            "profileCount": profile_count,
        },
    }


def build_glossary(
    seasons: list[str],
    maps: dict[str, dict[str, dict[int, str]]],
    all_frames: dict[str, list[pd.DataFrame]],
) -> dict[str, list[dict[str, Any]]]:
    minimum_examples = 4
    season_groups = [
        ("20212022", "20252026"),
        ("20172018", "20202021"),
        ("20122013", "20162017"),
        ("20082009", "20112012"),
    ]
    output: dict[str, list[dict[str, Any]]] = {}
    for group in ("forwards", "defense"):
        variants: dict[str, dict[str, Counter[str]]] = defaultdict(
            lambda: {"high": Counter(), "low": Counter()}
        )
        for season in seasons:
            path = REPORTS_DIR / f"archetype_traits_{group}_{season}.csv"
            if not path.exists():
                continue
            traits = pd.read_csv(path)
            for _, row in traits.iterrows():
                cluster = int(row["cluster"])
                name = maps[group][season].get(cluster)
                if not name:
                    continue
                variants[name]["high"][str(row.get("top_traits", ""))] += 1
                variants[name]["low"][str(row.get("low_traits", ""))] += 1

        candidates: dict[
            str,
            dict[int, dict[int, dict[str, Any]]],
        ] = defaultdict(lambda: defaultdict(dict))
        for frame in all_frames[group]:
            season = str(frame["season"].iloc[0])
            season_map = maps[group][season]
            season_group = next(
                (
                    index
                    for index, (start, end) in enumerate(season_groups)
                    if start <= season <= end
                ),
                None,
            )
            if season_group is None:
                continue
            clusters_by_name: dict[str, list[int]] = defaultdict(list)
            for cluster, name in season_map.items():
                if f"p{cluster}" in frame.columns:
                    clusters_by_name[name].append(cluster)

            for _, row in frame.iterrows():
                player_id = int(row["player_id"])
                games_value = row.get("reg_games", 0)
                games = float(games_value) if pd.notna(games_value) else 0.0
                if games <= 0:
                    continue
                assigned_name = season_map.get(int(row["top_cluster"]))
                probability_weight = max(games, 1.0)

                for name, clusters in clusters_by_name.items():
                    probabilities = [
                        float(row[f"p{cluster}"])
                        for cluster in clusters
                        if pd.notna(row[f"p{cluster}"])
                    ]
                    if not probabilities:
                        continue
                    probability = max(probabilities)
                    candidate = candidates[name][season_group].setdefault(
                        player_id,
                        {
                            "id": player_id,
                            "name": str(row["full_name"]),
                            "games": 0.0,
                            "assignedGames": 0.0,
                            "probabilityTotal": 0.0,
                            "probabilityWeight": 0.0,
                            "maxProbability": 0.0,
                            "seasonGroup": season_group,
                        },
                    )
                    candidate["games"] += games
                    candidate["probabilityTotal"] += (
                        probability * probability_weight
                    )
                    candidate["probabilityWeight"] += probability_weight
                    candidate["maxProbability"] = max(
                        float(candidate["maxProbability"]),
                        probability,
                    )
                    if assigned_name == name:
                        candidate["assignedGames"] += games

        examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for name in variants:
            grouped_candidates = candidates.get(name, {})
            used_ids: set[int] = set()

            def average_probability(player: dict[str, Any]) -> float:
                weight = float(player["probabilityWeight"])
                return (
                    float(player["probabilityTotal"]) / weight
                    if weight
                    else 0.0
                )

            selected: list[dict[str, Any]] = []

            def add_example(player: dict[str, Any]) -> None:
                player_id = int(player["id"])
                if player_id in used_ids:
                    return
                used_ids.add(player_id)
                selected.append(player)

            # Preserve the four-era structure whenever a style has a distinct
            # top-assigned player in that era.
            for season_group in range(len(season_groups)):
                ranked = sorted(
                    (
                        player
                        for player in grouped_candidates.get(
                            season_group,
                            {},
                        ).values()
                        if float(player["assignedGames"]) > 0
                    ),
                    key=lambda player: (
                        -float(player["assignedGames"]),
                        -average_probability(player),
                        -float(player["games"]),
                        str(player["name"]),
                        int(player["id"]),
                    ),
                )
                selected_player = next(
                    (
                        player
                        for player in ranked
                        if int(player["id"]) not in used_ids
                    ),
                    None,
                )
                if selected_player is not None:
                    add_example(selected_player)

            # A style may only exist in one or two eras. Use additional
            # top-assigned players from those eras before considering close
            # probability matches.
            assigned_fallbacks = sorted(
                (
                    player
                    for period in grouped_candidates.values()
                    for player in period.values()
                    if float(player["assignedGames"]) > 0
                    and int(player["id"]) not in used_ids
                ),
                key=lambda player: (
                    int(player["seasonGroup"]),
                    -float(player["assignedGames"]),
                    -average_probability(player),
                    -float(player["games"]),
                    str(player["name"]),
                    int(player["id"]),
                ),
            )
            for player in assigned_fallbacks:
                if len(selected) >= minimum_examples:
                    break
                add_example(player)

            # Very small or short-lived clusters can have fewer than four
            # assigned players. Fill those final slots with the strongest
            # probability-weighted matches from seasons where the style
            # existed.
            probability_fallbacks = sorted(
                (
                    player
                    for period in grouped_candidates.values()
                    for player in period.values()
                    if int(player["id"]) not in used_ids
                ),
                key=lambda player: (
                    -(average_probability(player) * float(player["games"])),
                    -average_probability(player),
                    -float(player["maxProbability"]),
                    int(player["seasonGroup"]),
                    -float(player["games"]),
                    str(player["name"]),
                    int(player["id"]),
                ),
            )
            for player in probability_fallbacks:
                if len(selected) >= minimum_examples:
                    break
                add_example(player)

            if len(selected) < minimum_examples:
                raise RuntimeError(
                    f"{group} style {name!r} has only "
                    f"{len(selected)} unique glossary examples"
                )

            selected.sort(
                key=lambda player: (
                    int(player["seasonGroup"]),
                    -float(player["assignedGames"]),
                    -average_probability(player),
                    -float(player["games"]),
                    str(player["name"]),
                    int(player["id"]),
                )
            )
            examples[name] = [
                {
                    "id": int(player["id"]),
                    "name": str(player["name"]),
                    "games": clean(player["games"], 0),
                }
                for player in selected[:minimum_examples]
            ]

        rows: list[dict[str, Any]] = []
        for name in sorted(variants):
            high_raw = variants[name]["high"].most_common(1)[0][0]
            low_raw = variants[name]["low"].most_common(1)[0][0]
            high = parse_trait_string(high_raw)
            low = parse_trait_string(low_raw)
            rows.append(
                {
                    "name": name,
                    "description": describe_profile(name, high, low),
                    "high": [
                        {
                            "label": readable_trait_label(feature),
                            "z": clean(z, 1),
                        }
                        for feature, z in high[:4]
                    ],
                    "low": [
                        {
                            "label": readable_trait_label(feature),
                            "z": clean(z, 1),
                        }
                        for feature, z in low[:3]
                    ],
                    "examples": examples.get(name, []),
                }
            )
        output[group] = rows
    return output


def switch_rate(frames: list[pd.DataFrame]) -> float | None:
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["player_id", "season"]
    )
    combined["previous"] = combined.groupby("player_id")["top_cluster"].shift(1)
    combined["changed"] = (
        combined["previous"].notna()
        & (combined["top_cluster"].astype(int) != combined["previous"])
    )
    per_player = combined.groupby("player_id").agg(
        seasons=("season", "nunique"),
        switches=("changed", "sum"),
    )
    eligible = per_player[per_player["seasons"] >= 3].copy()
    if eligible.empty:
        return None
    rates = eligible["switches"] / (eligible["seasons"] - 1)
    return clean(float(rates.median()), 2)


def player_record(
    row: pd.Series,
    names: dict[int, str],
) -> dict[str, Any]:
    probability_columns = sorted(
        [
            col
            for col in row.index
            if isinstance(col, str)
            and col.startswith("p")
            and col[1:].isdigit()
        ],
        key=lambda col: int(col[1:]),
    )
    probabilities = sorted(
        [
            {
                "profile": names.get(int(column[1:]), f"Profile {column[1:]}"),
                "value": clean(float(row[column]), 4),
            }
            for column in probability_columns
            if clean(row[column]) is not None
        ],
        key=lambda item: item["value"],
        reverse=True,
    )[:3]
    return {
        "id": int(row["player_id"]),
        "name": str(row["full_name"]),
        "team": str(row.get("teams_played", "")),
        "position": str(row.get("position", "")),
        "games": clean(row.get("reg_games"), 0),
        "goals": clean(row.get("reg_goals"), 0),
        "assists": clean(row.get("reg_assists"), 0),
        "points": clean(row.get("reg_points"), 0),
        "shots": clean(row.get("reg_shots"), 0),
        "toi": clean(row.get("reg_avg_toi_min"), 1),
        "plusMinus": clean(row.get("reg_plus_minus"), 0),
        "pim": clean(row.get("reg_pim"), 0),
        "playoffGames": clean(row.get("po_games"), 0),
        "playoffPoints": clean(row.get("po_points"), 0),
        "cluster": int(row["top_cluster"]),
        "profile": names.get(int(row["top_cluster"]), "Unlabeled profile"),
        "confidence": clean(float(row["confidence"]), 4),
        "probabilities": probabilities,
    }


def playoff_records(
    seasons: list[str],
    maps: dict[str, dict[str, dict[int, str]]],
    frames_by_key: dict[tuple[str, str], pd.DataFrame],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group in ("forwards", "defense"):
        for season in seasons:
            projection_path = (
                DATA_DIR
                / f"playoff_archetype_projection_{group}_{season}.parquet"
            )
            if not projection_path.exists():
                continue
            projection = pd.read_parquet(projection_path)
            players = frames_by_key[(group, season)]
            stats = players.set_index("player_id")
            names = maps[group][season]
            for _, row in projection.iterrows():
                player_id = int(row["player_id"])
                if player_id not in stats.index:
                    continue
                player = stats.loc[player_id]
                if isinstance(player, pd.DataFrame):
                    player = player.iloc[0]
                reg_cluster = int(row["reg_top_cluster"])
                playoff_cluster = int(row["po_top_cluster"])
                reg_games = float(player.get("reg_games", 0) or 0)
                playoff_games = float(player.get("po_games", 0) or 0)
                records.append(
                    {
                        "season": season,
                        "group": group,
                        "id": player_id,
                        "name": str(player["full_name"]),
                        "team": str(player.get("teams_played", "")),
                        "position": str(player.get("position", "")),
                        "regGames": clean(reg_games, 0),
                        "playoffGames": clean(playoff_games, 0),
                        "regProfile": names.get(reg_cluster, f"Profile {reg_cluster}"),
                        "playoffProfile": names.get(
                            playoff_cluster,
                            f"Profile {playoff_cluster}",
                        ),
                        "regConfidence": clean(row.get("reg_confidence"), 4),
                        "playoffConfidence": clean(row.get("po_confidence"), 4),
                        "distance": clean(row.get("probability_distance"), 4),
                        "changed": bool(row.get("archetype_changed", False)),
                        "regPpg": clean(
                            float(player.get("reg_points", 0) or 0) / reg_games
                            if reg_games
                            else 0,
                            3,
                        ),
                        "playoffPpg": clean(
                            float(player.get("po_points", 0) or 0) / playoff_games
                            if playoff_games
                            else 0,
                            3,
                        ),
                        "regToi": clean(player.get("reg_avg_toi_min"), 2),
                        "playoffToi": clean(player.get("po_avg_toi_min"), 2),
                    }
                )
    return records


def main() -> None:
    seasons = available_seasons()
    maps = profile_maps(seasons)
    all_frames: dict[str, list[pd.DataFrame]] = {
        "forwards": [],
        "defense": [],
    }
    frames_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    season_payload: dict[str, dict[str, Any]] = {}
    confidence_trend: list[dict[str, Any]] = []
    unique_ids: set[int] = set()
    career_records: list[dict[str, Any]] = []
    player_season_count = 0

    for season in seasons:
        season_payload[season] = {}
        trend_row: dict[str, Any] = {
            "season": season,
            "label": season_label(season),
        }
        for group in ("forwards", "defense"):
            frame = pd.read_parquet(
                DATA_DIR / f"players_{group}_{season}.parquet"
            ).copy()
            frame["season"] = season
            frames_by_key[(group, season)] = frame
            all_frames[group].append(frame)
            unique_ids.update(int(value) for value in frame["player_id"])
            names = maps[group][season]
            players = [
                player_record(row, names)
                for _, row in frame.sort_values(
                    ["reg_points", "confidence"],
                    ascending=False,
                ).iterrows()
            ]
            player_season_count += len(players)
            career_records.extend(
                {
                    "season": season,
                    "group": group,
                    "id": record["id"],
                    "name": record["name"],
                    "team": record["team"],
                    "position": record["position"],
                    "games": record["games"],
                    "points": record["points"],
                    "toi": record["toi"],
                    "profile": record["profile"],
                    "confidence": record["confidence"],
                }
                for record in players
            )
            profile_counts = Counter(record["profile"] for record in players)
            season_payload[season][group] = {
                "players": players,
                "profiles": [
                    {
                        "name": name,
                        "count": count,
                        "share": clean(100 * count / len(players), 4),
                    }
                    for name, count in profile_counts.most_common()
                ],
                "averageConfidence": clean(
                    float(frame["confidence"].mean()),
                    4,
                ),
                "mixedCount": int((frame["confidence"] < 0.8).sum()),
            }
            trend_row[group] = clean(
                float(frame["confidence"].mean()) * 100,
                1,
            )
        confidence_trend.append(trend_row)

    generated_reads: list[str] = []
    for season in sorted(seasons):
        for group in ("forwards", "defense"):
            payload = season_payload[season][group]
            payload["seasonRead"] = build_season_read(
                season,
                group,
                payload,
            )
            generated_reads.append(
                " ".join(payload["seasonRead"]["paragraphs"])
            )

    expected_reads = len(seasons) * 2
    if (
        len(generated_reads) != expected_reads
        or len(set(generated_reads)) != expected_reads
    ):
        raise RuntimeError(
            "Season reads must be present and unique for every season and group"
        )

    confidence_trend.sort(key=lambda row: row["season"])
    profile_definition_counts = {
        group: sum(len(maps[group][season]) for season in seasons)
        for group in ("forwards", "defense")
    }
    playoffs = playoff_records(seasons, maps, frames_by_key)
    glossary = build_glossary(seasons, maps, all_frames)
    latest_season = seasons[0]
    latest_season_breakdown = {
        group: len(frames_by_key[(group, latest_season)])
        for group in ("forwards", "defense")
    }
    average_model_confidence = clean(
        float(
            pd.concat(
                [*all_frames["forwards"], *all_frames["defense"]],
                ignore_index=True,
            )["confidence"].mean()
        )
        * 100,
        1,
    )

    core_payload = {
        "meta": {
            "generated": pd.Timestamp.now(tz="UTC").isoformat(),
            "seasons": [
                {"key": season, "label": season_label(season)}
                for season in seasons
            ],
            "seasonCount": len(seasons),
            "playerCount": len(unique_ids),
            "playerSeasonCount": player_season_count,
            "profileDefinitions": profile_definition_counts,
            "namedStyleCount": sum(len(rows) for rows in glossary.values()),
            "namedStyleBreakdown": {
                group: len(glossary[group])
                for group in ("forwards", "defense")
            },
            "latestSeasonPlayerCount": sum(latest_season_breakdown.values()),
            "latestSeasonBreakdown": latest_season_breakdown,
            "averageModelConfidence": average_model_confidence,
            "switchRates": {
                group: switch_rate(all_frames[group])
                for group in ("forwards", "defense")
            },
            "confidenceTrend": confidence_trend,
        },
        "glossary": glossary,
    }

    data_output = WEB_DIR / "data"
    seasons_output = data_output / "seasons"
    seasons_output.mkdir(parents=True, exist_ok=True)
    outputs = {
        data_output / "core.json": core_payload,
        data_output / "careers.json": career_records,
        data_output / "playoffs.json": playoffs,
    }
    outputs.update(
        {
            seasons_output / f"{season}.json": payload
            for season, payload in season_payload.items()
        }
    )

    for output_path, output_payload in outputs.items():
        output_path.write_text(
            json.dumps(
                output_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        print(
            f"Wrote {output_path.relative_to(ROOT)} "
            f"({output_path.stat().st_size / 1024 / 1024:.2f} MB)"
        )


if __name__ == "__main__":
    main()
