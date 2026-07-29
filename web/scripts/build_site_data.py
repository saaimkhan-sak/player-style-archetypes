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

import numpy as np
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


def minute_clock(value: Any) -> str:
    numeric = 0.0
    try:
        if not pd.isna(value):
            numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    total_seconds = int(round(numeric * 60))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


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


def format_trait_detail(
    tokens: list[tuple[str, float]],
    max_items: int,
) -> str:
    pieces = []
    for feature, z_score in tokens[:max_items]:
        direction = "higher" if z_score >= 0 else "lower"
        pieces.append(
            f"{readable_trait_label(feature)} "
            f"({direction}, {z_score:+.1f}σ)"
        )
    return "; ".join(pieces) or "None"


def need_finder_metadata(
    season: str,
    group: str,
    names: dict[int, str],
) -> dict[str, Any]:
    path = REPORTS_DIR / f"archetype_traits_{group}_{season}.csv"
    traits = pd.read_csv(path) if path.exists() else pd.DataFrame()
    details: dict[str, dict[str, Any]] = {}
    target_order: list[str] = []
    target_clusters: dict[str, int] = {}

    if not traits.empty:
        for _, row in traits.sort_values("cluster").iterrows():
            cluster = int(row["cluster"])
            high = parse_trait_string(row.get("top_traits", ""))
            low = parse_trait_string(row.get("low_traits", ""))
            generated_name, summary = build_archetype_name_summary(
                cluster,
                high,
                low,
                group=group,
            )
            name = names.get(
                cluster,
                canonical_profile_name(generated_name),
            )
            details[str(cluster)] = {
                "name": name,
                "summary": summary,
                "higher": format_trait_detail(high, 5),
                "lower": format_trait_detail(low, 4),
            }
            if name not in target_clusters:
                target_order.append(name)
            # Streamlit's {display_name: k} target map keeps the last raw
            # cluster when multiple learned clusters share a display name.
            target_clusters[name] = cluster
    else:
        for cluster, name in sorted(names.items()):
            details[str(cluster)] = {
                "name": name,
                "summary": "",
                "higher": "None",
                "lower": "None",
            }
            if name not in target_clusters:
                target_order.append(name)
            target_clusters[name] = cluster

    return {
        "targets": [
            {
                "profile": name,
                "cluster": target_clusters[name],
            }
            for name in target_order
        ],
        "details": details,
    }


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

# These reads interpret only the within-season profile mix. They describe role
# and statistical identity, not player quality, causal change, or year-over-year
# movement. Blocks, hits, and giveaways are treated as contextual signals rather
# than automatic evidence of defense, forechecking success, or carelessness.
SEASON_EDITORIALS: dict[
    str,
    dict[str, tuple[str, tuple[str, str]]],
] = {
    "20082009": {
        "forwards": (
            "Speed sets the table; playmaking makes the next play",
            (
                "{dominant_name} is the clearest attacking identity in this "
                "pool. Its shot and possession-continuation signals read like "
                "a role built to push pace and create before five defenders "
                "can settle into shape.",
                "{runner_up_name} adds a connective layer through assists and "
                "takeaways, giving the attack another route to its next chance. "
                "{third_name} supplies a defensive-detail counterweight, so the "
                "forward pool separates creation from support instead of asking "
                "every player to solve the same problem.",
            ),
        ),
        "defense": (
            "The blue line splits between surviving and advancing",
            (
                "{dominant_name} describes a block-heavy workload under "
                "sustained shot pressure. That is a role signal, not proof of "
                "better defense: the opponent has to own enough of the shift "
                "for those blocks to accumulate.",
                "The roster-construction story is the combined presence of "
                "{runner_up_name} and {third_name}. One adds production from "
                "the back end; the other links recoveries to the next play, "
                "creating natural complements for a stay-at-home partner.",
            ),
        ),
    },
    "20092010": {
        "forwards": (
            "The puck runs through the creators",
            (
                "{dominant_name} carries the signature of offensive "
                "responsibility: scoring, frequent involvement and more failed "
                "plays as the price of attempting difficult ones. The giveaway "
                "signal is better read as workload than carelessness.",
                "{runner_up_name} offers a more connective version of creation, "
                "with assists and takeaways doing more of the separating. With "
                "{third_name} carrying more block-and-exposure work, the pool "
                "shows how creators and support roles divide the burden of a "
                "shift.",
            ),
        ),
        "defense": (
            "No single defense job owns the blue line",
            (
                "{dominant_name} and {runner_up_name} are almost level, but "
                "they describe opposite shift economies. One tries to move "
                "play out of danger; the other is identified by the work that "
                "happens when danger has already arrived.",
                "{third_name} prevents this from becoming a simple mover-versus-"
                "stopper split. In pairing terms, the season offers several "
                "ways to distribute puck advancement, shooting-lane work and "
                "offensive responsibility.",
            ),
        ),
    },
    "20102011": {
        "forwards": (
            "Inside finishing becomes the common scoring identity",
            (
                "The named-profile map compresses heavily around "
                "{dominant_name}. Rebound, high-danger and finishing signals "
                "make second-chance offense the clearest separator, even if "
                "the broad label contains more than one kind of player.",
                "{runner_up_name} and {third_name} describe much of the labor "
                "around that offense—blocking lanes, disrupting touches and "
                "keeping shifts competitive. The hockey lesson is "
                "complementarity, not that three quarters of the league played "
                "an identical crease game.",
            ),
        ),
        "defense": (
            "The blue line needs an anchor and an outlet",
            (
                "{dominant_name} is the largest workload family, marked by "
                "blocks and repeated defensive-zone involvement. It tells us "
                "what those defenders were asked to absorb, not whether the "
                "team spent enough time on offense.",
                "{runner_up_name} and {third_name} supply two different "
                "outlets: measured distribution and a higher-variance offensive "
                "role. A useful pair can divide those jobs instead of asking "
                "one defender to absorb pressure and handle every puck-"
                "advancement touch.",
            ),
        ),
    },
    "20112012": {
        "forwards": (
            "Rush creation leads; the support jobs are defensive",
            (
                "{dominant_name} reads as the primary offensive lane: "
                "shot volume and extended attacking sequences with less of "
                "the profile devoted to physical work.",
                "{runner_up_name} is better understood as defensive detail "
                "than pure contact, while {third_name} carries more puck and "
                "creation risk. That division gives a coach three distinct "
                "tools—pace, recovery work and high-touch offense—rather than "
                "one generic forward type.",
            ),
        ),
        "defense": (
            "Pressure matters only if the next pass works",
            (
                "{dominant_name} leads a genuinely plural blue-line map. Its "
                "takeaway and continuation signals suggest defenders whose "
                "value proposition begins with recovering the puck and doing "
                "something useful with the next touch.",
                "{runner_up_name} absorbs more shooting-lane work, while "
                "{third_name} adds creation from the back end. Together they "
                "form a recognizable pairing architecture: recover, protect, "
                "then advance.",
            ),
        ),
    },
    "20122013": {
        "forwards": (
            "Skill lives outside the paint; support keeps it playable",
            (
                "In the lockout-shortened snapshot, {dominant_name} dominates "
                "the named profile mix. The underlying combination of goals, "
                "shots and lower inside-shot concentration points to offense "
                "created from space rather than a roster living at the crease.",
                "{runner_up_name} is the important counterbalance: takeaways "
                "and playmaking travel with the scoring. When the first attack "
                "comes from the outside, that second layer helps a line recover "
                "the puck and create another decision.",
            ),
        ),
        "defense": (
            "A pressure-first back end tries to end defense quickly",
            (
                "{dominant_name} combines takeaways with possession "
                "continuation, a profile that reads less like passive "
                "containment and more like ending the defensive phase with a "
                "useful next play.",
                "{runner_up_name} reflects the shifts that last long enough to "
                "require blocks; {third_name} reflects the defenders who can "
                "tilt play after the recovery. One manages the emergency, the "
                "other two try to prevent the next one.",
            ),
        ),
    },
    "20132014": {
        "forwards": (
            "The cycle is not possession for possession’s sake",
            (
                "{dominant_name} carries a credible sustained-pressure "
                "fingerprint: rebounds, continued offensive-zone play and a "
                "favorable share of the chance environment. The point is to "
                "make the defense survive another rotation, not simply hold "
                "the puck along the wall.",
                "{runner_up_name} reaches a similar territorial outcome with "
                "more direct scoring and playmaking. {third_name} supplies "
                "defensive detail beneath those lanes, giving the roster both "
                "pressure players and the shifts that earn them another start.",
            ),
        ),
        "defense": (
            "Puck movement carries one shift; shot blocking carries another",
            (
                "{dominant_name} is the clearest advancement profile, built "
                "around distribution and a healthier on-ice chance balance. "
                "{runner_up_name} is the mirror image: more blocks, more "
                "defensive exposure and less event creation.",
                "The contrast between {dominant_name} and {runner_up_name} is "
                "the useful hockey read. A blue line needs defenders who can "
                "shorten the trip through the neutral zone and defenders who "
                "can stabilize a shift when that trip never starts; the labels "
                "describe different jobs, not a value ranking.",
            ),
        ),
    },
    "20142015": {
        "forwards": (
            "The forward pool is split between pace and resistance",
            (
                "{dominant_name} owns the clearest attacking brief. Its shot "
                "and continuation signals read like a role built to carry "
                "possession into the next phase and generate attempts before "
                "the defense can reset.",
                "{runner_up_name} is less a hitting identity than a "
                "defensive-detail one, with blocks and takeaways doing the "
                "separating. {third_name} supplies the higher-touch bridge "
                "between those poles, which is the kind of role balance a "
                "coach can turn into coherent lines.",
            ),
        ),
        "defense": (
            "Transport and containment divide the workload",
            (
                "{dominant_name} and {runner_up_name} form a sharp "
                "specialization spectrum: one is associated with assists, "
                "points and team offense; the other with blocks and repeated "
                "defensive-zone labor.",
                "{third_name} fills most of the remaining space as a lower-"
                "creation role. The blue-line question is not which label is "
                "best in isolation, but whether each pair has a way to move the "
                "puck when containment finally wins it back.",
            ),
        ),
    },
    "20152016": {
        "forwards": (
            "Puck responsibility separates creators from support",
            (
                "{dominant_name} combines scoring, possession extension and "
                "giveaways—the familiar statistical footprint of players "
                "trusted to try difficult things with the puck and carry more "
                "of the attack.",
                "{runner_up_name} creates through a more connective blend of "
                "assists, takeaways and chance quality. {third_name} carries "
                "more of the block-and-support workload, so the forward pool "
                "looks built around who drives the decision and who makes the "
                "next decision possible.",
            ),
        ),
        "defense": (
            "The best defense starts with the puck moving north",
            (
                "{dominant_name} is the largest blue-line family, pairing "
                "production and takeaways with a favorable chance share. In "
                "hockey terms, the role is to turn a recovery into progression "
                "rather than settle for a blind clear.",
                "{runner_up_name} and {third_name} cover the other half of the "
                "job: shooting-lane work, contact and the physical cost of "
                "absorbing pressure. High block volume still describes exposure "
                "as much as effectiveness, which is why a puck mover matters "
                "beside it.",
            ),
        ),
    },
    "20162017": {
        "forwards": (
            "High-touch offense needs a stabilizer",
            (
                "{dominant_name} is the season’s majority identity, with "
                "transition shot volume and failed plays rising together. "
                "That is what offensive burden often looks like in a box score: "
                "more creation attempts also mean more ways a possession can "
                "end.",
                "{runner_up_name} adds higher-quality playmaking, while "
                "{third_name} brings takeaways and assists into the scoring "
                "mix. The roster logic is clear—surround the high-touch driver "
                "with players who can extend or repair the play.",
            ),
        ),
        "defense": (
            "Several routes to the next possession coexist",
            (
                "{dominant_name} and {runner_up_name} sit almost level, but "
                "the useful distinction is not simply safety versus aggression. "
                "Both labels contain active puck involvement, and the broad "
                "names should not be mistaken for two rigid systems.",
                "{third_name} adds a clearer production-and-distribution lane. "
                "The hockey read is a plural blue line: different defenders "
                "influence the next possession through individual chance "
                "creation, recovery pressure or puck movement.",
            ),
        ),
    },
    "20172018": {
        "forwards": (
            "Puck pressure and creation collapse into one broad job",
            (
                "{dominant_name} absorbs an extraordinary share of the named "
                "profile map. That should be read as label compression, not a "
                "claim that four out of five forwards played identical hockey; "
                "takeaways, chance share and continuation simply traveled "
                "together often enough to crowd the other labels.",
                "{runner_up_name} provides the sharper contrast: more frequent "
                "puck responsibility, special-situation creation and failed "
                "touches. The distinction is less who works and who creates "
                "than how much of the offense runs through one player.",
            ),
        ),
        "defense": (
            "The broad label hides a sharper blue-line contrast",
            (
                "{dominant_name} contains several very different kinds of "
                "defender, so its headline share should not be treated as one "
                "uniform stay-at-home identity or one repeatable tactical "
                "assignment.",
                "The cleaner hockey comparison sits underneath it: "
                "{runner_up_name} carries the block-heavy, low-advancement "
                "footprint, while {third_name} carries the puck-moving one. "
                "Those are the roles a pairing decision can actually balance.",
            ),
        ),
    },
    "20182019": {
        "forwards": (
            "There is no single recipe for a useful forward",
            (
                "{dominant_name} and {runner_up_name} put skill first, but in "
                "different ways: one scores from space, the other mixes shot "
                "threat with distribution and recovery.",
                "The next layer is just as revealing. {third_name} and "
                "{fourth_name} form a substantial support tier, separating "
                "disruption and defensive detail from primary creation. That "
                "is a lineup economy with specialized jobs, not one prototype "
                "repeated four times.",
            ),
        ),
        "defense": (
            "Puck movement leads a three-part blue line",
            (
                "{dominant_name} carries the most complete play-driving "
                "signature: distribution, team chance share and the willingness "
                "to accept some puck risk in exchange for moving the attack "
                "forward.",
                "{runner_up_name} and {third_name} divide much of the remaining "
                "work between blocks and lower-event responsibility. The "
                "practical challenge is to keep enough puck advancement on "
                "every pair that one bad shift does not become two.",
            ),
        ),
    },
    "20192020": {
        "forwards": (
            "Transition and possession offer two paths into attack",
            (
                "{dominant_name} and {third_name} form the clearest tactical "
                "poles in the interrupted season: create in motion or hold the "
                "puck long enough to extend the sequence.",
                "{runner_up_name} fills the space between them, but its name "
                "should not be treated as proof of constant crease work; the "
                "interior signal is modest here. The stronger roster read is "
                "that transport, possession and finishing responsibility were "
                "distributed across different players.",
            ),
        ),
        "defense": (
            "The blue line is almost a binary choice",
            (
                "{dominant_name} and {runner_up_name} create one of the cleanest "
                "stylistic splits in the dataset. One is associated with "
                "distribution and favorable team chances; the other with blocks "
                "and lower point involvement.",
                "{third_name} adds a smaller, more confrontational lane. The "
                "pairing question is how much puck-advancement responsibility "
                "to place beside a defender whose statistical identity is built "
                "around blocks and contact.",
            ),
        ),
    },
    "20202021": {
        "forwards": (
            "Rush creation is the clearest offensive lane",
            (
                "Within a short, division-only schedule, {dominant_name} is the "
                "most distinct attacking signature. Its shot and continuation "
                "profile reads like quick-strike offense designed to attack "
                "before the defense can reset.",
                "{runner_up_name} supplies a more patient skill lane. Beneath "
                "them, {third_name} is best read as an exposure-and-workload "
                "profile—not proof of successful suppression—while the support "
                "tier absorbs the lower-event minutes around the creators.",
            ),
        ),
        "defense": (
            "Pressure turns defense into the first pass",
            (
                "{dominant_name} is the largest of four meaningful blue-line "
                "jobs. Its takeaway and advancement signals point toward "
                "defenders expected to convert pressure into the next "
                "possession rather than merely end the current threat.",
                "{runner_up_name} and {third_name} carry more own-zone exposure "
                "and blocking work; {fourth_name} adds a cleaner play-driving "
                "lane. The mix is about distributing responsibility, not "
                "crowning one universal defense type.",
            ),
        ),
    },
    "20212022": {
        "forwards": (
            "Shot creation drives the first full-season attack",
            (
                "In the first 82-game schedule after two shortened seasons, "
                "{dominant_name} becomes the central offensive identity. Shots, "
                "expected goals, rebounds and continued zone time all point to "
                "players who manufacture attempts rather than wait for one "
                "perfect look.",
                "{runner_up_name} adds takeaways and playmaking to that offense, "
                "while {third_name} carries more high-touch variance. The "
                "hockey tradeoff is familiar: volume creates pressure, but "
                "another player still has to recover the puck and make the "
                "next chance better.",
            ),
        ),
        "defense": (
            "One broad defense label; one clearly active counterweight",
            (
                "The consolidated profile map is unusually compressed around "
                "{dominant_name}. Several distinct statistical groups collapse "
                "into that one name, so its share describes a broad family of "
                "roles rather than one quiet, low-creation tactical job.",
                "{runner_up_name} is the useful counterweight, carrying more "
                "takeaways and production from the back end. A coach can "
                "concentrate that activation on selected pairs without asking "
                "the entire blue line to play at the same risk level.",
            ),
        ),
    },
    "20222023": {
        "forwards": (
            "Territory, defensive detail and inside scoring share the attack",
            (
                "{dominant_name} is best understood through the lower rate of "
                "opponent attempts and chances attached to the group. The "
                "identity is territorial: spend less of the shift defending "
                "and give the offense another turn.",
                "{runner_up_name} supplies block-and-support work rather than "
                "a pure contact story, while {third_name} owns the clearest "
                "rebound and high-danger fingerprint. Those jobs form a useful "
                "sequence—stabilize the shift, tilt the ice, then finish inside.",
            ),
        ),
        "defense": (
            "Shot blocking is the workload; transition is the escape",
            (
                "{dominant_name} is the majority identity, defined by blocks "
                "and heavy shot exposure. That can be valuable labor, but it "
                "also means the opponent had enough possession to make the "
                "block necessary.",
                "{runner_up_name} offers the countermeasure: take the puck away "
                "and attach offense to the recovery. In pairing terms, the "
                "contrast is between absorbing shot pressure and getting the "
                "shift moving in the other direction.",
            ),
        ),
    },
    "20232024": {
        "forwards": (
            "Pressure is useful only when it becomes the next chance",
            (
                "{dominant_name} and {runner_up_name} are broad statistical "
                "umbrellas, not literal descriptions of two clean systems. The "
                "first combines puck-pressure signals with heavier defensive "
                "exposure; the second looks more like takeaways and chance "
                "quality than raw contact.",
                "{third_name} is the sharpest specialty in the pool, with "
                "rebounds, high-danger attempts and continued zone play. The "
                "hockey read is a forward group that separates recovery, "
                "support and inside finishing rather than asking one archetype "
                "to own the whole sequence.",
            ),
        ),
        "defense": (
            "The blue line splits between driving play and absorbing it",
            (
                "{dominant_name} and {runner_up_name} form a clean "
                "advancement-versus-workload contrast. Distribution and team "
                "chance share sit on one side; blocks and lower transition "
                "involvement sit on the other.",
                "{third_name} remains a small physical specialty rather than "
                "the season’s organizing principle. The practical roster "
                "question is whether each pair has enough puck movement to "
                "keep shot-blocking from becoming its default state.",
            ),
        ),
    },
    "20242025": {
        "forwards": (
            "A modern forward corps is a portfolio, not a prototype",
            (
                "{dominant_name} leads, but three other profiles occupy "
                "substantial parts of the lineup. The clearest contrast is "
                "between scoring from space and the territorial contribution "
                "captured by {fourth_name}.",
                "{runner_up_name} and {third_name} describe separate support "
                "lanes, although their names are stronger than the direct "
                "contact evidence underneath them. The useful hockey insight "
                "is role coverage: creation, disruption, defensive detail and "
                "shot-share work all need roster space.",
            ),
        ),
        "defense": (
            "Puck movement comes in several risk settings",
            (
                "{dominant_name} leads a blue-line spectrum that runs from "
                "shot-and-chance exposure to raw production and active rush "
                "involvement. The broad label is a starting point, not a "
                "complete description of the defenders inside it.",
                "{runner_up_name} carries the clearer scoring signature, while "
                "{third_name} is quieter and lower-contact rather than a proven "
                "transition driver. {fourth_name} is the more assertive play-"
                "driving option, whether through extending possessions or "
                "adding dangerous offense. Roster construction is about "
                "choosing the right setting for each pair.",
            ),
        ),
    },
    "20252026": {
        "forwards": (
            "High-touch creation sets the attack; balance makes the line work",
            (
                "{dominant_name} owns the majority of the forward map. The "
                "combination of scoring, extended possessions and giveaways "
                "describes offensive burden: these players are asked to make "
                "more plays, including the difficult ones that sometimes fail.",
                "{runner_up_name} adds takeaways and playmaking, but it should "
                "not be mistaken for automatic defensive suppression. Its value "
                "in this mix is connective: a second decision-maker who can keep "
                "the attack from stalling around one high-touch creator.",
            ),
        ),
        "defense": (
            "The blue line keeps several answers in the bag",
            (
                "{dominant_name} is only a plurality, and its blocks reflect "
                "own-zone labor as much as defensive success. "
                "{runner_up_name} supplies the clearest production-and-"
                "distribution alternative.",
                "{third_name} is too broad to carry a literal defensive claim, "
                "while {fourth_name} owns the more credible physical specialty. "
                "The season’s hockey story is specialization: one pair can "
                "advance, another can absorb pressure and a third can impose "
                "contact without any single job swallowing the blue line.",
            ),
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
    third = profiles[2] if len(profiles) > 2 else runner_up
    fourth = profiles[3] if len(profiles) > 3 else third
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
    short_style = STYLE_READS.get(
        str(dominant["name"]),
        (
            str(dominant["name"]),
            "The leading profile sets the season's tactical center of gravity.",
        ),
    )[0]

    editorial_headline, editorial_templates = SEASON_EDITORIALS.get(
        season,
        {},
    ).get(
        group,
        (
            f"{short_style} leads the season",
            (
                "{dominant_name} is the clearest within-season role family, "
                "but its share describes statistical identity rather than "
                "player quality.",
                "{runner_up_name} provides the counterweight, giving the "
                "roster another way to distribute responsibility.",
            ),
        ),
    )
    editorial_context = {
        "dominant_name": str(dominant["name"]),
        "runner_up_name": str(runner_up["name"]),
        "third_name": str(third["name"]),
        "fourth_name": str(fourth["name"]),
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
    headline = f"{season_label(season)}: {editorial_headline}"
    editorial_paragraphs = [
        template.format(**editorial_context)
        for template in editorial_templates
    ]

    return {
        "headline": headline,
        "paragraphs": editorial_paragraphs,
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
    target_scores = [
        clean(float(row[column]) * 100, 1)
        if clean(row[column]) is not None
        else 0.0
        for column in probability_columns
    ]
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
        "regAtoi": minute_clock(row.get("reg_avg_toi_min")),
        "plusMinus": clean(row.get("reg_plus_minus"), 0),
        "pim": clean(row.get("reg_pim"), 0),
        "playoffGames": clean(row.get("po_games"), 0),
        "playoffToi": clean(row.get("po_avg_toi_min"), 3),
        "playoffAtoi": minute_clock(row.get("po_avg_toi_min")),
        "playoffPoints": clean(row.get("po_points"), 0),
        "playoffGoals": clean(row.get("po_goals"), 0),
        "playoffAssists": clean(row.get("po_assists"), 0),
        "playoffShots": clean(row.get("po_shots"), 0),
        "playoffPlusMinus": clean(row.get("po_plus_minus"), 0),
        "playoffPim": clean(row.get("po_pim"), 0),
        "cluster": int(row["top_cluster"]),
        "profile": names.get(int(row["top_cluster"]), "Unlabeled profile"),
        "confidence": clean(float(row["confidence"]), 4),
        "probabilities": probabilities,
        "targetScores": target_scores,
        "needOrder": int(row.get("_need_order", 0)),
    }


def career_record(
    row: pd.Series,
    record: dict[str, Any],
    season: str,
    group: str,
) -> dict[str, Any]:
    confidence = float(row.get("confidence", 0) or 0)
    return {
        "season": season,
        "group": group,
        "id": record["id"],
        "name": record["name"],
        "team": record["team"],
        "position": record["position"],
        "profile": record["profile"],
        "confidence": record["confidence"],
        "confidencePct": clean(confidence * 100, 1),
        "mixedness": clean(1.0 - confidence, 3),
        "games": record["games"],
        "regAtoi": record["regAtoi"],
        "points": record["points"],
        "goals": record["goals"],
        "assists": record["assists"],
        "shots": record["shots"],
        "plusMinus": record["plusMinus"],
        "pim": record["pim"],
        "playoffGames": record["playoffGames"],
        "playoffAtoi": record["playoffAtoi"],
        "playoffPoints": record["playoffPoints"],
        "playoffGoals": record["playoffGoals"],
        "playoffAssists": record["playoffAssists"],
        "playoffShots": record["playoffShots"],
        "playoffPlusMinus": record["playoffPlusMinus"],
        "playoffPim": record["playoffPim"],
    }


def surname_key(name: Any) -> str:
    parts = str(name or "").replace(".", "").replace("'", "").split()
    return parts[-1].lower() if parts else ""


def team_codes_value(value: Any) -> list[str]:
    return [
        team.strip()
        for team in str(value or "").split("/")
        if team.strip()
    ]


def weighted_probability(
    frame: pd.DataFrame,
    column: str,
) -> float:
    if frame.empty:
        return 0.0
    weights = (
        pd.to_numeric(frame["reg_toi_total"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        + 1e-9
    )
    values = (
        pd.to_numeric(frame[column], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    return float(np.average(values, weights=weights))


def team_constructions(
    season: str,
    group: str,
    frame: pd.DataFrame,
    names: dict[int, str],
    line_data: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    probability_columns = sorted(
        [
            column
            for column in frame.columns
            if isinstance(column, str)
            and column.startswith("p")
            and column[1:].isdigit()
        ],
        key=lambda column: int(column[1:]),
    )
    if not probability_columns:
        return {}

    clusters = [int(column[1:]) for column in probability_columns]
    teams = sorted(
        {
            team
            for value in frame["teams_played"].dropna()
            for team in team_codes_value(value)
        }
    )
    team_frames = {
        team: frame[
            frame["teams_played"]
            .fillna("")
            .map(lambda value, code=team: code in team_codes_value(value))
        ].copy()
        for team in teams
    }

    # League-context metrics use every model-eligible player associated with
    # the team, matching the Streamlit tab rather than the selected depth chart.
    metric_rows: list[dict[str, Any]] = []
    for team, team_frame in team_frames.items():
        team_frame["reg_toi_total"] = (
            pd.to_numeric(
                team_frame["reg_avg_toi_min"],
                errors="coerce",
            ).fillna(0.0)
            * pd.to_numeric(
                team_frame["reg_games"],
                errors="coerce",
            ).fillna(0.0)
        )
        weights = team_frame["reg_toi_total"].to_numpy(dtype=float) + 1e-9
        weight_total = float(weights.sum())
        for cluster, column in zip(clusters, probability_columns):
            probabilities = (
                pd.to_numeric(team_frame[column], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            share = (
                float(np.average(probabilities, weights=weights))
                if len(probabilities)
                else 0.0
            )
            strong = probabilities >= 0.60
            coverage = (
                float(weights[strong].sum() / weight_total)
                if weight_total > 0
                else 0.0
            )
            contributions = weights * probabilities
            contribution_total = float(contributions.sum())
            concentration = (
                float(np.sort(contributions)[-2:].sum() / contribution_total)
                if contribution_total > 0
                else 1.0
            )
            metric_rows.append(
                {
                    "team": team,
                    "cluster": cluster,
                    "share": share,
                    "coverage": coverage,
                    "concentration": concentration,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        baseline = metrics.groupby("cluster", as_index=False).agg(
            mean_share=("share", "mean"),
            std_share=("share", "std"),
        )
        metrics = metrics.merge(baseline, on="cluster", how="left")
        metrics["coverage_rank"] = metrics.groupby("cluster")[
            "coverage"
        ].rank(pct=True)
        metrics["concentration_rank"] = metrics.groupby("cluster")[
            "concentration"
        ].rank(pct=True)
        valid_std = metrics["std_share"].replace({0: np.nan})
        metrics["z"] = (
            (metrics["share"] - metrics["mean_share"]) / valid_std
        ).fillna(0.0)
        metrics["risk"] = (
            -metrics["z"]
            + np.maximum(0, 0.35 - metrics["coverage_rank"]) * 2.0
            + np.maximum(0, metrics["concentration_rank"] - 0.75) * 1.5
        )

    slot_size = 3 if group == "forwards" else 2
    roster_size = 12 if group == "forwards" else 8
    top_size = 6 if group == "forwards" else 4
    unit_label = "Line" if group == "forwards" else "Pair"
    top_label = "Top 6" if group == "forwards" else "Top 4"
    bottom_label = "Bottom 6" if group == "forwards" else "Bottom 4"
    combination_position = "line" if group == "forwards" else "pairing"
    relevant_lines = pd.DataFrame()
    if not line_data.empty:
        relevant_lines = line_data[
            (line_data["season_key"].astype(str) == str(season))
            & (
                line_data["position"].astype(str)
                == combination_position
            )
        ].copy()

    output: dict[str, dict[str, Any]] = {}
    for team, team_frame in team_frames.items():
        if team_frame.empty:
            continue
        base_roster = team_frame.copy()
        base_roster["reg_toi_total"] = (
            pd.to_numeric(
                base_roster["reg_avg_toi_min"],
                errors="coerce",
            ).fillna(0.0)
            * pd.to_numeric(
                base_roster["reg_games"],
                errors="coerce",
            ).fillna(0.0)
        )
        base_roster = base_roster.sort_values(
            ["reg_toi_total", "confidence"],
            ascending=False,
        ).reset_index(drop=True)
        base_roster["last_key"] = base_roster["full_name"].map(
            surname_key
        )

        combo_frame = pd.DataFrame()
        if not relevant_lines.empty:
            combo_frame = relevant_lines[
                relevant_lines["playerTeam"].astype(str) == str(team)
            ].sort_values("toi_min", ascending=False)

        used_last: set[str] = set()
        roster_rows: list[pd.Series] = []
        unit_cards: dict[int, dict[str, Any]] = {}
        if not combo_frame.empty:
            for _, combo in combo_frame.iterrows():
                combo_names = [
                    player.strip()
                    for player in str(combo["name"]).split("-")
                    if player.strip()
                ]
                if len(combo_names) != slot_size:
                    continue
                keys = [surname_key(player) for player in combo_names]
                if any(key in used_last for key in keys):
                    continue
                matched: list[pd.Series] = []
                matched_ids: set[int] = set()
                for key in keys:
                    candidates = base_roster[
                        (base_roster["last_key"] == key)
                        & (~base_roster["last_key"].isin(used_last))
                        & (~base_roster["player_id"].isin(matched_ids))
                    ]
                    if candidates.empty:
                        break
                    player = candidates.iloc[0].copy()
                    matched.append(player)
                    matched_ids.add(int(player["player_id"]))
                if len(matched) != slot_size:
                    continue
                unit = len(unit_cards) + 1
                for player in matched:
                    player["Unit"] = unit
                    roster_rows.append(player)
                    used_last.add(str(player["last_key"]))
                unit_cards[unit] = {
                    "minutes": clean(combo.get("toi_min"), 1),
                    "xgPct": clean(combo.get("xg_pct"), 4),
                }
                if len(unit_cards) == roster_size // slot_size:
                    break

        if len(roster_rows) < roster_size:
            remaining = base_roster[
                ~base_roster["last_key"].isin(used_last)
            ]
            for _, player in remaining.iterrows():
                player = player.copy()
                player["Unit"] = len(roster_rows) // slot_size + 1
                roster_rows.append(player)
                used_last.add(str(player["last_key"]))
                if len(roster_rows) == roster_size:
                    break

        if not roster_rows:
            continue
        roster = (
            pd.DataFrame(roster_rows)
            .head(roster_size)
            .reset_index(drop=True)
        )
        roster["Depth"] = roster.index + 1
        roster["Unit"] = roster["Unit"].astype(int)
        roster["Archetype"] = roster["top_cluster"].map(
            lambda cluster: names.get(
                int(cluster),
                f"Archetype {int(cluster)}",
            )
        )

        shares = {
            cluster: weighted_probability(roster, column)
            for cluster, column in zip(clusters, probability_columns)
        }
        dominant_cluster = max(shares, key=shares.get)
        dominant_name = names.get(
            dominant_cluster,
            f"Archetype {dominant_cluster}",
        )
        top_half = roster.head(top_size)
        bottom_half = roster.iloc[top_size:roster_size]
        top_share = weighted_probability(
            top_half,
            f"p{dominant_cluster}",
        )
        bottom_share = weighted_probability(
            bottom_half,
            f"p{dominant_cluster}",
        )

        grouped_mix: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "overall": 0.0,
                "top": 0.0,
                "bottom": 0.0,
            }
        )
        for cluster, column in zip(clusters, probability_columns):
            profile_name = names.get(cluster, f"Archetype {cluster}")
            grouped_mix[profile_name]["overall"] += shares[cluster] * 100
            grouped_mix[profile_name]["top"] += (
                weighted_probability(top_half, column) * 100
            )
            grouped_mix[profile_name]["bottom"] += (
                weighted_probability(bottom_half, column) * 100
            )
        mix = sorted(
            [
                {
                    "profile": profile_name,
                    "overall": clean(values["overall"], 1),
                    "top": clean(values["top"], 1),
                    "bottom": clean(values["bottom"], 1),
                }
                for profile_name, values in grouped_mix.items()
            ],
            key=lambda row: float(row["overall"] or 0),
            reverse=True,
        )

        units: list[dict[str, Any]] = []
        for unit, unit_frame in roster.groupby("Unit", sort=True):
            profile_counts = unit_frame["Archetype"].value_counts()
            largest_count = int(profile_counts.max())
            unit_profile = sorted(
                [
                    str(profile_name)
                    for profile_name, count in profile_counts.items()
                    if int(count) == largest_count
                ]
            )[0]
            combination = unit_cards.get(int(unit))
            players = []
            for _, player in unit_frame.iterrows():
                players.append(
                    {
                        "id": int(player["player_id"]),
                        "name": str(player["full_name"]),
                        "team": team,
                        "position": str(player.get("position", "")),
                        "games": clean(player.get("reg_games"), 0),
                        "atoi": clean(
                            player.get("reg_avg_toi_min"),
                            3,
                        ),
                        "goals": clean(player.get("reg_goals"), 0),
                        "assists": clean(
                            player.get("reg_assists"),
                            0,
                        ),
                        "points": clean(player.get("reg_points"), 0),
                        "profile": str(player["Archetype"]),
                        "confidence": clean(
                            player.get("confidence"),
                            4,
                        ),
                        "depth": int(player["Depth"]),
                    }
                )
            units.append(
                {
                    "number": int(unit),
                    "profile": unit_profile,
                    "minutes": (
                        combination["minutes"]
                        if combination
                        else clean(
                            float(
                                pd.to_numeric(
                                    unit_frame["reg_toi_total"],
                                    errors="coerce",
                                ).fillna(0.0).sum()
                            ),
                            1,
                        )
                    ),
                    "xgPct": (
                        combination["xgPct"]
                        if combination
                        else None
                    ),
                    "fromCombination": bool(combination),
                    "players": players,
                }
            )

        gaps: list[dict[str, Any]] = []
        if not metrics.empty:
            team_gaps = metrics[
                metrics["team"] == team
            ].sort_values("risk", ascending=False)
            for _, row in team_gaps.iterrows():
                z_score = float(row["z"])
                coverage_rank = float(row["coverage_rank"])
                concentration_rank = float(row["concentration_rank"])
                note = ""
                if z_score < -0.75 or (
                    z_score < -0.5 and coverage_rank < 0.35
                ):
                    note = "Underrepresented"
                elif (
                    concentration_rank > 0.75
                    and coverage_rank < 0.5
                ):
                    note = "Thin coverage"
                cluster = int(row["cluster"])
                gaps.append(
                    {
                        "profile": names.get(
                            cluster,
                            f"Archetype {cluster}",
                        ),
                        "teamShare": clean(
                            float(row["share"]) * 100,
                            1,
                        ),
                        "leagueAverage": clean(
                            float(row["mean_share"]) * 100,
                            1,
                        ),
                        "zScore": clean(z_score, 2),
                        "strongCoverage": clean(
                            float(row["coverage"]) * 100,
                            1,
                        ),
                        "topTwoReliance": clean(
                            float(row["concentration"]) * 100,
                            1,
                        ),
                        "note": note,
                    }
                )

        output[team] = {
            "team": team,
            "unitLabel": unit_label,
            "topLabel": top_label,
            "bottomLabel": bottom_label,
            "source": (
                "MoneyPuck 5v5 line/pairing minutes"
                if unit_cards
                else "regular-season player TOI fallback"
            ),
            "usesMoneyPuck": bool(unit_cards),
            "hasFallbackUnits": any(
                not bool(unit["fromCombination"])
                for unit in units
            ),
            "dominant": {
                "profile": dominant_name,
                "overall": clean(shares[dominant_cluster] * 100, 1),
                "top": clean(top_share * 100, 1),
                "bottom": clean(bottom_share * 100, 1),
                "gap": clean(
                    abs(top_share - bottom_share) * 100,
                    1,
                ),
            },
            "units": units,
            "mix": mix,
            "gaps": gaps,
        }
    return output


def playoff_records(
    seasons: list[str],
    maps: dict[str, dict[str, dict[int, str]]],
    frames_by_key: dict[tuple[str, str], pd.DataFrame],
) -> list[dict[str, Any]]:
    def numeric(value: Any) -> float:
        try:
            return 0.0 if pd.isna(value) else float(value)
        except (TypeError, ValueError):
            return 0.0

    def rate(numerator: Any, denominator: float) -> float:
        numeric_denominator = numeric(denominator)
        return (
            numeric(numerator) / numeric_denominator
            if numeric_denominator
            else 0.0
        )

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
                reg_games = numeric(player.get("reg_games", 0))
                playoff_games = numeric(player.get("po_games", 0))
                reg_ppg = rate(player.get("reg_points", 0), reg_games)
                playoff_ppg = rate(player.get("po_points", 0), playoff_games)
                reg_shots_per_game = rate(
                    player.get("reg_shots", 0),
                    reg_games,
                )
                playoff_shots_per_game = rate(
                    player.get("po_shots", 0),
                    playoff_games,
                )
                reg_pim_per_game = rate(
                    player.get("reg_pim", 0),
                    reg_games,
                )
                playoff_pim_per_game = rate(
                    player.get("po_pim", 0),
                    playoff_games,
                )
                reg_plus_minus_per_game = rate(
                    player.get("reg_plus_minus", 0),
                    reg_games,
                )
                playoff_plus_minus_per_game = rate(
                    player.get("po_plus_minus", 0),
                    playoff_games,
                )
                reg_toi = numeric(player.get("reg_avg_toi_min", 0))
                playoff_toi = numeric(player.get("po_avg_toi_min", 0))
                probability_distance = numeric(
                    row.get("probability_distance", 0)
                )
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
                        "distance": clean(probability_distance, 4),
                        "shiftBand": (
                            "Held steady"
                            if probability_distance <= 0.25
                            else (
                                "Moderate shift"
                                if probability_distance <= 0.75
                                else "Major shift"
                            )
                        ),
                        "changed": bool(row.get("archetype_changed", False)),
                        "regPpg": clean(reg_ppg, 4),
                        "playoffPpg": clean(playoff_ppg, 4),
                        "ppgChange": clean(playoff_ppg - reg_ppg, 8),
                        "regToi": clean(reg_toi, 2),
                        "playoffToi": clean(playoff_toi, 2),
                        "toiChange": clean(playoff_toi - reg_toi, 8),
                        "shotRateChange": clean(
                            playoff_shots_per_game - reg_shots_per_game,
                            8,
                        ),
                        "pimRateChange": clean(
                            playoff_pim_per_game - reg_pim_per_game,
                            8,
                        ),
                        "plusMinusRateChange": clean(
                            playoff_plus_minus_per_game
                            - reg_plus_minus_per_game,
                            8,
                        ),
                    }
                )

    shift_fields = [
        "ppgChange",
        "shotRateChange",
        "toiChange",
        "pimRateChange",
        "plusMinusRateChange",
    ]
    population: dict[str, list[float]] = {
        field: [] for field in shift_fields
    }
    for players in frames_by_key.values():
        for _, player in players.iterrows():
            reg_games = numeric(player.get("reg_games", 0))
            playoff_games = numeric(player.get("po_games", 0))
            reg_ppg = rate(player.get("reg_points", 0), reg_games)
            playoff_ppg = rate(player.get("po_points", 0), playoff_games)
            reg_shots_per_game = rate(
                player.get("reg_shots", 0),
                reg_games,
            )
            playoff_shots_per_game = rate(
                player.get("po_shots", 0),
                playoff_games,
            )
            reg_pim_per_game = rate(
                player.get("reg_pim", 0),
                reg_games,
            )
            playoff_pim_per_game = rate(
                player.get("po_pim", 0),
                playoff_games,
            )
            reg_plus_minus_per_game = rate(
                player.get("reg_plus_minus", 0),
                reg_games,
            )
            playoff_plus_minus_per_game = rate(
                player.get("po_plus_minus", 0),
                playoff_games,
            )
            has_both_splits = reg_games > 0 and playoff_games > 0
            population["ppgChange"].append(
                playoff_ppg - reg_ppg if has_both_splits else 0.0
            )
            population["shotRateChange"].append(
                playoff_shots_per_game - reg_shots_per_game
                if has_both_splits
                else 0.0
            )
            population["toiChange"].append(
                numeric(player.get("po_avg_toi_min", 0))
                - numeric(player.get("reg_avg_toi_min", 0))
            )
            population["pimRateChange"].append(
                playoff_pim_per_game - reg_pim_per_game
                if has_both_splits
                else 0.0
            )
            population["plusMinusRateChange"].append(
                playoff_plus_minus_per_game
                - reg_plus_minus_per_game
                if has_both_splits
                else 0.0
            )

    centers: dict[str, float] = {}
    spreads: dict[str, float] = {}
    for field in shift_fields:
        values = np.array(population[field], dtype=float)
        centers[field] = float(values.mean())
        spreads[field] = float(values.std(ddof=0))

    for record in records:
        stat_shift = math.sqrt(
            sum(
                (
                    (
                        float(record.get(field, 0) or 0)
                        - centers[field]
                    )
                    / spreads[field]
                )
                ** 2
                for field in shift_fields
                if math.isfinite(spreads[field])
                and spreads[field] != 0
            )
        )
        record["statShift"] = clean(stat_shift, 4)
        record["statBand"] = (
            "Held steady"
            if stat_shift <= 2
            else ("Moderate shift" if stat_shift <= 3.5 else "Major shift")
        )
    return records


def main() -> None:
    seasons = available_seasons()
    maps = profile_maps(seasons)
    line_path = DATA_DIR / "line_combinations.parquet"
    line_data = (
        pd.read_parquet(line_path)
        if line_path.exists()
        else pd.DataFrame()
    )
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
            frame["_need_order"] = np.arange(len(frame))
            frame["season"] = season
            frames_by_key[(group, season)] = frame
            all_frames[group].append(frame)
            unique_ids.update(int(value) for value in frame["player_id"])
            names = maps[group][season]
            ordered_frame = frame.sort_values(
                ["reg_points", "confidence"],
                ascending=False,
            )
            players: list[dict[str, Any]] = []
            for _, row in ordered_frame.iterrows():
                record = player_record(row, names)
                players.append(record)
                career_records.append(
                    career_record(row, record, season, group)
                )
            player_season_count += len(players)
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
                "needFinder": need_finder_metadata(
                    season,
                    group,
                    names,
                ),
                "teamConstructions": team_constructions(
                    season,
                    group,
                    frame,
                    names,
                    line_data,
                ),
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
