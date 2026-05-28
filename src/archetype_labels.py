from __future__ import annotations

import re
from typing import Iterable


TraitToken = tuple[str, float]


def parse_trait_string(value: str) -> list[TraitToken]:
    if not isinstance(value, str):
        return []
    out: list[TraitToken] = []
    for part in value.split(","):
        part = part.strip()
        match = re.match(r"^([A-Za-z0-9_]+)\(([+-]?\d+\.?\d*)\)$", part)
        if match:
            out.append((match.group(1), float(match.group(2))))
    return out


def _has(features: set[str], names: Iterable[str]) -> bool:
    return any(name in features for name in names)


def _ordered_categories(tokens: list[TraitToken]) -> list[str]:
    categories = []
    for feature, _ in tokens:
        if feature in {"reg_points_per60", "reg_goals_per60"}:
            categories.append("Scoring")
        elif feature == "reg_shots_per60":
            categories.append("Shooting")
        elif feature == "reg_assists_per60":
            categories.append("Playmaking")
        elif feature == "reg_hits_per60":
            categories.append("Contact")
        elif feature == "reg_blocked_shots_per60":
            categories.append("Shot-Blocking")
        elif feature == "reg_pim_per60":
            categories.append("Penalty-Prone")
        elif feature == "reg_takeaways_per60":
            categories.append("Puck-Pressure")
        elif feature == "reg_giveaways_per60":
            categories.append("High-Risk Puck-Handling")
        elif feature == "reg_pp_share":
            categories.append("Power-Play Usage")
        elif feature == "reg_pk_share":
            categories.append("Penalty-Kill Usage")
        elif feature in {"reg_fo_taken_per_game", "reg_fo_pct"}:
            categories.append("Faceoff")

    deduped = []
    for category in categories:
        if category not in deduped:
            deduped.append(category)
    return deduped


def _low_context(low_features: set[str]) -> str:
    offense = {"reg_points_per60", "reg_goals_per60", "reg_assists_per60", "reg_shots_per60"}
    physical = {"reg_hits_per60", "reg_blocked_shots_per60", "reg_pim_per60"}
    puck_risk = {"reg_giveaways_per60"}

    if _has(low_features, offense):
        return "Defensive"
    if _has(low_features, physical):
        return "Low-Contact"
    if _has(low_features, puck_risk):
        return "Controlled"
    return "Balanced"


def build_archetype_name_summary(cluster: int, high_tokens: list[TraitToken], low_tokens: list[TraitToken]) -> tuple[str, str]:
    high_features = {feature for feature, _ in high_tokens}
    low_features = {feature for feature, _ in low_tokens}

    offense_hi = _has(high_features, ["reg_points_per60", "reg_goals_per60", "reg_assists_per60", "reg_shots_per60"])
    playmaking_hi = "reg_assists_per60" in high_features
    shooting_hi = "reg_shots_per60" in high_features
    scoring_hi = _has(high_features, ["reg_points_per60", "reg_goals_per60"])
    blocks_hi = "reg_blocked_shots_per60" in high_features
    hits_hi = "reg_hits_per60" in high_features
    pim_hi = "reg_pim_per60" in high_features
    takeaways_hi = "reg_takeaways_per60" in high_features
    giveaways_hi = "reg_giveaways_per60" in high_features
    giveaways_lo = "reg_giveaways_per60" in low_features
    pk_hi = "reg_pk_share" in high_features
    pp_hi = "reg_pp_share" in high_features
    fo_hi = _has(high_features, ["reg_fo_taken_per_game", "reg_fo_pct"])

    if pim_hi and hits_hi:
        return "Agitating Heavy-Contact Forward", "High-contact profile: delivers hits and takes more penalties."
    if blocks_hi and hits_hi:
        return "Shot-Blocking Contact Specialist", "Defense-tilted profile: blocks shots and plays physically."
    if offense_hi and playmaking_hi and shooting_hi:
        return "High-Volume Playmaking Scorer", "Offense driver: generates shots and assists at high rates."
    if takeaways_hi and giveaways_lo:
        return "Puck-Pressure Two-Way Creator", "Pressure-and-recover profile: creates takeaways while limiting giveaways."
    if fo_hi and not offense_hi:
        return "Deployment / Role Specialist", "Deployment-driven: reflects coach usage, draws, and role minutes."
    if pk_hi and not pp_hi:
        return "PK-Leaning Defensive Role", "Shorthanded-leaning: value shows up in defensive usage."
    if pp_hi and not pk_hi:
        return "PP-Leaning Offensive Role", "Power-play leaning: production is driven by scoring-role deployment."
    if blocks_hi and not offense_hi:
        return "Defense-First Shot Suppressor", "Low-offense profile built around blocked shots and defensive minutes."
    if hits_hi and not offense_hi:
        return "Checking-Line Contact Profile", "Physical depth profile: contact and disruption matter more than scoring."
    if scoring_hi and not hits_hi:
        return "Low-Contact Scoring Profile", "Skill-leaning profile: offense without much physical play."
    if takeaways_hi and scoring_hi:
        return "Puck-Pressure Scoring Profile", "Creates offense while pressuring puck carriers."
    if giveaways_hi and playmaking_hi:
        return "High-Touch Risk/Reward Playmaker", "High-event puck profile: creates plays while carrying turnover risk."

    categories = _ordered_categories(high_tokens)
    if not categories:
        return f"Balanced Role Profile {cluster}", "Balanced profile: no single trait dominates the cluster."

    context = _low_context(low_features)
    if len(categories) == 1:
        name = f"{context} {categories[0]} Profile"
    else:
        name = f"{categories[0]} / {categories[1]} Profile"

    summary = "Blended profile: combines the listed high traits rather than fitting a single extreme role."
    return name, summary
