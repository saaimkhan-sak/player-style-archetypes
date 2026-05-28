from __future__ import annotations

import re
from typing import Iterable


TraitToken = tuple[str, float]
ARCHETYPE_LABEL_VERSION = "role-names-v2"

PROFILE_COLOR_MAP: dict[str, tuple[str, str]] = {
    "High-Volume Playmaking Scorer": ("#2563EB", "#FFFFFF"),
    "Low-Contact Scorer": ("#F97316", "#111827"),
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

PROFILE_ORDER = list(PROFILE_COLOR_MAP.keys())


def canonical_profile_name(name: str) -> str:
    replacements = {
        "Shot-Creating Playmaker": "High-Volume Playmaking Scorer",
        "Setup Playmaker": "High-Volume Playmaking Scorer",
        "Shot-Volume Scorer": "Low-Contact Scorer",
        "Volume Shooter": "Low-Contact Scorer",
        "Finisher": "Low-Contact Scorer",
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
        "Low-Contact Scoring Profile": "Low-Contact Scorer",
        "Shooting / Scoring Profile": "Low-Contact Scorer",
    }
    text = str(name)
    if text in PROFILE_COLOR_MAP:
        return text
    return replacements.get(text, text)


def profile_colors(name: str) -> tuple[str, str]:
    return PROFILE_COLOR_MAP.get(canonical_profile_name(name), ("#E5E7EB", "#111827"))


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


def _fallback_role_name(cluster: int, high_tokens: list[TraitToken], low_features: set[str]) -> tuple[str, str]:
    high_features = {feature for feature, _ in high_tokens}
    categories = _ordered_categories(high_tokens)

    if _has(high_features, ["reg_shots_per60", "reg_points_per60", "reg_goals_per60"]):
        if "reg_assists_per60" in high_features:
            return "Shot-Creating Playmaker", "Creates offense through both shot volume and setup play, with scoring chances flowing through the puck on their stick."
        if "reg_shots_per60" in high_features and _has(high_features, ["reg_points_per60", "reg_goals_per60"]):
            return "Shot-Volume Scorer", "Generates offense by getting pucks on net and converting that shot volume into goals or points."
        if "reg_shots_per60" in high_features:
            return "Volume Shooter", "Looks to create through shot generation first, even when the broader scoring profile is less extreme."
        return "Finisher", "Stands out most through goal and point production rather than physical play or defensive usage."

    if "reg_assists_per60" in high_features:
        return "Setup Playmaker", "Creates value primarily by setting up teammates and driving assisted offense."
    if "reg_takeaways_per60" in high_features:
        return "Puck Hunter", "Stands out by pressuring puck carriers and turning defensive pressure into regained possessions."
    if "reg_giveaways_per60" in high_features:
        return "High-Touch Puck Mover", "Handles the puck often enough to create plays, with more turnover risk attached to that responsibility."
    if "reg_hits_per60" in high_features and _has(low_features, ["reg_points_per60", "reg_goals_per60", "reg_assists_per60"]):
        return "Checking Forward", "Impacts games through contact and forechecking pressure more than scoring output."
    if "reg_hits_per60" in high_features:
        return "Physical Disruptor", "Creates separation through contact and pressure, adding value away from pure scoring."
    if "reg_blocked_shots_per60" in high_features:
        return "Shot Suppressor", "Absorbs defensive minutes and blocks shots, with value showing up in prevention work."
    if "reg_pim_per60" in high_features:
        return "Penalty-Drawn Edge Player", "Plays an abrasive style where penalties are part of the statistical footprint."
    if "reg_pp_share" in high_features:
        return "Power-Play Specialist", "Receives offensive-zone and power-play usage, so production is tied to scoring-role deployment."
    if "reg_pk_share" in high_features:
        return "Penalty-Kill Specialist", "Leans into shorthanded and defensive usage rather than scoring-role deployment."
    if _has(high_features, ["reg_fo_taken_per_game", "reg_fo_pct"]):
        return "Role-Center Specialist", "Shows up through deployment details like draws, matchups, and role minutes."

    if categories:
        return f"Balanced {categories[0]} Contributor", "Has a real statistical signature, but it is more moderate than the extreme archetypes."
    return f"Balanced Role Contributor {cluster}", "Does not lean heavily into one boxscore trait, so the cluster reads as a balanced role."


def _build_defense_name_summary(cluster: int, high_features: set[str], low_features: set[str], high_tokens: list[TraitToken]) -> tuple[str, str]:
    offense_hi = _has(high_features, ["reg_points_per60", "reg_goals_per60", "reg_assists_per60", "reg_shots_per60"])
    playmaking_hi = "reg_assists_per60" in high_features
    shooting_hi = "reg_shots_per60" in high_features
    blocks_hi = "reg_blocked_shots_per60" in high_features
    hits_hi = "reg_hits_per60" in high_features
    pim_hi = "reg_pim_per60" in high_features
    takeaways_hi = "reg_takeaways_per60" in high_features
    giveaways_hi = "reg_giveaways_per60" in high_features
    giveaways_lo = "reg_giveaways_per60" in low_features
    pk_hi = "reg_pk_share" in high_features
    pp_hi = "reg_pp_share" in high_features

    if pim_hi and giveaways_hi:
        return "High-Event Physical Defenseman", "High-event defense profile: plays physically and handles enough puck touches to carry added turnover risk."
    if blocks_hi and hits_hi:
        return "Shot-Blocking Defensive Defenseman", "Defense-first profile: blocks shots, plays through contact, and absorbs hard minutes."
    if pim_hi and (hits_hi or blocks_hi):
        return "Physical Shutdown Defenseman", "Defense profile built around contact, crease-area resistance, and a higher-penalty edge."
    if takeaways_hi and (giveaways_lo or offense_hi):
        return "Puck-Pressure Transition Defenseman", "Transition defender: pressures puck carriers and turns recoveries into clean exits."
    if giveaways_hi and offense_hi:
        return "Transition Risk/Reward Defenseman", "High-touch defense profile: moves the puck often, with turnover risk attached."
    if offense_hi and (playmaking_hi or shooting_hi):
        return "Offensive Puck-Moving Defenseman", "Blue-line offense driver: creates through point shots, exits, and puck movement."
    if pk_hi and not pp_hi:
        return "Penalty-Kill Defensive Defenseman", "Shorthanded defense profile: value is tied to defensive usage and penalty-kill minutes."
    if pp_hi and not pk_hi:
        return "Point-Usage Power-Play Defenseman", "Power-play defense profile: offense is tied to point usage and special-teams deployment."
    if blocks_hi and not offense_hi:
        return "Shot-Blocking Defensive Defenseman", "Stay-at-home profile: suppression, blocked shots, and defensive-zone work drive the role."
    if hits_hi and not offense_hi:
        return "Physical Shutdown Defenseman", "Physical defense profile: contact and disruption matter more than puck production."
    if offense_hi:
        return "Low-Event Puck-Moving Defenseman", "Puck-moving defense profile with offense showing up without a heavy-contact footprint."

    categories = _ordered_categories(high_tokens)
    if categories:
        return "Defensive Role Defenseman", "Role-driven defense profile whose statistical lean is moderate rather than extreme."
    return f"Defensive Role Defenseman {cluster}", "Balanced defense profile without one dominant statistical trait."


def build_archetype_name_summary(
    cluster: int,
    high_tokens: list[TraitToken],
    low_tokens: list[TraitToken],
    group: str = "forwards",
) -> tuple[str, str]:
    high_features = {feature for feature, _ in high_tokens}
    low_features = {feature for feature, _ in low_tokens}

    if group == "defense":
        return _build_defense_name_summary(cluster, high_features, low_features, high_tokens)

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
        return "Shot-Blocking Contact Specialist", "Low-offense profile built around blocked shots and defensive minutes."
    if hits_hi and not offense_hi:
        return "Checking-Line Disruptor", "Physical depth role: contact and disruption matter more than scoring."
    if scoring_hi and not hits_hi:
        return "Low-Contact Scorer", "Skill-leaning scorer: creates offense without much physical play."
    if takeaways_hi and scoring_hi:
        return "Puck-Pressure Two-Way Creator", "Creates offense while pressuring puck carriers."
    if giveaways_hi and playmaking_hi:
        return "High-Touch Risk/Reward Playmaker", "High-event puck profile: creates plays while carrying turnover risk."

    name, summary = _fallback_role_name(cluster, high_tokens, low_features)
    return canonical_profile_name(name), summary
