from __future__ import annotations

import re
from typing import Iterable


TraitToken = tuple[str, float]
ARCHETYPE_LABEL_VERSION = "role-names-v2"

PROFILE_COLOR_MAP: dict[str, tuple[str, str]] = {
    "High-Volume Playmaking Scorer": ("#BFDBFE", "#1E3A8A"),
    "Perimeter Skill Scorer": ("#FED7AA", "#7C2D12"),
    "Shot-Creation Scorer": ("#FBCFE8", "#831843"),
    "Two-Way Skill Scorer": ("#DCFCE7", "#14532D"),
    "High-Touch Risk/Reward Scorer": ("#FAE8FF", "#701A75"),
    "Shot-Blocking Contact Specialist": ("#A5F3FC", "#164E63"),
    "Agitating Heavy-Contact Forward": ("#FECACA", "#7F1D1D"),
    "Puck-Pressure Two-Way Creator": ("#CFFAFE", "#155E75"),
    "Interior Net-Front Finisher": ("#99F6E4", "#134E4A"),
    "Rush / Transition Chance Creator": ("#DDD6FE", "#4C1D95"),
    "Possession-Carrying Forward": ("#E0E7FF", "#3730A3"),
    "Cycle Pressure Play-Driver": ("#A7F3D0", "#064E3B"),
    "Two-Way Shot-Share Driver": ("#BAE6FD", "#0C4A6E"),
    "Suppression Workload Forward": ("#F1F5F9", "#334155"),
    "Shutdown Suppression Center": ("#C7D2FE", "#312E81"),
    "Deployment / Role Specialist": ("#E2E8F0", "#334155"),
    "PP-Leaning Offensive Role": ("#FDE68A", "#78350F"),
    "PK-Leaning Defensive Role": ("#C4B5FD", "#312E81"),
    "High-Touch Risk/Reward Playmaker": ("#E9D5FF", "#581C87"),
    "Checking-Line Disruptor": ("#FECDD3", "#881337"),
    "Physical Shutdown Defenseman": ("#FCA5A5", "#7F1D1D"),
    "Shot-Blocking Defensive Defenseman": ("#67E8F9", "#164E63"),
    "Offensive Puck-Moving Defenseman": ("#93C5FD", "#1E3A8A"),
    "Low-Event Puck-Moving Defenseman": ("#FEF3C7", "#78350F"),
    "Point-Usage Power-Play Defenseman": ("#FCD34D", "#78350F"),
    "Penalty-Kill Defensive Defenseman": ("#A5B4FC", "#312E81"),
    "Transition Risk/Reward Defenseman": ("#E879F9", "#701A75"),
    "Defensive Role Defenseman": ("#CBD5E1", "#334155"),
    "Puck-Pressure Transition Defenseman": ("#D9F99D", "#365314"),
    "Crease-Clearing Suppression Defenseman": ("#5EEAD4", "#134E4A"),
    "Play-Driving Puck-Moving Defenseman": ("#7DD3FC", "#0C4A6E"),
    "High-Event Physical Defenseman": ("#FDA4AF", "#881337"),
}

PROFILE_ORDER = list(PROFILE_COLOR_MAP.keys())


def canonical_profile_name(name: str) -> str:
    replacements = {
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
        "Low-Contact Scoring Profile": "Perimeter Skill Scorer",
        "Shooting / Scoring Profile": "Shot-Creation Scorer",
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


TRAIT_LABELS: dict[str, str] = {
    "reg_points_per60": "Points per 60",
    "reg_goals_per60": "Goals per 60",
    "reg_assists_per60": "Assists per 60",
    "reg_shots_per60": "Shots on goal per 60",
    "reg_hits_per60": "Hits per 60",
    "reg_blocked_shots_per60": "Blocked shots per 60",
    "reg_takeaways_per60": "Takeaways per 60",
    "reg_giveaways_per60": "Giveaways per 60",
    "reg_pim_per60": "Penalty minutes per 60",
    "reg_pp_share": "Power-play usage",
    "reg_pk_share": "Penalty-kill usage",
    "reg_fo_pct": "Faceoff win rate",
    "reg_fo_taken_per_game": "Faceoffs per game",
    "mp_reg_5on5_I_F_highDangerShots_per60": "High-danger shots per 60",
    "mp_reg_5on5_I_F_highDangerShotShare": "Share of shots from high-danger areas",
    "mp_reg_5on5_I_F_xGoals_per60": "Individual expected goals per 60",
    "mp_reg_5on5_I_F_xGoalsPerAttempt": "Expected goals per shot attempt",
    "mp_reg_5on5_I_F_rebounds_per60": "Rebound chances per 60",
    "mp_reg_5on5_I_F_reboundxGoals_per60": "Expected goals from rebounds per 60",
    "mp_reg_5on5_I_F_playContinuedInZone_per60": "Offensive-zone possessions extended per 60",
    "mp_reg_5on5_I_F_playContinuedOutsideZone_per60": "Rush possessions extended per 60",
    "mp_reg_5on5_OnIce_xGoalsPercentage_calc": "On-ice expected-goal share",
    "mp_reg_5on5_OnIce_F_xGoals_per60": "Team expected goals while on ice per 60",
    "mp_reg_5on5_OnIce_A_xGoals_per60": "Expected goals against while on ice per 60",
    "mp_reg_5on5_OnIce_A_shotAttempts_per60": "Shot attempts against while on ice per 60",
    "mp_reg_4on5_OnIce_A_xGoals_per60": "Penalty-kill expected goals against per 60",
    "mp_reg_5on5_shotsBlockedByPlayer_per60": "Shot blocks per 60",
    "mp_reg_5on5_penaltiesDrawn_per60": "Penalties drawn per 60",
    "mp_reg_5on4_I_F_xGoals_per60": "Power-play expected goals per 60",
}


def readable_trait_label(feature: str) -> str:
    if feature in TRAIT_LABELS:
        return TRAIT_LABELS[feature]
    text = re.sub(r"^(mp_)?reg_", "", str(feature))
    text = text.replace("5on5_", "").replace("5on4_", "power_play_").replace("4on5_", "penalty_kill_")
    text = text.replace("OnIce_", "on_ice_").replace("I_F_", "individual_")
    text = text.replace("_per60", "_per_60").replace("_calc", "")
    return text.replace("_", " ").title()


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
        elif feature in {"mp_reg_5on5_I_F_highDangerShots_per60", "mp_reg_5on5_I_F_highDangerShotShare", "mp_reg_5on5_I_F_rebounds_per60", "mp_reg_5on5_I_F_reboundxGoals_per60"}:
            categories.append("Net-Front")
        elif feature in {"mp_reg_5on5_I_F_playContinuedInZone_per60", "mp_reg_5on5_I_F_playContinuedOutsideZone_per60"}:
            categories.append("Puck-Carrying")
        elif feature in {"mp_reg_5on5_OnIce_xGoalsPercentage_calc", "mp_reg_5on5_OnIce_F_xGoals_per60"}:
            categories.append("Play-Driving")
        elif feature in {"mp_reg_5on5_OnIce_A_xGoals_per60", "mp_reg_5on5_OnIce_A_shotAttempts_per60", "mp_reg_4on5_OnIce_A_xGoals_per60"}:
            categories.append("Suppression")

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
        if categories[0] == "Suppression":
            return "Suppression Workload Forward", "Defensive-workload profile: the clearest signal is time spent absorbing shots and expected chances against."
        if categories[0] == "Puck-Carrying":
            return "Possession-Carrying Forward", "Puck-carrying profile: keeps plays moving through carries and continued possessions more than net-front finishing."
        high_text = ", ".join(readable_trait_label(feature).lower() for feature, _ in high_tokens[:2])
        low_text = ", ".join(readable_trait_label(feature).lower() for feature in list(low_features)[:2])
        detail = f"Leans most toward {high_text}"
        if low_text:
            detail += f", with less emphasis on {low_text}"
        return f"Balanced {categories[0]} Contributor", f"{detail}."
    top_text = ", ".join(readable_trait_label(feature).lower() for feature, _ in high_tokens[:2])
    return f"Balanced Role Contributor {cluster}", f"Blended role profile with its clearest signals in {top_text or 'usage and secondary production'}."


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
    netfront_hi = _has(high_features, ["mp_reg_5on5_shotsBlockedByPlayer_per60", "mp_reg_5on5_OnIce_A_xGoals_per60", "mp_reg_4on5_OnIce_A_xGoals_per60"])
    xg_share_hi = "mp_reg_5on5_OnIce_xGoalsPercentage_calc" in high_features
    xga_lo = _has(low_features, ["mp_reg_5on5_OnIce_A_xGoals_per60", "mp_reg_5on5_OnIce_A_shotAttempts_per60"])
    continuation_hi = _has(high_features, ["mp_reg_5on5_I_F_playContinuedInZone_per60", "mp_reg_5on5_I_F_playContinuedOutsideZone_per60"])

    if xg_share_hi and xga_lo:
        return "Play-Driving Puck-Moving Defenseman", "Drives five-on-five shot quality while keeping chances against under control."
    if blocks_hi and (xga_lo or pk_hi):
        return "Crease-Clearing Suppression Defenseman", "Suppression profile: blocks shots, protects dangerous areas, and leans into defensive usage."
    if continuation_hi and offense_hi:
        return "Play-Driving Puck-Moving Defenseman", "Puck-moving profile: keeps plays alive and turns possession into offensive-zone pressure."
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
        high_text = ", ".join(readable_trait_label(feature).lower() for feature, _ in high_tokens[:2])
        low_text = ", ".join(readable_trait_label(feature).lower() for feature in list(low_features)[:2])
        detail = f"Role-driven defense profile that leans toward {high_text}"
        if low_text:
            detail += f" while showing less {low_text}"
        return "Defensive Role Defenseman", f"{detail}."
    top_text = ", ".join(readable_trait_label(feature).lower() for feature, _ in high_tokens[:2])
    return f"Defensive Role Defenseman {cluster}", f"Blended defense profile with its clearest signals in {top_text or 'usage and suppression metrics'}."


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
    high_danger_hi = _has(high_features, ["mp_reg_5on5_I_F_highDangerShots_per60", "mp_reg_5on5_I_F_highDangerShotShare", "mp_reg_5on5_I_F_reboundxGoals_per60"])
    rebound_hi = _has(high_features, ["mp_reg_5on5_I_F_rebounds_per60", "mp_reg_5on5_I_F_reboundxGoals_per60"])
    xg_share_hi = "mp_reg_5on5_OnIce_xGoalsPercentage_calc" in high_features
    xga_lo = _has(low_features, ["mp_reg_5on5_OnIce_A_xGoals_per60", "mp_reg_5on5_OnIce_A_shotAttempts_per60"])
    xga_hi = _has(high_features, ["mp_reg_5on5_OnIce_A_xGoals_per60", "mp_reg_5on5_OnIce_A_shotAttempts_per60"])
    in_zone_hi = "mp_reg_5on5_I_F_playContinuedInZone_per60" in high_features
    outside_zone_hi = "mp_reg_5on5_I_F_playContinuedOutsideZone_per60" in high_features

    if high_danger_hi and rebound_hi:
        return "Interior Net-Front Finisher", "Interior scoring profile: creates high-danger chances and rebound-based offense around the net."
    if giveaways_hi and offense_hi:
        return "High-Touch Risk/Reward Scorer", "Shot-creation profile with extra puck touches and some turnover risk."
    if outside_zone_hi and offense_hi:
        return "Rush / Transition Chance Creator", "Transition-leaning creator: pushes plays forward and turns movement through the neutral zone into offense."
    if in_zone_hi and xg_share_hi:
        return "Cycle Pressure Play-Driver", "Cycle-pressure profile: keeps offensive-zone plays alive and tilts shot quality in his team's favor."
    if xg_share_hi and xga_lo:
        return "Two-Way Shot-Share Driver", "Two-way driver: wins the five-on-five chance-quality battle without giving much back defensively."
    if fo_hi and pk_hi and xga_lo:
        return "Shutdown Suppression Center", "Defensive-center profile: handles draws and shorthanded usage while suppressing chances against."
    if xga_hi and blocks_hi:
        return "Shot-Blocking Contact Specialist", "Defensive workload profile: absorbs heavy chance volume and shows up through blocked shots."
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
    if takeaways_hi and scoring_hi:
        return "Two-Way Skill Scorer", "Blends scoring with puck-pressure and recovery value."
    if scoring_hi and not hits_hi:
        return "Perimeter Skill Scorer", "Skill-leaning scorer whose value comes more from shots and points than physical involvement."
    if giveaways_hi and playmaking_hi:
        return "High-Touch Risk/Reward Playmaker", "High-event puck profile: creates plays while carrying turnover risk."

    name, summary = _fallback_role_name(cluster, high_tokens, low_features)
    return canonical_profile_name(name), summary
