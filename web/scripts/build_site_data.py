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


def build_glossary(
    seasons: list[str],
    maps: dict[str, dict[str, dict[int, str]]],
    all_frames: dict[str, list[pd.DataFrame]],
) -> dict[str, list[dict[str, Any]]]:
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
            for _, row in frame.iterrows():
                cluster = int(row["top_cluster"])
                name = maps[group][season].get(cluster)
                if not name:
                    continue
                player_id = int(row["player_id"])
                games_value = row.get("reg_games", 0)
                games = float(games_value) if pd.notna(games_value) else 0.0
                candidate = candidates[name][season_group].setdefault(
                    player_id,
                    {
                        "id": player_id,
                        "name": str(row["full_name"]),
                        "games": 0.0,
                    },
                )
                candidate["games"] += games

        examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for name, grouped_candidates in candidates.items():
            used_ids: set[int] = set()
            for season_group in range(len(season_groups)):
                ranked = sorted(
                    grouped_candidates.get(season_group, {}).values(),
                    key=lambda player: (
                        -float(player["games"]),
                        str(player["name"]),
                        int(player["id"]),
                    ),
                )
                selected = next(
                    (
                        player
                        for player in ranked
                        if int(player["id"]) not in used_ids
                    ),
                    None,
                )
                if selected is None:
                    continue
                player_id = int(selected["id"])
                used_ids.add(player_id)
                examples[name].append(
                    {
                        "id": player_id,
                        "name": str(selected["name"]),
                        "games": clean(selected["games"], 0),
                    }
                )

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
                        "share": clean(100 * count / len(players), 2),
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
