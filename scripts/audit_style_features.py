#!/usr/bin/env python3
"""Emit a machine-readable quality report for the v2 style feature contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import yaml
except ImportError:  # Keep the audit runnable before the next dependency install.
    yaml = None


def load_contract(path: Path) -> list[dict]:
    if yaml is not None:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return [item for item in payload.get("features", []) if item.get("included_in_model")]
    # Minimal parser for this intentionally flat contract shape. Full YAML
    # parsing is used when PyYAML is installed; this fallback makes the
    # release audit usable in a clean data-runner image.
    items: list[dict] = []
    current: dict | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("- feature_id:"):
            if current and current.get("included_in_model"):
                items.append(current)
            current = {"feature_id": line.split(":", 1)[1].strip()}
        elif current is not None and ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            value = value.strip()
            if value.lower() in {"true", "false"}:
                parsed: object = value.lower() == "true"
            elif value.startswith("["):
                parsed = [part.strip() for part in value[1:-1].split(",")]
            else:
                parsed = value
            current[key.strip()] = parsed
    if current and current.get("included_in_model"):
        items.append(current)
    return items


def feature_summary(frame: pd.DataFrame, feature: dict) -> dict:
    column = feature["source_column"]
    if column not in frame.columns:
        return {"feature": feature["feature_id"], "column": column, "state": "unknown", "missing": True}
    values = pd.to_numeric(frame[column], errors="coerce")
    observed = values.notna()
    nonzero = values[observed].ne(0)
    q1, q3 = values[observed].quantile([0.25, 0.75]) if observed.any() else (np.nan, np.nan)
    allowed_min, allowed_max = feature.get("allowed_range", [None, None])
    try:
        allowed_min = float(allowed_min) if allowed_min is not None else None
        allowed_max = float(allowed_max) if allowed_max is not None else None
    except (TypeError, ValueError):
        allowed_min = allowed_max = None
    out_of_range = int(((values < allowed_min) | (values > allowed_max)).fillna(False).sum()) if allowed_min is not None else 0
    return {
        "feature": feature["feature_id"],
        "column": column,
        "state": "observed" if observed.all() else "unknown",
        "observedCount": int(observed.sum()),
        "unknownCount": int((~observed).sum()),
        "nonzeroRate": float(nonzero.mean()) if observed.any() else 0.0,
        "distinctCount": int(values[observed].nunique()),
        "median": float(values[observed].median()) if observed.any() else None,
        "iqr": float(q3 - q1) if observed.any() else None,
        "outOfRangeCount": out_of_range,
        "allZero": bool(observed.any() and not nonzero.any()),
        "nearConstant": bool(observed.any() and values[observed].nunique() <= 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season_label", required=True)
    parser.add_argument("--contract", default="config/style_feature_contract_v2.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fail_on_quality_gate", action="store_true")
    args = parser.parse_args()

    path = Path("data/features") / f"player_season_boxscore_{args.season_label}.parquet"
    frame = pd.read_parquet(path)
    features = load_contract(Path(args.contract))
    rows = []
    for group, positions in {"forwards": {"C", "LW", "RW", "W", "F"}, "defense": {"D", "LD", "RD"}}.items():
        subset = frame[frame["position"].astype(str).str.upper().isin(positions)]
        for feature in features:
            row = feature_summary(subset, feature)
            row.update({"season": args.season_label, "group": group, "rows": len(subset)})
            rows.append(row)
    report = {"season": args.season_label, "contract": args.contract, "features": rows}
    output = Path(args.output or f"reports/style_feature_audit_{args.season_label}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    failures = [row for row in rows if row.get("missing") or row.get("allZero") or row.get("outOfRangeCount", 0) > 0]
    if failures and args.fail_on_quality_gate:
        raise SystemExit(f"Style feature quality gate failed for {len(failures)} feature/group rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
