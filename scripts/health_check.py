from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODULES = [
    "altair",
    "numpy",
    "pandas",
    "pyarrow",
    "st_aggrid",
    "streamlit",
]


def season_label(k: str) -> str:
    return f"{k[:4]}-{k[4:]}" if len(k) == 8 and k.isdigit() else k


def check_modules() -> list[str]:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing Python packages: " + ", ".join(missing))
        return [f"missing packages: {', '.join(missing)}"]
    print("Python packages: ok")
    return []


def check_app_files() -> list[str]:
    errors: list[str] = []
    expected = [
        ROOT / "app" / "Home.py",
        ROOT / "app" / "lib.py",
        ROOT / "app" / "pages" / "01_Season_Level_Analysis.py",
        ROOT / "app" / "pages" / "02_Player_Evolution.py",
    ]
    missing = [p.relative_to(ROOT).as_posix() for p in expected if not p.exists()]
    if missing:
        errors.append("missing app files: " + ", ".join(missing))
        print(errors[-1])
    else:
        print("Streamlit app files: ok")
    return errors


def check_data_inventory() -> list[str]:
    errors: list[str] = []
    data_app = ROOT / "data" / "app"
    reports = ROOT / "reports"

    forwards = {p.stem.replace("players_forwards_", "") for p in data_app.glob("players_forwards_*.parquet")}
    defense = {p.stem.replace("players_defense_", "") for p in data_app.glob("players_defense_*.parquet")}
    seasons = sorted(forwards & defense)

    if not seasons:
        errors.append("no complete seasons found in data/app")
        print(errors[-1])
        return errors

    print(f"Complete app seasons: {len(seasons)} ({season_label(seasons[0])} to {season_label(seasons[-1])})")

    missing_sides = sorted((forwards ^ defense))
    if missing_sides:
        errors.append("incomplete app season pairs: " + ", ".join(missing_sides))
        print(errors[-1])

    missing_reports = []
    for season in seasons:
        for group in ("forwards", "defense"):
            for kind in ("cards", "traits"):
                path = reports / f"archetype_{kind}_{group}_{season}.csv"
                if not path.exists():
                    missing_reports.append(path.name)

    if missing_reports:
        preview = ", ".join(missing_reports[:12])
        suffix = " ..." if len(missing_reports) > 12 else ""
        errors.append(f"missing report CSVs: {preview}{suffix}")
        print(errors[-1])
    else:
        print("Report CSVs for app seasons: ok")

    return errors


def check_parquet_readability() -> list[str]:
    if importlib.util.find_spec("pandas") is None:
        return []

    import pandas as pd

    errors: list[str] = []
    samples = sorted((ROOT / "data" / "app").glob("players_*_*.parquet"))[:2]
    samples += sorted((ROOT / "data" / "app").glob("players_*_*.parquet"))[-2:]
    for path in samples:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            errors.append(f"cannot read {path.relative_to(ROOT)}: {exc}")
            continue
        required = {"season", "player_id", "full_name", "top_cluster", "confidence"}
        missing = required - set(df.columns)
        if missing:
            errors.append(f"{path.relative_to(ROOT)} missing columns: {', '.join(sorted(missing))}")
    if errors:
        for err in errors:
            print(err)
    else:
        print("Sample app parquet reads: ok")
    return errors


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    errors: list[str] = []
    errors.extend(check_modules())
    errors.extend(check_app_files())
    errors.extend(check_data_inventory())
    errors.extend(check_parquet_readability())

    if errors:
        print(f"\nHealth check failed with {len(errors)} issue(s).")
        return 1

    print("\nHealth check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
