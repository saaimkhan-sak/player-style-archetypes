# Player Style Archetypes (NHL)

Data pipeline and Streamlit app for NHL player style archetypes.

The project:
1. Pulls public NHL game data into a local cache.
2. Builds player-season boxscore and usage feature vectors.
3. Fits NMF + GMM models for soft player archetypes.
4. Publishes Streamlit views for season-level analysis, roster fit, comps, and player evolution.

## Current Shape

- Streamlit entrypoint: `app/Home.py`
- App-ready data: `data/app/players_{forwards,defense}_<season>.parquet`
- App reports: `reports/archetype_{cards,traits}_{forwards,defense}_<season>.csv`
- Pipeline scripts: `pipelines/`
- Local build artifacts ignored by git: `data/raw/`, `data/features/`, `data/processed/`, `logs/`

The app currently expects paired forwards and defense files in `data/app`.

## Setup

Use Python 3.11, matching `runtime.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/health_check.py
```

Run the app locally:

```bash
streamlit run app/Home.py
```

## Rebuild Data

Build a single season after raw game data is present:

```bash
python pipelines/08_build_app_tables.py --season_label 20252026
```

Rebuild a range of seasons:

```bash
python pipelines/99_build_all_seasons.py --start_year 2000 --end_year 2025 --download_missing
```

The all-season runner uses the active Python interpreter, so run it from the activated virtual environment.

## Deployment Notes

For Streamlit Cloud, use `app/Home.py` as the app entrypoint. The repo intentionally tracks app-facing data in `data/app` and reports in `reports`, while larger intermediate files stay local.
