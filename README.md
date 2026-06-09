# Player Style Archetypes (NHL)

Data pipeline and Streamlit app for NHL player style archetypes.

The project:
1. Pulls public NHL game data into a local cache.
2. Aggregates MoneyPuck player game files into advanced player-season features.
3. Builds fused player-season feature vectors from NHL boxscore/usage data plus MoneyPuck advanced metrics.
4. Fits NMF + GMM models for soft player archetypes.
5. Publishes Streamlit views for season-level analysis, roster fit, comps, and player evolution.

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
python pipelines/03b_build_moneypuck_player_season_features.py --season_label 20252026
python pipelines/03_build_player_season_features_boxscore.py --schedule_parquet data/processed/schedule_20252026.parquet --season_label 20252026
python pipelines/04_build_model_matrices.py --season_label 20252026
python pipelines/05_fit_nmf_gmm.py --season_label 20252026
python pipelines/07_make_archetype_cards.py --season_label 20252026
python pipelines/08_build_app_tables.py --season_label 20252026
```

Rebuild a range of seasons:

```bash
python pipelines/99_build_all_seasons.py --start_year 2008 --end_year 2025 --download_missing
```

The advanced-data build starts at 2008-09 because that is the first season covered by the MoneyPuck player files currently used by the project. The all-season runner uses the active Python interpreter, so run it from the activated virtual environment.

## Daily Refresh

GitHub Actions runs `.github/workflows/refresh-data.yml` every day at 9am America/New_York time. The workflow:

1. Downloads the latest current-season MoneyPuck skater and line game-by-game files.
2. Reconciles the NHL schedule and downloads completed game boxscores/play-by-play.
3. Rebuilds the current season models, app tables, reports, playoff projections, and line combinations.
4. Runs the health check and smoke tests.
5. Commits `data/app`, `reports`, and `models` back to `main` only when generated artifacts changed.

You can also start it manually from the GitHub Actions tab and optionally pass a season label such as `20252026`.

## Deployment Notes

For Streamlit Cloud, use `app/Home.py` as the app entrypoint. The repo intentionally tracks app-facing data in `data/app` and reports in `reports`, while larger intermediate files stay local.
