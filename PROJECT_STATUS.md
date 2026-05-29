# Project Status

Last reviewed locally on 2026-05-28.

## GitHub Latest

GitHub repository: `saaimkhan-sak/player-style-archetypes`

Latest remote commit seen through the GitHub connector:

- `ce172b42fc0036309f72d81023e2f941c200f6cb`
- Message: `Auto-update generated artifacts (2026-01-08 01:14:49)`

This local checkout is based on:

- `ad4d207d0f36352b212a8627891f09bcd2309a4f`
- Message: `Auto-update generated artifacts (2026-01-07 18:41:13)`

So GitHub is ahead of this local checkout.

## Local State

The local working tree contains an advanced-data rebuild:

- Old single-file Streamlit app `app/app.py` is deleted locally.
- New multi-page Streamlit app exists under `app/Home.py`, `app/lib.py`, and `app/pages/`.
- App-facing data now focuses on 18 complete advanced-data seasons: 2008-2009 through 2025-2026.
- The model combines NHL Gamecenter boxscore/usage data with MoneyPuck player-level advanced metrics.
- Pre-2008 app-facing player/report/model artifacts were removed because the MoneyPuck player files begin in 2008.
- All 2008+ app-season report CSVs and playoff projection files are present.
- `data/features/` and `data/processed/` are intermediate build artifacts and should stay out of future commits.

## Environment

Use Python 3.11, matching `runtime.txt`, then run:

```bash
python -m pip install -r requirements.txt
python scripts/health_check.py
```

## Cleanup Plan

1. Run `python scripts/health_check.py`.
2. Run `streamlit run app/Home.py` and smoke-test the three app pages.
3. Remove tracked intermediate artifacts from git history going forward:
   - `data/features/`
   - `data/processed/`
   - `data/raw/`
4. Keep tracked deploy artifacts:
   - `app/`
   - `.streamlit/config.toml`
   - `requirements.txt`
   - `runtime.txt`
   - `data/app/`
   - `reports/`
5. Fix or recreate the local git remote. This checkout currently cannot write `.git/config` from the sandbox, but the intended origin is:

```bash
git remote add origin https://github.com/saaimkhan-sak/player-style-archetypes.git
```
