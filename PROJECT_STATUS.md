# Project Status

Last reviewed locally on 2026-05-27.

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

The local working tree contains a partially completed cleanup/restructure:

- Old single-file Streamlit app `app/app.py` is deleted locally.
- New multi-page Streamlit app exists under `app/Home.py`, `app/lib.py`, and `app/pages/`.
- App-facing data exists for 25 complete seasons: 2000-2001 through 2025-2026, excluding the cancelled 2004-2005 season.
- All app-season report CSVs are present.
- `data/features/` and `data/processed/` are intermediate build artifacts and should stay out of future commits.

## Environment

The previous `.venv` pointed to a missing Homebrew Python 3.13 binary. The project requirements do not resolve cleanly on Python 3.11 or 3.14:

- Python 3.11: `altair==6.0.0` is unavailable.
- Python 3.14: some scientific stack pins are incompatible, and `numpy==2.4.0` is yanked upstream.

Use Python 3.11, matching `runtime.txt`, then run:

```bash
python -m pip install -r requirements.txt
python scripts/health_check.py
```

## Cleanup Plan

1. Install/link Python 3.13 locally and recreate `.venv`.
2. Run `python scripts/health_check.py`.
3. Run `streamlit run app/Home.py` and smoke-test the three app pages.
4. Decide whether to keep the local generated `data/app` and `reports` updates or replace them with GitHub latest.
5. Remove tracked intermediate artifacts from git history going forward:
   - `data/features/`
   - `data/processed/`
   - `data/raw/`
6. Keep tracked deploy artifacts:
   - `app/`
   - `.streamlit/config.toml`
   - `requirements.txt`
   - `runtime.txt`
   - `data/app/`
   - `reports/`
7. Fix or recreate the local git remote. This checkout currently cannot write `.git/config` from the sandbox, but the intended origin is:

```bash
git remote add origin https://github.com/saaimkhan-sak/player-style-archetypes.git
```
