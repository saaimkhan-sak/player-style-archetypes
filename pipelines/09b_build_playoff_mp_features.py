"""
Pipeline 09b: Build playoff MoneyPuck advanced features from saved HTML season-summary pages.

Reads the scraped HTML files in data/raw/moneypuck/mp-playoffs-raw-html-files/,
computes mp_po_* features that match the mp_reg_* naming convention used by the
NMF/GMM model, and merges them into the player_season_boxscore_{season}.parquet
so that pipeline 09 can use them for playoff archetype projection.

Usage:
    python pipelines/09b_build_playoff_mp_features.py --all
    python pipelines/09b_build_playoff_mp_features.py --season_label 20242025
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


HTML_BASE = Path("data/raw/moneypuck/mp-playoffs-raw-html-files")
FEATURES_DIR = Path("data/features")

# Situations to process and their folder suffixes
SITUATIONS = {
    "5on5": "5on5-playoff-data",
    "5on4": "5on4-playoff-data",
    "4on5": "4on5-playoff-data",
}

# season_label (e.g. "20242025") -> year string (e.g. "2024-2025")
def season_label_to_year(label: str) -> str:
    return f"{label[:4]}-{label[4:]}"


def parse_html(path: Path) -> pd.DataFrame:
    """Parse a MoneyPuck saved HTML page and return a clean DataFrame."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    m = re.search(
        r'id="includedContent"[^>]*>(.*?)(?=<div id="(?!includedContent)|\Z)',
        content, re.DOTALL,
    )
    if not m:
        return pd.DataFrame()
    inner = m.group(1)

    thead = re.search(r'<thead>(.*?)</thead>', inner, re.DOTALL)
    if not thead:
        return pd.DataFrame()
    raw_ths = re.findall(r'<th[^>]*>(.*?)</th>', thead.group(1), re.DOTALL)
    headers = [re.sub(r'<[^>]+>', '', h).strip().replace('\n', ' ') for h in raw_ths]

    chunks = re.split(r'<tbody>', inner)[1:]
    rows = []
    for chunk in chunks:
        name_m = re.search(r'<a href="[^"]*">([^<]+)</a>', chunk)
        pid_m = re.search(r'\?p=(\d+)', chunk)
        team_m = re.search(r'alt="([^"]+)"', chunk)
        name = name_m.group(1).strip() if name_m else ''
        pid = pid_m.group(1) if pid_m else ''
        team = team_m.group(1).strip() if team_m else ''
        after = chunk[chunk.find('</tbody></table>') + 16:]
        tds = re.findall(r'<td[^>]*>(.*?)</td>', after, re.DOTALL)
        vals = [re.sub(r'<[^>]+>', '', t).strip() for t in tds]
        if vals:
            rows.append([name, pid, team] + vals)

    final_headers = ['name', 'playerId', 'team'] + headers[1:]
    # Deduplicate column names (e.g. 'ExpectedGoals' appears twice)
    seen: dict[str, int] = {}
    deduped = []
    for h in final_headers:
        if h in seen:
            seen[h] += 1
            deduped.append(f'{h}.{seen[h]}')
        else:
            seen[h] = 0
            deduped.append(h)
    final_headers = deduped

    if not rows:
        return pd.DataFrame(columns=final_headers)

    # Pad/truncate rows to match header count
    n = len(final_headers)
    rows = [r[:n] + [''] * max(0, n - len(r)) for r in rows]
    df = pd.DataFrame(rows, columns=final_headers)
    df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce')
    return df.dropna(subset=['playerId']).copy()


def pct_to_float(s: pd.Series) -> pd.Series:
    """Convert '52.3%' or '52.3' to 52.3 (keep as 0-100 scale)."""
    cleaned = s.astype(str).str.replace('%', '', regex=False).str.strip()
    return pd.to_numeric(cleaned, errors='coerce').fillna(0.0)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce').fillna(0.0)


def per60(total: pd.Series, icetime_hours: pd.Series) -> pd.Series:
    return (total / icetime_hours.replace(0, np.nan)).fillna(0.0)


def compute_features(df: pd.DataFrame, situation: str) -> pd.DataFrame:
    """Compute mp_po_{situation}_* features from a raw season-summary DataFrame."""
    c = df.columns.tolist()

    icetime_h = to_num(df['Icetime(Minutes)']) / 60.0  # hours

    xgoals      = to_num(df['ExpectedGoals'])
    shot_att    = to_num(df['Shot Attempts']).replace(0, np.nan)
    hd_shots    = to_num(df['High DangerUnblockedShot Attempts'])
    hd_xgoals   = to_num(df['High DangerxGoals'])
    rebounds    = to_num(df['ReboundsCreated'])
    rebound_xg  = to_num(df['xGoals OnRebounds Shots'])
    created_xg  = to_num(df['CreatedxGoals'])
    hits        = to_num(df['Hits'])
    takeaways   = to_num(df['Takeaways'])
    giveaways   = to_num(df['Giveaways'])
    pim_drawn   = to_num(df['PIMDrawn'])
    blk_per60   = to_num(df['Shots BlockedBy PlayerPer 60'])

    xg_per60    = to_num(df['ExpectedGoals Per 60Minutes'])
    att_per60   = to_num(df['Shot AttemptsPer 60Minutes'])
    onice_xgf60 = to_num(df['On-Ice ExpectedGoals ForPer 60 Minutes'])
    onice_attf60= to_num(df['On-Ice ShotAttempts ForPer 60 Minutes'])
    onice_xga60 = to_num(df['On-Ice ExpectedGoals AgainstPer 60 Minutes'])
    onice_atta60= to_num(df['On-Ice ShotAttempts AgainstPer 60 Minutes'])
    onice_xgpct = pct_to_float(df['On-IceExpectedGoals %']) / 100.0
    fo_pct      = pct_to_float(df['FaceoffWin %'])

    rebound_xg_share = to_num(df['Share of xGoalsFrom ReboundsShots'])
    # If stored as percentage (0-100) convert to fraction
    if rebound_xg_share.max() > 1.5:
        rebound_xg_share = rebound_xg_share / 100.0

    feat: dict[str, pd.Series] = {
        'I_F_xGoals_per60':                    xg_per60,
        'I_F_xGoalsPerAttempt':                (xgoals / shot_att).fillna(0.0),
        'I_F_shotAttempts_per60':              att_per60,
        'I_F_highDangerShots_per60':           per60(hd_shots, icetime_h),
        'I_F_highDangerxGoals_per60':          per60(hd_xgoals, icetime_h),
        'I_F_highDangerShotShare':             (hd_shots / shot_att).fillna(0.0),
        'I_F_rebounds_per60':                  per60(rebounds, icetime_h),
        'I_F_reboundxGoals_per60':             per60(rebound_xg, icetime_h),
        'I_F_reboundxGoalsShare':              rebound_xg_share,
        'I_F_xGoals_with_earned_rebounds_per60': per60(created_xg, icetime_h),
        'OnIce_xGoalsPercentage_calc':         onice_xgpct,
        'OnIce_F_xGoals_per60':                onice_xgf60,
        'OnIce_F_shotAttempts_per60':          onice_attf60,
        'OnIce_A_xGoals_per60':                onice_xga60,
        'OnIce_A_shotAttempts_per60':          onice_atta60,
        'shotsBlockedByPlayer_per60':          blk_per60,
        'I_F_hits_per60':                      per60(hits, icetime_h),
        'I_F_takeaways_per60':                 per60(takeaways, icetime_h),
        'I_F_giveaways_per60':                 per60(giveaways, icetime_h),
        'penaltiesDrawn_per60':                per60(pim_drawn, icetime_h),
        'faceoffPct':                          fo_pct,
        'faceoffsWon':                         to_num(df.get('FaceoffsWon', pd.Series(0, index=df.index))),
    }

    # 5on5-only: after-shift xGoals columns
    if situation == '5on5':
        if 'xGoals For5 Seconds AfterFly Shift Ended' in c:
            feat['xGoalsForAfterShifts_per60'] = per60(
                to_num(df['xGoals For5 Seconds AfterFly Shift Ended']), icetime_h)
        if 'xGoals Against5 Seconds AfterFly Shift Ended' in c:
            feat['xGoalsAgainstAfterShifts_per60'] = per60(
                to_num(df['xGoals Against5 Seconds AfterFly Shift Ended']), icetime_h)

    prefix = f'mp_po_{situation}_'
    out = pd.DataFrame({'player_id': df['playerId'].astype(int)})
    for name, series in feat.items():
        out[prefix + name] = series.values
    return out


def build_season(season_label: str) -> bool:
    year = season_label_to_year(season_label)
    year_dir = HTML_BASE / f"mp-playoff-data-{year}"
    if not year_dir.exists():
        print(f"  No HTML dir for {season_label} ({year_dir})")
        return False

    boxscore_path = FEATURES_DIR / f"player_season_boxscore_{season_label}.parquet"
    if not boxscore_path.exists():
        print(f"  No boxscore parquet for {season_label}")
        return False

    all_features: list[pd.DataFrame] = []

    for situation, suffix in SITUATIONS.items():
        html_path = year_dir / f"{year}-{suffix}.html"
        if not html_path.exists():
            print(f"  Missing {html_path.name}, skipping {situation}")
            continue

        raw = parse_html(html_path)
        if raw.empty:
            print(f"  Empty parse for {html_path.name}")
            continue

        feats = compute_features(raw, situation)
        all_features.append(feats)
        print(f"  {situation}: {len(feats)} players, {len(feats.columns)-1} features")

    # Also parse all-situations to get po_games and position
    all_sit_path = year_dir / f"{year}-all-situations-combined-playoff-data.html"
    po_games_df: pd.DataFrame | None = None
    if all_sit_path.exists():
        raw_all = parse_html(all_sit_path)
        if not raw_all.empty:
            po_games_df = pd.DataFrame({
                'player_id': raw_all['playerId'].astype('Int64'),
                'mp_po_games': to_num(raw_all['GamesPlayed']).astype(int),
                'mp_po_position': raw_all['Pos'].str.upper().str.strip(),
            })

    if not all_features:
        print(f"  No features built for {season_label}")
        return False

    merged = all_features[0]
    for df in all_features[1:]:
        merged = merged.merge(df, on='player_id', how='outer')
    merged['player_id'] = merged['player_id'].astype('Int64')
    if po_games_df is not None:
        merged = merged.merge(po_games_df, on='player_id', how='left')
    merged = merged.fillna(0.0)

    boxscore = pd.read_parquet(boxscore_path)
    # Inject po_games from MoneyPuck if missing from boxscore
    if 'po_games' not in boxscore.columns and 'mp_po_games' in merged.columns:
        po_map = merged.set_index('player_id')['mp_po_games'].to_dict()
        boxscore['po_games'] = boxscore['player_id'].map(po_map).fillna(0).astype(int)
        print(f"  Injected po_games from MoneyPuck all-situations")
    # Drop any existing mp_po_ columns so we can replace them cleanly
    existing_mp_po = [c for c in boxscore.columns if c.startswith('mp_po_')]
    if existing_mp_po:
        boxscore = boxscore.drop(columns=existing_mp_po)

    boxscore = boxscore.merge(merged, on='player_id', how='left')
    # Fill NaN for players with no playoff MP data
    new_cols = [c for c in merged.columns if c != 'player_id']
    str_cols = boxscore[new_cols].select_dtypes(include='object').columns
    num_cols = [c for c in new_cols if c not in str_cols]
    boxscore[num_cols] = boxscore[num_cols].fillna(0.0)
    boxscore[str_cols] = boxscore[str_cols].fillna('')

    boxscore.to_parquet(boxscore_path, index=False)
    mp_po_count = len([c for c in boxscore.columns if c.startswith('mp_po_')])
    print(f"  Saved {season_label}: {mp_po_count} mp_po_ cols merged into boxscore parquet")
    return True


def available_seasons() -> list[str]:
    return sorted(
        p.stem.replace('player_season_boxscore_', '')
        for p in FEATURES_DIR.glob('player_season_boxscore_*.parquet')
        if p.stem.replace('player_season_boxscore_', '')[:4].isdigit()
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--season_label', help='e.g. 20242025')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args(argv)

    if not args.all and not args.season_label:
        ap.error('Provide --season_label or --all')

    seasons = available_seasons() if args.all else [args.season_label]
    ok = 0
    for season in seasons:
        print(f"\n{season}:")
        if build_season(season):
            ok += 1

    print(f"\nDone: {ok}/{len(seasons)} seasons enriched.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
