"""
Scrape MoneyPuck playoff skater stats for a given season and save as CSV
matching the game-by-game-player-data column format used by this project.

Usage:
    python scripts/scrape_moneypuck_playoffs.py --season 2025 --out data/raw/moneypuck/playoffs_skaters_2025.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

PAGE_URL = "https://peter-tanner.com/moneypuck/players.htm"


def scrape(season: str, out_path: Path) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Navigating to {PAGE_URL} …")
        page.goto(PAGE_URL, wait_until="networkidle", timeout=60_000)

        # Select season
        page.select_option("#season_type", season)
        # Select playoffs
        page.select_option("#table_playoff_type", "playoffs")

        # Wait for table to re-render (XHR fires on change)
        print("Waiting for table to load …")
        page.wait_for_function(
            """() => {
                const rows = document.querySelectorAll('#includedContent tbody tr');
                return rows.length > 10;
            }""",
            timeout=30_000,
        )
        time.sleep(2)  # let any remaining rendering finish

        # Extract headers from thead
        headers = page.eval_on_selector_all(
            "#includedContent thead th",
            "els => els.map(e => e.innerText.trim().replace(/\\n/g, ' '))",
        )
        print(f"Found {len(headers)} columns")

        # Extract all data rows
        rows_data = page.evaluate("""() => {
            const rows = document.querySelectorAll('#includedContent tbody tr');
            return Array.from(rows).map(row => {
                const cells = row.querySelectorAll('td');
                return Array.from(cells).map(c => c.innerText.trim());
            });
        }""")

        # Filter out rows that are just rank/logo/name (3 cells) - those are mobile collapsed rows
        data_rows = [r for r in rows_data if len(r) == len(headers)]
        print(f"Found {len(data_rows)} data rows")

        if not data_rows:
            # Try alternate extraction: some layouts put stats in nested elements
            print("Trying alternate extraction …")
            rows_data = page.evaluate("""() => {
                const tbodies = document.querySelectorAll('#includedContent tbody');
                return Array.from(tbodies).map(tb => {
                    const cells = tb.querySelectorAll('td');
                    return Array.from(cells).map(c => c.innerText.trim());
                }).filter(r => r.length > 5);
            }""")
            data_rows = [r for r in rows_data if len(r) == len(headers)]
            print(f"Alternate: {len(data_rows)} matching rows")

        browser.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data_rows)

    print(f"Saved {len(data_rows)} rows → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="2025", help="Season year (e.g. 2025 for 2025-26)")
    parser.add_argument("--out", default="data/raw/moneypuck/playoffs_skaters_2025.csv")
    args = parser.parse_args()
    scrape(args.season, Path(args.out))
