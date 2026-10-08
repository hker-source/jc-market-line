#!/usr/bin/env python3
"""poll_once.py — single poll, then exit. GitHub Actions entry point.

Run locally for validation:
    python poll_once.py --db /tmp/t.db
"""
import argparse
import os
import sqlite3
import sys

from HKJC_odds13 import HKJCGraphQLClient
from HKJC_warehouse135 import HKJCWarehouse
from HKJC_odds_modelling15 import LiveOddsModel

DEFAULT_TYPES = "HAD,SGA,HDC,CHL,HIL,CHD"


def main() -> int:
    ap = argparse.ArgumentParser(description="Single HKJC odds poll for CI.")
    ap.add_argument("--db", default="data/hkjc_odds.db",
                    help="SQLite database path (default: data/hkjc_odds.db)")
    ap.add_argument("--odds-types",
                    default=os.environ.get("HKJC_ODDS_TYPES", DEFAULT_TYPES),
                    help="Comma-separated odds types (default: %(default)s)")
    args = ap.parse_args()
    odds_types = [t.strip().upper() for t in args.odds_types.split(",") if t.strip()]

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    wh = HKJCWarehouse(args.db)
    model = LiveOddsModel(args.db)

    with HKJCGraphQLClient(headless=True) as client:
        n_raw = wh.ingest_from_client(
            client, odds_types=odds_types,
            start_index=1, end_index=60,
            navigate_first=True, save_raw=False,
        )
    if n_raw == 0:
        print("::error::poll returned 0 raw rows (possible block / throttle / no matches)")
        return 1

    n_events = model.ingest_from_warehouse_df(wh.latest())

    con = sqlite3.connect(args.db)
    integrity = con.execute("PRAGMA integrity_check").fetchone()[0]
    con.close()

    summary = f"raw={n_raw} events={n_events} integrity={integrity}"
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(f"### HKJC poll\n\n{summary}\n")
    return 0 if integrity == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
