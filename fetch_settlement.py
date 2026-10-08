"""
fetch_settlement.py -- fetch SETTLED matches (foPools resultOnly:true) and ingest.

Usage:
  venv/bin/python fetch_settlement.py [startDate] [endDate] [ODDS,ODDS,...] [--db DB]
  venv/bin/python fetch_settlement.py --help

Defaults: startDate=2026-09-20  endDate=2026-10-06  odds=HAD,HDC,CHL,HIL,SGA,CHD
"""
import argparse

from HKJC_odds13 import HKJCGraphQLClient
from HKJC_results17 import MatchResultsStore
from HKJC_backtest_pnl import pnl_table, pnl_summary

DB_DEFAULT = "hkjc_odds.db"


def all_matches(data):
    out = []
    for payload in data.values():
        out.extend((payload.get("data") or {}).get("matches") or [])
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Fetch settled matches (foPools resultOnly:true) and ingest.",
    )
    ap.add_argument("start", nargs="?", default="2026-09-20",
                    help="start date YYYY-MM-DD (default: 2026-09-20)")
    ap.add_argument("end", nargs="?", default="2026-10-06",
                    help="end date YYYY-MM-DD (default: 2026-10-06)")
    ap.add_argument("odds", nargs="?", default="HAD,HDC,CHL,HIL,SGA,CHD",
                    help="comma-separated odds types (default: HAD,HDC,CHL,HIL,SGA,CHD)")
    ap.add_argument("--db", default=DB_DEFAULT,
                    help=f"SQLite DB path (default: {DB_DEFAULT})")
    args = ap.parse_args()

    start = args.start
    end = args.end
    odds = args.odds.split(",")
    db = args.db

    with HKJCGraphQLClient(headless=True) as client:
        data = client.fetch_settlement(odds, start_index=1, end_index=60,
                                       start_date=start, end_date=end, save_raw=True)

    matches = all_matches(data)
    print(f"\nfetched matches (all types): {len(matches)}")

    samples, settled_pools = 0, 0
    for m in matches:
        for pool in (m.get("foPools") or []):
            hit = False
            for line in (pool.get("lines") or []):
                for comb in (line.get("combinations") or []):
                    if comb.get("status") in ("WIN", "LOSE"):
                        hit = True
                        if samples < 6:
                            print("  sample:", m.get("frontEndId"), m.get("status"),
                                  pool.get("oddsType"), "pool", pool.get("id"),
                                  "lineId", line.get("lineId"), "cond", line.get("condition"),
                                  "comb", comb.get("str"), comb.get("status"))
                            samples += 1
            if hit:
                settled_pools += 1
    print(f"settled pools (WIN/LOSE present): {settled_pools}")

    if matches:
        store = MatchResultsStore(db)
        wrote = store.ingest_settlement(matches)
        print(f"settlement rows written: {wrote} | total in db: {store.settlement_count()}")
        t = pnl_table(db)
        print(f"pnl rows (joinable to opening_odds): {len(t)}")
        print(pnl_summary(t))


if __name__ == "__main__":
    main()
