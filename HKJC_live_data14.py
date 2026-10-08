"""
HKJC_live_data14.py

Pure poller. Does NOT implement its own change-detection anymore -- that
logic used to be duplicated three ways across this file, HKJC_odds_modelling15's
diff_and_store(), and an earlier ad-hoc version. Now there is exactly one
diff engine (LiveOddsModel.ingest_batch in HKJC_odds_modelling15), and this
file's only job is: fetch -> warehouse.ingest -> model.ingest_batch -> report.
"""

import sys
import time
from pathlib import Path

from HKJC_odds13 import HKJCGraphQLClient
from HKJC_warehouse135 import HKJCWarehouse
from HKJC_odds_modelling15 import LiveOddsModel

_ROOT = Path(__file__).parent

MONITOR_ODDS_TYPES = ["HAD", "HDC", "CHL", "HIL", "SGA"]
POLL_INTERVAL = 328
RUN_CYCLES = 0
DB_PATH = str(_ROOT / "hkjc_odds.db")

for arg in sys.argv[1:]:
    if arg.startswith("--cycles="):
        RUN_CYCLES = int(arg.split("=")[1])
    elif arg.startswith("--poll-interval="):
        POLL_INTERVAL = int(arg.split("=")[1])
    elif arg.startswith("--db="):
        DB_PATH = arg.split("=")[1]


def main():
    print("HKJC Live Odds Monitor starting...")
    print(f"Odds types: {MONITOR_ODDS_TYPES}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"DB: {DB_PATH}")
 

    warehouse = HKJCWarehouse(DB_PATH)
    model = LiveOddsModel(DB_PATH)  # same db file, different tables -- one file, two concerns

    with HKJCGraphQLClient(headless=True) as client:
        iteration = 0
        while True:
            iteration += 1
            if RUN_CYCLES > 0 and iteration > RUN_CYCLES:
                print(f"Reached {RUN_CYCLES} cycles. Stopping.")
                break

            print(f"\n{'=' * 60}\nPoll #{iteration} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 60}")

            raw_rows_written = warehouse.ingest_from_client(
                client, odds_types=MONITOR_ODDS_TYPES, start_index=1, end_index=60,
                save_raw=True,
            )
            print(f"Warehouse: +{raw_rows_written} raw rows")

            if raw_rows_written == 0:
                print("No data fetched, retrying next cycle...")
                time.sleep(POLL_INTERVAL)
                continue

            latest_df = warehouse.latest()
            n_events = model.ingest_from_warehouse_df(latest_df)
            print(f"Model: {n_events} odds_events written")

            if n_events > 0:
                recent = model.steam_moves(match_id=None)
                recent = recent[recent["event_time"] >= latest_df["scraped_at"].max()] \
                    if not recent.empty else recent
                if not recent.empty:
                    print("\n--- STEAM / ANTICIPATORY MOVES THIS POLL ---")
                    cols = ["match_id", "odds_type", "selection_name", "old_odds",
                            "new_odds", "delta_prob", "movement_type", "minutes_to_ko"]
                    print(recent[cols].to_string(index=False))

            print(f"\nNext poll in {POLL_INTERVAL} seconds... (Ctrl+C to stop)")
            try:
                time.sleep(POLL_INTERVAL)
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
                break


if __name__ == "__main__":
    main()
