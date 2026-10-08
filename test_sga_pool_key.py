"""
test_sga_pool_key.py -- regression test for the SGA pool_id primary-key fix.

Before the fix:
  odds_snapshot / opening_odds were keyed (match_id, odds_type, line_id, comb_id).
  SGA publishes ~20 pools per match that all share comb_id='1', so 19 of every
  20 SGA pools were silently overwritten (FB5602: 20 raw pools -> 1 snapshot row).
  The single-outcome de-vig also normalised every SGA pool to probability 1.0.

Run: venv/bin/python test_sga_pool_key.py
"""

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from HKJC_odds_modelling15 import LiveOddsModel, RawRow
from HKJC_warehouse135 import HKJCWarehouse


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    return path


def test_sga_pools_do_not_collide():
    print("TEST 1: 20 SGA pools (one comb_id) -> 20 snapshot + 20 opening rows")
    db = _tmp_db()
    model = LiveOddsModel(db)
    rows = [
        RawRow(match_id="FBX", odds_type="SGA", line_id="0", comb_id="1",
               selection_name=f"acca {i}", odds=4.0 + i, scraped_at="2026-09-27T00:00:00+00:00",
               pool_id=f"P{i:02d}")
        for i in range(20)
    ]
    model.ingest_batch(rows)
    con = sqlite3.connect(db)
    snap = con.execute("SELECT COUNT(*) FROM odds_snapshot WHERE match_id='FBX' AND odds_type='SGA'").fetchone()[0]
    opn = con.execute("SELECT COUNT(*) FROM opening_odds WHERE match_id='FBX' AND odds_type='SGA'").fetchone()[0]
    pools = con.execute("SELECT COUNT(DISTINCT pool_id) FROM odds_snapshot WHERE match_id='FBX' AND odds_type='SGA'").fetchone()[0]
    assert snap == 20, f"expected 20 snapshot rows, got {snap}"
    assert opn == 20, f"expected 20 opening rows, got {opn}"
    assert pools == 20, f"expected 20 distinct pool_id, got {pools}"
    con.close()
    os.remove(db)
    print("  PASS")


def test_single_outcome_prob_not_forced_to_one():
    print("TEST 2: single-outcome pool keeps 1/odds (not normalised to 1.0)")
    db = _tmp_db()
    model = LiveOddsModel(db)
    model.ingest_batch([
        RawRow(match_id="FBX", odds_type="SGA", line_id="0", comb_id="1",
               selection_name="acca", odds=4.0, scraped_at="2026-09-27T00:00:00+00:00",
               pool_id="P1"),
    ])
    con = sqlite3.connect(db)
    prob = con.execute("SELECT implied_prob_devig FROM odds_snapshot WHERE match_id='FBX' AND pool_id='P1'").fetchone()[0]
    assert abs(prob - 0.25) < 1e-9, f"expected 0.25 (=1/4.0), got {prob}"
    con.close()
    os.remove(db)
    print("  PASS")


def test_move_on_one_pool_does_not_touch_siblings():
    print("TEST 3: a move on one SGA pool updates only that pool")
    db = _tmp_db()
    model = LiveOddsModel(db)
    first = [
        RawRow(match_id="FBX", odds_type="SGA", line_id="0", comb_id="1",
               selection_name=f"acca {i}", odds=4.0, scraped_at="2026-09-27T00:00:00+00:00",
               pool_id=f"P{i:02d}")
        for i in range(20)
    ]
    model.ingest_batch(first)
    # second poll: only P00 shortens 4.0 -> 3.0
    moved = [RawRow(match_id="FBX", odds_type="SGA", line_id="0", comb_id="1",
                    selection_name="acca 0", odds=3.0, scraped_at="2026-09-27T00:10:00+00:00",
                    pool_id="P00")]
    model.ingest_batch(moved)
    con = sqlite3.connect(db)
    n = con.execute("SELECT COUNT(*) FROM odds_snapshot WHERE match_id='FBX'").fetchone()[0]
    p00 = con.execute("SELECT odds FROM odds_snapshot WHERE pool_id='P00'").fetchone()[0]
    p01 = con.execute("SELECT odds FROM odds_snapshot WHERE pool_id='P01'").fetchone()[0]
    ev = con.execute("SELECT COUNT(*) FROM odds_events WHERE pool_id='P00'").fetchone()[0]
    assert n == 20, f"snapshot rows changed: {n}"
    assert p00 == 3.0, f"P00 not updated: {p00}"
    assert p01 == 4.0, f"sibling P01 disturbed: {p01}"
    assert ev == 1, f"expected 1 event on P00, got {ev}"
    con.close()
    os.remove(db)
    print("  PASS")


def test_multi_outcome_devig_still_normalises():
    print("TEST 4: 3-way HAD still de-vigs to sum=1 (regression guard)")
    db = _tmp_db()
    model = LiveOddsModel(db)
    model.ingest_batch([
        RawRow(match_id="FBY", odds_type="HAD", line_id="L1", comb_id=c, selection_name=c,
               odds=o, scraped_at="2026-09-27T00:00:00+00:00", pool_id="PH")
        for c, o in (("H", 2.0), ("D", 3.5), ("A", 4.0))
    ])
    con = sqlite3.connect(db)
    tot = con.execute("SELECT SUM(implied_prob_devig) FROM odds_snapshot WHERE match_id='FBY'").fetchone()[0]
    assert abs(tot - 1.0) < 1e-9, f"HAD probs should sum to 1, got {tot}"
    con.close()
    os.remove(db)
    print("  PASS")


def test_warehouse_latest_keeps_all_pools():
    print("TEST 5: HKJCWarehouse.latest() returns both SGA pools, not one")
    db = _tmp_db()
    wh = HKJCWarehouse(db)
    payload = {"SGA": {"data": {"matches": [{
        "id": "50077222", "frontEndId": "FB9999",
        "homeTeam": {"name_en": "A"}, "awayTeam": {"name_en": "B"},
        "tournament": {"name_en": "T"}, "venue": {"name_en": "V"},
        "matchDate": "2026-09-27", "kickOffTime": "02:45", "status": "PREEVENT",
        "runningResult": {"homeScore": 0, "awayScore": 0, "homeCorner": 0, "awayCorner": 0},
        "foPools": [
            {"id": "P1", "oddsType": "SGA", "status": "ACTIVE", "inplay": False,
             "lines": [{"lineId": "0", "condition": "0.0", "combinations": [
                 {"combId": "1", "str": "01", "status": "ACTIVE", "currentOdds": 4.9,
                  "selections": [{"str": "01", "name_en": "BTTS & A win"}]}]}]},
            {"id": "P2", "oddsType": "SGA", "status": "ACTIVE", "inplay": False,
             "lines": [{"lineId": "0", "condition": "0.0", "combinations": [
                 {"combId": "1", "str": "01", "status": "ACTIVE", "currentOdds": 7.25,
                  "selections": [{"str": "01", "name_en": "A win & Over 2.5"}]}]}]},
        ],
    }]}}}
    wh.ingest(payload)
    df = wh.latest(match_id="FB9999")
    sga = df[df.odds_type == "SGA"]
    assert len(sga) == 2, f"latest() returned {len(sga)} SGA rows, expected 2"
    assert sga.pool_id.nunique() == 2, "latest() lost a pool"
    os.remove(db)
    print("  PASS")


if __name__ == "__main__":
    tests = [
        test_sga_pools_do_not_collide,
        test_single_outcome_prob_not_forced_to_one,
        test_move_on_one_pool_does_not_touch_siblings,
        test_multi_outcome_devig_still_normalises,
        test_warehouse_latest_keeps_all_pools,
    ]
    for t in tests:
        t()
    print("\n" + "=" * 60 + "\nALL SGA POOL-KEY TESTS PASSED\n" + "=" * 60)
