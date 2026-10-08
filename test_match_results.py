"""
test_match_results.py

Plain-script tests for HKJC_results17.MatchResultsStore (no pytest needed).
Everything runs on synthetic payloads shaped like the expected HKJC matchList
`results[]` response; no network is touched.
"""

import os
from pathlib import Path

from HKJC_results17 import MatchResultsStore

DB_PATH = "test_match_results.db"


def fresh_store():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    return MatchResultsStore(DB_PATH)


def sample_match(front_end_id="FB5952", internal_id="50076228"):
    return {
        "id": internal_id,
        "frontEndId": front_end_id,
        "status": "INPLAYMATCHENDED",
        "kickOffTime": "2026-09-30T02:45:00.000+08:00",
        "homeTeam": {"id": "H1", "name_en": "Home FC", "name_ch": "主隊"},
        "awayTeam": {"id": "A1", "name_en": "Away FC", "name_ch": "客隊"},
        "results": [
            # stageId 2/4/5 x resultType 1/2/4 = 9 long rows
            {"stageId": 2, "resultType": 1, "homeResult": 1, "awayResult": 0,
             "resultConfirmType": 1, "payoutConfirmed": True},
            {"stageId": 4, "resultType": 1, "homeResult": 2, "awayResult": 1,
             "resultConfirmType": 1, "payoutConfirmed": True},
            {"stageId": 5, "resultType": 1, "homeResult": 2, "awayResult": 1,
             "resultConfirmType": None, "payoutConfirmed": None},
            {"stageId": 2, "resultType": 2, "homeResult": 3, "awayResult": 4,
             "resultConfirmType": None, "payoutConfirmed": None},
            {"stageId": 4, "resultType": 2, "homeResult": 5, "awayResult": 6,
             "resultConfirmType": None, "payoutConfirmed": None},
            {"stageId": 5, "resultType": 2, "homeResult": 5, "awayResult": 6,
             "resultConfirmType": None, "payoutConfirmed": None},
            {"stageId": 2, "resultType": 4, "homeResult": None, "awayResult": None,
             "resultConfirmType": None, "payoutConfirmed": None},
            {"stageId": 4, "resultType": 4, "homeResult": None, "awayResult": None,
             "resultConfirmType": None, "payoutConfirmed": None},
            {"stageId": 5, "resultType": 4, "homeResult": None, "awayResult": None,
             "resultConfirmType": None, "payoutConfirmed": None},
        ],
    }


def test_raw_append_only():
    print("=" * 60, "\nTEST 1: raw table is append-only (ingest twice -> 2 raw rows)\n", "=" * 60)
    store = fresh_store()
    m = sample_match()
    store.ingest_results([m])
    store.ingest_results([sample_match()])
    assert store.raw_count() == 2, f"expected 2 raw rows, got {store.raw_count()}"
    with store._conn() as conn:
        ids = [r["id"] for r in conn.execute("SELECT id FROM match_results_raw ORDER BY id")]
    assert ids == sorted(ids) and len(set(ids)) == 2, "raw rows must be distinct appends"
    print("PASS")
    return store


def test_long_flatten():
    print("\n" + "=" * 60, "\nTEST 2: results[] flattens to one long row per (stage_id, result_type)\n", "=" * 60)
    store = fresh_store()
    n = store.ingest_results([sample_match()])
    assert n == 9, f"expected 9 long rows, got {n}"
    df = store.latest_results("FB5952")
    assert len(df) == 9, f"expected 9 latest rows, got {len(df)}"
    assert set(df["stage_id"]) == {2, 4, 5}
    assert set(df["result_type"]) == {1, 2, 4}

    r = df[(df["stage_id"] == 4) & (df["result_type"] == 1)].iloc[0]
    assert r["home_result"] == 2 and r["away_result"] == 1
    c = df[(df["stage_id"] == 5) & (df["result_type"] == 2)].iloc[0]
    assert c["home_result"] == 5 and c["away_result"] == 6
    print(f"PASS ({n} rows)")


def test_null_handling():
    print("\n" + "=" * 60, "\nTEST 3: resultType=4 with null results stores NULL, no crash\n", "=" * 60)
    store = fresh_store()
    m = {
        "id": "1", "frontEndId": "FE9",
        "results": [{"stageId": 4, "resultType": 4, "homeResult": None,
                     "awayResult": None, "resultConfirmType": None,
                     "payoutConfirmed": None}],
    }
    n = store.ingest_results([m])
    assert n == 1
    row = store.latest_results("FE9").iloc[0]
    assert row["result_type"] == 4
    assert row["home_result"] is None and row["away_result"] is None
    assert row["result_confirm_type"] is None and row["payout_confirmed"] is None
    # missing `results` key entirely must also be graceful
    assert store.ingest_results([{"id": "2", "frontEndId": "FE8"}]) == 0
    print("PASS")


def test_resolve_90min():
    print("\n" + "=" * 60, "\nTEST 4: resolve_90min_result basis depends on explicit assumption\n", "=" * 60)
    store = fresh_store()
    store.ingest_results([sample_match()])

    assumed = store.resolve_90min_result("FB5952", assume_no_extra_time=True)
    assert assumed["settlement_basis"] == "90", assumed
    assert assumed["home"] == 2 and assumed["away"] == 1, assumed
    assert assumed["provenance"] == [(5, 1, 2, 1)], assumed["provenance"]

    unknown = store.resolve_90min_result("FB5952", assume_no_extra_time=False)
    assert unknown["settlement_basis"] == "unknown", unknown
    assert unknown["home"] == 2 and unknown["away"] == 1
    assert len(unknown["provenance"]) == 3, unknown["provenance"]

    empty = store.resolve_90min_result("DOES_NOT_EXIST", assume_no_extra_time=True)
    assert empty["settlement_basis"] == "unknown" and empty["home"] is None
    print("PASS")


def test_latest_one_row_per_key():
    print("\n" + "=" * 60, "\nTEST 5: latest_results returns one row per (match_id, stage_id, result_type)\n", "=" * 60)
    store = fresh_store()
    store.ingest_results([sample_match()])
    store.ingest_results([sample_match()])  # second capture, later timestamp
    df = store.latest_results("FB5952")
    keys = list(zip(df["match_id"], df["stage_id"], df["result_type"]))
    assert len(keys) == len(set(keys)) == 9, f"expected 9 unique keys, got {len(keys)}/{len(set(keys))}"
    # all returned rows must be the newest capture
    assert df["captured_at"].nunique() == 1, "latest_results must pin a single newest capture"
    print("PASS")


if __name__ == "__main__":
    test_raw_append_only()
    test_long_flatten()
    test_null_handling()
    test_resolve_90min()
    test_latest_one_row_per_key()
    print("\n" + "=" * 60 + "\nALL TESTS PASSED\n" + "=" * 60)
