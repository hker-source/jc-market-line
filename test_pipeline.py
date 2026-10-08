"""
test_pipeline.py (updated for HKJC_warehouse135 / HKJC_live_data14 / HKJC_odds_modelling15)

Old version tested OddsTimeSeriesModel / ingest_raw / batch_id, none of
which exist anymore -- this replaces it. No live site needed; everything
runs on synthetic payloads shaped like HKJC_odds13's real GraphQL output.
"""

from pathlib import Path

import HKJC_warehouse135 as wh_mod
import HKJC_odds_modelling15 as model_mod
from HKJC_warehouse135 import HKJCWarehouse
from HKJC_odds_modelling15 import LiveOddsModel

_ROOT = Path(__file__).parent

DB_PATH = "test_pipeline.db"


def _payload(h, d, a, is_live=False, hs=0, aws=0, match_id="999", front_end_id="FEX"):
    return {"HAD": {"data": {"matches": [{
        "id": match_id, "frontEndId": front_end_id,
        "homeTeam": {"name_en": "TeamA"}, "awayTeam": {"name_en": "TeamB"},
        "tournament": {"name_en": "TestLeague"}, "venue": {"name_en": "Stadium"},
        "matchDate": "2026-09-19", "kickOffTime": "10:00:00",
        "status": "LIVE" if is_live else "PRE",
        "runningResult": {"homeScore": hs, "awayScore": aws, "homeCorner": 0, "awayCorner": 0},
        "foPools": [{"id": "P1", "oddsType": "HAD", "status": "ACTIVE", "inplay": is_live,
            "lines": [{"lineId": "L1", "condition": "0.0", "combinations": [
                {"combId": "H", "str": "H", "status": "ACTIVE", "currentOdds": h,
                 "selections": [{"str": "H", "name_en": "Home"}]},
                {"combId": "D", "str": "D", "status": "ACTIVE", "currentOdds": d,
                 "selections": [{"str": "D", "name_en": "Draw"}]},
                {"combId": "A", "str": "A", "status": "ACTIVE", "currentOdds": a,
                 "selections": [{"str": "A", "name_en": "Away"}]},
            ]}]}]}]}}}


def test_warehouse_ingest_keeps_real_ids():
    print("=" * 60, "\nTEST 1: Warehouse ingest uses real IDs, not display strings\n", "=" * 60)
    wh = HKJCWarehouse(DB_PATH)
    n = wh.ingest(_payload(2.0, 3.2, 3.8))
    assert n == 3
    df = wh.latest(match_id="FEX")
    assert set(df["comb_id"]) == {"H", "D", "A"}
    assert df["match_id"].iloc[0] == "FEX"
    assert "scraped_at" in df.columns
    print("PASS")


def test_warehouse_is_append_only():
    print("\n" + "=" * 60, "\nTEST 2: odds_raw never overwrites, only appends\n", "=" * 60)
    wh = HKJCWarehouse(DB_PATH)
    before = wh.row_count()
    wh.ingest(_payload(1.9, 3.3, 3.9))
    after = wh.row_count()
    assert after == before + 3, "ingest must append, not upsert"
    print(f"PASS ({before} -> {after})")


def test_opening_line_no_event():
    print("\n" + "=" * 60, "\nTEST 3: first sighting of a selection = opening line, no event\n", "=" * 60)
    import os
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    wh = HKJCWarehouse(DB_PATH)
    model = LiveOddsModel(DB_PATH)
    wh.ingest(_payload(2.0, 3.2, 3.8))
    n_events = model.ingest_from_warehouse_df(wh.latest(match_id="FEX"))
    assert n_events == 0
    print("PASS")


def test_devig_sums_to_one():
    print("\n" + "=" * 60, "\nTEST 4: de-vigged probabilities sum to 1 within a market\n", "=" * 60)
    model = LiveOddsModel(DB_PATH)
    rows = [
        model_mod.RawRow("FEX", "HAD", "L1", "H", "Home", 2.0, "2026-09-19T10:00:00+00:00"),
        model_mod.RawRow("FEX", "HAD", "L1", "D", "Draw", 3.2, "2026-09-19T10:00:00+00:00"),
        model_mod.RawRow("FEX", "HAD", "L1", "A", "Away", 3.8, "2026-09-19T10:00:00+00:00"),
    ]
    probs = model._devig(rows)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    print(f"PASS (sum={sum(probs.values()):.6f})")


def test_steam_classification():
    print("\n" + "=" * 60, "\nTEST 5: fast pre-match move classifies as steam\n", "=" * 60)
    wh = HKJCWarehouse(DB_PATH)
    model = LiveOddsModel(DB_PATH)
    wh.ingest(_payload(1.65, 3.5, 4.6))  # sharp shorten on Home
    n_events = model.ingest_from_warehouse_df(wh.latest(match_id="FEX"))
    assert n_events == 3
    steam = model.steam_moves(match_id="FEX")
    assert (steam["movement_type"] == "steam").any()
    print("PASS")


def test_inplay_anticipatory_vs_reactive():
    print("\n" + "=" * 60, "\nTEST 6: in-play — score unchanged = anticipatory, score changed = reactive\n", "=" * 60)
    wh = HKJCWarehouse(DB_PATH)
    model = LiveOddsModel(DB_PATH)

    wh.ingest(_payload(1.5, 3.9, 5.5, is_live=True, hs=0, aws=0))
    model.ingest_from_warehouse_df(wh.latest(match_id="FEX"))
    events = model.steam_moves(match_id="FEX")
    assert (events["movement_type"] == "inplay_anticipatory").any(), "score-unchanged in-play move should be anticipatory"

    wh.ingest(_payload(1.3, 4.5, 8.0, is_live=True, hs=1, aws=0))
    model.ingest_from_warehouse_df(wh.latest(match_id="FEX"))
    # reactive events are deliberately excluded from steam_moves(); check odds_events directly
    with model._conn() as conn:
        row = conn.execute(
            "SELECT movement_type FROM odds_events WHERE match_id='FEX' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row["movement_type"] == "inplay_reactive", f"expected inplay_reactive, got {row['movement_type']}"
    print("PASS")


def test_noise_floor_suppresses_micro_moves():
    print("\n" + "=" * 60, "\nTEST 7: sub-threshold odds jitter is not logged as an event\n", "=" * 60)
    wh = HKJCWarehouse(DB_PATH)
    model = LiveOddsModel(DB_PATH)
    wh.ingest(_payload(2.00, 3.20, 3.80, match_id="777", front_end_id="FEY"))
    model.ingest_from_warehouse_df(wh.latest(match_id="FEY"))
    wh.ingest(_payload(2.001, 3.199, 3.801, match_id="777", front_end_id="FEY"))  # negligible
    n_events = model.ingest_from_warehouse_df(wh.latest(match_id="FEY"))
    assert n_events == 0, "tiny float jitter should not create events"
    print("PASS")


def test_clv_and_market_summary():
    print("\n" + "=" * 60, "\nTEST 8: CLV + market_summary aggregation\n", "=" * 60)
    model = LiveOddsModel(DB_PATH)
    clv = model.clv("FEX", "HAD", "H")
    assert clv is not None and "clv_prob_shift" in clv
    summary = model.market_summary("FEX")
    assert set(summary["comb_id"]) == {"H", "D", "A"}
    assert "steam_count" in summary.columns and "anticipatory_count" in summary.columns
    print(f"PASS (CLV={clv})")


def test_minutes_to_ko_real_hkjc_format():
    print("\n" + "=" * 60, "\nTEST 10: minutes_to_ko parses HKJC's real matchDate/kickOffTime shapes\n", "=" * 60)
    mtk = model_mod._minutes_to_ko(
        "2026-09-27+08:00",
        "2026-09-27T02:45:00.000+08:00",
        "2026-09-26T17:06:41+00:00",
    )
    assert mtk is not None, "full-ISO kickOffTime must parse, not silently return None"
    assert abs(mtk - (98 + 19 / 60)) < 0.01, f"kickoff 18:45Z minus event 17:06:41Z is ~98.32 min, got {mtk}"
    assert model_mod._minutes_to_ko("2026-09-19", "10:00:00", "2026-09-19T09:00:00+00:00") == 60.0
    assert model_mod._minutes_to_ko(None, None, "2026-09-19T09:00:00+00:00") is None
    assert model_mod._minutes_to_ko(float("nan"), float("nan"), "2026-09-19T09:00:00+00:00") is None
    print(f"PASS (real-shape mtk={mtk:.2f})")


def test_no_duplicate_diff_logic_in_poller():
    print("\n" + "=" * 60, "\nTEST 9: 1.4 no longer defines its own change-detection\n", "=" * 60)
    poller_src = (_ROOT / "HKJC_live_data14.py").read_text()
    assert "detect_changes" not in poller_src, "poller must not re-implement diffing"
    assert "INSERT INTO odds_events" not in poller_src, "poller must not write events itself"
    assert "model.ingest_from_warehouse_df" in poller_src, "poller must delegate to the one diff engine"
    print("PASS")


if __name__ == "__main__":
    import os
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    tests = [
        test_warehouse_ingest_keeps_real_ids,
        test_warehouse_is_append_only,
        test_opening_line_no_event,
        test_devig_sums_to_one,
        test_steam_classification,
        test_inplay_anticipatory_vs_reactive,
        test_noise_floor_suppresses_micro_moves,
        test_clv_and_market_summary,
        test_minutes_to_ko_real_hkjc_format,
        test_no_duplicate_diff_logic_in_poller,
    ]
    for t in tests:
        t()
    print("\n" + "=" * 60 + "\nALL TESTS PASSED\n" + "=" * 60)
