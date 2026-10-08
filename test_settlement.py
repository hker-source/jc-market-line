"""
test_settlement.py -- tests for the foPools settlement layer in HKJC_results17.py.

Fixture mirrors the real FB5809 (U20 WWC, extra-time cup) matchResultDetails
payload: foPools with resultOnly-style settlement (comb.status WIN/LOSE,
winOrd display order), including 2 SGA pools that share comb str '01', a
multi-line ECD/CHL pool, and an empty additionalResults array.

Run: venv/bin/python test_settlement.py
"""

import os
import tempfile

from HKJC_results17 import MatchResultsStore


def _match():
    return {
        "id": "50076973",
        "frontEndId": "FB5809",
        "additionalResults": [],
        "foPools": [
            # SGA pool 1 (instNo 1) -- LOSE
            {"id": "1980980044", "status": "PAYOUTSTARTED", "oddsType": "SGA", "instNo": 1,
             "lines": [{"combinations": [{"str": "01", "status": "LOSE", "winOrd": "1",
                        "selections": [{"selId": "1", "str": "",
                                        "name_en": "Both teams to score & Italy Women U20 to win",
                                        "name_ch": "兩隊均取得入球 & 意大利女足U20勝"}]}]}]},
            # SGA pool 15 (instNo 15) -- WIN, same comb str '01'
            {"id": "1981023044", "status": "PAYOUTSTARTED", "oddsType": "SGA", "instNo": 15,
             "lines": [{"combinations": [{"str": "01", "status": "WIN", "winOrd": "1",
                        "selections": [{"selId": "1", "str": "",
                                        "name_en": "Under [1.5] goals in the first half & Under [2.5] goals in the match",
                                        "name_ch": "半場總入球少於[1.5]球 & 全場總入球少於[2.5]球"}]}]}]},
            # HAD -- Draw wins
            {"id": "1977556001", "status": "PAYOUTSTARTED", "oddsType": "HAD", "instNo": 0,
             "lines": [{"combinations": [
                 {"str": "A", "status": "LOSE", "winOrd": "3", "selections": [{"selId": "2", "str": "A", "name_en": "Away"}]},
                 {"str": "D", "status": "WIN", "winOrd": "2", "selections": [{"selId": "3", "str": "D", "name_en": "Draw"}]},
                 {"str": "H", "status": "LOSE", "winOrd": "1", "selections": [{"selId": "1", "str": "H", "name_en": "Home"}]},
             ]}]},
            # TTG -- 2 goals
            {"id": "1977566012", "status": "PAYOUTSTARTED", "oddsType": "TTG", "instNo": 0,
             "lines": [{"combinations": [
                 {"str": "0", "status": "LOSE", "winOrd": "1", "selections": [{"selId": "1", "str": "0", "name_en": "0"}]},
                 {"str": "2", "status": "WIN", "winOrd": "3", "selections": [{"selId": "3", "str": "2", "name_en": "2"}]},
             ]}]},
            # OOE -- Even
            {"id": "1977565011", "status": "PAYOUTSTARTED", "oddsType": "OOE", "instNo": 0,
             "lines": [{"combinations": [
                 {"str": "O", "status": "LOSE", "winOrd": "1", "selections": [{"selId": "1", "str": "O", "name_en": "Odd"}]},
                 {"str": "E", "status": "WIN", "winOrd": "2", "selections": [{"selId": "2", "str": "E", "name_en": "Even"}]},
             ]}]},
            # ELH (single-line HiLo) -- Low wins
            {"id": "1984354061", "status": "PAYOUTSTARTED", "oddsType": "ELH", "instNo": 0,
             "lines": [{"combinations": [
                 {"str": "H", "status": "LOSE", "winOrd": "1", "selections": [{"selId": "1", "str": "H", "name_en": "High"}]},
                 {"str": "L", "status": "WIN", "winOrd": "2", "selections": [{"selId": "2", "str": "L", "name_en": "Low"}]},
             ]}]},
            # ECD -- TWO lines: line0 Away wins, line1 Home wins (line_index matters)
            {"id": "1984352049", "status": "PAYOUTSTARTED", "oddsType": "ECD", "instNo": 0,
             "lines": [
                 {"combinations": [
                     {"str": "H", "status": "LOSE", "winOrd": "1", "selections": [{"selId": "1", "str": "H", "name_en": "Home"}]},
                     {"str": "A", "status": "WIN", "winOrd": "2", "selections": [{"selId": "2", "str": "A", "name_en": "Away"}]},
                 ]},
                 {"combinations": [
                     {"str": "H", "status": "WIN", "winOrd": "1", "selections": [{"selId": "1", "str": "H", "name_en": "Home"}]},
                     {"str": "A", "status": "LOSE", "winOrd": "2", "selections": [{"selId": "2", "str": "A", "name_en": "Away"}]},
                 ]},
             ]},
            # ENT -- No Goals wins
            {"id": "1984345036", "status": "PAYOUTSTARTED", "oddsType": "ENT", "instNo": 1,
             "lines": [{"combinations": [
                 {"str": "H", "status": "LOSE", "winOrd": "1", "selections": [{"selId": "1", "str": "H", "name_en": "Home"}]},
                 {"str": "N", "status": "WIN", "winOrd": "3", "selections": [{"selId": "3", "str": "N", "name_en": "No Goals"}]},
             ]}]},
        ],
    }


def main():
    fd, db = tempfile.mkstemp(suffix=".db"); os.close(fd); os.remove(db)
    store = MatchResultsStore(db)
    n = store.ingest_settlement([_match()])

    failures = []

    # 1 -- SGA pools share comb str '01' but must stay distinct (pool_id)
    sga = store.latest_settlements("FB5809")
    sga = sga[sga.odds_type == "SGA"]
    if len(sga) != 2 or sga.pool_id.nunique() != 2:
        failures.append(f"1. SGA rows={len(sga)} pools={sga.pool_id.nunique()} (want 2/2)")
    if set(sga.inst_no) != {1, 15}:
        failures.append(f"1b. inst_no not preserved: {sorted(sga.inst_no)}")
    winners = sga[sga.comb_status == "WIN"]
    if len(winners) != 1 or winners.iloc[0].pool_id != "1981023044":
        failures.append("1c. SGA winner not isolated to pool 1981023044")

    # 2 -- straight single-line winners
    for ot, want in (("HAD", "D"), ("TTG", "2"), ("OOE", "E"), ("ELH", "L"), ("ENT", "N")):
        w = store.settled_winner("FB5809", ot)
        got = w[0]["comb_str"] if w else None
        if got != want:
            failures.append(f"2. {ot} winner expected {want!r} got {got!r}")

    # 3 -- multi-line: line_index disambiguates
    ecd0 = store.settled_winner("FB5809", "ECD", line_index=0)
    ecd1 = store.settled_winner("FB5809", "ECD", line_index=1)
    if not (ecd0 and ecd1 and ecd0[0]["comb_str"] == "A" and ecd1[0]["comb_str"] == "H"):
        failures.append("3. ECD line disambiguation failed")

    # 4 -- comb key normalisation bridges '01' (settlement) <-> '1' (odds)
    if store._norm_comb("01") != "1" or store._norm_comb("H:H") != "H:H":
        failures.append("4. _norm_comb wrong")

    # 5 -- additionalResults ignored gracefully; is_settled true; counts
    if not store.is_settled("FB5809"):
        failures.append("5a. is_settled False")
    if store.settlement_count() != n:
        failures.append("5b. settlement_count mismatch")

    # 6 -- idempotent re-ingest (same keys overwrite, no growth)
    n2 = store.ingest_settlement([_match()])
    if store.settlement_count() != n:
        failures.append(f"6. re-ingest changed row count {n} -> {store.settlement_count()}")

    os.remove(db)

    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    print(f"PASS  (settlement rows={n}, idempotent re-ingest={n2}, SGA pools=2, "
          f"HAD=D TTG=2 OOE=E ELH=L ENT=N, ECD lines disambiguated)")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
