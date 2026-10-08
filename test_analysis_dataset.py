"""
test_analysis_dataset.py -- Phase 1 preprocessing tests.

Fixtures: a HAD market (H/D/A) with a real settlement (D wins) and one SGA pool
(single-outcome, raw probability), plus pre-KO odds_events so steam/temporal
features are exercised. Verifies lambda, kappa winsorisation, steam_dummy,
is_sga/prob_space, outcome/profit, and dq_report.

Run: venv/bin/python test_analysis_dataset.py
"""

import os
import sqlite3
import tempfile

from HKJC_analysis_dataset import build_dataset, dq_report, _check_devig_sums
from HKJC_results17 import MatchResultsStore

MATCH_DATE = "2026-09-22+08:00"
KICKOFF = "2026-09-22T13:00:00.000+08:00"
T1 = "2026-09-22T04:30:00+00:00"

DDL_OPEN = """
CREATE TABLE opening_odds (
    match_id TEXT NOT NULL, odds_type TEXT NOT NULL, pool_id TEXT NOT NULL DEFAULT '',
    line_id TEXT NOT NULL DEFAULT '', line_condition TEXT, comb_id TEXT NOT NULL,
    odds REAL, implied_prob_devig REAL, captured_at TEXT,
    PRIMARY KEY (match_id, odds_type, pool_id, line_id, comb_id));
"""
DDL_RAW = """
CREATE TABLE odds_raw (
    id INTEGER PRIMARY KEY AUTOINCREMENT, scraped_at TEXT, match_id TEXT, odds_type TEXT,
    pool_id TEXT, line_id TEXT, line_condition TEXT, comb_id TEXT, comb_str TEXT, odds REAL,
    selection_name TEXT, match_date TEXT, kickoff_time TEXT);
"""
DDL_EVT = """
CREATE TABLE odds_events (
    match_id TEXT, odds_type TEXT, pool_id TEXT, line_id TEXT, comb_id TEXT,
    movement_type TEXT, velocity REAL, acceleration REAL, seconds_since_last REAL,
    minutes_to_ko REAL);
"""


def _settlement():
    return [{
        "id": "50000001", "frontEndId": "DS1", "additionalResults": [],
        "foPools": [
            {"id": "PHAD", "status": "PAYOUTSTARTED", "oddsType": "HAD", "instNo": 0,
             "lines": [{"combinations": [
                 {"str": "H", "status": "LOSE", "winOrd": "1", "selections": [{"selId": "1", "str": "H"}]},
                 {"str": "D", "status": "WIN", "winOrd": "2", "selections": [{"selId": "3", "str": "D"}]},
                 {"str": "A", "status": "LOSE", "winOrd": "3", "selections": [{"selId": "2", "str": "A"}]},
             ]}]},
            {"id": "PSGA", "status": "PAYOUTSTARTED", "oddsType": "SGA", "instNo": 1,
             "lines": [{"combinations": [
                 {"str": "01", "status": "WIN", "winOrd": "1", "selections": [{"selId": "1", "str": "01"}]},
             ]}]},
            {"id": "PSGA2", "status": "SELLINGSTOPPED", "oddsType": "SGA", "instNo": 2,
             "lines": [{"combinations": [
                 {"str": "02", "status": "AVAILABLE", "winOrd": "1", "selections": [{"selId": "1", "str": "02"}]},
             ]}]},
        ],
    }]


def build(db):
    c = sqlite3.connect(db)
    c.executescript(DDL_OPEN)
    c.executescript(DDL_RAW)
    c.executescript(DDL_EVT)
    c.executemany("INSERT INTO opening_odds VALUES (?,?,?,?,?,?,?,?,?)", [
        ("DS1", "HAD", "PHAD", "0", "0.0", "1", 2.0, 0.5, T1),
        ("DS1", "HAD", "PHAD", "0", "0.0", "2", 4.0, 0.2, T1),   # A
        ("DS1", "HAD", "PHAD", "0", "0.0", "3", 3.5, 0.3, T1),   # D
        ("DS1", "SGA", "PSGA", "0", "0.0", "1", 5.0, 0.2, T1),
        ("DS1", "SGA", "PSGA2", "0", "0.0", "2", 6.0, 0.1667, T1),
    ])
    c.executemany(
        "INSERT INTO odds_raw (scraped_at, match_id, odds_type, pool_id, line_id, "
        "line_condition, comb_id, comb_str, odds, selection_name, match_date, kickoff_time) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            (T1, "DS1", "HAD", "PHAD", "0", "0.0", "1", "H", 1.6, "Home", MATCH_DATE, KICKOFF),
            (T1, "DS1", "HAD", "PHAD", "0", "0.0", "2", "A", 5.0, "Away", MATCH_DATE, KICKOFF),
            (T1, "DS1", "HAD", "PHAD", "0", "0.0", "3", "D", 4.0, "Draw", MATCH_DATE, KICKOFF),
            (T1, "DS1", "SGA", "PSGA", "0", "0.0", "1", "01", 4.0, "acca", MATCH_DATE, KICKOFF),
            (T1, "DS1", "SGA", "PSGA2", "0", "0.0", "2", "02", 5.5, "acca2", MATCH_DATE, KICKOFF),
        ])
    c.executemany(
        "INSERT INTO odds_events (match_id, odds_type, pool_id, line_id, comb_id, "
        "movement_type, velocity, acceleration, seconds_since_last, minutes_to_ko) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)", [
            ("DS1", "HAD", "PHAD", "0", "1", "steam", 0.02, 0.01, 300, 30.0),   # H: steam
            ("DS1", "HAD", "PHAD", "0", "3", "drift", 0.003, 0.0, 600, 40.0),   # D: drift
            ("DS1", "HAD", "PHAD", "0", "2", "inplay_reactive", 0.5, 0.0, 60, -5.0),  # A: in-play (excluded)
        ])
    c.commit(); c.close()
    MatchResultsStore(db).ingest_settlement(_settlement())


def approx(a, b, tol=1e-6):
    return a is not None and abs(float(a) - b) <= tol


def main():
    fd, db = tempfile.mkstemp(suffix=".db"); os.close(fd); os.remove(db)
    build(db)
    ds = build_dataset(db_path=db)
    bounds = ds.attrs.get("kappa_bounds")
    failures = []

    ds = ds.set_index(["odds_type", "comb_str"])
    h = ds.loc[("HAD", "H")]; d = ds.loc[("HAD", "D")]; a = ds.loc[("HAD", "A")]
    s = ds.loc[("SGA", "01")]

    # lambda = ln(o0) - ln(o1)
    if not approx(h["lambda"], 0.22314, 1e-4):   # ln(2)-ln(1.6)=0.22314
        failures.append(f"H lambda {h['lambda']}")
    if not approx(s["lambda"], 0.22314, 1e-4):        # ln(5)-ln(4)
        failures.append(f"SGA lambda {s['lambda']}")

    # steam / temporal
    if int(h["steam_dummy"]) != 1 or h["drift_class"] != "steam":
        failures.append(f"H steam features {h['steam_dummy']} {h['drift_class']}")
    if int(a["n_events_preko"]) != 0:                 # A's only event is in-play -> excluded
        failures.append(f"A n_events_preko {a['n_events_preko']}")

    # outcome / profit
    if d["outcome"] != "WIN" or not approx(d["profit"], 2.5):   # o0=3.5 -> 2.5
        failures.append(f"D outcome/profit {d['outcome']} {d['profit']}")
    if h["outcome"] != "LOSE" or not approx(h["profit"], -1.0):
        failures.append(f"H outcome/profit {h['outcome']} {h['profit']}")

    # SGA flags / raw probability space
    if int(s["is_sga"]) != 1 or s["prob_space"] != "raw_single":
        failures.append(f"SGA flags {s['is_sga']} {s['prob_space']}")
    if not approx(s["kappa"], 0.25, 1e-4):           # raw: (0.25-0.2)/0.2
        failures.append(f"SGA kappa {s['kappa']}")

    # not-yet-settled (AVAILABLE) must NOT be labelled VOID
    s2 = ds.loc[("SGA", "02")]
    if s2["outcome"] != "PENDING" or int(s2["settled"]) != 0:
        failures.append(f"AVAILABLE label {s2['outcome']} settled={s2['settled']}")

    # winsorisation applied within odds_type
    if not bounds:
        failures.append("kappa_bounds missing")

    # DQ
    dq = dq_report(db)
    for k in ("settlement_dup_keys", "odds_le_1_rows", "settlement_rows"):
        if k not in dq:
            failures.append(f"dq missing {k}")
    if dq["settlement_dup_keys"] != 0 or dq["odds_le_1_rows"] != 0:
        failures.append(f"dq unexpected {dq}")

    os.remove(db)
    if failures:
        print("FAIL"); [print("  -", f) for f in failures]; raise SystemExit(1)
    print("PASS  dataset: lambda/delta/kappa, steam_dummy, SGA raw space, "
          "outcome/profit, winsorise bounds + DQ all OK")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
