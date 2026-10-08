"""Plain-script tests for HKJC_backtest_clv.py.

Run:  venv/bin/python test_backtest_clv.py
Builds a temporary SQLite DB with the minimal opening_odds + odds_raw
tables, then asserts de-vig, CLV sign/magnitude, pre-kickoff filtering,
coverage accounting, and summary grouping.
"""

import os
import sqlite3
import tempfile

import pandas as pd

from HKJC_backtest_clv import clv_table, clv_summary, coverage

MATCH_DATE = "2026-09-22+08:00"
KICKOFF = "2026-09-22T13:00:00.000+08:00"   # 13:00 HKT == 05:00 UTC
T0 = "2026-09-22T03:00:00+00:00"            # earliest pre-KO observation
T1 = "2026-09-22T04:30:00+00:00"            # last pre-KO observation (30min to KO)
POST_KO = "2026-09-22T05:30:00+00:00"       # 30min after KO (minutes_to_ko < 0)

OPENING_DDL = """
CREATE TABLE opening_odds (
    match_id TEXT NOT NULL, odds_type TEXT NOT NULL,
    pool_id TEXT NOT NULL DEFAULT '',
    line_id TEXT NOT NULL DEFAULT '', line_condition TEXT,
    comb_id TEXT NOT NULL, odds REAL, implied_prob_devig REAL, captured_at TEXT,
    PRIMARY KEY (match_id, odds_type, pool_id, line_id, comb_id)
);
"""

RAW_DDL = """
CREATE TABLE odds_raw (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scraped_at TEXT, match_id TEXT, odds_type TEXT, pool_id TEXT, line_id TEXT,
    line_condition TEXT, comb_id TEXT, odds REAL,
    selection_name TEXT, match_date TEXT, kickoff_time TEXT
);
"""


def build_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(OPENING_DDL)
    conn.executescript(RAW_DDL)

    opening = [
        # M1 HAD 0.0: A=0.5, B=0.5
        ("M1", "HAD", "P0", "0", "0.0", "H", 2.0, 0.5, T0),
        ("M1", "HAD", "P0", "0", "0.0", "A", 2.0, 0.5, T0),
        # M2 HAD 0.0: only the away selection has an opening row
        ("M2", "HAD", "P0", "0", "0.0", "A", 2.0, 0.5, T0),
        # M3 HAD 0.0: opening row but no pre-KO raw observation
        ("M3", "HAD", "P0", "0", "0.0", "H", 1.8, 0.6, T0),
    ]
    conn.executemany(
        "INSERT INTO opening_odds VALUES (?,?,?,?,?,?,?,?,?)", opening)

    raw = [
        # M1 at T0 (earlier): both 2.0 -> de-vig 0.5/0.5  (test 1)
        (T0, "M1", "HAD", "P0", "0", "0.0", "H", 2.0, "Home", MATCH_DATE, KICKOFF),
        (T0, "M1", "HAD", "P0", "0", "0.0", "A", 2.0, "Away", MATCH_DATE, KICKOFF),
        # M1 at T1 (later, still pre-KO): H=2.0, A=3.0 -> de-vig 0.6/0.4 (test 2)
        (T1, "M1", "HAD", "P0", "0", "0.0", "H", 2.0, "Home", MATCH_DATE, KICKOFF),
        (T1, "M1", "HAD", "P0", "0", "0.0", "A", 3.0, "Away", MATCH_DATE, KICKOFF),
        # M1 POST-KO (must be ignored, test 3)
        (POST_KO, "M1", "HAD", "P0", "0", "0.0", "H", 10.0, "Home", MATCH_DATE, KICKOFF),
        (POST_KO, "M1", "HAD", "P0", "0", "0.0", "A", 1.5, "Away", MATCH_DATE, KICKOFF),
        # M2 pre-KO: no opening row for H (test 4)
        (T1, "M2", "HAD", "P0", "0", "0.0", "H", 2.0, "Home", MATCH_DATE, KICKOFF),
        (T1, "M2", "HAD", "P0", "0", "0.0", "A", 2.0, "Away", MATCH_DATE, KICKOFF),
        # M3 deliberately has no raw rows
    ]
    conn.executemany(
        "INSERT INTO odds_raw (scraped_at, match_id, odds_type, pool_id, line_id, "
        "line_condition, comb_id, odds, selection_name, match_date, kickoff_time) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)", raw)
    conn.commit()
    conn.close()


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def main():
    tmpdir = tempfile.mkdtemp(prefix="clv_test_")
    db = os.path.join(tmpdir, "test_clv.db")
    build_db(db)

    df = clv_table(db_path=db)
    df = df.set_index(["match_id", "comb_id"]).sort_index()
    failures = []

    # 1. de-vig correctness: 2.0 / 2.0 -> 0.5 / 0.5
    #    Verified directly on the T0 (earliest) M1 observations, since the
    #    closing line itself legitimately uses the later T1 de-vig.
    conn = sqlite3.connect(db)
    t0 = pd.read_sql_query(
        "SELECT comb_id, odds FROM odds_raw WHERE match_id='M1' AND scraped_at=?",
        conn, params=(T0,))
    conn.close()
    imp = 1.0 / t0["odds"]
    devig = imp / imp.sum()
    if not (list(devig.round(9)) == [0.5, 0.5]):
        failures.append(f"1. de-vig of 2.0/2.0 != 0.5/0.5 -> {list(devig)}")

    # 2. CLV sign/magnitude: opening 0.5 -> closing 0.6 => +0.1
    m1h = df.loc[("M1", "H")]
    if not approx(m1h["closing_prob"], 0.6):
        failures.append(f"2a. M1/H closing_prob expected 0.6 got {m1h['closing_prob']}")
    if not approx(m1h["clv_prob_shift"], 0.1):
        failures.append(f"2b. M1/H clv_prob_shift expected +0.1 got {m1h['clv_prob_shift']}")
    if not approx(m1h["clv_odds_ratio"], 0.0):
        failures.append(f"2c. M1/H clv_odds_ratio expected 0.0 got {m1h['clv_odds_ratio']}")

    # 3. post-kickoff row ignored: H must use T1 (2.0), not POST_KO (10.0)
    if m1h["last_pre_ko_scraped_at"] != T1:
        failures.append(
            f"3a. M1/H used {m1h['last_pre_ko_scraped_at']} not last pre-KO {T1}")
    if approx(m1h["closing_odds"], 10.0):
        failures.append("3b. post-kickoff row leaked into closing")

    # 4. coverage: M2/H has no opening row -> excluded and counted
    if ("M2", "H") in df.index:
        failures.append("4a. M2/H (no opening row) unexpectedly included")
    cov = coverage(df)
    if cov["excluded_no_opening"] != 1:
        failures.append(f"4b. excluded_no_opening expected 1 got {cov['excluded_no_opening']}")
    if cov["excluded_no_closing"] != 1:
        failures.append(f"4c. excluded_no_closing expected 1 got {cov['excluded_no_closing']}")
    if cov["included"] != 3:
        failures.append(f"4d. included expected 3 got {cov['included']}")
    if cov["total_selections"] != 5:
        failures.append(f"4e. total_selections expected 5 got {cov['total_selections']}")

    # 5. clv_summary groups correctly
    summary = clv_summary(df)
    type_rows = summary[summary["group_kind"] == "odds_type"]
    if len(type_rows) != 1:
        failures.append(f"5a. expected 1 odds_type group got {len(type_rows)}")
    else:
        row = type_rows.iloc[0]
        if row["odds_type"] != "HAD" or int(row["count"]) != 3:
            failures.append(f"5b. HAD group wrong: {row.to_dict()}")
        # shifts: M1/H +0.1, M1/A -0.1, M2/A 0.0 -> mean 0.0, positive_rate 1/3
        if not approx(row["mean_clv_prob_shift"], 0.0, tol=1e-9):
            failures.append(f"5c. mean shift expected 0.0 got {row['mean_clv_prob_shift']}")
        if not approx(row["positive_rate"], 1.0 / 3.0, tol=1e-9):
            failures.append(f"5d. positive_rate expected 1/3 got {row['positive_rate']}")

    sel_rows = summary[summary["group_kind"] == "selection"]
    if len(sel_rows) != 2:
        failures.append(f"5e. expected 2 selection groups (HAD/H, HAD/A) got {len(sel_rows)}")

    # ------------------------------------------------------------------ #
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)

    print("PASS  (5/5 test groups)")
    print(f"  included={cov['included']} excluded={cov['excluded']} "
          f"(no_opening={cov['excluded_no_opening']}, no_closing={cov['excluded_no_closing']})")
    print("  de-vig 2.0/2.0 -> 0.50/0.50; M1/H closing 0.6, clv_prob_shift +0.10; "
          "post-KO ignored; summary groups OK")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
