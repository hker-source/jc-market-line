"""
test_pnl.py -- verifies HKJC_backtest_pnl math on a synthetic settled market.

Mirrors reality: odds carry numeric comb_id (1/2) plus comb_str (H/L); the
settlement keys on comb_str. Market HIL / line 0 / pool PHIL.
  opening_odds : H 2.0 / L 2.0            -> p0 = 0.5 / 0.5
  odds_raw last pre-KO: H 1.6 / L 2.5     -> p1_H = 0.625/1.025 = 0.609756...
  settlement   : H WIN, L LOSE
Expected H: profit = o0-1 = 1.0 ; drift_abs = +0.109756 ; drift_coeff = +0.219512 ; EV = +0.219512
Expected L: profit = -1.0 ; ROI = 0.0 ; win_rate = 0.5

Run: venv/bin/python test_pnl.py
"""

import os
import sqlite3
import tempfile

from HKJC_backtest_pnl import pnl_table, pnl_summary, clv_vs_outcome
from HKJC_results17 import MatchResultsStore

MATCH_DATE = "2026-09-22+08:00"
KICKOFF = "2026-09-22T13:00:00.000+08:00"
T1 = "2026-09-22T04:30:00+00:00"

OPENING_DDL = """
CREATE TABLE opening_odds (
    match_id TEXT NOT NULL, odds_type TEXT NOT NULL, pool_id TEXT NOT NULL DEFAULT '',
    line_id TEXT NOT NULL DEFAULT '', line_condition TEXT, comb_id TEXT NOT NULL,
    odds REAL, implied_prob_devig REAL, captured_at TEXT,
    PRIMARY KEY (match_id, odds_type, pool_id, line_id, comb_id));
"""
RAW_DDL = """
CREATE TABLE odds_raw (
    id INTEGER PRIMARY KEY AUTOINCREMENT, scraped_at TEXT, match_id TEXT, odds_type TEXT,
    pool_id TEXT, line_id TEXT, line_condition TEXT, comb_id TEXT, comb_str TEXT, odds REAL,
    selection_name TEXT, match_date TEXT, kickoff_time TEXT);
"""


def _payload():
    return [{
        "id": "50000001", "frontEndId": "PNL1", "additionalResults": [],
        "foPools": [{
            "id": "PHIL", "status": "PAYOUTSTARTED", "oddsType": "HIL", "instNo": 0,
            "lines": [{"combinations": [
                {"str": "H", "status": "WIN", "winOrd": "1",
                 "selections": [{"selId": "1", "str": "H", "name_en": "High"}]},
                {"str": "L", "status": "LOSE", "winOrd": "2",
                 "selections": [{"selId": "2", "str": "L", "name_en": "Low"}]},
            ]}],
        }],
    }]


def build(db):
    c = sqlite3.connect(db)
    c.executescript(OPENING_DDL)
    c.executescript(RAW_DDL)
    c.executemany("INSERT INTO opening_odds VALUES (?,?,?,?,?,?,?,?,?)", [
        ("PNL1", "HIL", "PHIL", "0", "2.5", "1", 2.0, 0.5, T1),
        ("PNL1", "HIL", "PHIL", "0", "2.5", "2", 2.0, 0.5, T1),
    ])
    c.executemany(
        "INSERT INTO odds_raw (scraped_at, match_id, odds_type, pool_id, line_id, "
        "line_condition, comb_id, comb_str, odds, selection_name, match_date, kickoff_time) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
            (T1, "PNL1", "HIL", "PHIL", "0", "2.5", "1", "H", 1.6, "High", MATCH_DATE, KICKOFF),
            (T1, "PNL1", "HIL", "PHIL", "0", "2.5", "2", "L", 2.5, "Low", MATCH_DATE, KICKOFF),
        ])
    c.commit(); c.close()
    MatchResultsStore(db).ingest_settlement(_payload())


def approx(a, b, tol=1e-6):
    return a is not None and abs(float(a) - b) <= tol


def main():
    fd, db = tempfile.mkstemp(suffix=".db"); os.close(fd); os.remove(db)
    build(db)
    t = pnl_table(db_path=db)
    df = t.set_index("comb_str")
    failures = []

    p1h = 0.625 / 1.025
    h, l = df.loc["H"], df.loc["L"]
    if not approx(h["opening_odds"], 2.0): failures.append("H o0")
    if not approx(h["closing_prob"], p1h): failures.append(f"H p1 {h['closing_prob']}")
    if not approx(h["drift_abs"], p1h - 0.5): failures.append(f"H drift_abs {h['drift_abs']}")
    if not approx(h["drift_coeff"], (p1h - 0.5) / 0.5): failures.append(f"H drift_coeff {h['drift_coeff']}")
    if not approx(h["ev_closing"], p1h * 2.0 - 1.0): failures.append(f"H EV {h['ev_closing']}")
    if not approx(h["profit"], 1.0): failures.append(f"H profit {h['profit']}")
    if not approx(l["profit"], -1.0): failures.append(f"L profit {l['profit']}")

    s = pnl_summary(t)
    if (s["n_settled"], s["n_win"], s["n_lose"]) != (2, 1, 1):
        failures.append(f"counts {s}")
    if not approx(s["roi"], 0.0): failures.append(f"ROI {s['roi']}")
    if not approx(s["win_rate"], 0.5): failures.append(f"win_rate {s['win_rate']}")

    cv = clv_vs_outcome(t)
    if set(cv["outcome"]) != {"WIN", "LOSE"}:
        failures.append(f"clv_vs_outcome {list(cv['outcome'])}")

    os.remove(db)
    if failures:
        print("FAIL"); [print("  -", f) for f in failures]; raise SystemExit(1)
    print("PASS  P&L: H profit=+1.0 (o0-1), L profit=-1.0, ROI=0.0, "
          f"drift_coeff(H)={(p1h-0.5)/0.5:.4f}, EV(H)={p1h*2-1:.4f}")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
