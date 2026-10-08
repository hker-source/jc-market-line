"""
HKJC_backtest_pnl.py -- realized P&L from opening odds + closing odds + settlement.

Pipeline
--------
    opening_odds  (stake price)            }  from HKJC_backtest_clv.clv_table()
    closing odds -> p1 (last pre-kickoff)  }
    match_settlements.comb_status          }  from HKJC_results17 (foPools resultOnly)
        -> WIN / LOSE / VOID

JOIN KEY: settlement keys a combination by its `str` ('H'/'A'/'D', '01'), while
the odds pipeline's `comb_id` is a numeric id ('1'/'2'/'3'). So we join on
COMB_STR, not comb_id. Keys are
    (match_id, odds_type, pool_id, comb_str-normalised)
with line_id applied when the settlement carries it (the live resultOnly query
does; the standalone matchResultDetails payload does not).

MATH (per selection i, flat stake = 1)
--------------------------------------
Odds / probability
    o0_i = opening decimal odds
    o1_i = closing decimal odds (last observation with minutes_to_ko > 0)
    p0_i = opening de-vigged implied prob = (1/o0_i) / SUM_{j in group}(1/o0_j)
    p1_i = closing de-vigged implied prob (same grouping)

Cumulative drift
    absolute drift      d_i = p1_i - p0_i
    drift coefficient   k_i = d_i / p0_i = p1_i / p0_i - 1        (relative drift)
    log-odds drift      l_i = ln(o0_i) - ln(o1_i)                 (>0 if shortened)
    odds ratio          r_i = o1_i / o0_i - 1

Settlement & profit (stake 1)
    g_i in {WIN, LOSE, VOID}
    WIN :  profit_i = o0_i - 1        return_i = o0_i
    LOSE:  profit_i = -1              return_i = 0
    VOID:  profit_i =  0              return_i = 1     (stake refunded)

Aggregates
    ROI        = SUM(profit_i) / N
    win rate   = #WIN / (#WIN + #LOSE)
    EV (closing-fair) per bet = p1_i * o0_i - 1     ; mean over bets
    CLV vs outcome = mean(d_i | WIN) vs mean(d_i | LOSE)

Read-only DB access; never mutates the live database.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import pandas as pd

from HKJC_backtest_clv import clv_table, _connect_ro
from HKJC_results17 import MatchResultsStore

DB_DEFAULT = "hkjc_odds.db"
JOIN_KEYS = ["match_id", "odds_type", "pool_key", "comb_key"]


def _norm_comb(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and value != value):
        return None
    s = str(value).strip()
    return str(int(s)) if s.isdigit() else s


def _comb_str_map(db_path: str) -> pd.DataFrame:
    """comb_id -> comb_str bridge, taken from the append-only raw log."""
    with _connect_ro(db_path) as conn:
        return pd.read_sql_query(
            "SELECT DISTINCT match_id, odds_type, COALESCE(pool_id,'') AS pool_key, "
            "COALESCE(line_id,'') AS line_key, comb_id, comb_str FROM odds_raw", conn)


def _settlements_frame(db_path: str) -> pd.DataFrame:
    store = MatchResultsStore(db_path)          # creates tables if absent
    df = store.latest_settlements()
    if df.empty:
        return df
    df = df.copy()
    df["pool_key"] = df["pool_id"].fillna("")
    df["line_key"] = df["line_id"].fillna("")
    df["comb_key"] = df["comb_str"].map(_norm_comb)
    return df


def pnl_table(db_path: str = DB_DEFAULT,
              match_id: Optional[str] = None,
              odds_type: Optional[str] = None) -> pd.DataFrame:
    """Per-selection realized P&L table (see module docstring for the math)."""
    df = clv_table(db_path=db_path, match_id=match_id, odds_type=odds_type)
    if df.empty:
        return df

    df = df.copy()
    df["pool_key"] = df["pool_id"].fillna("")
    df["line_key"] = df["line_id"].fillna("")

    # attach comb_str (settlement keys on str, not the numeric comb_id)
    df = df.merge(_comb_str_map(db_path),
                  on=["match_id", "odds_type", "pool_key", "line_key", "comb_id"],
                  how="left")
    df["comb_key"] = df["comb_str"].map(_norm_comb)

    s = _settlements_frame(db_path)
    if s.empty:
        for c in ("comb_status", "sel_name_en", "inst_no"):
            df[c] = None
        merged = df
    else:
        merged = df.merge(
            s[["match_id", "odds_type", "pool_key", "comb_key", "line_key",
               "comb_status", "sel_name_en", "inst_no"]],
            on=["match_id", "odds_type", "pool_key", "comb_key"],
            how="left", suffixes=("", "_set"),
        )
        if "line_key_set" in merged.columns:
            # Prefer an explicit settlement line_id; if the settlement carried
            # none, only trust markets that have a single line for that (pool, comb).
            has_line = merged["line_key_set"].notna() & (merged["line_key_set"] != "")
            ok_line = merged["line_key"] == merged["line_key_set"]
            single = ~merged.duplicated(JOIN_KEYS, keep=False)
            merged = merged[(has_line & ok_line) | (~has_line & single)].copy()
            merged = merged.drop(columns=["line_key_set"])

    merged = merged.drop_duplicates(subset=JOIN_KEYS + ["line_key"])

    o0 = pd.to_numeric(merged["opening_odds"], errors="coerce")
    o1 = pd.to_numeric(merged["closing_odds"], errors="coerce")
    p0 = pd.to_numeric(merged["opening_prob"], errors="coerce")
    p1 = pd.to_numeric(merged["closing_prob"], errors="coerce")

    merged["drift_abs"] = p1 - p0
    merged["drift_coeff"] = (p1 - p0) / p0.replace(0, pd.NA)
    merged["log_odds_drift"] = [
        (math.log(a) - math.log(b)) if a and b and a > 0 and b > 0 else None
        for a, b in zip(o0, o1)
    ]
    merged["clv_odds_ratio"] = o1 / o0 - 1.0
    merged["ev_closing"] = p1 * o0 - 1.0

    def _profit(row):
        st = row.get("comb_status")
        if st == "WIN":
            return row["opening_odds"] - 1.0
        if st == "LOSE":
            return -1.0
        return 0.0

    merged["profit"] = merged.apply(_profit, axis=1)
    merged["stake"] = 1.0
    merged["outcome"] = merged["comb_status"].apply(
        lambda v: v if v in ("WIN", "LOSE") else ("void/pending" if v is not None else "unsettled"))

    cols = ["match_id", "odds_type", "pool_id", "line_id", "line_condition",
            "comb_id", "comb_str", "selection_name", "inst_no",
            "opening_odds", "closing_odds", "opening_prob", "closing_prob",
            "drift_abs", "drift_coeff", "log_odds_drift", "clv_odds_ratio",
            "ev_closing", "comb_status", "outcome", "stake", "profit"]
    for c in cols:
        if c not in merged.columns:
            merged[c] = None
    return merged[cols].reset_index(drop=True)


def pnl_summary(df: pd.DataFrame) -> Dict[str, Any]:
    settled = df[df["outcome"].isin(["WIN", "LOSE"])].copy()
    wins = settled[settled["outcome"] == "WIN"]
    losses = settled[settled["outcome"] == "LOSE"]
    n = len(settled)
    total_profit = float(settled["profit"].sum()) if n else 0.0
    return {
        "n_rows": int(len(df)),
        "n_settled": int(n),
        "n_win": int(len(wins)),
        "n_lose": int(len(losses)),
        "win_rate": (len(wins) / n) if n else None,
        "total_stake": n,
        "total_profit": total_profit,
        "roi": (total_profit / n) if n else None,
        "mean_drift_coeff": float(df["drift_coeff"].mean()) if not df.empty else None,
        "mean_clv_odds_ratio": float(df["clv_odds_ratio"].mean()) if not df.empty else None,
        "mean_ev_closing": float(df["ev_closing"].mean()) if not df.empty else None,
    }


def clv_vs_outcome(df: pd.DataFrame) -> pd.DataFrame:
    settled = df[df["outcome"].isin(["WIN", "LOSE"])].copy()
    if settled.empty:
        return pd.DataFrame(columns=["outcome", "n", "mean_drift_abs",
                                     "mean_drift_coeff", "mean_clv_odds_ratio", "mean_ev_closing"])
    return (settled.groupby("outcome", dropna=False)
            .agg(n=("outcome", "size"),
                 mean_drift_abs=("drift_abs", "mean"),
                 mean_drift_coeff=("drift_coeff", "mean"),
                 mean_clv_odds_ratio=("clv_odds_ratio", "mean"),
                 mean_ev_closing=("ev_closing", "mean"))
            .reset_index())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Realized P&L from opening odds + settlement")
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--match")
    ap.add_argument("--odds-type")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    table = pnl_table(args.db, match_id=args.match, odds_type=args.odds_type)
    with pd.option_context("display.width", 240, "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print("MATH: profit(WIN)=o0-1 ; profit(LOSE)=-1 ; profit(VOID)=0 ; "
              "ROI=sum(profit)/N ; drift_coeff=(p1-p0)/p0 ; EV=p1*o0-1")
        print(f"\nrows: {len(table)}")
        if table.empty:
            raise SystemExit(0)
        print("\n=== summary ===")
        for k, v in pnl_summary(table).items():
            print(f"  {k:20s}: {v}")
        print("\n=== CLV vs outcome ===")
        print(clv_vs_outcome(table).to_string(index=False))
        settled = table[table["outcome"].isin(["WIN", "LOSE"])]
        if not settled.empty:
            print(f"\n=== all settled selections (by profit) ===")
            print(settled.sort_values("profit", ascending=False)[
                ["match_id", "odds_type", "line_condition", "comb_str", "selection_name",
                 "opening_odds", "closing_odds", "closing_prob", "drift_coeff",
                 "ev_closing", "outcome", "profit"]].to_string(index=False))
