"""
HKJC_backtest_clv.py -- closing-line-value (CLV) backtest.

CLV does NOT need match results or settlement. It measures whether the
market moved toward a selection between the opening snapshot and the last
pre-kickoff observation, so it can be measured today even while the results
endpoint is unavailable.

Method (per selection key match_id, odds_type, pool_id, line_id, comb_id):
  opening_prob  <- opening_odds.implied_prob_devig (already de-vigged)
  closing_prob  <- de-vigged probability of the LAST pre-kickoff row in
                   odds_raw; a row is pre-kickoff when
                   _minutes_to_ko(match_date, kickoff_time, scraped_at) > 0.
                   For each selection take max scraped_at among pre-KO rows,
                   then de-vig across the market group
                   (match_id, odds_type, line_id): implied = 1/odds,
                   prob = implied / sum(implied).  => sums to 1 per group.
  clv_prob_shift = closing_prob - opening_prob
                   positive => market moved TOWARD this selection
                   (the opening price was value).
  clv_odds_ratio = closing_odds / opening_odds - 1

All DB access is read-only (sqlite file:...?mode=ro), so the backtest can
never mutate the live database.

Public API
----------
clv_table(db_path="hkjc_odds.db", match_id=None, odds_type=None) -> DataFrame
    columns: match_id, odds_type, pool_id, line_id, line_condition, comb_id,
             selection_name, opening_odds, opening_prob, closing_odds,
             closing_prob, clv_prob_shift, clv_odds_ratio,
             last_pre_ko_scraped_at
    The DataFrame carries `.attrs` with the coverage breakdown
    (total_selections, excluded_no_opening, excluded_no_closing).

clv_summary(df) -> DataFrame
    one row per odds_type and one row per (odds_type, comb_id) selection,
    with count / mean / median clv_prob_shift and positive_rate.

coverage(df, total_selections=None) -> dict
    included / excluded counts and the exclusion reasons.

CLI: venv/bin/python HKJC_backtest_clv.py [--db=] [--match=] [--odds-type=] [--top=N]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from HKJC_odds_modelling15 import _minutes_to_ko

KEY = ["match_id", "odds_type", "pool_id", "line_id", "comb_id"]
MARKET = ["match_id", "odds_type", "pool_id", "line_id"]
DB_DEFAULT = "hkjc_odds.db"


# --------------------------------------------------------------------------- #
# Read-only connection
# --------------------------------------------------------------------------- #
def _connect_ro(db_path: str = DB_DEFAULT) -> sqlite3.Connection:
    """Open the SQLite DB read-only so a backtest can never mutate it.

    Uses a file: URI so `mode=ro` is honoured, and Path.as_uri() to safely
    percent-encode spaces/special characters in the workspace path.
    """
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _where(filters: Dict[str, Optional[str]], prefix: str = "") -> Tuple[str, List[Any]]:
    clauses, params = [], []
    for col, val in filters.items():
        if val is not None:
            clauses.append(f"{prefix}{col} = ?")
            params.append(val)
    return (f" WHERE {' AND '.join(clauses)}" if clauses else "", params)


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def clv_table(db_path: str = DB_DEFAULT,
              match_id: Optional[str] = None,
              odds_type: Optional[str] = None) -> pd.DataFrame:
    """Build the per-selection CLV table.  See module docstring for columns."""
    filters = {"match_id": match_id, "odds_type": odds_type}
    where, params = _where(filters)

    with _connect_ro(db_path) as conn:
        opening = pd.read_sql_query(
            "SELECT match_id, odds_type, pool_id, line_id, line_condition, comb_id, "
            "odds, implied_prob_devig, captured_at FROM opening_odds" + where,
            conn, params=params,
        )
        raw = pd.read_sql_query(
            "SELECT id, match_id, odds_type, pool_id, line_id, line_condition, comb_id, "
            "odds, selection_name, match_date, kickoff_time, scraped_at "
            "FROM odds_raw" + where, conn, params=params,
        )
        snapshot = pd.DataFrame(columns=["match_id", "odds_type", "pool_id",
                                          "line_id", "comb_id", "selection_name"])
        if _table_exists(conn, "odds_snapshot"):
            snapshot = pd.read_sql_query(
                "SELECT match_id, odds_type, pool_id, line_id, comb_id, selection_name "
                "FROM odds_snapshot" + where, conn, params=params,
            )

    for frame in (opening, raw, snapshot):
        if "line_id" in frame.columns:
            frame["line_id"] = frame["line_id"].fillna("")
        if "pool_id" in frame.columns:
            frame["pool_id"] = frame["pool_id"].fillna("")

    # ---- opening side ----------------------------------------------------- #
    opening["opening_odds"] = pd.to_numeric(opening["odds"], errors="coerce")
    opening["opening_prob"] = pd.to_numeric(opening["implied_prob_devig"], errors="coerce")
    opening = opening[KEY + ["line_condition", "opening_odds", "opening_prob", "captured_at"]]

    # ---- closing side: last pre-kickoff observation per selection --------- #
    raw["odds"] = pd.to_numeric(raw["odds"], errors="coerce")
    raw = raw[(raw["odds"] > 0) & raw["odds"].notna()].copy()
    raw["minutes_to_ko"] = [
        _minutes_to_ko(md, kt, sa)
        for md, kt, sa in zip(raw["match_date"], raw["kickoff_time"], raw["scraped_at"])
    ]
    pre = raw[raw["minutes_to_ko"] > 0].copy()

    closing = pre.sort_values(["scraped_at", "id"]).groupby(KEY, dropna=False).tail(1).copy()
    if not closing.empty:
        closing["_implied"] = 1.0 / closing["odds"]
        grp = closing.groupby(MARKET, dropna=False)["_implied"]
        group_total = grp.transform("sum")
        group_n = grp.transform("size")
        closing["closing_prob"] = closing["_implied"] / group_total
        # single-outcome pools (SGA) have nothing to de-vig against; keep 1/odds
        single = group_n == 1
        closing.loc[single, "closing_prob"] = closing.loc[single, "_implied"]
        closing = closing.rename(columns={
            "odds": "closing_odds",
            "scraped_at": "last_pre_ko_scraped_at",
        })
        closing = closing[KEY + ["line_condition", "closing_odds", "closing_prob",
                                 "last_pre_ko_scraped_at"]]
    else:
        closing = pd.DataFrame(columns=KEY + ["line_condition", "closing_odds",
                                              "closing_prob", "last_pre_ko_scraped_at"])

    # ---- coverage bookkeeping -------------------------------------------- #
    opening_keys = set(map(tuple, opening[KEY].astype(str).to_numpy()))
    closing_keys = set(map(tuple, closing[KEY].astype(str).to_numpy()))
    no_opening = len(closing_keys - opening_keys)
    no_closing = len(opening_keys - closing_keys)

    # ---- join opening + closing ------------------------------------------ #
    merged = opening.merge(closing, on=KEY, how="inner", suffixes=("", "_closing"))
    if "line_condition_closing" in merged.columns:
        merged["line_condition"] = merged["line_condition"].fillna(
            merged["line_condition_closing"])
        merged = merged.drop(columns=["line_condition_closing"])

    merged = merged.merge(snapshot, on=KEY, how="left", suffixes=("", "_snap"))
    if "selection_name_snap" in merged.columns:
        merged["selection_name"] = merged["selection_name"].fillna(
            merged["selection_name_snap"])
        merged = merged.drop(columns=["selection_name_snap"])

    merged["clv_prob_shift"] = merged["closing_prob"] - merged["opening_prob"]
    merged["clv_odds_ratio"] = merged["closing_odds"] / merged["opening_odds"] - 1.0

    cols = ["match_id", "odds_type", "pool_id", "line_id", "line_condition", "comb_id",
            "selection_name", "opening_odds", "opening_prob", "closing_odds",
            "closing_prob", "clv_prob_shift", "clv_odds_ratio",
            "last_pre_ko_scraped_at"]
    for col in cols:
        if col not in merged.columns:
            merged[col] = None
    merged = merged[cols].copy()
    if "selection_name" not in merged or merged["selection_name"].isna().all():
        merged["selection_name"] = None

    merged = merged.sort_values(
        ["odds_type", "match_id", "pool_id", "line_id", "comb_id"]).reset_index(drop=True)

    merged.attrs["total_selections"] = len(opening_keys | closing_keys)
    merged.attrs["included"] = int(len(merged))
    merged.attrs["excluded_no_opening"] = no_opening
    merged.attrs["excluded_no_closing"] = no_closing
    return merged


def coverage(df: pd.DataFrame, total_selections: Optional[int] = None) -> Dict[str, Any]:
    """Included / excluded selection counts and exclusion reasons.

    If total_selections is None, the value recorded by clv_table (via
    df.attrs) is used.
    """
    included = int(len(df))
    if total_selections is None:
        total_selections = df.attrs.get("total_selections", included)
    total_selections = int(total_selections)
    excluded = max(total_selections - included, 0)
    return {
        "total_selections": total_selections,
        "included": included,
        "excluded": excluded,
        "excluded_no_opening": int(df.attrs.get("excluded_no_opening", 0)),
        "excluded_no_closing": int(df.attrs.get("excluded_no_closing", 0)),
        "coverage_rate": (included / total_selections) if total_selections else 0.0,
    }


def _agg(group: pd.DataFrame) -> pd.Series:
    shift = group["clv_prob_shift"]
    return pd.Series({
        "count": int(shift.notna().sum()),
        "mean_clv_prob_shift": shift.mean(),
        "median_clv_prob_shift": shift.median(),
        "positive_rate": (shift > 0).mean(),
    })


def clv_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate clv_prob_shift by odds_type and by (odds_type, selection)."""
    if df.empty:
        return pd.DataFrame(columns=["group_kind", "odds_type", "selection",
                                     "count", "mean_clv_prob_shift",
                                     "median_clv_prob_shift", "positive_rate"])

    by_type = df.groupby("odds_type", dropna=False).apply(_agg, include_groups=False).reset_index()
    by_type.insert(0, "group_kind", "odds_type")
    by_type["selection"] = None

    by_sel = (df.groupby(["odds_type", "comb_id"], dropna=False)
                .apply(_agg, include_groups=False).reset_index()
                .rename(columns={"comb_id": "selection"}))
    by_sel.insert(0, "group_kind", "selection")

    out = pd.concat([by_type, by_sel], ignore_index=True)
    return out[["group_kind", "odds_type", "selection", "count",
                "mean_clv_prob_shift", "median_clv_prob_shift", "positive_rate"]]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _fmt(x, nd=4):
    return "n/a" if pd.isna(x) else f"{x:.{nd}f}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="HKJC closing-line-value (CLV) backtest (read-only).")
    ap.add_argument("--db", default=DB_DEFAULT, help="path to hkjc_odds.db (opened read-only)")
    ap.add_argument("--match", default=None, help="filter by match_id")
    ap.add_argument("--odds-type", default=None, help="filter by odds_type (HAD/HDC/HIL/...)")
    ap.add_argument("--top", type=int, default=10, help="show top N most positive/negative CLV")
    args = ap.parse_args(argv)

    df = clv_table(db_path=args.db, match_id=args.match, odds_type=args.odds_type)
    cov = coverage(df)

    print("=" * 72)
    print(f"CLV coverage  (db={args.db}"
          f"{', match=' + args.match if args.match else ''}"
          f"{', odds_type=' + args.odds_type if args.odds_type else ''})")
    print("=" * 72)
    print(f"  total selections considered : {cov['total_selections']}")
    print(f"  included (opening + closing) : {cov['included']}  "
          f"({cov['coverage_rate']:.1%})")
    print(f"  excluded                     : {cov['excluded']}")
    print(f"    - no opening row           : {cov['excluded_no_opening']}")
    print(f"    - no pre-kickoff closing   : {cov['excluded_no_closing']}")

    if df.empty:
        print("\nNo selections with both opening and pre-kickoff closing observations.")
        return 0

    print("\n" + "=" * 72)
    print("CLV summary")
    print("=" * 72)
    summary = clv_summary(df)
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(summary.to_string(index=False))

    show_cols = ["match_id", "odds_type", "line_id", "line_condition", "comb_id",
                 "selection_name", "opening_prob", "closing_prob",
                 "clv_prob_shift", "clv_odds_ratio", "last_pre_ko_scraped_at"]
    print("\n" + "=" * 72)
    print(f"Top {args.top} POSITIVE CLV (market moved toward selection)")
    print("=" * 72)
    pos = df.sort_values("clv_prob_shift", ascending=False).head(args.top)[show_cols]
    with pd.option_context("display.max_columns", None, "display.width", 240,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(pos.to_string(index=False))

    print("\n" + "=" * 72)
    print(f"Top {args.top} NEGATIVE CLV")
    print("=" * 72)
    neg = df.sort_values("clv_prob_shift", ascending=True).head(args.top)[show_cols]
    with pd.option_context("display.max_columns", None, "display.width", 240,
                           "display.float_format", lambda v: f"{v:.4f}"):
        print(neg.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
