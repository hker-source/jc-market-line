"""
HKJC_analysis_dataset.py -- Phase 1 preprocessing.

Builds ONE model-ready flat table (one row per selection) from the live DB, per
docs/METHODOLOGY.md (frozen pre-registration v1):

  * primary drift regressor  lambda = ln(o0) - ln(o1)   (raw odds; SGA-safe)
  * confounder control       ln(o0)
  * secondary                delta_prob = p1-p0 ; kappa = delta/p0 (winsorised P1/P99 per odds_type)
  * temporal features        pre-KO velocity/acceleration aggregates + steam_dummy
  * market FE                odds_type (kept as a column, not one-hot here)
  * SGA kept with flags      is_sga / prob_space ('raw_single' vs 'devig_multi')
  * join key                 comb_str normalised (NOT comb_id, NOT line_index)

Read-only DB access. No settlement -> settled=0 (row still emitted for coverage).
Also provides dq_report() for the pre-modelling data-quality gate.

Run: venv/bin/python HKJC_analysis_dataset.py [--match FB5809] [--odds-type HAD]
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import pandas as pd

from HKJC_backtest_clv import clv_table
from HKJC_backtest_pnl import _norm_comb, _comb_str_map, _settlements_frame

DB_DEFAULT = "hkjc_odds.db"
KEY = ["match_id", "odds_type", "pool_id", "line_id", "comb_id"]
TAU = 0.02  # pre-registered price-direction threshold (nats), see METHODOLOGY R5

# Genuine void/refund statuses. Anything else that is non-null and not WIN/LOSE
# is a NOT-YET-SETTLED state (e.g. comb_status='AVAILABLE', pool_status
# 'SELLINGSTOPPED') and must NOT be labelled VOID.
VOID_STATUSES = {"VOID", "REFUND", "REFUNDED", "CANCELLED", "CANCELED",
                 "ABANDONED", "DEADHEAT", "PUSH"}


def _outcome_label(value: Any) -> str:
    if value in ("WIN", "LOSE"):
        return value
    if value is None:
        return "UNSETTLED"
    return "VOID" if str(value).strip().upper() in VOID_STATUSES else "PENDING"


# --------------------------------------------------------------------------- #
# Temporal features from odds_events (pre-KO only)
# --------------------------------------------------------------------------- #
def _velocity_features(db_path: str) -> pd.DataFrame:
    from HKJC_backtest_clv import _connect_ro
    empty = pd.DataFrame(columns=KEY + ["n_events_preko", "vel_mean_preko", "vel_absmax_preko",
                                        "vel_timeweighted_preko", "acc_mean_preko",
                                        "steam_dummy", "drift_class"])
    with _connect_ro(db_path) as conn:
        try:
            ev = pd.read_sql_query(
                "SELECT match_id, odds_type, pool_id, line_id, comb_id, movement_type, "
                "velocity, acceleration, seconds_since_last, minutes_to_ko FROM odds_events", conn)
        except Exception:
            return empty
    if ev.empty:
        return empty
    for c in ("pool_id", "line_id"):
        ev[c] = ev[c].fillna("").astype(str)
    pre = ev[(ev["minutes_to_ko"] > 0) &
             (ev["movement_type"].isin(["steam", "drift", "micro"]))].copy()
    if pre.empty:
        return empty
    pre["_w"] = pd.to_numeric(pre["seconds_since_last"], errors="coerce").fillna(0).clip(lower=0)

    g = pre.groupby(KEY, dropna=False)
    feat = g.agg(n_events_preko=("velocity", "size"),
                 vel_mean_preko=("velocity", "mean"),
                 acc_mean_preko=("acceleration", "mean")).reset_index()
    feat["vel_absmax_preko"] = (
        pre.assign(_a=pd.to_numeric(pre["velocity"], errors="coerce").abs())
        .groupby(KEY, dropna=False)["_a"].max().reset_index(drop=True))
    num = pre.assign(_p=pd.to_numeric(pre["velocity"], errors="coerce") * pre["_w"]) \
             .groupby(KEY, dropna=False)["_p"].sum()
    den = pre.groupby(KEY, dropna=False)["_w"].sum()
    tw = (num / den.replace(0, pd.NA)).reset_index(name="vel_timeweighted_preko")
    feat = feat.merge(tw, on=KEY, how="left")

    def _any(mt):
        return (pre.assign(_b=(pre["movement_type"] == mt))
                .groupby(KEY, dropna=False)["_b"].any().reset_index(name=f"_has_{mt}"))
    feat = feat.merge(_any("steam"), on=KEY, how="left")
    feat = feat.merge(_any("drift"), on=KEY, how="left")
    feat = feat.merge(_any("micro"), on=KEY, how="left")
    feat["steam_dummy"] = feat["_has_steam"].fillna(False).astype(int)
    feat["drift_class"] = "none"
    feat.loc[feat["_has_micro"], "drift_class"] = "micro_only"
    feat.loc[feat["_has_drift"], "drift_class"] = "drift_only"
    feat.loc[feat["_has_steam"], "drift_class"] = "steam"
    return feat.drop(columns=["_has_steam", "_has_drift", "_has_micro"])


def _winsorise_kappa(df: pd.DataFrame, kappa_col: str = "kappa",
                     lo: float = 0.01, hi: float = 0.99):
    """Clip kappa to P1/P99 per odds_type. Returns (series, bounds, applied)."""
    kap = pd.to_numeric(df[kappa_col], errors="coerce")
    out = pd.Series(index=df.index, dtype="float64")
    bounds: Dict[str, Any] = {}
    applied = pd.Series(0, index=df.index, dtype=int)
    for gval, idx in df.groupby("odds_type").groups.items():
        s = kap.loc[idx].dropna()
        if s.empty:
            out.loc[idx] = kap.loc[idx]
            continue
        lo_v, hi_v = float(s.quantile(lo)), float(s.quantile(hi))
        bounds[str(gval)] = (lo_v, hi_v)
        clipped = kap.loc[idx].clip(lo_v, hi_v)
        applied.loc[idx] = (clipped != kap.loc[idx]).astype(int)
        out.loc[idx] = clipped
    return out, bounds, applied


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def build_dataset(db_path: str = DB_DEFAULT,
                  match_id: Optional[str] = None,
                  odds_type: Optional[str] = None) -> pd.DataFrame:
    base = clv_table(db_path=db_path, match_id=match_id, odds_type=odds_type)
    if base.empty:
        return base
    df = base.copy()
    for c in ("pool_id", "line_id"):
        df[c] = df[c].fillna("").astype(str) if c in df.columns else ""
    df["comb_id"] = df["comb_id"].astype(str)

    # comb_str bridge (settlement keys on str, not comb_id)
    cmap = _comb_str_map(db_path).rename(columns={"pool_key": "pool_id", "line_key": "line_id"})
    df = df.merge(cmap, on=KEY, how="left")
    df["comb_key"] = df["comb_str"].map(_norm_comb)

    # settlement (left join; never drop rows -- flag instead)
    s = _settlements_frame(db_path)
    df = df.reset_index(drop=True)
    if not s.empty:
        s2 = (s[["match_id", "odds_type", "pool_key", "comb_key", "line_key",
                 "comb_status", "sel_name_en", "inst_no"]]
                .rename(columns={"pool_key": "pool_id_join", "line_key": "line_id_set"}))
        df = df.merge(s2, left_on=["match_id", "odds_type", "pool_id", "comb_key"],
                      right_on=["match_id", "odds_type", "pool_id_join", "comb_key"],
                      how="left").drop(columns=["pool_id_join"])
        has_line = df["line_id_set"].notna() & (df["line_id_set"] != "")
        ok_line = df["line_id"] == df["line_id_set"]
        single = ~df.duplicated(["match_id", "odds_type", "pool_id", "comb_key"], keep=False)
        df["settlement_ok"] = (has_line & ok_line) | (~has_line & single)
        df["comb_status"] = df["comb_status"].where(df["settlement_ok"])
        df = df.drop(columns=["line_id_set"])
    else:
        for c in ("comb_status", "sel_name_en", "inst_no"):
            df[c] = None
        df["settlement_ok"] = False

    # ---- drift features (raw-odds based) --------------------------------- #
    o0 = pd.to_numeric(df["opening_odds"], errors="coerce")
    o1 = pd.to_numeric(df["closing_odds"], errors="coerce")
    p0 = pd.to_numeric(df["opening_prob"], errors="coerce")
    p1 = pd.to_numeric(df["closing_prob"], errors="coerce")
    df["ln_o0"] = o0.map(lambda v: math.log(v) if v and v > 0 else None)
    df["lambda"] = [math.log(a) - math.log(b) if a and b and a > 0 and b > 0 else None
                    for a, b in zip(o0, o1)]
    df["delta_prob"] = p1 - p0
    df["kappa"] = df["delta_prob"] / p0.replace(0, pd.NA)
    kwin, bounds, applied = _winsorise_kappa(df)
    df["kappa_wins"] = kwin
    df["kappa_wins_applied"] = applied
    df.attrs["kappa_bounds"] = bounds

    # ---- flags / space --------------------------------------------------- #
    df["is_sga"] = (df["odds_type"] == "SGA").astype(int)
    df["prob_space"] = df["is_sga"].map({1: "raw_single", 0: "devig_multi"})

    # ---- temporal features ----------------------------------------------- #
    df = df.merge(_velocity_features(db_path), on=KEY, how="left")
    for c in ("n_events_preko",):
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
    if "steam_dummy" in df.columns:
        df["steam_dummy"] = df["steam_dummy"].fillna(0).astype(int)
    else:
        df["steam_dummy"] = 0
        df["drift_class"] = "none"

    # ---- outcome / economics -------------------------------------------- #
    st = df["comb_status"]
    df["outcome"] = st.apply(_outcome_label)
    df["settled"] = st.isin(["WIN", "LOSE"]).astype(int)
    df["stake"] = 1.0
    df["profit"] = [
        (a - 1.0) if s_ == "WIN" else (-1.0 if s_ == "LOSE" else 0.0)
        for s_, a in zip(st, o0)
    ]
    df["ev_closing"] = p1 * o0 - 1.0

    # ---- price-direction bucket (METHODOLOGY R2/R5) --------------------- #
    lam = pd.to_numeric(df["lambda"], errors="coerce")
    df["price_direction"] = "flat"
    df.loc[lam > TAU, "price_direction"] = "shortened"
    df.loc[lam < -TAU, "price_direction"] = "lengthened"

    cols = ["match_id", "odds_type", "pool_id", "line_id", "line_condition",
            "comb_id", "comb_str", "selection_name", "inst_no",
            "opening_odds", "closing_odds", "opening_prob", "closing_prob",
            "prob_space", "is_sga", "ln_o0", "lambda", "delta_prob",
            "kappa", "kappa_wins", "kappa_wins_applied", "clv_odds_ratio",
            "n_events_preko", "vel_mean_preko", "vel_absmax_preko",
            "vel_timeweighted_preko", "acc_mean_preko", "steam_dummy", "drift_class",
            "price_direction", "comb_status", "outcome", "settled", "stake",
            "profit", "ev_closing", "last_pre_ko_scraped_at"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    df.attrs["kappa_bounds"] = bounds
    df.attrs["coverage"] = {
        "rows": int(len(df)),
        "settled": int(df["settled"].sum()),
        "win": int((df["outcome"] == "WIN").sum()),
        "lose": int((df["outcome"] == "LOSE").sum()),
        "sga": int(df["is_sga"].sum()),
    }
    return df


# --------------------------------------------------------------------------- #
# Data-quality checks
# --------------------------------------------------------------------------- #
def dq_report(db_path: str = DB_DEFAULT) -> Dict[str, Any]:
    from HKJC_backtest_clv import _connect_ro, _table_exists
    out: Dict[str, Any] = {}
    with _connect_ro(db_path) as conn:
        one = lambda q: conn.execute(q).fetchone()[0]
        # duplicate settlement keys
        out["settlement_dup_keys"] = one(
            "SELECT COUNT(*) FROM (SELECT match_id,odds_type,COALESCE(pool_id,''),line_index,"
            "COALESCE(comb_str,''),COALESCE(sel_id,''),COUNT(*) c FROM match_settlements "
            "GROUP BY 1,2,3,4,5,6 HAVING c>1)")
        # odds <= 1
        out["odds_le_1_rows"] = one("SELECT COUNT(*) FROM odds_raw WHERE odds IS NULL OR odds<=1")
        # settlement orphans (no opening_odds)
        out["settlement_orphans"] = one(
            "SELECT COUNT(*) FROM match_settlements s WHERE NOT EXISTS "
            "(SELECT 1 FROM opening_odds o WHERE o.match_id=s.match_id AND o.odds_type=s.odds_type)")
        # multi-line markets with missing line_id
        out["multiline_missing_line_id"] = one(
            "SELECT COUNT(*) FROM (SELECT match_id,odds_type,pool_id FROM opening_odds "
            "GROUP BY 1,2,3 HAVING COUNT(DISTINCT COALESCE(line_id,''))>1) t "
            "JOIN opening_odds o ON o.match_id=t.match_id AND o.odds_type=t.odds_type "
            "AND o.pool_id=t.pool_id WHERE o.line_id IS NULL OR o.line_id=''")
        # SGA pool integrity: matches where distinct pools < 2 (should be ~20)
        out["sga_matches_lt_2_pools"] = one(
            "SELECT COUNT(*) FROM (SELECT match_id,COUNT(DISTINCT pool_id) n FROM odds_raw "
            "WHERE odds_type='SGA' GROUP BY 1 HAVING n<2)")
        # settlement coverage
        out["settlement_rows"] = one("SELECT COUNT(*) FROM match_settlements")
        out["settlement_matches"] = one("SELECT COUNT(DISTINCT match_id) FROM match_settlements")
        out["odds_matches"] = one("SELECT COUNT(DISTINCT match_id) FROM odds_raw")
    return out


def _check_devig_sums(db_path: str, tol: float = 1e-6) -> int:
    from HKJC_backtest_clv import _connect_ro
    with _connect_ro(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT match_id, odds_type, pool_id, line_id, SUM(implied_prob_devig) s, COUNT(*) n "
            "FROM opening_odds GROUP BY match_id, odds_type, pool_id, line_id", conn)
    multi = df[df["n"] > 1]
    return int((multi["s"].sub(1.0).abs() > tol).sum())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Phase 1 analysis dataset builder")
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--match")
    ap.add_argument("--odds-type")
    ap.add_argument("--dq", action="store_true")
    args = ap.parse_args()

    ds = build_dataset(args.db, match_id=args.match, odds_type=args.odds_type)
    print(f"rows={len(ds)}  coverage={ds.attrs.get('coverage')}")
    print("kappa_bounds(P1,P99 per odds_type):", ds.attrs.get("kappa_bounds"))
    if not ds.empty:
        with pd.option_context("display.width", 240, "display.max_columns", 40,
                               "display.float_format", lambda v: f"{v:.4f}"):
            print("\nby odds_type x outcome:")
            print(pd.crosstab(ds["odds_type"], ds["outcome"]).to_string())
            print("\nhead:")
            print(ds[["match_id", "odds_type", "comb_str", "selection_name", "opening_odds",
                      "closing_odds", "lambda", "ln_o0", "kappa_wins", "steam_dummy",
                      "drift_class", "outcome", "settled", "profit"]].head(12).to_string(index=False))
    if args.dq:
        print("\nDQ report:", dq_report(args.db))
        print("devig sum violations:", _check_devig_sums(args.db))
