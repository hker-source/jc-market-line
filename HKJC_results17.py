"""
1.17 HKJC_results.py

Match-RESULTS storage layer. Companion to HKJC_warehouse135.py and deliberately
built with the SAME philosophy:

  - An immutable, append-only raw log (match_results_raw) holding the exact
    JSON of every match as HKJC returned it. This is the ground truth; nothing
    downstream re-derives history on its own.
  - A derived long-format table (match_results) with one row per
    (stage_id, result_type) so half-time / full-time / corner / special
    settlements never overwrite each other.
  - No collapsing of ambiguous flags. `result_confirm_type` and
    `payout_confirmed` are stored as SEPARATE nullable columns because their
    exact semantics are NOT verified; a later fact-check needs them intact.

This module performs NO network fetch. The live results endpoint is not yet
pinned down (see probe_match_results.py); this layer only ingests and stores
whatever the GraphQL `data.matches` array turns out to contain.

Ambiguity note (important):
  stageId = 4 ("Full Time / Normal Time") is ambiguous: it can be the score at
  the end of 90' OR at the end of 120'. stageId = 5 ("Final Confirmed") is
  preferred, but is itself not proof that extra time did or did not happen.
  Therefore resolve_90min_result() NEVER silently derives a settlement basis:
  it returns "90" only when explicitly told to assume no extra time, and
  "unknown" otherwise.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS match_results_raw (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id          TEXT NOT NULL,   -- frontEndId if present, else internal id
    match_internal_id TEXT,            -- numeric HKJC id
    status            TEXT,
    captured_at       TEXT NOT NULL,
    payload_json      TEXT NOT NULL    -- exact match dict, ensure_ascii=False
);

CREATE INDEX IF NOT EXISTS idx_match_results_raw_lookup
    ON match_results_raw(match_id, captured_at);

CREATE TABLE IF NOT EXISTS match_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id            TEXT NOT NULL,
    match_internal_id   TEXT,
    stage_id            INTEGER,
    result_type         INTEGER,
    home_result         NUMERIC,
    away_result         NUMERIC,
    result_confirm_type NUMERIC,       -- stored as-is; semantics UNVERIFIED
    payout_confirmed    NUMERIC,       -- stored as-is; semantics UNVERIFIED
    captured_at         TEXT
);

-- UNIQUE (not merely INDEX): the result table is a derived snapshot, so a
-- repeated (match_id, stage_id, result_type, captured_at) is a duplicate write,
-- not new history. INSERT OR REPLACE below relies on this constraint to stay
-- idempotent-safe without aborting an ingest batch. The raw table is the
-- append-only log and intentionally has no such constraint.
--
-- Caveat: SQLite treats NULLs as distinct in a UNIQUE index, so rows with a
-- NULL stage_id or result_type are not deduped. That is acceptable: a missing
-- discriminator is already ambiguous data and must not be silently merged.
CREATE UNIQUE INDEX IF NOT EXISTS idx_match_results_lookup
    ON match_results(match_id, stage_id, result_type, captured_at);

-- ---- settlement (foPools) ------------------------------------------------ #
-- Fed by matchResultDetails, or by the main matchList query with
-- foPools(fbOddsTypes: ..., resultOnly: true). Each combination carries its own
-- settlement `status` (WIN / LOSE); `winOrd` is DISPLAY ORDER, not a rank.
CREATE TABLE IF NOT EXISTS match_settlements_raw (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id          TEXT NOT NULL,
    match_internal_id TEXT,
    captured_at       TEXT NOT NULL,
    payload_json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_match_settlements_raw_lookup
    ON match_settlements_raw(match_id, captured_at);

CREATE TABLE IF NOT EXISTS match_settlements (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id          TEXT NOT NULL,
    match_internal_id TEXT,
    odds_type         TEXT NOT NULL,
    pool_id           TEXT,
    inst_no           INTEGER,
    pool_status       TEXT,      -- e.g. PAYOUTSTARTED
    line_index        INTEGER,   -- position of the line within the pool (settlement may omit lineId)
    line_id           TEXT,      -- present when the source query selects it
    line_condition    TEXT,
    comb_str          TEXT,      -- '01', 'H:H', ...
    comb_status       TEXT,      -- WIN / LOSE / ...
    win_ord           TEXT,      -- display order, NOT a rank
    sel_id            TEXT,
    sel_str           TEXT,
    sel_name_en       TEXT,
    sel_name_ch       TEXT,
    captured_at       TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_match_settlements_lookup
    ON match_settlements(match_id, odds_type, COALESCE(pool_id,''), line_index,
                         COALESCE(comb_str,''), COALESCE(sel_id,''));
"""

# Field mapping (from the pending results probe), kept here as documentation:
#   frontEndId   -> match_id ("FB5952")
#   id           -> match_internal_id ("50076228")
#   status       -> status ("INPLAYMATCHENDED")
#   results[]    -> one row per element
# stageId:    2 = Half Time / 1st Half
#             4 = Full Time / Normal Time  (AMBIGUOUS: 90' vs 120')
#             5 = Final Confirmed
# resultType: 1 = Goals / scoreline (homeResult, awayResult)
#             2 = Corner count
#             4 = Null / Special settlement
# resultConfirmType / payoutConfirmed: settlement-finalized flags, stored
#   as-is and nullable because their exact meaning is NOT verified.


@dataclass
class ResultRow:
    match_id: str
    match_internal_id: Optional[str]
    stage_id: Optional[int]
    result_type: Optional[int]
    home_result: Any
    away_result: Any
    result_confirm_type: Any
    payout_confirmed: Any
    captured_at: str


class MatchResultsStore:
    def __init__(self, db_path: str = "hkjc_odds.db"):
        self.db_path = db_path
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_results(self, matches: List[Dict[str, Any]]) -> int:
        """Ingest the `data.matches` array from an HKJC GraphQL matchList payload.

        For each match: append ONE immutable match_results_raw row, then flatten
        every element of `results[]` into a match_results long row. Missing keys,
        None and an empty/absent `results` list are all handled gracefully.

        captured_at is computed once per call so an entire batch shares a
        timestamp and stays a coherent snapshot.

        Idempotency: match_results has a UNIQUE index on
        (match_id, stage_id, result_type, captured_at) and writes use
        INSERT OR REPLACE, so a duplicate snapshot overwrites rather than
        aborting the batch. match_results_raw uses a plain INSERT and therefore
        always appends (that is the immutable log's job).

        Returns the number of long rows written.
        """
        if not matches:
            return 0

        captured_at = datetime.now(timezone.utc).isoformat()
        raw_rows: List[Tuple[Any, ...]] = []
        long_rows: List[ResultRow] = []

        for m in matches:
            if not isinstance(m, dict):
                continue
            raw_id = m.get("frontEndId") or m.get("id")
            match_id = str(raw_id) if raw_id is not None else "UNKNOWN"
            internal = m.get("id")
            match_internal_id = str(internal) if internal is not None else None

            raw_rows.append((
                match_id,
                match_internal_id,
                m.get("status"),
                captured_at,
                json.dumps(m, ensure_ascii=False),
            ))

            for r in (m.get("results") or []):
                if not isinstance(r, dict):
                    continue
                long_rows.append(ResultRow(
                    match_id=match_id,
                    match_internal_id=match_internal_id,
                    stage_id=r.get("stageId"),
                    result_type=r.get("resultType"),
                    home_result=r.get("homeResult"),
                    away_result=r.get("awayResult"),
                    result_confirm_type=r.get("resultConfirmType"),
                    payout_confirmed=r.get("payoutConfirmed"),
                    captured_at=captured_at,
                ))

        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO match_results_raw (
                    match_id, match_internal_id, status, captured_at, payload_json
                ) VALUES (?,?,?,?,?)""",
                raw_rows,
            )
            if long_rows:
                conn.executemany(
                    """INSERT OR REPLACE INTO match_results (
                        match_id, match_internal_id, stage_id, result_type,
                        home_result, away_result,
                        result_confirm_type, payout_confirmed, captured_at
                    ) VALUES (?,?,?,?,?,?,?,?,?)""",
                    [
                        (r.match_id, r.match_internal_id, r.stage_id, r.result_type,
                         r.home_result, r.away_result,
                         r.result_confirm_type, r.payout_confirmed, r.captured_at)
                        for r in long_rows
                    ],
                )
        return len(long_rows)

    # ------------------------------------------------------------------ #
    # Resolution (pure, no network)
    # ------------------------------------------------------------------ #
    def resolve_90min_result(self, match_id: str,
                             assume_no_extra_time: bool = False) -> Dict[str, Any]:
        """Resolve the 90-minute score from stored rows. PURE: reads SQLite only.

        The ambiguity: stageId=4 is the end of 90' OR the end of 120'; stageId=5
        ("Final Confirmed") is preferred but does not by itself prove whether
        extra time was played. This method therefore NEVER silently derives a
        basis:

          - assume_no_extra_time=True and a coherent full-time score exists
            (prefer stageId=5, else stageId=4, resultType=1, both legs present):
            settlement_basis="90" and home/away come from that row.
          - otherwise settlement_basis="unknown". A best-effort score may still
            be returned (stageId=5 preferred, else stageId=4) but it is flagged
            as unconfirmed via the basis and the provenance list.

        settlement_basis="120" is a reserved value: no stored field unambiguously
        identifies a 120' result today, so it is never returned implicitly. A
        later fact-check may enable it.

        Returns:
          {
            "match_id": str,
            "home": int|None,
            "away": int|None,
            "settlement_basis": "90"|"120"|"unknown",
            "provenance": [(stage_id, result_type, home_result, away_result), ...],
          }
        provenance lists the exact rows the returned numbers were drawn from (and,
        when unknown, every resultType=1 candidate consulted).
        """
        result: Dict[str, Any] = {
            "match_id": match_id,
            "home": None,
            "away": None,
            "settlement_basis": "unknown",
            "provenance": [],
        }

        df = self.latest_results(match_id)
        if df.empty:
            return result

        score_rows = df[df["result_type"] == 1]

        def _num(v: Any) -> Any:
            # pandas widens a NUMERIC column to float when it contains NaN;
            # collapse whole numbers back to int so callers get int|None.
            if v is None or (isinstance(v, float) and v != v):
                return None
            if isinstance(v, float) and v.is_integer():
                return int(v)
            return v

        def _row(stage_id: int):
            sub = score_rows[score_rows["stage_id"] == stage_id]
            return sub.iloc[0] if not sub.empty else None

        def _tuple(row) -> Tuple[Any, Any, Any, Any]:
            return (int(row["stage_id"]), int(row["result_type"]),
                    _num(row["home_result"]), _num(row["away_result"]))

        s5, s4 = _row(5), _row(4)
        chosen = s5 if s5 is not None else s4

        if assume_no_extra_time:
            if chosen is not None:
                result["home"] = _num(chosen["home_result"])
                result["away"] = _num(chosen["away_result"])
                result["provenance"] = [_tuple(chosen)]
                if chosen["home_result"] is not None and chosen["away_result"] is not None:
                    result["settlement_basis"] = "90"
            return result

        # Not authorised to assume: expose a best-effort score but keep the basis
        # explicitly unknown and show every resultType=1 row that was consulted.
        if chosen is not None:
            result["home"] = _num(chosen["home_result"])
            result["away"] = _num(chosen["away_result"])
        result["provenance"] = [
            (int(r["stage_id"]), int(r["result_type"]),
             _num(r["home_result"]), _num(r["away_result"]))
            for _, r in score_rows.sort_values("stage_id").iterrows()
        ]
        return result

    # ------------------------------------------------------------------ #
    # Read helpers
    # ------------------------------------------------------------------ #
    def latest_results(self, match_id: Optional[str] = None) -> pd.DataFrame:
        """Newest capture per (match_id, stage_id, result_type), optionally filtered.

        NULL stage_id/result_type are joined with SQLite's null-safe `IS` so such
        rows are not silently dropped.
        """
        query = """
            SELECT r.* FROM match_results r
            JOIN (
                SELECT match_id, stage_id, result_type, MAX(captured_at) AS max_ts
                FROM match_results {where}
                GROUP BY match_id, stage_id, result_type
            ) latest
            ON r.match_id IS latest.match_id
           AND r.stage_id IS latest.stage_id
           AND r.result_type IS latest.result_type
           AND r.captured_at IS latest.max_ts
        """
        where = ""
        params: List[Any] = []
        if match_id is not None:
            where = "WHERE match_id = ?"
            params.append(match_id)
        with self._conn() as conn:
            return pd.read_sql_query(query.format(where=where), conn, params=params)

    def raw_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM match_results_raw").fetchone()[0]

    def result_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM match_results").fetchone()[0]

    # ------------------------------------------------------------------ #
    # Settlement (foPools with resultOnly:true, or matchResultDetails)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _norm_comb(value: Optional[str]) -> Optional[str]:
        """Normalise a combination key so settlement and odds can join.

        Live odds return combId like '1' while the settlement payload returns the
        combination `str` like '01'. Strip leading zeros from all-numeric keys so
        the two sides match; non-numeric keys (e.g. 'H:H') pass through unchanged.
        """
        if value is None:
            return None
        s = str(value).strip()
        return str(int(s)) if s.isdigit() else s

    def ingest_settlement(self, matches: List[Dict[str, Any]]) -> int:
        """Ingest a `data.matches` array whose matches carry `foPools` settlement.

        For each match: append ONE immutable match_settlements_raw row, then
        flatten foPools -> lines -> combinations -> selections into
        match_settlements. The settlement signal is a combination's `status`
        (WIN / LOSE); `winOrd` is display order only and is stored verbatim.

        Returns the number of settlement rows written.
        """
        if not matches:
            return 0

        captured_at = datetime.now(timezone.utc).isoformat()
        raw_rows: List[Tuple[Any, ...]] = []
        rows: List[Tuple[Any, ...]] = []

        for m in matches:
            if not isinstance(m, dict):
                continue
            raw_id = m.get("frontEndId") or m.get("id")
            match_id = str(raw_id) if raw_id is not None else "UNKNOWN"
            internal = m.get("id")
            match_internal_id = str(internal) if internal is not None else None

            raw_rows.append((match_id, match_internal_id, captured_at,
                             json.dumps(m, ensure_ascii=False)))

            for pool in (m.get("foPools") or []):
                if not isinstance(pool, dict):
                    continue
                odds_type = pool.get("oddsType")
                pid = pool.get("id")
                pool_id = str(pid) if pid is not None else None
                inst_no = pool.get("instNo")
                pool_status = pool.get("status")
                for line_index, line in enumerate(pool.get("lines") or []):
                    if not isinstance(line, dict):
                        continue
                    lid = line.get("lineId")
                    line_id = str(lid) if lid is not None else None
                    for comb in (line.get("combinations") or []):
                        if not isinstance(comb, dict):
                            continue
                        win_ord = comb.get("winOrd")
                        for sel in (comb.get("selections") or []):
                            sel = sel if isinstance(sel, dict) else {}
                            sid = sel.get("selId")
                            rows.append((
                                match_id, match_internal_id, odds_type, pool_id,
                                inst_no, pool_status, line_index, line_id,
                                line.get("condition"), comb.get("str"),
                                comb.get("status"),
                                str(win_ord) if win_ord is not None else None,
                                str(sid) if sid is not None else None,
                                sel.get("str"), sel.get("name_en"), sel.get("name_ch"),
                                captured_at,
                            ))

        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO match_settlements_raw
                   (match_id, match_internal_id, captured_at, payload_json)
                   VALUES (?,?,?,?)""",
                raw_rows,
            )
            if rows:
                conn.executemany(
                    """INSERT OR REPLACE INTO match_settlements (
                        match_id, match_internal_id, odds_type, pool_id, inst_no,
                        pool_status, line_index, line_id, line_condition,
                        comb_str, comb_status, win_ord, sel_id, sel_str,
                        sel_name_en, sel_name_ch, captured_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    rows,
                )
        return len(rows)

    def latest_settlements(self, match_id: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT * FROM match_settlements"
        params: List[Any] = []
        if match_id is not None:
            query += " WHERE match_id = ?"
            params.append(match_id)
        query += " ORDER BY match_id, odds_type, pool_id, line_index, comb_str"
        with self._conn() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def settled_winner(self, match_id: str, odds_type: str,
                       pool_id: Optional[str] = None,
                       line_index: Optional[int] = None) -> List[Dict[str, Any]]:
        """Winning combination rows (comb_status='WIN') for a market.

        Multi-line markets (CHL/HDC/...) may need `line_index` because the
        settlement payload can omit `lineId`.
        """
        clauses = ["match_id = ?", "odds_type = ?", "comb_status = 'WIN'"]
        params: List[Any] = [match_id, odds_type]
        if pool_id is not None:
            clauses.append("pool_id = ?")
            params.append(pool_id)
        if line_index is not None:
            clauses.append("line_index = ?")
            params.append(line_index)
        with self._conn() as conn:
            fetched = conn.execute(
                f"SELECT * FROM match_settlements WHERE {' AND '.join(clauses)}",
                params,
            ).fetchall()
        return [dict(r) for r in fetched]

    def is_settled(self, match_id: str) -> bool:
        with self._conn() as conn:
            return conn.execute(
                "SELECT 1 FROM match_settlements "
                "WHERE match_id = ? AND comb_status IN ('WIN','LOSE') LIMIT 1",
                (match_id,),
            ).fetchone() is not None

    def settlement_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM match_settlements").fetchone()[0]


if __name__ == "__main__":
    import os

    db_path = "match_results_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    store = MatchResultsStore(db_path)
    synthetic = [{
        "id": "50076228",
        "frontEndId": "FB5952",
        "status": "INPLAYMATCHENDED",
        "kickOffTime": "2026-09-30T02:45:00.000+08:00",
        "homeTeam": {"id": "H1", "name_en": "Home FC", "name_ch": "主隊"},
        "awayTeam": {"id": "A1", "name_en": "Away FC", "name_ch": "客隊"},
        "results": [
            {"stageId": 2, "resultType": 1, "homeResult": 1, "awayResult": 0,
             "resultConfirmType": 1, "payoutConfirmed": True},
            {"stageId": 4, "resultType": 1, "homeResult": 2, "awayResult": 1,
             "resultConfirmType": 1, "payoutConfirmed": True},
            {"stageId": 5, "resultType": 1, "homeResult": 2, "awayResult": 1,
             "resultConfirmType": None, "payoutConfirmed": None},
            {"stageId": 2, "resultType": 2, "homeResult": 3, "awayResult": 4,
             "resultConfirmType": None, "payoutConfirmed": None},
            {"stageId": 4, "resultType": 4, "homeResult": None, "awayResult": None,
             "resultConfirmType": None, "payoutConfirmed": None},
        ],
    }]

    n = store.ingest_results(synthetic)
    print(f"Ingested {n} long rows; raw={store.raw_count()} results={store.result_count()}")

    print("\n--- match_results_raw ---")
    with store._conn() as conn:
        for row in conn.execute(
            "SELECT id, match_id, match_internal_id, status, captured_at FROM match_results_raw"
        ):
            print(dict(row))

    print("\n--- latest_results() ---")
    print(store.latest_results("FB5952").to_string(index=False))

    print("\n--- resolve_90min_result(assume_no_extra_time=True) ---")
    print(store.resolve_90min_result("FB5952", assume_no_extra_time=True))
    print("\n--- resolve_90min_result(assume_no_extra_time=False) ---")
    print(store.resolve_90min_result("FB5952", assume_no_extra_time=False))
