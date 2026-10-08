"""
1.35 HKJC_warehouse.py  (merged version)

Single warehouse layer. Responsibilities:
  - Flatten 1.3 HKJC_odds.py's raw GraphQL output into rows keyed by HKJC's
    own IDs (match_id/comb_id) -- never by display strings like team names
    or "H"/"D"/"A", which collide across matches.
  - Append every row to an immutable SQLite log (odds_raw). This table is
    the single source of historical truth; nothing downstream re-derives
    history on its own.
  - Carry the extra context 1.5's classifier needs: running score, corners,
    inplay flag, kickoff time -- computed once here, not re-parsed later.
  - Convenience: DataFrame view + CSV/parquet export for offline work.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_raw (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scraped_at      TEXT NOT NULL,
    match_id        TEXT NOT NULL,   -- frontEndId if present, else internal id
    match_internal_id TEXT,
    match_label     TEXT,            -- "Home VS Away", display only, never a key
    home_team       TEXT,
    away_team       TEXT,
    tournament      TEXT,
    venue           TEXT,
    match_date      TEXT,
    kickoff_time    TEXT,
    match_status    TEXT,
    home_score      INTEGER,
    away_score      INTEGER,
    home_corners    INTEGER,
    away_corners    INTEGER,
    odds_type       TEXT NOT NULL,
    betting_type    TEXT,            -- pool name_ch, display only
    pool_id         TEXT,
    pool_status     TEXT,
    is_live         INTEGER,         -- 0/1, from pool.inplay
    line_id         TEXT,
    line_condition  TEXT,
    comb_id         TEXT NOT NULL,   -- real key, unique within a line
    comb_str        TEXT,
    comb_status     TEXT,
    selection_str   TEXT,
    selection_name  TEXT,
    odds            REAL,
    offer_early_settlement INTEGER
);

CREATE INDEX IF NOT EXISTS idx_odds_raw_lookup
    ON odds_raw(match_id, odds_type, comb_id, scraped_at);
CREATE INDEX IF NOT EXISTS idx_odds_raw_time ON odds_raw(scraped_at);
"""


@dataclass
class OddsRow:
    scraped_at: str
    match_id: str
    match_internal_id: Optional[str]
    match_label: str
    home_team: Optional[str]
    away_team: Optional[str]
    tournament: Optional[str]
    venue: Optional[str]
    match_date: Optional[str]
    kickoff_time: Optional[str]
    match_status: Optional[str]
    home_score: Optional[int]
    away_score: Optional[int]
    home_corners: Optional[int]
    away_corners: Optional[int]
    odds_type: str
    betting_type: Optional[str]
    pool_id: Optional[str]
    pool_status: Optional[str]
    is_live: bool
    line_id: Optional[str]
    line_condition: Optional[str]
    comb_id: str
    comb_str: Optional[str]
    comb_status: Optional[str]
    selection_str: Optional[str]
    selection_name: Optional[str]
    odds: Optional[float]
    offer_early_settlement: bool


class HKJCWarehouse:
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
    @staticmethod
    def flatten(raw_results: Dict[str, Dict[str, Any]]) -> List[OddsRow]:
        """raw_results: {odds_type: {"data": {"matches": [...]}}} from
        HKJCGraphQLClient.fetch_multiple_odds_types()."""
        scraped_at = datetime.now(timezone.utc).isoformat()
        rows: List[OddsRow] = []

        for odds_type, payload in raw_results.items():
            matches = (payload.get("data") or {}).get("matches") or []
            for m in matches:
                home = (m.get("homeTeam") or {}).get("name_en") or (m.get("homeTeam") or {}).get("name_ch") or ""
                away = (m.get("awayTeam") or {}).get("name_en") or (m.get("awayTeam") or {}).get("name_ch") or ""
                match_label = f"{home} VS {away}"
                match_id = str(m.get("frontEndId") or m.get("id"))
                match_internal_id = str(m.get("id")) if m.get("id") is not None else None
                tourn = (m.get("tournament") or {}).get("name_en") or (m.get("tournament") or {}).get("name_ch")
                venue = (m.get("venue") or {}).get("name_en") or (m.get("venue") or {}).get("name_ch")
                rr = m.get("runningResult") or {}

                for pool in (m.get("foPools") or []):
                    pool_odds_type = pool.get("oddsType") or odds_type
                    is_live = bool(pool.get("inplay", False))
                    for line in (pool.get("lines") or []):
                        for comb in (line.get("combinations") or []):
                            selections = comb.get("selections") or []
                            sel_str = "/".join(s.get("str") or "" for s in selections) or None
                            sel_name = " & ".join(
                                s.get("name_en") or s.get("name_ch") or "" for s in selections
                            ) or (comb.get("str") or None)
                            rows.append(OddsRow(
                                scraped_at=scraped_at,
                                match_id=match_id,
                                match_internal_id=match_internal_id,
                                match_label=match_label,
                                home_team=home, away_team=away,
                                tournament=tourn, venue=venue,
                                match_date=m.get("matchDate"),
                                kickoff_time=m.get("kickOffTime"),
                                match_status=m.get("status"),
                                home_score=rr.get("homeScore"),
                                away_score=rr.get("awayScore"),
                                home_corners=rr.get("homeCorner"),
                                away_corners=rr.get("awayCorner"),
                                odds_type=pool_odds_type,
                                betting_type=pool.get("name_en") or pool.get("name_ch"),
                                pool_id=str(pool.get("id")) if pool.get("id") is not None else None,
                                pool_status=pool.get("status"),
                                is_live=is_live,
                                line_id=str(line.get("lineId")) if line.get("lineId") is not None else None,
                                line_condition=line.get("condition"),
                                comb_id=str(comb.get("combId")),
                                comb_str=comb.get("str"),
                                comb_status=comb.get("status"),
                                selection_str=sel_str,
                                selection_name=sel_name,
                                odds=comb.get("currentOdds"),
                                offer_early_settlement=bool(comb.get("offerEarlySettlement")),
                            ))
        return rows

    def ingest(self, raw_results: Dict[str, Dict[str, Any]]) -> int:
        rows = self.flatten(raw_results)
        if not rows:
            return 0
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO odds_raw (
                    scraped_at, match_id, match_internal_id, match_label,
                    home_team, away_team, tournament, venue,
                    match_date, kickoff_time, match_status,
                    home_score, away_score, home_corners, away_corners,
                    odds_type, betting_type, pool_id, pool_status, is_live,
                    line_id, line_condition,
                    comb_id, comb_str, comb_status,
                    selection_str, selection_name, odds, offer_early_settlement
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (r.scraped_at, r.match_id, r.match_internal_id, r.match_label,
                     r.home_team, r.away_team, r.tournament, r.venue,
                     r.match_date, r.kickoff_time, r.match_status,
                     r.home_score, r.away_score, r.home_corners, r.away_corners,
                     r.odds_type, r.betting_type, r.pool_id, r.pool_status, int(r.is_live),
                     r.line_id, r.line_condition,
                     r.comb_id, r.comb_str, r.comb_status,
                     r.selection_str, r.selection_name, r.odds, int(r.offer_early_settlement))
                    for r in rows
                ],
            )
        return len(rows)

    def ingest_from_client(self, client, odds_types: List[str], start_index: int = 1,
                            end_index: int = 60, navigate_first: bool = True,
                            save_raw: bool = False) -> int:
        """Convenience: pull straight from HKJCGraphQLClient and ingest in one call."""
        results = client.fetch_multiple_odds_types(
            odds_types_list=odds_types, start_index=start_index, end_index=end_index,
            save_raw=save_raw, navigate_first=navigate_first,
        )
        return self.ingest(results)

    # ------------------------------------------------------------------ #
    # Aggregation / export
    # ------------------------------------------------------------------ #
    def latest(self, match_id: Optional[str] = None, odds_type: Optional[str] = None) -> pd.DataFrame:
        query = """
            SELECT r.* FROM odds_raw r
            JOIN (
                SELECT match_id, odds_type, COALESCE(pool_id,'') AS pool_id,
                       COALESCE(line_id,'') AS line_id, comb_id, MAX(scraped_at) AS max_ts
                FROM odds_raw {where}
                GROUP BY match_id, odds_type, COALESCE(pool_id,''), COALESCE(line_id,''), comb_id
            ) latest
            ON r.match_id=latest.match_id AND r.odds_type=latest.odds_type
           AND COALESCE(r.pool_id,'')=latest.pool_id AND COALESCE(r.line_id,'')=latest.line_id
           AND r.comb_id=latest.comb_id AND r.scraped_at=latest.max_ts
        """
        clauses, params = [], []
        if match_id:
            clauses.append("match_id = ?"); params.append(match_id)
        if odds_type:
            clauses.append("odds_type = ?"); params.append(odds_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._conn() as conn:
            return pd.read_sql_query(query.format(where=where), conn, params=params)

    def tournament_summary(self, odds_type: Optional[str] = None) -> pd.DataFrame:
        """Quick health-check aggregation: rows/matches/avg odds per odds_type, off latest snapshot."""
        df = self.latest(odds_type=odds_type)
        if df.empty:
            return df
        g = df.groupby("odds_type").agg(
            total_selections=("comb_id", "count"),
            unique_matches=("match_id", "nunique"),
            inplay_matches=("is_live", lambda s: df.loc[s.index][df.loc[s.index, "is_live"] == 1]["match_id"].nunique()),
            avg_odds=("odds", "mean"),
            median_odds=("odds", "median"),
            suspended_count=("comb_status", lambda s: (s == "SUSPENDED").sum()),
        ).reset_index()
        return g

    def export(self, out_path: str, fmt: str = "parquet", since: Optional[str] = None) -> str:
        query = "SELECT * FROM odds_raw"
        params: List[Any] = []
        if since:
            query += " WHERE scraped_at >= ?"; params.append(since)
        with self._conn() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "parquet":
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)
        return str(out)

    def row_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM odds_raw").fetchone()[0]


if __name__ == "__main__":
    wh = HKJCWarehouse("hkjc_odds_test.db")
    fake = {"HAD": {"data": {"matches": [{
        "id": "12345", "frontEndId": "FE1",
        "homeTeam": {"name_en": "Arsenal"}, "awayTeam": {"name_en": "Chelsea"},
        "tournament": {"name_en": "EPL"}, "venue": {"name_en": "Emirates"},
        "matchDate": "2026-09-20", "kickOffTime": "19:30", "status": "PRE",
        "runningResult": {"homeScore": 0, "awayScore": 0, "homeCorner": 0, "awayCorner": 0},
        "foPools": [{"id": "P1", "oddsType": "HAD", "status": "ACTIVE", "inplay": False,
            "lines": [{"lineId": "L1", "condition": "0.0", "combinations": [
                {"combId": "H", "str": "H", "status": "ACTIVE", "currentOdds": 2.1,
                 "selections": [{"str": "H", "name_en": "Home"}]},
                {"combId": "D", "str": "D", "status": "ACTIVE", "currentOdds": 3.3,
                 "selections": [{"str": "D", "name_en": "Draw"}]},
                {"combId": "A", "str": "A", "status": "ACTIVE", "currentOdds": 3.6,
                 "selections": [{"str": "A", "name_en": "Away"}]},
            ]}]}]}]}}}
    n = wh.ingest(fake)
    print(f"Ingested {n} rows, total {wh.row_count()}")
    print(wh.latest(match_id="FE1"))
    print(wh.tournament_summary())
