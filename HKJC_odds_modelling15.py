"""
1.5 live_oddsmodelling.py  (merged version)

Single, authoritative diff engine -- 1.4's poller should NOT run its own
separate change-detection anymore; it just calls ingest_batch() here.

Pipeline per market (match_id, odds_type, line_id):
  1. de-vig all combinations in the market together (sum implied prob = 1)
  2. compare de-vigged prob to the stored snapshot -> skip if below noise floor
  3. classify:
       in-play + score changed      -> inplay_reactive   (mechanical, low value)
       in-play + score unchanged    -> inplay_anticipatory (market moved ahead of the score -- interesting)
       pre-match + fast (velocity)  -> steam
       pre-match + slow but big     -> drift
       otherwise                    -> micro (not written, just updates snapshot)
  4. store event (incl. minutes_to_ko, acceleration), upsert snapshot
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from itertools import groupby
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

# Tunables -- calibrate against backtest before trusting live signals
STEAM_VELOCITY_THRESHOLD = 0.005    # >= 0.5pp implied-prob move per minute
DRIFT_PROB_THRESHOLD = 0.02         # >= 2pp cumulative move, any speed
MIN_PROB_CHANGE_TO_LOG = 0.0005     # noise floor -- below this, don't even write "micro"

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshot (
    match_id TEXT NOT NULL, odds_type TEXT NOT NULL,
    pool_id TEXT NOT NULL DEFAULT '',   -- SGA reuses ONE comb_id across ~20 pools; pool_id is part of the key
    line_id TEXT NOT NULL DEFAULT '',   -- comb_id repeats across lines, so the line is part of the key
    line_condition TEXT,                -- handicap / over-under label, e.g. '+0.5', '2.5/3.0', '0.0'
    comb_id TEXT NOT NULL,
    selection_name TEXT, odds REAL, implied_prob_devig REAL,
    home_score INTEGER, away_score INTEGER, is_live INTEGER,
    velocity REAL,                  -- kept so next event can compute acceleration
    last_updated TEXT,
    PRIMARY KEY (match_id, odds_type, pool_id, line_id, comb_id)
);

CREATE TABLE IF NOT EXISTS opening_odds (
    match_id TEXT NOT NULL, odds_type TEXT NOT NULL,
    pool_id TEXT NOT NULL DEFAULT '', line_id TEXT NOT NULL DEFAULT '', line_condition TEXT,
    comb_id TEXT NOT NULL,
    odds REAL, implied_prob_devig REAL, captured_at TEXT,
    PRIMARY KEY (match_id, odds_type, pool_id, line_id, comb_id)
);

CREATE TABLE IF NOT EXISTS odds_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT, odds_type TEXT,
    pool_id TEXT,
    line_id TEXT, line_condition TEXT,
    comb_id TEXT, selection_name TEXT,
    old_odds REAL, new_odds REAL, old_prob REAL, new_prob REAL,
    delta_prob REAL, delta_pct REAL, direction TEXT,
    seconds_since_last REAL, velocity REAL, acceleration REAL,
    minutes_to_ko REAL, is_live INTEGER,
    movement_type TEXT,   -- steam | drift | inplay_reactive | inplay_anticipatory
    event_time TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_match ON odds_events(match_id, odds_type, event_time);
CREATE INDEX IF NOT EXISTS idx_events_type ON odds_events(movement_type, event_time);
"""


@dataclass
class RawRow:
    match_id: str
    odds_type: str
    line_id: Optional[str]
    comb_id: str
    selection_name: Optional[str]
    odds: Optional[float]
    scraped_at: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    is_live: bool = False
    match_date: Optional[str] = None
    kickoff_time: Optional[str] = None
    line_condition: Optional[str] = None
    pool_id: Optional[str] = None


_HKT = timezone(timedelta(hours=8))


def _fmt_hkt(iso_str: Optional[str]) -> Optional[str]:
    """'2026-09-25T09:46:12+00:00' -> '0925 17:46:12 (HKT)'."""
    if not iso_str or pd.isna(iso_str):
        return iso_str
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_HKT).strftime("%m%d %H:%M:%S (HKT)")
    except Exception:
        return iso_str


def _minutes_to_ko(match_date: Optional[str], kickoff_time: Optional[str], scraped_at: str) -> Optional[float]:
    """Positive = minutes before kickoff, negative = minutes after (in-play).

    HKJC's kickOffTime is a full ISO datetime ('2026-09-27T02:45:00.000+08:00')
    and matchDate is a date with tz suffix ('2026-09-27+08:00'), so the two can
    never be concatenated; parse kickOffTime directly and only fall back to a
    match_date + plain-time join for plain 'YYYY-MM-DD' / 'HH:MM:SS' payloads.
    """
    if pd.isna(match_date) or pd.isna(kickoff_time) or not match_date or not kickoff_time:
        return None
    try:
        if "T" in kickoff_time:
            ko = datetime.fromisoformat(kickoff_time)
        else:
            date_part, _, off = str(match_date).partition("+")
            ko = datetime.fromisoformat(f"{date_part}T{kickoff_time}")
            if ko.tzinfo is None and off:
                hh, mm = off.split(":")[:2]
                ko = ko.replace(tzinfo=timezone(int(hh) * 60 + int(mm)))
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)
        now = datetime.fromisoformat(scraped_at)
        return (ko - now).total_seconds() / 60.0
    except Exception:
        return None


class LiveOddsModel:
    def __init__(self, db_path: str = "hkjc_odds.db"):
        self.db_path = db_path
        with self._conn() as conn:
            self._migrate(conn)

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
    # Schema migration
    # ------------------------------------------------------------------ #
    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone() is not None

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Idempotent upgrade to the pool+line-level schema.

        Older DBs keyed odds_snapshot/opening_odds by (match_id, odds_type, comb_id),
        then by (..., line_id, comb_id). Both are insufficient: comb_id repeats
        across lines (CHD/HDC/CHL/HIL) AND across pools (SGA publishes ~20 pools
        that all share comb_id='1'). Those tables are rebuilt from odds_raw,
        which is the append-only source of truth.
        """
        snap_cols = self._columns(conn, "odds_snapshot")
        if snap_cols and ("line_id" not in snap_cols or "pool_id" not in snap_cols):
            conn.execute("DROP TABLE IF EXISTS odds_snapshot")
        open_cols = self._columns(conn, "opening_odds")
        if open_cols and ("line_id" not in open_cols or "pool_id" not in open_cols):
            conn.execute("DROP TABLE IF EXISTS opening_odds")

        conn.executescript(SCHEMA)

        ev_cols = self._columns(conn, "odds_events")
        if ev_cols and "line_id" not in ev_cols:
            conn.execute("ALTER TABLE odds_events ADD COLUMN line_id TEXT")
            conn.execute("ALTER TABLE odds_events ADD COLUMN line_condition TEXT")
        if ev_cols and "pool_id" not in ev_cols:
            conn.execute("ALTER TABLE odds_events ADD COLUMN pool_id TEXT")

        # Backfill line attribution for events written before the line-key existed.
        # Only possible where the raw log maps the comb to a single line (HAD/SGA);
        # multi-line events can't be attributed retroactively and stay NULL, which
        # just excludes them from per-line counts.
        if self._table_exists(conn, "odds_raw"):
            conn.execute("""
                UPDATE odds_events
                SET line_id = (SELECT r.line_id FROM odds_raw r
                               WHERE r.match_id=odds_events.match_id
                                 AND r.odds_type=odds_events.odds_type
                                 AND r.comb_id=odds_events.comb_id LIMIT 1),
                    line_condition = (SELECT r.line_condition FROM odds_raw r
                               WHERE r.match_id=odds_events.match_id
                                 AND r.odds_type=odds_events.odds_type
                                 AND r.comb_id=odds_events.comb_id LIMIT 1)
                WHERE line_id IS NULL
                  AND (SELECT COUNT(DISTINCT COALESCE(r.line_id,'')) FROM odds_raw r
                       WHERE r.match_id=odds_events.match_id
                         AND r.odds_type=odds_events.odds_type
                         AND r.comb_id=odds_events.comb_id) = 1
            """)
            # Backfill pool attribution only where (odds_type, comb_id) maps to a
            # single pool. SGA's ~20 pools per match share a comb_id, so those
            # historical events cannot be attributed and stay NULL.
            conn.execute("""
                UPDATE odds_events
                SET pool_id = (SELECT r.pool_id FROM odds_raw r
                               WHERE r.match_id=odds_events.match_id
                                 AND r.odds_type=odds_events.odds_type
                                 AND r.comb_id=odds_events.comb_id LIMIT 1)
                WHERE pool_id IS NULL
                  AND (SELECT COUNT(DISTINCT COALESCE(r.pool_id,'')) FROM odds_raw r
                       WHERE r.match_id=odds_events.match_id
                         AND r.odds_type=odds_events.odds_type
                         AND r.comb_id=odds_events.comb_id) = 1
            """)

        empty = conn.execute("SELECT COUNT(*) FROM odds_snapshot").fetchone()[0] == 0
        if empty and self._table_exists(conn, "odds_raw"):
            self._rebuild_from_raw(conn)

    @staticmethod
    def _rebuild_from_raw(conn: sqlite3.Connection) -> None:
        """Recompute current + opening state (de-vig included) straight from odds_raw.

        Keys on (match_id, odds_type, pool_id, line_id, comb_id) so SGA's ~20
        pools survive. A single-outcome pool (SGA) is NOT normalised -- doing so
        would force its probability to 1.0 and erase movement; it keeps 1/odds.
        """
        conn.executescript("""
        INSERT INTO odds_snapshot (
            match_id, odds_type, pool_id, line_id, line_condition, comb_id, selection_name,
            odds, implied_prob_devig, home_score, away_score, is_live, velocity, last_updated)
        WITH ranked AS (
            SELECT r.*, ROW_NUMBER() OVER (
                       PARTITION BY r.match_id, r.odds_type, COALESCE(r.pool_id,''),
                                    COALESCE(r.line_id,''), r.comb_id
                       ORDER BY r.scraped_at DESC, r.id DESC) AS rn
            FROM odds_raw r WHERE r.odds IS NOT NULL AND r.odds > 0
        ),
        cur AS (SELECT * FROM ranked WHERE rn = 1),
        dev AS (
            SELECT match_id, odds_type, COALESCE(pool_id,'') AS pid, COALESCE(line_id,'') AS lid,
                   SUM(1.0/odds) AS tot, COUNT(*) AS n
            FROM cur GROUP BY 1, 2, 3, 4
        )
        SELECT c.match_id, c.odds_type, COALESCE(c.pool_id,''), COALESCE(c.line_id,''),
               c.line_condition, c.comb_id,
               c.selection_name, c.odds,
               CASE WHEN d.n = 1 THEN 1.0 / c.odds ELSE 1.0 / c.odds / d.tot END,
               c.home_score, c.away_score, c.is_live, 0.0, c.scraped_at
        FROM cur c
        JOIN dev d ON d.match_id=c.match_id AND d.odds_type=c.odds_type
                  AND d.pid=COALESCE(c.pool_id,'') AND d.lid=COALESCE(c.line_id,'')
        WHERE d.tot > 0;

        INSERT INTO opening_odds (
            match_id, odds_type, pool_id, line_id, line_condition, comb_id,
            odds, implied_prob_devig, captured_at)
        WITH ranked AS (
            SELECT r.*, ROW_NUMBER() OVER (
                       PARTITION BY r.match_id, r.odds_type, COALESCE(r.pool_id,''),
                                    COALESCE(r.line_id,''), r.comb_id
                       ORDER BY r.scraped_at ASC, r.id ASC) AS rn
            FROM odds_raw r WHERE r.odds IS NOT NULL AND r.odds > 0
        ),
        first AS (SELECT * FROM ranked WHERE rn = 1),
        dev AS (
            SELECT match_id, odds_type, COALESCE(pool_id,'') AS pid, COALESCE(line_id,'') AS lid,
                   SUM(1.0/odds) AS tot, COUNT(*) AS n
            FROM first GROUP BY 1, 2, 3, 4
        )
        SELECT f.match_id, f.odds_type, COALESCE(f.pool_id,''), COALESCE(f.line_id,''),
               f.line_condition, f.comb_id,
               f.odds, CASE WHEN d.n = 1 THEN 1.0 / f.odds ELSE 1.0 / f.odds / d.tot END,
               f.scraped_at
        FROM first f
        JOIN dev d ON d.match_id=f.match_id AND d.odds_type=f.odds_type
                  AND d.pid=COALESCE(f.pool_id,'') AND d.lid=COALESCE(f.line_id,'')
        WHERE d.tot > 0;
        """)

    @staticmethod
    def _devig(rows: List[RawRow]) -> Dict[str, float]:
        raw_probs = {r.comb_id: (1.0 / r.odds) for r in rows if r.odds and r.odds > 0}
        if not raw_probs:
            return {}
        # A single-outcome pool has nothing to de-vig against: normalising it
        # would always yield 1.0 and erase all movement signal. SGA is exactly
        # this case (one combination per pool), so keep the raw implied prob.
        if len(raw_probs) == 1:
            return dict(raw_probs)
        total = sum(raw_probs.values())
        return {k: v / total for k, v in raw_probs.items()} if total > 0 else {}

    def ingest_batch(self, rows: List[RawRow]) -> int:
        # pool_id is part of the market identity: SGA carries ~20 pools that all
        # share ONE comb_id, so without it they collapse into a single record.
        keyfn = lambda r: (r.match_id, r.odds_type, r.pool_id or "", r.line_id or "")
        rows_sorted = sorted(rows, key=keyfn)
        events_written = 0

        with self._conn() as conn:
            for market_key, group_iter in groupby(rows_sorted, key=keyfn):
                match_id, odds_type, pool_id, _line_id = market_key
                market_rows = list(group_iter)
                probs = self._devig(market_rows)

                for r in market_rows:
                    new_prob = probs.get(r.comb_id)
                    if new_prob is None or r.odds is None:
                        continue

                    prev = conn.execute(
                        """SELECT odds, implied_prob_devig, home_score, away_score,
                                  velocity, last_updated
                           FROM odds_snapshot
                           WHERE match_id=? AND odds_type=? AND pool_id=? AND line_id=? AND comb_id=?""",
                        (match_id, odds_type, r.pool_id or "", r.line_id or "", r.comb_id),
                    ).fetchone()

                    if prev is None:
                        conn.execute(
                            """INSERT OR IGNORE INTO opening_odds
                               (match_id, odds_type, pool_id, line_id, line_condition,
                                comb_id, odds, implied_prob_devig, captured_at)
                               VALUES (?,?,?,?,?,?,?,?,?)""",
                            (match_id, odds_type, r.pool_id or "", r.line_id or "", r.line_condition,
                             r.comb_id, r.odds, new_prob, r.scraped_at),
                        )
                        self._upsert_snapshot(conn, match_id, odds_type, r, new_prob, velocity=0.0)
                        continue

                    delta_prob = new_prob - prev["implied_prob_devig"]
                    if abs(delta_prob) < MIN_PROB_CHANGE_TO_LOG:
                        continue  # noise floor -- don't even touch last_updated

                    seconds = self._seconds_between(prev["last_updated"], r.scraped_at)
                    velocity = (delta_prob / (seconds / 60.0)) if seconds > 0 else 0.0
                    prev_velocity = prev["velocity"] or 0.0
                    minutes = (seconds / 60.0) if seconds > 0 else 1.0
                    acceleration = (velocity - prev_velocity) / minutes

                    score_changed = (r.home_score, r.away_score) != (prev["home_score"], prev["away_score"])
                    movement_type = self._classify(r.is_live, score_changed, delta_prob, velocity)
                    direction = "shorten" if delta_prob > 0 else "drift_out"
                    mtc = _minutes_to_ko(r.match_date, r.kickoff_time, r.scraped_at)

                    conn.execute(
                        """INSERT INTO odds_events (
                            match_id, odds_type, pool_id, line_id, line_condition,
                            comb_id, selection_name,
                            old_odds, new_odds, old_prob, new_prob,
                            delta_prob, delta_pct, direction,
                            seconds_since_last, velocity, acceleration,
                            minutes_to_ko, is_live, movement_type, event_time
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (match_id, odds_type, r.pool_id or "", r.line_id or "", r.line_condition,
                         r.comb_id, r.selection_name,
                         prev["odds"], r.odds, prev["implied_prob_devig"], new_prob,
                         delta_prob,
                         delta_prob / prev["implied_prob_devig"] if prev["implied_prob_devig"] else None,
                         direction, seconds, velocity, acceleration,
                         mtc, int(r.is_live), movement_type, r.scraped_at),
                    )
                    events_written += 1
                    self._upsert_snapshot(conn, match_id, odds_type, r, new_prob, velocity=velocity)

        return events_written

    @staticmethod
    def _upsert_snapshot(conn: sqlite3.Connection, match_id: str, odds_type: str,
                          r: RawRow, new_prob: float, velocity: float) -> None:
        conn.execute(
            """INSERT INTO odds_snapshot
               (match_id, odds_type, pool_id, line_id, line_condition, comb_id, selection_name,
                odds, implied_prob_devig, home_score, away_score, is_live, velocity, last_updated)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(match_id, odds_type, pool_id, line_id, comb_id) DO UPDATE SET
                   line_condition=excluded.line_condition,
                   odds=excluded.odds, implied_prob_devig=excluded.implied_prob_devig,
                   selection_name=excluded.selection_name,
                   home_score=excluded.home_score, away_score=excluded.away_score,
                   is_live=excluded.is_live, velocity=excluded.velocity,
                   last_updated=excluded.last_updated""",
            (match_id, odds_type, r.pool_id or "", r.line_id or "", r.line_condition,
             r.comb_id, r.selection_name,
             r.odds, new_prob, r.home_score, r.away_score, int(r.is_live), velocity, r.scraped_at),
        )

    @staticmethod
    def _seconds_between(t1: str, t2: str) -> float:
        try:
            return (datetime.fromisoformat(t2) - datetime.fromisoformat(t1)).total_seconds()
        except Exception:
            return 0.0

    @staticmethod
    def _classify(is_live: bool, score_changed: bool, delta_prob: float, velocity: float) -> str:
        if is_live:
            return "inplay_reactive" if score_changed else "inplay_anticipatory"
        if abs(velocity) >= STEAM_VELOCITY_THRESHOLD:
            return "steam"
        if abs(delta_prob) >= DRIFT_PROB_THRESHOLD:
            return "drift"
        return "micro"

    # ------------------------------------------------------------------ #
    # Bridge from warehouse's flattened DataFrame
    # ------------------------------------------------------------------ #
    def ingest_from_warehouse_df(self, df: pd.DataFrame) -> int:
        rows = [
            RawRow(
                match_id=row.match_id, odds_type=row.odds_type, line_id=row.line_id,
                line_condition=getattr(row, "line_condition", None),
                comb_id=row.comb_id, selection_name=row.selection_name, odds=row.odds,
                scraped_at=row.scraped_at,
                home_score=getattr(row, "home_score", None),
                away_score=getattr(row, "away_score", None),
                is_live=bool(getattr(row, "is_live", 0)),
                match_date=getattr(row, "match_date", None),
                kickoff_time=getattr(row, "kickoff_time", None),
                pool_id=getattr(row, "pool_id", None),
            )
            for row in df.itertuples(index=False)
        ]
        return self.ingest_batch(rows)

    # ------------------------------------------------------------------ #
    # Quant queries
    # ------------------------------------------------------------------ #
    def steam_moves(self, since: Optional[str] = None, match_id: Optional[str] = None) -> pd.DataFrame:
        query = "SELECT * FROM odds_events WHERE movement_type IN ('steam','inplay_anticipatory')"
        params: List[Any] = []
        if since:
            query += " AND event_time >= ?"; params.append(since)
        if match_id:
            query += " AND match_id = ?"; params.append(match_id)
        query += " ORDER BY event_time DESC"
        with self._conn() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def clv(self, match_id: str, odds_type: str, comb_id: str,
            line_id: Optional[str] = None, pool_id: Optional[str] = None) -> Optional[Dict[str, float]]:
        where = "match_id=? AND odds_type=? AND comb_id=?"
        params: List[Any] = [match_id, odds_type, comb_id]
        if line_id is not None:
            where += " AND line_id=?"
            params.append(line_id)
        if pool_id is not None:
            where += " AND pool_id=?"
            params.append(pool_id)
        # No line/pool given: a market may have several lines AND several pools
        # (SGA) sharing one comb_id, so pick a deterministic one instead of
        # letting fetchone() choose at random.
        tail = "" if (line_id is not None or pool_id is not None) else " ORDER BY pool_id, line_id LIMIT 1"
        with self._conn() as conn:
            opening = conn.execute(
                f"SELECT implied_prob_devig, pool_id, line_id, line_condition FROM opening_odds "
                f"WHERE {where}{tail}", tuple(params)).fetchone()
            current = conn.execute(
                f"SELECT implied_prob_devig, pool_id, line_id, line_condition FROM odds_snapshot "
                f"WHERE {where}{tail}", tuple(params)).fetchone()
        if not opening or not current:
            return None
        return {"pool_id": current["pool_id"],
                "line_id": current["line_id"],
                "line_condition": current["line_condition"],
                "opening_prob": opening["implied_prob_devig"],
                "current_prob": current["implied_prob_devig"],
                "clv_prob_shift": current["implied_prob_devig"] - opening["implied_prob_devig"]}

    def market_summary(self, match_id: str) -> pd.DataFrame:
        query = """
        SELECT s.match_id, s.odds_type, s.pool_id, s.line_id, s.line_condition, s.selection_name,
           s.comb_id,
           o.odds AS opening_odds,
           s.odds AS current_odds,
           o.implied_prob_devig AS opening_prob, 
           s.implied_prob_devig AS current_prob,
           s.implied_prob_devig - o.implied_prob_devig AS cumulative_drift,
           s.last_updated,
           (SELECT COUNT(*) FROM odds_events e WHERE e.match_id=s.match_id
            AND e.odds_type=s.odds_type AND e.pool_id=s.pool_id AND e.line_id=s.line_id
            AND e.comb_id=s.comb_id
            AND e.movement_type='steam') AS steam_count,
           (SELECT COUNT(*) FROM odds_events e WHERE e.match_id=s.match_id
            AND e.odds_type=s.odds_type AND e.pool_id=s.pool_id AND e.line_id=s.line_id
            AND e.comb_id=s.comb_id
            AND e.movement_type='inplay_anticipatory') AS anticipatory_count,
           (SELECT MAX(event_time) FROM odds_events e WHERE e.match_id=s.match_id
            AND e.odds_type=s.odds_type AND e.pool_id=s.pool_id AND e.line_id=s.line_id
            AND e.comb_id=s.comb_id) AS last_event_time
        FROM odds_snapshot s
        LEFT JOIN opening_odds o
        ON s.match_id=o.match_id AND s.odds_type=o.odds_type
        AND s.pool_id=o.pool_id AND s.line_id=o.line_id AND s.comb_id=o.comb_id
        WHERE s.match_id = ?
        ORDER BY s.odds_type, s.pool_id, s.line_id, s.comb_id
        """
        with self._conn() as conn:
            df = pd.read_sql_query(query, conn, params=(match_id,))
        if not df.empty:
            df["last_updated"] = df["last_updated"].apply(_fmt_hkt)
        return df

    def debug_snapshot_freshness(self, match_id: str, odds_type: str) -> pd.DataFrame:
        """
        Diagnostic for 'current_prob looks frozen across polls'.
        For each comb: snapshot.last_updated vs the latest odds_events.event_time.
        - If last_event_time is NEWER than snapshot.last_updated -> real bug:
          events are being written but _upsert_snapshot isn't landing (or you're
          reading a different db file than the poller is writing to).
        - If both timestamps stop advancing across polls -> odds genuinely
          aren't changing for this comb (not a bug); check a different comb
          in the same market, or confirm via liveodds_summary() below.
        """
        query = """
        SELECT s.pool_id, s.line_id, s.line_condition, s.comb_id, s.selection_name,
               s.odds, s.implied_prob_devig,
               s.last_updated AS snapshot_last_updated,
               (SELECT MAX(event_time) FROM odds_events e
                WHERE e.match_id=s.match_id AND e.odds_type=s.odds_type
                  AND e.pool_id=s.pool_id AND e.line_id=s.line_id AND e.comb_id=s.comb_id) AS last_event_time,
               (SELECT COUNT(*) FROM odds_events e
                WHERE e.match_id=s.match_id AND e.odds_type=s.odds_type
                  AND e.pool_id=s.pool_id AND e.line_id=s.line_id AND e.comb_id=s.comb_id) AS total_events
        FROM odds_snapshot s
        WHERE s.match_id = ? AND s.odds_type = ?
        """
        with self._conn() as conn:
            df = pd.read_sql_query(query, conn, params=(match_id, odds_type))
        if not df.empty:
            df["snapshot_matches_latest_event"] = df["snapshot_last_updated"] == df["last_event_time"]
        return df

    def liveodds_summary(self, match_id: str, odds_type: str) -> pd.DataFrame:
        """
        Outputs odds history formatted like HKJC UI:
        - Index: Scraped timestamp (MMDD HH:MM)
        - Columns: Condition + Selection (e.g. '主', '和', '客' or '2.5 大', '2.5 小')
        - Values: Formatted string with odds and % change vs previous poll (e.g. '⬆1.89 (6.78%)')
        """
        query = """
            SELECT scraped_at, line_condition, selection_name, odds, comb_id, pool_id
            FROM odds_raw
            WHERE match_id = ? AND odds_type = ?
            ORDER BY scraped_at, pool_id, comb_id
        """
        with self._conn() as conn:
            df = pd.read_sql_query(query, conn, params=(match_id, odds_type))

        if df.empty:
            return pd.DataFrame()

        # 1. Build column label: '2.5 大' for Handicap/Totals, or '主' for 1X2
        def build_label(row):
            cond = str(row["line_condition"]).strip() if pd.notna(row["line_condition"]) and str(row["line_condition"]).strip().lower() != "none" else ""
            sel = str(row["selection_name"]).strip() if pd.notna(row["selection_name"]) else ""
            return f"{cond} {sel}".strip()

        df["label"] = df.apply(build_label, axis=1)

        # Keep natural selection order (主 -> 和 -> 客 or 大 -> 小)
        col_order = df["label"].drop_duplicates().tolist()

        # Format timestamp for row header (e.g. 0924 12:04)
        df["scraped_at_fmt"] = pd.to_datetime(df["scraped_at"]).dt.strftime("%m%d %H:%M")

        # 2. Pivot raw odds matrix
        odds_pivot = df.pivot_table(
            index="scraped_at_fmt", 
            columns="label", 
            values="odds", 
            aggfunc="last"
        )

        # Sort rows chronologically & columns by original order
        time_order = df.groupby("scraped_at_fmt")["scraped_at"].min().sort_values().index
        odds_pivot = odds_pivot.reindex(index=time_order, columns=col_order)

        # Forward fill missing odds across polls if any
        odds_pivot = odds_pivot.ffill()

        # 3. Calculate % change vs PREVIOUS poll row
        pct_pivot = odds_pivot.pct_change()

        # 4. Format cell strings (⬆/⬇ + percentage)
        out_df = odds_pivot.copy().astype(object)

        for col in odds_pivot.columns:
            for idx in range(len(odds_pivot)):
                val = odds_pivot.iloc[idx][col]
                pct = pct_pivot.iloc[idx][col]

                if pd.isna(val):
                    out_df.iloc[idx, out_df.columns.get_loc(col)] = ""
                elif idx == 0 or pd.isna(pct) or pct == 0:
                    out_df.iloc[idx, out_df.columns.get_loc(col)] = f"{val:g}"
                elif pct > 0:
                    out_df.iloc[idx, out_df.columns.get_loc(col)] = f"⬆{val:g} ({pct:.2%})"
                else:
                    out_df.iloc[idx, out_df.columns.get_loc(col)] = f"⬇{val:g} ({abs(pct):.2%})"

        out_df.index.name = "#"
        return out_df



    @staticmethod
    def compute_clv(entry_odds: float, closing_odds: float, stake: float = 1.0) -> float:
        """Simple CLV in stake units: positive = you beat the closing line."""
        return (closing_odds / entry_odds - 1.0) * stake


if __name__ == "__main__":
    model = LiveOddsModel("hkjc_odds_test.db")

    batch1 = [
        RawRow("FE1", "HAD", "L1", "H", "Home", 2.10, "2026-09-20T18:00:00+00:00",
               0, 0, False, "2026-09-20", "19:30:00"),
        RawRow("FE1", "HAD", "L1", "D", "Draw", 3.30, "2026-09-20T18:00:00+00:00",
               0, 0, False, "2026-09-20", "19:30:00"),
        RawRow("FE1", "HAD", "L1", "A", "Away", 3.60, "2026-09-20T18:00:00+00:00",
               0, 0, False, "2026-09-20", "19:30:00"),
    ]
    print("Batch1 events:", model.ingest_batch(batch1), "(expect 0, opening line)")

    # pre-match steam: Home shortens fast
    batch2 = [
        RawRow("FE1", "HAD", "L1", "H", "Home", 1.85, "2026-09-20T18:02:00+00:00",
               0, 0, False, "2026-09-20", "19:30:00"),
        RawRow("FE1", "HAD", "L1", "D", "Draw", 3.40, "2026-09-20T18:02:00+00:00",
               0, 0, False, "2026-09-20", "19:30:00"),
        RawRow("FE1", "HAD", "L1", "A", "Away", 3.90, "2026-09-20T18:02:00+00:00",
               0, 0, False, "2026-09-20", "19:30:00"),
    ]
    print("Batch2 events:", model.ingest_batch(batch2), "(expect 3, steam)")

    # in-play, odds move BEFORE the score changes -> anticipatory
    batch3 = [
        RawRow("FE1", "HAD", "L1", "H", "Home", 1.60, "2026-09-20T20:15:00+00:00",
               0, 0, True, "2026-09-20", "19:30:00"),
        RawRow("FE1", "HAD", "L1", "D", "Draw", 3.80, "2026-09-20T20:15:00+00:00",
               0, 0, True, "2026-09-20", "19:30:00"),
        RawRow("FE1", "HAD", "L1", "A", "Away", 5.00, "2026-09-20T20:15:00+00:00",
               0, 0, True, "2026-09-20", "19:30:00"),
    ]
    print("Batch3 events:", model.ingest_batch(batch3), "(expect 3, inplay_anticipatory, score unchanged)")

    df = model.steam_moves(match_id="FE1")
    print(df[["comb_id", "delta_prob", "velocity", "acceleration", "minutes_to_ko", "movement_type"]])
    print(model.market_summary("FE1"))
