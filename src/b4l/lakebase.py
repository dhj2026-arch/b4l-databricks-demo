"""Lakebase operational state (SPEC.md 5.C), shared by 04_setup_lakebase and the tests.

Postgres layout in the project's `databricks_postgres` database:
- `{prefix}_sync.slot_price_recommendations`: snapshot synced table of the gold recommendations (read-only).
- `{prefix}_ops.live_court_slots`: one row per court and upcoming slot start. The live price is the tariff until staff
  approve the recommendation or override it, one court at a time, in the staff app.
- `{prefix}_ops.matchmaking_lobbies`: open doubles games (Phase 3 concierge tools).
- `{prefix}_ops.agent_action_log`: append-only log of every staff or agent action.

The setup notebook (running as the project owner) owns every table; the staff app's service principal only gets the
privileges below, so it can change a court's price and status and append to the log, nothing else.
"""

from __future__ import annotations

import re

import pandas as pd

from b4l import pricing_rules as pr

OPEN, BOOKED, BLOCKED = "open", "booked", "blocked"
PROPOSED, APPROVED, OVERRIDDEN, EXPIRED = "proposed", "approved", "overridden", "expired"
STAFF_DECISIONS = (APPROVED, OVERRIDDEN)
DEFAULT_GAME_MINUTES = 120

_IDENT = re.compile(r"^[a-z][a-z0-9_]*$")


def schema_names(prefix: str) -> tuple[str, str]:
    """(operational schema, synced schema) in Postgres; the synced one mirrors a UC schema of the same name."""
    ops, sync = f"{prefix}_ops", f"{prefix}_sync"
    for s in (ops, sync):
        if not _IDENT.match(s):
            raise ValueError(f"Invalid schema name: {s!r}")
    return ops, sync


def ddl(ops: str) -> list[str]:
    return [
        f"CREATE SCHEMA IF NOT EXISTS {ops}",
        f"""CREATE TABLE IF NOT EXISTS {ops}.live_court_slots (
            slot_id TEXT PRIMARY KEY,
            slot_key TEXT NOT NULL,
            court_id TEXT NOT NULL,
            hall_zone TEXT NOT NULL,
            slot_start TIMESTAMPTZ NOT NULL,
            tier TEXT NOT NULL,
            base_price_chf DOUBLE PRECISION NOT NULL,
            recommended_price_chf DOUBLE PRECISION NOT NULL,
            dynamic_price_chf DOUBLE PRECISION NOT NULL,
            applied_rule TEXT,
            reason TEXT,
            status TEXT NOT NULL CHECK (status IN ('proposed', 'approved', 'overridden', 'booked', 'blocked', 'expired')),
            locked BOOLEAN NOT NULL DEFAULT FALSE,
            decided_by TEXT,
            decided_at TIMESTAMPTZ,
            decision_reason TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CHECK (dynamic_price_chf BETWEEN {pr.PRICE_FLOOR_CHF} AND {pr.PRICE_CEILING_CHF})
        )""",
        f"CREATE INDEX IF NOT EXISTS live_court_slots_start ON {ops}.live_court_slots (slot_start)",
        f"""CREATE TABLE IF NOT EXISTS {ops}.matchmaking_lobbies (
            lobby_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            court_id TEXT,
            slot_start TIMESTAMPTZ NOT NULL,
            duration_minutes INTEGER NOT NULL DEFAULT {DEFAULT_GAME_MINUTES}
                CHECK (duration_minutes BETWEEN 30 AND 240 AND duration_minutes % 30 = 0),
            skill_tier TEXT NOT NULL,
            spots_open INTEGER NOT NULL CHECK (spots_open BETWEEN 0 AND 3),
            player_uuids TEXT[] NOT NULL DEFAULT '{{}}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
        f"""ALTER TABLE {ops}.matchmaking_lobbies ADD COLUMN IF NOT EXISTS duration_minutes INTEGER NOT NULL
            DEFAULT {DEFAULT_GAME_MINUTES} CHECK (duration_minutes BETWEEN 30 AND 240 AND duration_minutes % 30 = 0)""",
        f"""CREATE TABLE IF NOT EXISTS {ops}.agent_action_log (
            action_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            ts TIMESTAMPTZ NOT NULL DEFAULT now(),
            actor TEXT NOT NULL,
            action_type TEXT NOT NULL,
            details JSONB NOT NULL,
            status TEXT NOT NULL
        )""",
        f"CREATE INDEX IF NOT EXISTS agent_action_log_ts ON {ops}.agent_action_log (ts DESC)",
    ]


APP_UPDATABLE_COLUMNS = ("dynamic_price_chf", "status", "locked", "decided_by", "decided_at", "decision_reason",
                         "updated_at")


def grant_statements(ops: str, role: str) -> list[str]:
    """Least privilege for the staff app's service principal (role = its client id)."""
    if not re.fullmatch(r"[A-Za-z0-9-]+", role):
        raise ValueError(f"Invalid role: {role!r}")
    r = f'"{role}"'
    return [
        f"GRANT USAGE ON SCHEMA {ops} TO {r}",
        f"GRANT SELECT, UPDATE ({', '.join(APP_UPDATABLE_COLUMNS)}) ON {ops}.live_court_slots TO {r}",
        f"GRANT SELECT, INSERT, UPDATE ON {ops}.matchmaking_lobbies TO {r}",
        f"GRANT SELECT, INSERT ON {ops}.agent_action_log TO {r}",
    ]


def court_slot_states(slots: pd.DataFrame, courts: pd.DataFrame, bookings: pd.DataFrame,
                      blocks: pd.DataFrame) -> pd.DataFrame:
    """One row per (slot start, court): open, booked (an upcoming booking covers the start) or blocked.

    slots: slot_key, slot_start_ts. courts: court_id, hall_zone. bookings/blocks: court_id, start_ts, end_ts
    (bookings filtered to upcoming ones by the caller). A block wins over a booking on the same court and time.
    """
    grid = slots[["slot_key", "slot_start_ts"]].merge(courts[["court_id", "hall_zone"]], how="cross")
    grid["court_state"] = OPEN
    t = pd.to_datetime(grid["slot_start_ts"])
    for frame, state in ((bookings, BOOKED), (blocks, BLOCKED)):
        if frame.empty:
            continue
        f = frame[["court_id", "start_ts", "end_ts"]].copy()
        f["start_ts"], f["end_ts"] = pd.to_datetime(f["start_ts"]), pd.to_datetime(f["end_ts"])
        m = grid[["court_id"]].assign(t=t, i=grid.index).merge(f, on="court_id")
        hit = m.loc[(m["start_ts"] <= m["t"]) & (m["t"] < m["end_ts"]), "i"].unique()
        grid.loc[hit, "court_state"] = state
    grid["slot_id"] = grid["slot_key"] + "_" + grid["court_id"]
    return grid[["slot_id", "slot_key", "court_id", "hall_zone", "court_state"]]


STAGE_DDL = """CREATE TEMP TABLE court_state (
    slot_id TEXT PRIMARY KEY, slot_key TEXT, court_id TEXT, hall_zone TEXT, court_state TEXT
) ON COMMIT DROP"""


def merge_sql(ops: str, sync: str) -> str:
    """Upsert the staged court states joined to the synced recommendations.

    Open courts are `proposed` at the tariff until staff decide. Staff decisions on a court that is still open are
    kept on reruns (only the recommendation and reason refresh); booked and blocked courts follow the latest data.
    """
    return f"""
    INSERT INTO {ops}.live_court_slots AS live (
        slot_id, slot_key, court_id, hall_zone, slot_start, tier, base_price_chf, recommended_price_chf,
        dynamic_price_chf, applied_rule, reason, status, locked, updated_at)
    SELECT cs.slot_id, cs.slot_key, cs.court_id, cs.hall_zone, r.slot_start_ts, r.tier, r.base_price_chf,
           r.recommended_price_chf, r.base_price_chf, r.applied_rule, r.reason,
           CASE cs.court_state WHEN '{OPEN}' THEN '{PROPOSED}' ELSE cs.court_state END,
           cs.court_state <> '{OPEN}', now()
    FROM court_state cs JOIN {sync}.slot_price_recommendations r USING (slot_key)
    ON CONFLICT (slot_id) DO UPDATE SET
        tier = EXCLUDED.tier,
        base_price_chf = EXCLUDED.base_price_chf,
        recommended_price_chf = EXCLUDED.recommended_price_chf,
        applied_rule = EXCLUDED.applied_rule,
        reason = EXCLUDED.reason,
        updated_at = now(),
        dynamic_price_chf = CASE WHEN live.status IN ('{APPROVED}', '{OVERRIDDEN}') AND EXCLUDED.status = '{PROPOSED}'
                                 THEN live.dynamic_price_chf ELSE EXCLUDED.dynamic_price_chf END,
        locked = CASE WHEN live.status IN ('{APPROVED}', '{OVERRIDDEN}') AND EXCLUDED.status = '{PROPOSED}'
                      THEN TRUE ELSE EXCLUDED.locked END,
        status = CASE WHEN live.status IN ('{APPROVED}', '{OVERRIDDEN}') AND EXCLUDED.status = '{PROPOSED}'
                      THEN live.status ELSE EXCLUDED.status END
    """


def expire_sql(ops: str) -> str:
    """Slots no longer in the upcoming window keep their history but can no longer be changed."""
    return f"""UPDATE {ops}.live_court_slots SET status = '{EXPIRED}', locked = TRUE, updated_at = now()
    WHERE status <> '{EXPIRED}' AND slot_id NOT IN (SELECT slot_id FROM court_state)"""
