"""Live court slots, price decisions and the action log in Lakebase (Postgres), plus an in-memory mock.

The tables are created and refreshed by src/04_setup_lakebase.py (copy of the layout in src/b4l/lakebase.py). The app's
service principal can only change a court's price, status and decision columns, add doubles lobbies and append to
the action log. Each decision is one transaction: lock the court row, check it, update it, log it.
"""

from __future__ import annotations

import os
import threading
import uuid
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import pandas as pd

import agent_tools as at
import domain as d
import mock_data

PROPOSED, APPROVED, OVERRIDDEN = "proposed", "approved", "overridden"
BOOKED, BLOCKED, EXPIRED = "booked", "blocked", "expired"
DECIDABLE = (PROPOSED, APPROVED, OVERRIDDEN)
APPROVE, OVERRIDE = "approve_price", "override_price"
CONCIERGE, LOBBY, INVITE = "send_concierge_message", "create_doubles_lobby", "send_lobby_invitation"
SENT_DEMO = "approved (demo: not sent)"

COURT_SLOT_COLUMNS = [
    "slot_id", "slot_key", "court_id", "hall_zone", "tier", "base_price_chf", "recommended_price_chf",
    "dynamic_price_chf", "applied_rule", "reason", "status", "decided_by", "decided_at", "decision_reason",
]
SUMMARY_COLUMNS = [
    "slot_key", "tier", "base_price_chf", "recommended_price_chf", "reason", "proposed", "approved", "overridden",
    "booked", "blocked",
]
LOG_COLUMNS = ["ts", "actor", "action_type", "status", "summary", "court_id", "slot_key", "old_price_chf",
               "new_price_chf", "recommended_price_chf", "reason"]
LOBBY_COLUMNS = ["lobby_id", "slot_start", "duration_minutes", "court_id", "skill_tier", "spots_open", "player_uuids",
                 "created_at"]
LOBBY_SPOTS = 3
LOG_LIMIT = 50
TOKEN_TTL_S = 45 * 60


class DecisionError(ValueError):
    """A decision the guardrails or the court's state do not allow; nothing was written."""


def check_decidable(row: dict) -> None:
    if row["status"] not in DECIDABLE:
        raise DecisionError(f"{row['court_id']} at {slot_time(row['slot_key'])} is {row['status']}; "
                            "only open courts can be re-priced.")


def check_override(tier: str, price: float, reason: str) -> float:
    if not (reason or "").strip():
        raise DecisionError("Give a reason for the manual override.")
    lo, hi = d.price_bounds(tier)
    if price != round(price):
        raise DecisionError("Prices are whole francs.")
    if not lo <= price <= hi:
        raise DecisionError(f"CHF {price:.0f} is outside the {tier} range CHF {lo:.0f}–{hi:.0f} "
                            f"(CHF {d.PRICE_FLOOR_CHF:.0f}–{d.PRICE_CEILING_CHF:.0f}, max ±{d.MAX_MOVE_PCT:.0%} "
                            "from the tariff).")
    return float(price)


def slot_time(slot_key: str) -> str:
    """'2026-11-24T1200' -> 'Tue 24 Nov 12:00'."""
    return f"{datetime.strptime(slot_key, '%Y-%m-%dT%H%M'):%a %d %b %H:%M}"


def log_details(row: dict, new_price: float, reason: str | None) -> dict:
    return {
        "slot_id": row["slot_id"], "slot_key": row["slot_key"], "court_id": row["court_id"], "tier": row["tier"],
        "old_status": row["status"], "old_price_chf": float(row["dynamic_price_chf"]), "new_price_chf": new_price,
        "base_price_chf": float(row["base_price_chf"]),
        "recommended_price_chf": float(row["recommended_price_chf"]), "reason": reason,
    }


def short_uuid(u: str | None) -> str:
    return f"{u[:8]}…" if u else "?"


def log_summary(action: str, x: dict) -> str:
    if action in (APPROVE, OVERRIDE):
        return (f"{x.get('court_id')} {slot_time(x['slot_key']) if x.get('slot_key') else ''}: "
                f"CHF {x.get('old_price_chf', 0):.0f} → {x.get('new_price_chf', 0):.0f}")
    if action == CONCIERGE:
        return f"{x.get('trigger_type')} to {short_uuid(x.get('player_uuid'))} ({x.get('language')})"
    if action == INVITE:
        return f"Lobby invitation to {short_uuid(x.get('player_uuid'))} ({x.get('role')}, {x.get('language')})"
    if action == LOBBY:
        return (f"Doubles {x.get('skill_tier')} {x.get('slot_start', '')[:16].replace('T', ' ')} "
                f"{at.duration_label(x.get('duration_minutes', 0), False)} {x.get('court_id')} "
                f"CHF {at.chf_amount(x.get('price_chf', 0))}: organiser {short_uuid(x.get('organiser_uuid'))}, "
                f"{len(x.get('invited_uuids') or [])} invited")
    return action


def _log_frame(entries: list[tuple]) -> pd.DataFrame:
    rows = [(ts, actor, action, status, log_summary(action, x), x.get("court_id"), x.get("slot_key"),
             x.get("old_price_chf"),
             x.get("new_price_chf"), x.get("recommended_price_chf"), x.get("reason"))
            for ts, actor, action, status, x in entries]
    return pd.DataFrame(rows, columns=LOG_COLUMNS)


def _summary(courts: pd.DataFrame) -> pd.DataFrame:
    if courts.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    counts = pd.crosstab(courts["slot_key"], courts["status"])
    first = courts.groupby("slot_key")[["tier", "base_price_chf", "recommended_price_chf", "reason"]].first()
    out = first.join(counts).reset_index()
    for c in ("proposed", "approved", "overridden", "booked", "blocked"):
        out[c] = out[c].astype(int) if c in out else 0
    return out[SUMMARY_COLUMNS].sort_values("slot_key").reset_index(drop=True)


def concierge_details(draft: dict, message: str) -> dict:
    """Log entry for an approved concierge message; staff may have edited the draft."""
    if not draft.get("eligible"):
        raise DecisionError(draft.get("note") or "The trigger does not apply to this player.")
    if not (message or "").strip():
        raise DecisionError("The message is empty.")
    return {"player_uuid": draft["player_uuid"], "trigger_type": draft["trigger_type"],
            "language": draft["language"], "message": message.strip(), "edited": message.strip() != draft["message"]}


def lobby_details(organiser_uuid: str, skill_tier: str, game: dict, invited: list[str]) -> dict:
    """game: court_id, slot_key (start), minutes, price_chf and tariff_chf as quoted in the invitations."""
    if not invited:
        raise DecisionError("Pick at least one partner to invite.")
    if len(invited) > LOBBY_SPOTS:
        raise DecisionError(f"A doubles game has {LOBBY_SPOTS} spots besides the organiser.")
    if organiser_uuid in invited:
        raise DecisionError("The organiser cannot invite themselves.")
    if game["minutes"] not in at.GAME_MINUTES:
        raise DecisionError(f"Game length must be one of {', '.join(str(m) for m in at.GAME_MINUTES)} minutes.")
    start = datetime.strptime(game["slot_key"], "%Y-%m-%dT%H%M")
    return {"organiser_uuid": organiser_uuid, "skill_tier": skill_tier, "slot_start": start.isoformat(),
            "duration_minutes": int(game["minutes"]), "court_id": game["court_id"],
            "price_chf": float(game["price_chf"]), "tariff_chf": float(game["tariff_chf"]),
            "invited_uuids": list(invited), "spots_open": LOBBY_SPOTS}


def game_keys(lobby: dict) -> list[str]:
    start = datetime.fromisoformat(lobby["slot_start"])
    return [at.slot_key(start + pd.Timedelta(minutes=30 * i)) for i in range(lobby["duration_minutes"] // 30)]


def check_lobby_game(rows: list[dict], lobby: dict) -> None:
    """The court must still be open for the whole game at the price quoted in the invitations."""
    g = at.court_game(at.court_index(pd.DataFrame(rows)) if rows else {}, lobby["court_id"],
                      datetime.fromisoformat(lobby["slot_start"]), lobby["duration_minutes"])
    when = slot_time(at.slot_key(datetime.fromisoformat(lobby["slot_start"])))
    if g is None:
        raise DecisionError(f"{lobby['court_id']} is no longer open for the whole game from {when}; pick another "
                            "court or time.")
    if g["price_chf"] != lobby["price_chf"]:
        raise DecisionError(f"The price of the game on {lobby['court_id']} changed to CHF "
                            f"{at.chf_amount(g['price_chf'])}; check the invitations again.")


def invitation_details(lobby: dict, invitations: list[dict]) -> list[dict]:
    """One log entry per person; the drafts must cover exactly the organiser and the invited partners."""
    people = [lobby["organiser_uuid"], *lobby["invited_uuids"]]
    if sorted(i["player_uuid"] for i in invitations) != sorted(people):
        raise DecisionError("The invitations do not match the organiser and the invited partners.")
    if any(not (i.get("message") or "").strip() for i in invitations):
        raise DecisionError("An invitation is empty.")
    return [{"player_uuid": i["player_uuid"], "role": i["role"], "language": i["language"],
             "message": i["message"].strip(), "slot_start": lobby["slot_start"],
             "duration_minutes": lobby["duration_minutes"], "court_id": lobby["court_id"],
             "price_chf": lobby["price_chf"]} for i in invitations]


class LakebaseStore:
    name = "live"

    def __init__(self, w, endpoint: str, database: str, ops_schema: str):
        self.w = w
        self.endpoint = endpoint
        self.database = os.getenv("PGDATABASE") or database
        self.ops = ops_schema
        self._lock = threading.Lock()
        self._token: str | None = None
        self._token_at = 0.0
        self._host: str | None = None
        self._user: str | None = None

    def _credentials(self) -> tuple[str, str, str]:
        with self._lock:
            if self._host is None:
                self._host = os.getenv("PGHOST") or self.w.postgres.get_endpoint(name=self.endpoint).status.hosts.host
                self._user = os.getenv("PGUSER") or self.w.config.client_id or self.w.current_user.me().user_name
            if self._token is None or time.time() - self._token_at > TOKEN_TTL_S:
                self._token = self.w.postgres.generate_database_credential(endpoint=self.endpoint).token
                self._token_at = time.time()
            return self._host, self._user, self._token

    def _connect(self):
        import psycopg

        host, user, token = self._credentials()
        # autocommit so that conn.transaction() opens a real transaction instead of a savepoint.
        return psycopg.connect(host=host, dbname=self.database, user=user, password=token, sslmode="require",
                               autocommit=True, connect_timeout=30)

    def _frame(self, sql: str, params: tuple, columns: list[str]) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return pd.DataFrame(rows, columns=columns)

    def court_slots_for_day(self, day: date) -> pd.DataFrame:
        return self._frame(
            f"SELECT {', '.join(COURT_SLOT_COLUMNS)} FROM {self.ops}.live_court_slots "
            "WHERE slot_key LIKE %s AND status <> %s ORDER BY slot_key, court_id",
            (f"{day.isoformat()}T%", EXPIRED), COURT_SLOT_COLUMNS)

    def slot_summary(self, day: date) -> pd.DataFrame:
        return _summary(self.court_slots_for_day(day))

    def court_slots_between(self, start: date, end: date) -> pd.DataFrame:
        """Court slots with start day in [start, end)."""
        return self._frame(
            f"SELECT {', '.join(COURT_SLOT_COLUMNS)} FROM {self.ops}.live_court_slots "
            "WHERE slot_key >= %s AND slot_key < %s AND status <> %s ORDER BY slot_key, court_id",
            (start.isoformat(), end.isoformat(), EXPIRED), COURT_SLOT_COLUMNS)

    def court_slots(self, slot_key: str) -> pd.DataFrame:
        return self._frame(
            f"SELECT {', '.join(COURT_SLOT_COLUMNS)} FROM {self.ops}.live_court_slots WHERE slot_key = %s "
            "ORDER BY court_id", (slot_key,), COURT_SLOT_COLUMNS)

    def _decide(self, slot_id: str, actor: str, action: str, price: float | None, reason: str | None) -> dict:
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb

        with self._connect() as conn, conn.transaction():
            cur = conn.cursor(row_factory=dict_row)
            row = cur.execute(f"SELECT * FROM {self.ops}.live_court_slots WHERE slot_id = %s FOR UPDATE",
                              (slot_id,)).fetchone()
            if row is None:
                raise DecisionError(f"Unknown court slot {slot_id}.")
            check_decidable(row)
            if action == APPROVE:
                new_price, status = float(row["recommended_price_chf"]), APPROVED
            else:
                new_price, status = check_override(row["tier"], price, reason), OVERRIDDEN
            updated = cur.execute(
                f"""UPDATE {self.ops}.live_court_slots SET dynamic_price_chf = %s, status = %s, locked = TRUE,
                    decided_by = %s, decided_at = now(), decision_reason = %s, updated_at = now()
                    WHERE slot_id = %s RETURNING slot_id, court_id, slot_key, status, dynamic_price_chf""",
                (new_price, status, actor, reason, slot_id)).fetchone()
            cur.execute(
                f"INSERT INTO {self.ops}.agent_action_log (actor, action_type, details, status) VALUES (%s, %s, %s, %s)",
                (actor, action, Jsonb(log_details(row, new_price, reason)), "applied"))
        return updated

    def _insert_log(self, cur, actor: str, action: str, details: dict, status: str) -> None:
        from psycopg.types.json import Jsonb

        cur.execute(
            f"INSERT INTO {self.ops}.agent_action_log (actor, action_type, details, status) VALUES (%s, %s, %s, %s)",
            (actor, action, Jsonb(details), status))

    def send_concierge(self, draft: dict, message: str, actor: str) -> dict:
        details = concierge_details(draft, message)
        with self._connect() as conn:
            self._insert_log(conn.cursor(), actor, CONCIERGE, details, SENT_DEMO)
        return details

    def create_lobby(self, organiser_uuid: str, skill_tier: str, game: dict, invited: list[str],
                     invitations: list[dict], actor: str) -> dict:
        from psycopg.rows import dict_row

        details = lobby_details(organiser_uuid, skill_tier, game, invited)
        invites = invitation_details(details, invitations)
        with self._connect() as conn, conn.transaction():
            cur = conn.cursor(row_factory=dict_row)
            rows = cur.execute(
                f"SELECT {', '.join(COURT_SLOT_COLUMNS)} FROM {self.ops}.live_court_slots "
                "WHERE court_id = %s AND slot_key = ANY(%s) FOR SHARE",
                (details["court_id"], game_keys(details))).fetchall()
            check_lobby_game(rows, details)
            lobby_id = cur.execute(
                f"""INSERT INTO {self.ops}.matchmaking_lobbies (court_id, slot_start, duration_minutes, skill_tier,
                    spots_open, player_uuids) VALUES (%s, %s, %s, %s, %s, %s) RETURNING lobby_id""",
                (details["court_id"], datetime.fromisoformat(details["slot_start"]), details["duration_minutes"],
                 skill_tier, LOBBY_SPOTS, [organiser_uuid])).fetchone()["lobby_id"]
            details["lobby_id"] = str(lobby_id)
            self._insert_log(cur, actor, LOBBY, details, "open")
            for i in invites:
                self._insert_log(cur, actor, INVITE, {**i, "lobby_id": details["lobby_id"]}, SENT_DEMO)
        return details

    def lobbies(self, limit: int = LOG_LIMIT) -> pd.DataFrame:
        return self._frame(f"SELECT {', '.join(LOBBY_COLUMNS)} FROM {self.ops}.matchmaking_lobbies "
                           "ORDER BY created_at DESC LIMIT %s", (limit,), LOBBY_COLUMNS)

    def approve(self, slot_id: str, actor: str) -> dict:
        return self._decide(slot_id, actor, APPROVE, None, None)

    def override(self, slot_id: str, price: float, reason: str, actor: str) -> dict:
        return self._decide(slot_id, actor, OVERRIDE, price, reason.strip())

    def action_log(self, limit: int = LOG_LIMIT) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT ts, actor, action_type, status, details FROM {self.ops}.agent_action_log "
                "ORDER BY ts DESC LIMIT %s", (limit,)).fetchall()
        return _log_frame(rows)


def _mock_court_slots(as_of: date) -> pd.DataFrame:
    recs = mock_data.price_recommendations(as_of)
    booked_now_by_key: dict[str, int] = {}
    for day in sorted(pd.to_datetime(recs["slot_start_ts"]).dt.date.unique()):
        f = mock_data.slot_features(day, as_of)
        booked_now_by_key.update(zip(f["slot_key"], f["booked_courts_now"].astype(int)))
    rows = []
    for r in recs.itertuples(index=False):
        start = pd.Timestamp(r.slot_start_ts).to_pydatetime()
        day, hour = start.date(), start.hour
        need = 60 if start.minute == 0 else 30
        blocked = [c for c in d.COURT_IDS if mock_data._blocked(day, c, hour) >= need]
        free = [c for c in d.COURT_IDS if c not in blocked]
        booked_now = booked_now_by_key[r.slot_key]
        booked = set(free[len(free) - booked_now:]) if booked_now else set()
        base, rec = float(r.base_price_chf), float(r.recommended_price_chf)
        reason = (f"Recommended CHF {rec:.0f} instead of CHF {base:.0f} (mock demand curve)" if rec != base
                  else f"Tariff CHF {base:.0f} kept (mock demand curve)")
        for c in d.COURT_IDS:
            status = BLOCKED if c in blocked else BOOKED if c in booked else PROPOSED
            rows.append((f"{r.slot_key}_{c}", r.slot_key, c, d.HALL_ZONE[c], r.tier, base, rec, base,
                         r.applied_rule, reason, status, None, None, None))
    return pd.DataFrame(rows, columns=COURT_SLOT_COLUMNS)


@dataclass
class MockOpsStore:
    """Same interface as LakebaseStore; state lives in this process only."""

    as_of: date
    name: str = "mock"
    _slots: pd.DataFrame = field(init=False, repr=False)
    _log: list[tuple] = field(default_factory=list, init=False, repr=False)
    _lobbies: list[tuple] = field(default_factory=list, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        self._slots = _mock_court_slots(self.as_of).set_index("slot_id", drop=False)

    def court_slots_for_day(self, day: date) -> pd.DataFrame:
        s = self._slots[self._slots["slot_key"].str.startswith(day.isoformat())]
        return s.reset_index(drop=True).copy()

    def slot_summary(self, day: date) -> pd.DataFrame:
        return _summary(self.court_slots_for_day(day))

    def court_slots_between(self, start: date, end: date) -> pd.DataFrame:
        k = self._slots["slot_key"]
        return self._slots[(k >= start.isoformat()) & (k < end.isoformat())].reset_index(drop=True).copy()

    def court_slots(self, slot_key: str) -> pd.DataFrame:
        return self._slots[self._slots["slot_key"] == slot_key].reset_index(drop=True).copy()

    def _decide(self, slot_id: str, actor: str, action: str, price: float | None, reason: str | None) -> dict:
        with self._lock:
            if slot_id not in self._slots.index:
                raise DecisionError(f"Unknown court slot {slot_id}.")
            row = self._slots.loc[slot_id].to_dict()
            check_decidable(row)
            if action == APPROVE:
                new_price, status = float(row["recommended_price_chf"]), APPROVED
            else:
                new_price, status = check_override(row["tier"], price, reason), OVERRIDDEN
            now = datetime.now(timezone.utc)
            for col, value in (("dynamic_price_chf", new_price), ("status", status), ("decided_by", actor),
                               ("decided_at", now), ("decision_reason", reason)):
                self._slots.at[slot_id, col] = value
            self._log.append((now, actor, action, "applied", log_details(row, new_price, reason)))
        return {"slot_id": slot_id, "court_id": row["court_id"], "slot_key": row["slot_key"], "status": status,
                "dynamic_price_chf": new_price}

    def approve(self, slot_id: str, actor: str) -> dict:
        return self._decide(slot_id, actor, APPROVE, None, None)

    def override(self, slot_id: str, price: float, reason: str, actor: str) -> dict:
        return self._decide(slot_id, actor, OVERRIDE, price, (reason or "").strip())

    def action_log(self, limit: int = LOG_LIMIT) -> pd.DataFrame:
        return _log_frame(list(reversed(self._log))[:limit])

    def send_concierge(self, draft: dict, message: str, actor: str) -> dict:
        details = concierge_details(draft, message)
        with self._lock:
            self._log.append((datetime.now(timezone.utc), actor, CONCIERGE, SENT_DEMO, details))
        return details

    def create_lobby(self, organiser_uuid: str, skill_tier: str, game: dict, invited: list[str],
                     invitations: list[dict], actor: str) -> dict:
        details = lobby_details(organiser_uuid, skill_tier, game, invited)
        invites = invitation_details(details, invitations)
        now = datetime.now(timezone.utc)
        with self._lock:
            s = self._slots
            check_lobby_game(s[(s["court_id"] == details["court_id"]) & s["slot_key"].isin(game_keys(details))]
                             .to_dict("records"), details)
            details["lobby_id"] = str(uuid.uuid4())
            self._lobbies.append((details["lobby_id"], datetime.fromisoformat(details["slot_start"]),
                                  details["duration_minutes"], details["court_id"], skill_tier, LOBBY_SPOTS,
                                  [organiser_uuid], now))
            self._log.append((now, actor, LOBBY, "open", details))
            self._log.extend((now, actor, INVITE, SENT_DEMO, {**i, "lobby_id": details["lobby_id"]}) for i in invites)
        return details

    def lobbies(self, limit: int = LOG_LIMIT) -> pd.DataFrame:
        return pd.DataFrame(list(reversed(self._lobbies))[:limit], columns=LOBBY_COLUMNS)


def make_ops_store(backend, endpoint: str, database: str, ops_schema: str):
    """(store, reason): Lakebase when the gold backend is live and an endpoint is configured, else the mock."""
    if backend.name != "live":
        return MockOpsStore(backend.as_of_date()), "mock gold data"
    if not endpoint:
        return MockOpsStore(backend.as_of_date()), "B4L_LAKEBASE_ENDPOINT is not set"
    try:
        import psycopg  # noqa: F401
    except ImportError:
        return MockOpsStore(backend.as_of_date()), "psycopg is not installed"
    return LakebaseStore(backend.w, endpoint, database, ops_schema), endpoint.split("/")[1]
