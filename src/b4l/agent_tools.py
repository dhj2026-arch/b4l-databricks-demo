"""Read-only UC functions for staff actions and agents (SPEC.md 5.D), with a Python twin of each for the checks.

- `apply_dynamic_price_with_guardrails(slot_id, proposed_price)`: final price, rule and reason (SPEC.md 3.8).
- `trigger_player_concierge(player_uuid, trigger_type)`: a French or English message for one player.
- `find_doubles_partners(player_uuid)`: up to 10 players at the same level with similar habits.
- `get_open_waitlist(start_date, n_days)`: open waitlist requests, so staff can pick whom to offer a freed court.

The functions only compute. The staff app writes (Lakebase action log, lobbies) after a person approves.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from b4l import pricing_rules as pr
from b4l.day_context import _ident, _sql_str

GUARDRAIL_FN = "apply_dynamic_price_with_guardrails"
CONCIERGE_FN = "trigger_player_concierge"
PARTNERS_FN = "find_doubles_partners"
WAITLIST_FN = "get_open_waitlist"
FUNCTIONS = (GUARDRAIL_FN, CONCIERGE_FN, PARTNERS_FN, WAITLIST_FN)

WAITLIST_BACKFILL, DOUBLES_MATCH, PASS_EXPIRY, RETENTION_VOUCHER = (
    "waitlist_backfill", "doubles_match", "pass_expiry", "retention_voucher")
TRIGGERS = (WAITLIST_BACKFILL, DOUBLES_MATCH, PASS_EXPIRY, RETENTION_VOUCHER)
VOUCHER_CHF = 10
VOUCHER_VALID_DAYS = 30
PASS_REMINDER_DAYS = 30
OFFER_REPLY_MINUTES = 30
MAX_PARTNERS = 10
PARTNER_HOUR_WINDOW = 2

FR_WEEKDAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
FR_WEEKDAYS_SHORT = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
FR_MONTHS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
EN_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FR_SKILL = {"Beginner": "débutant", "Intermediate": "intermédiaire", "Advanced": "avancé"}
SIGNATURE = " - B4L Fondation"

GUARDRAIL_COLUMNS = ["slot_key", "tier", "base_price_chf", "proposed_price_chf", "final_price_chf", "applied_rule",
                     "reason"]
CONCIERGE_COLUMNS = ["player_uuid", "trigger_type", "language", "eligible", "message", "note"]
PARTNER_COLUMNS = ["partner_uuid", "skill_tier", "preferred_day_of_week", "preferred_hour", "plays_last_30d",
                   "played_together", "match_score", "preferred_lang"]
WAITLIST_COLUMNS = ["waitlist_id", "player_uuid", "requested_date", "window_start_ts", "window_end_ts", "party_size",
                    "created_at", "channel", "preferred_lang", "skill_tier"]

Row = Mapping[str, Any]


# ---------------------------------------------------------------- Python twins

def guardrail(slot: Row | None, proposed_price: float | None) -> dict | None:
    """slot: slot_features row (slot_key, tier, base_price_chf, is_future, booked_courts_now, courts_available)."""
    if slot is None:
        return None
    base = float(slot["base_price_chf"])
    out = {"slot_key": slot["slot_key"], "tier": slot["tier"], "base_price_chf": base,
           "proposed_price_chf": proposed_price}
    if not slot["is_future"]:
        return {**out, "final_price_chf": base, "applied_rule": "past_slot",
                "reason": "The slot has started or passed; the tariff stays"}
    booked, available = slot["booked_courts_now"], slot["courts_available"]
    if booked is not None and available is not None and not pd.isna(booked) and booked >= available:
        return {**out, "final_price_chf": base, "applied_rule": "fully_booked",
                "reason": "Every court is booked; booked prices are locked"}
    if proposed_price is None:
        return {**out, "final_price_chf": base, "applied_rule": "no_price", "reason": "No price proposed"}
    r = pr.apply_guardrails(slot["tier"], float(proposed_price))
    return {**out, "final_price_chf": r.final_price_chf, "applied_rule": r.applied_rule, "reason": r.reason}


def fr_date(d: date | datetime) -> str:
    return f"{FR_WEEKDAYS_SHORT[d.weekday()]} {d.day} {FR_MONTHS[d.month - 1]}"


def en_date(d: date | datetime) -> str:
    return f"{d:%a} {d.day} {d:%b}"


def concierge_message(player: Row | None, trigger_type: str, as_of: date, waitlist: Row | None = None) -> dict | None:
    """player: player_features row; waitlist: the player's earliest open request from as_of (or None)."""
    if player is None:
        return None
    trigger = (trigger_type or "").strip().lower()
    lang = "en" if player.get("preferred_lang") == "EN" else "fr"
    out = {"player_uuid": player["player_uuid"], "trigger_type": trigger, "language": lang}
    fr = lang == "fr"
    if trigger == WAITLIST_BACKFILL:
        if waitlist is None:
            return {**out, "eligible": False, "message": None, "note": "No open waitlist request"}
        s, e = waitlist["window_start_ts"], waitlist["window_end_ts"]
        when_fr = f"à {s:%H:%M}" if s == e else f"avec un début entre {s:%H:%M} et {e:%H:%M}"
        when_en = f"at {s:%H:%M}" if s == e else f"starting between {s:%H:%M} and {e:%H:%M}"
        msg = (f"Bonjour, bonne nouvelle : un terrain s'est libéré le {fr_date(s)} {when_fr}, comme demandé sur la "
               f"liste d'attente. Répondez OUI dans les {OFFER_REPLY_MINUTES} minutes pour le réserver." if fr else
               f"Hello, good news: a court is free on {en_date(s)} {when_en}, matching your waitlist request. "
               f"Reply YES within {OFFER_REPLY_MINUTES} minutes to book it.")
        return {**out, "eligible": True, "message": msg + SIGNATURE, "note": None}
    if trigger == DOUBLES_MATCH:
        day, hour, skill = player.get("preferred_day_of_week"), player.get("preferred_hour"), player["skill_tier"]
        if day is None or hour is None or pd.isna(day) or pd.isna(hour):
            return {**out, "eligible": False, "message": None, "note": "No usual playing time yet"}
        day, hour = int(day), int(hour)
        msg = (f"Bonjour, envie d'un double ? Des joueurs de votre niveau ({FR_SKILL.get(skill, skill)}) jouent "
               f"souvent le {FR_WEEKDAYS[day - 1]} vers {hour}h00. Répondez OUI et nous organisons la partie." if fr
               else f"Hello, fancy a doubles game? Players at your level ({skill.lower()}) often play on "
               f"{EN_WEEKDAYS[day - 1]}s around {hour:02d}:00. Reply YES and we will set up the game.")
        return {**out, "eligible": True, "message": msg + SIGNATURE, "note": None}
    if trigger == PASS_EXPIRY:
        days, hours = player.get("days_to_pass_expiry"), player.get("pass_hours_remaining")
        if (player.get("pass_product") != pr.PASS_CHRONO or days is None or hours is None or pd.isna(days)
                or pd.isna(hours) or not 0 <= days <= PASS_REMINDER_DAYS or hours <= 0):
            return {**out, "eligible": False, "message": None,
                    "note": f"No Pass Chrono with hours left expiring within {PASS_REMINDER_DAYS} days"}
        exp = pd.Timestamp(player["pass_expires_at"])
        msg = (f"Bonjour, votre Pass Chrono a encore {int(hours)} heures et expire le {fr_date(exp)} {exp.year}. "
               "Réservez un terrain d'ici là pour ne pas les perdre." if fr else
               f"Hello, your Pass Chrono has {int(hours)} hours left and expires on {en_date(exp)} {exp.year}. "
               "Book a court before then so you don't lose them.")
        return {**out, "eligible": True, "message": msg + SIGNATURE, "note": None}
    if trigger == RETENTION_VOUCHER:
        if not player.get("churn_risk_rule"):
            return {**out, "eligible": False, "message": None, "note": "Not flagged as an at-risk regular"}
        until = as_of + timedelta(days=VOUCHER_VALID_DAYS)
        msg = (f"Bonjour, vous nous manquez sur les terrains ! Voici un bon de CHF {VOUCHER_CHF} pour le cordage de "
               f"votre raquette chez B4L, valable jusqu'au {fr_date(until)} {until.year}. Présentez ce message à "
               "l'accueil." if fr else
               f"Hello, we miss you on court! Here is a CHF {VOUCHER_CHF} voucher for racket stringing at B4L, valid "
               f"until {en_date(until)} {until.year}. Show this message at reception.")
        return {**out, "eligible": True, "message": msg + SIGNATURE, "note": None}
    return {**out, "eligible": False, "message": None, "note": f"Unknown trigger; use one of {', '.join(TRIGGERS)}"}


def doubles_partners(players: pd.DataFrame, player_uuid: str) -> pd.DataFrame:
    """players: player_features. Same skill tier, active in the last 30 days, preferred hour within 2 h."""
    me = players[players["player_uuid"] == player_uuid]
    if me.empty:
        return pd.DataFrame(columns=PARTNER_COLUMNS)
    t = me.iloc[0]
    c = players[(players["skill_tier"] == t["skill_tier"]) & (players["player_uuid"] != player_uuid)
                & ((players["preferred_hour"] - t["preferred_hour"]).abs() <= PARTNER_HOUR_WINDOW)
                & (players["plays_last_30d"] > 0)].copy()
    partners = set(t["top_partner_uuids"]) if t["top_partner_uuids"] is not None else set()
    c["played_together"] = c["player_uuid"].isin(partners)
    c["match_score"] = (2 * (c["preferred_day_of_week"] == t["preferred_day_of_week"]).astype(int)
                        + PARTNER_HOUR_WINDOW - (c["preferred_hour"] - t["preferred_hour"]).abs()
                        + c["played_together"].astype(int))
    c = c.sort_values(["match_score", "plays_last_30d", "player_uuid"], ascending=[False, False, True])
    return c.rename(columns={"player_uuid": "partner_uuid"})[PARTNER_COLUMNS].head(MAX_PARTNERS).reset_index(drop=True)


# ---------------------------------------------------------------- SQL

def _schemas(catalog: str, schema_prefix: str) -> tuple[str, str]:
    _ident(catalog, schema_prefix)
    return f"`{catalog}`.`{schema_prefix}_gold`", f"`{catalog}`.`{schema_prefix}_silver`"


def function_name(catalog: str, schema_prefix: str, fn: str) -> str:
    if fn not in FUNCTIONS:
        raise ValueError(f"Unknown function: {fn}")
    return f"{_schemas(catalog, schema_prefix)[0]}.{fn}"


def _arr(values: list[str]) -> str:
    return "array(" + ", ".join(_sql_str(v) for v in values) + ")"


def _fr_date_sql(expr: str) -> str:
    return (f"concat(element_at({_arr(FR_WEEKDAYS_SHORT)}, weekday({expr}) + 1), ' ', day({expr}), ' ', "
            f"element_at({_arr(FR_MONTHS)}, month({expr})))")


def _en_date_sql(expr: str) -> str:
    return f"concat(date_format({expr}, 'EEE'), ' ', day({expr}), ' ', date_format({expr}, 'MMM'))"


def guardrail_sql(catalog: str, schema_prefix: str) -> str:
    gold, _ = _schemas(catalog, schema_prefix)
    f, step = GUARDRAIL_FN, pr.PRICE_STEP_CHF
    lo = (f"ceil(greatest({pr.PRICE_FLOOR_CHF!r}D, base_price_chf * {1 - pr.MAX_MOVE_PCT!r}D) / {step!r}D) "
          f"* {step!r}D")
    hi = (f"floor(least({pr.PRICE_CEILING_CHF!r}D, base_price_chf * {1 + pr.MAX_MOVE_PCT!r}D) / {step!r}D) "
          f"* {step!r}D")
    return f"""CREATE OR REPLACE FUNCTION {gold}.{f}(
  slot_id STRING COMMENT 'Slot key (2026-11-24T1200) or court slot id (2026-11-24T1200_Court_02)',
  proposed_price DOUBLE COMMENT 'Proposed price in CHF per court per hour'
)
RETURNS TABLE (
  slot_key STRING COMMENT 'Slot start',
  tier STRING COMMENT 'Tariff tier',
  base_price_chf DOUBLE COMMENT 'Published tariff',
  proposed_price_chf DOUBLE COMMENT 'Price asked for',
  final_price_chf DOUBLE COMMENT 'Price allowed by the guardrails',
  applied_rule STRING COMMENT 'past_slot, fully_booked, no_price, floor, ceiling or within_bounds',
  reason STRING COMMENT 'Explanation for staff'
)
COMMENT 'B4L pricing guardrails (read-only): CHF {pr.PRICE_FLOOR_CHF:.0f}-{pr.PRICE_CEILING_CHF:.0f}, at most +/-{pr.MAX_MOVE_PCT:.0%} from the tier, whole francs; past and fully booked slots keep the tariff. Garde-fous de prix.'
RETURN
WITH s AS (
  SELECT slot_key, tier, base_price_chf, is_future, booked_courts_now >= courts_available AS fully_booked,
         {lo} AS lo, {hi} AS hi,
         bround({f}.proposed_price / {step!r}D) * {step!r}D AS p
  FROM {gold}.slot_features
  WHERE slot_key = substr({f}.slot_id, 1, 15)
)
SELECT slot_key, tier, base_price_chf, {f}.proposed_price AS proposed_price_chf,
  CASE WHEN NOT is_future OR fully_booked OR p IS NULL THEN base_price_chf
       WHEN p < lo THEN lo WHEN p > hi THEN hi ELSE p END AS final_price_chf,
  CASE WHEN NOT is_future THEN 'past_slot' WHEN fully_booked THEN 'fully_booked' WHEN p IS NULL THEN 'no_price'
       WHEN p < lo THEN 'floor' WHEN p > hi THEN 'ceiling' ELSE 'within_bounds' END AS applied_rule,
  CASE WHEN NOT is_future THEN 'The slot has started or passed; the tariff stays'
       WHEN fully_booked THEN 'Every court is booked; booked prices are locked'
       WHEN p IS NULL THEN 'No price proposed'
       WHEN p < lo THEN format_string('Raised to the floor of CHF %.2f', lo)
       WHEN p > hi THEN format_string('Capped at the ceiling of CHF %.2f', hi)
       ELSE 'Within guardrails' END AS reason
FROM s"""


def concierge_sql(catalog: str, schema_prefix: str) -> str:
    gold, silver = _schemas(catalog, schema_prefix)
    f = CONCIERGE_FN
    sig = _sql_str(SIGNATURE)
    t = f"lower(trim({f}.trigger_type))"
    fr_skill = "CASE p.skill_tier " + " ".join(f"WHEN {_sql_str(k)} THEN {_sql_str(v)}" for k, v in FR_SKILL.items()) \
        + " ELSE p.skill_tier END"
    pass_ok = (f"p.pass_product = {_sql_str(pr.PASS_CHRONO)} AND p.days_to_pass_expiry BETWEEN 0 AND "
               f"{PASS_REMINDER_DAYS} AND p.pass_hours_remaining > 0")
    until = f"date_add(a.d, {VOUCHER_VALID_DAYS})"
    s = lambda v: _sql_str(v)  # noqa: E731
    messages = {
        (WAITLIST_BACKFILL, "fr"): f"""concat({s("Bonjour, bonne nouvelle : un terrain s'est libéré le ")}, {_fr_date_sql('w.window_start_ts')},
            CASE WHEN w.window_start_ts = w.window_end_ts THEN concat({s(' à ')}, date_format(w.window_start_ts, 'HH:mm'))
                 ELSE concat({s(' avec un début entre ')}, date_format(w.window_start_ts, 'HH:mm'), ' et ',
                             date_format(w.window_end_ts, 'HH:mm')) END,
            {s(f", comme demandé sur la liste d'attente. Répondez OUI dans les {OFFER_REPLY_MINUTES} minutes pour le réserver.")}, {sig})""",
        (WAITLIST_BACKFILL, "en"): f"""concat('Hello, good news: a court is free on ', {_en_date_sql('w.window_start_ts')},
            CASE WHEN w.window_start_ts = w.window_end_ts THEN concat(' at ', date_format(w.window_start_ts, 'HH:mm'))
                 ELSE concat(' starting between ', date_format(w.window_start_ts, 'HH:mm'), ' and ',
                             date_format(w.window_end_ts, 'HH:mm')) END,
            {s(f", matching your waitlist request. Reply YES within {OFFER_REPLY_MINUTES} minutes to book it.")}, {sig})""",
        (DOUBLES_MATCH, "fr"): f"""concat({s("Bonjour, envie d'un double ? Des joueurs de votre niveau (")}, {fr_skill},
            ') jouent souvent le ', element_at({_arr(FR_WEEKDAYS)}, p.preferred_day_of_week), ' vers ',
            p.preferred_hour, {s('h00. Répondez OUI et nous organisons la partie.')}, {sig})""",
        (DOUBLES_MATCH, "en"): f"""concat('Hello, fancy a doubles game? Players at your level (', lower(p.skill_tier),
            ') often play on ', element_at({_arr(EN_WEEKDAYS)}, p.preferred_day_of_week), 's around ',
            lpad(cast(p.preferred_hour AS STRING), 2, '0'), ':00. Reply YES and we will set up the game.', {sig})""",
        (PASS_EXPIRY, "fr"): f"""concat('Bonjour, votre Pass Chrono a encore ', p.pass_hours_remaining,
            ' heures et expire le ', {_fr_date_sql('p.pass_expires_at')}, ' ', year(p.pass_expires_at),
            {s(". Réservez un terrain d'ici là pour ne pas les perdre.")}, {sig})""",
        (PASS_EXPIRY, "en"): f"""concat('Hello, your Pass Chrono has ', p.pass_hours_remaining,
            ' hours left and expires on ', {_en_date_sql('p.pass_expires_at')}, ' ', year(p.pass_expires_at),
            {s(". Book a court before then so you don't lose them.")}, {sig})""",
        (RETENTION_VOUCHER, "fr"): f"""concat({s(f"Bonjour, vous nous manquez sur les terrains ! Voici un bon de CHF {VOUCHER_CHF} pour le cordage de votre raquette chez B4L, valable jusqu'au ")},
            {_fr_date_sql(until)}, ' ', year({until}), {s(". Présentez ce message à l'accueil.")}, {sig})""",
        (RETENTION_VOUCHER, "en"): f"""concat({s(f"Hello, we miss you on court! Here is a CHF {VOUCHER_CHF} voucher for racket stringing at B4L, valid until ")},
            {_en_date_sql(until)}, ' ', year({until}), '. Show this message at reception.', {sig})""",
    }
    eligible = {
        WAITLIST_BACKFILL: ("w.window_start_ts IS NOT NULL", "No open waitlist request"),
        DOUBLES_MATCH: ("p.preferred_day_of_week IS NOT NULL AND p.preferred_hour IS NOT NULL",
                        "No usual playing time yet"),
        PASS_EXPIRY: (pass_ok, f"No Pass Chrono with hours left expiring within {PASS_REMINDER_DAYS} days"),
        RETENTION_VOUCHER: ("coalesce(p.churn_risk_rule, FALSE)", "Not flagged as an at-risk regular"),
    }
    elig_case = "CASE " + " ".join(f"WHEN {t} = {s(k)} THEN coalesce({c}, FALSE)" for k, (c, _) in eligible.items()) \
        + " ELSE FALSE END"
    note_case = "CASE " + " ".join(f"WHEN {t} = {s(k)} THEN " + (f"CASE WHEN coalesce({c}, FALSE) THEN NULL ELSE "
                                                               f"{s(n)} END" if n else "NULL")
                                   for k, (c, n) in eligible.items()) \
        + f" ELSE {s('Unknown trigger; use one of ' + ', '.join(TRIGGERS))} END"
    msg_case = "CASE " + " ".join(
        f"WHEN {t} = {s(k)} AND coalesce({eligible[k][0]}, FALSE) THEN CASE WHEN p.preferred_lang = 'EN' THEN "
        f"{messages[(k, 'en')]} ELSE {messages[(k, 'fr')]} END" for k in TRIGGERS) + " END"
    return f"""CREATE OR REPLACE FUNCTION {gold}.{f}(
  player_uuid STRING COMMENT 'Pseudonymised player id',
  trigger_type STRING COMMENT '{", ".join(TRIGGERS)}'
)
RETURNS TABLE (
  player_uuid STRING COMMENT 'Pseudonymised player id',
  trigger_type STRING COMMENT 'Trigger',
  language STRING COMMENT 'fr or en (player preference)',
  eligible BOOLEAN COMMENT 'Whether the trigger applies to this player',
  message STRING COMMENT 'Draft message for staff to approve; NULL when not eligible',
  note STRING COMMENT 'Why the trigger does not apply'
)
COMMENT 'B4L player concierge (read-only): drafts a French or English message for a waitlist backfill, a doubles match, a Pass Chrono expiry reminder or a CHF {VOUCHER_CHF} stringing voucher. Staff approve and the app logs it.'
RETURN
WITH a AS (SELECT max(as_of_date) AS d FROM {gold}.player_features),
w AS (
  SELECT wl.window_start_ts, wl.window_end_ts
  FROM {silver}.waitlist wl CROSS JOIN a
  WHERE wl.player_uuid = {f}.player_uuid AND wl.status = 'open' AND wl.requested_date >= a.d
  ORDER BY wl.window_start_ts, wl.waitlist_id
  LIMIT 1
)
SELECT p.player_uuid, {t} AS trigger_type, CASE WHEN p.preferred_lang = 'EN' THEN 'en' ELSE 'fr' END AS language,
  {elig_case} AS eligible,
  {msg_case} AS message,
  {note_case} AS note
FROM {gold}.player_features p CROSS JOIN a LEFT JOIN w ON TRUE
WHERE p.player_uuid = {f}.player_uuid"""


def partners_sql(catalog: str, schema_prefix: str) -> str:
    gold, _ = _schemas(catalog, schema_prefix)
    f, win = PARTNERS_FN, PARTNER_HOUR_WINDOW
    together = "array_contains(coalesce(t.top_partner_uuids, array()), c.player_uuid)"
    return f"""CREATE OR REPLACE FUNCTION {gold}.{f}(
  player_uuid STRING COMMENT 'Pseudonymised player id looking for a doubles game'
)
RETURNS TABLE (
  partner_uuid STRING COMMENT 'Suggested partner (pseudonymised)',
  skill_tier STRING COMMENT 'Same level as the player',
  preferred_day_of_week INT COMMENT '1=Monday .. 7=Sunday',
  preferred_hour INT COMMENT 'Usual start hour',
  plays_last_30d INT COMMENT 'Plays in the last 30 days',
  played_together BOOLEAN COMMENT 'Among the usual partners of the player',
  match_score INT COMMENT '2 same day + (2 - hour gap) + 1 played together',
  preferred_lang STRING COMMENT 'FR or EN'
)
COMMENT 'B4L doubles matchmaking (read-only): up to {MAX_PARTNERS} active players at the same level whose usual start hour is within {win} h. Partenaires de double.'
RETURN
SELECT c.player_uuid AS partner_uuid, c.skill_tier, c.preferred_day_of_week, c.preferred_hour, c.plays_last_30d,
  {together} AS played_together,
  CAST(CASE WHEN c.preferred_day_of_week = t.preferred_day_of_week THEN 2 ELSE 0 END
       + {win} - abs(c.preferred_hour - t.preferred_hour)
       + CASE WHEN {together} THEN 1 ELSE 0 END AS INT) AS match_score,
  c.preferred_lang
FROM {gold}.player_features t
JOIN {gold}.player_features c
  ON c.skill_tier = t.skill_tier AND c.player_uuid <> t.player_uuid
  AND abs(c.preferred_hour - t.preferred_hour) <= {win} AND c.plays_last_30d > 0
WHERE t.player_uuid = {f}.player_uuid
ORDER BY match_score DESC, c.plays_last_30d DESC, c.player_uuid
LIMIT {MAX_PARTNERS}"""


def waitlist_sql(catalog: str, schema_prefix: str) -> str:
    gold, silver = _schemas(catalog, schema_prefix)
    f = WAITLIST_FN
    return f"""CREATE OR REPLACE FUNCTION {gold}.{f}(
  start_date DATE COMMENT 'First day',
  n_days INT DEFAULT 7 COMMENT 'Number of days (1-31)'
)
RETURNS TABLE (
  waitlist_id STRING COMMENT 'Waitlist request',
  player_uuid STRING COMMENT 'Pseudonymised player id',
  requested_date DATE COMMENT 'Day asked for',
  window_start_ts TIMESTAMP COMMENT 'Earliest start the player accepts',
  window_end_ts TIMESTAMP COMMENT 'Latest end the player accepts',
  party_size INT COMMENT 'Players',
  created_at TIMESTAMP COMMENT 'When the request was made',
  channel STRING COMMENT 'phone or reception',
  preferred_lang STRING COMMENT 'FR or EN',
  skill_tier STRING COMMENT 'Level'
)
COMMENT 'B4L open waitlist requests (read-only), for offering a freed court. Liste d attente.'
RETURN
SELECT wl.waitlist_id, wl.player_uuid, wl.requested_date, wl.window_start_ts, wl.window_end_ts, wl.party_size,
  wl.created_at, wl.channel, p.preferred_lang, p.skill_tier
FROM {silver}.waitlist wl
LEFT JOIN {gold}.player_features p ON p.player_uuid = wl.player_uuid
WHERE wl.status = 'open' AND wl.requested_date >= {f}.start_date
  AND wl.requested_date < date_add({f}.start_date, least(greatest(coalesce({f}.n_days, 7), 1), 31))
ORDER BY wl.window_start_ts, wl.waitlist_id"""


def all_function_sql(catalog: str, schema_prefix: str) -> dict[str, str]:
    return {GUARDRAIL_FN: guardrail_sql(catalog, schema_prefix), CONCIERGE_FN: concierge_sql(catalog, schema_prefix),
            PARTNERS_FN: partners_sql(catalog, schema_prefix), WAITLIST_FN: waitlist_sql(catalog, schema_prefix)}


def grant_statements(catalog: str, schema_prefix: str, principal: str) -> list[str]:
    import re

    if not re.fullmatch(r"[A-Za-z0-9-]+", principal):
        raise ValueError(f"Invalid principal: {principal!r}")
    return [f"GRANT EXECUTE ON FUNCTION {function_name(catalog, schema_prefix, fn)} TO `{principal}`"
            for fn in FUNCTIONS]

