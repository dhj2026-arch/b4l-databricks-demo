"""Staff action tools: guardrails, concierge messages, doubles partners and the open waitlist.

Live, the app calls the read-only UC functions (src/b4l/agent_tools.py). Offline, the mock backend uses the copies
below; tests/test_agent_tools.py fails if they drift from src/b4l/agent_tools.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

import domain as d

WAITLIST_BACKFILL, DOUBLES_MATCH, PASS_EXPIRY, RETENTION_VOUCHER = (
    "waitlist_backfill", "doubles_match", "pass_expiry", "retention_voucher")
TRIGGERS = (WAITLIST_BACKFILL, DOUBLES_MATCH, PASS_EXPIRY, RETENTION_VOUCHER)
TRIGGER_LABELS = {
    WAITLIST_BACKFILL: "Waitlist backfill (court freed)",
    DOUBLES_MATCH: "Doubles match",
    PASS_EXPIRY: "Pass Chrono expiry reminder",
    RETENTION_VOUCHER: "Retention voucher (CHF 10 stringing)",
}
VOUCHER_CHF = 10
VOUCHER_VALID_DAYS = 30
PASS_REMINDER_DAYS = 30
OFFER_REPLY_MINUTES = 30
MAX_PARTNERS = 10
PARTNER_HOUR_WINDOW = 2
LOBBY_SPOTS = 3

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
PLAYER_PICK_COLUMNS = ["player_uuid", "skill_tier", "preferred_day_of_week", "preferred_hour", "plays_last_30d",
                       "preferred_lang"]

Row = Mapping[str, Any]


def _missing(v: Any) -> bool:
    return v is None or bool(pd.isna(v))


def guardrail(slot: Row | None, proposed_price: float | None) -> dict | None:
    if slot is None:
        return None
    base = float(slot["base_price_chf"])
    out = {"slot_key": slot["slot_key"], "tier": slot["tier"], "base_price_chf": base,
           "proposed_price_chf": proposed_price}
    if not slot["is_future"]:
        return {**out, "final_price_chf": base, "applied_rule": "past_slot",
                "reason": "The slot has started or passed; the tariff stays"}
    booked, available = slot["booked_courts_now"], slot["courts_available"]
    if not _missing(booked) and not _missing(available) and booked >= available:
        return {**out, "final_price_chf": base, "applied_rule": "fully_booked",
                "reason": "Every court is booked; booked prices are locked"}
    if proposed_price is None:
        return {**out, "final_price_chf": base, "applied_rule": "no_price", "reason": "No price proposed"}
    low, high = d.price_bounds(slot["tier"])
    price = float(round(float(proposed_price)))
    if price < low:
        return {**out, "final_price_chf": low, "applied_rule": "floor", "reason": f"Raised to the floor of CHF {low:.2f}"}
    if price > high:
        return {**out, "final_price_chf": high, "applied_rule": "ceiling",
                "reason": f"Capped at the ceiling of CHF {high:.2f}"}
    return {**out, "final_price_chf": price, "applied_rule": "within_bounds", "reason": "Within guardrails"}


def fr_date(x: date | datetime) -> str:
    return f"{FR_WEEKDAYS_SHORT[x.weekday()]} {x.day} {FR_MONTHS[x.month - 1]}"


def en_date(x: date | datetime) -> str:
    return f"{x:%a} {x.day} {x:%b}"


def concierge_message(player: Row | None, trigger_type: str, as_of: date, waitlist: Row | None = None) -> dict | None:
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
        if _missing(day) or _missing(hour):
            return {**out, "eligible": False, "message": None, "note": "No usual playing time yet"}
        day, hour = int(day), int(hour)
        msg = (f"Bonjour, envie d'un double ? Des joueurs de votre niveau ({FR_SKILL.get(skill, skill)}) jouent "
               f"souvent le {FR_WEEKDAYS[day - 1]} vers {hour}h00. Répondez OUI et nous organisons la partie." if fr
               else f"Hello, fancy a doubles game? Players at your level ({skill.lower()}) often play on "
               f"{EN_WEEKDAYS[day - 1]}s around {hour:02d}:00. Reply YES and we will set up the game.")
        return {**out, "eligible": True, "message": msg + SIGNATURE, "note": None}
    if trigger == PASS_EXPIRY:
        days, hours = player.get("days_to_pass_expiry"), player.get("pass_hours_remaining")
        if (player.get("pass_product") != d.PASS_CHRONO or _missing(days) or _missing(hours)
                or not 0 <= days <= PASS_REMINDER_DAYS or hours <= 0):
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


ORGANISER, PARTNER = "organiser", "partner"
INVITE_REPLY_HOURS = 24
INVITATION_COLUMNS = ["player_uuid", "role", "language", "message"]


OPEN_STATUSES = ("proposed", "approved", "overridden")
PLAYERS_PER_GAME = 4
MAX_LOBBY_OPTIONS = 30
DEFAULT_GAME_MINUTES = 120
GAME_MINUTES = (60, 90, 120, 180)
LOBBY_OPTION_COLUMNS = ["slot_key", "start", "end", "tier", "open_courts", "discounted_courts", "best_court_id",
                        "best_price_chf", "tariff_chf", "ai_price_chf", "booked_share", "hist_fill_rate_4w",
                        "underfilled", "usual_day", "hour_gap"]
COURT_GAME_COLUMNS = ["court_id", "hall_zone", "price_chf", "tariff_chf", "ai_price_chf", "slot_ids", "to_approve"]


def court_label(court_id: str, fr: bool) -> str:
    n = int(court_id.split("_")[1])
    return f"terrain {n}" if fr else f"Court {n}"


def duration_label(minutes: int, fr: bool) -> str:
    h, m = divmod(int(minutes), 60)
    if fr:
        return f"{h} h {m:02d}" if m else f"{h} h"
    return f"{minutes / 60:g} hour{'s' if minutes != 60 else ''}"


def chf_amount(x: float) -> str:
    return f"{x:.0f}" if x == round(x) else f"{x:.2f}"


def slot_key(t: datetime) -> str:
    return f"{t:%Y-%m-%dT%H%M}"


def price_status(price: float, tariff: float, ai: float) -> str:
    if price < tariff:
        return ("approved discount" if ai >= price else
                f"partly discounted · AI suggests CHF {chf_amount(ai)} (not approved)")
    if price > tariff:
        return "above tariff"
    if ai < price:
        return f"tariff · AI suggests CHF {chf_amount(ai)} (not approved)"
    return "tariff"


def court_game(index: Mapping[tuple[str, str], Row], court_id: str, start: datetime, minutes: int) -> dict | None:
    """The game on one court, or None unless the court is open for its whole length.

    index: live_court_slots rows by (court_id, slot_key). Every half-hour start in the game must be open; the game is
    priced as consecutive 1-hour blocks, each at the court's live price for its start (a last block of 30 min pays half).
    """
    blocks = [(start + timedelta(minutes=60 * k), min(60, minutes - 60 * k)) for k in range(-(-minutes // 60))]
    block_starts = {b for b, _ in blocks}
    for i in range(minutes // 30):
        t = start + timedelta(minutes=30 * i)
        r = index.get((court_id, slot_key(t)))
        if r is None and t in block_starts:
            return None
        if r is not None and r["status"] not in OPEN_STATUSES:
            return None
    rows = [(index[(court_id, slot_key(b))], m / 60) for b, m in blocks]
    ai = lambda r: r["recommended_price_chf"] if r["status"] == "proposed" else r["dynamic_price_chf"]  # noqa: E731
    return {
        "court_id": court_id, "hall_zone": rows[0][0].get("hall_zone"),
        "price_chf": float(sum(r["dynamic_price_chf"] * h for r, h in rows)),
        "tariff_chf": float(sum(r["base_price_chf"] * h for r, h in rows)),
        "ai_price_chf": float(sum(ai(r) * h for r, h in rows)),
        "slot_ids": [r["slot_id"] for r, _ in rows],
        "to_approve": [r["slot_id"] for r, _ in rows
                       if r["status"] == "proposed" and r["recommended_price_chf"] < r["dynamic_price_chf"]],
    }


def court_index(courts: pd.DataFrame) -> dict[tuple[str, str], dict]:
    return {(r["court_id"], r["slot_key"]): r for r in courts.to_dict("records")}


def court_games(index: Mapping[tuple[str, str], Row], start: datetime, minutes: int) -> pd.DataFrame:
    """Every court open for the whole game, cheapest first."""
    courts = sorted({c for c, _ in index})
    games = [g for g in (court_game(index, c, start, minutes) for c in courts) if g is not None]
    return pd.DataFrame(games, columns=COURT_GAME_COLUMNS).sort_values(
        ["price_chf", "court_id"]).reset_index(drop=True)


def lobby_slot_options(courts: pd.DataFrame, slot_feats: pd.DataFrame, day_of_week: int | None, hour: int,
                       minutes: int = DEFAULT_GAME_MINUTES) -> pd.DataFrame:
    """Upcoming starts within PARTNER_HOUR_WINDOW h of the organiser's usual hour with a court open for the whole game.

    courts: live_court_slots rows (ops store); slot_feats: slot_features with booked_share and underfilled
    (data.underfilled_slots). Under-filled starts first, then the organiser's usual day, the closest hour, the price.
    """
    index = court_index(courts)
    keys = sorted(courts.loc[courts["status"].isin(OPEN_STATUSES), "slot_key"].unique())
    feats = slot_feats.set_index("slot_key") if not slot_feats.empty else pd.DataFrame()
    rows = []
    for k in keys:
        start = datetime.strptime(k, "%Y-%m-%dT%H%M")
        gap = abs(start.hour + start.minute / 60 - hour)
        if gap > PARTNER_HOUR_WINDOW:
            continue
        games = court_games(index, start, minutes)
        if games.empty:
            continue
        best = games.iloc[0]
        f = feats.loc[k] if k in feats.index else {}
        rows.append((k, start, start + timedelta(minutes=minutes), index[(best["court_id"], k)]["tier"], len(games),
                     int((games["price_chf"] < games["tariff_chf"]).sum()), best["court_id"], best["price_chf"],
                     best["tariff_chf"], best["ai_price_chf"], f.get("booked_share"), f.get("hist_fill_rate_4w"),
                     bool(f.get("underfilled", False)),
                     not _missing(day_of_week) and start.weekday() + 1 == int(day_of_week), gap))
    out = pd.DataFrame(rows, columns=LOBBY_OPTION_COLUMNS)
    out = out.sort_values(["underfilled", "usual_day", "hour_gap", "best_price_chf", "start"],
                          ascending=[False, False, True, True, True])
    return out.head(MAX_LOBBY_OPTIONS).reset_index(drop=True)


def _game_text(fr: bool, start: datetime, minutes: int, court_id: str, price: float, tariff: float) -> str:
    end = start + timedelta(minutes=minutes)
    each = f"{price / PLAYERS_PER_GAME:.2f}"
    if fr:
        return (f"{fr_date(start)} de {start:%H:%M} à {end:%H:%M} sur le {court_label(court_id, True)}, à "
                f"CHF {chf_amount(price)} pour {duration_label(minutes, True)}"
                + (f" au lieu de CHF {chf_amount(tariff)}" if price < tariff else "") + f" (CHF {each} par joueur)")
    return (f"{en_date(start)} from {start:%H:%M} to {end:%H:%M} on {court_label(court_id, False)}, at "
            f"CHF {chf_amount(price)} for {duration_label(minutes, False)}"
            + (f" instead of CHF {chf_amount(tariff)}" if price < tariff else "") + f" (CHF {each} each)")


def lobby_invitation(lang: str | None, role: str, skill_tier: str, start: datetime, minutes: int, n_invited: int,
                     court_id: str, price: float, tariff: float) -> str:
    """Invitation for one lobby member; built in the app (no UC function), staff review it before logging."""
    fr = lang != "EN"
    level = FR_SKILL.get(skill_tier, skill_tier) if fr else skill_tier.lower()
    game = _game_text(fr, start, minutes, court_id, price, tariff)
    if role == ORGANISER:
        msg = (f"Bonjour, votre double est ouvert : {game}, niveau {level}. Nous avons invité {n_invited} "
               f"partenaire{'s' if n_invited > 1 else ''} et vous confirmons dès que la partie est complète." if fr else
               f"Hello, your doubles game is open: {game}, {level} level. We have invited {n_invited} "
               f"partner{'s' if n_invited > 1 else ''} and will confirm once the game is full.")
    else:
        msg = (f"Bonjour, une partie de double de votre niveau ({level}) est prévue le {game}. Il reste des places : "
               f"répondez OUI dans les {INVITE_REPLY_HOURS} heures pour réserver la vôtre." if fr else
               f"Hello, a doubles game at your level ({level}) is planned on {game}. Spots are open: reply YES "
               f"within {INVITE_REPLY_HOURS} hours to take yours.")
    return msg + SIGNATURE


def lobby_invitations(organiser: Row, partners: pd.DataFrame, invited: list[str], start: datetime, minutes: int,
                      court_id: str, price: float, tariff: float) -> pd.DataFrame:
    """One draft per person: the organiser first, then the invited partners in the order picked."""
    lang = partners.set_index("partner_uuid")["preferred_lang"].to_dict()
    people = [(organiser["player_uuid"], ORGANISER, organiser.get("preferred_lang"))] + [
        (u, PARTNER, lang.get(u)) for u in invited]
    rows = [(u, role, "en" if lg == "EN" else "fr",
             lobby_invitation(lg, role, organiser["skill_tier"], start, minutes, len(invited), court_id, price, tariff))
            for u, role, lg in people]
    return pd.DataFrame(rows, columns=INVITATION_COLUMNS)


def mock_open_waitlist(players: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    """Deterministic waitlist for offline runs: one request per waitlist count on upcoming slots."""
    rows, pool, i = [], players.sort_values("player_uuid").reset_index(drop=True), 0
    for s in slots[slots["waitlist_requests"].fillna(0) > 0].itertuples(index=False):
        for _ in range(int(s.waitlist_requests)):
            p = pool.iloc[i % len(pool)]
            start = pd.Timestamp(s.slot_start_ts)
            rows.append((f"WL-MOCK-{i:04d}", p["player_uuid"], s.slot_date, start, start, 2 + 2 * (i % 2),
                         start - pd.Timedelta(hours=20), "phone", p["preferred_lang"], p["skill_tier"]))
            i += 1
    return pd.DataFrame(rows, columns=WAITLIST_COLUMNS)
