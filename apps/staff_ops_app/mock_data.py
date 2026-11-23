"""Deterministic synthetic gold frames for offline runs (`B4L_USE_MOCK=true` or no Databricks auth).

Column names and order match src/b4l/schemas.py GOLD exactly (checked by tests/test_staff_app_contract.py).
Numbers are plausible but simpler than the real generator; every frame depends only on the as-of date.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, time, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

import domain as d

GOLD_COLUMNS: dict[str, list[str]] = {
    "court_utilization_hourly": [
        "date", "hour_start_ts", "hour", "court_id", "hall_zone", "is_show_court", "day_type", "day_name",
        "tier", "available_minutes", "booked_minutes", "blocked_minutes", "occupancy_rate", "revenue_chf",
        "bookings_started", "check_in_persons", "is_rainy", "is_school_holiday", "is_public_holiday",
    ],
    "daily_kpis": [
        "date", "day_name", "day_type", "revenue_chf", "block_revenue_chf", "occupancy_rate", "bookings",
        "cancellations_early", "cancellations_late", "no_shows", "waitlist_requests", "waitlist_resolved",
        "check_in_persons", "unique_players", "is_rainy", "is_school_holiday",
    ],
    "player_features": [
        "player_uuid", "segment", "skill_tier", "preferred_lang", "age_band", "tenure_days", "plays_last_30d",
        "plays_prev_30d", "plays_last_90d", "avg_monthly_plays_baseline", "play_change_pct", "days_since_last_play", "preferred_day_of_week",
        "preferred_hour", "preferred_tier", "avg_lead_time_h", "late_cancel_rate", "no_show_rate",
        "has_active_pass", "pass_product", "pass_hours_remaining", "pass_balance_chf", "pass_expires_at",
        "days_to_pass_expiry", "top_partner_uuids", "churn_risk_rule", "as_of_date",
    ],
    "slot_features": [
        "slot_key", "slot_date", "slot_start_ts", "start_hour", "day_of_week", "day_type", "tier",
        "base_price_chf", "is_future", "is_school_holiday", "is_public_holiday", "arena_event", "is_rainy",
        "temp_mean_c", "courts_available", "booked_courts_at_d3", "booked_courts_now", "final_booked_courts",
        "fill_rate", "avg_price_shown_chf", "hist_fill_rate_4w", "hist_fill_rate_8w", "waitlist_requests",
    ],
    "sponsor_metrics": [
        "month", "metric_group", "metric_key", "unique_players", "player_visits", "check_in_persons",
        "court_hours", "suppressed",
    ],
}

N_PLAYERS = 3000
BADMATIN_COURTS = d.COURT_IDS[:4]
SCHOOL_COURTS = d.COURT_IDS[8:10]
CLUB_COURTS = d.COURT_IDS[12:]
LESSON_COURT = d.COURT_IDS[15]
AGE_BANDS = ("<18", "18-25", "26-35", "36-50", "51-65", "65+")
CHURN_MIN_MONTHLY_PLAYS = 2.0
CHURN_DROP_RATIO = 0.35


def _rng(*key: object) -> np.random.Generator:
    digest = hashlib.sha256("|".join(map(str, key)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def is_rainy(day: date) -> bool:
    return bool(_rng("rain", day).random() < 0.35)


def _base_occupancy(day: date, hour: int, tier: str) -> float:
    dtype = d.day_type(day)
    school = d.is_school_holiday(day)
    if dtype == "weekend_holiday":
        p = 0.80 + (0.12 if is_rainy(day) else 0.0)
    elif tier == d.LUNCH:
        p = 0.68 - (0.10 if school else 0.0)
    elif tier == d.EVENING:
        p = 0.90 - (0.10 if school else 0.0)
    elif tier == d.WEEKEND:
        p = 0.72
    elif hour < 11:
        p = 0.38 + (0.06 if school else 0.0)
    else:
        p = 0.50
    if day.month in (7, 8):
        p -= 0.12
    return float(np.clip(p, 0.05, 0.98))


def _booked_share_so_far(day: date, as_of: date) -> float:
    """Future days only hold the bookings made so far."""
    ahead = (day - as_of).days
    return 1.0 if ahead < 0 else max(0.2, 0.95 - 0.06 * ahead)


def _blocked(day: date, court: str, hour: int) -> int:
    """Mirrors the generator's block schedule (src/b4l/synth.py build_blocks); lessons are simplified."""
    wd = day.weekday()
    if d.day_type(day) != "weekday":
        return 0
    term = not d.is_school_holiday(day)
    if wd == 1 and court in BADMATIN_COURTS and term:
        return {9: 30, 10: 60}.get(hour, 0)
    if wd in (1, 3) and court in SCHOOL_COURTS and term:
        return {13: 30, 14: 60, 15: 30}.get(hour, 0)
    if court in CLUB_COURTS and ((wd == 0 and term and hour in (18, 19)) or (wd == 3 and hour in (20, 21))):
        return 60
    if wd in (0, 2, 4) and court == LESSON_COURT and hour in (17, 18):
        return 60
    return 0


def _hour_tier(day: date, hour: int) -> str:
    dtype = d.day_type(day)
    first = d.rentable_window(dtype)[0]
    start = max(time(hour, 0), first)
    return d.tier_for(start, day.weekday(), dtype)


def _is_demo_lunch(day: date, hour: int, as_of: date) -> bool:
    """Seeded moment: the first upcoming Tuesday lunch is under-filled."""
    first_tuesday = as_of + timedelta(days=(1 - as_of.weekday()) % 7 or 7)
    return day == first_tuesday and hour in (11, 12, 13)


@lru_cache(maxsize=512)
def _court_day(day: date, as_of: date) -> pd.DataFrame:
    dtype = d.day_type(day)
    cols = GOLD_COLUMNS["court_utilization_hourly"]
    if dtype == "closed":
        return pd.DataFrame(columns=cols)
    rng = _rng("court_day", day)
    rainy = is_rainy(day)
    school = d.is_school_holiday(day)
    public = day in d.vaud_public_holidays(day.year)
    share = _booked_share_so_far(day, as_of)
    rows = []
    for court in d.COURT_IDS:
        for hour in d.heatmap_hours(dtype):
            avail = d.available_minutes(dtype, hour)
            tier = _hour_tier(day, hour)
            blocked = min(_blocked(day, court, hour), avail)
            p = _base_occupancy(day, hour, tier) * share
            if _is_demo_lunch(day, hour, as_of):
                p *= 0.55
            u = rng.random()
            booked = avail if u < p else (30 if u < p + 0.08 and avail >= 30 else 0)
            booked = min(booked, avail - blocked)
            started = int(round(booked / 60)) if booked else 0
            persons = 0 if day >= as_of else started * int(rng.choice([2, 4], p=[0.45, 0.55]))
            rows.append(
                (
                    day, datetime.combine(day, time(hour, 0)), hour, court, d.HALL_ZONE[court],
                    court in d.SHOW_COURTS, dtype, d.DAY_NAMES[day.weekday()], tier, avail, booked, blocked,
                    round((booked + blocked) / avail, 4) if avail else 0.0,
                    round(booked / 60 * d.TIER_PRICE_CHF[tier], 2), started, persons, rainy, school, public,
                )
            )
    return pd.DataFrame(rows, columns=cols)


def court_utilization_hourly(day: date, as_of: date, hall: str | None = None) -> pd.DataFrame:
    df = _court_day(day, as_of)
    if hall:
        df = df[df["hall_zone"] == hall]
    return df.reset_index(drop=True).copy()


def window(as_of: date) -> tuple[date, date]:
    return as_of - timedelta(days=d.HISTORY_DAYS), as_of + timedelta(days=d.FUTURE_DAYS - 1)


def daily_kpis(start: date, end: date, as_of: date) -> pd.DataFrame:
    rows = []
    day = start
    while day <= end:
        hourly = _court_day(day, as_of)
        if not hourly.empty:
            rng = _rng("kpi", day)
            avail = int(hourly["available_minutes"].sum())
            used = int(hourly["booked_minutes"].sum() + hourly["blocked_minutes"].sum())
            bookings = int(hourly["bookings_started"].sum())
            occ = used / avail if avail else 0.0
            past = day < as_of
            persons = int(hourly["check_in_persons"].sum())
            block_rev = 0.0
            if day.weekday() == 1 and not d.is_school_holiday(day) and d.day_type(day) == "weekday":
                block_rev += 30.0 * int(rng.integers(10, 17))
            if day.weekday() == 0 and d.day_type(day) == "weekday":
                block_rev += 50.0 * 1
            wl_req = int(rng.poisson(6 * max(occ - 0.6, 0.05) * 4))
            rows.append(
                (
                    day, d.DAY_NAMES[day.weekday()], d.day_type(day), round(float(hourly["revenue_chf"].sum()), 2),
                    block_rev, round(occ, 4), bookings,
                    int(rng.binomial(bookings, 0.08)) if past else int(rng.binomial(bookings, 0.03)),
                    int(rng.binomial(bookings, 0.05)) if past else 0,
                    int(rng.binomial(bookings, 0.025)) if past else 0,
                    wl_req, int(rng.binomial(wl_req, 0.45)) if past else 0, persons,
                    int(persons * 0.82) if past else 0, is_rainy(day), d.is_school_holiday(day),
                )
            )
        day += timedelta(days=1)
    return pd.DataFrame(rows, columns=GOLD_COLUMNS["daily_kpis"])


def slot_features(day: date, as_of: date) -> pd.DataFrame:
    dtype = d.day_type(day)
    cols = GOLD_COLUMNS["slot_features"]
    if dtype == "closed":
        return pd.DataFrame(columns=cols)
    rng = _rng("slots", day)
    future = day >= as_of
    share = _booked_share_so_far(day, as_of)
    rainy = is_rainy(day)
    temp = round(float(rng.normal(9 if day.month in (11, 12, 1, 2) else 18, 3)), 1)
    rows = []
    for start in d.start_times(dtype):
        hour = start.hour
        tier = d.tier_for(start, day.weekday(), dtype)
        blocked_courts = sum(1 for c in d.COURT_IDS if _blocked(day, c, hour) >= (60 if start.minute == 0 else 30))
        available = d.N_COURTS - blocked_courts
        base_p = _base_occupancy(day, hour, tier)
        hist4 = float(np.clip(base_p + rng.normal(0, 0.05), 0.05, 1.0))
        hist8 = float(np.clip(base_p + rng.normal(0, 0.04), 0.05, 1.0))
        p_now = base_p * share * (0.5 if _is_demo_lunch(day, hour, as_of) else 1.0)
        booked_now = int(np.clip(rng.binomial(available, float(np.clip(p_now, 0, 1))), 0, available))
        final = None if future else booked_now
        d3 = min(booked_now, int(round(booked_now * (0.7 if future else 0.8))))
        rows.append(
            (
                f"{day.isoformat()}T{start.strftime('%H%M')}", day, datetime.combine(day, start),
                hour + start.minute / 60, day.weekday() + 1, dtype, tier, d.TIER_PRICE_CHF[tier], future,
                d.is_school_holiday(day), day in d.vaud_public_holidays(day.year),
                day.weekday() in (1, 4) and hour >= 18, rainy, temp, available, d3, booked_now, final,
                None if future else round(booked_now / available, 4) if available else 0.0,
                d.TIER_PRICE_CHF[tier], round(hist4, 4), round(hist8, 4),
                int(rng.poisson(2.0 if base_p > 0.85 else 0.2)),
            )
        )
    df = pd.DataFrame(rows, columns=cols)
    for c in ("final_booked_courts",):
        df[c] = df[c].astype("Int64")
    return df


def _uuid(i: int) -> str:
    return hashlib.sha256(f"mock-player-{i}".encode()).hexdigest()


@lru_cache(maxsize=4)
def player_features(as_of: date) -> pd.DataFrame:
    rng = _rng("players", 20261123)
    rows = []
    pref_hours = {
        "Corporate_Lunch": 12, "Evening_Regular": 19, "Family_Weekend": 10, "Student_OffPeak": 14,
        "Club_Member": 20,
    }
    for i in range(N_PLAYERS):
        seg = str(rng.choice(d.SEGMENTS, p=[0.22, 0.28, 0.18, 0.22, 0.10]))
        regular = rng.random() < 0.25
        baseline = round(int(rng.poisson(3 * (6 if regular else 1.2))) / 3.0, 4)
        drifting = rng.random() < 0.13
        prev = int(rng.poisson(baseline * (0.6 if drifting else 1.0)))
        last = int(rng.poisson(baseline * (0.15 if drifting else 1.0)))
        change = round((last - prev) / prev, 4) if prev else None
        product = rng.choice([None, d.CARTE_PREPAYEE, d.PASS_LIBERTE, d.PASS_CHRONO], p=[0.55, 0.2, 0.12, 0.13])
        hours_left = balance = expires = days_to = None
        if product == d.PASS_CHRONO:
            days_to = int(rng.integers(-10, 120))
            expires = datetime.combine(as_of + timedelta(days=days_to), time(23, 59))
            hours_left = int(rng.integers(0, 26))
        elif product == d.PASS_LIBERTE:
            hours_left = int(rng.integers(1, 60))
        elif product == d.CARTE_PREPAYEE:
            balance = round(float(rng.uniform(5, 600)), 2)
        active = product is not None and (days_to is None or days_to >= 0)
        hour = pref_hours[seg]
        dow = 6 if seg == "Family_Weekend" else int(rng.integers(1, 6))
        dtype = "weekend_holiday" if dow >= 6 else "weekday"
        tier = d.tier_for(time(hour, 30 if dtype == "weekend_holiday" and hour == 9 else 0), dow - 1, dtype)
        partners = [_uuid(int(j)) for j in rng.choice(N_PLAYERS, size=3, replace=False) if j != i][:3]
        rows.append(
            (
                _uuid(i), seg, str(rng.choice(d.SKILL_TIERS, p=[0.3, 0.5, 0.2])),
                str(rng.choice(["FR", "EN"], p=[0.78, 0.22])), str(rng.choice(AGE_BANDS)),
                int(rng.integers(20, 900)), last, prev, last + prev + int(rng.poisson(baseline + 0.5)), baseline,
                change,
                int(rng.integers(0, 6)) if last else int(rng.integers(31, 120)), dow, hour, tier,
                round(float(rng.gamma(2.0, 30.0)), 1), round(float(rng.beta(1, 18)), 4),
                round(float(rng.beta(1, 30)), 4), bool(active), product, hours_left, balance, expires, days_to,
                partners,
                bool(baseline >= CHURN_MIN_MONTHLY_PLAYS and last <= CHURN_DROP_RATIO * baseline), as_of,
            )
        )
    df = pd.DataFrame(rows, columns=GOLD_COLUMNS["player_features"])
    for c in ("pass_hours_remaining", "days_to_pass_expiry"):
        df[c] = df[c].astype("Int64")
    for c in ("play_change_pct", "pass_balance_chf"):
        df[c] = df[c].astype(float)
    # Seeded moment 5: one Pass Chrono with unused hours expiring in 6 days.
    k = df.index[df["pass_product"] == d.PASS_CHRONO][0]
    df.loc[k, "days_to_pass_expiry"] = 6
    df.loc[k, "pass_hours_remaining"] = 9
    df.loc[k, "has_active_pass"] = True
    df.loc[k, "pass_expires_at"] = datetime.combine(as_of + timedelta(days=6), time(23, 59))
    return df


@lru_cache(maxsize=4)
def sponsor_metrics(as_of: date) -> pd.DataFrame:
    rng = _rng("sponsor", as_of.year)
    groups = {
        "segment": list(d.SEGMENTS),
        "age_band": list(AGE_BANDS),
        "time_band": ["morning", "lunch", "afternoon", "evening", "weekend"],
        "hall_zone": list(d.HALLS),
        "show_court": ["show_court", "standard_court"],
    }
    rows = []
    month = date(as_of.year, as_of.month, 1)
    months = []
    for _ in range(6):
        months.append(month)
        month = (month - timedelta(days=1)).replace(day=1)
    for m in sorted(months):
        summer = m.month in (7, 8)
        for group, keys in groups.items():
            for key in keys:
                players = int(rng.integers(40, 260) * (0.7 if summer else 1.0))
                if key == "65+" or (summer and key == "<18"):
                    players = int(rng.integers(3, 10))
                suppressed = players < d.K_ANONYMITY_MIN
                visits = int(players * rng.uniform(2.5, 6))
                rows.append(
                    (
                        m, group, key, None if suppressed else players, None if suppressed else visits,
                        None if suppressed else int(visits * 1.05),
                        None if suppressed else round(visits / 3.1, 1), suppressed,
                    )
                )
    df = pd.DataFrame(rows, columns=GOLD_COLUMNS["sponsor_metrics"])
    for c in ("unique_players", "player_visits", "check_in_persons"):
        df[c] = df[c].astype("Int64")
    return df
