"""Opening calendar for the centre: closures, Vaud public holidays and Vaud school holidays.

School holiday dates follow the official Vaud calendar (vd.ch, "Vacances scolaires vaudoises 2022-2031").
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

# The centre is closed on these month/day pairs (b4l-fondation.ch/horaires-tarifs).
CLOSED_DAYS = {(12, 24), (12, 25), (12, 31), (1, 1)}

SCHOOL_HOLIDAYS = [
    ("Vacances d'automne", date(2025, 10, 11), date(2025, 10, 26)),
    ("Vacances d'hiver", date(2025, 12, 20), date(2026, 1, 4)),
    ("Relâches", date(2026, 2, 14), date(2026, 2, 22)),
    ("Vacances de Pâques", date(2026, 4, 3), date(2026, 4, 19)),
    ("Pont de l'Ascension", date(2026, 5, 14), date(2026, 5, 17)),
    ("Vacances d'été", date(2026, 6, 27), date(2026, 8, 16)),
    ("Vacances d'automne", date(2026, 10, 10), date(2026, 10, 25)),
    ("Vacances d'hiver", date(2026, 12, 24), date(2027, 1, 10)),
    ("Relâches", date(2027, 2, 6), date(2027, 2, 14)),
    ("Vacances de Pâques", date(2027, 3, 26), date(2027, 4, 11)),
    ("Pont de l'Ascension", date(2027, 5, 6), date(2027, 5, 9)),
    ("Vacances d'été", date(2027, 7, 3), date(2027, 8, 22)),
]

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def easter_sunday(year: int) -> date:
    """Gregorian Easter (Anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day + 1)


def vaud_public_holidays(year: int) -> dict[date, str]:
    easter = easter_sunday(year)
    sept_first = date(year, 9, 1)
    first_sunday = sept_first + timedelta(days=(6 - sept_first.weekday()) % 7)
    jeune_monday = first_sunday + timedelta(days=15)
    return {
        date(year, 1, 1): "Nouvel An",
        date(year, 1, 2): "Saint-Berchtold",
        easter - timedelta(days=2): "Vendredi-Saint",
        easter + timedelta(days=1): "Lundi de Pâques",
        easter + timedelta(days=39): "Ascension",
        easter + timedelta(days=50): "Lundi de Pentecôte",
        date(year, 8, 1): "Fête nationale",
        jeune_monday: "Lundi du Jeûne fédéral",
        date(year, 12, 25): "Noël",
    }


def school_holiday_name(d: date) -> str | None:
    for name, start, end in SCHOOL_HOLIDAYS:
        if start <= d <= end:
            return name
    return None


def day_type(d: date) -> str:
    """'closed', 'weekend_holiday' (weekend hours) or 'weekday'."""
    if (d.month, d.day) in CLOSED_DAYS:
        return "closed"
    if d.weekday() >= 5 or d in vaud_public_holidays(d.year):
        return "weekend_holiday"
    return "weekday"


def arena_event_days(start: date, end: date, seed: int) -> set[date]:
    """ASSUMPTION: Vaudoise Aréna evening events (ice hockey season Sep-Apr, a few concerts otherwise)."""
    rng = np.random.default_rng(seed + 7)
    out = set()
    d = start
    while d <= end:
        in_season = d.month >= 9 or d.month <= 4
        p = {1: 0.35, 4: 0.45, 5: 0.5}.get(d.weekday(), 0.05) if in_season else 0.04
        if rng.random() < p:
            out.add(d)
        d += timedelta(days=1)
    return out


def build_calendar(start: date, end: date, seed: int) -> list[dict]:
    arena = arena_event_days(start, end, seed)
    rows = []
    d = start
    while d <= end:
        holidays = vaud_public_holidays(d.year)
        school = school_holiday_name(d)
        dtype = day_type(d)
        rows.append(
            {
                "date": d,
                "day_of_week": d.weekday() + 1,
                "day_name": DAY_NAMES[d.weekday()],
                "is_weekend": d.weekday() >= 5,
                "is_public_holiday": d in holidays,
                "holiday_name": holidays.get(d),
                "is_closed": dtype == "closed",
                "is_school_holiday": school is not None,
                "school_holiday_name": school,
                "day_type": dtype,
                "arena_event": d in arena,
                "iso_week": d.isocalendar()[1],
                "month": d.month,
            }
        )
        d += timedelta(days=1)
    return rows
