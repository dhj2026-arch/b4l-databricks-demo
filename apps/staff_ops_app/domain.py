"""Domain constants for the staff app.

The app is deployed from apps/staff_ops_app only, so it cannot import src/b4l at runtime. These values are
copies of src/b4l/pricing_rules.py, src/b4l/calendar.py and src/b4l/schemas.py;
tests/test_staff_app_contract.py fails if they drift.
"""

from __future__ import annotations

from datetime import date, time, timedelta

N_COURTS = 16
COURT_IDS = [f"Court_{i:02d}" for i in range(1, N_COURTS + 1)]
SHOW_COURTS = COURT_IDS[:4]
MAIN_HALL = "Main Hall"
WEST_HALL = "West Hall"
HALLS = (MAIN_HALL, WEST_HALL)
HALL_ZONE = {c: (MAIN_HALL if i < 8 else WEST_HALL) for i, c in enumerate(COURT_IDS)}

WEEKDAY_OPEN = time(8, 0)
WEEKDAY_CLOSE = time(22, 0)
WEEKEND_OPEN = time(9, 30)
WEEKEND_CLOSE = time(19, 0)
SLOT_MINUTES = 30

DAY = "Day"
WEEKEND = "Weekend"
LUNCH = "Lunch"
EVENING = "Evening"
TIERS = (DAY, WEEKEND, LUNCH, EVENING)
TIER_PRICE_CHF = {DAY: 25.0, WEEKEND: 25.0, LUNCH: 32.0, EVENING: 35.0}
TIER_LABEL_FR = {DAY: "Tarif journée", WEEKEND: "Tarif week-end", LUNCH: "Tarif midi", EVENING: "Tarif soir"}

CARTE_PREPAYEE = "Carte_Prepayee"
PASS_LIBERTE = "Pass_Liberte"
PASS_CHRONO = "Pass_Chrono"
PASS_PRODUCTS = (CARTE_PREPAYEE, PASS_LIBERTE, PASS_CHRONO)

PRICE_FLOOR_CHF = 18.0
PRICE_CEILING_CHF = 42.0

HISTORY_DAYS = 168
FUTURE_DAYS = 14
K_ANONYMITY_MIN = 10

SEGMENTS = ("Corporate_Lunch", "Evening_Regular", "Family_Weekend", "Student_OffPeak", "Club_Member")
SKILL_TIERS = ("Beginner", "Intermediate", "Advanced")

CLOSED_DAYS = {(12, 24), (12, 25), (12, 31), (1, 1)}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

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


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def easter_sunday(year: int) -> date:
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


def is_school_holiday(d: date) -> bool:
    return any(start <= d <= end for _, start, end in SCHOOL_HOLIDAYS)


def day_type(d: date) -> str:
    """'closed', 'weekend_holiday' (weekend hours) or 'weekday'."""
    if (d.month, d.day) in CLOSED_DAYS:
        return "closed"
    if d.weekday() >= 5 or d in vaud_public_holidays(d.year):
        return "weekend_holiday"
    return "weekday"


def rentable_window(dtype: str) -> tuple[time, time] | None:
    if dtype == "closed":
        return None
    if dtype == "weekend_holiday":
        return WEEKEND_OPEN, WEEKEND_CLOSE
    return WEEKDAY_OPEN, WEEKDAY_CLOSE


def heatmap_hours(dtype: str) -> list[int]:
    """Clock hours that contain rentable minutes: weekday 8-21, weekend/holiday 9-18."""
    window = rentable_window(dtype)
    if window is None:
        return []
    first, close = (_minutes(t) for t in window)
    return list(range(first // 60, (close - 1) // 60 + 1))


def available_minutes(dtype: str, hour: int) -> int:
    window = rentable_window(dtype)
    if window is None:
        return 0
    first, close = (_minutes(t) for t in window)
    return max(0, min(close, (hour + 1) * 60) - max(first, hour * 60))


def start_times(dtype: str, duration_minutes: int = 60) -> list[time]:
    window = rentable_window(dtype)
    if window is None:
        return []
    first, close = (_minutes(t) for t in window)
    return [time(m // 60, m % 60) for m in range(first, close - duration_minutes + 1, SLOT_MINUTES)]


def tier_for(start: time, weekday: int, dtype: str) -> str:
    """Base tier set by the booking start time. weekday: 0=Mon .. 6=Sun."""
    if dtype == "weekend_holiday":
        return WEEKEND
    m = _minutes(start)
    if 11 * 60 + 30 <= m < 13 * 60 + 30:
        return LUNCH
    if weekday == 4 and m >= 13 * 60 + 30:
        return WEEKEND
    if m >= 17 * 60:
        return EVENING
    return DAY


def tier_label(tier: str) -> str:
    return f"{tier} ({TIER_LABEL_FR.get(tier, tier)})"
