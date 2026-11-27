"""Day outlook: weather, holidays, Aréna events and a demand hint per day.

Live, the app calls the UC function `get_day_context` (src/b4l/day_context.py). Offline, `mock_day_context` builds the
same columns from deterministic mock weather. RULES copies the function's hint rules;
tests/test_day_context.py fails if they drift.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, timedelta
from typing import Any

import pandas as pd

import domain as d
import mock_data

COLUMNS = [
    "date", "day_name", "day_type", "opening_hours", "condition", "temp_mean_c", "temp_max_c", "precip_mm",
    "is_rainy", "is_forecast", "public_holiday", "school_holiday", "arena_event", "demand_hint",
]
HOT_DAY_C = 28.0
WARM_WEEKEND_C = 22.0
SUMMER_HOLIDAYS = "Vacances d'été"
CLOSED_HINT = "Centre closed"
NORMAL_HINT = "Normal day"
HINT_SEPARATOR = " · "

Row = Mapping[str, Any]


def _rainy(r: Row) -> bool:
    v = r.get("is_rainy")
    return v is not None and v == v and bool(v)


def _dry(r: Row) -> bool:
    v = r.get("is_rainy")
    return v is not None and v == v and not bool(v)


def _temp(r: Row) -> float:
    v = r.get("temp_max_c")
    return 0.0 if v is None or v != v else float(v)


def _weekend(r: Row) -> bool:
    return r["day_type"] == "weekend_holiday"


def _tuesday_weekday(r: Row) -> bool:
    return r["day_type"] == "weekday" and int(r["day_of_week"]) == 2


RULES: list[tuple[str, str, Callable[[Row], bool]]] = [
    ("public_holiday", "Public holiday: weekend hours, about 10% quieter than a normal weekend",
     lambda r: bool(r["is_public_holiday"])),
    ("rain_weekend", "Rainy weekend: expect busier courts than usual (about +25%); players book earlier",
     lambda r: _rainy(r) and _weekend(r)),
    ("rain_weekday", "Rain: slightly busier than usual (about +5-8%)",
     lambda r: _rainy(r) and r["day_type"] == "weekday"),
    ("hot", f"Hot day ({HOT_DAY_C:.0f} °C or more): daytime demand drops (about -20 to -25%)",
     lambda r: _dry(r) and _temp(r) >= HOT_DAY_C),
    ("warm_weekend", f"Warm weekend ({WARM_WEEKEND_C:.0f} °C or more): slightly quieter (about -8%)",
     lambda r: _dry(r) and WARM_WEEKEND_C <= _temp(r) < HOT_DAY_C and _weekend(r)),
    ("summer_holidays", "Summer school holidays: far fewer corporate, evening, family and club players",
     lambda r: r.get("school_holiday_name") == SUMMER_HOLIDAYS),
    ("school_holidays", "School holidays: fewer corporate and club players, more students and families in the daytime",
     lambda r: bool(r["is_school_holiday"]) and r.get("school_holiday_name") != SUMMER_HOLIDAYS),
    ("badmatin", "Bad'matin 09:30-11:00 blocks Courts 1-4",
     lambda r: _tuesday_weekday(r) and not r["is_school_holiday"]),
    ("no_badmatin", "No Bad'matin in school holidays: Courts 1-4 free on Tuesday morning",
     lambda r: _tuesday_weekday(r) and bool(r["is_school_holiday"])),
    ("arena", "Event at the Vaudoise Aréna: evening demand a little lower (about -7%), parking busier",
     lambda r: bool(r["arena_event"])),
]


def demand_hint(row: Row) -> str:
    if row["is_closed"]:
        return CLOSED_HINT
    return HINT_SEPARATOR.join(hint for _, hint, pred in RULES if pred(row)) or NORMAL_HINT


def opening_hours(day_type: str) -> str | None:
    window = d.rentable_window(day_type)
    return None if window is None else f"{window[0]:%H:%M}-{window[1]:%H:%M}"


# Lausanne monthly mean of daily max temperature (°C), for mock weather only.
_MONTH_MAX_C = {1: 4, 2: 6, 3: 11, 4: 15, 5: 19, 6: 23, 7: 26, 8: 25, 9: 21, 10: 15, 11: 9, 12: 5}


def _school_holiday_name(day: date) -> str | None:
    return next((name for name, start, end in d.SCHOOL_HOLIDAYS if start <= day <= end), None)


def mock_day_context(start: date, n_days: int, as_of: date) -> pd.DataFrame:
    rows = []
    for i in range(max(1, min(n_days, 31))):
        day = start + timedelta(days=i)
        rng = mock_data._rng("weather", day)
        dtype = d.day_type(day)
        rainy = mock_data.is_rainy(day)
        t_max = round(float(_MONTH_MAX_C[day.month] + rng.normal(0, 3)), 1)
        holiday = d.vaud_public_holidays(day.year).get(day)
        school = _school_holiday_name(day)
        arena = bool(rng.random() < (0.35 if day.weekday() in (1, 4) and (day.month >= 9 or day.month <= 4) else 0.03))
        calendar = {
            "day_type": dtype, "day_of_week": day.weekday() + 1, "is_closed": dtype == "closed",
            "is_public_holiday": holiday is not None, "is_school_holiday": school is not None,
            "school_holiday_name": school, "arena_event": arena, "is_rainy": rainy, "temp_max_c": t_max,
        }
        rows.append((
            day, d.DAY_NAMES[day.weekday()], dtype, opening_hours(dtype),
            "rain" if rainy else ("sunny" if rng.random() < 0.5 else "cloudy"), round(t_max - 4.5, 1), t_max,
            round(float(rng.uniform(1, 15)), 1) if rainy else round(float(rng.uniform(0, 0.9)), 1), rainy,
            day >= as_of, holiday, school, arena, demand_hint(calendar),
        ))
    return pd.DataFrame(rows, columns=COLUMNS)


CONDITION_ICON = {"sunny": "☀️", "cloudy": "☁️", "rain": "🌧️", "snow": "🌨️"}
