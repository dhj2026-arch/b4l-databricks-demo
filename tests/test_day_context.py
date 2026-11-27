"""get_day_context: demand hint rules, generated SQL, grants, and the staff app's copy of the rules."""

import itertools
import re
import sys
from datetime import date
from pathlib import Path

import pytest

from b4l import day_context as dc

APP_DIR = Path(__file__).resolve().parents[1] / "apps" / "staff_ops_app"
sys.path.insert(0, str(APP_DIR))

import day_outlook  # noqa: E402

BASE = {"day_type": "weekday", "day_of_week": 3, "is_closed": False, "is_public_holiday": False,
        "is_school_holiday": False, "school_holiday_name": None, "arena_event": False, "is_rainy": False,
        "temp_max_c": 12.0}


def row(**kw):
    return {**BASE, **kw}


WEEKEND = {"day_type": "weekend_holiday", "day_of_week": 6}


def test_quiet_weekday_is_a_normal_day():
    assert dc.demand_hint(row()) == dc.NORMAL_HINT


def test_closed_day_shows_only_closed():
    assert dc.demand_hint(row(is_closed=True, day_type="closed", is_rainy=True, arena_event=True)) == dc.CLOSED_HINT


def test_rainy_weekend_with_arena_event_lists_both_in_order():
    hints = dc.demand_hint(row(**WEEKEND, is_rainy=True, arena_event=True)).split(dc.HINT_SEPARATOR)
    assert hints[0].startswith("Rainy weekend") and hints[1].startswith("Event at the Vaudoise")


def test_rain_on_a_weekday_is_the_milder_hint():
    assert dc.demand_hint(row(is_rainy=True)).startswith("Rain: slightly busier")


@pytest.mark.parametrize("temp, weekend, expected", [
    (29.0, False, "Hot day"), (29.0, True, "Hot day"), (24.0, True, "Warm weekend"), (24.0, False, dc.NORMAL_HINT),
])
def test_heat_hints(temp, weekend, expected):
    r = row(**WEEKEND, temp_max_c=temp) if weekend else row(temp_max_c=temp)
    assert dc.demand_hint(r).startswith(expected)


def test_missing_weather_gives_no_weather_hint():
    assert dc.demand_hint(row(is_rainy=None, temp_max_c=None)) == dc.NORMAL_HINT
    assert dc.demand_hint(row(is_rainy=float("nan"), temp_max_c=float("nan"))) == dc.NORMAL_HINT


def test_summer_and_other_school_holidays_differ():
    summer = dc.demand_hint(row(is_school_holiday=True, school_holiday_name="Vacances d'été"))
    autumn = dc.demand_hint(row(is_school_holiday=True, school_holiday_name="Vacances d'automne"))
    assert summer.startswith("Summer school holidays") and autumn.startswith("School holidays")


def test_badmatin_on_term_tuesdays_only():
    assert "Bad'matin 09:30" in dc.demand_hint(row(day_of_week=2))
    assert "No Bad'matin" in dc.demand_hint(row(day_of_week=2, is_school_holiday=True,
                                                 school_holiday_name="Vacances d'automne"))
    assert "Bad'matin" not in dc.demand_hint(row(**{**WEEKEND, "day_of_week": 2}, is_public_holiday=True))


def test_opening_hours():
    assert dc.opening_hours("weekday") == "08:00-22:00"
    assert dc.opening_hours("weekend_holiday") == "09:30-19:00"
    assert dc.opening_hours("closed") is None


def test_sql_escapes_apostrophes_with_backslashes():
    sql = dc.function_sql("cat", "b4l")
    assert "Bad\\'matin" in sql and "d\\'été" in sql
    assert not re.search(r"[^\\]''[^)]", sql.replace("nullif(concat_ws", "")), "doubled quotes concatenate in Spark"


def test_sql_returns_every_column_and_reads_silver():
    sql = dc.function_sql("cat", "b4l")
    for name, typ, _ in dc.COLUMNS:
        assert f"{name} {typ} COMMENT" in sql
    assert "`cat`.`b4l_silver`.calendar" in sql and "`cat`.`b4l_silver`.weather" in sql
    assert "CREATE OR REPLACE FUNCTION `cat`.`b4l_gold`.get_day_context" in sql
    assert f", {dc.MAX_DAYS}))" in sql
    for _, hint, cond, _ in dc.RULES:
        assert cond in sql


def test_grant_is_execute_only_and_validates_input():
    (stmt,) = dc.grant_statements("cat", "b4l", "7d1c34b3-f25f-40a4-bfea-2950bde54b1d")
    assert stmt.startswith("GRANT EXECUTE ON FUNCTION `cat`.`b4l_gold`.get_day_context TO")
    with pytest.raises(ValueError):
        dc.grant_statements("cat", "b4l", "x`; DROP")
    with pytest.raises(ValueError):
        dc.function_sql("cat; drop", "b4l")


# ---------------------------------------------------------------- staff app copy

def test_app_copy_has_the_same_columns_hints_and_thresholds():
    assert day_outlook.COLUMNS == [n for n, _, _ in dc.COLUMNS]
    assert [(k, h) for k, h, _ in day_outlook.RULES] == [(k, h) for k, h, _, _ in dc.RULES]
    for name in ("HOT_DAY_C", "WARM_WEEKEND_C", "SUMMER_HOLIDAYS", "CLOSED_HINT", "NORMAL_HINT", "HINT_SEPARATOR"):
        assert getattr(day_outlook, name) == getattr(dc, name), name


def test_app_copy_gives_the_same_hint_on_every_combination():
    grid = itertools.product(
        [("weekday", 2), ("weekday", 3), ("weekend_holiday", 6), ("closed", 4)], [False, True],
        [None, "Vacances d'été", "Vacances d'automne"], [False, True], [None, False, True], [None, 15.0, 24.0, 30.0],
    )
    n = 0
    for (dtype, dow), public, school, arena, rainy, temp in grid:
        r = row(day_type=dtype, day_of_week=dow, is_closed=dtype == "closed", is_public_holiday=public,
                is_school_holiday=school is not None, school_holiday_name=school, arena_event=arena, is_rainy=rainy,
                temp_max_c=temp)
        assert day_outlook.demand_hint(r) == dc.demand_hint(r), r
        n += 1
    assert n == 4 * 2 * 3 * 2 * 3 * 4


def test_mock_day_context_matches_the_calendar():
    df = day_outlook.mock_day_context(date(2026, 7, 31), 3, as_of=date(2026, 11, 23))
    assert list(df.columns) == day_outlook.COLUMNS and len(df) == 3
    aug1 = df.set_index("date").loc[date(2026, 8, 1)]
    assert aug1["public_holiday"] == "Fête nationale" and aug1["school_holiday"] == "Vacances d'été"
    assert aug1["opening_hours"] == "09:30-19:00" and not aug1["is_forecast"]
    assert day_outlook.mock_day_context(date(2026, 12, 25), 1, date(2026, 11, 23)).iloc[0]["demand_hint"] == dc.CLOSED_HINT
