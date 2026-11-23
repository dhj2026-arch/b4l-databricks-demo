from datetime import date, datetime, time

import pytest

from b4l import pricing_rules as pr
from b4l.calendar import build_calendar, day_type, easter_sunday, vaud_public_holidays


@pytest.mark.parametrize(
    "start,weekday,dtype,expected",
    [
        (time(8, 0), 0, "weekday", pr.DAY),
        (time(11, 0), 1, "weekday", pr.DAY),
        (time(11, 30), 1, "weekday", pr.LUNCH),
        (time(13, 0), 2, "weekday", pr.LUNCH),
        (time(13, 30), 2, "weekday", pr.DAY),
        (time(16, 30), 3, "weekday", pr.DAY),
        (time(17, 0), 3, "weekday", pr.EVENING),
        (time(21, 0), 0, "weekday", pr.EVENING),
        (time(12, 0), 4, "weekday", pr.LUNCH),
        (time(13, 30), 4, "weekday", pr.WEEKEND),
        (time(19, 0), 4, "weekday", pr.WEEKEND),
        (time(10, 0), 5, "weekend_holiday", pr.WEEKEND),
        (time(12, 0), 0, "weekend_holiday", pr.WEEKEND),
    ],
)
def test_tier_by_start_time(start, weekday, dtype, expected):
    assert pr.tier_for(start, weekday, dtype) == expected


def test_start_grid():
    weekday = pr.start_times("weekday")
    weekend = pr.start_times("weekend_holiday")
    assert weekday[0] == time(8, 0) and weekday[-1] == time(21, 0)
    assert weekend[0] == time(9, 30) and weekend[-1] == time(18, 0)
    assert pr.start_times("closed") == []


def test_capacity_close_to_site_figure():
    hours = pr.court_hours_per_week()
    assert hours == 16 * (5 * 14 + 2 * 9.5)
    assert abs(hours - pr.SITE_SLOTS_PER_WEEK) / pr.SITE_SLOTS_PER_WEEK < 0.05


def test_family_rate_only_day_and_weekend():
    assert pr.base_price(pr.DAY, pr.FAMILY) == 18.0
    with pytest.raises(ValueError):
        pr.base_price(pr.EVENING, pr.FAMILY)


def test_guardrails_bounds():
    assert pr.price_bounds(pr.DAY) == (18.0, 35.0)
    assert pr.price_bounds(pr.LUNCH) == (20.0, 42.0)
    assert pr.price_bounds(pr.EVENING) == (21.0, 42.0)
    assert pr.apply_guardrails(pr.EVENING, 60).final_price_chf == 42.0
    assert pr.apply_guardrails(pr.DAY, 5).applied_rule == "floor"
    assert pr.apply_guardrails(pr.LUNCH, 27.4).final_price_chf == 27.0


def test_guardrails_protect_special_cases():
    assert pr.apply_guardrails(pr.DAY, 30, rate_category=pr.FAMILY).final_price_chf == 18.0
    assert pr.apply_guardrails(pr.EVENING, 40, payment_product=pr.PASS_CHRONO).applied_rule == "pass_protected"
    assert pr.apply_guardrails(pr.EVENING, 40, payment_product=pr.CARTE_PREPAYEE).final_price_chf == 40.0
    assert pr.apply_guardrails(pr.DAY, 30, is_block=True).final_price_chf == 25.0
    assert pr.apply_guardrails(pr.DAY, 30, locked_price=22.0).final_price_chf == 22.0


def test_candidate_prices_within_policy():
    for tier in pr.TIERS:
        prices = pr.candidate_prices(tier)
        assert min(prices) >= pr.PRICE_FLOOR_CHF and max(prices) <= pr.PRICE_CEILING_CHF
        assert pr.TIER_PRICE_CHF[tier] in prices


def test_pass_coverage():
    assert pr.pass_covers(pr.MIDI, pr.LUNCH)
    assert not pr.pass_covers(pr.MIDI, pr.EVENING)
    assert pr.pass_covers(pr.SOIR, pr.EVENING)


def test_late_cancellation():
    start = datetime(2026, 11, 26, 19, 0)
    assert pr.is_late_cancellation(datetime(2026, 11, 26, 8, 0), start)
    assert not pr.is_late_cancellation(datetime(2026, 11, 25, 18, 0), start)


def test_calendar_facts():
    assert easter_sunday(2026) == date(2026, 4, 5)
    holidays = vaud_public_holidays(2026)
    assert holidays[date(2026, 9, 21)] == "Lundi du Jeûne fédéral"
    assert day_type(date(2026, 12, 24)) == "closed"
    assert day_type(date(2027, 1, 2)) == "weekend_holiday"
    assert day_type(date(2026, 8, 1)) == "weekend_holiday"
    assert day_type(date(2026, 11, 24)) == "weekday"
    rows = {r["date"]: r for r in build_calendar(date(2026, 6, 8), date(2026, 12, 7), seed=1)}
    assert rows[date(2026, 7, 15)]["is_school_holiday"]
    assert not rows[date(2026, 8, 20)]["is_school_holiday"]
    assert rows[date(2026, 10, 12)]["school_holiday_name"] == "Vacances d'automne"
