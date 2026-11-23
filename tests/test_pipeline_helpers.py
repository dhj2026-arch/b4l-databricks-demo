import hashlib
import hmac
import re
from datetime import date, time
from pathlib import Path

import pytest

from b4l import pipeline_helpers as ph
from b4l import pricing_rules as pr
from b4l import schemas

PIPELINE_SRC = (Path(__file__).resolve().parents[1] / "src" / "02_lakeflow_etl.py").read_text()


def _grid(day_type, weekday0):
    cols = ["day_type", "weekday0", "hour", "avail_start_minute", "available_minutes", "tier"]
    return [dict(zip(cols, row)) for row in ph.hour_buckets(day_type, weekday0)]


def test_weekday_hour_grid():
    rows = _grid("weekday", 1)
    assert [r["hour"] for r in rows] == list(range(8, 22))
    assert all(r["available_minutes"] == 60 for r in rows)
    tiers = {r["hour"]: r["tier"] for r in rows}
    assert tiers[8] == pr.DAY and tiers[11] == pr.DAY and tiers[12] == pr.LUNCH
    assert tiers[13] == pr.LUNCH and tiers[14] == pr.DAY and tiers[17] == pr.EVENING and tiers[21] == pr.EVENING


def test_friday_afternoon_is_weekend_tier():
    tiers = {r["hour"]: r["tier"] for r in _grid("weekday", 4)}
    assert tiers[12] == pr.LUNCH and tiers[14] == pr.WEEKEND and tiers[20] == pr.WEEKEND


def test_weekend_hour_grid_has_half_first_hour():
    rows = _grid("weekend_holiday", 5)
    assert [r["hour"] for r in rows] == list(range(9, 19))
    first = rows[0]
    assert first["avail_start_minute"] == 9 * 60 + 30 and first["available_minutes"] == 30
    assert all(r["available_minutes"] == 60 for r in rows[1:])
    assert {r["tier"] for r in rows} == {pr.WEEKEND}


@pytest.mark.parametrize("day_type", ph.OPEN_DAY_TYPES)
def test_hour_grid_minutes_match_inventory_cells(day_type):
    minutes = sum(r["available_minutes"] for r in _grid(day_type, 2))
    assert minutes == len(pr.cell_starts(day_type)) * pr.SLOT_MINUTES


def test_closed_day_has_no_grid():
    assert ph.hour_buckets("closed", 3) == []


def test_hour_grid_rows_cover_all_weekdays():
    rows = ph.hour_grid_rows()
    assert len(rows) == 7 * (14 + 10)
    assert len({(r[0], r[1], r[2]) for r in rows}) == len(rows)
    assert len(rows[0]) == len(ph.HOUR_GRID_DDL.split(","))


def test_slot_grid_rows():
    rows = ph.slot_grid_rows()
    assert len(rows[0]) == len(ph.SLOT_GRID_DDL.split(","))
    tuesday = [r for r in rows if r[0] == "weekday" and r[1] == 1]
    assert [r[2] for r in tuesday] == [t.hour * 60 + t.minute for t in pr.start_times("weekday")]
    assert len(tuesday) == 27
    lunch_noon = next(r for r in tuesday if r[2] == 12 * 60)
    assert lunch_noon[3] == pr.LUNCH and lunch_noon[4] == 32.0
    saturday = [r for r in rows if r[0] == "weekend_holiday" and r[1] == 5]
    assert saturday[0][2] == 9 * 60 + 30 and saturday[-1][2] == 18 * 60
    assert all(r[4] == pr.TIER_PRICE_CHF[r[3]] for r in rows)


def test_hmac_matches_rfc4231_vector():
    expected = "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
    assert ph.hmac_hex("Jefe", "what do ya want for nothing?") == expected


def test_hmac_is_keyed_and_null_safe():
    assert ph.hmac_hex("k1", "P0001") != ph.hmac_hex("k2", "P0001")
    assert ph.hmac_hex("k1", "P0001") != hashlib.sha256(b"P0001").hexdigest()
    assert ph.hmac_hex("k1", None) is None
    assert ph.hmac_hex("k1", "P0001") == hmac.new(b"k1", b"P0001", hashlib.sha256).hexdigest()


@pytest.mark.parametrize(
    "birth_year,expected",
    [(2010, "<18"), (2009, "<18"), (2008, "18-25"), (2001, "18-25"), (2000, "26-35"), (1991, "26-35"),
     (1990, "36-50"), (1976, "36-50"), (1975, "51-65"), (1961, "51-65"), (1960, "65+"), (None, None)],
)
def test_age_band(birth_year, expected):
    assert ph.age_band(birth_year, date(2026, 11, 23)) == expected


def test_age_band_sql_lists_every_band():
    sql = ph.age_band_sql("year(as_of_date) - birth_year")
    for _, label in ph.AGE_BANDS:
        assert f"'{label}'" in sql
    assert sql.startswith("CASE") and sql.endswith("ELSE '65+' END")


@pytest.mark.parametrize(
    "start,day_type,expected",
    [(time(8, 0), "weekday", "morning"), (time(11, 0), "weekday", "morning"), (time(11, 30), "weekday", "lunch"),
     (time(13, 0), "weekday", "lunch"), (time(13, 30), "weekday", "afternoon"), (time(16, 30), "weekday", "afternoon"),
     (time(17, 0), "weekday", "evening"), (time(21, 0), "weekday", "evening"),
     (time(10, 0), "weekend_holiday", "weekend"), (time(18, 0), "weekend_holiday", "weekend")],
)
def test_time_band(start, day_type, expected):
    assert ph.time_band(start.hour * 60 + start.minute, day_type) == expected


def test_time_band_sql_shape():
    sql = ph.time_band_sql("start_minute", "day_type")
    assert "WHEN (day_type) = 'weekend_holiday' THEN 'weekend'" in sql
    assert "WHEN (start_minute) < 690 THEN 'morning'" in sql
    assert sql.endswith("ELSE 'evening' END")


@pytest.mark.parametrize(
    "value,ok",
    [("anna.muller12@bluewin.ch", True), ("j.doe@mail.example.com", True), ("anna.muller@", False),
     ("anna muller@gmail.com", False), ("annagmail.com", False), ("a@b", False)],
)
def test_email_pattern(value, ok):
    assert bool(re.match(ph.EMAIL_PATTERN, value)) is ok


@pytest.mark.parametrize(
    "value,ok",
    [("+41 79 123 45 67", True), ("079 123 45 67", True), ("+41791234567", True), ("12-34", False),
     ("+41 79 abc 45 67", False), ("123", False)],
)
def test_phone_pattern(value, ok):
    assert bool(re.match(ph.PHONE_PATTERN, value)) is ok


def test_patterns_are_safe_in_sql_literals():
    for pattern in (ph.EMAIL_PATTERN, ph.PHONE_PATTERN):
        assert "\\" not in pattern and "'" not in pattern
    assert ph.regex_match_sql("email", "^a$") == "coalesce(email RLIKE '^a$', false)"


def test_booking_rules_sql():
    rules = ph.BOOKING_RULES
    assert set(rules) >= {"known_court", "positive_duration", "valid_party_size", "start_before_end", "valid_status"}
    violates = ph.violates_any_sql(rules)
    assert violates.startswith("NOT (") and violates.count(" AND (") >= len(rules) - 1
    reason = ph.violation_reason_sql(rules)
    assert reason.startswith("concat_ws(',', ")
    for name in rules:
        assert f"THEN '{name}' END" in reason
    for status in ph.BOOKING_STATUSES:
        assert f"'{status}'" in rules["valid_status"]


def test_billed_statuses():
    assert "cancelled_early" not in ph.BILLED_OR_UPCOMING
    assert set(ph.BILLED_STATUSES) < set(ph.BILLED_OR_UPCOMING) <= set(ph.BOOKING_STATUSES)


def test_pipeline_defines_every_contract_table():
    gold_defined = set(re.findall(r'@gold_view\(\s*"(\w+)"', PIPELINE_SRC))
    assert gold_defined == set(schemas.GOLD)
    for table in schemas.SILVER_TABLES:
        assert re.search(rf"@dp\.materialized_view\([^@]*?\)\n(@dp\.expect[^\n]*\n)*def {table}\(\):",
                         PIPELINE_SRC, re.S), table


def test_pipeline_uses_no_forbidden_actions():
    for forbidden in (".collect(", ".toPandas(", ".saveAsTable(", ".save(", "LIVE."):
        assert forbidden not in PIPELINE_SRC
