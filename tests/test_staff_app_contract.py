"""The staff app ships without src/b4l, so its copied constants and mock frames must match the contracts."""

import sys
from datetime import date, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from b4l import calendar as cal
from b4l import pricing_rules as pr
from b4l import schemas

APP_DIR = Path(__file__).resolve().parents[1] / "apps" / "staff_ops_app"
sys.path.insert(0, str(APP_DIR))

import data  # noqa: E402
import domain  # noqa: E402
import genie_client  # noqa: E402
import mock_data  # noqa: E402

AS_OF = date(2026, 11, 23)


def test_mock_column_lists_match_gold_schema():
    assert set(mock_data.GOLD_COLUMNS) == set(schemas.GOLD)
    for table, cols in schemas.GOLD.items():
        assert mock_data.GOLD_COLUMNS[table] == schemas.column_names(cols), table


def _frames():
    lo, hi = mock_data.window(AS_OF)
    return {
        "court_utilization_hourly": mock_data.court_utilization_hourly(AS_OF - timedelta(days=3), AS_OF),
        "daily_kpis": mock_data.daily_kpis(lo, hi, AS_OF),
        "player_features": mock_data.player_features(AS_OF),
        "slot_features": mock_data.slot_features(AS_OF + timedelta(days=2), AS_OF),
        "sponsor_metrics": mock_data.sponsor_metrics(AS_OF),
    }


def test_mock_frames_have_exact_gold_columns():
    for table, frame in _frames().items():
        assert list(frame.columns) == schemas.column_names(schemas.GOLD[table]), table
        assert not frame.empty, table


def test_domain_constants_match_pricing_rules():
    assert domain.N_COURTS == pr.N_COURTS
    assert domain.COURT_IDS == pr.COURT_IDS
    assert domain.SHOW_COURTS == pr.SHOW_COURTS
    assert domain.TIERS == pr.TIERS
    assert (domain.DAY, domain.WEEKEND, domain.LUNCH, domain.EVENING) == (pr.DAY, pr.WEEKEND, pr.LUNCH, pr.EVENING)
    assert domain.TIER_PRICE_CHF == pr.TIER_PRICE_CHF
    assert domain.TIER_LABEL_FR == pr.TIER_LABEL_FR
    assert (domain.WEEKDAY_OPEN, domain.WEEKDAY_CLOSE) == (pr.WEEKDAY_OPEN, pr.WEEKDAY_CLOSE)
    assert (domain.WEEKEND_OPEN, domain.WEEKEND_CLOSE) == (pr.WEEKEND_OPEN, pr.WEEKEND_CLOSE)
    assert domain.SLOT_MINUTES == pr.SLOT_MINUTES
    assert domain.PASS_PRODUCTS == pr.PASS_PRODUCTS
    assert (domain.PRICE_FLOOR_CHF, domain.PRICE_CEILING_CHF) == (pr.PRICE_FLOOR_CHF, pr.PRICE_CEILING_CHF)
    assert domain.K_ANONYMITY_MIN == schemas.K_ANONYMITY_MIN


def test_hall_zones():
    expected = pr.HALL_ZONE or {
        c: ("Main Hall" if i < 8 else "West Hall") for i, c in enumerate(pr.COURT_IDS)
    }
    assert domain.HALL_ZONE == expected


def test_domain_calendar_matches_b4l_calendar():
    assert domain.CLOSED_DAYS == cal.CLOSED_DAYS
    assert domain.SCHOOL_HOLIDAYS == cal.SCHOOL_HOLIDAYS
    assert domain.DAY_NAMES == cal.DAY_NAMES
    for year in (2025, 2026, 2027):
        assert domain.vaud_public_holidays(year) == cal.vaud_public_holidays(year)
    day = date(2025, 6, 1)
    while day <= date(2027, 6, 1):
        assert domain.day_type(day) == cal.day_type(day), day
        assert domain.is_school_holiday(day) == (cal.school_holiday_name(day) is not None), day
        day += timedelta(days=1)


@pytest.mark.parametrize("dtype", ["weekday", "weekend_holiday", "closed"])
def test_domain_tiers_and_windows_match(dtype):
    assert domain.rentable_window(dtype) == pr.rentable_window(dtype)
    assert domain.start_times(dtype) == pr.start_times(dtype)
    for weekday in range(7):
        for start in pr.start_times(dtype):
            assert domain.tier_for(start, weekday, dtype) == pr.tier_for(start, weekday, dtype)


def test_heatmap_hours_follow_rentable_window():
    assert domain.heatmap_hours("weekday") == list(range(8, 22))
    assert domain.heatmap_hours("weekend_holiday") == list(range(9, 19))
    assert domain.available_minutes("weekend_holiday", 9) == 30
    for dtype in ("weekday", "weekend_holiday"):
        total = sum(domain.available_minutes(dtype, h) for h in domain.heatmap_hours(dtype))
        assert total == len(pr.cell_starts(dtype)) * pr.SLOT_MINUTES


def test_mock_hourly_is_consistent():
    for offset in (-10, -3, 0, 5):
        day = AS_OF + timedelta(days=offset)
        df = mock_data.court_utilization_hourly(day, AS_OF)
        if domain.day_type(day) == "closed":
            continue
        assert set(df["court_id"]) == set(pr.COURT_IDS)
        assert (df["booked_minutes"] + df["blocked_minutes"] <= df["available_minutes"]).all()
        assert sorted(df["hour"].unique()) == domain.heatmap_hours(domain.day_type(day))
        for _, row in df.iterrows():
            start = max(time(int(row["hour"]), 0), pr.rentable_window(row["day_type"])[0])
            assert row["tier"] == pr.tier_for(start, day.weekday(), row["day_type"])


def test_mock_is_deterministic():
    a = mock_data.slot_features(AS_OF + timedelta(days=1), AS_OF)
    b = mock_data.slot_features(AS_OF + timedelta(days=1), AS_OF)
    assert a.equals(b)


def test_sponsor_mock_respects_k_anonymity():
    df = mock_data.sponsor_metrics(AS_OF)
    assert df["suppressed"].any()
    assert df.loc[df["suppressed"], "unique_players"].isna().all()
    assert (df.loc[~df["suppressed"], "unique_players"] >= schemas.K_ANONYMITY_MIN).all()


def test_mock_backend_queries():
    backend = data.MockBackend(AS_OF)
    lo, hi = backend.date_bounds()
    assert lo == AS_OF - timedelta(days=168) and hi == AS_OF + timedelta(days=13)
    main = backend.court_utilization(AS_OF - timedelta(days=1), "Main Hall")
    assert set(main["court_id"]) == set(pr.COURT_IDS[:8])
    risk = backend.at_risk_players()
    assert not risk.empty and list(risk.columns) == data.AT_RISK_COLUMNS
    assert risk["avg_monthly_plays_baseline"].is_monotonic_decreasing
    assert (risk["plays_last_30d"] <= 0.35 * risk["avg_monthly_plays_baseline"]).all()
    chrono = backend.expiring_chrono()
    assert not chrono.empty
    assert chrono["days_to_pass_expiry"].between(0, 14).all()
    slots = data.underfilled_slots(backend.slot_features(AS_OF + timedelta(days=1)))
    assert {"booked_share", "gap_vs_hist", "underfilled"} <= set(slots.columns)


def test_backend_selection(monkeypatch):
    monkeypatch.setenv("B4L_USE_MOCK", "true")
    backend, reason = data.make_backend(data.Settings())
    assert backend.name == "mock" and "B4L_USE_MOCK" in reason
    monkeypatch.setenv("B4L_USE_MOCK", "false")
    monkeypatch.delenv("DATABRICKS_WAREHOUSE_ID", raising=False)
    backend, _ = data.make_backend(data.Settings())
    assert backend.name == "mock"


def test_settings_reject_bad_identifiers():
    with pytest.raises(ValueError):
        _ = data.Settings(catalog="x; DROP TABLE y", schema_prefix="b4l").gold
    assert data.Settings(catalog="cat", schema_prefix="b4l").gold == "`cat`.`b4l_gold`"


def test_statement_to_frame_converts_types():
    cols = [
        SimpleNamespace(name="date", type_name=SimpleNamespace(value="DATE")),
        SimpleNamespace(name="n", type_name=SimpleNamespace(value="INT")),
        SimpleNamespace(name="x", type_name=SimpleNamespace(value="DOUBLE")),
        SimpleNamespace(name="flag", type_name=SimpleNamespace(value="BOOLEAN")),
        SimpleNamespace(name="ids", type_name=SimpleNamespace(value="ARRAY")),
    ]
    resp = SimpleNamespace(
        manifest=SimpleNamespace(schema=SimpleNamespace(columns=cols), total_chunk_count=1),
        result=SimpleNamespace(data_array=[["2026-11-23", "3", "0.5", "true", '["a","b"]'], [None, None, None, "false", None]]),
    )
    df = data.statement_to_frame(resp)
    assert df["date"].iloc[0] == date(2026, 11, 23)
    assert df["n"].iloc[0] == 3 and df["n"].isna().iloc[1]
    assert df["x"].iloc[0] == 0.5
    assert df["flag"].tolist() == [True, False]
    assert df["ids"].iloc[0] == ["a", "b"]


def test_parse_genie_message():
    msg = {
        "conversation_id": "c1",
        "message_id": "m1",
        "status": "COMPLETED",
        "attachments": [
            {"attachment_id": "a1", "query": {"query": "SELECT 1", "description": "One"}},
            {"text": {"content": "Le taux est de 64%."}},
            {"suggested_questions": {"questions": ["Et le soir ?"]}},
        ],
    }
    ans = genie_client.parse_message(msg)
    assert (ans.conversation_id, ans.message_id, ans.attachment_id) == ("c1", "m1", "a1")
    assert ans.sql == "SELECT 1" and ans.text == "Le taux est de 64%."
    assert ans.suggested_questions == ["Et le soir ?"]


def test_app_sample_questions_are_in_genie_space():
    from b4l import genie_space

    assert set(genie_client.SAMPLE_QUESTIONS) <= set(genie_space.SAMPLE_QUESTIONS)
