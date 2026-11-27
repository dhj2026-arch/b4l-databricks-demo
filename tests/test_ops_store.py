"""Staff app price decisions: guardrails, court-state rules and the action log (mock store; same rules as Lakebase)."""

import sys
from datetime import date
from pathlib import Path

import pytest

from b4l import lakebase as lb

APP_DIR = Path(__file__).resolve().parents[1] / "apps" / "staff_ops_app"
sys.path.insert(0, str(APP_DIR))

import ops_store as ops  # noqa: E402

AS_OF = date(2026, 11, 23)
TUESDAY = date(2026, 11, 24)


@pytest.fixture()
def store():
    return ops.MockOpsStore(AS_OF)


def _first(store, status, day=TUESDAY):
    s = store.court_slots_for_day(day)
    return s[s["status"] == status].iloc[0]


def test_statuses_and_columns_match_the_postgres_layout():
    assert (ops.PROPOSED, ops.APPROVED, ops.OVERRIDDEN) == (lb.PROPOSED, lb.APPROVED, lb.OVERRIDDEN)
    assert (ops.BOOKED, ops.BLOCKED, ops.EXPIRED) == (lb.BOOKED, lb.BLOCKED, lb.EXPIRED)
    live = next(s for s in lb.ddl("b4l_ops") if "live_court_slots (" in s)
    for c in ops.COURT_SLOT_COLUMNS:
        assert f"{c} " in live


def test_mock_has_sixteen_courts_per_slot_and_open_courts_at_the_tariff(store):
    s = store.court_slots_for_day(TUESDAY)
    assert (s.groupby("slot_key").size() == 16).all()
    open_ = s[s["status"] == ops.PROPOSED]
    assert len(open_) and (open_["dynamic_price_chf"] == open_["base_price_chf"]).all()
    assert {ops.BOOKED, ops.BLOCKED} <= set(s["status"])


def test_approve_sets_the_ai_price_and_logs_it(store):
    row = _first(store, ops.PROPOSED)
    r = store.approve(row["slot_id"], "staff@b4l.ch")
    assert r["status"] == ops.APPROVED and r["dynamic_price_chf"] == row["recommended_price_chf"]
    after = store.court_slots(row["slot_key"]).set_index("slot_id").loc[row["slot_id"]]
    assert after["decided_by"] == "staff@b4l.ch"
    log = store.action_log()
    assert len(log) == 1
    e = log.iloc[0]
    assert (e["action_type"], e["court_id"], e["old_price_chf"], e["new_price_chf"]) == (
        ops.APPROVE, row["court_id"], row["base_price_chf"], row["recommended_price_chf"])


def test_override_within_guardrails_needs_a_reason(store):
    row = _first(store, ops.PROPOSED)
    lo, hi = ops.d.price_bounds(row["tier"])
    with pytest.raises(ops.DecisionError, match="reason"):
        store.override(row["slot_id"], lo, "  ", "staff@b4l.ch")
    r = store.override(row["slot_id"], hi, "Club tournament", "staff@b4l.ch")
    assert r["status"] == ops.OVERRIDDEN and r["dynamic_price_chf"] == hi
    assert store.action_log().iloc[0]["reason"] == "Club tournament"


@pytest.mark.parametrize("delta", [-1, 1])
def test_override_outside_the_tier_range_is_refused_and_not_logged(store, delta):
    row = _first(store, ops.PROPOSED)
    lo, hi = ops.d.price_bounds(row["tier"])
    with pytest.raises(ops.DecisionError, match="outside"):
        store.override(row["slot_id"], (lo if delta < 0 else hi) + delta, "test", "staff@b4l.ch")
    assert store.action_log().empty


def test_override_rejects_fractional_francs(store):
    row = _first(store, ops.PROPOSED)
    with pytest.raises(ops.DecisionError, match="whole francs"):
        store.override(row["slot_id"], row["base_price_chf"] + 0.5, "test", "staff@b4l.ch")


@pytest.mark.parametrize("status", [ops.BOOKED, ops.BLOCKED])
def test_booked_and_blocked_courts_cannot_be_repriced(store, status):
    row = _first(store, status)
    with pytest.raises(ops.DecisionError, match=status):
        store.approve(row["slot_id"], "staff@b4l.ch")
    assert store.action_log().empty


def test_a_decided_court_can_be_changed_again(store):
    row = _first(store, ops.PROPOSED)
    store.approve(row["slot_id"], "a@b4l.ch")
    store.override(row["slot_id"], row["base_price_chf"], "Back to tariff", "b@b4l.ch")
    log = store.action_log()
    assert list(log["actor"]) == ["b@b4l.ch", "a@b4l.ch"]
    assert log.iloc[0]["old_price_chf"] == row["recommended_price_chf"]


def test_summary_counts_courts_by_status(store):
    s = store.slot_summary(TUESDAY)
    assert (s[["proposed", "approved", "overridden", "booked", "blocked"]].sum(axis=1) == 16).all()
    assert s["slot_key"].is_monotonic_increasing


def test_slot_time_label():
    assert ops.slot_time("2026-11-24T1200") == "Tue 24 Nov 12:00"
