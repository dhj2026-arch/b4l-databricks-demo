"""Lakebase helpers: court states, grants and SQL shape (the merge itself is exercised against Postgres in 04)."""

from datetime import datetime

import pandas as pd
import pytest

from b4l import lakebase as lb

T12, T1230, T13 = datetime(2026, 11, 24, 12), datetime(2026, 11, 24, 12, 30), datetime(2026, 11, 24, 13)
EMPTY = pd.DataFrame(columns=["court_id", "start_ts", "end_ts"])


def _states(bookings=EMPTY, blocks=EMPTY):
    slots = pd.DataFrame({"slot_key": ["2026-11-24T1200", "2026-11-24T1230"], "slot_start_ts": [T12, T1230]})
    courts = pd.DataFrame({"court_id": ["Court_01", "Court_02"], "hall_zone": ["Main Hall", "Main Hall"]})
    return lb.court_slot_states(slots, courts, bookings, blocks).set_index("slot_id")["court_state"]


def test_every_court_and_slot_is_open_without_bookings():
    s = _states()
    assert len(s) == 4 and (s == lb.OPEN).all()
    assert "2026-11-24T1200_Court_01" in s.index


def test_a_booking_covers_each_start_inside_it_but_not_its_end():
    s = _states(bookings=pd.DataFrame({"court_id": ["Court_01"], "start_ts": [T12], "end_ts": [T1230]}))
    assert s["2026-11-24T1200_Court_01"] == lb.BOOKED
    assert s["2026-11-24T1230_Court_01"] == lb.OPEN
    assert s["2026-11-24T1200_Court_02"] == lb.OPEN


def test_a_block_wins_over_a_booking():
    booking = pd.DataFrame({"court_id": ["Court_02"], "start_ts": [T12], "end_ts": [T13]})
    block = pd.DataFrame({"court_id": ["Court_02"], "start_ts": [T1230], "end_ts": [T13]})
    s = _states(booking, block)
    assert s["2026-11-24T1200_Court_02"] == lb.BOOKED
    assert s["2026-11-24T1230_Court_02"] == lb.BLOCKED


def test_grants_are_least_privilege():
    stmts = lb.grant_statements("b4l_ops", "7d1c34b3-f25f-40a4-bfea-2950bde54b1d")
    text = " ".join(stmts)
    assert "DELETE" not in text and "TRUNCATE" not in text and "ALL" not in text
    log = next(s for s in stmts if "agent_action_log" in s)
    assert "UPDATE" not in log
    slots = next(s for s in stmts if "live_court_slots" in s)
    assert "INSERT" not in slots and "base_price_chf" not in slots and "UPDATE (dynamic_price_chf" in slots


@pytest.mark.parametrize("role", ["x; DROP TABLE t", 'a"b', ""])
def test_grants_reject_unsafe_role_names(role):
    with pytest.raises(ValueError):
        lb.grant_statements("b4l_ops", role)


def test_schema_names_validate_prefix():
    assert lb.schema_names("b4l") == ("b4l_ops", "b4l_sync")
    with pytest.raises(ValueError):
        lb.schema_names("b4l; drop")


def test_ddl_guards_prices_and_statuses():
    live = next(s for s in lb.ddl("b4l_ops") if "live_court_slots (" in s)
    assert "BETWEEN 18.0 AND 42.0" in live
    for status in ("proposed", "approved", "overridden", "booked", "blocked", "expired"):
        assert f"'{status}'" in live


def test_merge_keeps_staff_decisions_only_on_open_courts():
    sql = lb.merge_sql("b4l_ops", "b4l_sync")
    assert "ON CONFLICT (slot_id) DO UPDATE" in sql
    assert sql.count("live.status IN ('approved', 'overridden') AND EXCLUDED.status = 'proposed'") == 3
    assert "JOIN b4l_sync.slot_price_recommendations" in sql
