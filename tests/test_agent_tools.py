"""Staff action tools: Python rules, generated SQL, grants, the staff app's copy and its Lakebase actions."""

import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from b4l import agent_tools as src

APP_DIR = Path(__file__).resolve().parents[1] / "apps" / "staff_ops_app"
sys.path.insert(0, str(APP_DIR))

import agent_tools as app  # noqa: E402
import mock_data  # noqa: E402
import ops_store  # noqa: E402

AS_OF = date(2026, 11, 23)
SLOT = {"slot_key": "2026-11-24T1200", "tier": "Lunch", "base_price_chf": 32.0, "is_future": True,
        "booked_courts_now": 5, "courts_available": 16}
PLAYER = {"player_uuid": "p1", "preferred_lang": "FR", "skill_tier": "Intermediate", "preferred_day_of_week": 2,
          "preferred_hour": 18, "pass_product": "Pass_Chrono", "pass_hours_remaining": 6,
          "days_to_pass_expiry": 5, "pass_expires_at": pd.Timestamp("2026-11-28"), "churn_risk_rule": True}
WAIT = {"window_start_ts": pd.Timestamp("2026-11-24 12:00"), "window_end_ts": pd.Timestamp("2026-11-24 12:00")}


# ---------------------------------------------------------------- guardrails

@pytest.mark.parametrize("price, rule, final", [
    (10.0, "floor", 20.0), (20.4, "within_bounds", 20.0), (30.5, "within_bounds", 30.0), (31.5, "within_bounds", 32.0),
    (50.0, "ceiling", 42.0), (None, "no_price", 32.0)])
def test_guardrail_rules(price, rule, final):
    g = src.guardrail(SLOT, price)
    assert (g["applied_rule"], g["final_price_chf"]) == (rule, final)


def test_past_and_fully_booked_slots_keep_the_tariff():
    assert src.guardrail({**SLOT, "is_future": False}, 20.0)["applied_rule"] == "past_slot"
    full = src.guardrail({**SLOT, "booked_courts_now": 16}, 20.0)
    assert (full["applied_rule"], full["final_price_chf"]) == ("fully_booked", 32.0)
    assert src.guardrail({**SLOT, "booked_courts_now": None}, 30.0)["applied_rule"] == "within_bounds"
    assert src.guardrail(None, 30.0) is None


# ---------------------------------------------------------------- concierge

def test_waitlist_message_uses_single_time_or_window():
    fr = src.concierge_message(PLAYER, "waitlist_backfill", AS_OF, WAIT)["message"]
    assert "mar. 24 nov. à 12:00" in fr and fr.endswith(src.SIGNATURE)
    window = {**WAIT, "window_end_ts": pd.Timestamp("2026-11-24 13:00")}
    en = src.concierge_message({**PLAYER, "preferred_lang": "EN"}, "waitlist_backfill", AS_OF, window)["message"]
    assert "Tue 24 Nov starting between 12:00 and 13:00" in en
    assert src.concierge_message(PLAYER, "waitlist_backfill", AS_OF, None)["eligible"] is False


def test_concierge_eligibility_rules():
    m = lambda p, t: src.concierge_message({**PLAYER, **p}, t, AS_OF)  # noqa: E731
    assert m({}, "doubles_match")["eligible"]
    assert m({"preferred_hour": None}, "doubles_match")["note"] == "No usual playing time yet"
    assert m({}, "pass_expiry")["eligible"]
    assert not m({"days_to_pass_expiry": 40}, "pass_expiry")["eligible"]
    assert not m({"pass_hours_remaining": 0}, "pass_expiry")["eligible"]
    assert not m({"pass_product": "Pass_Liberte"}, "pass_expiry")["eligible"]
    assert "23 déc. 2026" in m({}, "retention_voucher")["message"]
    assert not m({"churn_risk_rule": False}, "retention_voucher")["eligible"]
    assert m({}, " Retention_Voucher ")["trigger_type"] == "retention_voucher"
    assert m({}, "spam")["note"].startswith("Unknown trigger")


# ---------------------------------------------------------------- partners

def test_doubles_partners_same_level_close_hour_active_and_ranked():
    players = mock_data.player_features(AS_OF)
    me = players.iloc[0]
    out = src.doubles_partners(players, me["player_uuid"])
    assert 0 < len(out) <= src.MAX_PARTNERS and list(out.columns) == src.PARTNER_COLUMNS
    assert (out["skill_tier"] == me["skill_tier"]).all() and me["player_uuid"] not in set(out["partner_uuid"])
    assert ((out["preferred_hour"] - me["preferred_hour"]).abs() <= src.PARTNER_HOUR_WINDOW).all()
    assert (out["plays_last_30d"] > 0).all()
    assert out["match_score"].is_monotonic_decreasing
    assert src.doubles_partners(players, "nobody").empty


# ---------------------------------------------------------------- SQL and grants

def test_function_sql_names_and_escaping():
    sql = src.all_function_sql("cat", "b4l")
    assert set(sql) == set(src.FUNCTIONS)
    for name, stmt in sql.items():
        assert stmt.startswith(f"CREATE OR REPLACE FUNCTION `cat`.`b4l_gold`.{name}(")
    assert "''" not in sql[src.CONCIERGE_FN]
    assert "\\'" in sql[src.CONCIERGE_FN]
    assert "`cat`.`b4l_silver`.waitlist" in sql[src.WAITLIST_FN]
    with pytest.raises(ValueError):
        src.all_function_sql("cat; DROP", "b4l")


def test_grants_are_execute_only():
    g = src.grant_statements("cat", "b4l", "7d1c34b3-f25f")
    assert len(g) == len(src.FUNCTIONS) and all(re.match(r"GRANT EXECUTE ON FUNCTION ", s) for s in g)
    with pytest.raises(ValueError):
        src.grant_statements("cat", "b4l", "x`; DROP")


# ---------------------------------------------------------------- app copy matches src

def test_app_constants_match():
    for name in ("TRIGGERS", "GUARDRAIL_COLUMNS", "CONCIERGE_COLUMNS", "PARTNER_COLUMNS", "WAITLIST_COLUMNS",
                 "VOUCHER_CHF", "PASS_REMINDER_DAYS", "MAX_PARTNERS", "PARTNER_HOUR_WINDOW", "SIGNATURE"):
        assert getattr(app, name) == getattr(src, name), name
    assert set(app.TRIGGER_LABELS) == set(src.TRIGGERS)


def test_app_guardrail_matches_src():
    for slot in (SLOT, {**SLOT, "tier": "Day", "base_price_chf": 25.0}, {**SLOT, "is_future": False},
                 {**SLOT, "booked_courts_now": 16}, {**SLOT, "booked_courts_now": None}):
        for price in (None, 5.0, 17.5, 18.5, 24.5, 25.5, 33.4, 41.6, 60.0):
            assert app.guardrail(slot, price) == src.guardrail(slot, price), (slot, price)


def test_app_concierge_matches_src():
    players = mock_data.player_features(AS_OF).head(200).to_dict("records")
    for p in players + [{**PLAYER, "preferred_lang": "EN"}, {**PLAYER, "preferred_day_of_week": None}]:
        for t in (*src.TRIGGERS, "bad"):
            for w in (None, WAIT):
                assert app.concierge_message(p, t, AS_OF, w) == src.concierge_message(p, t, AS_OF, w)


def test_app_partners_match_src():
    players = mock_data.player_features(AS_OF)
    for u in players["player_uuid"].head(30):
        pd.testing.assert_frame_equal(app.doubles_partners(players, u), src.doubles_partners(players, u))


def test_mock_waitlist_shape():
    slots = mock_data.slot_features(date(2026, 11, 24), AS_OF)
    wl = app.mock_open_waitlist(mock_data.player_features(AS_OF), slots)
    assert list(wl.columns) == app.WAITLIST_COLUMNS and len(wl) == int(slots["waitlist_requests"].sum())


# ---------------------------------------------------------------- Lakebase actions (mock store)

def test_concierge_action_is_logged_and_checked():
    store = ops_store.MockOpsStore(AS_OF)
    draft = src.concierge_message(PLAYER, "retention_voucher", AS_OF)
    with pytest.raises(ops_store.DecisionError):
        store.send_concierge({**draft, "eligible": False, "note": "nope"}, "x", "a@b.ch")
    with pytest.raises(ops_store.DecisionError):
        store.send_concierge(draft, "  ", "a@b.ch")
    details = store.send_concierge(draft, draft["message"] + " Merci", "a@b.ch")
    assert details["edited"] is True
    log = store.action_log()
    assert log.iloc[0][["actor", "action_type", "status"]].tolist() == [
        "a@b.ch", ops_store.CONCIERGE, ops_store.SENT_DEMO]
    assert log.iloc[0]["summary"] == "retention_voucher to p1… (fr)"


PARTNERS = pd.DataFrame({"partner_uuid": ["p2", "p3", "p4"], "preferred_lang": ["EN", "FR", "FR"]})
START = datetime(2026, 11, 24, 12)


def courts(rows):
    cols = ["slot_id", "slot_key", "court_id", "hall_zone", "tier", "base_price_chf", "recommended_price_chf",
            "dynamic_price_chf", "status"]
    return pd.DataFrame([(f"{k}_{c}", k, c, "Main Hall", "Lunch", 32.0, rec, live, st)
                         for k, c, rec, live, st in rows], columns=cols)


def day(court, start_hhmm, n, rec=32.0, live=32.0, status="proposed", date_="2026-11-24"):
    """n consecutive half-hour rows on one court."""
    t0 = datetime.strptime(f"{date_}T{start_hhmm}", "%Y-%m-%dT%H%M")
    return [(app.slot_key(t0 + pd.Timedelta(minutes=30 * i)), court, rec, live, status) for i in range(n)]


def test_court_game_needs_every_half_hour_open_and_prices_hour_blocks():
    rows = day("Court_01", "1200", 4, rec=26.0, live=30.0)
    rows[0] = (*rows[0][:3], 26.0, "approved")                    # 12:00 approved at 26; 13:00 still 30, AI 26
    idx = app.court_index(courts(rows))
    g = app.court_game(idx, "Court_01", START, 120)
    assert (g["price_chf"], g["tariff_chf"], g["ai_price_chf"]) == (56.0, 64.0, 52.0)
    assert g["to_approve"] == ["2026-11-24T1300_Court_01"]
    assert app.court_game(idx, "Court_01", START, 90)["price_chf"] == 26.0 + 15.0
    assert app.court_game(idx, "Court_01", START, 180) is None    # 14:00 is outside the grid
    booked = rows[:]
    booked[1] = (*booked[1][:4], "booked")                        # someone booked 12:30
    assert app.court_game(app.court_index(courts(booked)), "Court_01", START, 120) is None
    assert app.court_game(app.court_index(courts(booked)), "Court_01", START, 60) is None


def test_lobby_slot_options_underfilled_first_then_usual_day():
    c = courts(day("Court_01", "1200", 4, rec=26.0, live=26.0, status="approved")   # Tue, usual day, discount
               + day("Court_02", "1200", 4)
               + day("Court_03", "1200", 1, status="booked")
               + day("Court_01", "1300", 4, date_="2026-11-25")                    # Wed, under-filled
               + day("Court_01", "1700", 4)                                        # 5 h from usual hour
               + day("Court_05", "1200", 4, status="blocked", date_="2026-11-26"))
    feats = pd.DataFrame({"slot_key": ["2026-11-24T1200", "2026-11-25T1300"], "booked_share": [0.5, 0.1],
                          "hist_fill_rate_4w": [0.5, 0.6], "underfilled": [False, True]})
    o = app.lobby_slot_options(c, feats, 2, 12, 120)
    assert o["slot_key"].tolist()[:2] == ["2026-11-25T1300", "2026-11-24T1200"]
    assert "2026-11-24T1700" not in o["slot_key"].tolist() and not o["slot_key"].str.startswith("2026-11-26").any()
    tue = o.iloc[1]
    assert (tue["open_courts"], tue["discounted_courts"], tue["best_court_id"], tue["best_price_chf"],
            tue["tariff_chf"], bool(tue["usual_day"])) == (2, 1, "Court_01", 52.0, 64.0, True)
    assert tue["end"] == datetime(2026, 11, 24, 14)
    assert "2026-11-24T1300" not in o["slot_key"].tolist()        # only 1 h left on the grid for a 2 h game
    assert "2026-11-24T1300" in app.lobby_slot_options(c, feats, 2, 12, 60)["slot_key"].tolist()
    assert app.lobby_slot_options(c, pd.DataFrame(), None, 12)["underfilled"].eq(False).all()


def test_price_status_and_labels():
    assert app.price_status(52, 64, 52) == "approved discount"
    assert app.price_status(56, 64, 52).startswith("partly discounted · AI suggests CHF 52")
    assert app.price_status(64, 64, 52).startswith("tariff · AI suggests CHF 52")
    assert app.price_status(64, 64, 64) == "tariff"
    assert [app.duration_label(m, True) for m in (60, 90, 120)] == ["1 h", "1 h 30", "2 h"]
    assert [app.duration_label(m, False) for m in (60, 90, 120)] == ["1 hour", "1.5 hours", "2 hours"]
    assert (app.chf_amount(41.0), app.chf_amount(40.5)) == ("41", "40.50")


def test_lobby_invitations_quote_court_time_and_game_price_in_their_language():
    inv = app.lobby_invitations(PLAYER, PARTNERS, ["p2", "p3"], START, 120, "Court_05", 52.0, 64.0)
    assert inv[["player_uuid", "role", "language"]].values.tolist() == [
        ["p1", app.ORGANISER, "fr"], ["p2", app.PARTNER, "en"], ["p3", app.PARTNER, "fr"]]
    assert inv.at[0, "message"].startswith(
        "Bonjour, votre double est ouvert : mar. 24 nov. de 12:00 à 14:00 sur le terrain 5, à CHF 52 pour 2 h "
        "au lieu de CHF 64 (CHF 13.00 par joueur), niveau intermédiaire. Nous avons invité 2 partenaires")
    assert ("Tue 24 Nov from 12:00 to 14:00 on Court 5, at CHF 52 for 2 hours instead of CHF 64 (CHF 13.00 each)"
            in inv.at[1, "message"])
    assert all(m.endswith(app.SIGNATURE) for m in inv["message"])
    tariff = app.lobby_invitations(PLAYER, PARTNERS, ["p3"], START, 90, "Court_05", 48.0, 48.0).at[0, "message"]
    assert "de 12:00 à 13:30" in tariff and "pour 1 h 30 (CHF 12.00" in tariff and "au lieu de" not in tariff
    assert "1 partenaire et" in tariff


def test_lobby_and_invitations_are_logged_together_at_the_quoted_price():
    store = ops_store.MockOpsStore(AS_OF)
    ahead = store.court_slots_between(date(2026, 11, 24), date(2026, 12, 1))
    opts = app.lobby_slot_options(ahead, pd.DataFrame(), None, 12, 120)
    o = opts.iloc[0]
    g = app.court_games(app.court_index(ahead), o["start"].to_pydatetime(), 120).iloc[0]
    game = {"court_id": g["court_id"], "slot_key": o["slot_key"], "minutes": 120, "price_chf": g["price_chf"],
            "tariff_chf": g["tariff_chf"]}
    inv = app.lobby_invitations(PLAYER, PARTNERS, ["p2", "p3"], o["start"], 120, g["court_id"], g["price_chf"],
                                g["tariff_chf"]).to_dict("records")
    for invited, invites in (([], inv[:1]), (["p1"], inv), (["p2", "p3", "p4", "p5"], inv), (["p2"], inv),
                             (["p2", "p3"], [*inv[:2], {**inv[2], "message": " "}])):
        with pytest.raises(ops_store.DecisionError):
            store.create_lobby("p1", "Intermediate", game, invited, invites, "a@b.ch")
    with pytest.raises(ops_store.DecisionError, match="Game length"):
        store.create_lobby("p1", "Intermediate", {**game, "minutes": 45}, ["p2", "p3"], inv, "a@b.ch")
    with pytest.raises(ops_store.DecisionError, match="price"):
        store.create_lobby("p1", "Intermediate", {**game, "price_chf": g["price_chf"] - 1}, ["p2", "p3"], inv, "a@b.ch")
    assert store.lobbies().empty and store.action_log().empty
    r = store.create_lobby("p1", "Intermediate", game, ["p2", "p3"], inv, "a@b.ch")
    lob = store.lobbies()
    assert lob.iloc[0][["lobby_id", "duration_minutes", "court_id", "spots_open", "player_uuids"]].tolist() == [
        r["lobby_id"], 120, g["court_id"], 3, ["p1"]]
    assert lob.iloc[0]["slot_start"] == o["start"]
    log = store.action_log()
    assert log["action_type"].value_counts().to_dict() == {ops_store.INVITE: 3, ops_store.LOBBY: 1}
    lobby_row = log.loc[log["action_type"] == ops_store.LOBBY, "summary"].iloc[0]
    assert f"2 hours {g['court_id']} CHF {app.chf_amount(g['price_chf'])}" in lobby_row and "2 invited" in lobby_row
    assert (log.loc[log["action_type"] == ops_store.INVITE, "status"] == ops_store.SENT_DEMO).all()
    assert "Lobby invitation to p2… (partner, en)" in log["summary"].tolist()
    later = f"{app.slot_key(o['start'] + pd.Timedelta(minutes=90))}_{g['court_id']}"
    store._slots.at[later, "status"] = ops_store.BOOKED                # the last half hour gets booked
    with pytest.raises(ops_store.DecisionError, match="whole game"):
        store.create_lobby("p1", "Intermediate", game, ["p2", "p3"], inv, "a@b.ch")
