"""Feature builders for the Phase 2 models: guardrails, label definitions and no look-ahead."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from b4l import demand, generator, synth
from b4l import ml_features as mf
from b4l import pricing_rules as pr

SEED = 20261123
AS_OF = date(2026, 11, 23)


@pytest.fixture(scope="module")
def world():
    return synth.build_world(SEED, AS_OF)


@pytest.fixture(scope="module")
def replay(world):
    intents = demand.build_intents(world)
    rng = np.random.default_rng(3)
    prices = mf.exploration_price_grid(world, rng)
    res = demand.simulate(world, intents, prices, 1.0)
    return prices, res, mf.fill_training_rows(world, res, prices, rng, replay_id=0)


@pytest.fixture(scope="module")
def silver_like():
    bronze, _, _ = generator.generate(SEED, AS_OF)
    b = bronze["bookings"].rename(columns={"booker_player_id": "booker_uuid"})
    b["booking_date"] = b["start_ts"].dt.date
    parts = bronze["booking_participants"].rename(columns={"player_id": "player_uuid"})
    plays = parts.merge(b[b["status"] == "completed"][["booking_id", "booking_date"]], on="booking_id")
    players = bronze["players"].rename(columns={"player_id": "player_uuid"})
    return b, plays, players, bronze["weather_lausanne"]


def test_exploration_prices_respect_guardrails_and_share_the_hour(world):
    grid = mf.exploration_price_grid(world, np.random.default_rng(0))
    tiers = demand.tier_grid(world)
    for day, prices, day_tiers in zip(world.days, grid, tiers):
        for c, tier in enumerate(day_tiers):
            lo, hi = pr.price_bounds(tier)
            assert lo <= prices[c] <= hi
            if c + 1 < day.n_cells and day.cell_time(c).hour == day.cell_time(c + 1).hour and tier == day_tiers[c + 1]:
                assert prices[c] == prices[c + 1]


def test_fill_rows_are_well_formed(replay):
    prices, res, rows = replay
    assert set(mf.FILL_FEATURES + [mf.FILL_LABEL]) <= set(rows.columns)
    assert rows["lead_days"].between(0, mf.MAX_LEAD_DAYS).all()
    assert rows["booked_share_now"].between(0, 1).all() and rows["fill_rate"].between(0, 1).all()
    assert (rows["courts_available"] <= pr.N_COURTS).all()
    assert (rows["booked_courts_now"] <= rows["courts_available"]).all()
    assert len(mf.FILL_MONOTONE) == len(mf.FILL_FEATURES)


def test_fill_label_matches_bookings(world, replay):
    _, res, rows = replay
    b = res.bookings
    hist = b[(b["start_ts"] < pd.Timestamp(world.now)) & b["status"].isin(mf.BILLED)]
    one = rows.drop_duplicates(["day_idx", "start_cell"])
    assert int(one["final_booked_courts"].sum()) == len(hist)


def test_fill_rate_falls_with_price(replay):
    rows = replay[2]
    low = rows[rows["price_ratio"] < 0.9].groupby("tier")["fill_rate"].mean()
    high = rows[rows["price_ratio"] > 1.1].groupby("tier")["fill_rate"].mean()
    assert (low > high).all()


def test_hist_fill_uses_only_the_previous_four_weeks():
    d0 = date(2026, 9, 7)
    slots = pd.DataFrame({
        "slot_date": [d0 + timedelta(days=7 * i) for i in range(6)],
        "day_of_week": 1, "start_minute": 720, "fill_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })
    hist = mf.add_hist_fill(slots)["hist_fill_rate_4w"].tolist()
    assert np.isnan(hist[0])
    assert hist[1] == pytest.approx(0.1)
    assert hist[5] == pytest.approx(np.mean([0.2, 0.3, 0.4, 0.5]))


def test_churn_snapshot_matches_the_gold_rule(silver_like):
    b, plays, players, _ = silver_like
    snap = mf.churn_snapshot(plays, b, players, AS_OF, with_label=False)
    assert (snap["avg_monthly_plays_baseline"] >= 2).all()
    rule = snap["plays_last_30d"] <= 0.35 * snap["avg_monthly_plays_baseline"]
    assert 0.05 < rule.mean() < 0.25
    assert mf.CHURN_LABEL not in snap


def test_churn_label_is_the_rule_one_month_later(silver_like):
    b, plays, players, _ = silver_like
    t = AS_OF - timedelta(days=30)
    labelled = mf.churn_snapshot(plays, b, players, t).set_index("player_uuid")
    later = mf.churn_snapshot(plays, b, players, AS_OF, with_label=False).set_index("player_uuid")
    common = labelled.index.intersection(later.index)
    assert (labelled.loc[common, "plays_next_30d"] == later.loc[common, "plays_last_30d"]).all()


def test_churn_snapshot_dates_fit_the_window():
    start = AS_OF - timedelta(days=168)
    dates = mf.churn_snapshot_dates(AS_OF, start)
    assert dates and min(dates) >= start + timedelta(days=120) and max(dates) <= AS_OF - timedelta(days=30)


def test_cancel_rows_use_only_known_history(silver_like):
    b, _, players, weather = silver_like
    rows = mf.cancel_rows(b, players, weather)
    assert "cancelled_early" not in set(rows["status"])
    sample = rows[rows["booker_prior_bookings"] > 0].sample(50, random_state=1)
    for r in sample.itertuples():
        mine = rows[(rows["booker_uuid"] == r.booker_uuid) & (rows["status"] != "upcoming")]
        known = mine[mine["start_ts"] < r.booked_at]
        assert r.booker_prior_bookings == len(known)
        assert r.booker_prior_late_rate == pytest.approx(known[mf.CANCEL_LABEL].mean())
