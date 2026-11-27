"""Static vs. dynamic backtest (SPEC.md 5.F tests), with stub fill models in place of LightGBM."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from b4l import backtest, demand, scoring, synth
from b4l import pricing_rules as pr

SEED = 20261123
AS_OF = date(2026, 11, 23)


def stub(elasticity: float):
    return lambda X: np.clip(X["hist_fill_rate_4w"].fillna(0.3) * X["price_ratio"] ** -elasticity, 0, 1).to_numpy()


@pytest.fixture(scope="module")
def setup():
    world = synth.build_world(SEED, AS_OF)
    intents = demand.build_intents(world)
    static = demand.simulate(world, intents)
    return world, intents, static, backtest.history_slots(world, static)


def test_tariff_prices_reproduce_the_static_world(setup):
    world, intents, static, slots = setup
    decisions = scoring.recommend_prices(slots, stub(1.0))
    assert (decisions["applied_rule"] == "keep_tariff").all()
    grid = backtest.dynamic_price_grid(world, decisions)
    for g, s in zip(grid, demand.static_price_grid(world)):
        np.testing.assert_array_equal(g, s)
    dyn = demand.simulate(world, intents, grid, 1.0)
    pd.testing.assert_frame_equal(dyn.bookings, static.bookings)
    cmp = backtest.compare(world, static, {"mid": dyn}, decisions)
    assert cmp["revenue_change_chf"].abs().max() == 0


def test_same_prices_give_an_identical_dynamic_world(setup):
    world, intents, _, slots = setup
    grid = backtest.dynamic_price_grid(world, scoring.recommend_prices(slots, stub(0.4)))
    a = demand.simulate(world, intents, grid, 1.0)
    b = demand.simulate(world, intents, grid, 1.0)
    pd.testing.assert_frame_equal(a.bookings, b.bookings)


@pytest.fixture(scope="module")
def repriced(setup):
    world, intents, static, slots = setup
    decisions = scoring.recommend_prices(slots, stub(0.4))
    grid = backtest.dynamic_price_grid(world, decisions)
    return decisions, grid, demand.simulate(world, intents, grid, 1.0)


def test_dynamic_prices_respect_guardrails(repriced):
    decisions, _, res = repriced
    for tier, g in decisions.groupby("tier"):
        lo, hi = pr.price_bounds(tier)
        assert g["recommended_price_chf"].between(lo, hi).all()
    b = res.bookings
    std = b[(b["rate_category"] == pr.STANDARD) & (b["payment_method"] != "pass")]
    assert (std["price_shown_chf"] >= pr.PRICE_FLOOR_CHF).all() and (std["price_shown_chf"] <= pr.PRICE_CEILING_CHF).all()


def test_protected_bookings_are_never_repriced(repriced):
    _, _, res = repriced
    b = res.bookings
    protected = b[(b["rate_category"] != pr.STANDARD) | (b["payment_method"] == "pass")]
    assert len(protected) > 0
    assert (protected["price_shown_chf"] == protected["base_price_chf"]).all()


def test_compare_covers_every_week_tier_and_scenario(setup, repriced):
    world, _, static, _ = setup
    decisions, _, res = repriced
    cmp = backtest.compare(world, static, {"low": res, "mid": res}, decisions)
    assert cmp["week_start"].nunique() == 24
    assert set(cmp["tier"]) == set(pr.TIERS)
    assert not cmp[["static_revenue_chf", "dynamic_revenue_chf", "static_occupancy", "dynamic_occupancy"]].isna().any().any()
    assert cmp["static_occupancy"].between(0, 1).all() and cmp["dynamic_occupancy"].between(0, 1).all()
    head = backtest.headline(cmp)
    assert set(head["elasticity_scenario"]) == {"low", "mid"}
