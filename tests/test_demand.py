"""Simulator properties the pricing backtest (SPEC.md 5.F) relies on."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from b4l import demand, generator, synth
from b4l import pricing_rules as pr

SEED = 20261123
AS_OF = date(2026, 11, 23)
ELASTICITY = {"low": 1.5, "mid": 1.0, "high": 0.67}


@pytest.fixture(scope="module")
def sim():
    world = synth.build_world(SEED, AS_OF)
    intents = demand.build_intents(world)
    return world, intents, demand.simulate(world, intents)


def _billed_hours(res: demand.SimResult, world, tier: str) -> float:
    b = res.bookings
    hist = b[(b["start_ts"] < pd.Timestamp(world.now)) & b["status"].isin(["completed", "cancelled_late", "no_show"])]
    return float(hist.loc[hist["tier"] == tier, "duration_h"].sum())


def _scaled_grid(world, tier: str, factor: float) -> list[np.ndarray]:
    grid = demand.static_price_grid(world)
    tiers = demand.tier_grid(world)
    out = []
    for prices, day_tiers in zip(grid, tiers):
        p = prices.copy()
        for c, t in enumerate(day_tiers):
            if t == tier:
                p[c] = float(np.clip(round(p[c] * factor), pr.PRICE_FLOOR_CHF, pr.PRICE_CEILING_CHF))
        out.append(p)
    return out


def test_same_seed_same_world_and_bookings(sim):
    world, intents, res = sim
    world2 = synth.build_world(SEED, AS_OF)
    intents2 = demand.build_intents(world2)
    pd.testing.assert_frame_equal(world.players, world2.players)
    pd.testing.assert_frame_equal(intents.df, intents2.df)
    pd.testing.assert_frame_equal(res.bookings, demand.simulate(world2, intents2).bookings)


@pytest.mark.parametrize("k", ELASTICITY.values(), ids=ELASTICITY.keys())
def test_static_replay_reproduces_history(sim, k):
    world, intents, res = sim
    replay = demand.simulate(world, intents, prices=demand.static_price_grid(world), elasticity_k=k)
    pd.testing.assert_frame_equal(res.bookings, replay.bookings)
    pd.testing.assert_frame_equal(res.turned_away, replay.turned_away)


def test_higher_evening_price_loses_more_hours_when_more_elastic(sim):
    world, intents, res = sim
    grid = _scaled_grid(world, pr.EVENING, 1.2)
    base = _billed_hours(res, world, pr.EVENING)
    low = _billed_hours(demand.simulate(world, intents, grid, ELASTICITY["low"]), world, pr.EVENING)
    high = _billed_hours(demand.simulate(world, intents, grid, ELASTICITY["high"]), world, pr.EVENING)
    assert high < low <= base
    # Evenings are near full: a 20% rise loses some hours, not most of them.
    assert high > 0.6 * base


def test_lower_day_price_fills_more_off_peak_hours(sim):
    world, intents, res = sim
    grid = _scaled_grid(world, pr.DAY, 0.8)
    base = _billed_hours(res, world, pr.DAY)
    cheaper = _billed_hours(demand.simulate(world, intents, grid, ELASTICITY["mid"]), world, pr.DAY)
    assert cheaper > base


def test_generated_bronze_passes_hard_rules():
    bronze, _, _ = generator.generate(SEED, AS_OF)
    report = bronze["data_quality_report"]
    failed = report[(report["category"] == "hard_rule") & ~report["passed"]]
    assert failed.empty, failed.to_dict("records")
    assert sorted(bronze["demo_moments"]["moment_id"]) == [1, 2, 3, 4, 5]
