"""Static vs. dynamic pricing backtest (SPEC.md 5.F), shared by 03c and the tests.

The static world is the generated history at the published tariff. The dynamic world replays the same 24 weeks
through the simulator with the same players, intents, weather and random draws; only the prices differ:

- Each slot start gets the price from the same search as 03b (scoring.recommend_prices), posted ex ante: two weeks
  ahead with nothing booked yet, using only what was known then (calendar, weather, the last 4 weeks' fill).
- Pass-hour and family-rate bookings, courses and club blocks are never re-priced (the simulator charges them the
  tariff whatever the grid says), and flexible players can move between slots, so gains are net of cannibalisation.
- The replay runs at three price sensitivities (low, mid, high); results are reported as a range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from b4l import demand, scoring
from b4l import ml_features as mf
from b4l import pricing_rules as pr
from b4l.synth import World

BILLED = mf.BILLED


def history_slots(world: World, static: demand.SimResult) -> pd.DataFrame:
    """Fill-model features for every historical slot start, as known two weeks ahead (nothing booked yet)."""
    slots = mf.slot_grid(world)
    b = static.bookings
    b = b[(b["start_ts"] < pd.Timestamp(world.now)) & b["status"].isin(BILLED)]
    key = ["day_idx", "start_cell"]
    final = b.groupby(key).size().rename("final_booked_courts")
    slots = slots.join(final, on=key).fillna({"final_booked_courts": 0})
    slots["fill_rate"] = (slots["final_booked_courts"] / slots["courts_available"].clip(lower=1)).clip(upper=1.0)
    slots = mf.add_hist_fill(slots)
    slots["slot_key"] = [pr.slot_key(d, t.time()) for d, t in zip(slots["slot_date"], slots["slot_start_ts"])]
    slots["lead_days"] = mf.MAX_LEAD_DAYS
    slots["booked_courts_now"] = 0
    slots["booked_share_now"] = 0.0
    slots["price_chf"] = slots["base_price_chf"]
    slots["price_ratio"] = 1.0
    return slots


def dynamic_price_grid(world: World, decisions: pd.DataFrame) -> list[np.ndarray]:
    """Static tariff grid with each recommended slot-start price written into its 30-minute cell."""
    grid = [p.copy() for p in demand.static_price_grid(world)]
    for d, c, p in zip(decisions["day_idx"], decisions["start_cell"], decisions["recommended_price_chf"]):
        grid[int(d)][int(c)] = float(p)
    return grid


def _available_hours(world: World) -> pd.DataFrame:
    """Rentable court-hours (courts not blocked) per history day and tier."""
    tiers = demand.tier_grid(world)
    blocked: dict[int, np.ndarray] = {}
    for b in world.blocks.itertuples():
        day = world.days[b.day_idx]
        c0 = (b.start_ts.hour * 60 + b.start_ts.minute - day.open_min) // pr.SLOT_MINUTES
        c1 = (b.end_ts.hour * 60 + b.end_ts.minute - day.open_min) // pr.SLOT_MINUTES
        arr = blocked.setdefault(b.day_idx, np.zeros((pr.N_COURTS, day.n_cells), dtype=bool))
        arr[pr.COURT_IDS.index(b.court_id), max(c0, 0):max(c1, 0)] = True
    rows = []
    for day in world.days:
        if day.n_cells == 0 or day.d >= world.as_of:
            continue
        free = pr.N_COURTS - (blocked[day.idx].sum(axis=0) if day.idx in blocked else np.zeros(day.n_cells))
        for c, tier in enumerate(tiers[day.idx]):
            rows.append({"date": day.d, "tier": tier, "available_hours": free[c] * pr.SLOT_MINUTES / 60})
    return pd.DataFrame(rows).groupby(["date", "tier"], as_index=False)["available_hours"].sum()


def _week(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return (d - pd.to_timedelta(d.dt.dayofweek, unit="D")).dt.date


def weekly_outcomes(world: World, res: demand.SimResult) -> pd.DataFrame:
    """Revenue, booked court-hours and turned-away requests per week and tier over the history."""
    b = res.bookings
    b = b[(b["start_ts"] < pd.Timestamp(world.now)) & b["status"].isin(BILLED)].copy()
    b["week_start"] = _week(b["start_ts"].dt.date)
    out = b.groupby(["week_start", "tier"]).agg(revenue_chf=("revenue_chf", "sum"),
                                                booked_hours=("duration_h", "sum")).reset_index()
    tiers = demand.tier_grid(world)
    ta = res.turned_away
    ta = ta[[world.days[d].d < world.as_of for d in ta["day_idx"]]].copy()
    ta["tier"] = [tiers[d][c] for d, c in zip(ta["day_idx"], ta["start_cell"])]
    ta["week_start"] = _week(pd.Series([world.days[d].d for d in ta["day_idx"]], index=ta.index))
    ta = ta.groupby(["week_start", "tier"], as_index=False)["turned_away"].sum()
    out = out.merge(ta, on=["week_start", "tier"], how="outer").fillna({"revenue_chf": 0, "booked_hours": 0,
                                                                       "turned_away": 0})
    avail = _available_hours(world)
    avail["week_start"] = _week(avail["date"])
    avail = avail.groupby(["week_start", "tier"], as_index=False)["available_hours"].sum()
    out = out.merge(avail, on=["week_start", "tier"], how="left")
    out["occupancy"] = out["booked_hours"] / out["available_hours"]
    return out


def compare(world: World, static: demand.SimResult, dynamic: dict[str, demand.SimResult],
            decisions: pd.DataFrame) -> pd.DataFrame:
    """gold.static_vs_dynamic_revenue: one row per week, tier and elasticity scenario."""
    s = weekly_outcomes(world, static).rename(columns={
        "revenue_chf": "static_revenue_chf", "booked_hours": "static_booked_hours",
        "occupancy": "static_occupancy", "turned_away": "turned_away_static"})
    d = decisions.copy()
    d["week_start"] = _week(pd.Series(d["slot_date"].to_numpy()))
    d["repriced"] = d["recommended_price_chf"] != d["base_price_chf"]
    d["abs_change"] = d["price_change_pct"].abs()
    prices = d.groupby(["week_start", "tier"]).agg(
        slots=("repriced", "size"), slots_repriced=("repriced", "sum"),
        avg_price_change_pct=("price_change_pct", "mean"), max_price_change_pct=("abs_change", "max"),
        avg_dynamic_price_chf=("recommended_price_chf", "mean")).reset_index()
    parts = []
    for scenario, res in dynamic.items():
        x = weekly_outcomes(world, res).rename(columns={
            "revenue_chf": "dynamic_revenue_chf", "booked_hours": "dynamic_booked_hours",
            "occupancy": "dynamic_occupancy", "turned_away": "turned_away_dynamic"}).drop(columns="available_hours")
        m = s.merge(x, on=["week_start", "tier"], how="outer").merge(prices, on=["week_start", "tier"], how="left")
        m["elasticity_scenario"] = scenario
        m["elasticity_k"] = demand.ELASTICITY_SCENARIOS[scenario]
        parts.append(m)
    out = pd.concat(parts, ignore_index=True)
    out["revenue_change_chf"] = out["dynamic_revenue_chf"] - out["static_revenue_chf"]
    out["revenue_change_pct"] = out["revenue_change_chf"] / out["static_revenue_chf"]
    return out.sort_values(["elasticity_scenario", "week_start", "tier"]).reset_index(drop=True)


def demand_curves(slots: pd.DataFrame, recs: pd.DataFrame, predict_fill) -> pd.DataFrame:
    """gold.slot_demand_curves: expected courts and revenue at each candidate price for upcoming slots.

    Courts already booked stay at their locked tariff; the candidate price applies to the courts still to be sold.
    """
    curves = scoring.price_curves(slots.reset_index(drop=True).assign(slot_idx=lambda d: d.index), predict_fill)
    curves["expected_courts_booked"] = np.maximum(curves["expected_courts"], curves["booked_courts_now"])
    curves["expected_revenue_chf"] = (curves["booked_courts_now"] * curves["base_price_chf"]
                                      + curves["expected_new_revenue_chf"])
    rec = recs.set_index("slot_key")["recommended_price_chf"]
    curves["is_tariff"] = curves["candidate_price_chf"] == curves["base_price_chf"]
    curves["is_recommended"] = curves["candidate_price_chf"] == curves["slot_key"].map(rec)
    curves["expected_occupancy"] = curves["expected_courts_booked"] / curves["courts_available"].clip(lower=1)
    return curves


def headline(compare_df: pd.DataFrame) -> pd.DataFrame:
    """Per scenario: weekly revenue change, off-peak occupancy change (points), share of slots re-priced."""
    weeks = compare_df["week_start"].nunique()
    rows = []
    for scenario, g in compare_df.groupby("elasticity_scenario"):
        off = g[g["tier"].isin([pr.DAY, pr.LUNCH])]
        one = g.drop_duplicates(["week_start", "tier"])
        rows.append({
            "elasticity_scenario": scenario,
            "static_revenue_per_week_chf": g["static_revenue_chf"].sum() / weeks,
            "dynamic_revenue_per_week_chf": g["dynamic_revenue_chf"].sum() / weeks,
            "revenue_change_per_week_chf": g["revenue_change_chf"].sum() / weeks,
            "revenue_change_pct": g["revenue_change_chf"].sum() / g["static_revenue_chf"].sum(),
            "offpeak_occupancy_change_pts": 100 * (off["dynamic_booked_hours"].sum() - off["static_booked_hours"].sum())
                                             / off["available_hours"].sum(),
            "share_slots_repriced": one["slots_repriced"].sum() / one["slots"].sum(),
        })
    return pd.DataFrame(rows)
