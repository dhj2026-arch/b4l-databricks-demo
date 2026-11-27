"""Price recommendation, risk bands and waitlist matching, with a stub fill model in place of LightGBM."""

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from b4l import ml_features as mf
from b4l import pricing_rules as pr
from b4l import scoring

AS_OF = date(2026, 11, 23)


def slot(tier: str, courts_available: int = 16, booked: int = 2, hist: float = 0.5, **kw) -> dict:
    base = pr.TIER_PRICE_CHF[tier]
    row = {
        "slot_key": f"k{len(kw)}", "slot_date": AS_OF, "slot_start_ts": datetime(2026, 11, 24, 12), "day_of_week": 2,
        "start_hour": 12.0, "is_weekend_day": 0, "tier": tier, "tier_code": mf.TIER_CODE[tier],
        "base_price_chf": base, "is_school_holiday": 0, "is_public_holiday": 0, "arena_event": 0, "is_rainy": 0,
        "temp_mean_c": 8.0, "courts_available": courts_available, "hist_fill_rate_4w": hist, "lead_days": 1,
        "booked_courts_now": booked, "booked_share_now": booked / max(courts_available, 1),
        "price_chf": base, "price_ratio": 1.0,
    }
    row.update(kw)
    return row


def stub(elasticity: float):
    """Constant-elasticity demand: fill = hist * (price / tariff) ** -elasticity."""
    return lambda X: np.clip(X["hist_fill_rate_4w"] * X["price_ratio"] ** -elasticity, 0, 1).to_numpy()


def test_inelastic_demand_raises_price_to_the_ceiling():
    out = scoring.recommend_prices(pd.DataFrame([slot(pr.LUNCH)]), stub(0.3)).iloc[0]
    assert out["recommended_price_chf"] == pr.price_bounds(pr.LUNCH)[1]
    assert out["applied_rule"] == "within_bounds"
    assert out["expected_revenue_uplift_chf"] > 0


def test_elastic_demand_with_free_courts_lowers_price():
    out = scoring.recommend_prices(pd.DataFrame([slot(pr.DAY, booked=0, hist=0.3)]), stub(2.5)).iloc[0]
    assert out["recommended_price_chf"] < pr.TIER_PRICE_CHF[pr.DAY]
    assert out["price_change_pct"] < 0


def test_unit_elasticity_keeps_the_tariff():
    out = scoring.recommend_prices(pd.DataFrame([slot(pr.EVENING, booked=0, hist=0.4)]), stub(1.0)).iloc[0]
    assert out["recommended_price_chf"] == pr.TIER_PRICE_CHF[pr.EVENING]
    assert out["applied_rule"] == "keep_tariff"


def test_full_and_blocked_slots_keep_the_tariff():
    df = pd.DataFrame([slot(pr.EVENING, courts_available=16, booked=16), slot(pr.DAY, courts_available=0, booked=0)])
    out = scoring.recommend_prices(df, stub(0.3))
    assert list(out["applied_rule"]) == ["fully_booked", "no_courts_available"]
    assert (out["recommended_price_chf"] == out["base_price_chf"]).all()


@pytest.mark.parametrize("elasticity", [0.2, 0.8, 1.2, 3.0])
def test_recommendations_respect_guardrails(elasticity):
    rows = [slot(t, booked=b, hist=h) for t in pr.TIERS for b in (0, 5, 12) for h in (0.1, 0.5, 0.9)]
    out = scoring.recommend_prices(pd.DataFrame(rows), stub(elasticity))
    for r in out.itertuples():
        lo, hi = pr.price_bounds(r.tier)
        assert lo <= r.recommended_price_chf <= hi
        assert pr.PRICE_FLOOR_CHF <= r.recommended_price_chf <= pr.PRICE_CEILING_CHF
        assert abs(r.price_change_pct) <= pr.MAX_MOVE_PCT + 1e-9
    assert (out["expected_revenue_uplift_chf"] >= -1e-9).all()


def test_policy_blocks_discounts_just_before_the_evening():
    policy = scoring.PricingPolicy(no_discount_hours_before_evening=2.0)
    late = slot(pr.DAY, booked=0, hist=0.3, start_hour=15.5, slot_start_ts=datetime(2026, 11, 24, 15, 30))
    early = slot(pr.DAY, booked=0, hist=0.3, start_hour=10.0, slot_start_ts=datetime(2026, 11, 24, 10))
    out = scoring.recommend_prices(pd.DataFrame([late, early]), stub(2.5), policy)
    assert out["recommended_price_chf"].iloc[0] == pr.TIER_PRICE_CHF[pr.DAY]
    assert "evening" in out["reason"].iloc[0]
    assert out["recommended_price_chf"].iloc[1] < pr.TIER_PRICE_CHF[pr.DAY]
    lunch_guard = scoring.PricingPolicy(no_discount_hours_before_lunch=1.0)
    pre_lunch = slot(pr.DAY, booked=0, hist=0.3, start_hour=11.0, slot_start_ts=datetime(2026, 11, 24, 11))
    out = scoring.recommend_prices(pd.DataFrame([pre_lunch, early]), stub(2.5), lunch_guard)
    assert out["recommended_price_chf"].iloc[0] == pr.TIER_PRICE_CHF[pr.DAY]
    assert out["recommended_price_chf"].iloc[1] < pr.TIER_PRICE_CHF[pr.DAY]


def test_policy_still_allows_raises_when_discounts_are_blocked():
    policy = scoring.PricingPolicy(no_discount_tiers=frozenset({pr.LUNCH}))
    up = scoring.recommend_prices(pd.DataFrame([slot(pr.LUNCH)]), stub(0.3), policy).iloc[0]
    assert up["recommended_price_chf"] == pr.price_bounds(pr.LUNCH)[1]
    down = scoring.recommend_prices(pd.DataFrame([slot(pr.LUNCH, booked=0, hist=0.3)]), stub(2.5), policy).iloc[0]
    assert down["recommended_price_chf"] == pr.TIER_PRICE_CHF[pr.LUNCH]


def test_policy_min_gain_by_tier():
    weekend = slot(pr.WEEKEND, booked=0, hist=0.4)
    loose = scoring.recommend_prices(pd.DataFrame([weekend]), stub(0.9), scoring.PricingPolicy()).iloc[0]
    assert loose["recommended_price_chf"] != pr.TIER_PRICE_CHF[pr.WEEKEND]
    strict = scoring.PricingPolicy(min_gain_pct_by_tier={pr.WEEKEND: 0.5})
    out = scoring.recommend_prices(pd.DataFrame([weekend]), stub(0.9), strict).iloc[0]
    assert out["recommended_price_chf"] == pr.TIER_PRICE_CHF[pr.WEEKEND]
    assert "50%" in out["reason"]


def test_bands():
    p = pd.Series([0.05, 0.10, 0.20, 0.50])
    assert list(scoring.band(p, scoring.CHURN_BANDS)) == ["low", "low", "medium", "high"]
    assert list(scoring.band(p, scoring.CANCEL_BANDS)) == ["low", "medium", "high", "high"]


def test_waitlist_matches_count_open_windows_only():
    bookings = pd.DataFrame({"start_ts": [datetime(2026, 11, 26, 19), datetime(2026, 11, 26, 8)]})
    waitlist = pd.DataFrame({
        "window_start_ts": [datetime(2026, 11, 26, 18), datetime(2026, 11, 26, 18), datetime(2026, 11, 26, 19)],
        "window_end_ts": [datetime(2026, 11, 26, 20), datetime(2026, 11, 26, 20), datetime(2026, 11, 26, 19)],
        "status": ["open", "expired", "open"],
    })
    assert list(scoring.waitlist_matches(bookings, waitlist)) == [2, 0]


def test_churn_scores_rule_and_action():
    snap = pd.DataFrame({"plays_last_30d": [0, 6], "avg_monthly_plays_baseline": [5.0, 5.0]})
    out = scoring.churn_scores(snap, np.array([0.6, 0.05]))
    assert list(out["churn_risk_rule"]) == [True, False]
    assert out.loc[0, "suggested_action"].startswith("Retention offer")
    assert out.loc[1, "suggested_action"] == "No action"
