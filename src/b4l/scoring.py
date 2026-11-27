"""Batch scoring logic for 03b (SPEC.md 5.B), kept free of MLflow so it can be tested with a stub model.

Price recommendation per upcoming slot:
1. Try every candidate price in the tier's guardrail range (CHF 1 steps).
2. Predict the final fill at each price. Courts already booked keep their locked price, so only the courts
   still expected to be sold earn the candidate price: expected new revenue = price x (predicted courts - booked).
3. Keep the price with the highest expected new revenue, subject to the pricing policy (DEFAULT_POLICY): stay at the
   published tariff unless the gain is at least the minimum (10%, 15% at weekends), and never discount in the 2 hours
   before the evening tariff. Then apply the guardrails and record the rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from b4l import ml_features as mf
from b4l import pricing_rules as pr

MIN_GAIN_PCT = 0.03
LUNCH_START_HOUR = 11.5
EVENING_START_HOUR = 17.0


@dataclass(frozen=True)
class PricingPolicy:
    """Business rules on top of the per-slot revenue optimum.

    The fill model prices each slot on its own, so it cannot see that a discount may only pull players from a busy
    slot nearby; these rules stop the cases the backtest showed to lose money net of that cannibalisation.
    """

    min_gain_pct: float = MIN_GAIN_PCT
    min_gain_pct_by_tier: dict[str, float] = field(default_factory=dict)
    no_discount_hours_before_evening: float = 0.0
    no_discount_hours_before_lunch: float = 0.0
    no_discount_tiers: frozenset[str] = frozenset()

    def min_gain(self, tier: str) -> float:
        return self.min_gain_pct_by_tier.get(tier, self.min_gain_pct)

    def discount_blocked(self, tier: str, start_hour: float) -> str | None:
        if tier in self.no_discount_tiers:
            return f"Discounts are not offered in the {tier} tier"
        if tier != pr.DAY:
            return None
        h = self.no_discount_hours_before_evening
        if h and start_hour >= EVENING_START_HOUR - h:
            return f"No discount in the {h:.0f} h before the evening tariff: it would pull players from busy evenings"
        h = self.no_discount_hours_before_lunch
        if h and LUNCH_START_HOUR - h <= start_hour < LUNCH_START_HOUR:
            return f"No discount in the {h:.0f} h before the lunch tariff: it would pull players from lunch"
        return None


POLICY_VARIANTS: dict[str, PricingPolicy] = {
    "per_slot_3pct": PricingPolicy(),
    "min_gain_10pct": PricingPolicy(min_gain_pct=0.10),
    "protect_evening": PricingPolicy(min_gain_pct=0.10, no_discount_hours_before_evening=2.0),
    "protect_evening_weekend": PricingPolicy(min_gain_pct=0.10, no_discount_hours_before_evening=2.0,
                                             min_gain_pct_by_tier={pr.WEEKEND: 0.15}),
    "evening_guard_3pct": PricingPolicy(no_discount_hours_before_evening=2.0,
                                        no_discount_tiers=frozenset({pr.WEEKEND})),
    "evening_lunch_guard_5pct": PricingPolicy(min_gain_pct=0.05, no_discount_hours_before_evening=2.0,
                                              no_discount_hours_before_lunch=1.0,
                                              no_discount_tiers=frozenset({pr.WEEKEND})),
    "robust_8pct": PricingPolicy(min_gain_pct=0.08, no_discount_hours_before_evening=2.0,
                                 min_gain_pct_by_tier={pr.WEEKEND: 0.15}),
}
# The only tested set with a positive 03c backtest at low, mid and high price sensitivity.
DEFAULT_POLICY_NAME = "protect_evening_weekend"
DEFAULT_POLICY = POLICY_VARIANTS[DEFAULT_POLICY_NAME]
CHURN_BANDS = ((0.35, "high"), (0.15, "medium"))
CANCEL_BANDS = ((0.15, "high"), (0.08, "medium"))

Predict = Callable[[pd.DataFrame], np.ndarray]


def band(p: pd.Series, cuts: tuple[tuple[float, str], ...]) -> pd.Series:
    out = pd.Series("low", index=p.index)
    for threshold, label in reversed(cuts):
        out[p >= threshold] = label
    return out


def price_curves(slots: pd.DataFrame, predict_fill: Predict) -> pd.DataFrame:
    """One row per (slot, candidate price) with predicted fill, courts and expected new revenue."""
    parts = []
    for tier, g in slots.groupby("tier"):
        for price in pr.candidate_prices(tier):
            parts.append(mf.with_price(g, price).assign(candidate_price_chf=price))
    grid = pd.concat(parts)
    grid["pred_fill"] = np.clip(predict_fill(grid[mf.FILL_FEATURES]), 0.0, 1.0)
    grid["expected_courts"] = grid["pred_fill"] * grid["courts_available"]
    grid["expected_new_courts"] = (grid["expected_courts"] - grid["booked_courts_now"]).clip(lower=0.0)
    grid["expected_new_revenue_chf"] = grid["candidate_price_chf"] * grid["expected_new_courts"]
    return grid


def recommend_prices(slots: pd.DataFrame, predict_fill: Predict, policy: PricingPolicy | None = None) -> pd.DataFrame:
    """slots: rows from ml_features.fill_features_from_gold (upcoming slots). Returns one row per slot."""
    policy = policy or DEFAULT_POLICY
    slots = slots.reset_index(drop=True).assign(slot_idx=lambda d: d.index)
    full = slots["booked_courts_now"] >= slots["courts_available"]
    curves = price_curves(slots[~full], predict_fill) if (~full).any() else pd.DataFrame()

    rows = []
    for s in slots.itertuples(index=False):
        base = float(s.base_price_chf)
        rec = {"slot_idx": s.slot_idx, "model_price_chf": base, "recommended_price_chf": base,
               "expected_fill_at_base": np.nan, "expected_fill_at_recommended": np.nan,
               "expected_revenue_at_base_chf": 0.0, "expected_revenue_at_recommended_chf": 0.0}
        if s.courts_available <= 0:
            rec.update(applied_rule="no_courts_available", reason="All courts are blocked for courses or clubs")
            rows.append(rec)
            continue
        if s.booked_courts_now >= s.courts_available:
            rec.update(applied_rule="fully_booked", reason="Every available court is already booked at a locked price")
            rows.append(rec)
            continue
        c = curves[curves["slot_idx"] == s.slot_idx]
        at_base = c[c["candidate_price_chf"] == base].iloc[0]
        # Highest expected revenue; ties go to the price closest to the tariff.
        c = c.assign(_dist=(c["candidate_price_chf"] - base).abs())
        best = c.sort_values(["expected_new_revenue_chf", "_dist"], ascending=[False, True]).iloc[0]
        rec.update(model_price_chf=float(best["candidate_price_chf"]),
                   expected_fill_at_base=float(at_base["pred_fill"]),
                   expected_revenue_at_base_chf=float(at_base["expected_new_revenue_chf"]))
        blocked = policy.discount_blocked(s.tier, float(s.start_hour)) if best["candidate_price_chf"] < base else None
        if blocked:
            c = c[c["candidate_price_chf"] >= base]
            best = c.sort_values(["expected_new_revenue_chf", "_dist"], ascending=[False, True]).iloc[0]
        gain = best["expected_new_revenue_chf"] - at_base["expected_new_revenue_chf"]
        min_gain = policy.min_gain(s.tier)
        if best["candidate_price_chf"] == base:
            chosen = at_base
            rec.update(applied_rule="keep_tariff",
                       reason=f"{blocked}; the published tariff is kept" if blocked
                       else "The published tariff has the highest expected revenue")
        elif gain < min_gain * max(at_base["expected_new_revenue_chf"], 1.0):
            chosen = at_base
            gain_pct = gain / max(at_base["expected_new_revenue_chf"], 1.0)
            rec.update(applied_rule="keep_tariff",
                       reason=f"Best price CHF {best['candidate_price_chf']:.0f} would add only {gain_pct:.0%} "
                              f"(minimum {min_gain:.0%}); the published tariff is kept")
        else:
            g = pr.apply_guardrails(s.tier, float(best["candidate_price_chf"]))
            chosen = c[c["candidate_price_chf"] == g.final_price_chf].iloc[0]
            direction = "raised" if g.final_price_chf > base else "lowered"
            rec.update(applied_rule=g.applied_rule,
                       reason=f"Price {direction}: expected +CHF {gain:.0f} of new revenue vs the tariff. {g.reason}")
        rec.update(recommended_price_chf=float(chosen["candidate_price_chf"]),
                   expected_fill_at_recommended=float(chosen["pred_fill"]),
                   expected_revenue_at_recommended_chf=float(chosen["expected_new_revenue_chf"]))
        rows.append(rec)

    out = slots.merge(pd.DataFrame(rows), on="slot_idx").drop(columns="slot_idx")
    out["free_courts"] = (out["courts_available"] - out["booked_courts_now"]).clip(lower=0)
    out["price_change_pct"] = out["recommended_price_chf"] / out["base_price_chf"] - 1.0
    out["expected_revenue_uplift_chf"] = out["expected_revenue_at_recommended_chf"] - out["expected_revenue_at_base_chf"]
    return out


def churn_scores(snapshot: pd.DataFrame, probability: np.ndarray) -> pd.DataFrame:
    """Regulars at the snapshot with churn probability, band, the gold rule and a suggested action."""
    s = snapshot.copy()
    s["churn_probability"] = probability
    s["churn_risk_band"] = band(s["churn_probability"], CHURN_BANDS)
    s["churn_risk_rule"] = s["plays_last_30d"] <= mf.CHURN_DROP_RATIO * s["avg_monthly_plays_baseline"]
    base = s["avg_monthly_plays_baseline"].replace(0, np.nan)
    s["play_change_vs_usual_pct"] = s["plays_last_30d"] / base - 1.0
    s["suggested_action"] = np.select(
        [s["churn_risk_band"] == "high", s["churn_risk_band"] == "medium"],
        ["Retention offer: CHF 10 stringing voucher or a doubles invite with usual partners",
         "Watch: invite to a doubles match"],
        default="No action",
    )
    return s


def waitlist_matches(bookings: pd.DataFrame, waitlist: pd.DataFrame) -> pd.Series:
    """Open waitlist requests whose window covers each booking's start (backfill candidates)."""
    if waitlist.empty or bookings.empty:
        return pd.Series(0, index=bookings.index)
    wl = waitlist[waitlist["status"] == "open"]
    starts = pd.to_datetime(bookings["start_ts"]).to_numpy()
    lo = pd.to_datetime(wl["window_start_ts"]).to_numpy()
    hi = pd.to_datetime(wl["window_end_ts"]).to_numpy()
    counts = ((starts[:, None] >= lo[None, :]) & (starts[:, None] <= hi[None, :])).sum(axis=1)
    return pd.Series(counts, index=bookings.index)


def cancellation_scores(upcoming: pd.DataFrame, probability: np.ndarray, waitlist: pd.DataFrame) -> pd.DataFrame:
    s = upcoming.copy()
    s["late_cancel_probability"] = probability
    s["risk_band"] = band(s["late_cancel_probability"], CANCEL_BANDS)
    s["open_waitlist_matches"] = waitlist_matches(s, waitlist)
    s["suggested_action"] = np.select(
        [(s["risk_band"] == "high") & (s["open_waitlist_matches"] > 0), s["risk_band"] == "high"],
        ["Confirm with the booker 24 h before; line up the next waitlisted player",
         "Send a reminder 24 h before (late cancellations are billed in full)"],
        default="No action",
    )
    return s
