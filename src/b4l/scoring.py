"""Batch scoring logic for 03b (SPEC.md 5.B), kept free of MLflow so it can be tested with a stub model.

Price recommendation per upcoming slot:
1. Try every candidate price in the tier's guardrail range (CHF 1 steps).
2. Predict the final fill at each price. Courts already booked keep their locked price, so only the courts
   still expected to be sold earn the candidate price: expected new revenue = price x (predicted courts - booked).
3. Keep the price with the highest expected new revenue, but stay at the published tariff unless the gain is at
   least MIN_GAIN_PCT (avoids churning prices on noise), then apply the guardrails and record the rule.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from b4l import ml_features as mf
from b4l import pricing_rules as pr

MIN_GAIN_PCT = 0.03
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


def recommend_prices(slots: pd.DataFrame, predict_fill: Predict, min_gain_pct: float = MIN_GAIN_PCT) -> pd.DataFrame:
    """slots: rows from ml_features.fill_features_from_gold (upcoming slots). Returns one row per slot."""
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
        gain = best["expected_new_revenue_chf"] - at_base["expected_new_revenue_chf"]
        if best["candidate_price_chf"] == base:
            chosen = at_base
            rec.update(applied_rule="keep_tariff", reason="The published tariff has the highest expected revenue")
        elif gain < min_gain_pct * max(at_base["expected_new_revenue_chf"], 1.0):
            chosen = at_base
            rec.update(applied_rule="keep_tariff",
                       reason=f"Expected gain below {min_gain_pct:.0%}; the published tariff is kept")
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
