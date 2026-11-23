"""Data quality: hard consistency rules and statistical calibration checks (SPEC.md section 4.1)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from b4l import pricing_rules as pr
from b4l.synth import World

BILLED = ("completed", "upcoming", "cancelled_late", "no_show")

OCCUPANCY_TARGETS = {  # band -> (low, high) for term-time history
    "evening_mon_thu": (0.85, 0.97),
    "lunch": (0.60, 0.75),
    "weekday_morning": (0.30, 0.45),
    "weekend": (0.70, 0.90),
}


def cell_band(weekday: int, day_type: str, minute: int) -> str:
    if day_type == "weekend_holiday":
        return "weekend"
    if 690 <= minute < 810:
        return "lunch"
    if minute < 690:
        return "weekday_morning"
    if weekday == 4:
        return "friday_pm"
    if minute < 1020:
        return "afternoon"
    return "evening_mon_thu"


def occupancy_cells(world: World, bookings: pd.DataFrame, statuses=BILLED) -> list[np.ndarray]:
    """Per day, a (courts x cells) boolean array of time used by billed bookings or court blocks."""
    court_index = {c: i for i, c in enumerate(pr.COURT_IDS)}
    grids = [np.zeros((pr.N_COURTS, max(d.n_cells, 1)), dtype=bool) for d in world.days]
    for b in world.blocks.itertuples():
        day = world.days[b.day_idx]
        c0 = ((b.start_ts.hour * 60 + b.start_ts.minute) - day.open_min) // pr.SLOT_MINUTES
        c1 = ((b.end_ts.hour * 60 + b.end_ts.minute) - day.open_min) // pr.SLOT_MINUTES
        grids[b.day_idx][court_index[b.court_id], c0:c1] = True
    sel = bookings[bookings["status"].isin(statuses)]
    for day_idx, cell, dur, court in zip(sel["day_idx"], sel["start_cell"], sel["dur_cells"], sel["court_idx"]):
        grids[day_idx][court, cell:cell + dur] = True
    return grids


def occupancy_by_band(world: World, res, history_only: bool = True) -> pd.DataFrame:
    grids = occupancy_cells(world, res.bookings)
    rows = []
    for day in world.days:
        if day.n_cells == 0 or (history_only and day.is_future):
            continue
        g = grids[day.idx]
        period = "summer_holiday" if day.school_holiday_name == "Vacances d'été" else (
            "other_holiday" if day.is_school_holiday else "term")
        for c in range(day.n_cells):
            minute = day.open_min + c * pr.SLOT_MINUTES
            rows.append((period, day.d.month, cell_band(day.weekday, day.day_type, minute), g[:, c].mean()))
    df = pd.DataFrame(rows, columns=["period", "month", "band", "occ"])
    out = df.pivot_table(index="band", columns="period", values="occ", aggfunc="mean")
    out["all"] = df.groupby("band")["occ"].mean()
    return out.round(3)


def occupancy_by_month(world: World, res) -> pd.DataFrame:
    grids = occupancy_cells(world, res.bookings)
    rows = []
    for day in world.days:
        if day.n_cells == 0 or day.is_future:
            continue
        rows.append((day.d.month, grids[day.idx].mean()))
    return pd.DataFrame(rows, columns=["month", "occ"]).groupby("month")["occ"].mean().round(3)


def summary_stats(world: World, res) -> dict[str, object]:
    b = res.bookings
    hist = b[b["start_ts"] < world.now]
    weeks = (world.as_of - world.window_start).days / 7
    stats: dict[str, object] = {}
    stats["bookings_per_week"] = round(len(hist) / weeks, 1)
    stats["late_cancel_or_no_show_rate"] = round(hist["status"].isin(["cancelled_late", "no_show"]).mean(), 4)
    stats["early_cancel_rate"] = round((hist["status"] == "cancelled_early").mean(), 4)
    billed = hist[hist["status"].isin(BILLED)]
    stats["revenue_chf_per_week"] = round(billed["revenue_chf"].sum() / weeks, 0)
    per_player = hist.groupby("player_idx").size().sort_values(ascending=False)
    top = per_player.head(max(1, int(len(per_player) * 0.2))).sum() / per_player.sum()
    stats["top20pct_booker_share"] = round(top, 3)
    stats["active_bookers"] = int(len(per_player))
    stats["pass_booking_share"] = round((hist["payment_method"] == "pass").mean(), 3)
    stats["prepaid_booking_share"] = round((hist["payment_method"] == "prepaid").mean(), 3)
    stats["family_rate_share"] = round((hist["rate_category"] == "family").mean(), 3)
    wl = res.waitlist
    if len(wl):
        stats["waitlist_entries"] = int(len(wl))
        stats["waitlist_resolved_rate"] = round(wl["resolved_booking_id"].notna().mean(), 3)
    stats["upcoming_bookings"] = int((b["status"] == "upcoming").sum())
    stats["half_hour_start_share"] = round((hist["start_ts"].dt.minute == 30).mean(), 3)
    return stats


from b4l.pipeline_helpers import CHURN_DROP_RATIO, CHURN_MIN_MONTHLY_PLAYS  # noqa: E402


def churn_base_rate(plays: pd.DataFrame, as_of) -> float:
    """Share of regulars flagged by the churn rule (see schemas.GOLD player_features.churn_risk_rule).

    plays: columns player_id, start_ts of completed participations.
    """
    now = pd.Timestamp(as_of)
    last = plays[(plays["start_ts"] >= now - pd.Timedelta(days=30)) & (plays["start_ts"] < now)]
    base = plays[(plays["start_ts"] >= now - pd.Timedelta(days=120)) & (plays["start_ts"] < now - pd.Timedelta(days=30))]
    monthly = base.groupby("player_id").size() / 3.0
    regulars = monthly[monthly >= CHURN_MIN_MONTHLY_PLAYS]
    lc = last.groupby("player_id").size().reindex(regulars.index, fill_value=0)
    label = lc <= CHURN_DROP_RATIO * regulars
    return float(label.mean()) if len(label) else float("nan")


def _check(rows: list, name: str, category: str, passed: bool, observed, expected) -> None:
    rows.append({"check_name": name, "category": category, "passed": bool(passed), "observed": str(observed),
                 "expected": str(expected)})


def run_checks(world: World, bronze: dict[str, pd.DataFrame], res) -> pd.DataFrame:
    """Hard rules on clean bronze frames plus statistical calibration checks."""
    rows: list[dict] = []
    bookings = bronze["bookings"]
    blocks = bronze["court_blocks"]
    cal = bronze["calendar"].set_index("date")

    # No two live bookings on the same court overlap (late-cancelled slots may be resold).
    live = bookings[bookings["status"].isin(["completed", "upcoming", "no_show"])].sort_values(["court_id", "start_ts"])
    prev_end = live.groupby("court_id")["end_ts"].shift()
    overlaps = int((live["start_ts"] < prev_end).sum())
    _check(rows, "no_double_booking", "hard_rule", overlaps == 0, overlaps, 0)

    merged = live.merge(blocks[["court_id", "start_ts", "end_ts"]], on="court_id", suffixes=("", "_blk"))
    block_overlap = int(((merged["start_ts"] < merged["end_ts_blk"]) & (merged["start_ts_blk"] < merged["end_ts"])).sum())
    _check(rows, "no_booking_during_court_block", "hard_rule", block_overlap == 0, block_overlap, 0)

    def within_hours(r) -> bool:
        dtype = cal.at[r.start_ts.date(), "day_type"]
        window = pr.rentable_window(dtype)
        if window is None:
            return False
        open_dt = datetime.combine(r.start_ts.date(), window[0])
        close_dt = datetime.combine(r.start_ts.date(), window[1])
        return open_dt <= r.start_ts and r.end_ts <= close_dt and r.start_ts.minute in (0, 30)

    outside = int((~bookings.apply(within_hours, axis=1)).sum())
    _check(rows, "bookings_within_rentable_hours_on_open_days", "hard_rule", outside == 0, outside, 0)

    def expected_tier(r) -> str:
        dtype = cal.at[r.start_ts.date(), "day_type"]
        return pr.tier_for(r.start_ts.time(), r.start_ts.weekday(), dtype)

    tier_bad = int((bookings.apply(expected_tier, axis=1) != bookings["tier"]).sum())
    _check(rows, "tier_matches_start_time", "hard_rule", tier_bad == 0, tier_bad, 0)
    base_bad = int((bookings.apply(lambda r: pr.base_price(r.tier, r.rate_category), axis=1)
                    != bookings["base_price_chf"]).sum())
    _check(rows, "base_price_matches_tariff", "hard_rule", base_bad == 0, base_bad, 0)
    static_bad = int(((bookings["payment_method"] != "pass") & (bookings["status"] != "cancelled_early")
                      & ((bookings["price_paid_chf"] - bookings["price_shown_chf"] * bookings["duration_h"]).abs() > 0.01)
                      ).sum())
    static_bad += int((bookings["price_shown_chf"] != bookings["base_price_chf"]).sum())
    _check(rows, "static_history_pays_published_tariff", "hard_rule", static_bad == 0, static_bad, 0)
    pass_paid = int(((bookings["payment_method"] == "pass") & (bookings["price_paid_chf"] != 0)).sum())
    _check(rows, "hour_pass_bookings_pay_zero_cash", "hard_rule", pass_paid == 0, pass_paid, 0)

    checks = bronze["check_ins"]
    status = bookings.set_index("booking_id")["status"]
    bad_ci = int((~checks["booking_id"].map(status).isin(["completed"])).sum())
    _check(rows, "check_ins_only_for_completed_bookings", "hard_rule", bad_ci == 0, bad_ci, 0)
    completed = set(bookings.loc[bookings["status"] == "completed", "booking_id"])
    missing_ci = len(completed - set(checks["booking_id"]))
    _check(rows, "every_completed_booking_checked_in", "hard_rule", missing_ci == 0, missing_ci, 0)
    billed_bad = int(((bookings["status"].isin(["cancelled_late", "no_show"])) & (bookings["revenue_chf"] <= 0)).sum())
    _check(rows, "late_cancel_and_no_show_billed", "hard_rule", billed_bad == 0, billed_bad, 0)

    ledger = bronze["pass_transactions"].sort_values("ts")
    wallets = bronze["pass_wallets"].set_index("wallet_id")
    neg = 0
    for wid, grp in ledger.groupby("wallet_id"):
        w = wallets.loc[wid]
        if w["product"] == pr.CARTE_PREPAYEE:
            neg += int(((w["initial_balance_chf"] + grp["delta_chf"].cumsum()) < -0.01).any())
        else:
            neg += int(((w["initial_hours"] + grp["delta_hours"].cumsum()) < 0).any())
    _check(rows, "pass_wallets_never_negative", "hard_rule", neg == 0, neg, 0)

    pass_b = bookings[bookings["payment_method"] == "pass"].merge(
        bronze["pass_wallets"][["wallet_id", "expires_at"]], on="wallet_id", how="left")
    after_exp = int((pass_b["expires_at"].notna() & (pass_b["start_ts"] > pass_b["expires_at"])).sum())
    _check(rows, "chrono_never_used_after_expiry", "hard_rule", after_exp == 0, after_exp, 0)

    wl = bronze["waitlist"]
    res_wl = wl[wl["resolved_booking_id"].notna()].merge(
        bookings[["booking_id", "start_ts", "booked_at", "source_waitlist_id"]], left_on="resolved_booking_id",
        right_on="booking_id", how="left")
    wl_bad = int((res_wl["source_waitlist_id"] != res_wl["waitlist_id"]).sum())
    wl_bad += int(((res_wl["start_ts"] < res_wl["window_start_ts"]) | (res_wl["start_ts"] > res_wl["window_end_ts"])).sum())
    wl_bad += int((res_wl["booked_at"] < res_wl["created_at"]).sum())
    _check(rows, "waitlist_resolutions_consistent", "hard_rule", wl_bad == 0, wl_bad, 0)

    parts = bronze["booking_participants"]
    counts = parts.groupby("booking_id").size()
    party = bookings.set_index("booking_id")["party_size"]
    part_bad = int((counts.reindex(party.index, fill_value=0) != party).sum())
    bookers = parts[parts["is_booker"]].set_index("booking_id")["player_id"]
    part_bad += int((bookers.reindex(party.index) != bookings.set_index("booking_id")["booker_player_id"]).sum())
    _check(rows, "participants_match_party_size_and_booker", "hard_rule", part_bad == 0, part_bad, 0)

    # Statistical checks.
    occ = occupancy_by_band(world, res)
    for band, (lo, hi) in OCCUPANCY_TARGETS.items():
        v = float(occ.loc[band, "term"])
        _check(rows, f"occupancy_{band}_term", "statistical", lo <= v <= hi, round(v, 3), f"{lo}-{hi}")
    if "summer_holiday" in occ.columns:
        summer, term = float(occ["summer_holiday"].mean()), float(occ["term"].mean())
        _check(rows, "summer_dip_visible", "statistical", summer < term - 0.05, round(summer, 3), f"< term {term:.3f} - 0.05")

    hist = bookings[bookings["start_ts"] < world.now]
    late_rate = float(hist["status"].isin(["cancelled_late", "no_show"]).mean())
    _check(rows, "late_cancel_no_show_rate", "statistical", 0.06 <= late_rate <= 0.10, round(late_rate, 4), "0.06-0.10")

    plays = parts.merge(bookings[["booking_id", "start_ts", "status"]], on="booking_id")
    plays = plays[plays["status"] == "completed"]
    churn = churn_base_rate(plays, world.as_of)
    _check(rows, "churn_label_base_rate", "statistical", 0.08 <= churn <= 0.20, round(churn, 4), "0.08-0.20")

    per_player = hist.groupby("booker_player_id").size().sort_values(ascending=False)
    top = float(per_player.head(int(len(per_player) * 0.2)).sum() / per_player.sum())
    _check(rows, "activity_skew_top20_share", "statistical", 0.5 <= top <= 0.7, round(top, 3), "0.50-0.70")

    rainy_weekend = _weekend_rain_lift(world, res)
    _check(rows, "rainy_weekend_demand_lift", "statistical", rainy_weekend >= 0.10, round(rainy_weekend, 3), ">= 0.10")

    month_occ = occupancy_by_month(world, res)
    sponsor_ok = _sponsor_cells_ok(bronze)
    _check(rows, "sponsor_monthly_segments_k_anonymous", "statistical", sponsor_ok[0], sponsor_ok[1], ">= 10 per cell")
    autumn = [month_occ[m] for m in (10, 11) if m in month_occ.index]
    if 7 in month_occ.index and autumn:
        _check(rows, "autumn_peak_above_july", "statistical", bool(max(autumn) > month_occ[7] + 0.05),
               month_occ.round(3).to_dict(), "Oct/Nov > Jul + 0.05")
    return pd.DataFrame(rows)


def _weekend_rain_lift(world: World, res) -> float:
    """Demand lift on rainy weekend days, measured as bookings + turned-away requests."""
    b = res.bookings
    w = world.weather.set_index("date")
    demand = b.groupby("day_idx").size()
    ta = res.turned_away.groupby("day_idx")["turned_away"].sum() if len(res.turned_away) else pd.Series(dtype=float)
    vals = {True: [], False: []}
    for day in world.days:
        if day.day_type != "weekend_holiday" or day.is_future or day.is_school_holiday:
            continue
        vals[bool(w.at[day.d, "is_rainy"])].append(demand.get(day.idx, 0) + ta.get(day.idx, 0))
    if not vals[True] or not vals[False]:
        return float("nan")
    return float(np.mean(vals[True]) / np.mean(vals[False]) - 1)


def _sponsor_cells_ok(bronze: dict[str, pd.DataFrame]) -> tuple[bool, str]:
    b = bronze["bookings"]
    parts = bronze["booking_participants"].merge(b[["booking_id", "start_ts", "status"]], on="booking_id")
    parts = parts[parts["status"] == "completed"].merge(bronze["players"][["player_id", "segment"]], on="player_id")
    parts["month"] = parts["start_ts"].dt.to_period("M")
    counts = parts.groupby(["month", "segment"])["player_id"].nunique()
    return bool((counts >= 10).all()), f"min={int(counts.min())}"
