"""Feature builders for the Phase 2 models (SPEC.md 5.B), shared by training (03), scoring (03b) and the backtest (03c).

Pure pandas/numpy so they run on a serverless driver and in local tests. Categoricals are integer-coded with fixed
mappings so MLflow signatures stay numeric and scoring uses the same codes as training.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from b4l import demand
from b4l import pricing_rules as pr
from b4l.pipeline_helpers import CHURN_DROP_RATIO, CHURN_MIN_MONTHLY_PLAYS
from b4l.synth import SEGMENTS, World

BILLED = ("completed", "cancelled_late", "no_show")
LATE_OUTCOMES = ("cancelled_late", "no_show")
MAX_LEAD_DAYS = 13

TIER_CODE = {t: i for i, t in enumerate(pr.TIERS)}
SEGMENT_CODE = {s: i for i, s in enumerate(SEGMENTS)}
SKILL_CODE = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
PAYMENT_CODE = {"on_site": 0, "prepaid": 1, "pass": 2}


def _code(series: pd.Series, mapping: dict) -> pd.Series:
    return series.map(mapping).fillna(-1).astype("int64")


# ------------------------------------------------------------------------------------------------ fill probability

FILL_FEATURES = [
    "day_of_week", "start_hour", "is_weekend_day", "tier_code", "is_school_holiday", "is_public_holiday",
    "arena_event", "is_rainy", "temp_mean_c", "courts_available", "hist_fill_rate_4w", "lead_days",
    "booked_share_now", "price_chf", "price_ratio",
]
# LightGBM monotone constraints: a higher price never raises the predicted fill.
FILL_MONOTONE = [-1 if f in ("price_chf", "price_ratio") else 0 for f in FILL_FEATURES]
FILL_CATEGORICAL = ["day_of_week", "tier_code"]
FILL_LABEL = "fill_rate"


def exploration_price_grid(world: World, rng: np.random.Generator) -> list[np.ndarray]:
    """Simulated price test: every (day, clock hour) gets a random price from its tier's guardrail range.

    Both 30-minute cells of an hour share the draw so a slot and its half-hour neighbour are priced alike.
    """
    tiers = demand.tier_grid(world)
    grid = []
    for day, day_tiers in zip(world.days, tiers):
        p = np.full(day.n_cells, np.nan)
        draws: dict[tuple[int, str], float] = {}
        for c, tier in enumerate(day_tiers):
            key = (day.cell_time(c).hour, tier)
            if key not in draws:
                draws[key] = float(rng.choice(pr.candidate_prices(tier)))
            p[c] = draws[key]
        grid.append(p)
    return grid


def _blocked_courts(world: World) -> dict[tuple[int, int], int]:
    """(day_idx, start minute) -> courts blocked for any part of the hour starting there."""
    out: dict[tuple[int, int], set] = {}
    for b in world.blocks.itertuples():
        day = world.days[b.day_idx]
        b0 = b.start_ts.hour * 60 + b.start_ts.minute
        b1 = b.end_ts.hour * 60 + b.end_ts.minute
        for st in pr.start_times(day.day_type):
            m = st.hour * 60 + st.minute
            if b0 < m + pr.BOOKING_MINUTES and m < b1:
                out.setdefault((b.day_idx, m), set()).add(b.court_id)
    return {k: len(v) for k, v in out.items()}


def slot_grid(world: World, history_only: bool = True) -> pd.DataFrame:
    """One row per open day and valid booking start, with calendar, weather and courts available."""
    weather = world.weather.set_index("date")
    blocked = _blocked_courts(world)
    rows = []
    for day in world.days:
        if day.day_type == "closed" or (history_only and day.d >= world.as_of):
            continue
        w = weather.loc[day.d]
        for st in pr.start_times(day.day_type):
            m = st.hour * 60 + st.minute
            tier = pr.tier_for(st, day.weekday, day.day_type)
            rows.append({
                "day_idx": day.idx, "slot_date": day.d, "start_minute": m,
                "start_cell": (m - day.open_min) // pr.SLOT_MINUTES,
                "slot_start_ts": datetime.combine(day.d, st),
                "day_of_week": day.weekday + 1, "start_hour": m / 60.0,
                "is_weekend_day": int(day.day_type == "weekend_holiday"), "tier": tier,
                "tier_code": TIER_CODE[tier], "base_price_chf": pr.TIER_PRICE_CHF[tier],
                "is_school_holiday": int(day.is_school_holiday), "is_public_holiday": int(day.is_public_holiday),
                "arena_event": int(day.arena_event), "is_rainy": int(bool(w["is_rainy"])),
                "temp_mean_c": float(w["temp_mean_c"]),
                "courts_available": pr.N_COURTS - blocked.get((day.idx, m), 0),
            })
    return pd.DataFrame(rows)


def add_hist_fill(slots: pd.DataFrame) -> pd.DataFrame:
    """hist_fill_rate_4w: mean fill of the same weekday and start over the previous 28 days (as in gold)."""
    out = slots.copy()
    out["hist_fill_rate_4w"] = np.nan
    day_num = (pd.to_datetime(out["slot_date"]) - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    fill = out["fill_rate"].to_numpy(dtype=float)
    hist = np.full(len(out), np.nan)
    for idx in out.groupby(["day_of_week", "start_minute"]).indices.values():
        d, f = day_num[idx], fill[idx]
        for j, i in enumerate(idx):
            mask = (d >= d[j] - 28) & (d < d[j])
            if mask.any():
                hist[i] = f[mask].mean()
    out["hist_fill_rate_4w"] = hist
    return out


def fill_training_rows(world: World, res: demand.SimResult, prices: list[np.ndarray], rng: np.random.Generator,
                       replay_id: int, snapshots_per_slot: int = 2) -> pd.DataFrame:
    """Labelled slot rows from one replay: final fill, plus the booked share seen `lead_days` before the slot."""
    slots = slot_grid(world)
    b = res.bookings
    b = b[b["start_ts"] < pd.Timestamp(world.now)]
    key = ["day_idx", "start_cell"]
    final = b[b["status"].isin(BILLED)].groupby(key).size().rename("final_booked_courts")
    slots = slots.join(final, on=key).fillna({"final_booked_courts": 0})
    avail = slots["courts_available"].clip(lower=1)
    slots["fill_rate"] = (slots["final_booked_courts"] / avail).clip(upper=1.0)
    slots = add_hist_fill(slots)
    slots["price_chf"] = [float(prices[d][c]) for d, c in zip(slots["day_idx"], slots["start_cell"])]
    slots["price_ratio"] = slots["price_chf"] / slots["base_price_chf"]

    by_slot = {k: g for k, g in b.groupby(key)}
    rows = []
    for rec in slots.itertuples(index=False):
        g = by_slot.get((rec.day_idx, rec.start_cell))
        for lead in rng.integers(0, MAX_LEAD_DAYS + 1, size=snapshots_per_slot):
            snap = pd.Timestamp(datetime.combine(rec.slot_date, time(0)) - timedelta(days=int(lead)))
            on_books = 0
            if g is not None:
                cancelled = g["cancelled_at"]
                on_books = int(((g["booked_at"] < snap) & (cancelled.isna() | (cancelled >= snap))).sum())
            r = rec._asdict()
            r.update(lead_days=int(lead), booked_courts_now=on_books,
                     booked_share_now=min(on_books / max(rec.courts_available, 1), 1.0), replay_id=replay_id)
            rows.append(r)
    return pd.DataFrame(rows)


def fill_features_from_gold(slot_features: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Scoring rows from gold.slot_features (upcoming slots), at the published tariff."""
    s = slot_features.copy()
    s["is_weekend_day"] = (s["day_type"] == "weekend_holiday").astype(int)
    s["tier_code"] = _code(s["tier"], TIER_CODE)
    for c in ("is_school_holiday", "is_public_holiday", "arena_event", "is_rainy"):
        s[c] = s[c].fillna(False).astype(int)
    s["lead_days"] = (pd.to_datetime(s["slot_date"]) - pd.Timestamp(as_of)).dt.days.clip(0, MAX_LEAD_DAYS)
    avail = s["courts_available"].astype(float).clip(lower=1)
    s["booked_share_now"] = (s["booked_courts_now"].fillna(0).astype(float) / avail).clip(upper=1.0)
    s["price_chf"] = s["base_price_chf"].astype(float)
    s["price_ratio"] = 1.0
    return s


def with_price(features: pd.DataFrame, price: float | np.ndarray) -> pd.DataFrame:
    out = features.copy()
    out["price_chf"] = price
    out["price_ratio"] = out["price_chf"] / out["base_price_chf"]
    return out


# ------------------------------------------------------------------------------------------------ churn

CHURN_FEATURES = [
    "segment_code", "skill_code", "tenure_days", "plays_last_30d", "plays_prev_30d", "plays_last_90d",
    "avg_monthly_plays_baseline", "days_since_last_play", "bookings_as_booker_90d", "avg_lead_time_h",
    "late_cancel_rate", "partners_90d",
]
CHURN_LABEL = "churned_next_30d"


def churn_snapshot(plays: pd.DataFrame, bookings: pd.DataFrame, players: pd.DataFrame, t: date,
                   with_label: bool = True) -> pd.DataFrame:
    """Regulars at snapshot date t, with features from before t and (optionally) the next-30-day churn label.

    plays: player_uuid, booking_id, booking_date (completed participations)
    bookings: booker_uuid, start_ts, lead_time_h, status
    players: player_uuid, segment, skill_tier, joined_at
    """
    t_ts = pd.Timestamp(t)
    p = plays.assign(days_ago=(t_ts - pd.to_datetime(plays["booking_date"])).dt.days)
    past = p[p["days_ago"] >= 1]
    d = past["days_ago"]
    flags = pd.DataFrame({
        "player_uuid": past["player_uuid"],
        "plays_last_30d": (d <= 30).astype(int),
        "plays_prev_30d": ((d > 30) & (d <= 60)).astype(int),
        "plays_last_90d": (d <= 90).astype(int),
        "avg_monthly_plays_baseline": ((d > 30) & (d <= 120)).astype(int),
    })
    feats = flags.groupby("player_uuid").sum()
    feats["avg_monthly_plays_baseline"] = feats["avg_monthly_plays_baseline"] / 3.0
    feats["days_since_last_play"] = d.groupby(past["player_uuid"]).min()
    feats = feats[feats["avg_monthly_plays_baseline"] >= CHURN_MIN_MONTHLY_PLAYS]

    recent = past[past["days_ago"] <= 90][["booking_id", "player_uuid"]]
    pairs = recent.merge(recent, on="booking_id", suffixes=("", "_other"))
    pairs = pairs[pairs["player_uuid"] != pairs["player_uuid_other"]]
    feats["partners_90d"] = pairs.groupby("player_uuid")["player_uuid_other"].nunique()

    bk = bookings[pd.to_datetime(bookings["start_ts"]) < t_ts]
    bk = bk[(t_ts - pd.to_datetime(bk["start_ts"])).dt.days <= 90]
    bg = bk.assign(_late=bk["status"].isin(LATE_OUTCOMES).astype(float)).groupby("booker_uuid")
    feats["bookings_as_booker_90d"] = bg.size()
    feats["avg_lead_time_h"] = bg["lead_time_h"].mean()
    feats["late_cancel_rate"] = bg["_late"].mean()

    pl = players.set_index("player_uuid")
    feats = feats.join(pl[["segment", "skill_tier", "joined_at"]], how="left")
    feats["segment_code"] = _code(feats["segment"], SEGMENT_CODE)
    feats["skill_code"] = _code(feats["skill_tier"], SKILL_CODE)
    feats["tenure_days"] = (t_ts - pd.to_datetime(feats["joined_at"])).dt.days
    feats = feats.fillna({"partners_90d": 0, "bookings_as_booker_90d": 0, "late_cancel_rate": 0.0})
    feats["snapshot_date"] = t

    if with_label:
        ahead = p[(p["days_ago"] <= 0) & (p["days_ago"] > -30)].groupby("player_uuid").size()
        nxt = ahead.reindex(feats.index, fill_value=0)
        feats["plays_next_30d"] = nxt
        feats[CHURN_LABEL] = (nxt <= CHURN_DROP_RATIO * feats["avg_monthly_plays_baseline"]).astype(int)
    return feats.reset_index().rename(columns={"index": "player_uuid"})


def churn_snapshot_dates(as_of: date, window_start: date, step_days: int = 7) -> list[date]:
    """Training snapshots: the 120-day baseline must fit in the window and the 30-day label must end by as_of."""
    first = window_start + timedelta(days=120)
    last = as_of - timedelta(days=30)
    out, t = [], last
    while t >= first:
        out.append(t)
        t -= timedelta(days=step_days)
    return sorted(out)


# ------------------------------------------------------------------------------------------------ cancellation risk

CANCEL_FEATURES = [
    "lead_time_h", "segment_code", "tier_code", "day_of_week", "start_hour", "party_size", "payment_code",
    "is_rainy", "booker_prior_bookings", "booker_prior_late_rate",
]
CANCEL_LABEL = "late_cancel_or_no_show"


def cancel_rows(bookings: pd.DataFrame, players: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """One row per booking still on the books 24 h before start, with booker history known at booking time.

    bookings: booking_id, booker_uuid, start_ts, booked_at, lead_time_h, tier, party_size, payment_method, status
    weather: date, is_rainy
    """
    b = bookings[bookings["status"] != "cancelled_early"].copy()
    b["start_ts"] = pd.to_datetime(b["start_ts"])
    b["booked_at"] = pd.to_datetime(b["booked_at"])
    b[CANCEL_LABEL] = b["status"].isin(LATE_OUTCOMES).astype(int)

    # Booker history: outcomes of their bookings that started before this booking was made.
    hist = b[b["status"] != "upcoming"][["booker_uuid", "start_ts", CANCEL_LABEL]].sort_values("start_ts")
    hist["_n"] = hist.groupby("booker_uuid").cumcount() + 1
    hist["_late"] = hist.groupby("booker_uuid")[CANCEL_LABEL].cumsum()
    b = b.sort_values("booked_at")
    b = pd.merge_asof(b, hist[["booker_uuid", "start_ts", "_n", "_late"]].rename(columns={"start_ts": "_hist_ts"}),
                      left_on="booked_at", right_on="_hist_ts", by="booker_uuid", allow_exact_matches=False)
    b["booker_prior_bookings"] = b["_n"].fillna(0).astype(int)
    b["booker_prior_late_rate"] = (b["_late"] / b["_n"]).fillna(0.0)

    seg = players.set_index("player_uuid")["segment"]
    b["segment_code"] = _code(b["booker_uuid"].map(seg), SEGMENT_CODE)
    b["tier_code"] = _code(b["tier"], TIER_CODE)
    b["payment_code"] = _code(b["payment_method"], PAYMENT_CODE)
    b["day_of_week"] = b["start_ts"].dt.dayofweek + 1
    b["start_hour"] = b["start_ts"].dt.hour + b["start_ts"].dt.minute / 60.0
    rain = weather.assign(date=pd.to_datetime(weather["date"]).dt.date).set_index("date")["is_rainy"]
    b["is_rainy"] = b["start_ts"].dt.date.map(rain).fillna(False).astype(int)
    return b.drop(columns=["_n", "_late", "_hist_ts"]).reset_index(drop=True)
