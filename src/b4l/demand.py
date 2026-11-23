"""Deterministic demand simulator shared by the generator (static tariff) and the backtest (dynamic prices).

Booking intents (who wants to play, when, how early they book, how much they value it) are drawn once
from the seed and never depend on prices. `simulate` then allocates courts chronologically for a given
price grid. With the static tariff it produces the history; with another grid it produces the
counterfactual, holding players, weather, calendar and every random draw fixed.

Utility of an option j for an intent with desired start s:
    surplus_j = (mu + sigma * z) - ln(tariff_j / tariff_s) - c * |hours shifted| - ln(price_j / tariff_j) / k
The elasticity scenario k only scales the response to deviations from the published tariff, so all
scenarios reproduce the same static history.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from b4l import pricing_rules as pr
from b4l import synth
from b4l.synth import DayInfo, World, rng_for

SHIFT_COST_PER_HOUR = 0.35
FLEX_OFFSETS = (0, -1, 1, -2, 2, -3, 3, -4, 4)  # in 30-minute cells

ELASTICITY_SCENARIOS = {"low": 1.5, "mid": 1.0, "high": 0.67}

# Intents per clock hour across all segments in term time at season factor 1.0.
WEEKDAY_PROFILE = {8: 5.0, 9: 6.5, 10: 7.5, 11: 9.0, 12: 12.5, 13: 8.5, 14: 6.5, 15: 7.5, 16: 9.5,
                   17: 16.0, 18: 23.5, 19: 24.5, 20: 22.0, 21: 12.0}
WEEKEND_PROFILE = {9: 8.0, 10: 15.0, 11: 16.0, 12: 13.0, 13: 13.0, 14: 15.0, 15: 16.0, 16: 16.0, 17: 14.0, 18: 9.0}

BAND_MIX = {
    "morning": {"Student_OffPeak": 0.50, "Club_Member": 0.15, "Evening_Regular": 0.15, "Corporate_Lunch": 0.05,
                "Family_Weekend": 0.15},
    "lunch": {"Corporate_Lunch": 0.65, "Student_OffPeak": 0.15, "Evening_Regular": 0.10, "Club_Member": 0.10},
    "afternoon": {"Student_OffPeak": 0.50, "Family_Weekend": 0.20, "Club_Member": 0.15, "Evening_Regular": 0.15},
    "evening": {"Evening_Regular": 0.60, "Club_Member": 0.20, "Corporate_Lunch": 0.10, "Student_OffPeak": 0.10},
    "weekend": {"Family_Weekend": 0.40, "Evening_Regular": 0.25, "Student_OffPeak": 0.20, "Club_Member": 0.15},
}
BAND_CALIBRATION = {"morning": 1.0, "lunch": 0.82, "afternoon": 0.95, "evening": 1.1, "friday": 1.0, "weekend": 1.05}

SEASON = {1: 1.0, 2: 1.0, 3: 1.0, 4: 0.97, 5: 0.93, 6: 0.92, 7: 0.88, 8: 0.92, 9: 1.0, 10: 1.02, 11: 1.05, 12: 1.0}
HOLIDAY_SEGMENT = {
    "Vacances d'été": {"Corporate_Lunch": 0.55, "Evening_Regular": 0.65, "Family_Weekend": 0.6, "Student_OffPeak": 1.0,
                       "Club_Member": 0.6},
    "default": {"Corporate_Lunch": 0.85, "Evening_Regular": 0.9, "Family_Weekend": 0.85, "Student_OffPeak": 1.15,
                "Club_Member": 0.8},
}

LEAD_PARAMS = {  # (median hours, log sd)
    "Corporate_Lunch": (30.0, 0.6),
    "Evening_Regular": (50.0, 0.8),
    "Family_Weekend": (50.0, 0.7),
    "Student_OffPeak": (10.0, 0.9),
    "Club_Member": (70.0, 0.6),
}
DOUBLES_P = {"Corporate_Lunch": 0.45, "Evening_Regular": 0.6, "Family_Weekend": 0.75, "Student_OffPeak": 0.45,
             "Club_Member": 0.7}
TWO_HOUR_P = {"Corporate_Lunch": 0.02, "Evening_Regular": 0.12, "Family_Weekend": 0.2, "Student_OffPeak": 0.18,
              "Club_Member": 0.25}
WAITLIST_P = {"Corporate_Lunch": 0.2, "Evening_Regular": 0.5, "Family_Weekend": 0.3, "Student_OffPeak": 0.15,
              "Club_Member": 0.45}


def band_of(day: DayInfo, hour: float) -> str:
    if day.day_type == "weekend_holiday":
        return "weekend"
    if 11.5 <= hour < 13.5:
        return "lunch"
    if hour < 11.5:
        return "morning"
    if hour < 17.0:
        return "afternoon"
    return "evening"


@dataclass
class Intents:
    df: pd.DataFrame  # one row per intent, price independent


def _segment_multiplier(world: World, day: DayInfo, weather_row, hour: float) -> dict[str, float]:
    mult = {s: SEASON[day.d.month] for s in synth.SEGMENTS}
    if day.is_school_holiday:
        table = HOLIDAY_SEGMENT.get(day.school_holiday_name, HOLIDAY_SEGMENT["default"])
        for s in mult:
            mult[s] *= table[s]
    weekend = day.day_type == "weekend_holiday"
    daytime = hour < 17.0
    if weather_row.is_rainy:
        f = 1.25 if weekend else (1.08 if daytime else 1.05)
    elif weather_row.temp_max_c >= 28:
        f = 0.75 if weekend else (0.8 if daytime else 0.95)
    elif weekend and weather_row.temp_max_c >= 22:
        f = 0.92
    else:
        f = 1.0
    if day.arena_event and hour >= 18:
        f *= 0.93
    if day.is_public_holiday:
        f *= 0.9
    return {s: v * f for s, v in mult.items()}


def _start_lambda(day: DayInfo, st: time) -> float:
    """Expected intents for one start time (all segments) before segment and weather multipliers."""
    hour = st.hour + st.minute / 60
    if day.day_type == "weekend_holiday":
        base = WEEKEND_PROFILE.get(st.hour, 0.0)
        share = 1.0 if st.hour == 9 else (0.78 if st.minute == 0 else 0.22)
        return base * share * BAND_CALIBRATION["weekend"]
    base = WEEKDAY_PROFILE.get(st.hour, 0.0) * (0.78 if st.minute == 0 else 0.22)
    if day.weekday == 4 and hour >= 13.5:
        return base * (0.55 if hour >= 17 else 0.9) * BAND_CALIBRATION["friday"]
    if day.weekday == 0 and hour >= 17:
        base *= 0.95
    return base * BAND_CALIBRATION[band_of(day, hour)]


def build_intents(world: World) -> Intents:
    rng = rng_for(world.seed, 10)
    players = world.players
    seg_idx = {s: players.index[players["segment"] == s].to_numpy() for s in synth.SEGMENTS}
    activity = players["activity"].to_numpy()
    habit_day = players["habit_day"].to_numpy()
    habit_day2 = players["habit_day2"].to_numpy()
    habit_hour = players["habit_hour"].to_numpy()
    weather = world.weather.set_index("date")
    demo = world.demo
    rows = []
    for day in world.days:
        if day.n_cells == 0:
            continue
        wrow = weather.loc[day.d]
        life = synth.lifecycle_multiplier(players, day.idx, day.d)
        base_w = activity * life
        starts = pr.start_times(day.day_type)
        for st in starts:
            hour = st.hour + st.minute / 60
            lam_start = _start_lambda(day, st)
            band = band_of(day, hour)
            mix = dict(BAND_MIX[band])
            if day.is_school_holiday and band in ("morning", "afternoon"):
                mix["Family_Weekend"] = mix.get("Family_Weekend", 0) + 0.25
            smult = _segment_multiplier(world, day, wrow, hour)
            demo_mult = demo.get("demand_multiplier", {}).get((day.idx, band), 1.0)
            if demo.get("rainy_saturday_idx") == day.idx:
                demo_mult *= 1.5
            for seg, mshare in mix.items():
                lam = lam_start * mshare * smult[seg] * demo_mult
                n = rng.poisson(lam)
                if n == 0:
                    continue
                idx = seg_idx[seg]
                hd = (habit_day[idx] == day.weekday) | (habit_day2[idx] == day.weekday)
                hmatch = np.where(hd, 1.0 + 3.0 * np.exp(-((habit_hour[idx] - hour) / 0.6) ** 2), 1.0)
                w = base_w[idx] * hmatch
                tot = w.sum()
                if tot <= 0:
                    continue
                chosen = rng.choice(idx, size=n, p=w / tot)
                for pidx in chosen:
                    habit_match = bool(hd[np.searchsorted(idx, pidx)]) and abs(habit_hour[pidx] - hour) <= 1.0
                    rows.append(_make_intent(rng, world, day, st, int(pidx), seg, habit_match))
        rows.extend(_group_habit_intents(rng, world, day, wrow, life))
    for forced in demo.get("forced_intents", []):
        day = world.days[forced["day_idx"]]
        row = _make_intent(rng, world, day, forced["start"], forced["player_idx"],
                           players.at[forced["player_idx"], "segment"], True)
        row.update(forced.get("overrides", {}))
        row["lead_h"] = (day.cell_dt(row["start_cell"]) - row["booked_at"]).total_seconds() / 3600
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(["booked_at", "day_idx", "start_cell"], kind="stable").reset_index(drop=True)
    df["intent_id"] = np.arange(len(df))
    return Intents(df)


def _group_habit_intents(rng, world: World, day: DayInfo, wrow, life: np.ndarray) -> list[dict]:
    """Weekly fixed-slot bookings of regular groups; a group skips when members drift away."""
    gh = world.group_habits
    if gh.empty:
        return []
    valid = {st.hour + st.minute / 60: st for st in pr.start_times(day.day_type)}
    activity = world.players["activity"].to_numpy()
    rows = []
    for g in gh[gh["weekday"] == day.weekday].itertuples():
        st = valid.get(g.hour)
        if st is None:
            continue
        members = world.groups[g.group_id]
        member_life = life[members]
        if member_life.max() <= 0:
            continue
        smult = min(1.0, _segment_multiplier(world, day, wrow, g.hour)[g.segment])
        p = g.attend_p * smult * min(1.0, float(np.sort(member_life)[-2:].mean()) if len(members) > 1 else member_life[0])
        p *= world.demo.get("demand_multiplier", {}).get((day.idx, band_of(day, g.hour)), 1.0)
        if world.demo.get("rainy_saturday_idx") == day.idx:
            p = min(0.98, p * 1.3)
        if rng.random() >= p:
            continue
        w = activity[members] * member_life ** 3 + 1e-9
        booker = int(members[int(rng.choice(len(members), p=w / w.sum()))])
        row = _make_intent(rng, world, day, st, booker, g.segment, True)
        row["flexible"] = False
        two_hours = g.segment == "Club_Member" and rng.random() < 0.3 and row["start_cell"] + 4 <= day.n_cells
        row["dur_cells"] = 4 if two_hours else 2
        row["party_size"] = 4 if (len(members) >= 4 and rng.random() < 0.85) else 2
        lead = float(np.clip(rng.lognormal(math.log(130.0), 0.35), 24.0, 336.0))
        row["lead_h"] = lead
        row["booked_at"] = day.cell_dt(row["start_cell"]) - timedelta(hours=lead)
        rows.append(row)
    return rows


def _make_intent(rng, world: World, day: DayInfo, st: time, pidx: int, seg: str, habit_match: bool) -> dict:
    start_cell = ((st.hour * 60 + st.minute) - day.open_min) // pr.SLOT_MINUTES
    dur_cells = 4 if (rng.random() < TWO_HOUR_P[seg] and start_cell + 4 <= day.n_cells) else 2
    med, sd = LEAD_PARAMS[seg]
    if seg == "Evening_Regular" and habit_match:
        med, sd = 110.0, 0.5
    if world.demo.get("rainy_saturday_idx") == day.idx:
        med *= 1.8  # a heavy-rain forecast makes players secure courts earlier
    lead_h = float(np.clip(rng.lognormal(math.log(med), sd), 0.5, 336.0))
    start_dt = day.cell_dt(start_cell)
    booked_at = start_dt - timedelta(hours=lead_h)
    p = world.players
    family_rate = seg == "Family_Weekend" and rng.random() < 0.7
    ch = rng.random()
    phone_p = 0.25 if seg == "Family_Weekend" else 0.13
    channel = "phone" if ch < phone_p else ("app" if ch < phone_p + 0.32 else "online")
    return {
        "day_idx": day.idx,
        "start_cell": int(start_cell),
        "dur_cells": int(dur_cells),
        "player_idx": pidx,
        "segment": seg,
        "party_size": 4 if rng.random() < DOUBLES_P[seg] else 2,
        "family_rate": family_rate,
        "lead_h": lead_h,
        "booked_at": booked_at,
        "wtp_z": float(0.6 * p.at[pidx, "wtp_z"] + 0.8 * rng.normal()),
        "flexible": bool(p.at[pidx, "is_flexible"]),
        "channel": channel,
        "u_cancel": float(rng.random()),
        "u_cancel_time": float(rng.random()),
        "u_waitlist": float(rng.random()),
        "u_wait_accept": float(rng.random()),
        "u_court": float(rng.random()),
        "u_aux": float(rng.random()),
        "force_book": False,
        "cancel_mult": 1.0,
    }


def static_price_grid(world: World) -> list[np.ndarray]:
    grid = []
    for day in world.days:
        arr = np.full(day.n_cells, np.nan)
        for c in range(day.n_cells):
            tier = pr.tier_for(day.cell_time(c), day.weekday, day.day_type)
            arr[c] = pr.TIER_PRICE_CHF[tier]
        grid.append(arr)
    return grid


def tier_grid(world: World) -> list[list[str]]:
    return [[pr.tier_for(day.cell_time(c), day.weekday, day.day_type) for c in range(day.n_cells)]
            for day in world.days]


@dataclass
class Wallet:
    wallet_id: str
    player_idx: int
    product: str
    variant: str | None
    purchase_price: float
    credit: float
    hours_total: int
    purchased_at: datetime
    expires_at: datetime | None
    balance: float
    hours_left: int
    init_balance: float = 0.0
    init_hours: int = 0

    def __post_init__(self):
        self.init_balance = self.balance
        self.init_hours = self.hours_left

    def value_per_hour(self) -> float:
        return self.purchase_price / self.hours_total if self.hours_total else 0.0

    def credit_ratio(self) -> float:
        return self.purchase_price / self.credit if self.credit else 1.0


@dataclass
class SimResult:
    bookings: pd.DataFrame
    waitlist: pd.DataFrame
    ledger: pd.DataFrame
    wallets: pd.DataFrame
    turned_away: pd.DataFrame
    occupancy: list[np.ndarray] = field(repr=False, default_factory=list)


class _WalletBook:
    """Tracks pass products per player during the chronological simulation."""

    def __init__(self, world: World):
        self.world = world
        self.current: dict[int, Wallet] = {}
        self.pending: dict[int, list[Wallet]] = {}
        self.all: list[Wallet] = []
        self.ledger: list[dict] = []
        self.renewals: dict[int, int] = {}
        self.product_of: dict[int, str] = {}
        rng = rng_for(world.seed, 6)
        for w in world.initial_wallets.itertuples():
            wallet = self._new_wallet(w.player_idx, w.product, w.variant, w.purchased_at, w.size_u, w.used_fraction)
            self.product_of[w.player_idx] = w.product
            if wallet.purchased_at <= world.now - timedelta(days=synth.HISTORY_DAYS):
                self.current[w.player_idx] = wallet
            else:
                self.pending.setdefault(w.player_idx, []).append(wallet)
        del rng

    def _new_wallet(self, pidx, product, variant, purchased_at, size_u, used_fraction=0.0) -> Wallet:
        wid = f"W{len(self.all) + 1:06d}"
        if product == pr.CARTE_PREPAYEE:
            paid = [200.0, 300.0, 500.0][0 if size_u < 0.45 else (1 if size_u < 0.8 else 2)]
            credit = pr.CARTE_PREPAYEE_OPTIONS[paid]
            wallet = Wallet(wid, pidx, product, None, paid, credit, 0, purchased_at, None,
                            round(credit * (1 - used_fraction), 2), 0)
        else:
            packs = [p for p in pr.HOUR_PACKS if p.product == product and p.variant == variant]
            pack = packs[min(int(size_u * len(packs)), len(packs) - 1)] if product == pr.PASS_LIBERTE else (
                packs[0] if size_u < 0.7 else packs[-1])
            expires = purchased_at + timedelta(days=pack.validity_days) if pack.validity_days else None
            wallet = Wallet(wid, pidx, product, variant, pack.price_chf, 0.0, pack.hours, purchased_at, expires, 0.0,
                            int(round(pack.hours * (1 - used_fraction))))
        self.all.append(wallet)
        return wallet

    def wallet_at(self, pidx: int, ts: datetime) -> Wallet | None:
        pend = self.pending.get(pidx)
        if pend:
            while pend and pend[0].purchased_at <= ts:
                self.current[pidx] = pend.pop(0)
        return self.current.get(pidx)

    def covers(self, pidx: int, tier: str, booked_at: datetime, start: datetime, hours: int) -> Wallet | None:
        w = self.wallet_at(pidx, booked_at)
        if w is None or w.product == pr.CARTE_PREPAYEE:
            return None
        if not pr.pass_covers(w.variant, tier):
            return None
        if w.expires_at is not None and start > w.expires_at:
            return None
        return w if w.hours_left >= hours else None

    def renew_draw(self, pidx: int) -> float:
        n = self.renewals.get(pidx, 0)
        self.renewals[pidx] = n + 1
        return float(rng_for(self.world.seed, 50, pidx, n).random())

    def pay(self, pidx: int, tier: str, booked_at: datetime, start: datetime, hours: int, price_total: float):
        """Returns (method, wallet, price_paid, revenue). Renews products when exhausted."""
        w = self.wallet_at(pidx, booked_at)
        if w is not None and w.product in (pr.PASS_LIBERTE, pr.PASS_CHRONO):
            expired = w.expires_at is not None and start > w.expires_at
            if pr.pass_covers(w.variant, tier) and not expired and w.hours_left >= hours:
                w.hours_left -= hours
                return "pass", w, 0.0, round(w.value_per_hour() * hours, 2)
            if expired or w.hours_left < hours:
                renew_p = 0.4 if (expired and w.hours_left > 0) else 0.7
                if w.product == pr.PASS_LIBERTE:
                    renew_p = 0.8
                if self.renew_draw(pidx) < renew_p:
                    nw = self._new_wallet(pidx, w.product, w.variant, booked_at, 0.3 if w.hours_total <= 25 else 0.9)
                    self.current[pidx] = nw
                    if pr.pass_covers(nw.variant, tier):
                        nw.hours_left -= hours
                        return "pass", nw, 0.0, round(nw.value_per_hour() * hours, 2)
                else:
                    self.current.pop(pidx, None)
            return "on_site", None, price_total, price_total
        if w is not None and w.product == pr.CARTE_PREPAYEE:
            if w.balance < price_total:
                if self.renew_draw(pidx) < 0.85:
                    nw = self._new_wallet(pidx, w.product, None, booked_at, 0.3 if w.purchase_price <= 200 else 0.9)
                    self.current[pidx] = nw
                    w = nw
                else:
                    self.current.pop(pidx, None)
                    return "on_site", None, price_total, price_total
            w.balance = round(w.balance - price_total, 2)
            return "prepaid", w, price_total, round(price_total * w.credit_ratio(), 2)
        return "on_site", None, price_total, price_total

    def refund(self, wallet: Wallet, hours: int, price_paid: float):
        if wallet.product == pr.CARTE_PREPAYEE:
            wallet.balance = round(wallet.balance + price_paid, 2)
        else:
            wallet.hours_left += hours


def _cancel_plan(row, start: datetime, booked_at: datetime, has_pass: bool, rainy: bool, player_mult: float):
    """Returns (kind, ts) with kind in None/'early'/'late'/'no_show'."""
    lead = (start - booked_at).total_seconds() / 3600
    seg = row.segment
    m = player_mult * row.cancel_mult
    p_early = 0.075 * min(1.0, max(0.0, lead - 24) / 72) * m
    p_late = (0.042 + (0.0 if has_pass else 0.014) + (0.01 if seg in ("Student_OffPeak", "Corporate_Lunch") else 0.0)) * m
    p_ns = (0.02 + (0.015 if lead > 96 else 0.0) - (0.008 if has_pass else 0.0)) * m
    if rainy:
        p_late *= 0.8
        p_ns *= 0.8
    u = row.u_cancel
    if u < p_early:
        span = (start - timedelta(hours=24) - booked_at).total_seconds()
        return "early", booked_at + timedelta(seconds=row.u_cancel_time * span)
    if u < p_early + p_late:
        lo = max(booked_at, start - timedelta(hours=24))
        span = max(0.0, (start - timedelta(minutes=30) - lo).total_seconds())
        return "late", lo + timedelta(seconds=row.u_cancel_time * span)
    if u < p_early + p_late + p_ns:
        return "no_show", start + timedelta(minutes=15)
    return None, None


def simulate(world: World, intents: Intents, prices: list[np.ndarray] | None = None,
             elasticity_k: float = 1.0) -> SimResult:
    static = static_price_grid(world)
    prices = static if prices is None else prices
    tiers = tier_grid(world)
    now = world.now
    days = world.days
    weather_rainy = world.weather.set_index("date")["is_rainy"].to_dict()
    players = world.players
    player_cancel_mult = players["cancel_propensity"].to_numpy() if "cancel_propensity" in players else np.ones(len(players))

    occ = [np.zeros((pr.N_COURTS, max(d.n_cells, 1)), dtype=np.int32) for d in days]
    court_index = {c: i for i, c in enumerate(pr.COURT_IDS)}
    for b in world.blocks.itertuples():
        day = days[b.day_idx]
        c0 = ((b.start_ts.hour * 60 + b.start_ts.minute) - day.open_min) // pr.SLOT_MINUTES
        c1 = ((b.end_ts.hour * 60 + b.end_ts.minute) - day.open_min) // pr.SLOT_MINUTES
        occ[b.day_idx][court_index[b.court_id], max(c0, 0):max(c1, 0)] = -1

    wallets = _WalletBook(world)
    bookings: list[dict] = []
    waitlist: list[dict] = []
    open_waitlist: dict[int, list[int]] = {}
    turned_away: dict[tuple[int, int], int] = {}
    player_busy: dict[tuple[int, int], list[tuple[int, int]]] = {}
    events: list = []
    seq = 0
    recs = list(intents.df.itertuples(index=False))
    for i, row in enumerate(recs):
        if row.booked_at < now:
            heapq.heappush(events, (row.booked_at, 1, seq, "arrival", i))
            seq += 1
    wallet_by_id: dict[str, Wallet] = {}

    def free_courts(day_idx: int, cell: int, dur: int) -> np.ndarray:
        block = occ[day_idx][:, cell:cell + dur]
        return np.flatnonzero((block == 0).all(axis=1))

    def pick_court(day_idx: int, cell: int, dur: int, u: float) -> int | None:
        free = free_courts(day_idx, cell, dur)
        if len(free) == 0:
            return None
        n = days[day_idx].n_cells
        grid = occ[day_idx]
        before = (grid[free, cell - 1] != 0) if cell > 0 else np.ones(len(free), bool)
        after = (grid[free, cell + dur] != 0) if cell + dur < n else np.ones(len(free), bool)
        score = before.astype(int) + after.astype(int)
        rot = (free - int(u * pr.N_COURTS)) % pr.N_COURTS
        order = np.lexsort((rot, -score))
        return int(free[order[0]])

    def book(row, day_idx: int, cell: int, court: int, ts: datetime, source_wl: int | None) -> int:
        day = days[day_idx]
        dur = row.dur_cells
        hours = dur // 2
        start = day.cell_dt(cell)
        tier = tiers[day_idx][cell]
        family = row.family_rate and tier in pr.FAMILY_ELIGIBLE_TIERS
        rate = pr.FAMILY if family else pr.STANDARD
        base = pr.base_price(tier, rate)
        if family:
            shown = base
            method, wallet, paid, revenue = "on_site", None, base * hours, base * hours
        else:
            pass_wallet = wallets.covers(row.player_idx, tier, ts, start, hours)
            shown = base if pass_wallet is not None else float(prices[day_idx][cell])
            method, wallet, paid, revenue = wallets.pay(row.player_idx, tier, ts, start, hours, shown * hours)
            if method == "pass":
                shown = base
        bidx = len(bookings)
        occ[day_idx][court, cell:cell + dur] = bidx + 1
        player_busy.setdefault((row.player_idx, day_idx), []).append((cell, cell + dur))
        kind, cts = _cancel_plan(row, start, ts, method == "pass", weather_rainy.get(day.d, False),
                                 float(player_cancel_mult[row.player_idx]))
        bookings.append(
            {
                "intent_id": row.intent_id, "day_idx": day_idx, "start_cell": cell, "dur_cells": dur,
                "court_idx": court, "player_idx": row.player_idx, "segment": row.segment,
                "party_size": row.party_size, "rate_category": rate, "tier": tier, "base_price_chf": base,
                "price_shown_chf": shown, "price_paid_chf": round(paid, 2), "revenue_chf": round(revenue, 2),
                "payment_method": method, "wallet_id": wallet.wallet_id if wallet else None,
                "channel": "phone" if source_wl is not None else row.channel, "booked_at": ts,
                "start_ts": start, "cancel_kind": None, "cancelled_at": None, "source_waitlist_idx": source_wl,
                "planned_cancel": kind, "planned_cancel_ts": cts,
            }
        )
        if wallet is not None:
            wallet_by_id[wallet.wallet_id] = wallet
            wallets.ledger.append({"wallet_id": wallet.wallet_id, "booking_idx": bidx, "ts": ts,
                                   "delta_chf": -round(paid, 2) if method == "prepaid" else 0.0,
                                   "delta_hours": -hours if method == "pass" else 0, "reason": "booking"})
        if kind in ("early", "late") and cts < now:
            nonlocal_push(cts, 0, "cancel", bidx)
        return bidx

    def nonlocal_push(ts, prio, kind, payload):
        nonlocal seq
        heapq.heappush(events, (ts, prio, seq, kind, payload))
        seq += 1

    def busy(pidx: int, day_idx: int, cell: int, dur: int) -> bool:
        for a, b in player_busy.get((pidx, day_idx), []):
            if cell < b and a < cell + dur:
                return True
        return False

    while events:
        ts, _, _, kind, payload = heapq.heappop(events)
        if kind == "arrival":
            row = recs[payload]
            day = days[row.day_idx]
            dur = row.dur_cells
            if busy(row.player_idx, row.day_idx, row.start_cell, dur):
                continue
            desired_tier = tiers[row.day_idx][row.start_cell]
            tariff_s = pr.TIER_PRICE_CHF[desired_tier]
            start_desired = day.cell_dt(row.start_cell)
            insensitive = row.force_book or (row.family_rate and desired_tier in pr.FAMILY_ELIGIBLE_TIERS) or (
                wallets.covers(row.player_idx, desired_tier, ts, start_desired, dur // 2) is not None)
            mu, sigma = synth.WTP_PARAMS[row.segment]
            base_surplus = mu + sigma * row.wtp_z
            offsets = FLEX_OFFSETS if (row.flexible and not insensitive) else (0,)
            options = []
            for off in offsets:
                cell = row.start_cell + off
                if cell < 0 or cell + dur > day.n_cells:
                    continue
                if day.cell_dt(cell) <= ts:
                    continue
                if insensitive:
                    s = 10.0
                else:
                    tariff_j = static[row.day_idx][cell]
                    p_j = prices[row.day_idx][cell]
                    s = (base_surplus - math.log(tariff_j / tariff_s) - SHIFT_COST_PER_HOUR * abs(off) / 2
                         - math.log(p_j / tariff_j) / elasticity_k)
                options.append((s, abs(off), cell))
            options.sort(key=lambda o: (-o[0], o[1]))
            wanted = [o for o in options if o[0] >= 0]
            if not wanted:
                continue
            placed = False
            for _, _, cell in wanted:
                court = pick_court(row.day_idx, cell, dur, row.u_court)
                if court is not None:
                    book(row, row.day_idx, cell, court, ts, None)
                    placed = True
                    break
            if not placed:
                key = (row.day_idx, row.start_cell)
                turned_away[key] = turned_away.get(key, 0) + 1
                if row.lead_h >= 2 and row.u_waitlist < WAITLIST_P[row.segment]:
                    span = 2 if row.flexible else 0
                    widx = len(waitlist)
                    waitlist.append({"intent_idx": payload, "player_idx": row.player_idx, "day_idx": row.day_idx,
                                     "cell_lo": max(0, row.start_cell - span),
                                     "cell_hi": min(day.n_cells - dur, row.start_cell + span),
                                     "dur_cells": dur, "party_size": row.party_size, "created_at": ts,
                                     "channel": "phone" if row.u_aux < 0.75 else "app",
                                     "resolved_booking_idx": None, "resolved_at": None})
                    open_waitlist.setdefault(row.day_idx, []).append(widx)
        elif kind == "cancel":
            b = bookings[payload]
            b["cancel_kind"] = b["planned_cancel"]
            b["cancelled_at"] = ts
            day_idx, cell, dur = b["day_idx"], b["start_cell"], b["dur_cells"]
            occ[day_idx][b["court_idx"], cell:cell + dur] = 0
            spans = player_busy.get((b["player_idx"], day_idx), [])
            if (cell, cell + dur) in spans:
                spans.remove((cell, cell + dur))
            if b["cancel_kind"] == "early":
                b["revenue_chf"] = 0.0
                if b["wallet_id"] is not None:
                    w = wallet_by_id[b["wallet_id"]]
                    wallets.refund(w, dur // 2, b["price_paid_chf"])
                    wallets.ledger.append({"wallet_id": w.wallet_id, "booking_idx": payload, "ts": ts,
                                           "delta_chf": b["price_paid_chf"] if b["payment_method"] == "prepaid" else 0.0,
                                           "delta_hours": dur // 2 if b["payment_method"] == "pass" else 0,
                                           "reason": "refund_early_cancel"})
                b["price_paid_chf"] = 0.0
            start = b["start_ts"]
            if (start - ts) >= timedelta(hours=2):
                for widx in list(open_waitlist.get(day_idx, [])):
                    w = waitlist[widx]
                    if w["resolved_booking_idx"] is not None or w["created_at"] > ts:
                        continue
                    if not (w["cell_lo"] <= cell <= w["cell_hi"]) or cell + w["dur_cells"] > days[day_idx].n_cells:
                        continue
                    if not (occ[day_idx][b["court_idx"], cell:cell + w["dur_cells"]] == 0).all():
                        continue
                    wrow = recs[w["intent_idx"]]
                    if wrow.u_wait_accept < 0.65 and not busy(w["player_idx"], day_idx, cell, w["dur_cells"]):
                        delay = timedelta(minutes=10 + 110 * wrow.u_aux)
                        when = min(ts + delay, start - timedelta(minutes=45))
                        if when >= now:
                            continue
                        nonlocal_push(when, 0, "wl_book", (widx, cell, b["court_idx"]))
                        w["resolved_booking_idx"] = -1
                        open_waitlist[day_idx].remove(widx)
                        break
        elif kind == "wl_book":
            widx, cell, court = payload
            w = waitlist[widx]
            day_idx = w["day_idx"]
            if not (occ[day_idx][court, cell:cell + w["dur_cells"]] == 0).all():
                alt = pick_court(day_idx, cell, w["dur_cells"], 0.0)
                if alt is None:
                    w["resolved_booking_idx"] = None
                    open_waitlist[day_idx].append(widx)
                    continue
                court = alt
            wrow = recs[w["intent_idx"]]
            bidx = book(wrow._replace(dur_cells=w["dur_cells"]), day_idx, cell, court, ts, widx)
            w["resolved_booking_idx"] = bidx
            w["resolved_at"] = ts

    for b in bookings:
        if b["cancel_kind"] is None and b["planned_cancel"] == "no_show" and b["start_ts"] < now:
            b["cancel_kind"] = "no_show"
    return _assemble(world, bookings, waitlist, wallets, turned_away, occ)


def _status(b: dict, now: datetime) -> str:
    if b["cancel_kind"] == "early":
        return "cancelled_early"
    if b["cancel_kind"] == "late":
        return "cancelled_late"
    if b["cancel_kind"] == "no_show":
        return "no_show"
    return "completed" if b["start_ts"] < now else "upcoming"


def _assemble(world: World, bookings, waitlist, wallets: _WalletBook, turned_away, occ) -> SimResult:
    now = world.now
    bdf = pd.DataFrame(bookings)
    if len(bdf):
        bdf["status"] = [_status(b, now) for b in bookings]
        bdf["booking_id"] = [f"B{i + 1:06d}" for i in range(len(bdf))]
        bdf["duration_h"] = bdf["dur_cells"] / 2.0
        bdf["end_ts"] = bdf["start_ts"] + pd.to_timedelta(bdf["dur_cells"] * pr.SLOT_MINUTES, unit="m")
        bdf["lead_time_h"] = (bdf["start_ts"] - bdf["booked_at"]).dt.total_seconds() / 3600
        bdf["court_id"] = [pr.COURT_IDS[i] for i in bdf["court_idx"]]
    wdf = pd.DataFrame(waitlist)
    if len(wdf):
        wdf["waitlist_id"] = [f"WL{i + 1:05d}" for i in range(len(wdf))]
        wdf["resolved_booking_id"] = [
            f"B{int(x) + 1:06d}" if x is not None and not pd.isna(x) and x >= 0 else None
            for x in wdf["resolved_booking_idx"]
        ]
        if len(bdf):
            src = {int(b): f"WL{int(w) + 1:05d}" for b, w in zip(bdf.index, bdf["source_waitlist_idx"]) if w is not None and not pd.isna(w)}
            bdf["source_waitlist_id"] = [src.get(i) for i in bdf.index]
    elif len(bdf):
        bdf["source_waitlist_id"] = None
    ledger_rows = list(wallets.ledger)
    wallet_rows = []
    for w in wallets.all:
        expired = w.expires_at is not None and w.expires_at < now
        if expired and w.hours_left > 0:
            ledger_rows.append({"wallet_id": w.wallet_id, "booking_idx": None, "ts": w.expires_at,
                                "delta_chf": 0.0, "delta_hours": -w.hours_left, "reason": "expiry"})
        active_now = w.purchased_at <= now
        status = "expired" if expired else ("exhausted" if (
            (w.product == pr.CARTE_PREPAYEE and w.balance < pr.TIER_PRICE_CHF[pr.DAY]) or
            (w.product != pr.CARTE_PREPAYEE and w.hours_left <= 0)) else "active")
        wallet_rows.append({
            "wallet_id": w.wallet_id, "player_idx": w.player_idx, "product": w.product, "variant": w.variant,
            "purchase_price_chf": w.purchase_price, "credit_chf": w.credit if w.product == pr.CARTE_PREPAYEE else None,
            "hours_total": w.hours_total if w.product != pr.CARTE_PREPAYEE else None, "purchased_at": w.purchased_at,
            "expires_at": w.expires_at,
            "balance_chf_remaining": w.balance if w.product == pr.CARTE_PREPAYEE else None,
            "hours_remaining": (0 if expired else w.hours_left) if w.product != pr.CARTE_PREPAYEE else None,
            "status": status, "snapshot_at": now, "active_now": active_now,
            "initial_balance_chf": w.init_balance, "initial_hours": w.init_hours,
        })
    ta = pd.DataFrame([{"day_idx": k[0], "start_cell": k[1], "turned_away": v} for k, v in turned_away.items()],
                      columns=["day_idx", "start_cell", "turned_away"])
    return SimResult(bdf, wdf, pd.DataFrame(ledger_rows), pd.DataFrame(wallet_rows), ta, occ)
