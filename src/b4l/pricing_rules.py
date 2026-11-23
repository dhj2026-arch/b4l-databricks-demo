"""Hours, tariffs, passes and dynamic-pricing guardrails (SPEC.md section 3).

Facts come from b4l-fondation.ch/horaires-tarifs; anything marked ASSUMPTION is a demo choice.
Pure Python with no Spark dependency so the generator, pipeline, UC functions, apps and tests share it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

N_COURTS = 16
COURT_IDS = [f"Court_{i:02d}" for i in range(1, N_COURTS + 1)]
SHOW_COURTS = COURT_IDS[:4]  # ASSUMPTION
MAIN_HALL, WEST_HALL = "Main Hall", "West Hall"
HALL_ZONE = {c: (MAIN_HALL if i < 8 else WEST_HALL) for i, c in enumerate(COURT_IDS)}  # ASSUMPTION
SLOT_MINUTES = 30
BOOKING_MINUTES = 60

# Rentable windows (first start, end of play). Starts sit on a 30-minute grid.
WEEKDAY_OPEN = time(8, 0)
WEEKDAY_CLOSE = time(22, 0)
WEEKEND_OPEN = time(9, 30)
WEEKEND_CLOSE = time(19, 0)

SITE_SLOTS_PER_WEEK = 1368

# Tiers (CHF per court per hour, for 2 or 4 players).
DAY = "Day"
WEEKEND = "Weekend"
LUNCH = "Lunch"
EVENING = "Evening"
TIERS = (DAY, WEEKEND, LUNCH, EVENING)
TIER_PRICE_CHF = {DAY: 25.0, WEEKEND: 25.0, LUNCH: 32.0, EVENING: 35.0}
TIER_LABEL_FR = {DAY: "Tarif journée", WEEKEND: "Tarif week-end", LUNCH: "Tarif midi", EVENING: "Tarif soir"}

# Rate categories. Family is a published fact; school/association prices are ASSUMPTION.
STANDARD = "standard"
FAMILY = "family"
SCHOOL = "school"
ASSOCIATION = "association"
RATE_CATEGORIES = (STANDARD, FAMILY, SCHOOL, ASSOCIATION)
FAMILY_PRICE_CHF = 18.0
SCHOOL_PRICE_CHF = 20.0
ASSOCIATION_PRICE_CHF = 20.0
FAMILY_ELIGIBLE_TIERS = (DAY, WEEKEND)

BADMATIN_PRICE_PER_PERSON_CHF = 30.0
BADMATIN_CARD_PRICE_CHF = 300.0
BADMATIN_CARD_SESSIONS = 11
PRIVATE_LESSON_PRICE_CHF = 50.0

# Passes.
CARTE_PREPAYEE = "Carte_Prepayee"
PASS_LIBERTE = "Pass_Liberte"
PASS_CHRONO = "Pass_Chrono"
PASS_PRODUCTS = (CARTE_PREPAYEE, PASS_LIBERTE, PASS_CHRONO)
MIDI = "midi"
SOIR = "soir"

CARTE_PREPAYEE_OPTIONS = {200.0: 230.0, 300.0: 360.0, 500.0: 650.0}  # paid -> credit


@dataclass(frozen=True)
class HourPack:
    product: str
    hours: int
    variant: str
    price_chf: float
    validity_days: int | None

    @property
    def price_per_hour(self) -> float:
        return self.price_chf / self.hours


HOUR_PACKS = [
    # ASSUMPTION: the unlabeled 25/50/100 h pack on the tariffs page is Pass Liberte.
    HourPack(PASS_LIBERTE, 25, MIDI, 675.0, None),
    HourPack(PASS_LIBERTE, 25, SOIR, 750.0, None),
    HourPack(PASS_LIBERTE, 50, MIDI, 1250.0, None),
    HourPack(PASS_LIBERTE, 50, SOIR, 1400.0, None),
    HourPack(PASS_LIBERTE, 100, MIDI, 2300.0, None),
    HourPack(PASS_LIBERTE, 100, SOIR, 2600.0, None),
    HourPack(PASS_CHRONO, 25, MIDI, 500.0, 91),
    HourPack(PASS_CHRONO, 25, SOIR, 575.0, 91),
    HourPack(PASS_CHRONO, 50, MIDI, 900.0, 182),
    HourPack(PASS_CHRONO, 50, SOIR, 1000.0, 182),
]

# ASSUMPTION: a midi pack covers day, weekend and lunch slots; a soir pack covers every slot.
PASS_VARIANT_TIERS = {MIDI: (DAY, WEEKEND, LUNCH), SOIR: TIERS}

# Dynamic pricing guardrails.
PRICE_FLOOR_CHF = 18.0
PRICE_CEILING_CHF = 42.0
MAX_MOVE_PCT = 0.40  # ASSUMPTION
PRICE_STEP_CHF = 1.0

LATE_CANCEL_HOURS = 24


def is_weekend_hours(day_type: str) -> bool:
    return day_type == "weekend_holiday"


def rentable_window(day_type: str) -> tuple[time, time] | None:
    """Rentable window for a calendar day type ('weekday', 'weekend_holiday', 'closed')."""
    if day_type == "closed":
        return None
    if day_type == "weekend_holiday":
        return WEEKEND_OPEN, WEEKEND_CLOSE
    return WEEKDAY_OPEN, WEEKDAY_CLOSE


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def start_times(day_type: str, duration_minutes: int = BOOKING_MINUTES) -> list[time]:
    """All valid booking start times for a day type on the 30-minute grid."""
    window = rentable_window(day_type)
    if window is None:
        return []
    first, close = (_minutes(t) for t in window)
    last = close - duration_minutes
    return [time(m // 60, m % 60) for m in range(first, last + 1, SLOT_MINUTES)]


def cell_starts(day_type: str) -> list[time]:
    """30-minute inventory cells covering the rentable window."""
    window = rentable_window(day_type)
    if window is None:
        return []
    first, close = (_minutes(t) for t in window)
    return [time(m // 60, m % 60) for m in range(first, close, SLOT_MINUTES)]


def court_hours_per_week() -> float:
    """Rentable court-hours in a week with no holidays or blocks."""
    weekday = len(cell_starts("weekday")) * SLOT_MINUTES / 60
    weekend = len(cell_starts("weekend_holiday")) * SLOT_MINUTES / 60
    return N_COURTS * (5 * weekday + 2 * weekend)


def tier_for(start: time, weekday: int, day_type: str) -> str:
    """Base tier set by the booking start time. weekday: 0=Mon .. 6=Sun."""
    if day_type == "weekend_holiday":
        return WEEKEND
    m = _minutes(start)
    if 11 * 60 + 30 <= m < 13 * 60 + 30:
        return LUNCH
    if weekday == 4 and m >= 13 * 60 + 30:
        return WEEKEND
    if m >= 17 * 60:
        return EVENING
    # ASSUMPTION: an 08:00 start pays the day rate even though the published day window opens at 08:30.
    return DAY


def base_price(tier: str, rate_category: str = STANDARD) -> float:
    if rate_category == FAMILY:
        if tier not in FAMILY_ELIGIBLE_TIERS:
            raise ValueError(f"Family rate is not available on the {tier} tier")
        return FAMILY_PRICE_CHF
    if rate_category == SCHOOL:
        return SCHOOL_PRICE_CHF
    if rate_category == ASSOCIATION:
        return ASSOCIATION_PRICE_CHF
    return TIER_PRICE_CHF[tier]


def pass_covers(variant: str, tier: str) -> bool:
    return tier in PASS_VARIANT_TIERS[variant]


def is_repriceable(rate_category: str, payment_product: str | None = None, is_block: bool = False) -> bool:
    """Only standard-rate bookings paid on site or by Carte Prepayee follow the dynamic price."""
    if is_block or rate_category != STANDARD:
        return False
    return payment_product in (None, CARTE_PREPAYEE)


@dataclass(frozen=True)
class GuardrailResult:
    final_price_chf: float
    applied_rule: str
    reason: str


def apply_guardrails(
    tier: str,
    proposed_price: float,
    rate_category: str = STANDARD,
    payment_product: str | None = None,
    is_block: bool = False,
    locked_price: float | None = None,
) -> GuardrailResult:
    """Clamp a proposed price to the demo policy (SPEC.md 3.8)."""
    base = base_price(tier, rate_category)
    if locked_price is not None:
        return GuardrailResult(locked_price, "locked", "Price is locked once the slot is booked")
    if is_block:
        return GuardrailResult(base, "never_reprice_block", "Court blocks are never re-priced")
    if rate_category != STANDARD:
        return GuardrailResult(base, "never_reprice_rate", f"{rate_category} rate is never re-priced")
    if payment_product in (PASS_LIBERTE, PASS_CHRONO):
        return GuardrailResult(base, "pass_protected", "Hour-pass bookings deduct one hour; no re-pricing")

    low, high = price_bounds(tier)
    price = round(proposed_price / PRICE_STEP_CHF) * PRICE_STEP_CHF
    if price < low:
        return GuardrailResult(low, "floor", f"Raised to the floor of CHF {low:.2f}")
    if price > high:
        return GuardrailResult(high, "ceiling", f"Capped at the ceiling of CHF {high:.2f}")
    return GuardrailResult(price, "within_bounds", "Within guardrails")


def price_bounds(tier: str) -> tuple[float, float]:
    """Allowed dynamic price range for a tier, snapped inwards to the price step."""
    base = TIER_PRICE_CHF[tier]
    low = max(PRICE_FLOOR_CHF, base * (1 - MAX_MOVE_PCT))
    high = min(PRICE_CEILING_CHF, base * (1 + MAX_MOVE_PCT))
    return math.ceil(low / PRICE_STEP_CHF) * PRICE_STEP_CHF, math.floor(high / PRICE_STEP_CHF) * PRICE_STEP_CHF


def candidate_prices(tier: str) -> list[float]:
    low, high = price_bounds(tier)
    n = int(round((high - low) / PRICE_STEP_CHF))
    return [low + i * PRICE_STEP_CHF for i in range(n + 1)]


def is_late_cancellation(cancelled_at: datetime, start_ts: datetime) -> bool:
    return start_ts - cancelled_at < timedelta(hours=LATE_CANCEL_HOURS)


def slot_key(d: date, start: time) -> str:
    return f"{d.isoformat()}T{start.strftime('%H%M')}"
