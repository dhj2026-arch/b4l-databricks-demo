"""Assemble bronze tables from the simulated world (SPEC.md section 4).

`generate` returns clean bronze frames, the data-quality report and the world; `inject_dirty_rows`
adds the ~1% defects that the Lakeflow expectations are meant to catch.
"""

from __future__ import annotations

import uuid
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd

from b4l import demand, schemas, synth, validate
from b4l import pricing_rules as pr
from b4l.synth import World, rng_for


def generate(seed: int, as_of, n_players: int = 3000):
    world = synth.build_world(seed, as_of, n_players)
    intents = demand.build_intents(world)
    res = demand.simulate(world, intents)
    bronze = build_bronze(world, res)
    checks = validate.run_checks(world, bronze, res)
    checks = pd.concat([checks, demo_moment_checks(world, bronze)], ignore_index=True)
    run_id = uuid.uuid4().hex[:12]
    generated_at = datetime.now().replace(microsecond=0)
    checks.insert(0, "run_ts", generated_at)
    checks.insert(0, "run_id", run_id)
    bronze["data_quality_report"] = checks
    bronze["run_info"] = pd.DataFrame([{
        "run_id": run_id, "seed": int(seed), "as_of_date": world.as_of, "window_start": world.window_start,
        "window_end": world.window_end, "generated_at": generated_at,
    }])
    return bronze, world, res


def build_bronze(world: World, res: demand.SimResult) -> dict[str, pd.DataFrame]:
    players = world.players
    pid = players["player_id"].to_numpy()
    b = res.bookings.copy()
    b["booker_player_id"] = pid[b["player_idx"].to_numpy()]
    b["status"] = b["status"].astype(str)
    out: dict[str, pd.DataFrame] = {}

    out["courts"] = pd.DataFrame({
        "court_id": pr.COURT_IDS,
        "court_number": range(1, pr.N_COURTS + 1),
        "hall_zone": [pr.HALL_ZONE[c] for c in pr.COURT_IDS],
        "is_show_court": [c in pr.SHOW_COURTS for c in pr.COURT_IDS],
    })
    out["players"] = players[schemas.column_names(schemas.BRONZE["players"])].copy()
    out["corporate_accounts"] = world.companies.copy()

    wallets = res.wallets[res.wallets["purchased_at"] <= world.now].copy()
    wallets["player_id"] = pid[wallets["player_idx"].to_numpy()]
    out["pass_wallets"] = wallets
    ledger = res.ledger.copy()
    idx_to_id = dict(zip(b.index, b["booking_id"]))
    ledger["booking_id"] = [idx_to_id.get(int(x)) if x is not None and not pd.isna(x) else None
                            for x in ledger["booking_idx"]]
    ledger = ledger.sort_values("ts").reset_index(drop=True)
    ledger["transaction_id"] = [f"T{i + 1:07d}" for i in range(len(ledger))]
    out["pass_transactions"] = ledger

    blocks = world.blocks.copy()
    out["court_blocks"] = blocks
    b["cancelled_at"] = b["cancelled_at"].where(b["status"].isin(["cancelled_early", "cancelled_late"]))
    out["bookings"] = b
    out["booking_participants"] = _participants(world, b)
    checks = _check_ins(world, b)
    out["check_ins"] = checks
    out["booking_events"] = _events(world, b, checks)
    out["waitlist"] = _waitlist(world, res.waitlist)
    out["weather_lausanne"] = world.weather.copy()
    out["calendar"] = world.calendar.copy()
    out["demo_moments"] = _demo_moments(world, out)
    return out


def _participants(world: World, b: pd.DataFrame) -> pd.DataFrame:
    rng = rng_for(world.seed, 70)
    players = world.players
    pid = players["player_id"].to_numpy()
    group_of = players["group_id"].to_numpy()
    seg = players["segment"].to_numpy()
    skill = players["skill_tier"].to_numpy()
    activity = players["activity"].to_numpy()
    pool = {(s, k): players.index[(players["segment"] == s) & (players["skill_tier"] == k)].to_numpy()
            for s in synth.SEGMENTS for k in synth.SKILLS}
    life_cache: dict[int, np.ndarray] = {}
    rows_b, rows_p, rows_k = [], [], []
    for booking_id, booker, party, day_idx in zip(b["booking_id"], b["player_idx"], b["party_size"], b["day_idx"]):
        if day_idx not in life_cache:
            day = world.days[day_idx]
            life_cache[day_idx] = synth.lifecycle_multiplier(players, day_idx, day.d)
        life = life_cache[day_idx]
        chosen = [int(booker)]
        group = [m for m in world.groups[int(group_of[booker])] if m != booker]
        rng.shuffle(group)
        for m in group:
            if len(chosen) >= party:
                break
            if rng.random() < 0.85 * min(1.0, life[m] * 1.2):
                chosen.append(int(m))
        candidates = pool[(seg[booker], skill[booker])]
        cw = activity[candidates] * life[candidates] + 1e-3
        cw = cw / cw.sum()
        while len(chosen) < party:
            c = int(candidates[rng.choice(len(candidates), p=cw)])
            if c not in chosen:
                chosen.append(c)
        for j, c in enumerate(chosen):
            rows_b.append(booking_id)
            rows_p.append(pid[c])
            rows_k.append(j == 0)
    return pd.DataFrame({"booking_id": rows_b, "player_id": rows_p, "is_booker": rows_k})


def _check_ins(world: World, b: pd.DataFrame) -> pd.DataFrame:
    rng = rng_for(world.seed, 71)
    done = b[b["status"] == "completed"]
    offsets = pd.to_timedelta(rng.uniform(3, 20, len(done)), unit="m")
    return pd.DataFrame({
        "check_in_id": [f"CI{i + 1:06d}" for i in range(len(done))],
        "booking_id": done["booking_id"].to_numpy(),
        "court_id": done["court_id"].to_numpy(),
        "checked_in_at": (done["start_ts"] - offsets).dt.floor("min").to_numpy(),
        "persons": done["party_size"].to_numpy(),
    })


def _events(world: World, b: pd.DataFrame, checks: pd.DataFrame) -> pd.DataFrame:
    rng = rng_for(world.seed, 72)
    frames = [
        pd.DataFrame({"booking_id": b["booking_id"], "event_type": "created", "event_ts": b["booked_at"],
                      "start_ts": b["start_ts"]}),
    ]
    canc = b[b["status"].isin(["cancelled_early", "cancelled_late"])]
    frames.append(pd.DataFrame({"booking_id": canc["booking_id"], "event_type": "cancelled",
                                "event_ts": canc["cancelled_at"], "start_ts": canc["start_ts"]}))
    ns = b[b["status"] == "no_show"]
    frames.append(pd.DataFrame({"booking_id": ns["booking_id"], "event_type": "no_show",
                                "event_ts": ns["start_ts"] + pd.Timedelta(minutes=15), "start_ts": ns["start_ts"]}))
    ci = checks.merge(b[["booking_id", "start_ts"]], on="booking_id")
    frames.append(pd.DataFrame({"booking_id": ci["booking_id"], "event_type": "checked_in",
                                "event_ts": ci["checked_in_at"], "start_ts": ci["start_ts"]}))
    ev = pd.concat(frames, ignore_index=True)
    ev["event_ts"] = pd.to_datetime(ev["event_ts"]).dt.floor("s")
    ev = ev.sort_values(["event_ts", "booking_id"]).reset_index(drop=True)
    ev["hours_before_start"] = ((ev["start_ts"] - ev["event_ts"]).dt.total_seconds() / 3600).round(2)
    delay = pd.to_timedelta(rng.uniform(1, 15, len(ev)), unit="m")
    late = rng.random(len(ev)) < 0.006
    delay = delay.where(~late, pd.to_timedelta(rng.uniform(30, 96, len(ev)), unit="h"))
    ingested = (ev["event_ts"] + delay).clip(upper=pd.Timestamp(world.now) - pd.Timedelta(minutes=1))
    ev["ingested_at"] = ingested.where(ingested > ev["event_ts"], ev["event_ts"]).dt.floor("s")
    ev["event_id"] = [f"E{i + 1:07d}" for i in range(len(ev))]
    return ev[schemas.column_names(schemas.BRONZE["booking_events"])]


def _waitlist(world: World, wl: pd.DataFrame) -> pd.DataFrame:
    if not len(wl):
        return pd.DataFrame(columns=schemas.column_names(schemas.BRONZE["waitlist"]))
    pid = world.players["player_id"].to_numpy()
    rows = []
    for r in wl.itertuples():
        day = world.days[r.day_idx]
        ws, we = day.cell_dt(r.cell_lo), day.cell_dt(r.cell_hi)
        if r.resolved_booking_id is not None:
            status = "resolved"
        elif we < world.now:
            status = "expired"
        else:
            status = "open"
        rows.append({
            "waitlist_id": r.waitlist_id, "player_id": pid[r.player_idx], "requested_date": day.d,
            "window_start_ts": ws, "window_end_ts": we, "party_size": r.party_size, "created_at": r.created_at,
            "channel": r.channel, "status": status, "resolved_booking_id": r.resolved_booking_id,
            "resolved_at": r.resolved_at if r.resolved_booking_id is not None else None,
        })
    return pd.DataFrame(rows)


def _demo_moments(world: World, out: dict[str, pd.DataFrame]) -> pd.DataFrame:
    demo = world.demo
    pid = world.players["player_id"].to_numpy()
    days = world.days
    sat, tue, thu = days[demo["rainy_saturday_idx"]], days[demo["quiet_tuesday_idx"]], days[demo["late_cancel_day_idx"]]
    b = out["bookings"]
    canceller = pid[demo["canceller_idx"]]
    forced = b[(b["booker_player_id"] == canceller) & (b["day_idx"] == thu.idx) & (b["status"] == "upcoming")]
    forced_row = forced.iloc[0] if len(forced) else None
    chrono = out["pass_wallets"]
    chrono = chrono[(chrono["player_id"] == pid[demo["chrono_idx"]]) & (chrono["purchased_at"] == demo["chrono_purchased_at"])]
    rows = [
        {"moment_id": 1, "title": "Rainy Saturday spike",
         "description": "Heavy rain forecast for Saturday: weekend demand is up, courts fill early and the waitlist "
                        "grows. Dynamic pricing can raise the price within the ceiling; passes stay protected.",
         "slot_date": sat.d, "slot_start_ts": datetime.combine(sat.d, time(10, 0)), "court_id": None,
         "player_id": None, "booking_id": None},
        {"moment_id": 2, "title": "Quiet Tuesday lunch",
         "description": "Tuesday lunch is booked well below its usual level. A lower lunch price (Tarif midi) can "
                        "fill empty courts without touching the evening.",
         "slot_date": tue.d, "slot_start_ts": datetime.combine(tue.d, time(12, 0)), "court_id": None,
         "player_id": None, "booking_id": None},
        {"moment_id": 3, "title": "Drifting evening regular",
         "description": "A long-time evening regular has played far less over the last 30 days. Retention offer: "
                        "a CHF 10 stringing voucher or a doubles invite with usual partners.",
         "slot_date": world.as_of, "slot_start_ts": None, "court_id": None,
         "player_id": pid[demo["drifter_idx"]], "booking_id": None},
        {"moment_id": 4, "title": "Likely late cancellation, waitlist backfill",
         "description": "An upcoming Thursday 19:00 booking by a player with a history of late cancellations. If it "
                        "is cancelled, the next waitlisted player is offered the court.",
         "slot_date": thu.d, "slot_start_ts": datetime.combine(thu.d, time(19, 0)),
         "court_id": forced_row["court_id"] if forced_row is not None else None, "player_id": canceller,
         "booking_id": forced_row["booking_id"] if forced_row is not None else None},
        {"moment_id": 5, "title": "Pass Chrono about to expire",
         "description": "A Pass Chrono holder has unused hours and the pass expires within days. Reminder to book "
                        "before the expiry date.",
         "slot_date": chrono["expires_at"].iloc[-1].date() if len(chrono) else world.as_of, "slot_start_ts": None,
         "court_id": None, "player_id": pid[demo["chrono_idx"]], "booking_id": None},
    ]
    return pd.DataFrame(rows)


def demo_moment_checks(world: World, bronze: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    demo = world.demo
    pid = world.players["player_id"].to_numpy()
    b, parts = bronze["bookings"], bronze["booking_participants"]
    now = pd.Timestamp(world.now)
    plays = parts.merge(b[["booking_id", "start_ts", "status"]], on="booking_id")
    plays = plays[plays["status"] == "completed"]

    def count(player, lo, hi):
        p = plays[plays["player_id"] == player]
        return int(((p["start_ts"] >= now - pd.Timedelta(days=lo)) & (p["start_ts"] < now - pd.Timedelta(days=hi))).sum())

    drifter = pid[demo["drifter_idx"]]
    prev, last = count(drifter, 60, 30), count(drifter, 30, 0)
    ok = prev >= 5 and last <= 0.65 * prev
    validate._check(rows, "demo_drifter_drop", "hard_rule", ok, f"prev30={prev} last30={last}", "prev>=5, drop>35%")

    canceller = pid[demo["canceller_idx"]]
    mine = b[b["booker_player_id"] == canceller]
    late = int(mine["status"].isin(["cancelled_late", "no_show"]).sum())
    upcoming = int(((mine["day_idx"] == demo["late_cancel_day_idx"]) & (mine["status"] == "upcoming")).sum())
    validate._check(rows, "demo_late_canceller", "hard_rule", late >= 2 and upcoming >= 1,
                    f"late_or_no_show={late} upcoming_thu={upcoming}", "history >= 2 and an upcoming Thursday booking")

    w = bronze["pass_wallets"]
    w = w[(w["player_id"] == pid[demo["chrono_idx"]]) & (w["purchased_at"] == demo["chrono_purchased_at"])]
    days_left = (w["expires_at"].iloc[-1] - now).days if len(w) else -1
    hours = int(w["hours_remaining"].iloc[-1]) if len(w) else 0
    validate._check(rows, "demo_chrono_near_expiry", "hard_rule", 2 <= days_left <= 10 and hours >= 3,
                    f"days_left={days_left} hours={hours}", "2-10 days, >= 3 hours")

    live = b[b["status"] != "cancelled_early"]

    rainy = world.weather.set_index("date")["is_rainy"]

    def pace(day: synth.DayInfo, lo: time, hi: time, dry_only: bool = False) -> tuple[int, float]:
        """Bookings on the books now for the window vs the usual count at the same lead on past same weekdays."""
        lead = pd.Timestamp(world.now) - pd.Timestamp(day.d)
        t = live["start_ts"].dt.time
        in_window = (t >= lo) & (t < hi)
        on_books = int((in_window & (live["start_ts"].dt.date == day.d)).sum())
        past = [d.d for d in world.days if d.weekday == day.weekday and not d.is_future and d.n_cells
                and not d.is_school_holiday and not d.is_public_holiday and not (dry_only and rainy[d.d])]
        usual = [int((in_window & (live["start_ts"].dt.date == d) & (live["booked_at"] < pd.Timestamp(d) + lead)).sum())
                 for d in past]
        return on_books, float(np.mean(usual)) if usual else 0.0

    tue_now, tue_usual = pace(world.days[demo["quiet_tuesday_idx"]], time(11, 30), time(13, 30))
    validate._check(rows, "demo_quiet_tuesday_lunch_behind_pace", "hard_rule", tue_now < 0.75 * tue_usual,
                    f"on_books={tue_now} usual_at_same_lead={tue_usual:.1f}", "< 0.75x usual pace")
    sat_now, sat_usual = pace(world.days[demo["rainy_saturday_idx"]], time(9, 30), time(19, 0), dry_only=True)
    validate._check(rows, "demo_rainy_saturday_ahead_of_pace", "hard_rule", sat_now > 1.15 * sat_usual,
                    f"on_books={sat_now} usual_dry_at_same_lead={sat_usual:.1f}", "> 1.15x usual dry-Saturday pace")
    return pd.DataFrame(rows)


def inject_dirty_rows(bronze: dict[str, pd.DataFrame], seed: int) -> dict[str, pd.DataFrame]:
    """Add ~1% defects for the Lakeflow expectations (duplicates, malformed contacts, unknown courts)."""
    rng = rng_for(seed, 90)
    out = dict(bronze)
    ev = bronze["booking_events"]
    dup = ev.sample(frac=0.005, random_state=int(rng.integers(1 << 31)))
    dup = dup.assign(ingested_at=dup["ingested_at"] + pd.Timedelta(minutes=7))
    out["booking_events"] = pd.concat([ev, dup], ignore_index=True)

    players = bronze["players"].copy()
    bad_email = rng.random(len(players)) < 0.01
    players.loc[bad_email, "email"] = players.loc[bad_email, "email"].str.replace("@", " at ", regex=False)
    bad_phone = rng.random(len(players)) < 0.008
    players.loc[bad_phone, "phone"] = players.loc[bad_phone, "phone"].str.slice(0, 7)
    out["players"] = players

    b = bronze["bookings"]
    bad = b[b["status"] == "completed"].sample(n=max(5, int(len(b) * 0.002)), random_state=int(rng.integers(1 << 31)))
    bad = bad.copy()
    bad["booking_id"] = [f"BX{i + 1:05d}" for i in range(len(bad))]
    bad.loc[bad.index[: len(bad) // 2], "court_id"] = "Court_99"
    bad.loc[bad.index[len(bad) // 2:], "party_size"] = 0
    out["bookings"] = pd.concat([b, bad], ignore_index=True)
    return out


def to_schema(name: str, df: pd.DataFrame) -> pd.DataFrame:
    cols = schemas.column_names(schemas.BRONZE[name])
    out = df.reindex(columns=cols)
    for col, dtype, _ in schemas.BRONZE[name]:
        if dtype == "TIMESTAMP":
            out[col] = pd.to_datetime(out[col])
        elif dtype == "DATE":
            out[col] = pd.to_datetime(out[col]).dt.date
        elif dtype in ("INT", "BIGINT"):
            out[col] = pd.array(out[col], dtype="Int64")
        elif dtype == "DOUBLE":
            out[col] = pd.to_numeric(out[col]).astype("float64")
        elif dtype == "BOOLEAN":
            out[col] = out[col].astype("boolean")
        elif dtype == "STRING":
            out[col] = out[col].astype("string")
    return out
