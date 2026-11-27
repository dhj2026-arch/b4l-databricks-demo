"""Price-independent synthetic world: players, companies, weather, court blocks and pass holders.

Everything here depends only on the seed and the as-of date, so the static and dynamic pricing
replays in demand.py see exactly the same players, weather and calendar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from b4l import calendar as cal
from b4l import pricing_rules as pr

SEGMENTS = ("Corporate_Lunch", "Evening_Regular", "Family_Weekend", "Student_OffPeak", "Club_Member")
SEGMENT_SHARE = {
    "Corporate_Lunch": 0.17,
    "Evening_Regular": 0.31,
    "Family_Weekend": 0.17,
    "Student_OffPeak": 0.21,
    "Club_Member": 0.14,
}
SKILLS = ("Beginner", "Intermediate", "Advanced")
FLAKY_Z = 1.036  # standard-normal cut-off for the top ~15% of players by cancellation propensity

# Willingness to pay: log(WTP / tariff) ~ Normal(mu, sigma). Share booking at the tariff is
# Phi(mu / sigma); point elasticity at the tariff is phi(mu/sigma) / (sigma * Phi(mu/sigma)).
WTP_PARAMS = {
    "Corporate_Lunch": (0.60, 0.50),  # low elasticity (~0.4)
    "Evening_Regular": (0.35, 0.40),  # medium (~0.8)
    "Family_Weekend": (0.20, 0.35),  # medium-high (~1.4)
    "Student_OffPeak": (0.00, 0.30),  # high (~2.7)
    "Club_Member": (0.50, 0.40),  # low-medium (~0.6)
}
FLEXIBLE_SHARE = {
    "Corporate_Lunch": 0.15,
    "Evening_Regular": 0.30,
    "Family_Weekend": 0.40,
    "Student_OffPeak": 0.65,
    "Club_Member": 0.30,
}

FIRST_NAMES_F = [
    "Camille", "Léa", "Chloé", "Manon", "Julie", "Sarah", "Laura", "Emma", "Inès", "Zoé", "Céline",
    "Nathalie", "Sophie", "Isabelle", "Valérie", "Aline", "Mélanie", "Anaïs", "Clara", "Elodie",
    "Mei", "Priya", "Ana", "Giulia", "Fatima", "Lina", "Noémie", "Margaux", "Océane", "Lucie",
]
FIRST_NAMES_M = [
    "Lucas", "Nicolas", "Julien", "Thomas", "Maxime", "Antoine", "Kevin", "Mathieu", "David", "Yann",
    "Olivier", "Pierre", "Sébastien", "Loïc", "Florian", "Romain", "Raphaël", "Gaël", "Bastien",
    "Arnaud", "Wei", "Arjun", "João", "Marco", "Karim", "Luca", "Noah", "Hugo", "Théo", "Samuel",
]
LAST_NAMES = [
    "Favre", "Rochat", "Chevalley", "Bovet", "Monnier", "Pittet", "Cuendet", "Jaquier", "Mercier",
    "Blanc", "Martin", "Bonvin", "Perrin", "Dubois", "Rey", "Morel", "Girard", "Bühler", "Müller",
    "Rossi", "Ferreira", "Silva", "Nguyen", "Chen", "Kumar", "Haddad", "Gonzalez", "Meylan",
    "Pasche", "Golay", "Cherpillod", "Vuilleumier", "Junod", "Bolomey", "Chappuis", "Ruchet",
    "Tschumi", "Bezençon", "Rapin", "Jaccard", "Leresche", "Aubert", "Graf", "Keller", "Moret",
    "Sauthier", "Amstutz", "Ducret", "Pilet", "Vez", "Wang", "Li", "Santos", "Costa", "Bianchi",
]
POSTCODES = [
    ("1008", 0.22), ("1004", 0.10), ("1005", 0.06), ("1006", 0.06), ("1007", 0.07), ("1018", 0.07),
    ("1020", 0.09), ("1023", 0.04), ("1024", 0.05), ("1030", 0.05), ("1032", 0.03), ("1052", 0.03),
    ("1110", 0.04), ("1012", 0.05), ("1010", 0.04),
]
COMPANIES = [
    ("Malley Tech SA", "Technology"), ("Léman Pharma SA", "Pharma"), ("Prilly Logistique Sàrl", "Logistics"),
    ("Banque Lémanique", "Finance"), ("Helvetia Data Services", "Technology"), ("Clinique de la Gare", "Health"),
    ("Studio Archi Malley", "Architecture"), ("Romandie Assur Conseil", "Insurance"), ("EPI Lab Renens", "Research"),
    ("Groupe Mediacité", "Media"), ("Vaud Energie Services", "Energy"), ("Atelier Numérique SA", "Technology"),
    ("Fiduciaire du Viaduc", "Finance"), ("Cuisine Centrale Prilly", "Food"), ("Transports Ouest Lausanne", "Transport"),
    ("Biotech Valley SA", "Pharma"), ("Lausanne Legal Partners", "Legal"), ("Ingénieurs Associés Malley", "Engineering"),
    ("Hôtel du Léman", "Hospitality"), ("Swiss Retail Ouest", "Retail"), ("Ecole Pro Malley", "Education"),
    ("Cloud Romand SA", "Technology"), ("Régie Immobilière Prilly", "Real estate"), ("Centre Médical Malley", "Health"),
    ("Agence Comm' Ouest", "Marketing"),
]
COACHES = ("Alexeï", "Pavel")

# Lausanne climate norms by month: (rain-day probability, mean temperature, daily sd).
CLIMATE = {
    1: (0.38, 1.5, 3.0), 2: (0.35, 2.5, 3.0), 3: (0.38, 6.5, 3.0), 4: (0.40, 10.0, 3.0),
    5: (0.45, 14.5, 3.0), 6: (0.45, 18.0, 3.0), 7: (0.38, 20.5, 3.0), 8: (0.40, 20.0, 3.0),
    9: (0.35, 16.0, 3.0), 10: (0.42, 11.5, 3.0), 11: (0.42, 6.0, 3.0), 12: (0.42, 2.5, 3.0),
}

HISTORY_DAYS = 168
FUTURE_DAYS = 14


def rng_for(seed: int, *stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, *stream])


@dataclass
class DayInfo:
    idx: int
    d: date
    day_type: str
    open_min: int
    n_cells: int
    is_school_holiday: bool
    school_holiday_name: str | None
    is_public_holiday: bool
    arena_event: bool
    is_future: bool

    @property
    def weekday(self) -> int:
        return self.d.weekday()

    def cell_time(self, cell: int) -> time:
        m = self.open_min + cell * pr.SLOT_MINUTES
        return time(m // 60, m % 60)

    def cell_dt(self, cell: int) -> datetime:
        return datetime.combine(self.d, time(0)) + timedelta(minutes=self.open_min + cell * pr.SLOT_MINUTES)


@dataclass
class World:
    seed: int
    as_of: date
    window_start: date
    window_end: date
    days: list[DayInfo]
    calendar: pd.DataFrame
    weather: pd.DataFrame
    players: pd.DataFrame
    companies: pd.DataFrame
    groups: dict[int, list[int]]
    blocks: pd.DataFrame
    initial_wallets: pd.DataFrame
    group_habits: pd.DataFrame = field(default_factory=pd.DataFrame)
    demo: dict = field(default_factory=dict)

    @property
    def now(self) -> datetime:
        return datetime.combine(self.as_of, time(0))


def build_days(as_of: date, seed: int) -> tuple[list[DayInfo], pd.DataFrame]:
    start = as_of - timedelta(days=HISTORY_DAYS)
    end = as_of + timedelta(days=FUTURE_DAYS - 1)
    rows = cal.build_calendar(start, end, seed)
    days = []
    for i, r in enumerate(rows):
        window = pr.rentable_window(r["day_type"])
        if window is None:
            open_min, n_cells = 0, 0
        else:
            open_min = window[0].hour * 60 + window[0].minute
            n_cells = len(pr.cell_starts(r["day_type"]))
        days.append(
            DayInfo(
                idx=i, d=r["date"], day_type=r["day_type"], open_min=open_min, n_cells=n_cells,
                is_school_holiday=r["is_school_holiday"], school_holiday_name=r["school_holiday_name"],
                is_public_holiday=r["is_public_holiday"], arena_event=r["arena_event"],
                is_future=r["date"] >= as_of,
            )
        )
    return days, pd.DataFrame(rows)


def build_weather(days: list[DayInfo], seed: int) -> pd.DataFrame:
    rng = rng_for(seed, 1)
    rows = []
    rainy_prev = False
    anomaly = 0.0
    for day in days:
        p_rain, t_mean, t_sd = CLIMATE[day.d.month]
        p = min(0.85, p_rain + (0.22 if rainy_prev else -0.12))
        rainy = rng.random() < p
        anomaly = 0.7 * anomaly + rng.normal(0, t_sd * 0.7)
        temp = t_mean + anomaly - (1.5 if rainy else 0.0)
        precip = float(np.round(rng.gamma(1.4, 5.0) + 1.0, 1)) if rainy else float(np.round(rng.random() * 0.8, 1))
        temp_max = temp + 4.0 + rng.normal(0, 1.0) - (1.5 if rainy else 0.0)
        if rainy and temp < 1.0:
            condition = "snow"
        elif rainy:
            condition = "rain"
        else:
            condition = "sunny" if rng.random() < 0.55 else "cloudy"
        rows.append(
            {
                "date": day.d, "temp_mean_c": round(float(temp), 1), "temp_max_c": round(float(temp_max), 1),
                "precip_mm": precip, "is_rainy": bool(rainy), "condition": condition, "is_forecast": day.is_future,
            }
        )
        rainy_prev = rainy
    return pd.DataFrame(rows)


def _player_names(rng: np.random.Generator, n: int) -> tuple[list[str], list[str]]:
    firsts, lasts = [], []
    for _ in range(n):
        pool = FIRST_NAMES_F if rng.random() < 0.47 else FIRST_NAMES_M
        firsts.append(pool[rng.integers(len(pool))])
        lasts.append(LAST_NAMES[rng.integers(len(LAST_NAMES))])
    return firsts, lasts


def _ascii(s: str) -> str:
    table = str.maketrans("éèêëàâäïîôöùûüçñ'ß", "eeeeaaaiioouuucn-s")
    return s.lower().translate(table).replace(" ", "")


def build_players(seed: int, n_players: int, window_start: date, as_of: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = rng_for(seed, 2)
    companies = pd.DataFrame(
        {
            "corporate_account_id": [f"C{i + 1:03d}" for i in range(len(COMPANIES))],
            "company_name": [c[0] for c in COMPANIES],
            "sector": [c[1] for c in COMPANIES],
            "employees_nearby": rng.integers(40, 900, len(COMPANIES)),
            "distance_km": np.round(rng.uniform(0.2, 3.5, len(COMPANIES)), 1),
        }
    )
    company_weights = companies["employees_nearby"].to_numpy() / companies["employees_nearby"].sum()

    seg_names = list(SEGMENT_SHARE)
    segments = rng.choice(seg_names, size=n_players, p=[SEGMENT_SHARE[s] for s in seg_names])
    firsts, lasts = _player_names(rng, n_players)
    domains = ["gmail.com", "bluewin.ch", "hotmail.com", "outlook.com", "sunrise.ch", "protonmail.ch", "epfl.ch", "unil.ch"]

    rows = []
    for i in range(n_players):
        seg = segments[i]
        pid = f"P{i + 1:05d}"
        if seg == "Student_OffPeak":
            birth_year = int(as_of.year - rng.integers(17, 27)) if rng.random() < 0.75 else int(as_of.year - rng.integers(55, 75))
        elif seg == "Family_Weekend":
            birth_year = int(as_of.year - rng.integers(32, 52))
        elif seg == "Corporate_Lunch":
            birth_year = int(as_of.year - rng.integers(25, 60))
        elif seg == "Club_Member":
            birth_year = int(as_of.year - rng.integers(14, 65))
        else:
            birth_year = int(as_of.year - rng.integers(22, 58))
        skill_p = {"Club_Member": [0.1, 0.4, 0.5], "Evening_Regular": [0.2, 0.5, 0.3]}.get(seg, [0.4, 0.45, 0.15])
        domain = "unil.ch" if seg == "Student_OffPeak" and rng.random() < 0.4 else domains[rng.integers(len(domains))]
        if seg == "Corporate_Lunch":
            company = companies.iloc[rng.choice(len(companies), p=company_weights)]
            corp_id = company["corporate_account_id"]
            if rng.random() < 0.5:
                domain = _ascii(company["company_name"].split()[0]) + ".ch"
        else:
            corp_id = None
        # 20% of players register during the window; the rest are established.
        if rng.random() < 0.20:
            joined = window_start + timedelta(days=int(rng.integers(0, (as_of - window_start).days)))
        else:
            joined = window_start - timedelta(days=int(rng.integers(30, 2200)))
        rows.append(
            {
                "player_id": pid,
                "first_name": firsts[i],
                "last_name": lasts[i],
                "email": f"{_ascii(firsts[i])}.{_ascii(lasts[i])}{rng.integers(1, 99)}@{domain}",
                "phone": f"+41 7{rng.choice([6, 7, 8, 9])} {rng.integers(100, 999)} {rng.integers(10, 99)} {rng.integers(10, 99)}",
                "birth_year": birth_year,
                "home_postcode": POSTCODES[rng.choice(len(POSTCODES), p=_norm([p for _, p in POSTCODES]))][0],
                "skill_tier": SKILLS[rng.choice(3, p=skill_p)],
                "segment": seg,
                "preferred_lang": "EN" if rng.random() < (0.35 if domain in ("epfl.ch", "unil.ch") else 0.18) else "FR",
                "corporate_account_id": corp_id,
                "joined_at": datetime.combine(joined, time(int(rng.integers(8, 22)), int(rng.integers(0, 60)))),
                "marketing_opt_in": bool(rng.random() < 0.62),
                "is_flexible": bool(rng.random() < FLEXIBLE_SHARE[seg]),
                "activity": float(rng.lognormal(0.0, 1.15)),
                "join_date": joined,
            }
        )
    players = pd.DataFrame(rows)
    _assign_habits(players, rng)
    _assign_lifecycle(players, rng, window_start, as_of)
    _assign_wtp(players, rng)
    return players, companies


def _norm(p):
    a = np.asarray(p, dtype=float)
    return a / a.sum()


# Habit windows by segment: (weekday choices, start-hour choices).
HABITS = {
    "Corporate_Lunch": ([0, 1, 2, 3, 4], [11.5, 12.0, 12.0, 12.5, 12.0]),
    "Evening_Regular": ([0, 1, 2, 3], [17.5, 18.0, 18.5, 19.0, 19.0, 19.5, 20.0, 20.0, 20.5, 21.0]),
    "Family_Weekend": ([5, 6], [9.5, 10.0, 10.5, 11.0, 13.0, 14.0, 15.0, 16.0]),
    "Student_OffPeak": ([0, 1, 2, 3, 4], [8.5, 9.0, 10.0, 10.5, 14.0, 14.5, 15.0, 16.0]),
    "Club_Member": ([0, 1, 2, 3, 5, 6], [18.0, 19.0, 20.0, 10.0, 11.0, 17.5]),
}
HABIT_SHARE = {"Corporate_Lunch": 0.70, "Evening_Regular": 0.85, "Family_Weekend": 0.60, "Student_OffPeak": 0.30, "Club_Member": 0.60}


def _assign_habits(players: pd.DataFrame, rng: np.random.Generator) -> None:
    hd, hh, hd2 = [], [], []
    for seg in players["segment"]:
        days, hours = HABITS[seg]
        if rng.random() < HABIT_SHARE[seg]:
            d1 = int(rng.choice(days))
            h = float(rng.choice(hours))
            second = int(rng.choice(days)) if rng.random() < 0.35 else -1
            if second == d1:
                second = -1
            hd.append(d1)
            hh.append(h)
            hd2.append(second)
        else:
            hd.append(-1)
            hh.append(-1.0)
            hd2.append(-1)
    players["habit_day"] = hd
    players["habit_day2"] = hd2
    players["habit_hour"] = hh


def _assign_lifecycle(players: pd.DataFrame, rng: np.random.Generator, window_start: date, as_of: date) -> None:
    """Drifters lose interest gradually; newcomers ramp up over their first weeks."""
    n = len(players)
    span = (as_of - window_start).days
    drift_start = np.full(n, -1, dtype=int)
    drift_len = np.zeros(n, dtype=int)
    drift_floor = np.ones(n)
    established = players["join_date"].to_numpy() < np.datetime64(window_start - timedelta(days=30))
    for i in range(n):
        if established[i] and rng.random() < 0.16:
            drift_start[i] = int(rng.integers(20, span + 30))
            drift_len[i] = int(rng.integers(21, 49))
            drift_floor[i] = float(rng.choice([0.0, 0.1, 0.25, 0.4], p=[0.35, 0.25, 0.25, 0.15]))
    players["drift_start_day"] = drift_start
    players["drift_len"] = drift_len
    players["drift_floor"] = drift_floor


def _assign_wtp(players: pd.DataFrame, rng: np.random.Generator) -> None:
    players["wtp_z"] = rng.normal(0, 1, len(players))
    # ~15% habitual cancellers (~3x the base rate), the rest ~0.65x; the mean multiplier stays ~1.
    z = rng.normal(0, 1, len(players))
    flaky = z > FLAKY_Z
    players["cancel_propensity"] = np.where(flaky, 3.0 * np.exp(0.25 * (z - 1.5)), 0.65 * np.exp(0.25 * z))


def lifecycle_multiplier(players: pd.DataFrame, day_offset: int, day: date) -> np.ndarray:
    """Activity multiplier per player for the day at offset day_offset from the window start."""
    join = players["join_date"].to_numpy()
    days_since_join = (np.datetime64(day) - join.astype("datetime64[D]")).astype(int)
    mult = np.where(days_since_join < 0, 0.0, np.clip(0.35 + days_since_join / 42.0, 0.35, 1.0))
    ds = players["drift_start_day"].to_numpy()
    dl = np.maximum(players["drift_len"].to_numpy(), 1)
    floor = players["drift_floor"].to_numpy()
    progress = np.clip((day_offset - ds) / dl, 0.0, 1.0)
    drift = np.where(ds >= 0, 1.0 - progress * (1.0 - floor), 1.0)
    return mult * drift


GROUP_HABIT_SHARE = {"Corporate_Lunch": 0.45, "Evening_Regular": 0.55, "Family_Weekend": 0.35,
                     "Student_OffPeak": 0.15, "Club_Member": 0.40}


def build_groups(players: pd.DataFrame, seed: int) -> tuple[dict[int, list[int]], pd.DataFrame]:
    """Stable doubles groups of 3-5 players with the same segment and similar skill.

    Some groups have a weekly fixed slot (e.g. Tuesday 19:00) that they keep most weeks.
    """
    rng = rng_for(seed, 3)
    groups: dict[int, list[int]] = {}
    habit_rows = []
    gid = 0
    players["group_id"] = -1
    for seg in SEGMENTS:
        idx = players.index[players["segment"] == seg].to_numpy()
        skill_rank = players.loc[idx, "skill_tier"].map({"Beginner": 0, "Intermediate": 1, "Advanced": 2}).to_numpy()
        order = idx[np.lexsort((rng.random(len(idx)), skill_rank))]
        i = 0
        while i < len(order):
            size = int(rng.choice([3, 4, 4, 5]))
            members = order[i:i + size].tolist()
            groups[gid] = members
            players.loc[members, "group_id"] = gid
            if rng.random() < GROUP_HABIT_SHARE[seg]:
                days, hours = HABITS[seg]
                habit_rows.append({"group_id": gid, "segment": seg, "weekday": int(rng.choice(days)),
                                   "hour": float(rng.choice(hours)), "attend_p": float(rng.uniform(0.65, 0.92))})
            gid += 1
            i += size
    return groups, pd.DataFrame(habit_rows)


def build_blocks(days: list[DayInfo], seed: int) -> pd.DataFrame:
    """Bad'matin, private lessons, Club BLA and school sessions (never re-priced)."""
    rng = rng_for(seed, 4)
    rows = []

    def add(day: DayInfo, court: str, start: time, end: time, btype: str, coach, participants, revenue):
        rows.append(
            {
                "court_id": court,
                "block_type": btype,
                "start_ts": datetime.combine(day.d, start),
                "end_ts": datetime.combine(day.d, end),
                "coach": coach,
                "participants": participants,
                "revenue_chf": round(float(revenue), 2),
                "day_idx": day.idx,
            }
        )

    lessons = {
        0: [(time(15, 30), time(16, 30), "Pavel")],
        1: [(time(11, 0), time(12, 0), None), (time(12, 0), time(13, 0), None), (time(20, 30), time(21, 30), None)],
        2: [(time(13, 0), time(14, 0), None), (time(18, 0), time(19, 0), None)],
        4: [(time(15, 0), time(16, 0), None)],
    }
    for day in days:
        if day.day_type != "weekday":
            continue
        wd = day.weekday
        term = not day.is_school_holiday
        if wd == 1 and term:
            n = int(rng.integers(9, 17))
            for c in pr.SHOW_COURTS:
                add(day, c, time(9, 30), time(11, 0), "badmatin", COACHES[day.idx % 2], n if c == "Court_01" else 0,
                    n * pr.BADMATIN_PRICE_PER_PERSON_CHF * 0.93 if c == "Court_01" else 0.0)
        for start, end, coach in lessons.get(wd, []):
            if rng.random() < (0.65 if term else 0.4):
                tier = pr.tier_for(start, wd, day.day_type)
                add(day, "Court_16", start, end, "private_lesson", coach or COACHES[int(rng.integers(2))], 1,
                    pr.PRIVATE_LESSON_PRICE_CHF + pr.TIER_PRICE_CHF[tier])
        if wd == 0 and term:
            for c in pr.COURT_IDS[12:16]:
                add(day, c, time(18, 0), time(20, 0), "club_bla", None, 18 if c == "Court_13" else 0,
                    2 * pr.ASSOCIATION_PRICE_CHF)
        if wd == 3:
            for c in pr.COURT_IDS[12:16]:
                add(day, c, time(20, 0), time(22, 0), "club_bla", None, 14 if c == "Court_13" else 0,
                    2 * pr.ASSOCIATION_PRICE_CHF)
        if wd in (1, 3) and term:
            for c in ("Court_09", "Court_10"):
                add(day, c, time(13, 30), time(15, 30), "school", None, 24 if c == "Court_09" else 0,
                    2 * pr.SCHOOL_PRICE_CHF)
    blocks = pd.DataFrame(rows)
    # Private lessons on Court_16 can collide with club blocks; the club keeps the court.
    blocks = blocks.sort_values(["block_type"], key=lambda s: s.map({"club_bla": 0}).fillna(1)).reset_index(drop=True)
    keep = []
    taken: list[tuple[str, datetime, datetime]] = []
    for r in blocks.itertuples():
        if any(r.court_id == c and r.start_ts < e and s < r.end_ts for c, s, e in taken):
            keep.append(False)
            continue
        taken.append((r.court_id, r.start_ts, r.end_ts))
        keep.append(True)
    blocks = blocks[keep].sort_values("start_ts").reset_index(drop=True)
    blocks.insert(0, "block_id", [f"BL{i + 1:05d}" for i in range(len(blocks))])
    return blocks


def build_initial_wallets(players: pd.DataFrame, seed: int, window_start: date) -> pd.DataFrame:
    """Pass products held at the start of the window (product choice drives later renewals)."""
    rng = rng_for(seed, 5)
    rows = []
    products = []
    for p in players.itertuples():
        u = rng.random()
        heavy = p.activity > 1.2
        if u < (0.14 if heavy else 0.07):
            product = pr.CARTE_PREPAYEE
        elif u < (0.24 if heavy else 0.11):
            product = pr.PASS_LIBERTE
        elif u < (0.34 if heavy else 0.16):
            product = pr.PASS_CHRONO
        else:
            product = None
        purchased_at = datetime.combine(window_start, time(10)) - timedelta(days=int(rng.integers(1, 75)))
        used_fraction = float(rng.uniform(0.0, 0.7))
        if product is None and rng.random() < 0.07:
            # First purchase during the window.
            product = [pr.CARTE_PREPAYEE, pr.PASS_LIBERTE, pr.PASS_CHRONO][int(rng.choice(3, p=[0.4, 0.25, 0.35]))]
            purchased_at = datetime.combine(window_start, time(int(rng.integers(9, 21)))) + timedelta(
                days=int(rng.integers(0, HISTORY_DAYS - 7)))
            used_fraction = 0.0
        products.append(product)
        if product is None:
            continue
        variant = None
        if product != pr.CARTE_PREPAYEE:
            if p.segment == "Corporate_Lunch":
                variant = pr.MIDI
            elif p.segment == "Evening_Regular":
                variant = pr.SOIR
            else:
                variant = pr.MIDI if rng.random() < 0.5 else pr.SOIR
        rows.append(
            {
                "player_idx": p.Index,
                "product": product,
                "variant": variant,
                "purchased_at": purchased_at,
                "used_fraction": used_fraction,
                "size_u": float(rng.random()),
            }
        )
    players["pass_product"] = products
    return pd.DataFrame(rows)


def build_world(seed: int, as_of: date, n_players: int = 3000) -> World:
    days, cal_df = build_days(as_of, seed)
    window_start, window_end = days[0].d, days[-1].d
    weather = build_weather(days, seed)
    players, companies = build_players(seed, n_players, window_start, as_of)
    groups, group_habits = build_groups(players, seed)
    blocks = build_blocks(days, seed)
    wallets = build_initial_wallets(players, seed, window_start)
    world = World(seed, as_of, window_start, window_end, days, cal_df, weather, players, companies, groups, blocks,
                  wallets, group_habits=group_habits)
    setup_demo_moments(world)
    return world


def _first_future(world: World, weekday: int, min_offset: int) -> DayInfo:
    for day in world.days:
        if day.is_future and (day.d - world.as_of).days >= min_offset and day.weekday == weekday \
                and day.n_cells > 0 and not day.is_public_holiday:
            return day
    raise ValueError(f"No future day with weekday {weekday}")


def setup_demo_moments(world: World) -> None:
    """Seed the five scripted moments from SPEC.md section 4 into the next 14 days."""
    players = world.players
    demo: dict = {"demand_multiplier": {}, "forced_intents": []}

    saturday = _first_future(world, 5, 2)
    w = world.weather
    mask = w["date"] == saturday.d
    w.loc[mask, ["is_rainy", "precip_mm", "condition"]] = [True, 14.2, "rain"]
    w.loc[mask, "temp_mean_c"] = w.loc[mask, "temp_mean_c"] - 2.0
    w.loc[mask, "temp_max_c"] = w.loc[mask, "temp_max_c"] - 2.5
    demo["rainy_saturday_idx"] = saturday.idx

    tuesday = _first_future(world, 1, 1)
    demo["demand_multiplier"][(tuesday.idx, "lunch")] = 0.35
    demo["quiet_tuesday_idx"] = tuesday.idx

    established = players["join_date"] < world.window_start - timedelta(days=120)
    no_pass = players["pass_product"].isna()
    evening = (players["segment"] == "Evening_Regular") & established & (players["drift_start_day"] < 0)
    candidates = players[evening & players["habit_day"].between(0, 3) & no_pass].sort_values("activity", ascending=False)
    drifter = int(candidates.index[2])
    players.loc[drifter, ["activity", "drift_start_day", "drift_len", "drift_floor"]] = [
        max(float(players.at[drifter, "activity"]), 5.0), HISTORY_DAYS - 40, 14, 0.12]
    players.loc[drifter, "habit_day2"] = (int(players.at[drifter, "habit_day"]) + 2) % 4
    demo["drifter_idx"] = drifter

    thursday = _first_future(world, 3, 2)
    canceller = int(candidates.index[7])
    players.loc[canceller, ["habit_day", "habit_hour", "activity", "cancel_propensity", "is_flexible"]] = [
        3, 19.0, 4.0, 7.0, False]
    demo["forced_intents"].append({
        "day_idx": thursday.idx, "start": time(19, 0), "player_idx": canceller,
        "overrides": {"booked_at": datetime.combine(world.as_of, time(0)) - timedelta(days=5, hours=3),
                      "force_book": True, "cancel_mult": 0.0, "dur_cells": 2, "party_size": 4,
                      "family_rate": False, "flexible": False, "u_court": 0.3},
    })
    demo["canceller_idx"] = canceller
    demo["late_cancel_day_idx"] = thursday.idx

    chrono_pool = players[(players["segment"] == "Evening_Regular") & no_pass & established
                          & (players["drift_start_day"] < 0) & ~players.index.isin([drifter, canceller])]
    chrono_player = int(chrono_pool.sort_values("activity").index[len(chrono_pool) // 2])
    players.loc[chrono_player, ["activity", "pass_product"]] = [1.2, pr.PASS_CHRONO]
    chrono_purchase = datetime.combine(world.as_of - timedelta(days=84), time(18, 40))
    demo["chrono_purchased_at"] = chrono_purchase
    world.initial_wallets = pd.concat([world.initial_wallets, pd.DataFrame([{
        "player_idx": chrono_player, "product": pr.PASS_CHRONO, "variant": pr.SOIR,
        "purchased_at": chrono_purchase,
        "used_fraction": 0.0, "size_u": 0.1,
    }])], ignore_index=True)
    demo["chrono_idx"] = chrono_player
    world.demo = demo
