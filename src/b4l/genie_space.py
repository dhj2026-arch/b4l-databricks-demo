"""Genie space definition for the B4L gold tables (SPEC.md section 5.D).

Spark-free so it can be unit tested. `build_serialized_space` returns the version-2 `serialized_space`
payload accepted by POST /api/2.0/genie/spaces and PATCH /api/2.0/genie/spaces/{id}:
  - config.sample_questions[]            {id, question: [text]}
  - data_sources.tables[]                {identifier, column_configs: [{column_name, synonyms}]}
  - instructions.text_instructions[]     at most one, {id, content: [lines]}
  - instructions.example_question_sqls[] {id, question: [text], sql: [lines]}
IDs are 32-char lowercase hex and every collection is pre-sorted, as the API requires.
"""

from __future__ import annotations

import hashlib
import json

from b4l import pricing_rules as pr
from b4l import schemas

SPACE_TITLE = "B4L Smart Badminton Center"
STAFF_APP_NAME = "b4l-staff-ops"

GENIE_TABLES = (
    "court_utilization_hourly",
    "daily_kpis",
    "player_features",
    "slot_features",
    "sponsor_metrics",
)

SPACE_DESCRIPTION = (
    "Ask questions in French or English about the 16 badminton courts of B4L Fondation (Prilly, Lausanne): "
    "occupancy (taux d'occupation), revenue in CHF (chiffre d'affaires), tariffs, cancellations, waitlist, "
    "player retention, passes and anonymised sponsor footfall. Synthetic demo data: 24 weeks of history "
    "plus bookings made so far for the next 14 days."
)

# French/English business glossary: term -> meaning and how to express it in SQL on the gold tables.
GLOSSARY: dict[str, str] = {
    "Tarif journée": (
        f"Day tier, tier = '{pr.DAY}', CHF {pr.TIER_PRICE_CHF[pr.DAY]:.0f} per court per hour "
        "(Mon-Fri 08:00-11:29 and 13:30-16:59 starts)"
    ),
    "Tarif midi": (
        f"Lunch tier, tier = '{pr.LUNCH}', CHF {pr.TIER_PRICE_CHF[pr.LUNCH]:.0f} (Mon-Fri 11:30-13:29 starts); "
        "'créneaux de midi' = lunch slots"
    ),
    "Tarif soir": (
        f"Evening tier, tier = '{pr.EVENING}', CHF {pr.TIER_PRICE_CHF[pr.EVENING]:.0f} (Mon-Thu 17:00-21:00 starts)"
    ),
    "Tarif week-end": (
        f"Weekend tier, tier = '{pr.WEEKEND}', CHF {pr.TIER_PRICE_CHF[pr.WEEKEND]:.0f} "
        "(Friday from 13:30, Saturday, Sunday and public holidays)"
    ),
    "Famille": (
        f"Family rate CHF {pr.FAMILY_PRICE_CHF:.0f}, day and weekend times only, at least one adult and one "
        "child; a booking rate category, not a tier"
    ),
    "Bad'matin": (
        f"Coached drop-in session on Tuesday 09:30-11:00, CHF {pr.BADMATIN_PRICE_PER_PERSON_CHF:.0f} per person, "
        "not held in Vaud school holidays; it blocks courts (blocked_minutes) and its revenue is in "
        "daily_kpis.block_revenue_chf"
    ),
    "Taux d'occupation": (
        "Occupancy rate = (booked_minutes + blocked_minutes) / available_minutes. Aggregate as a ratio of sums, "
        "never as an average of occupancy_rate"
    ),
    "Carte prépayée": (
        f"Prepaid card, pass_product = '{pr.CARTE_PREPAYEE}': CHF 200/300/500 loaded with a 15/20/30% bonus; "
        "remaining credit in pass_balance_chf"
    ),
    "Pass Liberté": (
        f"Hour pack without expiry, pass_product = '{pr.PASS_LIBERTE}' (25/50/100 h, midi or soir variant); "
        "remaining hours in pass_hours_remaining"
    ),
    "Pass Chrono": (
        f"Hour pack with an expiry date, pass_product = '{pr.PASS_CHRONO}' (25 h valid 3 months, 50 h valid "
        "6 months); expiry in pass_expires_at and days_to_pass_expiry"
    ),
    "Liste d'attente": (
        "Waitlist: daily_kpis.waitlist_requests and waitlist_resolved; slot_features.waitlist_requests"
    ),
    "16 terrains / courts": (
        f"{pr.N_COURTS} courts Court_01..Court_16; Main Hall = Court_01-08, West Hall = Court_09-16; "
        "show courts (courts d'honneur) Court_01-04 (is_show_court)"
    ),
    "Chiffre d'affaires": "Revenue in CHF: revenue_chf (court bookings); block_revenue_chf for courses and club",
    "Fréquentation": (
        "Footfall: check_in_persons (persons checked in at reception); unique_players for distinct players"
    ),
    "Annulation tardive": (
        f"Late cancellation, less than {pr.LATE_CANCEL_HOURS} h before start, billed in full: "
        "daily_kpis.cancellations_late; no-shows are daily_kpis.no_shows"
    ),
}

# Column synonyms pushed into data_sources.tables[].column_configs.
COLUMN_SYNONYMS: dict[str, dict[str, list[str]]] = {
    "court_utilization_hourly": {
        "occupancy_rate": ["taux d'occupation", "occupancy", "utilisation", "remplissage"],
        "revenue_chf": ["chiffre d'affaires", "revenue", "CA", "recettes"],
        "court_id": ["terrain", "court", "court number"],
        "hall_zone": ["salle", "hall", "zone"],
        "tier": ["tarif", "tariff", "price tier", "créneau tarifaire"],
        "check_in_persons": ["fréquentation", "footfall", "entrées"],
        "blocked_minutes": ["cours", "Bad'matin", "club", "blocages"],
        "is_rainy": ["pluie", "rain", "rainy day"],
        "is_school_holiday": ["vacances scolaires", "school holidays"],
        "is_public_holiday": ["jour férié", "public holiday"],
        "is_show_court": ["court d'honneur", "show court", "terrain central"],
    },
    "daily_kpis": {
        "occupancy_rate": ["taux d'occupation", "occupancy"],
        "revenue_chf": ["chiffre d'affaires", "revenue", "CA"],
        "cancellations_late": ["annulations tardives", "late cancellations"],
        "cancellations_early": ["annulations", "free cancellations"],
        "no_shows": ["absences", "no-show"],
        "waitlist_requests": ["liste d'attente", "waitlist"],
        "check_in_persons": ["fréquentation", "footfall", "entrées"],
        "bookings": ["réservations", "bookings"],
        "is_rainy": ["pluie", "rain"],
    },
    "player_features": {
        "player_uuid": ["joueur", "player", "pseudonymised id"],
        "pass_product": ["abonnement", "pass", "carte prépayée", "Pass Liberté", "Pass Chrono"],
        "days_to_pass_expiry": ["expiration", "jours avant expiration", "days to expiry"],
        "churn_risk_rule": ["risque de départ", "churn risk", "at risk"],
        "plays_last_30d": ["parties 30 jours", "visits last 30 days"],
        "segment": ["segment", "profil"],
        "skill_tier": ["niveau", "level"],
    },
    "slot_features": {
        "slot_start_ts": ["créneau", "slot", "start time"],
        "tier": ["tarif", "tariff"],
        "base_price_chf": ["prix", "tarif CHF", "base price"],
        "booked_courts_now": ["terrains réservés", "courts booked so far"],
        "courts_available": ["terrains disponibles", "available courts"],
        "hist_fill_rate_4w": ["taux de remplissage historique", "historical fill rate"],
        "waitlist_requests": ["liste d'attente", "waitlist"],
    },
    "sponsor_metrics": {
        "check_in_persons": ["fréquentation", "footfall", "visiteurs"],
        "unique_players": ["joueurs uniques", "distinct players"],
        "metric_group": ["dimension", "groupe"],
        "suppressed": ["masqué", "k-anonymity suppressed"],
    },
}

SAMPLE_QUESTIONS: list[str] = [
    "Quel est le taux d'occupation des créneaux de midi la semaine dernière ?",
    "Which hours on weekdays have the lowest occupancy in October?",
    "How much revenue did rainy Saturdays bring vs dry Saturdays?",
    "Combien d'annulations tardives ce mois-ci ?",
    "Which players have a Pass Chrono expiring in the next 14 days?",
    "Quel est le chiffre d'affaires par tarif (journée, midi, soir, week-end) et par mois ?",
    "Which upcoming slots in the next 14 days are booked well below their 4-week fill rate?",
    "Combien de joueurs réguliers sont à risque de départ, par segment ?",
    "What was the monthly footfall (fréquentation) by player segment for sponsors?",
    "Compare Main Hall and West Hall occupancy during Vaud school holidays vs term time.",
]

_AS_OF = "(SELECT MAX(as_of_date) FROM {g}.player_features)"

# (question, SQL template). {g} is replaced with the fully qualified gold schema.
EXAMPLE_SQLS: list[tuple[str, str]] = [
    (
        SAMPLE_QUESTIONS[0],
        f"""SELECT ROUND(100 * SUM(booked_minutes + blocked_minutes) / SUM(available_minutes), 1)
  AS taux_occupation_pct
FROM {{g}}.court_utilization_hourly
WHERE tier = 'Lunch'
  AND date >= date_sub(CAST(date_trunc('WEEK', {_AS_OF}) AS DATE), 7)
  AND date < CAST(date_trunc('WEEK', {_AS_OF}) AS DATE)""",
    ),
    (
        SAMPLE_QUESTIONS[1],
        f"""SELECT hour,
  ROUND(100 * SUM(booked_minutes + blocked_minutes) / SUM(available_minutes), 1) AS occupancy_pct
FROM {{g}}.court_utilization_hourly
WHERE day_type = 'weekday' AND month(date) = 10 AND date < {_AS_OF}
GROUP BY hour
ORDER BY occupancy_pct ASC""",
    ),
    (
        SAMPLE_QUESTIONS[2],
        f"""SELECT CASE WHEN is_rainy THEN 'rainy' ELSE 'dry' END AS weather,
  COUNT(*) AS saturdays,
  ROUND(SUM(revenue_chf), 0) AS total_revenue_chf,
  ROUND(AVG(revenue_chf), 0) AS avg_revenue_chf_per_saturday
FROM {{g}}.daily_kpis
WHERE day_name = 'Saturday' AND date < {_AS_OF}
GROUP BY 1""",
    ),
    (
        SAMPLE_QUESTIONS[3],
        f"""SELECT SUM(cancellations_late) AS annulations_tardives, SUM(no_shows) AS no_shows
FROM {{g}}.daily_kpis
WHERE date >= trunc({_AS_OF}, 'MM') AND date < {_AS_OF}""",
    ),
    (
        SAMPLE_QUESTIONS[4],
        """SELECT player_uuid, segment, pass_hours_remaining, pass_expires_at, days_to_pass_expiry
FROM {g}.player_features
WHERE pass_product = 'Pass_Chrono' AND days_to_pass_expiry BETWEEN 0 AND 14
ORDER BY days_to_pass_expiry, pass_hours_remaining DESC""",
    ),
    (
        SAMPLE_QUESTIONS[5],
        f"""SELECT date_trunc('MONTH', date) AS month, tier, ROUND(SUM(revenue_chf), 0) AS revenue_chf
FROM {{g}}.court_utilization_hourly
WHERE date < {_AS_OF}
GROUP BY 1, 2
ORDER BY 1, 2""",
    ),
    (
        SAMPLE_QUESTIONS[8],
        """SELECT month, metric_key AS segment, check_in_persons, unique_players
FROM {g}.sponsor_metrics
WHERE metric_group = 'segment' AND NOT suppressed
ORDER BY month, segment""",
    ),
]


def gold_schema(catalog: str, schema_prefix: str) -> str:
    return f"{catalog}.{schema_prefix}_gold"


def table_identifiers(catalog: str, schema_prefix: str) -> list[str]:
    g = gold_schema(catalog, schema_prefix)
    return sorted(f"{g}.{t}" for t in GENIE_TABLES)


def stable_id(kind: str, key: str) -> str:
    """Deterministic 32-char lowercase hex id, so re-runs produce the same payload."""
    return hashlib.md5(f"b4l-genie/{kind}/{key}".encode()).hexdigest()


def instructions_text(catalog: str, schema_prefix: str) -> str:
    g = gold_schema(catalog, schema_prefix)
    glossary = "\n".join(f"- {term}: {meaning}." for term, meaning in GLOSSARY.items())
    return f"""You answer questions from the management and reception team of B4L Fondation, a 16-court \
badminton centre in Prilly (Lausanne, Switzerland). Users write in French or English; answer in the language \
of the question. All data lives in {g}.

Business rules:
- Currency is Swiss francs (CHF). Prices are per court per hour for 2 or 4 players. Format amounts as CHF.
- Tariff tiers (column tier) are set by the booking start time: Day = Tarif journée CHF 25, Lunch = Tarif midi \
CHF 32, Evening = Tarif soir CHF 35, Weekend = Tarif week-end CHF 25. Family rate CHF 18.
- Rentable hours: Monday-Friday 08:00-22:00 (hour 8-21), Saturday, Sunday and Vaud public holidays 09:30-19:00 \
(hour 9-18; the 09:00 hour has only 30 available minutes). day_type = 'weekday' or 'weekend_holiday'. Closed on \
24, 25 and 31 December and 1 January.
- Occupancy (taux d'occupation) = (booked_minutes + blocked_minutes) / available_minutes. When aggregating \
several hours, courts or days, always compute SUM(booked_minutes + blocked_minutes) / SUM(available_minutes); \
never average occupancy_rate. Express it as a percentage.
- Court blocks (Bad'matin, private lessons, club BLA, school sessions) appear as blocked_minutes; they are never re-priced.
- Halls: Main Hall = Court_01-Court_08, West Hall = Court_09-Court_16. Show courts Court_01-Court_04.

Time:
- The snapshot date ('today') is MAX(as_of_date) in player_features (it equals run_info.as_of_date of the \
generator). Use it instead of current_date() for 'today', 'this month', 'last week' and similar.
- Dates before the snapshot date are history (24 weeks). The next 14 days only contain the bookings made so far: \
they are not final, so exclude dates >= the snapshot date from historical occupancy and revenue unless the user \
asks about upcoming days. In slot_features, is_future = true marks those upcoming slots and booked_courts_now is \
the current booking count.
- Weeks start on Monday. 'Last week' means the previous full Monday-Sunday week.
- Vaud school holidays (vacances scolaires) are flagged by is_school_holiday; Vaud public holidays (jours fériés) \
by is_public_holiday and use weekend hours.

Players and privacy:
- player_uuid is a pseudonymised id. Never try to reveal names, emails or phone numbers.
- A 'regular' (régulier) has avg_monthly_plays_baseline >= 2 (average plays per 30 days over days 31-120 before \
the snapshot). churn_risk_rule = true means a regular whose plays_last_30d fell to 35% of that baseline or less; \
it is always false for non-regulars.
- sponsor_metrics is k-anonymised: any cell with fewer than {schemas.K_ANONYMITY_MIN} unique players is \
suppressed (suppressed = true, metrics null). Filter NOT suppressed and never try to derive suppressed values.

Glossary (French / English):
{glossary}
"""


def _column_configs(table: str) -> list[dict]:
    known = set(schemas.column_names(schemas.GOLD[table]))
    configs = [
        {"column_name": col, "synonyms": syns}
        for col, syns in COLUMN_SYNONYMS.get(table, {}).items()
        if col in known
    ]
    return sorted(configs, key=lambda c: c["column_name"])


def _lines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines or [text]


def build_serialized_space(
    catalog: str,
    schema_prefix: str,
    *,
    include_column_configs: bool = True,
    include_example_sqls: bool = True,
    include_instructions: bool = True,
) -> dict:
    """Version-2 serialized space. Flags allow a degraded payload if the API rejects a section."""
    g = gold_schema(catalog, schema_prefix)
    tables = []
    for ident in table_identifiers(catalog, schema_prefix):
        entry: dict = {"identifier": ident}
        if include_column_configs:
            configs = _column_configs(ident.rsplit(".", 1)[1])
            if configs:
                entry["column_configs"] = configs
        tables.append(entry)

    sample_questions = sorted(
        ({"id": stable_id("sample_question", q), "question": [q]} for q in SAMPLE_QUESTIONS),
        key=lambda x: x["id"],
    )
    space: dict = {
        "version": 2,
        "config": {"sample_questions": sample_questions},
        "data_sources": {"tables": tables},
    }

    instructions: dict = {}
    if include_instructions:
        instructions["text_instructions"] = [
            {"id": stable_id("text_instruction", "main"), "content": _lines(instructions_text(catalog, schema_prefix))}
        ]
    if include_example_sqls:
        instructions["example_question_sqls"] = sorted(
            (
                {"id": stable_id("example_sql", q), "question": [q], "sql": _lines(sql.format(g=g))}
                for q, sql in EXAMPLE_SQLS
            ),
            key=lambda x: x["id"],
        )
    if instructions:
        space["instructions"] = instructions
    return space


def serialize(space: dict) -> str:
    return json.dumps(space, ensure_ascii=False)


def payload_variants(catalog: str, schema_prefix: str) -> list[tuple[str, str]]:
    """Payloads from richest to minimal; the notebook tries them in order until the API accepts one."""
    return [
        ("full", serialize(build_serialized_space(catalog, schema_prefix))),
        (
            "no_column_configs",
            serialize(build_serialized_space(catalog, schema_prefix, include_column_configs=False)),
        ),
        (
            "instructions_only",
            serialize(
                build_serialized_space(
                    catalog, schema_prefix, include_column_configs=False, include_example_sqls=False
                )
            ),
        ),
        (
            "tables_and_questions",
            serialize(
                build_serialized_space(
                    catalog,
                    schema_prefix,
                    include_column_configs=False,
                    include_example_sqls=False,
                    include_instructions=False,
                )
            ),
        ),
    ]


def grant_statements(catalog: str, schema_prefix: str, principal: str) -> list[str]:
    """Least-privilege UC grants for an app service principal (application id) on the gold schema."""
    g = f"`{catalog}`.`{schema_prefix}_gold`"
    p = f"`{principal}`"
    return [
        f"GRANT USE CATALOG ON CATALOG `{catalog}` TO {p}",
        f"GRANT USE SCHEMA ON SCHEMA {g} TO {p}",
        f"GRANT SELECT ON SCHEMA {g} TO {p}",
    ]


def space_url(host: str, space_id: str) -> str:
    return f"{host.rstrip('/')}/genie/rooms/{space_id}"
