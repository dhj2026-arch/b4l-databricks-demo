"""B4L Lakeflow Spark Declarative Pipeline: bronze -> silver (clean, pseudonymised) -> gold (SPEC.md 5.A).

Every dataset is a materialized view over batch reads because the generator overwrites bronze on each run.
Silver lands in the pipeline default schema ({prefix}_silver); gold uses fully-qualified names in {prefix}_gold.
Players are pseudonymised with HMAC-SHA256 keyed by the secret-scope key (b4l.hmac_key).
"""

import pandas as pd
from pyspark import pipelines as dp
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from b4l import pipeline_helpers as ph
from b4l import pricing_rules as pr
from b4l import schemas

CATALOG = spark.conf.get("b4l.catalog")
PREFIX = spark.conf.get("b4l.schema_prefix")
HMAC_KEY = spark.conf.get("b4l.hmac_key")
BRONZE_SCHEMA = f"{CATALOG}.{PREFIX}_bronze"
GOLD_SCHEMA = f"{CATALOG}.{PREFIX}_gold"

GENIE_TERMS = (
    "16 terrains / courts; Tarif journée (Day), Tarif midi (Lunch), Tarif soir (Evening), "
    "Tarif week-end (Weekend); taux d'occupation = occupancy rate; chiffre d'affaires = revenue"
)


def bronze(table: str) -> DataFrame:
    return spark.read.table(f"{BRONZE_SCHEMA}.{table}")


def gold_name(table: str) -> str:
    return f"{GOLD_SCHEMA}.{table}"


def gold_view(table: str, comment: str):
    return dp.materialized_view(
        name=gold_name(table), comment=comment, schema=schemas.ddl(schemas.GOLD[table])
    )


def conform(df: DataFrame, table: str) -> DataFrame:
    """Select gold columns in contract order with contract types."""
    return df.select([F.col(c).cast(t).alias(c) for c, t, _ in schemas.GOLD[table]])


def at_minute(date_col: str, minute_expr: str):
    """Wall-clock timestamp at a minute of day (DST-safe, same session time zone as bronze timestamps)."""
    return F.expr(
        f"make_timestamp(year({date_col}), month({date_col}), day({date_col}), "
        f"({minute_expr}) div 60, ({minute_expr}) % 60, 0)"
    )


def count_when(cond):
    return F.sum(F.when(cond, 1).otherwise(0))


def _player_uuid_udf(key: str):
    key_bytes = key.encode("utf-8")

    # Self-contained (stdlib only): UDF workers may not have the b4l package on sys.path.
    # Must stay equivalent to pipeline_helpers.hmac_hex.
    @F.pandas_udf("string")
    def player_uuid(ids: pd.Series) -> pd.Series:
        import hashlib
        import hmac

        return ids.map(
            lambda v: None if pd.isna(v) else hmac.new(key_bytes, str(v).encode("utf-8"), hashlib.sha256).hexdigest()
        )

    return player_uuid


player_uuid = _player_uuid_udf(HMAC_KEY)


# --------------------------------------------------------------------------------------------------
# Silver
# --------------------------------------------------------------------------------------------------


@dp.temporary_view(comment="Snapshot dates of the generator run (single row): as_of_date and window")
def run_snapshot():
    return bronze("run_info").select("run_id", "as_of_date", "window_start", "window_end").limit(1)


@dp.materialized_view(comment="Silver: the 16 courts (terrains), hall (salle: Main Hall / West Hall), show courts")
@dp.expect_or_drop("has_court_id", "court_id IS NOT NULL")
def courts():
    return bronze("courts").dropDuplicates(["court_id"])


@dp.materialized_view(
    comment="Silver: pseudonymised players (joueurs). No names, email, phone or postcode; player_uuid is "
    "HMAC-SHA256 of the booking-system id; age_band as of the snapshot; postcode_area = first 2 digits. "
    "has_valid_email / has_valid_phone flag malformed contact data (kept, warn-only expectations)."
)
@dp.expect("valid_email_format", "has_valid_email")
@dp.expect("valid_phone_format", "has_valid_phone")
@dp.expect_or_drop("has_player_uuid", "player_uuid IS NOT NULL")
def players():
    snap = spark.read.table("run_snapshot").select("as_of_date")
    return (
        bronze("players")
        .dropDuplicates(["player_id"])
        .crossJoin(F.broadcast(snap))
        .select(
            player_uuid(F.col("player_id")).alias("player_uuid"),
            "skill_tier",
            "segment",
            "preferred_lang",
            F.expr(ph.age_band_sql("year(as_of_date) - birth_year")).alias("age_band"),
            F.substring("home_postcode", 1, 2).alias("postcode_area"),
            "corporate_account_id",
            "joined_at",
            "marketing_opt_in",
            F.expr(ph.regex_match_sql("email", ph.EMAIL_PATTERN)).alias("has_valid_email"),
            F.expr(ph.regex_match_sql("phone", ph.PHONE_PATTERN)).alias("has_valid_phone"),
        )
    )


@dp.materialized_view(comment="Silver: corporate accounts (entreprises) near Malley whose employees book lunch slots")
@dp.expect_or_drop("has_account_id", "corporate_account_id IS NOT NULL")
def corporate_accounts():
    return bronze("corporate_accounts").dropDuplicates(["corporate_account_id"])


@dp.materialized_view(
    comment="Silver: pass wallets (Carte prépayée, Pass Liberté, Pass Chrono) per pseudonymised player, "
    "balances at snapshot time"
)
@dp.expect_or_drop("has_player_uuid", "player_uuid IS NOT NULL")
@dp.expect("non_negative_balance", "coalesce(balance_chf_remaining, 0) >= 0 AND coalesce(hours_remaining, 0) >= 0")
def pass_wallets():
    return (
        bronze("pass_wallets")
        .dropDuplicates(["wallet_id"])
        .withColumn("player_uuid", player_uuid(F.col("player_id")))
        .drop("player_id")
    )


@dp.materialized_view(
    comment="Silver: court blocks (Bad'matin, private lessons / cours privés, club BLA, school / scolaire sessions)"
)
@dp.expect_or_drop("valid_block_window", "start_ts IS NOT NULL AND end_ts IS NOT NULL AND start_ts < end_ts")
def court_blocks():
    return bronze("court_blocks").dropDuplicates(["block_id"])


@dp.materialized_view(
    comment="Silver: booking events (created, cancelled, no_show, checked_in) deduplicated by event_id. "
    "delivery_count > 1 marks events delivered several times; is_late_arriving marks events ingested "
    f"more than {ph.LATE_ARRIVAL_HOURS} h after they happened."
)
@dp.expect("single_delivery", "delivery_count = 1")
@dp.expect("on_time_ingestion", "NOT is_late_arriving")
@dp.expect_or_drop("valid_event_type", ph.in_list_sql("event_type", ph.EVENT_TYPES))
def booking_events():
    first_ingested = Window.partitionBy("event_id").orderBy(F.col("ingested_at").asc_nulls_last())
    lag_h = (F.unix_timestamp("ingested_at") - F.unix_timestamp("event_ts")) / 3600.0
    return (
        bronze("booking_events")
        .withColumn("delivery_count", F.count(F.lit(1)).over(Window.partitionBy("event_id")))
        .withColumn("_rn", F.row_number().over(first_ingested))
        .filter("_rn = 1")
        .drop("_rn")
        .withColumn("ingest_lag_h", lag_h)
        .withColumn("is_late_arriving", F.coalesce(F.col("ingest_lag_h") > ph.LATE_ARRIVAL_HOURS, F.lit(False)))
    )


@dp.temporary_view(comment="Bookings with court check, event history, calendar and weather, before the DQ rules")
def bookings_validated():
    known_courts = spark.read.table("courts").select("court_id", F.lit(True).alias("is_known_court"))
    events = spark.read.table("booking_events").groupBy("booking_id").agg(
        F.max_by("event_type", "event_ts").alias("last_event_type"),
        (F.max((F.col("event_type") == "checked_in").cast("int")) == 1).alias("has_check_in_event"),
    )
    cal = spark.read.table("calendar").select(
        F.col("date").alias("booking_date"), "day_type", "day_of_week", "is_school_holiday", "is_public_holiday"
    )
    weather = spark.read.table("weather").select(F.col("date").alias("booking_date"), "is_rainy")
    return (
        bronze("bookings")
        .dropDuplicates(["booking_id"])
        .withColumn("booker_uuid", player_uuid(F.col("booker_player_id")))
        .drop("booker_player_id")
        .withColumn("booking_date", F.to_date("start_ts"))
        .withColumn("start_minute", F.hour("start_ts") * 60 + F.minute("start_ts"))
        .withColumn("is_billed", F.col("status").isin(*ph.BILLED_STATUSES))
        .join(F.broadcast(known_courts), "court_id", "left")
        .withColumn("is_known_court", F.coalesce("is_known_court", F.lit(False)))
        .join(events, "booking_id", "left")
        .withColumn("has_check_in_event", F.coalesce("has_check_in_event", F.lit(False)))
        .join(F.broadcast(cal), "booking_date", "left")
        .join(F.broadcast(weather), "booking_date", "left")
    )


@dp.materialized_view(
    comment="Silver: valid bookings (réservations) with pseudonymised booker_uuid, calendar and weather flags, "
    "and the last event from the event history. Rows failing a DQ rule are dropped here and kept in "
    "bookings_quarantine."
)
@dp.expect_all_or_drop(ph.BOOKING_RULES)
def bookings():
    return spark.read.table("bookings_validated")


@dp.materialized_view(
    comment="Silver: quarantined bookings that failed a DQ rule (e.g. unknown court Court_99); "
    "reason lists the failed rules"
)
def bookings_quarantine():
    return (
        spark.read.table("bookings_validated")
        .filter(F.expr(ph.violates_any_sql(ph.BOOKING_RULES)))
        .withColumn("reason", F.expr(ph.violation_reason_sql(ph.BOOKING_RULES)))
    )


@dp.materialized_view(comment="Silver: everyone on court per booking (2 or 4 players), pseudonymised player_uuid")
@dp.expect_or_drop("known_booking", "is_known_booking")
@dp.expect_or_drop("has_player_uuid", "player_uuid IS NOT NULL")
def booking_participants():
    known = spark.read.table("bookings").select("booking_id", F.lit(True).alias("is_known_booking"))
    return (
        bronze("booking_participants")
        .dropDuplicates(["booking_id", "player_id"])
        .join(known, "booking_id", "left")
        .select(
            "booking_id",
            player_uuid(F.col("player_id")).alias("player_uuid"),
            "is_booker",
            F.coalesce("is_known_booking", F.lit(False)).alias("is_known_booking"),
        )
    )


@dp.materialized_view(comment="Silver: waitlist (liste d'attente) requests with pseudonymised player_uuid")
@dp.expect_or_drop("has_player_uuid", "player_uuid IS NOT NULL")
@dp.expect_or_drop("valid_window", "window_start_ts IS NOT NULL AND window_start_ts <= window_end_ts")
@dp.expect("valid_party_size", "coalesce(party_size IN (2, 4), false)")
def waitlist():
    return (
        bronze("waitlist")
        .dropDuplicates(["waitlist_id"])
        .withColumn("player_uuid", player_uuid(F.col("player_id")))
        .drop("player_id")
    )


@dp.materialized_view(comment="Silver: check-ins at reception (fréquentation / footfall, persons arriving)")
@dp.expect_or_drop("known_booking", "is_known_booking")
@dp.expect_or_drop("positive_persons", "persons IS NOT NULL AND persons > 0")
def check_ins():
    known = spark.read.table("bookings").select("booking_id", F.lit(True).alias("is_known_booking"))
    return (
        bronze("check_ins")
        .dropDuplicates(["check_in_id"])
        .join(known, "booking_id", "left")
        .withColumn("is_known_booking", F.coalesce("is_known_booking", F.lit(False)))
    )


@dp.materialized_view(comment="Silver: Lausanne daily weather (météo: pluie / rain, temperature); forecast for future days")
@dp.expect_or_drop("has_date", "date IS NOT NULL")
def weather():
    return bronze("weather_lausanne").dropDuplicates(["date"])


@dp.materialized_view(
    comment="Silver: calendar with closures, Vaud public holidays (jours fériés), school holidays "
    "(vacances scolaires), day_type and Vaudoise Aréna event days"
)
@dp.expect_or_drop("has_date", "date IS NOT NULL")
@dp.expect("valid_day_type", "day_type IN ('weekday', 'weekend_holiday', 'closed')")
def calendar():
    return bronze("calendar").dropDuplicates(["date"])


# --------------------------------------------------------------------------------------------------
# Gold
# --------------------------------------------------------------------------------------------------


def _open_days() -> DataFrame:
    """Open calendar days in the generated window, with weather and snapshot columns."""
    snap = spark.read.table("run_snapshot")
    weather = spark.read.table("weather").select("date", "is_rainy", "temp_mean_c")
    return (
        spark.read.table("calendar")
        .crossJoin(F.broadcast(snap))
        .filter(F.col("date").between(F.col("window_start"), F.col("window_end")))
        .filter(F.col("day_type").isin(*ph.OPEN_DAY_TYPES))
        .join(weather, "date", "left")
        .withColumn("weekday0", F.col("day_of_week") - 1)
    )


def _hour_overlaps(df: DataFrame) -> DataFrame:
    """One row per (interval, clock hour) touched, with the overlapping minutes. Needs court_id, start_ts, end_ts."""
    return (
        df.withColumn("date", F.to_date("start_ts"))
        .withColumn("_start", F.hour("start_ts") * 60 + F.minute("start_ts"))
        .withColumn(
            "duration_minutes",
            ((F.unix_timestamp("end_ts") - F.unix_timestamp("start_ts")) / 60).cast("int"),
        )
        .withColumn("_end", F.col("_start") + F.col("duration_minutes"))
        .withColumn("hour", F.explode(F.sequence(F.floor(F.col("_start") / 60), F.floor((F.col("_end") - 1) / 60))))
        .withColumn("hour", F.col("hour").cast("int"))
        .withColumn(
            "overlap_minutes",
            F.least(F.col("_end"), (F.col("hour") + 1) * 60) - F.greatest(F.col("_start"), F.col("hour") * 60),
        )
    )


def _check_in_persons() -> DataFrame:
    return spark.read.table("check_ins").groupBy("booking_id").agg(F.sum("persons").alias("persons"))


@gold_view(
    "court_utilization_hourly",
    "Gold: taux d'occupation (occupancy rate) et chiffre d'affaires (revenue) par terrain et par heure. "
    "One row per court x open day x clock hour of the rentable window (weekdays 08:00-22:00, weekends and "
    "holidays 09:30-19:00, so the 09:00 hour has 30 minutes), for the 24 weeks of history and the next 14 days. "
    "Booked minutes count billed and upcoming bookings (completed, upcoming, annulation tardive / late cancel, "
    "no-show); blocked minutes are Bad'matin, private lessons and club BLA. Revenue is split by minutes. "
    + GENIE_TERMS,
)
def court_utilization_hourly():
    keys = ["court_id", "date", "hour"]
    hours = spark.createDataFrame(ph.hour_grid_rows(), ph.HOUR_GRID_DDL)
    courts_df = spark.read.table("courts").select("court_id", "hall_zone", "is_show_court")
    grid = _open_days().join(F.broadcast(hours), ["day_type", "weekday0"]).crossJoin(F.broadcast(courts_df))

    bookings_df = spark.read.table("bookings").filter(F.col("status").isin(*ph.BILLED_OR_UPCOMING))
    booked = (
        _hour_overlaps(bookings_df)
        .groupBy(*keys)
        .agg(
            F.sum("overlap_minutes").alias("booked_minutes"),
            F.sum(F.col("revenue_chf") * F.col("overlap_minutes") / F.col("duration_minutes")).alias("revenue_chf"),
        )
    )
    blocked = (
        _hour_overlaps(spark.read.table("court_blocks"))
        .groupBy(*keys)
        .agg(F.sum("overlap_minutes").alias("blocked_minutes"))
    )
    started = (
        bookings_df.join(_check_in_persons(), "booking_id", "left")
        .groupBy("court_id", F.col("booking_date").alias("date"), F.hour("start_ts").alias("hour"))
        .agg(F.count(F.lit(1)).alias("bookings_started"), F.sum("persons").alias("check_in_persons"))
    )

    booked_min = F.least(F.coalesce("booked_minutes", F.lit(0)), F.col("available_minutes"))
    df = (
        grid.join(booked, keys, "left")
        .join(blocked, keys, "left")
        .join(started, keys, "left")
        .withColumn("booked_minutes", booked_min)
        .withColumn(
            "blocked_minutes",
            F.least(F.coalesce("blocked_minutes", F.lit(0)), F.col("available_minutes") - F.col("booked_minutes")),
        )
        .withColumn(
            "occupancy_rate",
            (F.col("booked_minutes") + F.col("blocked_minutes")) / F.col("available_minutes"),
        )
        .withColumn("revenue_chf", F.coalesce("revenue_chf", F.lit(0.0)))
        .withColumn("bookings_started", F.coalesce("bookings_started", F.lit(0)))
        .withColumn("check_in_persons", F.coalesce("check_in_persons", F.lit(0)))
        .withColumn("hour_start_ts", at_minute("date", "hour * 60"))
    )
    return conform(df, "court_utilization_hourly")


@gold_view(
    "daily_kpis",
    "Gold: indicateurs journaliers (daily KPIs) per open day, history and next 14 days: chiffre d'affaires "
    "(court revenue) and block revenue (Bad'matin, cours, club), taux d'occupation over the 16 courts, "
    "bookings, annulations gratuites (early) and tardives (late, billed), no-shows, liste d'attente "
    "(waitlist) requests and resolutions, fréquentation (footfall) and distinct players. " + GENIE_TERMS,
)
def daily_kpis():
    hourly = (
        spark.read.table(gold_name("court_utilization_hourly"))
        .groupBy("date")
        .agg(
            F.sum("revenue_chf").alias("revenue_chf"),
            (
                F.sum(F.col("booked_minutes") + F.col("blocked_minutes")) / F.sum("available_minutes")
            ).alias("occupancy_rate"),
        )
    )
    bookings_df = spark.read.table("bookings")
    status = F.col("status")
    counts = bookings_df.groupBy(F.col("booking_date").alias("date")).agg(
        count_when(status.isin(*ph.BILLED_OR_UPCOMING)).alias("bookings"),
        count_when(status == "cancelled_early").alias("cancellations_early"),
        count_when(status == "cancelled_late").alias("cancellations_late"),
        count_when(status == "no_show").alias("no_shows"),
    )
    blocks = (
        spark.read.table("court_blocks")
        .groupBy(F.to_date("start_ts").alias("date"))
        .agg(F.sum("revenue_chf").alias("block_revenue_chf"))
    )
    waitlist_df = (
        spark.read.table("waitlist")
        .groupBy(F.col("requested_date").alias("date"))
        .agg(
            F.count(F.lit(1)).alias("waitlist_requests"),
            count_when(F.col("status") == "resolved").alias("waitlist_resolved"),
        )
    )
    footfall = (
        spark.read.table("check_ins")
        .groupBy(F.to_date("checked_in_at").alias("date"))
        .agg(F.sum("persons").alias("check_in_persons"))
    )
    on_court = bookings_df.filter(status.isin("completed", "upcoming")).select("booking_id", "booking_date")
    players_df = (
        spark.read.table("booking_participants")
        .join(on_court, "booking_id")
        .groupBy(F.col("booking_date").alias("date"))
        .agg(F.countDistinct("player_uuid").alias("unique_players"))
    )

    df = _open_days().select("date", "day_name", "day_type", "is_rainy", "is_school_holiday")
    for part in (hourly, counts, blocks, waitlist_df, footfall, players_df):
        df = df.join(part, "date", "left")
    zero_fill = {
        c: 0
        for c in (
            "bookings", "cancellations_early", "cancellations_late", "no_shows", "waitlist_requests",
            "waitlist_resolved", "check_in_persons", "unique_players",
        )
    }
    df = df.fillna(zero_fill).fillna({"revenue_chf": 0.0, "block_revenue_chf": 0.0})
    return conform(df, "daily_kpis")


def _mode(df: DataFrame, value: str, alias: str) -> DataFrame:
    """Most frequent value per player_uuid; ties go to the smallest value."""
    ranked = Window.partitionBy("player_uuid").orderBy(F.desc("count"), F.asc(value))
    return (
        df.groupBy("player_uuid", value)
        .count()
        .withColumn("_rn", F.row_number().over(ranked))
        .filter("_rn = 1")
        .select("player_uuid", F.col(value).alias(alias))
    )


@gold_view(
    "player_features",
    "Gold: player features (caractéristiques joueurs) at the snapshot date for churn and matchmaking. "
    "Plays = times on court in completed bookings; last 30 days vs the previous 30 days; preferred day, hour and "
    "tier over 90 days; late cancellation (annulation tardive) and no-show rates as booker; active pass "
    "(Carte prépayée, Pass Liberté, Pass Chrono) and expiry; top 3 doubles partners; churn_risk_rule = a regular "
    "(at least 2 plays per 30 days over days 31-120 before the snapshot, avg_monthly_plays_baseline) whose plays "
    "in the last 30 days fell to 35% of that baseline or less. player_uuid is pseudonymised.",
)
def player_features():
    snap = F.broadcast(spark.read.table("run_snapshot").select("as_of_date"))
    bookings_df = spark.read.table("bookings")

    plays = (
        spark.read.table("booking_participants")
        .select("booking_id", "player_uuid")
        .join(
            bookings_df.filter(F.col("status") == "completed").select(
                "booking_id", "booking_date", "start_ts", "tier", "day_of_week"
            ),
            "booking_id",
        )
        .crossJoin(snap)
        .filter(F.col("booking_date") < F.col("as_of_date"))
        .withColumn("days_ago", F.datediff("as_of_date", "booking_date"))
    )
    days_ago = F.col("days_ago")
    counts = plays.groupBy("player_uuid").agg(
        count_when(days_ago <= 30).alias("plays_last_30d"),
        count_when((days_ago > 30) & (days_ago <= 60)).alias("plays_prev_30d"),
        count_when(days_ago <= 90).alias("plays_last_90d"),
        count_when(days_ago.between(*ph.CHURN_BASELINE_DAYS)).alias("_plays_baseline"),
        F.max("booking_date").alias("last_play_date"),
    )
    recent = plays.filter(days_ago <= 90).withColumn("start_hour", F.hour("start_ts"))
    pref_day = _mode(recent, "day_of_week", "preferred_day_of_week")
    pref_hour = _mode(recent, "start_hour", "preferred_hour")
    pref_tier = _mode(recent, "tier", "preferred_tier")

    as_booker = (
        bookings_df.crossJoin(snap)
        .filter(F.col("booking_date") < F.col("as_of_date"))
        .groupBy(F.col("booker_uuid").alias("player_uuid"))
        .agg(
            F.avg("lead_time_h").alias("avg_lead_time_h"),
            F.avg((F.col("status") == "cancelled_late").cast("double")).alias("late_cancel_rate"),
            F.avg((F.col("status") == "no_show").cast("double")).alias("no_show_rate"),
        )
    )

    as_of_ts = F.col("as_of_date").cast("timestamp")
    latest_wallet = Window.partitionBy("player_uuid").orderBy(F.desc("purchased_at"), F.desc("wallet_id"))
    wallets = (
        spark.read.table("pass_wallets")
        .crossJoin(snap)
        .filter(
            (F.col("status") == "active")
            & (F.col("purchased_at") < as_of_ts)
            & (F.col("expires_at").isNull() | (F.col("expires_at") >= as_of_ts))
        )
        .withColumn("_rn", F.row_number().over(latest_wallet))
        .filter("_rn = 1")
        .select(
            "player_uuid",
            F.col("product").alias("pass_product"),
            F.col("hours_remaining").alias("pass_hours_remaining"),
            F.col("balance_chf_remaining").alias("pass_balance_chf"),
            F.col("expires_at").alias("pass_expires_at"),
        )
    )

    pairs = plays.select("booking_id", "player_uuid")
    partner_rank = Window.partitionBy("player_uuid").orderBy(F.desc("count"), F.asc("partner_uuid"))
    partners = (
        pairs.join(pairs.withColumnRenamed("player_uuid", "partner_uuid"), "booking_id")
        .filter(F.col("player_uuid") != F.col("partner_uuid"))
        .groupBy("player_uuid", "partner_uuid")
        .count()
        .withColumn("_rn", F.row_number().over(partner_rank))
        .filter("_rn <= 3")
        .groupBy("player_uuid")
        .agg(
            F.expr("transform(array_sort(collect_list(struct(_rn, partner_uuid))), x -> x.partner_uuid)").alias(
                "top_partner_uuids"
            )
        )
    )

    last, prev = F.col("plays_last_30d"), F.col("plays_prev_30d")
    baseline = F.col("avg_monthly_plays_baseline")
    lo, hi = ph.CHURN_BASELINE_DAYS
    baseline_months = (hi - lo + 1) / 30.0
    change = F.when(prev > 0, (last - prev) / prev)
    df = (
        spark.read.table("players")
        .select("player_uuid", "segment", "skill_tier", "preferred_lang", "age_band", "joined_at")
        .crossJoin(snap)
    )
    for part in (counts, pref_day, pref_hour, pref_tier, as_booker, wallets, partners):
        df = df.join(part, "player_uuid", "left")
    df = (
        df.fillna({"plays_last_30d": 0, "plays_prev_30d": 0, "plays_last_90d": 0, "_plays_baseline": 0})
        .withColumn("avg_monthly_plays_baseline", F.col("_plays_baseline") / baseline_months)
        .withColumn("play_change_pct", change)
        .withColumn("days_since_last_play", F.datediff("as_of_date", "last_play_date"))
        .withColumn("tenure_days", F.datediff("as_of_date", F.to_date("joined_at")))
        .withColumn("has_active_pass", F.col("pass_product").isNotNull())
        .withColumn("days_to_pass_expiry", F.datediff(F.to_date("pass_expires_at"), "as_of_date"))
        .withColumn("top_partner_uuids", F.coalesce("top_partner_uuids", F.array().cast("array<string>")))
        .withColumn(
            "churn_risk_rule",
            (baseline >= ph.CHURN_MIN_MONTHLY_PLAYS) & (last <= ph.CHURN_DROP_RATIO * baseline),
        )
    )
    return conform(df, "player_features")


@gold_view(
    "slot_features",
    "Gold: slot features (créneaux) for the fill-probability model. One row per open day and valid booking "
    "start on the 30-minute grid, court-agnostic across the 16 courts, for history and the next 14 days. "
    "Labels: final_booked_courts and fill_rate (billed bookings starting at the slot, history only). "
    "Features: base tier (Tarif journée, midi, soir, week-end) and price, calendar (vacances scolaires, jours "
    "fériés, Vaudoise Aréna), weather (pluie, forecast for future), courts blocked, bookings at D-3, same "
    "weekday/time fill rate over the previous 4 and 8 weeks, liste d'attente (waitlist) requests.",
)
def slot_features():
    slot_grid = spark.createDataFrame(ph.slot_grid_rows(), ph.SLOT_GRID_DDL)
    slots = (
        _open_days()
        .join(F.broadcast(slot_grid), ["day_type", "weekday0"])
        .withColumn("slot_start_ts", at_minute("date", "start_minute"))
        .withColumn("slot_end_ts", at_minute("date", "start_minute + 60"))
        .withColumn("is_future", F.col("date") >= F.col("as_of_date"))
    )
    slot_keys = slots.select("date", "start_minute", "slot_start_ts", "slot_end_ts")

    blocks = spark.read.table("court_blocks").select("court_id", "start_ts", "end_ts", F.to_date("start_ts").alias("date"))
    blocked = (
        slot_keys.join(blocks, "date")
        .filter((F.col("start_ts") < F.col("slot_end_ts")) & (F.col("end_ts") > F.col("slot_start_ts")))
        .groupBy("date", "start_minute")
        .agg(F.countDistinct("court_id").alias("blocked_courts"))
    )

    d3 = F.col("start_ts") - F.expr("INTERVAL 3 DAYS")
    status = F.col("status")
    at_slot = (
        spark.read.table("bookings")
        .groupBy(F.col("booking_date").alias("date"), "start_minute")
        .agg(
            count_when(status.isin(*ph.BILLED_STATUSES)).alias("final_booked_courts"),
            count_when(
                (F.col("booked_at") <= d3) & (F.col("cancelled_at").isNull() | (F.col("cancelled_at") > d3))
            ).alias("booked_courts_at_d3"),
            count_when(status == "upcoming").alias("booked_courts_now"),
            F.avg("price_shown_chf").alias("avg_price_shown_chf"),
        )
    )

    wl = spark.read.table("waitlist").select(
        F.col("requested_date").alias("date"), "window_start_ts", "window_end_ts"
    )
    waitlisted = (
        slot_keys.join(wl, "date")
        .filter(F.col("slot_start_ts").between(F.col("window_start_ts"), F.col("window_end_ts")))
        .groupBy("date", "start_minute")
        .agg(F.count(F.lit(1)).alias("waitlist_requests"))
    )

    keys = ["date", "start_minute"]
    future = F.col("is_future")
    day_num = F.datediff("date", F.to_date(F.lit("1970-01-01")))
    same_slot = Window.partitionBy("day_of_week", "start_minute").orderBy("_day_num")
    df = (
        slots.join(blocked, keys, "left")
        .join(at_slot, keys, "left")
        .join(waitlisted, keys, "left")
        .fillna({"blocked_courts": 0, "booked_courts_at_d3": 0, "waitlist_requests": 0})
        .withColumn("courts_available", F.lit(pr.N_COURTS) - F.col("blocked_courts"))
        .withColumn("final_booked_courts", F.when(~future, F.coalesce("final_booked_courts", F.lit(0))))
        .withColumn("booked_courts_now", F.when(future, F.coalesce("booked_courts_now", F.lit(0))))
        .withColumn(
            "fill_rate",
            F.when(F.col("courts_available") > 0, F.col("final_booked_courts") / F.col("courts_available")),
        )
        .withColumn("_day_num", day_num)
        .withColumn("hist_fill_rate_4w", F.avg("fill_rate").over(same_slot.rangeBetween(-28, -1)))
        .withColumn("hist_fill_rate_8w", F.avg("fill_rate").over(same_slot.rangeBetween(-56, -1)))
        .withColumn("slot_key", F.date_format("slot_start_ts", "yyyy-MM-dd'T'HHmm"))
        .withColumn("slot_date", F.col("date"))
        .withColumn("start_hour", F.col("start_minute") / 60.0)
    )
    return conform(df, "slot_features")


@gold_view(
    "sponsor_metrics",
    "Gold: anonymised monthly sponsor metrics (media kit sponsoring) from completed bookings in history: "
    "distinct players, visits on court, fréquentation (footfall from reception check-ins) and court-hours, "
    "by segment (e.g. Corporate_Lunch, Family_Weekend), age_band, time_band (morning, lunch / midi, "
    "afternoon, evening / soir, weekend), hall_zone (salle) and show_court (Prime Show Courts 1-4). "
    f"k-anonymity: cells with fewer than {schemas.K_ANONYMITY_MIN} players are suppressed (metrics null). "
    "Court-hours and footfall of a booking count once per group value present on court.",
)
def sponsor_metrics():
    snap = F.broadcast(spark.read.table("run_snapshot").select("as_of_date"))
    courts_df = spark.read.table("courts").select("court_id", "hall_zone", "is_show_court")
    played = (
        spark.read.table("bookings")
        .filter(F.col("status") == "completed")
        .crossJoin(snap)
        .filter(F.col("booking_date") < F.col("as_of_date"))
        .join(F.broadcast(courts_df), "court_id")
        .join(_check_in_persons(), "booking_id", "left")
        .select(
            "booking_id",
            F.trunc("booking_date", "month").alias("month"),
            "duration_h",
            F.coalesce("persons", F.lit(0)).alias("persons"),
            "hall_zone",
            F.when(F.col("is_show_court"), "show_court").otherwise("standard_court").alias("show_court"),
            F.expr(ph.time_band_sql("start_minute", "day_type")).alias("time_band"),
        )
    )
    players_df = spark.read.table("players").select("player_uuid", "segment", "age_band")
    visits = (
        spark.read.table("booking_participants")
        .select("booking_id", "player_uuid")
        .join(played, "booking_id")
        .join(players_df, "player_uuid", "left")
        .select(
            "month",
            "booking_id",
            "player_uuid",
            "duration_h",
            "persons",
            F.expr(
                "stack(5, 'segment', segment, 'age_band', age_band, 'time_band', time_band, "
                "'hall_zone', hall_zone, 'show_court', show_court) AS (metric_group, metric_key)"
            ),
        )
        .withColumn("metric_key", F.coalesce("metric_key", F.lit("unknown")))
    )
    cell = ["month", "metric_group", "metric_key"]
    people = visits.groupBy(*cell).agg(
        F.countDistinct("player_uuid").alias("unique_players"),
        F.count(F.lit(1)).alias("player_visits"),
    )
    courts_used = (
        visits.dropDuplicates(cell + ["booking_id"])
        .groupBy(*cell)
        .agg(F.sum("persons").alias("check_in_persons"), F.sum("duration_h").alias("court_hours"))
    )
    suppressed = F.col("unique_players") < schemas.K_ANONYMITY_MIN
    df = people.join(courts_used, cell).withColumn("suppressed", suppressed)
    for c in ("unique_players", "player_visits", "check_in_persons", "court_hours"):
        df = df.withColumn(c, F.when(~F.col("suppressed"), F.col(c)))
    return conform(df, "sponsor_metrics")
