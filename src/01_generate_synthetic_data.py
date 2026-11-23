# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Synthetic bronze data for the B4L Smart Badminton Center
# MAGIC
# MAGIC Simulates 24 weeks of history plus the next 14 days of bookings for the 16 courts at Malley (SPEC.md 4 and 4.1),
# MAGIC then overwrites the `{prefix}_bronze` tables with column comments. Steps:
# MAGIC
# MAGIC 1. Create the bronze/silver/gold/models schemas and the HMAC pseudonymisation secret if missing.
# MAGIC 2. Run the deterministic simulator (`src/b4l`), validate it (hard rules, statistical checks, demo moments).
# MAGIC 3. Inject ~1% deliberate defects for the Lakeflow expectations and write bronze.
# MAGIC 4. Fail the task if any hard rule or demo moment failed (statistical checks only warn).

# COMMAND ----------

dbutils.widgets.text("catalog", "serverless_stable_yqq46w_catalog")
dbutils.widgets.text("schema_prefix", "b4l")
dbutils.widgets.text("seed", "20261123")
dbutils.widgets.text("secret_scope", "b4l")
dbutils.widgets.text("as_of_date", "")
dbutils.widgets.text("n_players", "3000")

CATALOG = dbutils.widgets.get("catalog").strip()
PREFIX = dbutils.widgets.get("schema_prefix").strip()
SEED = int(dbutils.widgets.get("seed").strip())
SECRET_SCOPE = dbutils.widgets.get("secret_scope").strip()
AS_OF_RAW = dbutils.widgets.get("as_of_date").strip()
N_PLAYERS = int(dbutils.widgets.get("n_players").strip())

# COMMAND ----------

import os
import re
import secrets
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.abspath("."))

from b4l import generator, schemas

for part in (CATALOG, PREFIX):
    if not re.fullmatch(r"[A-Za-z0-9_]+", part):
        raise ValueError(f"Invalid identifier: {part!r}")

AS_OF = date.fromisoformat(AS_OF_RAW) if AS_OF_RAW else datetime.now(ZoneInfo("Europe/Zurich")).date()
BRONZE = f"`{CATALOG}`.`{PREFIX}_bronze`"
# Timestamps are naive Lausanne wall-clock times; a UTC session stores them unshifted (the pipeline reads in UTC too).
spark.conf.set("spark.sql.session.timeZone", "UTC")
print(f"catalog={CATALOG} prefix={PREFIX} seed={SEED} as_of={AS_OF} players={N_PLAYERS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schemas and pseudonymisation secret

# COMMAND ----------

LAYER_COMMENTS = {
    "bronze": "B4L raw synthetic booking-system data (contains fictitious PII)",
    "silver": "B4L cleaned, deduplicated and pseudonymised data (Lakeflow pipeline)",
    "gold": "B4L analytics tables for Genie, the staff app and the sponsor portal",
    "models": "B4L ML features, models and batch predictions",
}
for layer, comment in LAYER_COMMENTS.items():
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{PREFIX}_{layer}` COMMENT '{comment}'")

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
if SECRET_SCOPE not in {s.name for s in w.secrets.list_scopes()}:
    w.secrets.create_scope(scope=SECRET_SCOPE)
    print(f"Created secret scope '{SECRET_SCOPE}'")
if "pseudonymisation_key" not in {k.key for k in w.secrets.list_secrets(scope=SECRET_SCOPE)}:
    w.secrets.put_secret(scope=SECRET_SCOPE, key="pseudonymisation_key", string_value=secrets.token_hex(32))
    print(f"Created secret {SECRET_SCOPE}/pseudonymisation_key (random, never printed)")
else:
    print(f"Secret {SECRET_SCOPE}/pseudonymisation_key exists; keeping it so player_uuid values stay stable")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate and validate

# COMMAND ----------

bronze, world, res = generator.generate(SEED, AS_OF, n_players=N_PLAYERS)
report = bronze["data_quality_report"]
bronze = generator.inject_dirty_rows(bronze, SEED)

print(f"Window {world.window_start} .. {world.window_end}")
for name in schemas.BRONZE:
    print(f"  {name:<22} {len(bronze[name]):>8,} rows")
display(report)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write bronze (overwrite)

# COMMAND ----------

TABLE_COMMENTS = {
    "run_info": "Generator run: seed, snapshot date (as_of_date) and generated window",
    "courts": "The 16 badminton courts (terrains) at Malley, hall and show-court flag",
    "players": "Players (joueurs) of the booking system, fictitious PII; pseudonymised in silver",
    "corporate_accounts": "Fictitious companies near Malley whose employees book lunch slots",
    "pass_wallets": "Prepaid products: Carte prépayée (CHF balance), Pass Liberté and Pass Chrono (hours)",
    "pass_transactions": "Pass ledger: purchases, debits for bookings, refunds and expiries",
    "court_blocks": "Courts blocked for Bad'matin, private lessons, club BLA and school sessions (never re-priced)",
    "bookings": "Court bookings (réservations): 24 weeks of history and the bookings made so far for the next 14 days",
    "booking_participants": "Everyone on court per booking (2 or 4 players)",
    "booking_events": "Booking event stream (created, cancelled, no_show, checked_in) with ingestion time",
    "waitlist": "Waitlist (liste d'attente) requests and their outcome",
    "check_ins": "Reception check-ins (fréquentation / footfall)",
    "weather_lausanne": "Daily Lausanne weather (météo); forecast for future days",
    "calendar": "Calendar with Vaud public and school holidays, closures, day type and Vaudoise Aréna events",
    "demo_moments": "Seeded demo moments the presenter can point to (drifter, late canceller, Chrono, quiet lunch, rain)",
    "data_quality_report": "Generator data-quality report: hard rules, statistical checks and demo moments",
}

for name, columns in schemas.BRONZE.items():
    fq = f"{BRONZE}.`{name}`"
    pdf = generator.to_schema(name, bronze[name])
    comment = TABLE_COMMENTS.get(name, "").replace("'", "\\'")
    spark.sql(f"CREATE OR REPLACE TABLE {fq} ({schemas.ddl(columns)}) COMMENT '{comment}'")
    sdf = spark.createDataFrame(pdf, schema=", ".join(f"{c} {t}" for c, t, _ in columns))
    sdf.write.mode("append").saveAsTable(f"{CATALOG}.{PREFIX}_bronze.{name}")
    print(f"Wrote {fq}: {len(pdf):,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gate on data quality

# COMMAND ----------

failed = report[~report["passed"]]
warn = failed[failed["category"] != "hard_rule"]
hard = failed[failed["category"] == "hard_rule"]
for r in warn.itertuples():
    print(f"WARN {r.check_name}: observed {r.observed}, expected {r.expected}")
if len(hard):
    details = "\n".join(f"  {r.check_name}: observed {r.observed}, expected {r.expected}" for r in hard.itertuples())
    raise RuntimeError(f"{len(hard)} hard rule(s) failed; try another seed or as_of_date:\n{details}")

run_id = bronze["run_info"]["run_id"].iloc[0]
print(f"All hard rules and demo moments passed ({len(report)} checks, {len(warn)} warnings). run_id={run_id}")
dbutils.jobs.taskValues.set(key="run_id", value=run_id)
dbutils.jobs.taskValues.set(key="as_of_date", value=AS_OF.isoformat())
