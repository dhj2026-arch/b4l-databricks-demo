# Databricks notebook source
# MAGIC %md
# MAGIC # 03b · Batch scoring: price recommendations, churn scores, cancellation risk
# MAGIC
# MAGIC Loads the three `champion` models from `{prefix}_models` and writes three gold tables for the staff app,
# MAGIC Genie and (in step 4) the Lakebase synced table:
# MAGIC
# MAGIC | Table | One row per | What it says |
# MAGIC |---|---|---|
# MAGIC | `slot_price_recommendations` | upcoming slot (date + start time) | recommended price, rule applied, expected fill and revenue vs the tariff |
# MAGIC | `player_scores` | regular player | churn probability, risk band, suggested retention action |
# MAGIC | `booking_cancellation_risk` | upcoming booking | late-cancellation / no-show probability, open waitlist matches |
# MAGIC
# MAGIC **How a price is chosen.** For each upcoming slot, every price in the tier's guardrail range (CHF 1 steps) is
# MAGIC tried. Courts already booked keep their locked price, so only courts still expected to sell earn the new price:
# MAGIC expected new revenue = price × (predicted courts booked − courts already booked). The best price is kept only if it
# MAGIC beats the tariff by at least 10% (15% at weekends), and weekday slots in the 2 hours before the evening tariff are
# MAGIC never discounted (the 03c backtest showed those discounts pull players from full evenings). Otherwise the
# MAGIC published tariff stays. Prices never leave CHF 18–42 or ±40%.

# COMMAND ----------

# MAGIC %pip install -q lightgbm==4.5.0 "mlflow-skinny[databricks]>=2.16,<4"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "serverless_stable_yqq46w_catalog")
dbutils.widgets.text("schema_prefix", "b4l")
dbutils.widgets.text("secret_scope", "b4l")

CATALOG = dbutils.widgets.get("catalog").strip()
PREFIX = dbutils.widgets.get("schema_prefix").strip()
SECRET_SCOPE = dbutils.widgets.get("secret_scope").strip()

# COMMAND ----------

import os
import re
import sys

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

sys.path.insert(0, os.path.abspath("."))

from b4l import ml_features as mf
from b4l import pipeline_helpers as ph
from b4l import schemas, scoring

for part in (CATALOG, PREFIX):
    if not re.fullmatch(r"[A-Za-z0-9_]+", part):
        raise ValueError(f"Invalid identifier: {part!r}")

BRONZE, SILVER, GOLD = (f"`{CATALOG}`.`{PREFIX}_{layer}`" for layer in ("bronze", "silver", "gold"))
UC_MODELS = f"{CATALOG}.{PREFIX}_models"
spark.conf.set("spark.sql.session.timeZone", "UTC")
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

AS_OF = spark.table(f"{BRONZE}.run_info").first()["as_of_date"]
print(f"as_of={AS_OF}")


def champion(name: str):
    fq = f"{UC_MODELS}.{name}"
    version = int(client.get_model_version_by_alias(fq, "champion").version)
    return mlflow.sklearn.load_model(f"models:/{fq}@champion"), version


def write_gold(pdf: pd.DataFrame, table: str, comment: str) -> None:
    columns = schemas.SCORES[table]
    fq = f"{GOLD}.`{table}`"
    spark.sql(f"CREATE OR REPLACE TABLE {fq} ({schemas.ddl(columns)}) COMMENT '{comment}'")
    sdf = spark.createDataFrame(pdf[[c for c, _, _ in columns]])
    sdf.select(*[F.col(c).cast(t).alias(c) for c, t, _ in columns]).write.mode("append").saveAsTable(
        f"{CATALOG}.{PREFIX}_gold.{table}")
    print(f"Wrote {fq}: {len(pdf):,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Slot price recommendations

# COMMAND ----------

fill_model, fill_version = champion("fill_probability_model")
slot_features = spark.sql(f"SELECT * FROM {GOLD}.slot_features WHERE is_future").toPandas()
slots = mf.fill_features_from_gold(slot_features, AS_OF)
recs = scoring.recommend_prices(slots, fill_model.predict)
recs["model_version"] = fill_version
recs["as_of_date"] = AS_OF
recs["lead_days"] = recs["lead_days"].astype(int)
write_gold(recs.sort_values("slot_start_ts"), "slot_price_recommendations",
           "Recommended dynamic court prices for upcoming slots (fill_probability_model, guardrails applied)")

summary = recs.groupby(["tier", "applied_rule"]).agg(
    slots=("slot_key", "size"), avg_price=("recommended_price_chf", "mean"),
    avg_change_pct=("price_change_pct", "mean"), uplift_chf=("expected_revenue_uplift_chf", "sum"),
).round(2)
display(summary.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Player churn scores (regulars)

# COMMAND ----------

churn_model, churn_version = champion("player_churn_model")
bookings = spark.sql(f"""
    SELECT booking_id, booker_uuid, court_id, start_ts, booked_at, lead_time_h, tier, party_size, payment_method,
           status, booking_date
    FROM {SILVER}.bookings
""").toPandas()
plays = spark.sql(f"""
    SELECT p.player_uuid, p.booking_id, b.booking_date
    FROM {SILVER}.booking_participants p JOIN {SILVER}.bookings b USING (booking_id)
    WHERE b.status = 'completed'
""").toPandas()
players = spark.sql(f"SELECT player_uuid, segment, skill_tier, joined_at FROM {SILVER}.players").toPandas()

snap = mf.churn_snapshot(plays, bookings, players, AS_OF, with_label=False)
scores = scoring.churn_scores(snap, churn_model.predict_proba(snap[mf.CHURN_FEATURES])[:, 1])
scores["model_version"] = churn_version
scores["as_of_date"] = AS_OF
write_gold(scores.sort_values("churn_probability", ascending=False), "player_scores",
           "Churn scores for regular players (player_churn_model) with a suggested retention action")
display(scores.groupby("churn_risk_band").agg(players=("player_uuid", "size"),
                                              avg_probability=("churn_probability", "mean"),
                                              already_dropped=("churn_risk_rule", "mean")).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Booking cancellation risk (upcoming bookings)

# COMMAND ----------

cancel_model, cancel_version = champion("cancellation_risk_model")
weather = spark.sql(f"SELECT date, is_rainy FROM {SILVER}.weather").toPandas()
waitlist = spark.sql(f"SELECT window_start_ts, window_end_ts, status FROM {SILVER}.waitlist").toPandas()
rows = mf.cancel_rows(bookings, players, weather)
upcoming = rows[rows["status"] == "upcoming"].reset_index(drop=True)
risk = scoring.cancellation_scores(upcoming, cancel_model.predict_proba(upcoming[mf.CANCEL_FEATURES])[:, 1], waitlist)
risk["model_version"] = cancel_version
risk["as_of_date"] = AS_OF
write_gold(risk.sort_values("late_cancel_probability", ascending=False), "booking_cancellation_risk",
           "Late-cancellation and no-show risk for upcoming bookings (cancellation_risk_model) with waitlist matches")
display(risk.groupby("risk_band").agg(bookings=("booking_id", "size"),
                                      avg_probability=("late_cancel_probability", "mean"),
                                      with_waitlist=("open_waitlist_matches", lambda s: int((s > 0).sum()))).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo moments check
# MAGIC
# MAGIC The quiet Tuesday lunch, the rainy Saturday, the drifting regular and the likely late canceller, as scored.
# MAGIC Player ids are pseudonymised here with the same key as the pipeline (the key is never printed).

# COMMAND ----------

key = dbutils.secrets.get(SECRET_SCOPE, "pseudonymisation_key")
dm = spark.table(f"{BRONZE}.demo_moments").toPandas().set_index("moment_id")

cols = ["slot_key", "tier", "base_price_chf", "courts_available", "booked_courts_now", "hist_fill_rate_4w",
        "recommended_price_chf", "applied_rule", "expected_fill_at_base", "expected_fill_at_recommended",
        "expected_revenue_uplift_chf", "reason"]
for moment in (2, 1):
    ts = pd.Timestamp(dm.loc[moment, "slot_start_ts"])
    print(dm.loc[moment, "title"])
    display(recs[pd.to_datetime(recs["slot_start_ts"]) == ts][cols])

def show_rank(df: pd.DataFrame, mask: pd.Series, score: str, what: str, cols: list[str]) -> None:
    if not mask.any():
        print(f"  not found among {len(df):,} {what}")
        return
    print(f"  rank {int((df[score] > df.loc[mask, score].max()).sum()) + 1} of {len(df):,} {what}")
    display(df[mask][cols])


drifter = ph.hmac_hex(key, dm.loc[3, "player_id"])
print(dm.loc[3, "title"])
show_rank(scores, scores["player_uuid"] == drifter, "churn_probability", "regulars",
          ["churn_probability", "churn_risk_band", "plays_last_30d", "avg_monthly_plays_baseline", "suggested_action"])
if not (scores["player_uuid"] == drifter).any():
    mine = plays[plays["player_uuid"] == drifter]
    print(f"  in silver players: {(players['player_uuid'] == drifter).any()}, completed plays: {len(mine)}, "
          f"last play: {mine['booking_date'].max() if len(mine) else None}")

print(dm.loc[4, "title"])
show_rank(risk, risk["booking_id"] == dm.loc[4, "booking_id"], "late_cancel_probability", "upcoming bookings",
          ["booking_id", "court_id", "start_ts", "booker_prior_bookings", "booker_prior_late_rate",
           "late_cancel_probability", "risk_band", "open_waitlist_matches", "suggested_action"])
