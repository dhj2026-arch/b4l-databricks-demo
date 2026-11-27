# Databricks notebook source
# MAGIC %md
# MAGIC # 03c · Static vs dynamic pricing backtest
# MAGIC
# MAGIC What would the last 24 weeks have earned with dynamic prices? (SPEC.md 5.F)
# MAGIC
# MAGIC | Table | One row per | What it says |
# MAGIC |---|---|---|
# MAGIC | `static_vs_dynamic_revenue` | week × tier × price sensitivity | revenue, occupancy, turned-away requests at the tariff vs with dynamic prices |
# MAGIC | `slot_demand_curves` | upcoming slot × candidate price | expected courts and revenue at each price (the curve behind each recommendation) |
# MAGIC
# MAGIC **Method.** The simulator from notebook 01 is replayed over the same 24 weeks with the same players, booking
# MAGIC intents, weather and random draws; only the prices change. Each slot gets the price the fill model would have
# MAGIC recommended two weeks ahead (nothing booked yet, last 4 weeks' fill known). Pass-hour and family-rate bookings,
# MAGIC courses and club blocks keep their tariff. Flexible players can move to a cheaper slot nearby, so a discount
# MAGIC that only pulls players away from another slot shows up as a loss there. Because nobody knows how price-sensitive
# MAGIC B4L players really are, the replay runs at three sensitivities and the result is a range, to be confirmed by a pilot.

# COMMAND ----------

# MAGIC %pip install -q lightgbm==4.5.0 "mlflow-skinny[databricks]>=2.16,<4"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "serverless_stable_yqq46w_catalog")
dbutils.widgets.text("schema_prefix", "b4l")
dbutils.widgets.text("n_players", "3000")

CATALOG = dbutils.widgets.get("catalog").strip()
PREFIX = dbutils.widgets.get("schema_prefix").strip()
N_PLAYERS = int(dbutils.widgets.get("n_players").strip())

# COMMAND ----------

import os
import re
import sys

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

sys.path.insert(0, os.path.abspath("."))

from b4l import backtest, demand, schemas, scoring, synth
from b4l import ml_features as mf
from b4l import pricing_rules as pr

for part in (CATALOG, PREFIX):
    if not re.fullmatch(r"[A-Za-z0-9_]+", part):
        raise ValueError(f"Invalid identifier: {part!r}")

BRONZE, GOLD = (f"`{CATALOG}`.`{PREFIX}_{layer}`" for layer in ("bronze", "gold"))
UC_MODELS = f"{CATALOG}.{PREFIX}_models"
spark.conf.set("spark.sql.session.timeZone", "UTC")
mlflow.set_registry_uri("databricks-uc")

run_info = spark.table(f"{BRONZE}.run_info").toPandas().iloc[0]
SEED, AS_OF = int(run_info["seed"]), run_info["as_of_date"]
fq_model = f"{UC_MODELS}.fill_probability_model"
MODEL_VERSION = int(MlflowClient().get_model_version_by_alias(fq_model, "champion").version)
fill_model = mlflow.sklearn.load_model(f"models:/{fq_model}@champion")
print(f"seed={SEED} as_of={AS_OF} fill_probability_model v{MODEL_VERSION}")


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
# MAGIC ## Static world (what happened) and the dynamic prices it would have had

# COMMAND ----------

world = synth.build_world(SEED, AS_OF, N_PLAYERS)
intents = demand.build_intents(world)
static = demand.simulate(world, intents)
bronze_n = spark.sql(
    f"SELECT count(DISTINCT booking_id) FROM {BRONZE}.bookings WHERE booking_id NOT LIKE 'BX%'"
).first()[0]
if len(static.bookings) != bronze_n:
    raise RuntimeError(f"Rebuilt world has {len(static.bookings)} bookings, bronze has {bronze_n}: "
                       "seed, as_of_date or n_players differ from notebook 01")
print(f"Rebuilt world matches bronze: {bronze_n:,} bookings")

slots = backtest.history_slots(world, static)
print(f"Pricing policy: {scoring.DEFAULT_POLICY_NAME} {scoring.DEFAULT_POLICY}")
decisions = scoring.recommend_prices(slots, fill_model.predict)
display(decisions.groupby(["tier", "applied_rule"]).agg(
    slots=("slot_key", "size"), avg_price=("recommended_price_chf", "mean"),
    avg_change_pct=("price_change_pct", "mean")).round(3).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Replay the 24 weeks with dynamic prices at low, mid and high price sensitivity

# COMMAND ----------

grid = backtest.dynamic_price_grid(world, decisions)
dynamic = {s: demand.simulate(world, intents, grid, k) for s, k in demand.ELASTICITY_SCENARIOS.items()}
cmp = backtest.compare(world, static, dynamic, decisions)
cmp["model_version"] = MODEL_VERSION
cmp["as_of_date"] = AS_OF
for c in ("turned_away_static", "turned_away_dynamic", "slots", "slots_repriced"):
    cmp[c] = cmp[c].fillna(0).astype(int)
write_gold(cmp, "static_vs_dynamic_revenue",
           "Backtest: 24 weeks of revenue and occupancy at the tariff vs with dynamic prices, "
           "per week, tier and price-sensitivity scenario (simulated)")

display(backtest.headline(cmp).round(4))
display(cmp.groupby(["elasticity_scenario", "tier"]).agg(
    static_chf=("static_revenue_chf", "sum"), dynamic_chf=("dynamic_revenue_chf", "sum"),
    change_chf=("revenue_change_chf", "sum"), static_occ=("static_occupancy", "mean"),
    dynamic_occ=("dynamic_occupancy", "mean"), avg_price_change=("avg_price_change_pct", "mean"),
).round(3).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Do the off-peak discounts pay for themselves?
# MAGIC
# MAGIC Off-peak (Day, Lunch) and peak (Evening, Weekend) prices are switched on separately, so the effect of each,
# MAGIC including players moving between slots, is visible on its own. Not written to a table.

# COMMAND ----------

rows = []
for label, tiers in (("off-peak only", {pr.DAY, pr.LUNCH}), ("peak only", {pr.EVENING, pr.WEEKEND}),
                     ("all tiers", set(pr.TIERS))):
    part = decisions.copy()
    keep = ~part["tier"].isin(tiers)
    part.loc[keep, "recommended_price_chf"] = part.loc[keep, "base_price_chf"]
    part.loc[keep, "price_change_pct"] = 0.0
    g = backtest.dynamic_price_grid(world, part)
    for s, k in demand.ELASTICITY_SCENARIOS.items():
        c = backtest.compare(world, static, {s: demand.simulate(world, intents, g, k)}, part)
        by_tier = c.groupby("tier")["revenue_change_chf"].sum()
        rows.append({"repriced": label, "scenario": s, "total_change_chf": by_tier.sum(),
                     **{f"{t}_change_chf": by_tier.get(t, 0.0) for t in pr.TIERS}})
display(pd.DataFrame(rows).round(0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pricing rule variants
# MAGIC
# MAGIC The same replay for each rule set in `scoring.POLICY_VARIANTS` (higher minimum gain, no discounts just before
# MAGIC the evening, stricter weekends, no lunch discounts). The default policy is the one written above.

# COMMAND ----------

rows = []
for name, policy in scoring.POLICY_VARIANTS.items():
    dec = scoring.recommend_prices(slots, fill_model.predict, policy)
    g = backtest.dynamic_price_grid(world, dec)
    res = {s: demand.simulate(world, intents, g, k) for s, k in demand.ELASTICITY_SCENARIOS.items()}
    c = backtest.compare(world, static, res, dec)
    head = backtest.headline(c).set_index("elasticity_scenario")
    by_tier = c.groupby(["elasticity_scenario", "tier"])["revenue_change_chf"].sum()
    for s in demand.ELASTICITY_SCENARIOS:
        rows.append({"policy": name, "scenario": s,
                     "change_per_week_chf": head.loc[s, "revenue_change_per_week_chf"],
                     "change_pct": head.loc[s, "revenue_change_pct"],
                     "offpeak_occ_pts": head.loc[s, "offpeak_occupancy_change_pts"],
                     "share_repriced": head.loc[s, "share_slots_repriced"],
                     **{f"{t}_24w_chf": by_tier.get((s, t), 0.0) for t in pr.TIERS}})
variants = pd.DataFrame(rows)
display(variants.round(3))
print(variants.round(3).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demand curves for upcoming slots

# COMMAND ----------

slot_features = spark.sql(f"SELECT * FROM {GOLD}.slot_features WHERE is_future").toPandas()
upcoming = mf.fill_features_from_gold(slot_features, AS_OF)
recs = spark.sql(f"SELECT slot_key, recommended_price_chf FROM {GOLD}.slot_price_recommendations").toPandas()
curves = backtest.demand_curves(upcoming, recs, fill_model.predict)

dm = spark.table(f"{BRONZE}.demo_moments").toPandas()
labels = {pd.Timestamp(r.slot_start_ts): r.title for r in dm.itertuples() if pd.notna(r.slot_start_ts)
          and r.moment_id in (1, 2)}
curves["demo_label"] = pd.to_datetime(curves["slot_start_ts"]).map(labels)
curves["model_version"] = MODEL_VERSION
curves["as_of_date"] = AS_OF
write_gold(curves.sort_values(["slot_start_ts", "candidate_price_chf"]), "slot_demand_curves",
           "Expected courts and revenue at each candidate price for upcoming slots (fill_probability_model)")

display(curves[curves["demo_label"].notna()][
    ["demo_label", "candidate_price_chf", "is_tariff", "is_recommended", "pred_fill", "expected_courts_booked",
     "expected_revenue_chf"]].round(3))
