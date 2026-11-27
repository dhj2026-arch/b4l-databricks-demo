# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Read-only Unity Catalog functions (agent tools)
# MAGIC
# MAGIC UC functions cannot write to tables: they compute an answer, and the staff app does any write (to Lakebase)
# MAGIC after a person confirms. This notebook creates the functions, checks them, and grants the staff app
# MAGIC `EXECUTE` on each one.
# MAGIC
# MAGIC | Function | Returns |
# MAGIC |---|---|
# MAGIC | `get_day_context(start_date, n_days)` | One row per day: opening hours, weather, Vaud public and school holidays, Vaudoise Aréna events and a demand hint |
# MAGIC
# MAGIC The definitions live in `src/b4l/day_context.py`.

# COMMAND ----------

dbutils.widgets.text("catalog", "serverless_stable_yqq46w_catalog")
dbutils.widgets.text("schema_prefix", "b4l")
dbutils.widgets.text("staff_app_name", "b4l-staff-ops")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA_PREFIX = dbutils.widgets.get("schema_prefix").strip()
STAFF_APP_NAME = dbutils.widgets.get("staff_app_name").strip()

# COMMAND ----------

import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.abspath("."))

from databricks.sdk import WorkspaceClient

from b4l import day_context as dc

w = WorkspaceClient()
FN = dc.function_name(CATALOG, SCHEMA_PREFIX)
SILVER = f"`{CATALOG}`.`{SCHEMA_PREFIX}_silver`"
print(f"catalog={CATALOG} schema_prefix={SCHEMA_PREFIX} function={FN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## `get_day_context`

# COMMAND ----------

spark.sql(dc.function_sql(CATALOG, SCHEMA_PREFIX))
lo, hi = spark.sql(f"SELECT min(date), max(date) FROM {SILVER}.calendar").first()
n_days = (hi - lo).days + 1
got = {}
start = lo
while start <= hi:
    for r in spark.sql(f"SELECT date, demand_hint FROM {FN}(DATE'{start}', {dc.MAX_DAYS})").collect():
        got[r["date"]] = r["demand_hint"]
    start += timedelta(days=dc.MAX_DAYS)

calendar = spark.sql(f"""SELECT c.*, w.is_rainy, w.temp_max_c FROM {SILVER}.calendar c
                         LEFT JOIN {SILVER}.weather w USING (date)""").collect()
mismatches = [(r["date"], got.get(r["date"]), dc.demand_hint(r.asDict())) for r in calendar
              if got.get(r["date"]) != dc.demand_hint(r.asDict())]
print(f"{len(got)} of {n_days} days returned; {len(mismatches)} demand hints differ from src/b4l/day_context.py")
if len(got) != n_days or mismatches:
    raise AssertionError(f"get_day_context check failed: {mismatches[:5]}")

# COMMAND ----------

as_of = spark.sql(f"SELECT max(as_of_date) FROM `{CATALOG}`.`{SCHEMA_PREFIX}_gold`.player_features").first()[0]
display(spark.sql(f"SELECT * FROM {FN}(DATE'{as_of}', 14) ORDER BY date"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grant the staff app `EXECUTE`
# MAGIC `USE CATALOG` and `USE SCHEMA` on gold come from `06_setup_genie`.

# COMMAND ----------

try:
    sp_client_id = w.apps.get(name=STAFF_APP_NAME).service_principal_client_id
except Exception as e:  # noqa: BLE001
    sp_client_id = None
    print(f"App '{STAFF_APP_NAME}' not found ({e}). Deploy the bundle, then re-run this task to grant access.")
if sp_client_id:
    for stmt in dc.grant_statements(CATALOG, SCHEMA_PREFIX, sp_client_id):
        spark.sql(stmt)
        print(f"OK   {stmt}")
    display(spark.sql(f"SHOW GRANTS ON FUNCTION {FN}"))
