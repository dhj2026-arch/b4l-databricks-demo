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
# MAGIC | `apply_dynamic_price_with_guardrails(slot_id, proposed_price)` | Final price, rule and reason (CHF 18–42, ±40%, whole francs; past and fully booked slots keep the tariff) |
# MAGIC | `trigger_player_concierge(player_uuid, trigger_type)` | A French or English draft message: waitlist backfill, doubles match, Pass Chrono expiry, retention voucher |
# MAGIC | `find_doubles_partners(player_uuid)` | Up to 10 active players at the same level whose usual start hour is within 2 h |
# MAGIC | `get_open_waitlist(start_date, n_days)` | Open waitlist requests, to pick whom to offer a freed court |
# MAGIC
# MAGIC The definitions live in `src/b4l/day_context.py` and `src/b4l/agent_tools.py`. Each function is checked
# MAGIC against its Python twin before the grants.

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

import pandas as pd

from b4l import agent_tools as at
from b4l import day_context as dc

w = WorkspaceClient()
FN = dc.function_name(CATALOG, SCHEMA_PREFIX)
SILVER = f"`{CATALOG}`.`{SCHEMA_PREFIX}_silver`"
GOLD = f"`{CATALOG}`.`{SCHEMA_PREFIX}_gold`"
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
# MAGIC ## Staff action tools
# MAGIC Each check calls the function in SQL and compares the result with `src/b4l/agent_tools.py`.

# COMMAND ----------

for name, stmt in at.all_function_sql(CATALOG, SCHEMA_PREFIX).items():
    spark.sql(stmt)
    print(f"created {at.function_name(CATALOG, SCHEMA_PREFIX, name)}")
fn = lambda name: at.function_name(CATALOG, SCHEMA_PREFIX, name)  # noqa: E731


def union_rows(calls: list[str]) -> list:
    return spark.sql(" UNION ALL ".join(calls)).collect()


slots = spark.sql(f"""SELECT slot_key, tier, base_price_chf, is_future, booked_courts_now, courts_available
                      FROM {GOLD}.slot_features WHERE is_future OR slot_date = date_sub(DATE'{as_of}', 1)""").toPandas()
cases = [(s, p) for s in slots.sample(min(len(slots), 40), random_state=1).to_dict("records")
         for p in (5.0, 17.4, 19.5, 20.5, 26.5, 27.5, 35.2, 41.5, 50.0)]
got = union_rows([f"SELECT * FROM {fn(at.GUARDRAIL_FN)}('{s['slot_key']}_Court_03', {p})" for s, p in cases])
guard_bad = [(s["slot_key"], p, g.asDict()) for (s, p), g in zip(cases, got)
             if (g["final_price_chf"], g["applied_rule"], g["reason"])
             != tuple(at.guardrail(s, p)[k] for k in ("final_price_chf", "applied_rule", "reason"))]
print(f"guardrails: {len(cases)} cases, {len(guard_bad)} mismatches")

players = spark.sql(f"SELECT * FROM {GOLD}.player_features").toPandas()
waitlist = spark.sql(f"SELECT * FROM {fn(at.WAITLIST_FN)}(DATE'{as_of}', 14)").toPandas()
first_wl = waitlist.sort_values(["window_start_ts", "waitlist_id"]).groupby("player_uuid").first()
pick = {
    at.WAITLIST_BACKFILL: list(waitlist["player_uuid"].unique()[:10]) + list(players["player_uuid"].sample(5, random_state=2)),
    at.DOUBLES_MATCH: list(players["player_uuid"].sample(10, random_state=4)),
    at.PASS_EXPIRY: list(players.loc[players["pass_product"] == "Pass_Chrono", "player_uuid"].head(15)),
    at.RETENTION_VOUCHER: list(players.loc[players["churn_risk_rule"], "player_uuid"].head(10)),
}
calls = [(u, t) for t, us in pick.items() for u in us]
got = union_rows([f"SELECT * FROM {fn(at.CONCIERGE_FN)}('{u}', '{t}')" for u, t in calls])
pmap = players.set_index("player_uuid")
concierge_bad = []
for (u, t), g in zip(calls, got):
    e = at.concierge_message({**pmap.loc[u].to_dict(), "player_uuid": u}, t, as_of,
                             first_wl.loc[u].to_dict() if u in first_wl.index else None)
    if (g["eligible"], g["message"], g["note"]) != (e["eligible"], e["message"], e["note"]):
        concierge_bad.append((t, u, g.asDict(), e))
print(f"concierge: {len(calls)} calls, {len(concierge_bad)} mismatches")

who = list(players["player_uuid"].sample(10, random_state=5))
partners_bad = [u for u in who
                if [r["partner_uuid"] for r in spark.sql(f"SELECT * FROM {fn(at.PARTNERS_FN)}('{u}')").collect()]
                != at.doubles_partners(players, u)["partner_uuid"].tolist()]
print(f"doubles partners: {len(who)} players, {len(partners_bad)} mismatches; open waitlist requests: {len(waitlist)}")
if guard_bad or concierge_bad or partners_bad:
    raise AssertionError(f"Tool checks failed: {guard_bad[:3]} {concierge_bad[:3]} {partners_bad[:3]}")

# COMMAND ----------

example = pick[at.RETENTION_VOUCHER][0] if pick[at.RETENTION_VOUCHER] else players["player_uuid"].iloc[0]
display(spark.sql(f"SELECT * FROM {fn(at.CONCIERGE_FN)}('{example}', '{at.RETENTION_VOUCHER}')"))
display(spark.sql(f"SELECT * FROM {fn(at.GUARDRAIL_FN)}('{slots.loc[slots['is_future'], 'slot_key'].iloc[0]}', 45)"))

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
    for stmt in dc.grant_statements(CATALOG, SCHEMA_PREFIX, sp_client_id) + at.grant_statements(
            CATALOG, SCHEMA_PREFIX, sp_client_id):
        spark.sql(stmt)
        print(f"OK   {stmt}")
    grants = [(name, r["ActionType"]) for name in (FN, *(fn(t) for t in at.FUNCTIONS))
              for r in spark.sql(f"SHOW GRANTS `{sp_client_id}` ON FUNCTION {name}").collect()]
    display(pd.DataFrame(grants, columns=["function", "privilege"]))
