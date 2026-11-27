# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Lakebase operational state
# MAGIC
# MAGIC Prepares the Lakebase (Postgres) project `b4l-ops`, declared in `resources/b4l_lakebase.yml` (SPEC.md 5.C):
# MAGIC
# MAGIC 1. **Synced table** `{prefix}_sync.slot_price_recommendations`: a snapshot copy of the gold recommendations,
# MAGIC    refreshed on every run.
# MAGIC 2. **Operational tables** in `{prefix}_ops`: `live_court_slots`, `matchmaking_lobbies`, `agent_action_log`.
# MAGIC 3. **Merge** of the recommendations into `live_court_slots`, one row per court and upcoming slot start.
# MAGIC    Open courts stay at the tariff (`proposed`) until staff approve or override them in the staff app; those
# MAGIC    decisions survive reruns. Booked and blocked courts follow the latest bookings; past slots become `expired`.
# MAGIC 4. **Grants** for the staff app's service principal: change a court's price and status, append to the action
# MAGIC    log, nothing else.

# COMMAND ----------

# MAGIC %pip install -q "databricks-sdk>=0.81" "psycopg[binary]>=3.1"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "serverless_stable_yqq46w_catalog")
dbutils.widgets.text("schema_prefix", "b4l")
dbutils.widgets.text("lakebase_project", "b4l-ops")
dbutils.widgets.text("staff_app_name", "b4l-staff-ops")

CATALOG = dbutils.widgets.get("catalog").strip()
PREFIX = dbutils.widgets.get("schema_prefix").strip()
PROJECT = dbutils.widgets.get("lakebase_project").strip()
APP_NAME = dbutils.widgets.get("staff_app_name").strip()

# COMMAND ----------

import os
import re
import sys
import time

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service import postgres as pg

sys.path.insert(0, os.path.abspath("."))

from b4l import lakebase as lb

for part in (CATALOG, PREFIX):
    if not re.fullmatch(r"[A-Za-z0-9_]+", part):
        raise ValueError(f"Invalid identifier: {part!r}")

OPS, SYNC = lb.schema_names(PREFIX)
BRONZE, SILVER, GOLD = (f"`{CATALOG}`.`{PREFIX}_{layer}`" for layer in ("bronze", "silver", "gold"))
BRANCH = f"projects/{PROJECT}/branches/production"
ENDPOINT = f"{BRANCH}/endpoints/primary"
PG_DATABASE = "databricks_postgres"
spark.conf.set("spark.sql.session.timeZone", "UTC")

w = WorkspaceClient()
ME = w.current_user.me().user_name
AS_OF = spark.table(f"{BRONZE}.run_info").first()["as_of_date"]
print(f"project={PROJECT} as_of={AS_OF} schemas={OPS},{SYNC}")


def connect() -> psycopg.Connection:
    host = w.postgres.get_endpoint(name=ENDPOINT).status.hosts.host
    token = w.postgres.generate_database_credential(endpoint=ENDPOINT).token
    for attempt in range(5):
        try:
            conn = psycopg.connect(host=host, dbname=PG_DATABASE, user=ME, password=token, sslmode="require",
                                   autocommit=True, connect_timeout=30)
            conn.execute("SET TIME ZONE 'UTC'")
            return conn
        except psycopg.OperationalError:
            if attempt == 4:
                raise
            time.sleep(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Synced table (snapshot of `gold.slot_price_recommendations`)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SYNC}` COMMENT 'Lakebase synced tables (read-only copies of gold)'")
synced_id = f"{CATALOG}.{SYNC}.slot_price_recommendations"
source = f"{CATALOG}.{PREFIX}_gold.slot_price_recommendations"


def wait_synced(name: str, timeout_s: int = 1200) -> pg.SyncedTable:
    done = {pg.SyncedTableState.SYNCED_TABLE_ONLINE, pg.SyncedTableState.SYNCED_TABLE_ONLINE_NO_PENDING_UPDATE}
    failed = {pg.SyncedTableState.SYNCED_TABLE_OFFLINE_FAILED, pg.SyncedTableState.SYNCED_TABLE_ONLINE_PIPELINE_FAILED}
    deadline = time.time() + timeout_s
    while True:
        st = w.postgres.get_synced_table(name=name)
        state = st.status.detailed_state if st.status else None
        if state in done:
            return st
        if state in failed:
            raise RuntimeError(f"Synced table {name} failed: {st.status.message}")
        if time.time() > deadline:
            raise TimeoutError(f"Synced table {name} still {state}")
        time.sleep(15)


name = f"synced_tables/{synced_id}"
try:
    st = w.postgres.get_synced_table(name=name)
    pipeline_id = st.status.pipeline_id
    update = w.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=True)
    print(f"Refreshing snapshot: pipeline {pipeline_id}, update {update.update_id}")
    while True:
        u = w.pipelines.get_update(pipeline_id=pipeline_id, update_id=update.update_id).update
        if u.state.value in ("COMPLETED", "FAILED", "CANCELED"):
            break
        time.sleep(15)
    if u.state.value != "COMPLETED":
        raise RuntimeError(f"Snapshot refresh ended {u.state.value}")
except NotFound:
    print(f"Creating synced table {synced_id}")
    w.postgres.create_synced_table(
        synced_table=pg.SyncedTable(spec=pg.SyncedTableSyncedTableSpec(
            branch=BRANCH, postgres_database=PG_DATABASE, source_table_full_name=source,
            primary_key_columns=["slot_key"], create_database_objects_if_missing=True,
            scheduling_policy=pg.SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.SNAPSHOT,
            new_pipeline_spec=pg.NewPipelineSpec(storage_catalog=CATALOG, storage_schema=SYNC))),
        synced_table_id=synced_id,
    ).wait()
st = wait_synced(name)
print(f"{synced_id}: {st.status.detailed_state.value} {st.status.message or ''}".strip())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Operational tables and 3. merge into `live_court_slots`

# COMMAND ----------

slots = spark.sql(f"SELECT slot_key, slot_start_ts FROM {GOLD}.slot_price_recommendations").toPandas()
courts = spark.sql(f"SELECT court_id, hall_zone FROM {SILVER}.courts").toPandas()
bookings = spark.sql(f"SELECT court_id, start_ts, end_ts FROM {SILVER}.bookings WHERE status = 'upcoming'").toPandas()
blocks = spark.sql(f"SELECT court_id, start_ts, end_ts FROM {SILVER}.court_blocks WHERE end_ts >= '{AS_OF}'").toPandas()
states = lb.court_slot_states(slots, courts, bookings, blocks)
print(states["court_state"].value_counts().to_dict())

with connect() as conn:
    n_synced = conn.execute(f"SELECT count(*) FROM {SYNC}.slot_price_recommendations").fetchone()[0]
    if n_synced != len(slots):
        raise RuntimeError(f"Synced table has {n_synced} rows, gold has {len(slots)}")
    for stmt in lb.ddl(OPS):
        conn.execute(stmt)
    with conn.transaction():
        conn.execute(lb.STAGE_DDL)
        with conn.cursor().copy("COPY court_state FROM STDIN") as cp:
            for row in states.itertuples(index=False):
                cp.write_row(tuple(row))
        conn.execute(lb.merge_sql(OPS, SYNC))
        expired = conn.execute(lb.expire_sql(OPS)).rowcount
    summary = conn.execute(f"""SELECT status, count(*), round(avg(dynamic_price_chf)::numeric, 2)::float8
                               FROM {OPS}.live_court_slots GROUP BY 1 ORDER BY 1""").fetchall()
print(f"Synced rows: {n_synced}; newly expired: {expired}")
display(spark.createDataFrame(summary, "status string, court_slots long, avg_live_price_chf double"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Grants for the staff app's service principal

# COMMAND ----------

client_id = w.apps.get(APP_NAME).service_principal_client_id
roles = {r.status.postgres_role for r in w.postgres.list_roles(parent=BRANCH) if r.status}
if client_id not in roles:
    w.postgres.create_role(
        parent=BRANCH, role_id=f"app-{client_id[:8]}",
        role=pg.Role(spec=pg.RoleRoleSpec(identity_type=pg.RoleIdentityType.SERVICE_PRINCIPAL, postgres_role=client_id,
                                          auth_method=pg.RoleAuthMethod.LAKEBASE_OAUTH_V1)),
    ).wait()
    print(f"Created Postgres role for {APP_NAME}")
with connect() as conn:
    for stmt in lb.grant_statements(OPS, client_id):
        conn.execute(stmt)
    grants = conn.execute(
        "SELECT table_name, string_agg(DISTINCT privilege_type, ', ') FROM information_schema.role_table_grants "
        "WHERE grantee = %s AND table_schema = %s GROUP BY 1 ORDER BY 1", (client_id, OPS)).fetchall()
    column_grants = conn.execute(
        "SELECT table_name, privilege_type, string_agg(column_name, ', ' ORDER BY column_name) "
        "FROM information_schema.column_privileges WHERE grantee = %s AND table_schema = %s "
        "AND privilege_type <> 'SELECT' GROUP BY 1, 2 ORDER BY 1, 2", (client_id, OPS)).fetchall()
print(f"{APP_NAME} ({client_id}):")
for table, privileges in grants:
    print(f"  {OPS}.{table}: {privileges}")
for table, privilege, columns in column_grants:
    if not any(t == table and privilege in p for t, p in grants):
        print(f"  {OPS}.{table}: {privilege} only on ({columns})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo slots

# COMMAND ----------

dm = spark.table(f"{BRONZE}.demo_moments").toPandas()
starts = [r.slot_start_ts for r in dm.itertuples() if r.moment_id in (1, 2)]
with connect() as conn:
    rows = conn.execute(
        f"""SELECT slot_key, court_id, status, base_price_chf::float8, recommended_price_chf::float8,
                   dynamic_price_chf::float8, reason
            FROM {OPS}.live_court_slots WHERE slot_start = ANY(%s) ORDER BY slot_key, court_id""",
        ([s.to_pydatetime().replace(tzinfo=None) for s in starts],)).fetchall()
display(spark.createDataFrame(rows, "slot_key string, court_id string, status string, base double, recommended double, "
                                    "live double, reason string"))
