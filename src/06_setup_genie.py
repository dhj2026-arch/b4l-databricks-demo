# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Genie space on the B4L gold tables
# MAGIC
# MAGIC Creates or updates (matched by title) the **B4L Smart Badminton Center** Genie space on the five gold
# MAGIC tables, with French/English instructions, synonyms, 10 sample questions and example SQL. Then grants the
# MAGIC `b4l-staff-ops` app service principal read access to the gold schema and `CAN_RUN` on the space.
# MAGIC The space definition lives in `src/b4l/genie_space.py`; see `GENIE_SETUP.md` at the repo root.

# COMMAND ----------

dbutils.widgets.text("catalog", "serverless_stable_yqq46w_catalog")
dbutils.widgets.text("schema_prefix", "b4l")
dbutils.widgets.text("warehouse_id", "")
dbutils.widgets.text("staff_app_name", "b4l-staff-ops")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA_PREFIX = dbutils.widgets.get("schema_prefix").strip()
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id").strip()
STAFF_APP_NAME = dbutils.widgets.get("staff_app_name").strip()

# COMMAND ----------

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import iam, sql

from b4l import genie_space as gs

w = WorkspaceClient()
if not WAREHOUSE_ID:
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouse found; pass the warehouse_id job parameter")
    WAREHOUSE_ID = warehouses[0].id
print(f"catalog={CATALOG} schema_prefix={SCHEMA_PREFIX} warehouse_id={WAREHOUSE_ID}")
print("Tables:", *gs.table_identifiers(CATALOG, SCHEMA_PREFIX), sep="\n  ")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or update the space
# MAGIC Uses `w.genie.create_space` / `update_space` / `list_spaces` when the installed SDK has them, otherwise
# MAGIC the same REST endpoints through `w.api_client.do`.

# COMMAND ----------


def list_spaces() -> list[dict]:
    spaces, token = [], None
    while True:
        if hasattr(w.genie, "list_spaces"):
            resp = w.genie.list_spaces(page_token=token).as_dict()
        else:
            resp = w.api_client.do("GET", "/api/2.0/genie/spaces", query={"page_token": token} if token else None)
        spaces.extend(resp.get("spaces") or [])
        token = resp.get("next_page_token")
        if not token:
            return spaces


def create_space(serialized: str) -> dict:
    if hasattr(w.genie, "create_space"):
        return w.genie.create_space(
            warehouse_id=WAREHOUSE_ID,
            serialized_space=serialized,
            title=gs.SPACE_TITLE,
            description=gs.SPACE_DESCRIPTION,
        ).as_dict()
    body = {
        "warehouse_id": WAREHOUSE_ID,
        "serialized_space": serialized,
        "title": gs.SPACE_TITLE,
        "description": gs.SPACE_DESCRIPTION,
    }
    return w.api_client.do("POST", "/api/2.0/genie/spaces", body=body)


def update_space(space_id: str, serialized: str) -> dict:
    if hasattr(w.genie, "update_space"):
        return w.genie.update_space(
            space_id,
            warehouse_id=WAREHOUSE_ID,
            serialized_space=serialized,
            title=gs.SPACE_TITLE,
            description=gs.SPACE_DESCRIPTION,
        ).as_dict()
    body = {
        "warehouse_id": WAREHOUSE_ID,
        "serialized_space": serialized,
        "title": gs.SPACE_TITLE,
        "description": gs.SPACE_DESCRIPTION,
    }
    return w.api_client.do("PATCH", f"/api/2.0/genie/spaces/{space_id}", body=body)


existing = [s for s in list_spaces() if s.get("title") == gs.SPACE_TITLE]
if len(existing) > 1:
    print(f"WARNING: {len(existing)} spaces titled '{gs.SPACE_TITLE}'; updating the first one")
space_id = existing[0]["space_id"] if existing else None

errors = []
applied_variant = None
for variant, serialized in gs.payload_variants(CATALOG, SCHEMA_PREFIX):
    try:
        if space_id:
            update_space(space_id, serialized)
            action = "updated"
        else:
            space_id = create_space(serialized)["space_id"]
            action = "created"
        applied_variant = variant
        break
    except Exception as e:  # noqa: BLE001 - try a simpler payload before giving up
        errors.append(f"{variant}: {e}")
        print(f"Payload '{variant}' rejected: {e}")

if applied_variant is None:
    raise RuntimeError("Genie space could not be created or updated:\n" + "\n".join(errors))
if applied_variant != "full":
    print(f"NOTE: applied the degraded payload '{applied_variant}'. Add the missing parts in the UI (GENIE_SETUP.md).")

space_url = gs.space_url(w.config.host, space_id)
print(f"Genie space {action}: {space_id}\n{space_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grant the staff app service principal
# MAGIC `USE CATALOG`, `USE SCHEMA` and `SELECT` on the gold schema (through the SQL warehouse) plus `CAN_RUN` on
# MAGIC the space. The warehouse `CAN_USE` comes from the app's `sql-warehouse` resource.

# COMMAND ----------


def run_sql(statement: str) -> None:
    resp = w.statement_execution.execute_statement(statement=statement, warehouse_id=WAREHOUSE_ID, wait_timeout="30s")
    while resp.status.state in (sql.StatementState.PENDING, sql.StatementState.RUNNING):
        time.sleep(2)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state != sql.StatementState.SUCCEEDED:
        raise RuntimeError(f"{statement}: {resp.status.state} {resp.status.error}")


sp_client_id = None
try:
    app = w.apps.get(name=STAFF_APP_NAME)
    sp_client_id = app.service_principal_client_id
    print(f"App '{STAFF_APP_NAME}' service principal: {app.service_principal_name} ({sp_client_id})")
except Exception as e:  # noqa: BLE001
    print(f"App '{STAFF_APP_NAME}' not found ({e}). Deploy the bundle, then re-run this task to grant access.")

if sp_client_id:
    for stmt in gs.grant_statements(CATALOG, SCHEMA_PREFIX, sp_client_id):
        try:
            run_sql(stmt)
            print(f"OK   {stmt}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {stmt}\n     {e}\n     Run it manually as the schema owner.")
    try:
        w.permissions.update(
            request_object_type="genie",
            request_object_id=space_id,
            access_control_list=[
                iam.AccessControlRequest(
                    service_principal_name=sp_client_id, permission_level=iam.PermissionLevel.CAN_RUN
                )
            ],
        )
        print(f"OK   CAN_RUN on Genie space {space_id} for {sp_client_id}")
    except Exception as e:  # noqa: BLE001
        print(
            f"FAIL CAN_RUN on Genie space for {sp_client_id}: {e}\n"
            "     Share the space with the app service principal (Can Run) in the Genie UI."
        )

# COMMAND ----------

dbutils.jobs.taskValues.set(key="genie_space_id", value=space_id)
dbutils.jobs.taskValues.set(key="genie_space_url", value=space_url)
dbutils.notebook.exit(space_id)
