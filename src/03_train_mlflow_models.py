# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Train the B4L models (fill probability, churn, cancellation risk)
# MAGIC
# MAGIC Trains three LightGBM models, logs them to MLflow and registers them in Unity Catalog under `{prefix}_models`
# MAGIC with the alias `champion` (SPEC.md 5.B):
# MAGIC
# MAGIC | Model | Predicts | Training data |
# MAGIC |---|---|---|
# MAGIC | `fill_probability_model` | share of available courts booked for a slot, given its price | simulated price-test replays of the 24 weeks of history |
# MAGIC | `player_churn_model` | a regular drops below 35% of their usual monthly plays in the next 30 days | silver bookings, weekly snapshots |
# MAGIC | `cancellation_risk_model` | a booking ends as a late cancellation or no-show | silver bookings |
# MAGIC
# MAGIC **Why price-test replays?** The real history was sold at the published tariff, so price never varies and a model
# MAGIC can't learn how demand reacts to it. The notebook re-runs the same simulator as notebook 01 (same seed, same
# MAGIC players) with a random price inside the guardrails for every hour, like an A/B price test. The fill model is
# MAGIC constrained so a higher price never predicts a higher fill. Each model is validated on the last 4 weeks of
# MAGIC history, then refitted on all of it before registration.

# COMMAND ----------

# MAGIC %pip install -q lightgbm==4.5.0 "mlflow-skinny[databricks]>=2.16,<4"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "serverless_stable_yqq46w_catalog")
dbutils.widgets.text("schema_prefix", "b4l")
dbutils.widgets.text("n_players", "3000")
dbutils.widgets.text("n_replays", "4")

CATALOG = dbutils.widgets.get("catalog").strip()
PREFIX = dbutils.widgets.get("schema_prefix").strip()
N_PLAYERS = int(dbutils.widgets.get("n_players").strip())
N_REPLAYS = int(dbutils.widgets.get("n_replays").strip())

# COMMAND ----------

import os
import re
import sys
import time
from datetime import timedelta

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

sys.path.insert(0, os.path.abspath("."))

from b4l import demand, synth
from b4l import ml_features as mf
from b4l import pricing_rules as pr

for part in (CATALOG, PREFIX):
    if not re.fullmatch(r"[A-Za-z0-9_]+", part):
        raise ValueError(f"Invalid identifier: {part!r}")

BRONZE, SILVER, MODELS = (f"`{CATALOG}`.`{PREFIX}_{layer}`" for layer in ("bronze", "silver", "models"))
UC_MODELS = f"{CATALOG}.{PREFIX}_models"
spark.conf.set("spark.sql.session.timeZone", "UTC")

run_info = spark.table(f"{BRONZE}.run_info").toPandas().iloc[0]
SEED, AS_OF, WINDOW_START = int(run_info["seed"]), run_info["as_of_date"], run_info["window_start"]
VALID_FROM = AS_OF - timedelta(days=28)
print(f"seed={SEED} as_of={AS_OF} window_start={WINDOW_START} validation from {VALID_FROM}")

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/{PREFIX}_models")
client = MlflowClient()
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}")


def save(pdf: pd.DataFrame, table: str, comment: str) -> None:
    fq = f"{MODELS}.`{table}`"
    spark.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq)
    spark.sql(f"COMMENT ON TABLE {fq} IS '{comment}'")
    log(f"Wrote {fq}: {len(pdf):,} rows")


def register(model, X: pd.DataFrame, name: str, metrics: dict, params: dict, predict) -> int:
    """Log the model refitted on all rows, register it in UC and point the champion alias at the new version."""
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        info = mlflow.sklearn.log_model(
            model, artifact_path="model", signature=infer_signature(X, predict(X.head(200))),
            input_example=X.head(5), registered_model_name=f"{UC_MODELS}.{name}",
            pip_requirements=[f"lightgbm=={lgb.__version__}", "scikit-learn", "pandas", "numpy"],
        )
    version = int(info.registered_model_version)
    client.set_registered_model_alias(f"{UC_MODELS}.{name}", "champion", version)
    log(f"Registered {UC_MODELS}.{name} v{version} (alias champion): {metrics}")
    return version

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rebuild the simulator world and check it matches bronze

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fill probability model (price-test replays)

# COMMAND ----------

replays = []
for r in range(N_REPLAYS):
    rng = np.random.default_rng([SEED, 300, r])
    prices = mf.exploration_price_grid(world, rng)
    res = demand.simulate(world, intents, prices, 1.0)
    replays.append(mf.fill_training_rows(world, res, prices, rng, replay_id=r))
    print(f"replay {r}: {len(res.bookings):,} bookings, revenue CHF {res.bookings['revenue_chf'].sum():,.0f}")
fill = pd.concat(replays, ignore_index=True)
fill_cols = ["replay_id", "slot_date", "slot_start_ts", "tier", "base_price_chf", "final_booked_courts",
             "booked_courts_now"] + mf.FILL_FEATURES + [mf.FILL_LABEL]
save(fill[fill_cols], "fill_training_slots",
     "Fill-model training rows from simulated price-test replays of the history (random price per hour within guardrails)")

FILL_PARAMS = dict(objective="cross_entropy", n_estimators=500, learning_rate=0.04, num_leaves=31,
                   min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                   monotone_constraints=mf.FILL_MONOTONE, monotone_constraints_method="advanced",
                   random_state=SEED, verbose=-1)


def fit_fill(df: pd.DataFrame) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(**FILL_PARAMS)
    m.fit(df[mf.FILL_FEATURES], df[mf.FILL_LABEL], categorical_feature=mf.FILL_CATEGORICAL)
    return m


is_val = pd.to_datetime(fill["slot_date"]) >= pd.Timestamp(VALID_FROM)
fill_model = fit_fill(fill[~is_val])
val = fill[is_val]
pred = fill_model.predict(val[mf.FILL_FEATURES])
baseline = val["hist_fill_rate_4w"].fillna(fill.loc[~is_val, mf.FILL_LABEL].mean())
fill_metrics = {
    "val_mae": mean_absolute_error(val[mf.FILL_LABEL], pred),
    "val_mae_baseline_hist4w": mean_absolute_error(val[mf.FILL_LABEL], baseline),
    "val_mean_pred": float(pred.mean()), "val_mean_actual": float(val[mf.FILL_LABEL].mean()),
    "train_rows": int((~is_val).sum()), "val_rows": int(is_val.sum()),
}
print(fill_metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC **Price response learned by the model** (validation slots, average predicted fill and expected revenue per slot
# MAGIC at each multiple of the published tariff, guardrails applied). Fill must fall as the price rises.

# COMMAND ----------

curve = []
for mult in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
    for tier, g in val.groupby("tier"):
        lo, hi = pr.price_bounds(tier)
        price = float(np.clip(pr.TIER_PRICE_CHF[tier] * mult, lo, hi))
        p = fill_model.predict(mf.with_price(g, price)[mf.FILL_FEATURES])
        curve.append({"tier": tier, "multiplier": mult, "price_chf": price, "pred_fill": p.mean(),
                      "exp_revenue_per_slot": (p * g["courts_available"] * price).mean()})
curve = pd.DataFrame(curve)
display(curve.pivot(index="multiplier", columns="tier", values="pred_fill").round(3))
display(curve.pivot(index="multiplier", columns="tier", values="exp_revenue_per_slot").round(1))
mono = curve.sort_values("multiplier").groupby("tier")["pred_fill"].apply(lambda s: bool((s.diff().dropna() <= 1e-9).all()))
if not mono.all():
    raise RuntimeError(f"Fill model is not monotone in price for tiers {list(mono[~mono].index)}")
fill_metrics.update({f"pred_fill_x0.8_{t}": float(v) for t, v in
                     curve[curve["multiplier"] == 0.8].set_index("tier")["pred_fill"].items()})
fill_metrics.update({f"pred_fill_x1.2_{t}": float(v) for t, v in
                     curve[curve["multiplier"] == 1.2].set_index("tier")["pred_fill"].items()})

# COMMAND ----------

fill_final = fit_fill(fill)
fill_version = register(fill_final, fill[mf.FILL_FEATURES], "fill_probability_model", fill_metrics,
                        {**{k: str(v) for k, v in FILL_PARAMS.items()}, "n_replays": N_REPLAYS, "elasticity_k": 1.0},
                        fill_final.predict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver inputs for churn and cancellation

# COMMAND ----------

bookings = spark.sql(f"""
    SELECT booking_id, booker_uuid, start_ts, booked_at, lead_time_h, tier, party_size, payment_method, status,
           booking_date
    FROM {SILVER}.bookings
""").toPandas()
plays = spark.sql(f"""
    SELECT p.player_uuid, p.booking_id, b.booking_date
    FROM {SILVER}.booking_participants p JOIN {SILVER}.bookings b USING (booking_id)
    WHERE b.status = 'completed'
""").toPandas()
players = spark.sql(f"SELECT player_uuid, segment, skill_tier, joined_at FROM {SILVER}.players").toPandas()
weather = spark.sql(f"SELECT date, is_rainy FROM {SILVER}.weather").toPandas()
log(f"bookings {len(bookings):,}  completed participations {len(plays):,}  players {len(players):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Churn model (weekly snapshots of regulars)

# COMMAND ----------

dates = mf.churn_snapshot_dates(AS_OF, WINDOW_START)
churn = pd.concat([mf.churn_snapshot(plays, bookings, players, t) for t in dates], ignore_index=True)
log(f"churn snapshots built for {dates}")
save(churn[["player_uuid", "snapshot_date", "plays_next_30d"] + mf.CHURN_FEATURES + [mf.CHURN_LABEL]],
     "churn_training_snapshots", "Churn-model training rows: regulars at weekly snapshots, label = next 30 days")
print(churn.groupby("snapshot_date")[mf.CHURN_LABEL].agg(["size", "mean"]))

CLS_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.9, random_state=SEED, verbose=-1)


def fit_cls(df: pd.DataFrame, features: list[str], label: str) -> lgb.LGBMClassifier:
    m = lgb.LGBMClassifier(**CLS_PARAMS)
    m.fit(df[features], df[label])
    return m


def cls_metrics(y, p) -> dict:
    top = pd.Series(p).rank(ascending=False, method="first") <= max(1, int(len(p) * 0.1))
    return {"val_auc": roc_auc_score(y, p), "val_avg_precision": average_precision_score(y, p),
            "val_base_rate": float(np.mean(y)), "val_top_decile_rate": float(np.asarray(y)[top.to_numpy()].mean())}


last = max(dates)
churn_train, churn_val = churn[churn["snapshot_date"] < last], churn[churn["snapshot_date"] == last]
m = fit_cls(churn_train, mf.CHURN_FEATURES, mf.CHURN_LABEL)
churn_metrics = cls_metrics(churn_val[mf.CHURN_LABEL], m.predict_proba(churn_val[mf.CHURN_FEATURES])[:, 1])
churn_metrics.update(train_rows=len(churn_train), val_rows=len(churn_val))
print(churn_metrics)

churn_final = fit_cls(churn, mf.CHURN_FEATURES, mf.CHURN_LABEL)
churn_version = register(churn_final, churn[mf.CHURN_FEATURES], "player_churn_model", churn_metrics,
                         {**{k: str(v) for k, v in CLS_PARAMS.items()}, "snapshots": len(dates)},
                         lambda X: churn_final.predict_proba(X)[:, 1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cancellation risk model (late cancellation or no-show)

# COMMAND ----------

cancel = mf.cancel_rows(bookings, players, weather)
log("cancellation rows built")
cancel_hist = cancel[cancel["status"] != "upcoming"].reset_index(drop=True)
save(cancel_hist[["booking_id", "booker_uuid", "start_ts", "status"] + mf.CANCEL_FEATURES + [mf.CANCEL_LABEL]],
     "cancellation_training_bookings", "Cancellation-model training rows: past bookings still on the books 24 h before start")

is_val = cancel_hist["start_ts"] >= pd.Timestamp(VALID_FROM)
m = fit_cls(cancel_hist[~is_val], mf.CANCEL_FEATURES, mf.CANCEL_LABEL)
v = cancel_hist[is_val]
cancel_metrics = cls_metrics(v[mf.CANCEL_LABEL], m.predict_proba(v[mf.CANCEL_FEATURES])[:, 1])
cancel_metrics.update(train_rows=int((~is_val).sum()), val_rows=int(is_val.sum()))
print(cancel_metrics)

cancel_final = fit_cls(cancel_hist, mf.CANCEL_FEATURES, mf.CANCEL_LABEL)
cancel_version = register(cancel_final, cancel_hist[mf.CANCEL_FEATURES], "cancellation_risk_model", cancel_metrics,
                          {k: str(v) for k, v in CLS_PARAMS.items()},
                          lambda X: cancel_final.predict_proba(X)[:, 1])

# COMMAND ----------

summary = pd.DataFrame([
    {"model": "fill_probability_model", "version": fill_version, "metric": "val MAE (vs 4-week history)",
     "value": f"{fill_metrics['val_mae']:.3f} (baseline {fill_metrics['val_mae_baseline_hist4w']:.3f})"},
    {"model": "player_churn_model", "version": churn_version, "metric": "val AUC / top-decile churn rate",
     "value": f"{churn_metrics['val_auc']:.3f} / {churn_metrics['val_top_decile_rate']:.1%} "
              f"(base {churn_metrics['val_base_rate']:.1%})"},
    {"model": "cancellation_risk_model", "version": cancel_version, "metric": "val AUC / top-decile late rate",
     "value": f"{cancel_metrics['val_auc']:.3f} / {cancel_metrics['val_top_decile_rate']:.1%} "
              f"(base {cancel_metrics['val_base_rate']:.1%})"},
])
display(summary)
