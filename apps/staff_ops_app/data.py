"""Data-access layer for the staff app: live SQL warehouse or deterministic mock.

No Streamlit import here so the module can be tested and reused; caching lives in app.py.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Protocol

import pandas as pd

import agent_tools as at
import day_outlook
import domain as d
import mock_data

_IDENT = re.compile(r"^[A-Za-z0-9_\-]+$")


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    catalog: str = field(default_factory=lambda: os.getenv("B4L_CATALOG", "serverless_stable_yqq46w_catalog"))
    schema_prefix: str = field(default_factory=lambda: os.getenv("B4L_SCHEMA_PREFIX", "b4l"))
    warehouse_id: str = field(default_factory=lambda: os.getenv("DATABRICKS_WAREHOUSE_ID", ""))
    genie_space_title: str = field(
        default_factory=lambda: os.getenv("B4L_GENIE_SPACE_TITLE", "B4L Smart Badminton Center")
    )
    genie_space_id: str = field(default_factory=lambda: os.getenv("B4L_GENIE_SPACE_ID", ""))
    use_mock: bool = field(default_factory=lambda: _env_flag("B4L_USE_MOCK"))
    lakebase_endpoint: str = field(default_factory=lambda: os.getenv("B4L_LAKEBASE_ENDPOINT", ""))
    lakebase_database: str = field(default_factory=lambda: os.getenv("B4L_LAKEBASE_DATABASE", "databricks_postgres"))

    @property
    def ops_schema(self) -> str:
        """Postgres schema of the operational tables (src/b4l/lakebase.py schema_names)."""
        s = f"{self.schema_prefix}_ops"
        if not re.fullmatch(r"[a-z][a-z0-9_]*", s):
            raise ValueError(f"Invalid schema: {s!r}")
        return s

    @property
    def gold(self) -> str:
        for part in (self.catalog, f"{self.schema_prefix}_gold"):
            if not _IDENT.match(part):
                raise ValueError(f"Invalid identifier: {part!r}")
        return f"`{self.catalog}`.`{self.schema_prefix}_gold`"


class Backend(Protocol):
    name: str

    def as_of_date(self) -> date: ...

    def date_bounds(self) -> tuple[date, date]: ...

    def court_utilization(self, day: date, hall: str | None = None) -> pd.DataFrame: ...

    def daily_kpis(self, start: date, end: date) -> pd.DataFrame: ...

    def slot_features(self, day: date) -> pd.DataFrame: ...

    def at_risk_players(self, min_baseline: float = 2.0, limit: int = 200) -> pd.DataFrame: ...

    def expiring_chrono(self, within_days: int = 14) -> pd.DataFrame: ...

    def pricing_backtest(self) -> pd.DataFrame: ...

    def demand_curve_slots(self) -> pd.DataFrame: ...

    def demand_curve(self, slot_key: str) -> pd.DataFrame: ...

    def price_recommendations(self) -> pd.DataFrame: ...

    def day_context(self, start: date, n_days: int = 1) -> pd.DataFrame: ...

    def guardrail(self, slot_id: str, price: float) -> dict | None: ...

    def concierge(self, player_uuid: str, trigger: str) -> dict | None: ...

    def doubles_partners(self, player_uuid: str) -> pd.DataFrame: ...

    def open_waitlist(self, start: date, n_days: int = 7) -> pd.DataFrame: ...

    def organisers(self, limit: int = 300) -> pd.DataFrame: ...


CURVE_SLOT_COLUMNS = ["slot_key", "slot_start_ts", "tier", "demo_label", "base_price_chf", "recommended_price_chf"]


def _curve_slots(curves: pd.DataFrame) -> pd.DataFrame:
    rec = curves[curves["is_recommended"]].rename(columns={"candidate_price_chf": "recommended_price_chf"})
    return (rec.assign(_demo=rec["demo_label"].isna())
            .sort_values(["_demo", "slot_start_ts"])[CURVE_SLOT_COLUMNS].reset_index(drop=True))


AT_RISK_COLUMNS = [
    "player_uuid", "segment", "skill_tier", "preferred_lang", "avg_monthly_plays_baseline", "plays_prev_30d",
    "plays_last_30d",
    "play_change_pct", "days_since_last_play", "preferred_day_of_week", "preferred_hour", "preferred_tier",
    "pass_product",
]
CHRONO_COLUMNS = [
    "player_uuid", "segment", "preferred_lang", "pass_hours_remaining", "pass_expires_at", "days_to_pass_expiry",
    "plays_last_30d",
]


class MockBackend:
    name = "mock"

    def __init__(self, as_of: date | None = None):
        self._as_of = as_of or date.today()

    def as_of_date(self) -> date:
        return self._as_of

    def date_bounds(self) -> tuple[date, date]:
        return mock_data.window(self._as_of)

    def court_utilization(self, day: date, hall: str | None = None) -> pd.DataFrame:
        return mock_data.court_utilization_hourly(day, self._as_of, hall)

    def daily_kpis(self, start: date, end: date) -> pd.DataFrame:
        return mock_data.daily_kpis(start, end, self._as_of)

    def slot_features(self, day: date) -> pd.DataFrame:
        return mock_data.slot_features(day, self._as_of)

    def at_risk_players(self, min_baseline: float = 2.0, limit: int = 200) -> pd.DataFrame:
        df = mock_data.player_features(self._as_of)
        df = df[df["churn_risk_rule"] & (df["avg_monthly_plays_baseline"] >= min_baseline)]
        return (
            df.sort_values("avg_monthly_plays_baseline", ascending=False)[AT_RISK_COLUMNS]
            .head(limit)
            .reset_index(drop=True)
        )

    def expiring_chrono(self, within_days: int = 14) -> pd.DataFrame:
        df = mock_data.player_features(self._as_of)
        days = df["days_to_pass_expiry"]
        df = df[(df["pass_product"] == d.PASS_CHRONO) & days.between(0, within_days).fillna(False)]
        return df.sort_values(["days_to_pass_expiry", "pass_hours_remaining"], ascending=[True, False])[
            CHRONO_COLUMNS
        ].reset_index(drop=True)

    def pricing_backtest(self) -> pd.DataFrame:
        return mock_data.static_vs_dynamic_revenue(self._as_of)

    def demand_curve_slots(self) -> pd.DataFrame:
        return _curve_slots(mock_data.slot_demand_curves(self._as_of))

    def demand_curve(self, slot_key: str) -> pd.DataFrame:
        c = mock_data.slot_demand_curves(self._as_of)
        return c[c["slot_key"] == slot_key].sort_values("candidate_price_chf").reset_index(drop=True)

    def price_recommendations(self) -> pd.DataFrame:
        return mock_data.price_recommendations(self._as_of)

    def day_context(self, start: date, n_days: int = 1) -> pd.DataFrame:
        return day_outlook.mock_day_context(start, n_days, self._as_of)

    def _players(self) -> pd.DataFrame:
        return mock_data.player_features(self._as_of)

    def guardrail(self, slot_id: str, price: float) -> dict | None:
        key = slot_id[:15]
        try:
            f = mock_data.slot_features(date.fromisoformat(key[:10]), self._as_of)
        except ValueError:
            return None
        row = f[f["slot_key"] == key]
        return at.guardrail(row.iloc[0].to_dict(), price) if not row.empty else None

    def concierge(self, player_uuid: str, trigger: str) -> dict | None:
        p = self._players()
        row = p[p["player_uuid"] == player_uuid]
        if row.empty:
            return None
        wl = self.open_waitlist(self._as_of, d.FUTURE_DAYS)
        mine = wl[wl["player_uuid"] == player_uuid]
        return at.concierge_message(row.iloc[0].to_dict(), trigger, self._as_of,
                                    mine.iloc[0].to_dict() if not mine.empty else None)

    def doubles_partners(self, player_uuid: str) -> pd.DataFrame:
        return at.doubles_partners(self._players(), player_uuid)

    def open_waitlist(self, start: date, n_days: int = 7) -> pd.DataFrame:
        days = [start + timedelta(days=i) for i in range(max(1, min(int(n_days), 31)))]
        slots = pd.concat([mock_data.slot_features(x, self._as_of) for x in days], ignore_index=True)
        wl = at.mock_open_waitlist(self._players(), slots[slots["is_future"]])
        return wl.sort_values(["window_start_ts", "waitlist_id"]).reset_index(drop=True)

    def organisers(self, limit: int = 300) -> pd.DataFrame:
        p = self._players()
        p = p[p["preferred_hour"].notna() & (p["plays_last_30d"] > 0)]
        return (p.sort_values(["plays_last_30d", "player_uuid"], ascending=[False, True])[at.PLAYER_PICK_COLUMNS]
                .head(limit).reset_index(drop=True))


def _convert(series: pd.Series, type_name: str) -> pd.Series:
    t = type_name.upper()
    if t in ("INT", "LONG", "SHORT", "BYTE", "BIGINT", "SMALLINT", "TINYINT"):
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if t in ("DOUBLE", "FLOAT", "DECIMAL"):
        return pd.to_numeric(series, errors="coerce").astype(float)
    if t == "BOOLEAN":
        return series.map({"true": True, "false": False, True: True, False: False})
    if t == "DATE":
        return pd.to_datetime(series, errors="coerce").dt.date
    if t in ("TIMESTAMP", "TIMESTAMP_NTZ"):
        return pd.to_datetime(series, errors="coerce")
    if t == "ARRAY":
        return series.map(lambda v: json.loads(v) if isinstance(v, str) else v)
    return series


def statement_to_frame(resp: Any, fetch_chunk=None) -> pd.DataFrame:
    """Build a DataFrame from a StatementResponse (INLINE / JSON_ARRAY), converting types from the manifest."""
    manifest = resp.manifest
    columns = manifest.schema.columns if manifest and manifest.schema and manifest.schema.columns else []
    names = [c.name for c in columns]
    rows = list((resp.result.data_array or []) if resp.result else [])
    total_chunks = getattr(manifest, "total_chunk_count", None) or 1
    if fetch_chunk is not None and total_chunks > 1:
        for i in range(1, total_chunks):
            rows.extend(fetch_chunk(i).data_array or [])
    df = pd.DataFrame(rows, columns=names)
    for c in columns:
        type_name = c.type_name.value if hasattr(c.type_name, "value") else str(c.type_name or "")
        df[c.name] = _convert(df[c.name], type_name)
    return df


class SqlBackend:
    """Queries gold tables through the SQL warehouse with the Statement Execution API."""

    name = "live"

    def __init__(self, settings: Settings, client: Any = None):
        from databricks.sdk import WorkspaceClient

        self.settings = settings
        self.w = client or WorkspaceClient()
        self.gold = settings.gold
        self._as_of: date | None = None

    def query(self, statement: str, params: dict[str, tuple[Any, str]] | None = None) -> pd.DataFrame:
        from databricks.sdk.service import sql

        parameters = [
            sql.StatementParameterListItem(name=k, value=None if v is None else str(v), type=t)
            for k, (v, t) in (params or {}).items()
        ]
        resp = self.w.statement_execution.execute_statement(
            statement=statement,
            warehouse_id=self.settings.warehouse_id,
            parameters=parameters or None,
            wait_timeout="30s",
            disposition=sql.Disposition.INLINE,
            format=sql.Format.JSON_ARRAY,
        )
        deadline = time.time() + 120
        while resp.status.state in (sql.StatementState.PENDING, sql.StatementState.RUNNING):
            if time.time() > deadline:
                self.w.statement_execution.cancel_execution(resp.statement_id)
                raise TimeoutError("SQL statement timed out")
            time.sleep(1)
            resp = self.w.statement_execution.get_statement(resp.statement_id)
        if resp.status.state != sql.StatementState.SUCCEEDED:
            err = resp.status.error.message if resp.status.error else resp.status.state
            raise RuntimeError(f"SQL failed: {err}")
        return statement_to_frame(
            resp, lambda i: self.w.statement_execution.get_statement_result_chunk_n(resp.statement_id, i)
        )

    def as_of_date(self) -> date:
        if self._as_of is None:
            df = self.query(f"SELECT MAX(as_of_date) AS as_of_date FROM {self.gold}.player_features")
            value = df["as_of_date"].iloc[0] if not df.empty else None
            self._as_of = value if isinstance(value, date) else date.today()
        return self._as_of

    def date_bounds(self) -> tuple[date, date]:
        df = self.query(f"SELECT MIN(date) AS lo, MAX(date) AS hi FROM {self.gold}.daily_kpis")
        lo, hi = (df["lo"].iloc[0], df["hi"].iloc[0]) if not df.empty else (None, None)
        as_of = self.as_of_date()
        return (
            lo if isinstance(lo, date) else as_of - timedelta(days=d.HISTORY_DAYS),
            hi if isinstance(hi, date) else as_of + timedelta(days=d.FUTURE_DAYS - 1),
        )

    def court_utilization(self, day: date, hall: str | None = None) -> pd.DataFrame:
        return self.query(
            f"""SELECT * FROM {self.gold}.court_utilization_hourly
            WHERE date = :day AND (:hall IS NULL OR hall_zone = :hall)
            ORDER BY court_id, hour""",
            {"day": (day.isoformat(), "DATE"), "hall": (hall, "STRING")},
        )

    def daily_kpis(self, start: date, end: date) -> pd.DataFrame:
        return self.query(
            f"SELECT * FROM {self.gold}.daily_kpis WHERE date BETWEEN :start AND :end ORDER BY date",
            {"start": (start.isoformat(), "DATE"), "end": (end.isoformat(), "DATE")},
        )

    def slot_features(self, day: date) -> pd.DataFrame:
        return self.query(
            f"SELECT * FROM {self.gold}.slot_features WHERE slot_date = :day ORDER BY slot_start_ts",
            {"day": (day.isoformat(), "DATE")},
        )

    def at_risk_players(self, min_baseline: float = 2.0, limit: int = 200) -> pd.DataFrame:
        return self.query(
            f"""SELECT {", ".join(AT_RISK_COLUMNS)} FROM {self.gold}.player_features
            WHERE churn_risk_rule AND avg_monthly_plays_baseline >= :min_baseline
            ORDER BY avg_monthly_plays_baseline DESC LIMIT {int(limit)}""",
            {"min_baseline": (min_baseline, "DOUBLE")},
        )

    def expiring_chrono(self, within_days: int = 14) -> pd.DataFrame:
        return self.query(
            f"""SELECT {", ".join(CHRONO_COLUMNS)} FROM {self.gold}.player_features
            WHERE pass_product = 'Pass_Chrono' AND days_to_pass_expiry BETWEEN 0 AND :days
            ORDER BY days_to_pass_expiry, pass_hours_remaining DESC""",
            {"days": (within_days, "INT")},
        )

    def pricing_backtest(self) -> pd.DataFrame:
        return self.query(f"SELECT * FROM {self.gold}.static_vs_dynamic_revenue ORDER BY elasticity_scenario, "
                          "week_start, tier")

    def demand_curve_slots(self) -> pd.DataFrame:
        return self.query(
            f"""SELECT slot_key, slot_start_ts, tier, demo_label, base_price_chf,
                   candidate_price_chf AS recommended_price_chf
            FROM {self.gold}.slot_demand_curves WHERE is_recommended
            ORDER BY demo_label IS NULL, slot_start_ts"""
        )

    def demand_curve(self, slot_key: str) -> pd.DataFrame:
        return self.query(
            f"SELECT * FROM {self.gold}.slot_demand_curves WHERE slot_key = :k ORDER BY candidate_price_chf",
            {"k": (slot_key, "STRING")},
        )

    def price_recommendations(self) -> pd.DataFrame:
        return self.query(
            f"SELECT {', '.join(mock_data.PRICE_REC_COLUMNS)} FROM {self.gold}.slot_price_recommendations "
            "ORDER BY slot_start_ts"
        )

    def day_context(self, start: date, n_days: int = 1) -> pd.DataFrame:
        return self.query(
            f"SELECT {', '.join(day_outlook.COLUMNS)} FROM {self.gold}.get_day_context(:start, :n) ORDER BY date",
            {"start": (start.isoformat(), "DATE"), "n": (int(n_days), "INT")},
        )

    def _fn(self, name: str) -> str:
        return f"{self.gold}.{name}"

    @staticmethod
    def _first(df: pd.DataFrame) -> dict | None:
        if df.empty:
            return None
        return {k: (None if not isinstance(v, (list, tuple)) and pd.isna(v) else v)
                for k, v in df.iloc[0].to_dict().items()}

    def guardrail(self, slot_id: str, price: float) -> dict | None:
        return self._first(self.query(
            f"SELECT {', '.join(at.GUARDRAIL_COLUMNS)} FROM {self._fn('apply_dynamic_price_with_guardrails')}(:s, :p)",
            {"s": (slot_id, "STRING"), "p": (float(price), "DOUBLE")}))

    def concierge(self, player_uuid: str, trigger: str) -> dict | None:
        return self._first(self.query(
            f"SELECT {', '.join(at.CONCIERGE_COLUMNS)} FROM {self._fn('trigger_player_concierge')}(:u, :t)",
            {"u": (player_uuid, "STRING"), "t": (trigger, "STRING")}))

    def doubles_partners(self, player_uuid: str) -> pd.DataFrame:
        return self.query(
            f"SELECT {', '.join(at.PARTNER_COLUMNS)} FROM {self._fn('find_doubles_partners')}(:u)",
            {"u": (player_uuid, "STRING")})

    def open_waitlist(self, start: date, n_days: int = 7) -> pd.DataFrame:
        return self.query(
            f"SELECT {', '.join(at.WAITLIST_COLUMNS)} FROM {self._fn('get_open_waitlist')}(:start, :n)",
            {"start": (start.isoformat(), "DATE"), "n": (int(n_days), "INT")})

    def organisers(self, limit: int = 300) -> pd.DataFrame:
        return self.query(
            f"""SELECT {', '.join(at.PLAYER_PICK_COLUMNS)} FROM {self.gold}.player_features
            WHERE preferred_hour IS NOT NULL AND plays_last_30d > 0
            ORDER BY plays_last_30d DESC, player_uuid LIMIT {int(limit)}""")


def databricks_client() -> tuple[Any | None, str]:
    """A WorkspaceClient if the SDK is installed and credentials resolve, else (None, reason)."""
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        return None, "databricks-sdk is not installed"
    try:
        return WorkspaceClient(), "ok"
    except Exception as e:  # noqa: BLE001 - any auth/config failure means offline mode
        return None, f"no Databricks auth ({type(e).__name__})"


def make_backend(settings: Settings | None = None) -> tuple[Backend, str]:
    """Mock if B4L_USE_MOCK=true, the SDK or auth is missing, or no warehouse is configured."""
    settings = settings or Settings()
    if settings.use_mock:
        return MockBackend(), "B4L_USE_MOCK=true"
    if not settings.warehouse_id:
        return MockBackend(), "DATABRICKS_WAREHOUSE_ID is not set"
    client, reason = databricks_client()
    if client is None:
        return MockBackend(), reason
    return SqlBackend(settings, client), f"SQL warehouse {settings.warehouse_id}"


def occupancy(frame: pd.DataFrame) -> float:
    """Taux d'occupation as a ratio of sums (never an average of rates)."""
    avail = frame["available_minutes"].sum()
    return float((frame["booked_minutes"].sum() + frame["blocked_minutes"].sum()) / avail) if avail else 0.0


def pricing_headline(backtest: pd.DataFrame) -> pd.DataFrame:
    """Per scenario: weekly revenue change, off-peak (Day, Lunch) occupancy change in points, share re-priced.

    Same arithmetic as src/b4l/backtest.py headline (checked by tests/test_staff_app_contract.py).
    """
    weeks = backtest["week_start"].nunique()
    rows = []
    for scenario, g in backtest.groupby("elasticity_scenario"):
        off = g[g["tier"].isin([d.DAY, d.LUNCH])]
        one = g.drop_duplicates(["week_start", "tier"])
        rows.append({
            "elasticity_scenario": scenario,
            "static_revenue_per_week_chf": g["static_revenue_chf"].sum() / weeks,
            "dynamic_revenue_per_week_chf": g["dynamic_revenue_chf"].sum() / weeks,
            "revenue_change_per_week_chf": g["revenue_change_chf"].sum() / weeks,
            "revenue_change_pct": g["revenue_change_chf"].sum() / g["static_revenue_chf"].sum(),
            "offpeak_occupancy_change_pts": 100 * (off["dynamic_booked_hours"].sum() - off["static_booked_hours"].sum())
                                             / off["available_hours"].sum(),
            "share_slots_repriced": one["slots_repriced"].sum() / one["slots"].sum(),
        })
    return pd.DataFrame(rows)


def underfilled_slots(slots: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """Add booked_share and a flag for slots booked below `threshold` x their 4-week fill rate."""
    out = slots.copy()
    avail = out["courts_available"].astype(float).where(out["courts_available"] > 0)
    out["booked_share"] = (out["booked_courts_now"].astype(float) / avail).fillna(0.0).round(3)
    hist = out["hist_fill_rate_4w"].astype(float).fillna(0.0)
    out["gap_vs_hist"] = (hist - out["booked_share"]).round(3)
    out["underfilled"] = out["booked_share"] < threshold * hist
    return out
