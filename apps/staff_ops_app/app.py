"""B4L Reception & Ops Cockpit: reads the gold tables; writes price decisions and the action log to Lakebase only."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import domain as d
import guide
from data import Settings, make_backend, occupancy, pricing_headline, underfilled_slots
from genie_client import SAMPLE_QUESTIONS, GenieClient
from ops_store import DECIDABLE, LOG_LIMIT, DecisionError, make_ops_store, slot_time

st.set_page_config(page_title="B4L Reception & Ops Cockpit", page_icon="🏸", layout="wide")

SETTINGS = Settings()


@st.cache_resource(show_spinner=False)
def get_backend():
    return make_backend(SETTINGS)


@st.cache_data(ttl=300, show_spinner=False)
def load(_backend, backend_name: str, method: str, *args):
    return getattr(_backend, method)(*args)


@st.cache_resource(show_spinner=False)
def get_genie(_backend, backend_name: str) -> GenieClient:
    return GenieClient(_backend.w, SETTINGS.genie_space_title, SETTINGS.genie_space_id)


@st.cache_resource(show_spinner=False)
def get_ops_store(_backend, backend_name: str):
    return make_ops_store(_backend, SETTINGS.lakebase_endpoint, SETTINGS.lakebase_database, SETTINGS.ops_schema)


def current_actor() -> str:
    """The signed-in user's email (set by the Databricks Apps proxy), else a local placeholder."""
    return st.context.headers.get("X-Forwarded-Email") or "local-user (no sign-in header)"


def fetch(method: str, *args) -> pd.DataFrame | None:
    try:
        return load(backend, backend.name, method, *args)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load data ({method}): {e}")
        return None


def chf(x: float) -> str:
    return f"CHF {x:,.0f}".replace(",", "'")


def how_to_read(text: str) -> None:
    with st.expander(guide.EXPANDER_LABEL):
        st.markdown(text)


backend, backend_reason = get_backend()
try:
    AS_OF = backend.as_of_date()
    LO, HI = backend.date_bounds()
except Exception as e:  # noqa: BLE001
    st.error(f"Cannot reach the gold tables: {e}")
    st.stop()
ops, ops_reason = get_ops_store(backend, backend.name)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🏸 B4L Fondation")
    st.caption("Ch. du Viaduc 12, 1008 Prilly · 16 courts")
    if backend.name == "live":
        st.success(f"Data source: **live** · {backend_reason}")
        st.caption(f"`{SETTINGS.catalog}.{SETTINGS.schema_prefix}_gold`")
    else:
        st.warning(f"Data source: **mock** · {backend_reason}")
    st.metric("As-of date (snapshot)", AS_OF.strftime("%a %d %b %Y"))
    st.caption(
        f"History {LO:%d %b} → {AS_OF - timedelta(days=1):%d %b}; "
        f"bookings so far until {HI:%d %b}."
    )
    stale_days = (date.today() - AS_OF).days
    if stale_days > 0:
        st.warning(
            f"Snapshot is {stale_days} day{'s' if stale_days > 1 else ''} old: days from {AS_OF:%d %b} still "
            "count as upcoming. Re-run `b4l_demo_job` before a demo."
        )
    with st.expander("ℹ️ About the snapshot"):
        st.markdown(guide.SNAPSHOT)
    if st.button("↻ Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    if ops.name == "live":
        st.caption(f"Price decisions: **Lakebase** `{ops_reason}` · `{SETTINGS.ops_schema}`")
    else:
        st.caption(f"Price decisions: **in-memory mock** ({ops_reason}); lost when the app restarts.")

st.title("B4L Fondation · Smart Badminton Center")
st.caption("Reception & Ops Cockpit · Cockpit accueil et exploitation")

tab_heat, tab_slots, tab_pricing, tab_players, tab_genie = st.tabs(
    ["🗓️ Court heatmap", "💶 Slots & prices", "📈 Pricing impact", "👥 Players", "💬 Ask Genie"]
)

# ---------------------------------------------------------------- heatmap
with tab_heat:
    how_to_read(guide.HEATMAP)
    c1, c2, _ = st.columns([1, 1, 2])
    day = c1.date_input("Day (jour)", value=min(max(AS_OF, LO), HI), min_value=LO, max_value=HI, key="heat_day")
    hall_choice = c2.selectbox("Hall (salle)", ["All halls", *d.HALLS], key="heat_hall")
    hall = None if hall_choice == "All halls" else hall_choice
    dtype = d.day_type(day)

    kpis = fetch("daily_kpis", day - timedelta(days=7), day)
    if dtype == "closed":
        st.info(f"The centre is closed on {day:%A %d %B %Y}.")
    else:
        future = day >= AS_OF
        if future:
            st.caption("📅 Upcoming day: figures show the bookings made so far (réservations à ce jour).")
        if kpis is not None and not kpis.empty:
            today_k = kpis[kpis["date"] == day]
            prev_k = kpis[kpis["date"] == day - timedelta(days=7)]
            if not today_k.empty:
                k = today_k.iloc[0]
                p = prev_k.iloc[0] if not prev_k.empty and not future else None

                def delta(col, fmt="{:+.0f}"):
                    return None if p is None else fmt.format(k[col] - p[col])

                m = st.columns(5)
                m[0].metric(
                    "Occupancy (taux d'occupation)", f"{k['occupancy_rate']:.0%}",
                    None if p is None else f"{(k['occupancy_rate'] - p['occupancy_rate']) * 100:+.1f} pts",
                )
                m[1].metric("Revenue (chiffre d'affaires)", chf(k["revenue_chf"]), delta("revenue_chf", "{:+,.0f}"))
                m[2].metric("Bookings (réservations)", int(k["bookings"]), delta("bookings"))
                m[3].metric("Check-in persons (fréquentation)", int(k["check_in_persons"]), delta("check_in_persons"))
                m[4].metric("Waitlist (liste d'attente)", int(k["waitlist_requests"]), delta("waitlist_requests"))
                if p is not None:
                    st.caption("Deltas vs the same weekday one week earlier.")

        hourly = fetch("court_utilization", day, hall)
        if hourly is not None and not hourly.empty:
            hours = d.heatmap_hours(dtype)
            courts = sorted(hourly["court_id"].unique())
            idx = pd.MultiIndex.from_product([courts, hours], names=["court_id", "hour"])
            grid = hourly.set_index(["court_id", "hour"]).reindex(idx)
            occ = grid["occupancy_rate"].unstack("hour") * 100
            tier = grid["tier"].unstack("hour")
            booked = grid["booked_minutes"].unstack("hour")
            blocked = grid["blocked_minutes"].unstack("hour")
            avail = grid["available_minutes"].unstack("hour")
            revenue = grid["revenue_chf"].unstack("hour")
            price = tier.apply(lambda col: col.map(lambda t: d.TIER_PRICE_CHF.get(t)))
            label = tier.apply(lambda col: col.map(lambda t: d.tier_label(t) if isinstance(t, str) else ""))
            custom = np.dstack([f.to_numpy(dtype=object) for f in (label, price, booked, blocked, avail, revenue)])
            text = blocked.apply(lambda col: col.map(lambda b: "⛔" if b and b > 0 else ""))
            x_labels = [f"{h:02d}:00" for h in hours]
            y_labels = [f"{c} {'★' if c in d.SHOW_COURTS else ''}".strip() for c in courts]
            fig = go.Figure(
                go.Heatmap(
                    z=occ.to_numpy(), x=x_labels, y=y_labels, customdata=custom, text=text.to_numpy(),
                    texttemplate="%{text}", zmin=0, zmax=100, colorscale="Blues",
                    colorbar={"title": "Occupancy %"}, xgap=2, ygap=2,
                    hovertemplate=(
                        "<b>%{y} · %{x}</b><br>Tier: %{customdata[0]}<br>Base price: CHF %{customdata[1]:.0f}"
                        "<br>Occupancy: %{z:.0f}%<br>Booked: %{customdata[2]:.0f} min · "
                        "Blocked: %{customdata[3]:.0f} min / %{customdata[4]:.0f}"
                        "<br>Revenue: CHF %{customdata[5]:.0f}<extra></extra>"
                    ),
                )
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=560, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                              title=f"{day:%A %d %B %Y} · {hall_choice} · rentable hours "
                                    f"{d.rentable_window(dtype)[0]:%H:%M}–{d.rentable_window(dtype)[1]:%H:%M}")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "⛔ = court block (Bad'matin, private lesson, club BLA or school) · ★ = Prime Show Court · "
                f"Hall occupancy {occupancy(hourly):.0%}. Hover a cell for tier and base price."
            )
            with st.expander("Tariff tiers (tarifs)"):
                st.dataframe(
                    pd.DataFrame(
                        [{"Tier": d.tier_label(t), "Base price CHF / court / hour": d.TIER_PRICE_CHF[t]} for t in d.TIERS]
                    ),
                    hide_index=True,
                )
        elif hourly is not None:
            st.info("No hourly data for this day.")

# ---------------------------------------------------------------- slots & prices
with tab_slots:
    how_to_read(guide.SLOTS)
    first_future = max(AS_OF, LO)
    c1, c2, _ = st.columns([1, 1, 2])
    slot_day = c1.date_input(
        "Upcoming day", value=first_future, min_value=first_future, max_value=HI, key="slot_day"
    )
    threshold = c2.slider(
        "Under-filled if booked share < x · 4-week fill rate", 0.3, 1.0, 0.6, 0.05, key="slot_threshold"
    )
    slots = fetch("slot_features", slot_day)
    if d.day_type(slot_day) == "closed":
        st.info("Closed day.")
    elif slots is not None and not slots.empty:
        s = underfilled_slots(slots, threshold)
        s["start"] = pd.to_datetime(s["slot_start_ts"]).dt.strftime("%H:%M")
        s["tier_label"] = s["tier"].map(d.tier_label)
        n_under = int(s["underfilled"].sum())
        m = st.columns(4)
        m[0].metric("Slots (créneaux)", len(s))
        m[1].metric("Under-filled candidates", n_under)
        m[2].metric("Courts booked so far", int(s["booked_courts_now"].sum()))
        m[3].metric("Waitlist requests", int(s["waitlist_requests"].fillna(0).sum()))

        fig = go.Figure()
        fig.add_bar(
            x=s["start"], y=s["booked_share"] * 100, name="Booked now (%)",
            marker_color=["#e4572e" if u else "#4c78a8" for u in s["underfilled"]],
            customdata=s[["tier_label", "base_price_chf", "booked_courts_now", "courts_available"]],
            hovertemplate="%{x} · %{customdata[0]} · CHF %{customdata[1]:.0f}<br>"
                          "%{customdata[2]} / %{customdata[3]} courts booked<extra></extra>",
        )
        fig.add_scatter(
            x=s["start"], y=s["hist_fill_rate_4w"] * 100, name="Fill rate, last 4 weeks (%)",
            mode="lines+markers", line={"color": "#333", "dash": "dot"},
        )
        fig.update_layout(
            height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis_title="%", yaxis_range=[0, 105],
            legend={"orientation": "h", "y": 1.12}, title=f"{slot_day:%A %d %B %Y} · red = under-filled",
        )
        st.plotly_chart(fig, use_container_width=True)

        table = s[[
            "start", "tier_label", "base_price_chf", "courts_available", "booked_courts_now", "booked_share",
            "hist_fill_rate_4w", "gap_vs_hist", "waitlist_requests", "underfilled",
        ]]
        try:
            styled = table.style.apply(
                lambda row: ["background-color: #fde2dc" if row["underfilled"] else "" for _ in row], axis=1
            )
        except (AttributeError, ImportError):
            styled = table
        st.dataframe(
            styled, hide_index=True, use_container_width=True,
            column_config={
                "start": "Start",
                "tier_label": "Tier (tarif)",
                "base_price_chf": st.column_config.NumberColumn("Base price", format="CHF %.0f"),
                "courts_available": "Courts available",
                "booked_courts_now": "Booked now",
                "booked_share": st.column_config.ProgressColumn("Booked share", min_value=0, max_value=1),
                "hist_fill_rate_4w": st.column_config.ProgressColumn("Fill rate 4w", min_value=0, max_value=1),
                "gap_vs_hist": st.column_config.NumberColumn("Gap vs history", format="%.2f"),
                "waitlist_requests": "Waitlist",
                "underfilled": st.column_config.CheckboxColumn("Dynamic-pricing candidate"),
            },
        )
        st.caption(
            "Under-filled slots are candidates for demand-based pricing (CHF 18–42, max ±40% from the tier). "
            "Slots further ahead naturally have fewer bookings so far; pass holders are never re-priced."
        )
    elif slots is not None:
        st.info("No slot features for this day.")

    st.subheader("Price decisions · one court at a time")
    flash = st.session_state.pop("decision_flash", None)
    if flash:
        st.success(flash)
    try:
        summary = ops.slot_summary(slot_day)
    except Exception as e:  # noqa: BLE001
        st.error(f"Cannot reach Lakebase: {e}")
        summary = None
    if summary is not None and summary.empty:
        st.info("No AI prices for this day yet: recommendations cover the next 7 days (run `b4l_demo_job`).")
    elif summary is not None:
        def summary_label(i: int) -> str:
            r = summary.iloc[i]
            return (f"{slot_time(r['slot_key'])[-5:]} · {r['tier']} · CHF {r['base_price_chf']:.0f} → "
                    f"{r['recommended_price_chf']:.0f} · {r['proposed']} open, "
                    f"{r['approved'] + r['overridden']} decided")

        changed = summary.index[summary["recommended_price_chf"] != summary["base_price_chf"]]
        pick = st.selectbox("Start time (AI price shown after the arrow)", range(len(summary)),
                            index=int(changed[0]) if len(changed) else 0, format_func=summary_label,
                            key=f"decision_slot_{slot_day}")
        slot = summary.iloc[pick]
        st.caption(f"Why: {slot['reason']}")
        courts = ops.court_slots(slot["slot_key"])
        st.dataframe(
            courts[["court_id", "hall_zone", "status", "dynamic_price_chf", "recommended_price_chf", "decided_by",
                    "decided_at", "decision_reason"]],
            hide_index=True, use_container_width=True,
            column_config={
                "court_id": "Court", "hall_zone": "Hall", "status": "Status",
                "dynamic_price_chf": st.column_config.NumberColumn("Live price", format="CHF %.0f"),
                "recommended_price_chf": st.column_config.NumberColumn("AI price", format="CHF %.0f"),
                "decided_by": "Decided by",
                "decided_at": st.column_config.DatetimeColumn("Decided at", format="DD.MM HH:mm"),
                "decision_reason": "Reason",
            },
        )
        open_courts = courts[courts["status"].isin(DECIDABLE)]
        if open_courts.empty:
            st.info("No open court at this time: booked and blocked courts keep their price.")
        else:
            court = st.selectbox(
                "Court", open_courts["slot_id"].tolist(), key=f"decision_court_{slot['slot_key']}",
                format_func=lambda sid: (lambda r: f"{r['court_id']} · {r['status']} · live CHF "
                                                   f"{r['dynamic_price_chf']:.0f}")(
                    open_courts.set_index("slot_id").loc[sid]),
            )
            row = open_courts.set_index("slot_id").loc[court]
            lo, hi = d.price_bounds(row["tier"])
            actor = current_actor()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Approve AI price** · CHF {row['recommended_price_chf']:.0f}")
                if st.button(f"✅ Approve CHF {row['recommended_price_chf']:.0f} for {row['court_id']}",
                             key="approve_btn"):
                    try:
                        r = ops.approve(court, actor)
                        st.session_state["decision_flash"] = (
                            f"Approved CHF {r['dynamic_price_chf']:.0f} on {r['court_id']} at "
                            f"{slot_time(r['slot_key'])}.")
                        st.rerun()
                    except DecisionError as e:
                        st.error(str(e))
            with c2:
                with st.form("override_form", clear_on_submit=True):
                    st.markdown(f"**Manual override** · CHF {lo:.0f}–{hi:.0f} for {d.tier_label(row['tier'])}")
                    price = st.number_input("Price (CHF)", min_value=lo, max_value=hi, step=1.0,
                                            value=float(min(max(row["recommended_price_chf"], lo), hi)))
                    reason = st.text_input("Reason (required)", placeholder="e.g. Club tournament warm-up")
                    if st.form_submit_button(f"✏️ Override {row['court_id']}"):
                        try:
                            r = ops.override(court, price, reason, actor)
                            st.session_state["decision_flash"] = (
                                f"Override CHF {r['dynamic_price_chf']:.0f} on {r['court_id']} at "
                                f"{slot_time(r['slot_key'])}.")
                            st.rerun()
                        except DecisionError as e:
                            st.error(str(e))
            st.caption(f"Signed in as {actor}. Every decision is logged below.")

    st.subheader(f"Action log · latest {LOG_LIMIT}")
    try:
        log = ops.action_log(LOG_LIMIT)
    except Exception as e:  # noqa: BLE001
        st.error(f"Cannot read the action log: {e}")
        log = None
    if log is not None and log.empty:
        st.caption("No decisions yet.")
    elif log is not None:
        st.dataframe(
            log.assign(slot_key=log["slot_key"].map(slot_time)), hide_index=True, use_container_width=True,
            column_config={
                "ts": st.column_config.DatetimeColumn("When (UTC)", format="DD.MM HH:mm:ss"),
                "actor": "Who", "action_type": "Action", "status": "Status", "court_id": "Court",
                "slot_key": "Slot",
                "old_price_chf": st.column_config.NumberColumn("Old", format="CHF %.0f"),
                "new_price_chf": st.column_config.NumberColumn("New", format="CHF %.0f"),
                "recommended_price_chf": st.column_config.NumberColumn("AI price", format="CHF %.0f"),
                "reason": "Reason",
            },
        )

# ---------------------------------------------------------------- pricing impact
SCENARIOS = {"low": "Low sensitivity", "mid": "Mid sensitivity", "high": "High sensitivity"}

with tab_pricing:
    how_to_read(guide.PRICING)
    st.info(
        "Simulated backtest: the result depends on how price-sensitive players are, which is an assumption. "
        "Read it as a range and confirm it with a pilot before changing the tariff."
    )
    bt = fetch("pricing_backtest")
    if bt is not None and bt.empty:
        st.info("No backtest yet: run the `backtest` task of `b4l_demo_job`.")
    elif bt is not None:
        head = pricing_headline(bt).set_index("elasticity_scenario")
        order = [s for s in SCENARIOS if s in head.index]
        rev = head.loc[order, "revenue_change_per_week_chf"]
        pct = head.loc[order, "revenue_change_pct"]
        occ = head.loc[order, "offpeak_occupancy_change_pts"]
        m = st.columns(3)
        m[0].metric("Revenue change per week", f"{chf(rev.min())} to {chf(rev.max())}",
                    f"{pct.min():+.1%} to {pct.max():+.1%}", delta_color="off")
        m[1].metric("Off-peak occupancy (Day + Lunch)", f"{occ.min():+.1f} to {occ.max():+.1f} pts", delta_color="off")
        m[2].metric("Slots re-priced", f"{head['share_slots_repriced'].iloc[0]:.0%}")
        st.caption(
            f"Range over low, mid and high price sensitivity · 24 weeks of history · static revenue "
            f"{chf(head['static_revenue_per_week_chf'].iloc[0])} per week."
        )

        scenario = st.radio("Price sensitivity (sensibilité au prix)", order, index=order.index("mid") if "mid" in order else 0,
                            format_func=SCENARIOS.get, horizontal=True, key="pricing_scenario")
        g = bt[bt["elasticity_scenario"] == scenario]
        by_tier = g.groupby("tier")[["static_revenue_chf", "dynamic_revenue_chf"]].sum().reindex(d.TIERS)
        change = by_tier["dynamic_revenue_chf"] - by_tier["static_revenue_chf"]
        labels = [d.tier_label(t) for t in by_tier.index]
        fig = go.Figure()
        fig.add_bar(x=labels, y=by_tier["static_revenue_chf"], name="Tariff (static)", marker_color="#9aa5b1")
        fig.add_bar(x=labels, y=by_tier["dynamic_revenue_chf"], name="Dynamic prices", marker_color="#4c78a8",
                    text=[f"{c:+,.0f}".replace(",", "'") for c in change], textposition="outside")
        fig.update_layout(barmode="group", height=360, margin={"l": 10, "r": 10, "t": 40, "b": 10},
                          yaxis_title="CHF over 24 weeks", legend={"orientation": "h", "y": 1.12},
                          title=f"Revenue by tier · {SCENARIOS[scenario].lower()} · labels = change in CHF")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        weeks = sorted(g["week_start"].unique())
        x = [f"{w:%d %b}" for w in weeks]
        for col, field, title in ((c1, "static_occupancy", "Occupancy at the tariff"),
                                  (c2, "dynamic_occupancy", "Occupancy with dynamic prices")):
            z = g.pivot(index="tier", columns="week_start", values=field).reindex(index=list(d.TIERS), columns=weeks)
            hm = go.Figure(go.Heatmap(z=z.to_numpy() * 100, x=x, y=labels, zmin=0, zmax=100, colorscale="Blues",
                                      colorbar={"title": "%"},
                                      hovertemplate="%{y} · week of %{x}<br>Occupancy %{z:.0f}%<extra></extra>"))
            hm.update_layout(height=300, margin={"l": 10, "r": 10, "t": 40, "b": 10}, title=title)
            col.plotly_chart(hm, use_container_width=True)

    st.subheader("One slot: why this price?")
    curve_slots = fetch("demand_curve_slots")
    if curve_slots is not None and not curve_slots.empty:
        def slot_label(i: int) -> str:
            r = curve_slots.iloc[i]
            name = f"{r['demo_label']} · " if isinstance(r["demo_label"], str) else ""
            return f"{name}{pd.Timestamp(r['slot_start_ts']):%a %d %b %H:%M} · {r['tier']} · " \
                   f"CHF {r['base_price_chf']:.0f} → {r['recommended_price_chf']:.0f}"

        pick = st.selectbox("Upcoming slot", range(len(curve_slots)), format_func=slot_label, key="curve_slot")
        curve = fetch("demand_curve", curve_slots.iloc[pick]["slot_key"])
        if curve is not None and not curve.empty:
            tariff = curve[curve["is_tariff"]]
            rec = curve[curve["is_recommended"]]
            fig = go.Figure()
            fig.add_bar(x=curve["candidate_price_chf"], y=curve["expected_courts_booked"], name="Expected courts booked",
                        marker_color="#c9d6e3", yaxis="y2", hovertemplate="CHF %{x:.0f}: %{y:.1f} courts<extra></extra>")
            fig.add_scatter(x=curve["candidate_price_chf"], y=curve["expected_revenue_chf"], name="Expected revenue",
                            mode="lines+markers", line={"color": "#4c78a8"},
                            hovertemplate="CHF %{x:.0f}: CHF %{y:.0f} expected<extra></extra>")
            for frame, name, symbol, color in ((tariff, "Tariff", "diamond", "#333"),
                                               (rec, "Recommended", "star", "#e4572e")):
                fig.add_scatter(x=frame["candidate_price_chf"], y=frame["expected_revenue_chf"], name=name,
                                mode="markers", marker={"symbol": symbol, "size": 16, "color": color})
            fig.update_layout(
                height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10}, xaxis_title="Price per court per hour (CHF)",
                yaxis={"title": "Expected slot revenue (CHF)"},
                yaxis2={"title": "Courts", "overlaying": "y", "side": "right", "showgrid": False, "rangemode": "tozero"},
                legend={"orientation": "h", "y": 1.12},
            )
            st.plotly_chart(fig, use_container_width=True)
            r0 = curve.iloc[0]
            if not tariff.empty and not rec.empty:
                t, r = tariff.iloc[0], rec.iloc[0]
                text = (f"{int(r0['booked_courts_now'])} of {int(r0['courts_available'])} courts already booked. "
                        f"At the tariff CHF {t['candidate_price_chf']:.0f}: {t['expected_courts_booked']:.1f} courts, "
                        f"{chf(t['expected_revenue_chf'])}. ")
                if r["is_tariff"]:
                    top = curve.loc[curve["expected_new_revenue_chf"].idxmax()]
                    gain = top["expected_new_revenue_chf"] / max(t["expected_new_revenue_chf"], 1.0) - 1
                    text += (f"The tariff is kept: the best price, CHF {top['candidate_price_chf']:.0f}, adds only "
                             f"{gain:.0%} on the courts still to sell, or the slot is protected by the pricing rules "
                             "(see How to read this).")
                else:
                    text += (f"At CHF {r['candidate_price_chf']:.0f}: {r['expected_courts_booked']:.1f} courts, "
                             f"{chf(r['expected_revenue_chf'])}.")
                st.caption(text)

    st.subheader("Upcoming price changes")
    recs = fetch("price_recommendations")
    if recs is not None and not recs.empty:
        pc = recs["price_change_pct"].astype(float) * 100
        hist = go.Figure(go.Histogram(x=pc, xbins={"start": -42.5, "end": 42.5, "size": 5}, marker_color="#4c78a8",
                                      hovertemplate="%{x}%: %{y} slots<extra></extra>"))
        for edge in (-100 * d.MAX_MOVE_PCT, 100 * d.MAX_MOVE_PCT):
            hist.add_vline(x=edge, line={"color": "#e4572e", "dash": "dash"}, annotation_text="guardrail")
        hist.update_layout(height=300, margin={"l": 10, "r": 10, "t": 30, "b": 10}, bargap=0.05,
                           xaxis_title="Recommended vs tariff (%)", yaxis_title="Upcoming slots", xaxis_range=[-45, 45])
        st.plotly_chart(hist, use_container_width=True)
        st.caption(
            f"{len(recs)} upcoming slot starts · {int((pc != 0).sum())} with a new price · prices stay within "
            f"CHF {d.PRICE_FLOOR_CHF:.0f}–{d.PRICE_CEILING_CHF:.0f} and ±{d.MAX_MOVE_PCT:.0%} of the tariff."
        )

# ---------------------------------------------------------------- players
with tab_players:
    how_to_read(guide.PLAYERS)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("At-risk regulars (risque de départ)")
        min_baseline = st.number_input(
            "Regular = usual plays per 30 days ≥", 2.0, 20.0, 2.0, step=0.5, key="min_baseline",
            help="Average over days 31-120 before the snapshot; flagged when the last 30 days fell to ≤ 35% of it.",
        )
        risk = fetch("at_risk_players", float(min_baseline), 200)
        if risk is not None:
            st.metric("Players flagged", len(risk))
            st.dataframe(
                risk.assign(play_change_pct=pd.to_numeric(risk["play_change_pct"], errors="coerce") * 100),
                hide_index=True, use_container_width=True, height=420,
                column_config={
                    "player_uuid": st.column_config.TextColumn("Player (pseudonymised)", width="medium"),
                    "play_change_pct": st.column_config.NumberColumn("Change", format="%.0f%%"),
                    "avg_monthly_plays_baseline": st.column_config.NumberColumn("Usual / 30 d", format="%.1f"),
                    "plays_prev_30d": "Prev 30 d",
                    "plays_last_30d": "Last 30 d",
                    "days_since_last_play": "Days since last play",
                },
            )
    with c2:
        st.subheader("Pass Chrono expiring ≤ 14 days")
        chrono = fetch("expiring_chrono", 14)
        if chrono is not None:
            with_hours = chrono[chrono["pass_hours_remaining"].fillna(0) > 0]
            st.metric("Passes with unused hours", len(with_hours), f"{int(with_hours['pass_hours_remaining'].sum())} h left")
            st.dataframe(
                chrono, hide_index=True, use_container_width=True, height=420,
                column_config={
                    "player_uuid": st.column_config.TextColumn("Player (pseudonymised)", width="medium"),
                    "pass_hours_remaining": "Hours left",
                    "pass_expires_at": st.column_config.DatetimeColumn("Expires", format="DD.MM.YYYY"),
                    "days_to_pass_expiry": "Days to expiry",
                },
            )

# ---------------------------------------------------------------- genie
with tab_genie:
    st.subheader("Ask Genie (demandez à Genie)")
    st.caption(f"Natural-language questions in French or English on the gold tables · space “{SETTINGS.genie_space_title}”")
    how_to_read(guide.GENIE)
    if backend.name != "live":
        st.info(
            "Genie needs a live Databricks connection. Deploy the app (or set Databricks auth and "
            "DATABRICKS_WAREHOUSE_ID locally) and run the `setup_genie` job task. Try these questions then:"
        )
        for q in SAMPLE_QUESTIONS:
            st.markdown(f"- {q}")
    else:
        ss = st.session_state
        ss.setdefault("genie_conversation_id", None)
        ss.setdefault("genie_history", [])
        top = st.columns([3, 1])
        if top[1].button("🆕 New conversation"):
            ss.genie_conversation_id = None
            ss.genie_history = []
        picked = None
        cols = st.columns(len(SAMPLE_QUESTIONS))
        for i, q in enumerate(SAMPLE_QUESTIONS):
            if cols[i].button(q, key=f"sample_{i}", use_container_width=True):
                picked = q
        with st.form("genie_form", clear_on_submit=True):
            typed = st.text_input("Your question (votre question)", placeholder="Ex. Taux d'occupation du mardi midi ?")
            submitted = st.form_submit_button("Ask")
        question = picked or (typed if submitted and typed.strip() else None)
        if question:
            with st.spinner("Genie is thinking…"):
                try:
                    genie = get_genie(backend, backend.name)
                    ans = genie.ask(question, ss.genie_conversation_id)
                    ss.genie_conversation_id = ans.conversation_id
                    ss.genie_history.append((question, ans))
                except Exception as e:  # noqa: BLE001
                    st.error(f"Genie request failed: {e}")
        for q, ans in reversed(ss.genie_history):
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                if ans.error:
                    st.error(ans.error)
                if ans.text:
                    st.markdown(ans.text)
                elif not ans.sql and not ans.error:
                    st.write(f"Status: {ans.status}")
                if ans.sql:
                    with st.expander("Generated SQL"):
                        if ans.sql_description:
                            st.caption(ans.sql_description)
                        st.code(ans.sql, language="sql")
                if ans.result is not None:
                    st.dataframe(ans.result, hide_index=True, use_container_width=True)
                if ans.suggested_questions:
                    st.caption("Follow-ups: " + " · ".join(ans.suggested_questions))
        if ss.genie_conversation_id:
            st.caption(f"Conversation {ss.genie_conversation_id} · follow-up questions keep the context.")
