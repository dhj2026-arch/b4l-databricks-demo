# B4L Smart Badminton Center: Databricks demo

A Databricks demo for the [B4L Fondation](https://b4l-fondation.ch/) badminton centre at Malley, Lausanne (16 courts). It turns booking data into utilisation insight, a natural-language Q&A for staff (Genie) and, in Phase 2, revenue-maximising prices with a static vs. dynamic pricing backtest. See [SPEC.md](SPEC.md) for the full specification.

**Phase 1 (this repo today):** synthetic data, the Lakeflow pipeline, the Genie space and a read-only staff app.

```text
01_generate_synthetic_data  ->  b4l_bronze   (24 weeks + next 14 days, deliberate ~1% dirty rows)
02_lakeflow_etl (SDP)       ->  b4l_silver   (deduplicated, HMAC-pseudonymised, quarantine)
                            ->  b4l_gold     (court_utilization_hourly, daily_kpis, player_features,
                                              slot_features, sponsor_metrics)
06_setup_genie              ->  Genie space "B4L Smart Badminton Center"
apps/staff_ops_app          ->  Databricks App "b4l-staff-ops" (heatmap, slots, players, Ask Genie)
```

## Deploy and run (FEVM workspace)

```bash
databricks auth login --profile fevm-serverless-stable-yqq46w   # if the token has expired
databricks bundle validate
databricks bundle deploy                                         # job, pipeline, app
databricks bundle run b4l_demo_job                               # generate -> pipeline -> Genie (~10-15 min)
databricks bundle run staff_ops_app                              # deploy and start the app
```

- Everything goes into catalog `serverless_stable_yqq46w_catalog`, in schemas `b4l_bronze`, `b4l_silver`, `b4l_gold` and `b4l_models`. Override with `--var catalog=... --var schema_prefix=...`.
- Job parameters:
  - `seed` (default `20261123`).
  - `as_of_date` (default `2026-11-23`, a Monday; set by the bundle variable `as_of_date`). The demo moments are anchored to it. Pass `--var as_of_date=""` to use today in Lausanne, for example on the morning of a demo.
  - `n_players` (default 3000).
  - `secret_scope` (default `b4l`).
- On first run, the generator creates the secret scope `b4l` with a random `pseudonymisation_key`. Keep it: changing the key changes every `player_uuid`.
- The app is granted access by the `setup_genie` task. If the app didn't exist yet when the job ran, run `databricks bundle run b4l_demo_job --only setup_genie` after deploying it.

### Run the app locally

```bash
cd apps/staff_ops_app
B4L_USE_MOCK=true streamlit run app.py                    # offline, deterministic mock data
DATABRICKS_CONFIG_PROFILE=fevm-serverless-stable-yqq46w DATABRICKS_WAREHOUSE_ID=<id> streamlit run app.py
```

### Tests and data profile

```bash
python -m venv .venv && .venv/bin/pip install numpy pandas pytest   # pyspark is not needed
.venv/bin/python -m pytest -q
.venv/bin/python scripts/profile_synthetic.py                       # calibration printout
```

## The synthetic data

The generator in `src/b4l` runs a deterministic demand simulator. Players have segments, willingness to pay, habits, doubles groups and a lifecycle; then come price-independent booking intents, then court allocation with cancellations, a waitlist and pass wallets. Prices are an explicit input. The published tariff produces the history (the "static world"), and Phase 2 replays the same intents with dynamic prices.

Profile for seed `20261123`, as of 2026-11-23 (history 2026-06-08 to 2026-11-22):

- **Volume:** 3,000 players, about 19.9k bookings (about 810 a week), 64k participant rows, 17k check-ins, 1.3k waitlist entries. Revenue is about CHF 22.5k a week.
- **Occupancy in term time:**

  | Band | Occupancy |
  |---|---|
  | Mon–Thu evening | 90% |
  | Lunch | 73% |
  | Weekend | 81% |
  | Weekday morning | 38% |

  The summer dip runs from 47% in July to 72% in November.
- **Behaviour:** the top 20% of bookers make 59% of bookings. Late cancellations and no-shows are 7.1%. 16% of bookings use pass hours, and 15% use the family rate.
- **Quality gate:** each run logs 30 checks to `b4l_bronze.data_quality_report`: hard rules, statistical checks and the five demo moments. The job fails if a hard rule or demo moment fails. It has been checked across six seed and date combinations from September to March.

### Demo moments (`b4l_bronze.demo_moments`)

1. **Rainy Saturday spike.** The next Saturday at least 2 days ahead has heavy rain forecast and is booking ahead of a usual dry Saturday.
2. **Quiet Tuesday lunch.** It is 1–3 days ahead and well behind its usual pace: a candidate for a lower price.
3. **Drifting evening regular.** Plays in the last 30 days dropped to 35% of their baseline or less (`churn_risk_rule`).
4. **Likely late cancellation.** An upcoming Thursday 19:00 booking by a player with a history of late cancellations. Thursday evenings are full, so a freed court can go to the waitlist.
5. **Pass Chrono about to expire.** Unused hours expire within 2–10 days.

## Key assumptions (details in SPEC.md)

- **Prices:**
  - Tariffs and opening hours are taken from the website.
  - School and association rates are CHF 20 (not published).
  - Friday from 13:30 uses the weekend rate.
- **Blocks:**
  - Bad'matin on Court_01–04 (Tuesday morning, term).
  - School sessions on Court_09–10 (Tuesday and Thursday 13:30–15:30, term).
  - Club BLA on Court_13–16 (Monday 18–20 in term, Thursday 20–22).
  - Private lessons on Court_16.
- **Capacity:** the rentable grid gives 1,424 court-hours a week. The website quotes 1,368 slots, probably counting slightly different opening edges.
- **Churn rule:** a regular averages at least 2 plays per 30 days over days 31–120 before the snapshot; they are flagged when the last 30 days are at most 35% of that.
- **Data:** all names, companies and contacts are fictitious. Silver and gold contain only HMAC-SHA256 pseudonyms, and sponsor metrics suppress cells with fewer than 10 players.
