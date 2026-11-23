# Genie space: B4L Smart Badminton Center

The Genie space lets the B4L management and reception team ask questions in French or English about the 16 courts.
It is created by `src/06_setup_genie.py` (job task `setup_genie`, after the Lakeflow pipeline). The definition itself
(tables, instructions, synonyms, sample questions and example SQL) lives in `src/b4l/genie_space.py`, which is unit
tested in `tests/test_genie_space.py`.

## Tables

All five gold tables in `{catalog}.{schema_prefix}_gold` (default `serverless_stable_yqq46w_catalog.b4l_gold`). Their
column comments come from `src/b4l/schemas.py` and are set by the pipeline.

| Table | Grain | Typical questions |
|---|---|---|
| `court_utilization_hourly` | court × date × hour | occupancy by tier/hour/hall, revenue by tier, court blocks |
| `daily_kpis` | date | revenue, bookings, late cancellations, no-shows, waitlist, footfall |
| `player_features` | player (pseudonymised) | churn risk, pass state, Pass Chrono expiry |
| `slot_features` | date × start time (court-agnostic) | upcoming slots, bookings so far vs 4-week fill rate |
| `sponsor_metrics` | month × group × key | anonymised footfall by segment, age band, time band, hall, show court |

## Instructions (summary)

A single text instruction (the API allows one) covers:

- **Currency and tariffs:** CHF per court per hour; Day = *Tarif journée* CHF 25, Lunch = *Tarif midi* CHF 32,
  Evening = *Tarif soir* CHF 35, Weekend = *Tarif week-end* CHF 25, *Famille* CHF 18. The tier is set by the start time.
- **Rentable hours:** weekdays 08:00–22:00 (hours 8–21), weekends and Vaud public holidays 09:30–19:00 (hours 9–18).
  Closed 24, 25, 31 December and 1 January.
- **Occupancy (*taux d'occupation*)** = (booked + blocked minutes) / available minutes, always aggregated as a ratio of
  sums, never as an average of `occupancy_rate`.
- **Time:** "today" is `MAX(as_of_date)` in `player_features`, which equals `run_info.as_of_date`. Earlier dates are
  history. The next 14 days only hold the bookings made so far, so they are excluded from historical figures unless
  the question is about upcoming days.
- **Holidays:** `is_school_holiday` (Vaud *vacances scolaires*) and `is_public_holiday` (*jours fériés*, weekend hours).
- **Privacy:** only pseudonymised `player_uuid`. `sponsor_metrics` is k-anonymised: cells with fewer than 10 players
  are suppressed (`suppressed = true`, metrics null). Genie must filter them out and never back-derive them.
- **Halls:** Main Hall Court_01–08, West Hall Court_09–16, show courts Court_01–04.
- A French/English glossary (below).

## Synonyms

The glossary is embedded in the instruction. Column-level synonyms go into `data_sources.tables[].column_configs`.

| Term | Meaning on the gold tables |
|---|---|
| Tarif journée | `tier = 'Day'`, CHF 25 |
| Tarif midi | `tier = 'Lunch'`, CHF 32 ("créneaux de midi") |
| Tarif soir | `tier = 'Evening'`, CHF 35 |
| Tarif week-end | `tier = 'Weekend'`, CHF 25 (Fri from 13:30, Sat, Sun, holidays) |
| Famille | Family rate CHF 18 (rate category, not a tier) |
| Bad'matin | Tuesday 09:30–11:00 coached session; `blocked_minutes`, `block_revenue_chf` |
| Taux d'occupation | (booked + blocked) / available minutes |
| Carte prépayée | `pass_product = 'Carte_Prepayee'`, `pass_balance_chf` |
| Pass Liberté | `pass_product = 'Pass_Liberte'`, `pass_hours_remaining` |
| Pass Chrono | `pass_product = 'Pass_Chrono'`, `pass_expires_at`, `days_to_pass_expiry` |
| Liste d'attente | `waitlist_requests`, `waitlist_resolved` |
| 16 terrains / courts | `court_id` Court_01..Court_16, `hall_zone`, `is_show_court` |
| Chiffre d'affaires | `revenue_chf` (+ `block_revenue_chf`) |
| Fréquentation | `check_in_persons`, `unique_players` |
| Annulation tardive | `cancellations_late` (< 24 h, billed); `no_shows` |

## The 10 sample questions

1. Quel est le taux d'occupation des créneaux de midi la semaine dernière ?
2. Which hours on weekdays have the lowest occupancy in October?
3. How much revenue did rainy Saturdays bring vs dry Saturdays?
4. Combien d'annulations tardives ce mois-ci ?
5. Which players have a Pass Chrono expiring in the next 14 days?
6. Quel est le chiffre d'affaires par tarif (journée, midi, soir, week-end) et par mois ?
7. Which upcoming slots in the next 14 days are booked well below their 4-week fill rate?
8. Combien de joueurs réguliers sont à risque de départ, par segment ?
9. What was the monthly footfall (fréquentation) by player segment for sponsors?
10. Compare Main Hall and West Hall occupancy during Vaud school holidays vs term time.

Questions 1–6 and 9 also have **example SQL** (`instructions.example_question_sqls`), so Genie learns the
ratio-of-sums occupancy, the snapshot date and the k-anonymity filter.

## API used

- `serialized_space` version 2 (JSON string): `config.sample_questions`, `data_sources.tables` (sorted by identifier,
  with `column_configs`), `instructions.text_instructions` (one) and `instructions.example_question_sqls`. The IDs are
  deterministic 32-character hex values and every collection is pre-sorted, as the API requires.
- Create: `w.genie.create_space(warehouse_id, serialized_space, title, description)` = `POST /api/2.0/genie/spaces`.
- Update: `w.genie.update_space(space_id, serialized_space=..., ...)` = `PATCH /api/2.0/genie/spaces/{id}` (full
  replacement).
- Find: `w.genie.list_spaces()` = `GET /api/2.0/genie/spaces`, matched on the title `B4L Smart Badminton Center`.
- If the installed SDK lacks these methods, the notebook calls the same endpoints through `w.api_client.do`.
- If the API rejects the full payload, the notebook retries with simpler payloads, in this order: no column configs,
  then no example SQL, then tables and sample questions only. It prints which variant was applied.

## Access for the staff app

The notebook looks up the app `b4l-staff-ops` (`w.apps.get`) and uses its `service_principal_client_id` to:

- run `GRANT USE CATALOG`, `GRANT USE SCHEMA` and `GRANT SELECT` on the gold schema through the SQL warehouse;
- set **CAN_RUN** on the space (`w.permissions.update(request_object_type="genie", ...)`).

`CAN_USE` on the warehouse comes from the app's `sql-warehouse` resource (`resources/staff_ops_app.app.yml`). If the
app does not exist yet, the grants are skipped with a message; deploy the bundle and re-run the task.

## Run and re-run

```bash
databricks bundle deploy
databricks bundle run b4l_demo_job                          # full pipeline, ends with setup_genie
databricks bundle run b4l_demo_job --only setup_genie       # refresh only the Genie space and grants
```

The task is idempotent: it updates the existing space with the same title instead of creating a new one. Edits made in
the UI are overwritten on the next run, so change `src/b4l/genie_space.py` instead. The task prints the space ID and
URL (`<host>/genie/rooms/<space_id>`) and sets the task values `genie_space_id` and `genie_space_url`.

## Manual fallback (UI)

If the API path fails (for example, Genie isn't enabled or the payload is rejected):

1. **SQL → Genie → New**. Title `B4L Smart Badminton Center`; pick the Serverless Starter Warehouse.
2. Add the five tables from `serverless_stable_yqq46w_catalog.b4l_gold`.
3. **Instructions → General instructions:** paste the output of
   `python -c "from b4l import genie_space as g; print(g.instructions_text('serverless_stable_yqq46w_catalog','b4l'))"`
   (run from `src/`).
4. **Instructions → SQL queries:** add the example SQL from `genie_space.EXAMPLE_SQLS` (replace `{g}` with the gold
   schema).
5. **Settings → Sample questions:** add the 10 questions above.
6. **Share:** give the `b4l-staff-ops` service principal **Can run**. Then run the three GRANTs printed by
   `genie_space.grant_statements(...)` in the SQL editor.
7. Optional: under Data, add the column synonyms from `genie_space.COLUMN_SYNONYMS`.
