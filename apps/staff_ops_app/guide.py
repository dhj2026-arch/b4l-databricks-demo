"""In-app reading guidance (Markdown) shown in each tab's "How to read this" expander."""

EXPANDER_LABEL = "ℹ️ How to read this (comment lire)"

SNAPSHOT = """
The data is a snapshot taken on the **as-of date**. Days before it are final history; from the as-of date onwards
only the bookings made so far are counted. The app does not move the snapshot forward by itself: re-run
`b4l_demo_job` (empty `as_of_date` = today) on the morning of a demo.
"""

HEATMAP = """
**KPI tiles:** occupancy, revenue, bookings, check-in persons (fréquentation) and waitlist for the selected day.
- On a **past day**, the deltas compare with the same weekday one week earlier.
- On an **upcoming day** there are no deltas and the figures are *bookings so far*: a low number a few days
  ahead is normal, not a bad day. Check-ins stay at 0 until people arrive.

**Grid:** one row per court, one column per clock hour.
- **Colour** = occupancy of that court-hour (booked + blocked minutes / available minutes); darker is fuller.
- **⛔** = court block (Bad'matin, private lesson, club BLA or school). It counts as occupied but is never
  re-priced and earns no court-rental revenue.
- **★** = Prime Show Court (Court_01–04), the sponsor naming-rights inventory.
- **Hover** a cell for tier, base price, booked and blocked minutes and revenue.
- Weekends and public holidays open 09:30–19:00, so the 09:00 column has only 30 rentable minutes.

**What to look for:** on a past Mon–Thu, a dark band from 17:00 (Tarif soir, ~90% full), fairly dark lunches
and pale mornings and early afternoons. The same tariff for very different demand is the case for dynamic pricing.
"""

SLOTS = """
Each bar is a 1-hour **booking start time** on an upcoming day (bookings start on the hour or half hour).
- **Bar** = share of the available courts already booked for that start. **Red** = under-filled.
- **Dotted line** = usual fill rate for the same weekday and start time over the last 4 weeks.
- **Under-filled** = booked now < slider × usual rate (default 0.6). These are the dynamic-pricing candidates.
- **Courts available** excludes blocked courts (e.g. Bad'matin and school sessions on Tuesdays).

**Reading tips**
- The fill rate counts bookings *starting* at that exact time, not hours covered, so it is lower than the
  heatmap's occupancy. Compare slots with their own dotted line, not with the heatmap.
- Days further ahead always look emptier because people have not booked yet: compare slots within the same day.
- Story for an under-filled lunch: *"tomorrow's lunch is far behind its usual pace; in Phase 2 the model proposes
  a lower Tarif midi within the guardrails (CHF 18–42, max ±40%)."* Pass holders and family, school and
  association rates are never re-priced. The approve button is disabled until Phase 2.
"""

PRICING = """
**What this shows:** what the last 24 weeks would have earned if every slot had carried the model's price, posted
two weeks ahead. It is a simulation, not a measurement: the same simulated players and bookings are replayed with
only the prices changed.
- **Price sensitivity** (sensibilité au prix) is unknown, so each figure is shown for three assumptions:
  *low* (players barely react), *mid* and *high* (players switch slot or skip for a few francs). Quote the range.
- **Protected bookings** (passes, family rate, courses, club blocks) always keep the tariff.
- **Cannibalisation**: a flexible player can move to a cheaper slot up to 2 hours away. A discount that only pulls
  players from a full slot nearby shows up as a loss in that slot's tier.

**Single slot:** the curve is the expected revenue at each price. Courts already booked keep their price. The
recommended price (★) is the highest point, but only if it beats the tariff (◆) by at least 10% (15% at weekends);
weekday slots in the 2 hours before the evening tariff are never discounted. Otherwise ★ sits on the tariff.

**Why those rules:** the backtest showed that small weekend changes and late-afternoon discounts lose money once
players moving between slots are counted; with the rules the gain is positive at every price sensitivity.

**Heatmaps:** occupancy per week and tier at the tariff (left) and with dynamic prices (right).

**Histogram:** how far the upcoming recommended prices move from the tariff; the guardrails stop at ±40%.

**Before a decision:** confirm the effect with a pilot (e.g. a few weekday mornings for 4–6 weeks) before going live.
"""

PLAYERS = """
Only pseudonymised ids (HMAC-SHA256) are shown; in a real deployment reception looks the person up in the
booking system.

**At-risk regulars (left)**
- **Usual / 30 d** = the player's normal pace: average plays per 30 days over the 3 months before the last 30 days.
- **Last 30 d** = plays in the last 30 days. A regular is **flagged when Last 30 d ≤ 35% of Usual / 30 d**.
- **Change** compares with the previous 30 days only: context, not the rule.
- The list is sorted by usual pace, so the top rows are the most valuable regulars to call or send a voucher to.
  Raise *Regular = …* to narrow it to the heaviest players.

**Pass Chrono expiring ≤ 14 days (right)**
- Each row is a player with prepaid hours that will be lost at expiry. **Days to expiry 0–2** are urgent.
- Action: remind them to book, ideally into a quiet slot from *Slots & prices*, which links both use cases.
"""

GENIE = """
Ask in French or English; answers come from the same gold tables as the other tabs.
- Follow-up questions keep the context (e.g. *"et par segment ?"*). **New conversation** resets it.
- Open **Generated SQL** to show it is not a black box: occupancy is computed as
  SUM(booked + blocked) / SUM(available), and "today" is the snapshot date, not the calendar date.
- Questions that tie back to the other tabs:
  - *Quel est le taux d'occupation du mardi midi sur les 8 dernières semaines ?* (Slots & prices)
  - *Combien de joueurs réguliers sont à risque de départ, par segment ?* (Players)
  - *How much revenue did rainy Saturdays bring vs dry Saturdays?*

**5-minute walk-through:** heatmap of last Thursday (evenings full, middays not) → Slots & prices for the next
quiet lunch (capacity we can see coming) → Players (who is drifting, who has unused hours) → Genie (ask anything else).
"""
