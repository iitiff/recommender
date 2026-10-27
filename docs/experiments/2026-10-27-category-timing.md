# 2026-10-27: When is each category bought? Category timing and personal routines

**Status:** done (descriptive, observational)
**Data:** aggregates only for all three.
- **Instacart** online grocery, 2017: 3.2M orders, 21 departments. Released for non-commercial use.
- **Olist** marketplace, Brazil: 97k orders. CC BY-NC-SA 4.0.
- **REES46** electronics-heavy store: 893k orders. "© Original Authors".

## Question

The owner's hypothesis: categories have their own rhythm (e.g. food around
5–6pm, mostly Friday and Saturday), so a category campaign should follow
it. Two questions:
1. Does a category's time-of-day / day-of-week pattern differ from the
   **store's overall** pattern? Everything peaks when people are online; a
   category rhythm only helps timing if it differs from overall traffic.
2. Do shoppers have **personal routines** (usual day, hour, cadence)?
   `2026-10-24-shopper-traits-stability.md` rejected personal send hours on
   REES46, an electronics store; groceries may differ.

## Method

```bash
python -I docs/experiments/scripts/category_timing.py instacart DIR
python -I docs/experiments/scripts/category_timing.py olist DIR
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/category_timing.py rees46 events.parquet
```

**Unit:** an order counts once per category it contains. For REES46, an
order is a customer-day.

**Time zones:**
- Instacart hours appear to be local.
- Olist timestamps are Brazil local time.
- REES46 is UTC shifted to Moscow time.

**Instacart never documented which weekday 0 is.** Days are shown as raw
indices (0–6), and named days can't be confirmed from this data.

**Distinctiveness:** total variation distance between the category's time
profile and the store's, i.e. the share of the category's orders that would
have to move to match. The distance expected by chance for a category of
that size is subtracted, so small categories don't look distinctive just
from noise. Also reported: the share of orders in the category's peak 3-hour
window and peak two days, vs the store's share in the same window and days
(lift).

**Personal routines (Instacart, 101,696 shoppers with 10+ orders):** learn
each shopper's usual 3-hour window, usual day and median gap between orders
from their first half of orders. Test on the second half against the
store-wide equivalents.

## Results

**Overall rhythm**
- **Instacart:** orders peak 9am–4pm and fall after 5pm (6pm: 5.3% of
  orders vs 8.5% at 10am). Days 0 and 1 are busiest (17.3% each, vs 12.5–13.8%).
- **Olist:** flat from 10am to 10pm; Saturday is lowest.
- **REES46:** midday peak; heavy weekends (Sat 16.0%, Sun 20.5%).

**Categories whose timing differs from their store**

| Store | Category | Distinctiveness (share of orders) | Peak | Lift vs store |
|---|---|---|---|---|
| Instacart | **Alcohol** | **0.11** | days 5 + 4; 2–5pm | **1.25×** days (32.0% vs 25.7%); 1.18× window (28.9% vs 24.5%) |
| Instacart | Meat & seafood, canned, dry goods, international | 0.05–0.06 | days 0 + 1 | 1.10× (≈38% vs 34.7%) |
| REES46 | **Furniture** | 0.06 | Sun + Sat | **1.21×** (44.1% vs 36.5%) |
| REES46 | Appliances | 0.06 | Sun + Sat | 1.18× (43.1% vs 36.5%) |
| Instacart | Produce, dairy, snacks, beverages | ≤ 0.02 | same as the store | – |
| Olist | Every category with ≥300 orders, food (n=450) included | ≤ 0.03 | – | – |

**Personal routines (Instacart)**

| Share of a shopper's later orders… | Personal | Store-wide |
|---|---|---|
| …in the 3-hour window (personal usual window vs store peak) | **32.2%** | 25.0% |
| …on the day (personal usual day vs store's busiest day) | **22.8%** | 17.4% |

For comparison, REES46 (electronics): personal window 19.7% vs store peak 18.6%.

**Cadence (Instacart):**
- Median gap between orders: 7 days; 10.2% of gaps are exactly 7 days.
- Predicting the next gap from the shopper's own median cuts the error only
  from 5.7 to 5.2 days.
- Gaps are capped at 30 days in the data, so long gaps are lumped together.

**Lead time: does online ordering run 1–2 hours ahead of the in-store peak?**

The owner's hypothesis: delivery takes 1–2 hours, so online grocery orders
should peak 1–2 hours before the in-store dinner peak (around 3–5pm).
Shares of Instacart orders by hour band (`hour_bands` in the script output):

| Ordering time | 0–6 | 7–9 | 10–12 | 13–15 | 16–18 | 19–23 |
|---|---|---|---|---|---|---|
| All orders | 2.7% | 15.5% | 24.8% | 24.7% | 19.9% | 12.4% |
| Alcohol (lift) | 0.68× | 0.72× | 0.99× | **1.14×** | **1.23×** | 0.79× |
| Meat & seafood (lift) | 0.98× | 0.96× | 1.00× | 1.03× | 1.00× | 0.99× |
| Breakfast (lift) | 0.98× | **1.08×** | 0.99× | 0.97× | 0.97× | 1.04× |
| Babies (lift) | 0.98× | **1.18×** | 0.92× | 0.98× | 0.89× | **1.17×** |

- **The pattern fits alcohol:** afternoon ordering, up to 1.23×.
- **It doesn't fit dinner staples:** meat, deli and produce follow the store
  across the whole day. Only 20% of all orders fall in 4–6pm.
- **The likely reason:** online grocery customers choose a delivery slot,
  often later the same day or the next day. The lag between ordering and
  receiving varies, so it isn't a fixed 1–2 hours.
- **Not testable here:** Instacart has no delivery-slot times, so the lag
  can't be measured.

## Conclusion

1. **Category rhythm is real for a few categories, not for most.**
   - Alcohol has the clearest rhythm: late-week afternoons, about 1.2×. If
     day 0 is Sunday, as is commonly assumed, its peak days are Thursday and
     Friday. That fits weekend preparation and partly supports "Friday",
     but it can't be confirmed from this data.
   - Stock-up groceries cluster on the busiest days (the weekly shop), and
     furniture and appliances on weekends.
   - Most food (produce, dairy, snacks, beverages) is bought exactly when
     everything else is. For them, "send at the store's peak" is as good as
     a category rule.
2. **"5–6pm" didn't show up in online grocery.** Instacart orders peak late
   morning to mid-afternoon and drop after 5pm. That is ordering *for
   delivery*. In-store trips after work are a different channel and may
   well peak at 5–6pm. Timing must be learned **per channel** as well as per
   category and store.
3. **Correction: personal routines matter for routine categories.**
   - Grocery shoppers' own usual window and day beat the store's peak by
     about 1.3× (32% vs 25%; 23% vs 17%).
   - The earlier rejection of personal send hours (+1 point on REES46)
     holds for electronics, not for groceries. The exact next-order day stays
     hard to predict (±5 days).
4. **For campaigns:**
   - Default to the store's rhythm.
   - Add a category rhythm only where it is distinctive (alcohol,
     stock-up, home projects).
   - Use personal routine for repeat grocery shoppers.
   - All of these describe *when demand happens*. Whether sending then
     *lifts* response is a send-time experiment (randomize the send hour),
     not something this data can show.

**What this means for data requests:**
- **Timestamps in local time,** or the customer's or store's time zone. UTC
  alone forced guesses here (REES46).
- **Channel** (online, in store, pickup, delivery), since rhythms differ by
  channel.
- **Message send and open times,** for the send-time experiment.
- **Order placed time vs fulfilment slot** (delivery or pickup window), so
  the lead time between ordering and receiving is measured, not assumed.
  Online send times should be learned from *ordering* times per channel and
  customer, not derived as "store peak minus 2 hours".
- Category rhythms are computed from transactions and the catalog; they
  need no new customer attribute.

**Caveats:**
- One grocery dataset, from 2017, online only, and with unlabeled weekdays.
- Olist's food sample is small (450 orders).
- REES46 time zone assumed (Moscow).
- Observational: when people buy is not when a message works best.
