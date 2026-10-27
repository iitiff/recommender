# 2026-10-27: Paths to loyal and valuable in grocery (Complete Journey, X5, Instacart)

**Status:** done (descriptive, observational)
**Follows:** `2026-10-27-paths-to-value.md` (fashion) and `2026-10-27-grocery-first-test-design.md`.

**Data:** aggregates only, behavior only (no demographics used).

| Dataset | What it is | Use here |
|---|---|---|
| **84.51° Complete Journey** (R package `completejourney`, CC0) | One year (2017) of *every* purchase by 2,469 households who shop a US grocery chain frequently. Includes stores, loyalty-card and coupon discounts, coupon redemptions, departments including general merchandise | Value grid, grocery vs general merchandise, quarter-to-quarter paths |
| **X5 RetailHero** | Grocery chain; the client list was probably limited to active shoppers | New card holders' first 90 days |
| **Instacart** | Online grocery; only shoppers with 4+ orders | Path to a weekly routine among shoppers who stay |
| Groceries dataset (Kaggle mirror) | 3,898 members, median 4 visits in 2 years | Too sparse; not used |

None of these follow truly new grocery customers for a year.
- **Complete Journey's** households are established and frequent.
- **X5** covers 90 days of a selected list.
- **Instacart** excludes early dropouts.

So the grocery paths below describe *how frequency and value form among
customers who stay*, not who converts.

## Method

```bash
uv run --no-project --with pandas --with pyreadr python -I \
  docs/experiments/scripts/grocery_paths.py cj COMPLETEJOURNEY_DIR
uv run --no-project --with pandas python -I docs/experiments/scripts/grocery_paths.py x5 X5_DIR
uv run --no-project --with pandas python -I \
  docs/experiments/scripts/grocery_paths.py instacart INSTACART_DIR
```

**Complete Journey:**
- **Value grid:**
  - loyal = active in at least the median number of weeks, and active in
    the last 13 weeks;
  - valuable = spend at or above the median.
- **General merchandise (GM)** = drug GM, cosmetics and garden center
  departments.
- **Quarter states:** active weeks per quarter: 0, light (1–3), mid (4–8),
  regular (9–13).

**X5 new card holders:** what day 30 says about days 31–90.

**Instacart:** what orders 1–3 say about becoming a weekly regular (median
gap of 7 days or less over orders 4–10).

## Results

### 1. Loyal × valuable in grocery (Complete Journey)

| Group | Share | Active weeks | Trips | Spend | Spend per trip | Discount share | Coupons redeemed | Departments |
|---|---|---|---|---|---|---|---|---|
| **True friends** | 43% | 43 | 89 | $2,778 | ~$31 | 15.0% | **1.8** | **13.2** |
| **Barnacles** | **8.3%** | 33 | 52 | $897 | **~$17** | 16.4% | 0.2 | 10.1 |
| **Butterflies** | 6.8% | 23 | 33 | $1,582 | **~$48** | 15.3% | 0.5 | 11.6 |
| Strangers | 42% | 13 | 17 | $383 | ~$23 | 15.9% | 0.0 | 7.8 |

Medians for weeks, trips and spend; means otherwise. Every group shops its
main store for 72–75% of trips.

| Discount share (quartile) | Q1 (least) | Q2 | Q3 | Q4 (most) |
|---|---|---|---|---|
| True friends | 40.6% | **51.1%** | 46.4% | 34.7% |
| Barnacles | 6.8% | 7.1% | 9.6% | **9.9%** |

### 2. Grocery vs general merchandise (Complete Journey)

**General merchandise share in Q1 vs active weeks in Q2–Q4 (of 39),** within
households with the same Q1 activity:

| Q1 active weeks | Low GM share | Mid | High |
|---|---|---|---|
| 1–3 | 10.4 | 11.6 | **12.4** |
| 4–8 | 19.3 | **21.8** | 19.5 |
| 9–13 | 31.2 | **32.6** | 31.8 |

**Adding general merchandise:** households with no GM purchases in Q1 that
start buying GM in Q2 (n = 162), vs households with no GM in Q1–Q2 (n =
76). Active weeks go from 2.8 to 4.6 for adopters vs 2.3 to 2.6, a
difference of about +1.9 weeks per quarter after matching on Q1 activity.

### 3. The grocery frequency ladder (Complete Journey, quarter to quarter)

| From \ to | 0 weeks | Light (1–3) | Mid (4–8) | Regular (9–13) |
|---|---|---|---|---|
| 0 | 35% | 47% | 16% | 2% |
| Light | 14% | 45% | **36%** | 5% |
| Mid | 2% | 18% | 53% | **27%** |
| Regular | 0% | 2% | 18% | **80%** |

**The quarter before a move** (mean departments bought, spend per trip):

| Light households that next quarter… | Dropped to 0 | Stayed light | Moved to mid |
|---|---|---|---|
| Departments | 3.2 | 3.8 | **4.5** |
| Spend per trip | $28 | $31 | $34 |

| Mid households that next quarter… | Dropped to light | Stayed mid | Moved to regular |
|---|---|---|---|
| Departments | 6.2 | 6.9 | **7.7** |
| Spend per trip | $30 | $34 | $35 |

### 4. New grocery customers' first 90 days (X5, 12,202 new card holders)

| By day 30 | Weekly or more in days 31–90 (9+ visits) | High value in days 31–90 |
|---|---|---|
| **Visits:** 1 / 2–3 / 4–7 / 8+ | 26% / 24% / 45% / **79%** | 35% / 36% / 49% / 70% |
| **Stores used:** 1 / 2 / 3+ | 41% / 50% / **64%** | 46% / 52% / 60% |
| **Items below the usual price:** under 10% / 10–30% / 30%+ | 44% / **52%** / 38% | 48% / 53% / 39% |

### 5. Path to a weekly routine (Instacart, 101,696 shoppers with 10+ orders)

41% become weekly regulars.

| By order 3 | Become weekly regulars |
|---|---|
| **Second order within 7 days** / 8–14 / 15–29 / 30+ | **60%** / 35% / 22% / 21% |
| **Third order within 7 days** / 8–14 / 15–29 / 30+ | 60% / 35% / 21% / 19% |
| Basket size (smallest → largest quintile) | 41% / 43% / 43% / 41% / **38%** |
| Departments in orders 1–3 (low / mid / high) | 43% / 41% / **39%** |

## Conclusion

1. **In grocery, the path to loyal is a ladder of frequency, climbed one
   step at a time.**
   - Light households move up to mid 36% of the time, but jump straight to
     regular only 5% of the time.
   - Mid households move up to regular 27% of the time.
   - Regulars are sticky: 80% stay.
   - So programs should aim for **the next rung** (light → mid), not "make
     them weekly".
2. **Early speed is the strongest grocery signal, as in fashion.**
   - **Instacart:** a second or third order within 7 days → 60% become
     weekly regulars, vs about 20% after 15+ days.
   - **X5:** 8+ visits in the first 30 days → 79% shop weekly afterwards.
   - **The grocery version of the "60-day cliff" is about 1–2 weeks.**
3. **Bigger, broader first baskets don't make a weekly shopper.**
   - On Instacart, the largest and broadest early baskets are slightly
     *less* likely to turn weekly. They are stock-up shoppers.
   - Across a quarter (Complete Journey), buying from more departments
     comes before moving up a rung. But departments grow with trips, so
     this is partly mechanical.
4. **Grocery barnacles are different from fashion barnacles.**
   - Complete Journey has more of them (8% vs 1–3% in H&M).
   - They are *frequent small top-up shoppers*: about $17 a trip vs $31
     for true friends, with narrower baskets (10 vs 13 departments) and
     few coupons.
   - That is a **share-of-wallet** signal: they likely do their main shop
     elsewhere. In the profile, they belong to the "grow share of wallet"
     program (experiment 2.14), not a discount program.
5. **Some deal use goes with loyalty; heavy deal use doesn't**, the same
   inverted U as H&M.
   - True friends peak in the second discount quartile (51%) and fall in
     the top one (35%).
   - New X5 shoppers with 10–30% of items below the usual price become
     regular most often (52% vs 38% at 30%+).
   - True friends also redeem the most targeted coupons (1.8 vs 0.2 for
     barnacles). Engaged customers use coupons; coupons don't make
     barnacles.
6. **Grocery first: these data don't support "keep them in grocery".**
   - Light households that also buy general merchandise come back slightly
     more.
   - Households that add general merchandise grow their activity by about
     +1.9 weeks per quarter more than those that don't.
   - Both may be reverse causality: more active households discover more
     departments. That's exactly why the grocery-first question needs the
     randomized design (`2026-10-27-grocery-first-test-design.md`).
7. **Using more stores goes with regularity** (X5: 64% weekly with 3
   stores vs 41% with one), like H&M's store + online customers.
   Convenience and coverage may be part of the path. This is also
   selection-prone.

**Changes to the plan:**
- **Validation 1.1 for grocery:** measure the frequency ladder (quarter
  transitions) and the "first 1–2 weeks" cliff on new grocery customers.
  No public grocery dataset can show conversion.
- **Validation 1.16:** split barnacles into deal-driven (fashion-like) vs
  top-up (share-of-wallet) types, using spend per trip and department
  breadth.

**Caveats:**
- **Complete Journey** households are frequent shoppers of one chain, for
  one year: no new customers and no conversion, and value has no margin
  (spend after discounts).
- **X5 and Instacart are selected** (active or 4+ order shoppers), and
  Instacart caps gaps at 30 days.
- **The general merchandise "adopters" comparison is small** (162 vs 76)
  and adoption requires a Q2 trip, which biases it toward active
  households.
- **Everything is observational.**
