# 2026-10-27: What else predicts a second order? Fulfillment, ratings, returns, visits and intent

**Status:** done (descriptive, observational)
**Follows:** `2026-10-27-paths-across-datasets.md`. Grocery first is a separate test design: `2026-10-27-grocery-first-test-design.md`.

**Data:** aggregates only, behavior only.

| Dataset | Used for | License |
|---|---|---|
| **Olist** (Brazilian marketplace) | Delivery vs the promised date, delivery time, review score, payment | CC BY-NC-SA 4.0 |
| **Online Retail II** (UK gift wholesaler, B2B) | Cancellations and returns | CC BY 4.0 (unverified) |
| **OTTO** (German retailer, 25% shopper sample) | Visits, intent visits (focused views, carts) | CC BY 4.0 |
| **REES46** (electronics-heavy store) | Visits, intent visits (focused views, comparison within a category) | "© Original Authors" |

## Question

Beyond speed to the second order, what predicts that a new customer comes
back?
- Fulfillment: lateness, delivery time, failed orders.
- The rating they gave.
- Returns.
- Visits. The owner's theory: staying top of mind keeps visits high even
  without purchases, and *intent* visits (searches) matter most.

## Method

```bash
uv run --no-project --with pandas --with statsmodels python -I \
  docs/experiments/scripts/experience_factors.py {olist DIR | onlineretail FILE}
uv run --no-project --with pandas --with pyarrow --with statsmodels python -I \
  docs/experiments/scripts/visits_after_first_order.py {rees46 | otto} EVENTS.parquet
```

**Olist:**
- **Cohort:** 57,184 customers whose first order was placed between Jan
  2017 and Feb 2018.
- **Outcome:** another order within 180 days.
- **First-order factors:** delivered or not; delivery vs the promised date;
  days to delivery; the review they gave; value; shipping share;
  installments; payment type.
- **Regression:** a logistic regression adds month and region as
  adjustment controls (region only because delivery time depends on
  distance).
- **Late → low score:** reported separately, since the review is the
  customer's reaction to delivery.

**Online Retail II:**
- **Cohort:** 2,554 new customers (first order Mar–Nov 2010, after 3+
  months without one).
- **Exposure:** cancellations and returns in the first 90 days.
- **Outcome:** an order after day 90, and loyalty at 12 months.
- **Strata:** compared within groups by number of orders in the first 90
  days, since more orders give more chances to return something.

**Visits:**
- **Landmark design**, so that visits are never counted after the purchase
  they predict:
  - take first-time buyers;
  - keep those with **no second order** in the next 14 days (REES46) or 7
    days (OTTO);
  - count their visits in that window;
  - predict a second order in the following 14 or 7 days.
  - Concept: **landmark analysis**.
- **No search logs in either dataset, so intent is approximated:**
  - **focused:** a product viewed 2+ times in the visit;
  - **research:** 3+ products compared within one category (REES46);
  - **cart:** an add-to-cart (OTTO; REES46 logs carts fully only from Nov 8);
  - **shallow:** everything else.

## Results

### 1. Fulfillment and rating (Olist: 2.0% of new customers reorder within 180 days)

| First order | Reordered within 180 days |
|---|---|
| Delivered in 0–7 / 8–14 / 15–21 / 22–30 / 31+ days | **2.32%** / 2.02% / 1.89% / 1.73% / **1.26%** |
| Delivered 15+ days early / 1–7 early / on the day | 2.05% / 1.93% / 2.20% |
| Delivered 1–3 / 4–7 / 8+ days late | 1.83% / 1.44% / 1.58% |
| Review score 5 / 4 / 3 / 2 / 1 | **2.19%** / 1.84% / 1.95% / **1.53%** / 1.71% |
| Not delivered (unavailable) | 1.35% |

**Lateness drives the rating:**

| Delivered… | 8–14 days early | On the day | 1–3 days late | 4–7 days late | 8+ days late |
|---|---|---|---|---|---|
| Mean review score | 4.30 | 3.87 | 3.18 | 2.05 | **1.66** |
| Gave 1 or 2 stars | 9% | 14% | 35% | 67% | **78%** |

**After adjusting for everything else** (odds ratios, 95% CI):
- **Delivered 8+ days late vs 1–7 days early:** 0.90 [0.59, 1.36].
- **Gave 2 stars vs 5:** 0.70 [0.47, 1.03]; **1 star vs 5:** 0.83 [0.66,
  1.06].
- **Delivery took 31+ days vs 8–14:** 0.55 [0.30, 1.01].

None is clearly different from no effect.

### 2. Returns (Online Retail II, B2B)

| Returns in the first 90 days | Share | Ordered again after day 90 | Loyal at 12 months |
|---|---|---|---|
| None | 75% | 52.1% | 8.2% |
| Under 5% of the value bought | 15% | **76.9%** | **25.7%** |
| 5% or more | 10% | 59.8% | 12.6% |

**Within customers with the same number of first-90-day orders**, ordering
again after day 90 for those with no returns vs small returns:
- **1 order:** 44% vs 52%.
- **2–3 orders:** 68% vs 83%.
- **4+ orders:** 89% vs 92%.

### 3. Visits and intent

**OTTO** (140,974 first-time buyers with no second order in 7 days; 11.4% order in the next 7):

| Visits in the 7 days | 0 | 1 | 2–3 | 4–7 | 8+ |
|---|---|---|---|---|---|
| Second order in the next 7 days | 7.5% | 10.4% | 14.7% | 19.4% | **25.3%** |

| Same visits, different intent | 0 intent visits | 1 | 2+ |
|---|---|---|---|
| 1 visit | 9.2% | 13.1% | – |
| 2–3 visits | 12.4% | 15.1% | **19.8%** |
| 4–7 visits | 14.5% | 17.0% | **22.4%** |

**REES46** (90,427 first-time buyers with no second order in 14 days; 10.1% order in the next 14):

| Visits in the 14 days | 0 | 1 | 2–3 | 4–7 | 8+ |
|---|---|---|---|---|---|
| Second order in the next 14 days | 8.9% | 10.7% | 11.2% | 12.0% | 14.4% |

**Each kind of visit, adjusted for the others and for recency** (odds ratio
per doubling of visits of that kind, 95% CI):

| | Shallow | Focused | Research (compare in a category) | Cart | Visited in the last 2 days |
|---|---|---|---|---|---|
| OTTO | 1.12 [1.09, 1.14] | 1.20 [1.16, 1.23] | – | **1.60** [1.55, 1.65] | **1.54** [1.47, 1.61] |
| REES46 | 1.00 [0.96, 1.03] | 0.96 [0.92, 1.00] | **1.13** [1.09, 1.17] | – | **1.61** [1.51, 1.72] |

## Conclusion

1. **Visits after the first order predict the second, and intent visits
   predict it more than plain visits.** This supports the owner's theory.
   - **OTTO:** at the same number of visits, 2+ intent visits lift the
     second-order rate by about 8 points. Carts count most.
   - **REES46 (electronics):** only *comparison* visits matter. Plain and
     focused visits add nothing once comparison and recency are known.
   - **Recency is the strongest single signal in both:** a visit in the
     last 2 days means about 1.5–1.6× the odds.
   - **Plain visits help only in OTTO** (1.12 per doubling). So "top of
     mind" on its own is a weak signal; **intent visits plus recency** are
     the strong one.
   - Search logs would sharpen this: a search query is the clearest intent
     signal, and neither public dataset has one.
2. **Fulfillment and rating show the expected direction but no clear effect
   in a marketplace.**
   - Lateness strongly drives low ratings (78% of orders 8+ days late get
     1–2 stars).
   - Raw repeat rates fall with slow delivery (2.3% → 1.3%) and low
     ratings (2.2% → 1.5–1.7%).
   - After adjusting, none of these is clearly different from no effect.
   - With a 2% repeat base, Olist can't detect effects smaller than about
     ±40%. This needs a higher-repeat business (grocery) to measure.
   - Olist's promises were padded (most deliveries arrived 8+ days early),
     so "late" is rare and extreme.
3. **Returns go with *more* future business, not less.**
   - Small returns (under 5% of value) go with much higher repeat and
     loyalty, even among customers with the same number of early orders.
   - That's in line with research on product returns (Petersen & Kumar,
     *J. of Marketing* 2009).
   - The likely reason is engagement: customers who return a little are
     buying seriously. A smooth return is not a loyalty problem; a heavy
     returner might be.
4. **Category breadth** (from the paths write-up): loyal customers are more
   frequent, not broader, at the same number of orders. Breadth follows
   frequency.
5. **Grocery first** can't be tested on public data (no dataset sells both
   grocery and general merchandise to the same customers). The design for
   Walmart data is in `2026-10-27-grocery-first-test-design.md`.

**For lifecycle messaging:**
- **Add visit and intent signals to the shopper profile:** visits since
  the last order, intent visits (searches, product comparisons, carts), and
  days since the last visit.
- **Shoppers with recent intent visits but no second order** are the
  likeliest to buy next; they need relevance, not a discount.
- **Shoppers with no visits** after the first order are the reactivation
  group.
- **Use a fulfillment problem or a 1–2 star review as a trigger for service
  recovery,** not a promotion. The marketplace data can't show its effect,
  so measure it with a holdout.
- **Data to request:**
  - search queries tied to customers, plus visits and app opens;
  - fulfillment events (promised vs actual, substitutions, damage);
  - ratings and service contacts;
  - returns, with reasons.

**Caveats:**
- **Observational:** visits and orders share causes (a need comes up).
  Whether *raising* visits raises orders needs a randomized message with no
  offer.
- **Proxies:** intent is inferred from views and carts, not searches.
- **Short windows:** 7–14 days, on short datasets (4 weeks and 2 months).
- **Olist:** a cancelled first order may be followed by a re-placed one,
  which counts as a reorder.
- **Online Retail** is small (2,554) and B2B.
