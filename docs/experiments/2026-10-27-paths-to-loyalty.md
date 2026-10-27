# 2026-10-27: From first order to loyal: how many paths, and which are common? (H&M)

**Status:** done (descriptive, observational)
**Data:** H&M Personalized Fashion transactions, Sep 2018–Sep 2020 (Kaggle competition rules: aggregates only).
- Behavior only: the customer file (age, club status) is not used.
- Index groups (Ladieswear, Menswear, Baby/Children) are not used either: they would label shoppers rather than describe what they bought.

## Question

Do we have the data to study how a new customer becomes a loyal one? If so,
how many paths are there, and which are common?

## Which datasets can answer it

To follow customers from their first order to loyalty, we need three things:
- **a long window per customer** (a year or more);
- **true first orders**, with enough history before to know they are new;
- **repeat orders with product detail**.

| Dataset | Window | Verdict |
|---|---|---|
| **H&M** (fashion) | 2 years, 1.4M customers, 31.8M lines; product groups; store vs online | **Yes.** 6-month burn-in, then a year of follow-up per customer |
| Instacart (grocery) | Up to 100 orders per shopper | No: only shoppers with 4+ orders are included, so nobody who dropped out early is there (survivorship) |
| REES46 | 2 months | Too short for loyalty |
| OTTO | 4 weeks | Too short |
| RetailRocket | 4.5 months, anonymous | Too few repeat buyers |
| Olist (marketplace) | 2 years | About 3% ever order twice |
| Online Retail II | 2 years | Wholesale (B2B) buyers, a different business |
| dunnhumby Complete Journey | 2 years | Established households, not new ones (not downloaded) |

So this is **one dataset, one category (fashion)**. The method is built to
rerun on any store with two years of orders, including Walmart data.

## Method

```bash
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/paths_to_loyalty.py TRANSACTIONS_DIR articles.parquet
```

**Cohort:**
- **146,884 new customers:** first order between Mar 20 and Jun 19, 2019,
  after at least 6 months with no purchase. The data starts Sep 20, 2018.
- Each customer is followed for **365 days** after their first order.
- An order is a customer-day.

**Outcome at 12 months:**
- **Loyal:** 6+ orders *and* still buying in months 10–12. That is roughly
  the top third of established customers, whose median is 3 orders a year
  and 75th percentile 7.
- Otherwise: one order / 2–3 orders / 4+ orders but not loyal.

**Paths, three ways:**
1. **Order funnel:** the chance of an (n+1)-th order given n, and the days
   between orders.
2. **Activity paths:** which of the 12 thirty-day months after the first
   order had an order (a 12-month on/off pattern). Grouped into rule-based
   **families**.
3. **Step paths for the first three orders.** Each step is described by
   three things, giving 288 possible paths:
   - its gap: fast ≤ 30 days, medium 31–90, slow > 90;
   - whether it adds a **product group** new to the customer (garment
     upper body, shoes, accessories, …);
   - its channel.

   Concept: **sequence analysis** / customer path analysis.

**Channel:** `sales_channel_id` is undocumented. 2 is commonly read as
online and 1 as store; it's shown that way here, with that caveat.

## Results

**1. Outcome at 12 months**

| One order | 2–3 orders | 4+ orders, not loyal | **Loyal** |
|---|---|---|---|
| 36.4% | 31.5% | 19.8% | **12.2%** |

**Order funnel:** the second order is the biggest drop. After it, each next
order gets more likely and comes sooner.

| Step | 1→2 | 2→3 | 3→4 | 4→5 | 5→6 | 6→7 | 7→8 |
|---|---|---|---|---|---|---|---|
| Chance of the next order | **63.6%** | 70.2% | 71.8% | 72.8% | 73.3% | 73.3% | 73.6% |
| Median days between | 49 | 40 | 35 | 31 | 28 | 25 | 22 |

**2. How many paths? Nearly every possible one occurs.**
- **Activity paths:** 93,366 customers came back at least once. Between
  them they used **3,993 of the 4,095** possible 12-month patterns.
  - It takes 45 patterns to cover half of them, and 565 to cover 80%.
  - The most common pattern (one more order in month 1, then nothing)
    covers only 10%.
- **Step paths:** all **288 of 288** possible three-order paths occur. It
  takes 39 of them to cover half of the customers with 3+ orders.

Individual paths are too many to manage or message. **Families** of paths
are not:

| Family (12-month pattern) | Share of new customers | Loyal at 12 months |
|---|---|---|
| A. One and done | 36.4% | 0% |
| B. Quick repeat, then gone (months 1–3 only) | 12.9% | 0% |
| C. Early repeat, faded by month 9 | 12.3% | 0% |
| D. Slow return (after month 3), then gone | 10.7% | 0% |
| E. Slow start (after month 3), still buying in months 10–12 | 10.1% | 13% |
| F. Early repeat (months 1–3), still buying in months 10–12 | 10.9% | **46%** |
| G. Steady (6+ active months) | 6.6% | **88%** |

**There are three ways to become loyal, all through families E–G:**
- **Steady from the start (G):** 48% of loyal customers.
- **Early repeat that keeps going (F):** 41%.
- **Slow start that builds (E):** 11%.

**There are four ways to drop out:** never return (A), a quick burst (B), an
early repeat that fades (C), or one late return (D). B, C and D together
are **36%** of new customers. They came back at least once and still didn't
stay. That is the group lifecycle messaging can work on.

**3. What separates the paths early**

**Days to the second order**

| Days | 1–14 | 15–30 | 31–60 | 61–120 | 121–365 | None |
|---|---|---|---|---|---|---|
| Share of new customers | 16.8% | 8.1% | 10.5% | 11.7% | 16.5% | 36.4% |
| Loyal at 12 months | 25.9% | **27.9%** | 25.0% | 17.6% | **5.6%** | 0% |

**Second step, by element** (customers with 2+ orders)

| Element | Value | Loyal |
|---|---|---|
| Gap to the 2nd order | ≤ 30 days / 31–90 / > 90 | **26.6%** / 22.6% / **7.8%** |
| 2nd order adds a new product group | yes / no | 19.8% / 18.1% |
| Channel of the 2nd order | store / online | **22.5%** / 17.1% |

**First order** (all new customers)

| Trait | Value | Loyal | One order only |
|---|---|---|---|
| Channel | store / online | **15.0%** / 10.9% | 31.3% / 38.9% |
| Product groups in the basket | 1 / 2 / 3 / 4+ | 11.1% / 13.3% / 13.9% / 13.5% | 39.4% / 33.8% / 31.8% / 31.3% |
| Items in the basket | 1 / 2–3 / 4–6 / 7+ | 11.7% / 12.4% / 12.3% / 12.4% | 40.4% / 36.8% / 34.5% / 32.6% |

**Most common three-order paths** (customers with 3+ orders; first channel >
2nd step > 3rd step)

| Path | Share | Loyal |
|---|---|---|
| online > slow/new/online > fast/same/online | 2.4% | 17% |
| online > fast/same/online > fast/same/online | 2.4% | 31% |
| online > fast/new/online > fast/same/online | 2.3% | **38%** |
| online > medium/new/online > fast/same/online | 2.1% | 34% |
| online > fast/same/online > slow/new/online | 2.0% | 15% |
| store > fast/new/store > fast/same/store | 1.6% | **39%** |
| store > fast/new/store > fast/new/store | 1.5% | **42%** |

Mixed channels in the first two orders do even better: store first, then a
fast online order with a new group: 36.8%; online first, then a fast store
order with a new group: 32.6%.

## Conclusion

1. **Yes, we can study it, with H&M (fashion), and the answer to "how many
   paths" is: practically all of them.**
   - 3,993 of 4,095 possible monthly patterns, and 288 of 288 three-order
     paths, occur.
   - There is no single dominant journey. Designing for individual paths
     is a dead end.
2. **There are 7 path families, and only 3 lead to loyalty:** steady from
   the start, early repeat that keeps going, and slow start that builds.
   Together these hold every loyal customer.
3. **Speed to the second order is the strongest early sign.**
   - Second order within 60 days: 25–28% become loyal.
   - After 120 days: 5.6%.
   - The cliff is around 2 months. This is the same "second order is the
     bottleneck" story as LIFECYCLE.md, now with a timing threshold for
     fashion.
4. **Store and mixed-channel customers become loyal more often** (first
   order in store: 15.0% vs 10.9%; mixed first two orders: up to 37%).
   - This is consistent with the **omnichannel customer** finding in
     retail research (e.g. Sopadjieva, Dholakia & Benjamin, *HBR* 2017).
   - It's a correlation: store customers may simply live near a store.
5. **What's in the first basket matters little.** Breadth adds 2–3 points;
   size, nothing. Adding a new product group at the second order adds 1.7
   points.
   - This repeats the earlier cross-dataset result: expanding categories
     isn't what makes customers stay; coming back soon is.
6. **The group to work on is the 36% who return and then fade** (families
   B–D), plus the timing of the second order.
   - For messaging, that suggests:
     - a second-order nudge **before day 60**, timed from the customer's
       own first order;
     - then a "keep going" touch when an early repeater's gap starts
       stretching (C).
   - Both are hypotheses for an uplift test, not proven levers. A fast
     second order *predicts* loyalty; pushing one may not *create* it.

**What this means for data requests (e.g. Walmart):**
- **Full history from the account's first purchase,** so "new" is known
  rather than inferred from a burn-in.
- **Channel per order** (store, pickup, delivery, ship), documented.
- **Visits and message exposures,** to see paths that don't end in an order.
- **Three or more years,** so cohorts can be followed outside a shock like
  2020.

**Caveats:**
- **One retailer, one category (fashion).** Grocery paths (weekly rhythm)
  will look different, and Instacart can't show them because of its
  survivorship.
- **Burn-in:** 5.4% of all repeat gaps in the data exceed 6 months, so some
  "new" customers are returning ones.
- **COVID:** months 10–12 of the follow-up fall in Jan–Jun 2020. Store
  closures in spring 2020 depress "still buying" for store-led customers;
  even so, they came out ahead.
- **Channel coding is undocumented.**
- **The loyalty definition is a choice** (6+ orders, still buying). The
  funnel and the families don't depend on it.
