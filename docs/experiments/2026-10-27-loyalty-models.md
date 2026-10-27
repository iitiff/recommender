# 2026-10-27: Loyalty models: new factors, "still alive" models, time to second order, path clusters

**Status:** done (descriptive and predictive; observational)
**Follows:** `2026-10-27-loyalty-drivers.md` and `2026-10-27-paths-across-datasets.md`.
**Data:** aggregates only, behavior only.
- H&M, Olist, REES46, OTTO, CDNOW, as in the earlier write-ups.

**Packages:** `pymc-marketing` (CLV models), `lifelines` (survival),
`sequenzo` (sequence analysis), `scikit-learn`.
- They are used only by the experiment scripts (`uv run --with ...`).
- None is an engine dependency.

## Questions

1. **More first-order factors:** does a repeatable first product, a first
   order on sale, the acquisition month, or research before buying change
   what happens next?
2. **"Still alive" models:** can buy-till-you-die models replace the fixed
   "reactivate after 90 days" rule with a moment per customer?
3. **One model for the second order:** with visits that change day by day,
   what drives the time to the second order?
4. **Path families:** are the 7 hand-made families natural clusters in the
   data?

## Method

```bash
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/new_factors.py {hm DIR ARTICLES | olist DIR | rees46 PARQUET}
uv run --no-project --with pymc-marketing --with pyarrow --with scikit-learn python -I \
  docs/experiments/scripts/btyd_reactivation.py {cdnow FILE | hm DIR}
uv run --no-project --with pandas --with pyarrow --with lifelines python -I \
  docs/experiments/scripts/cox_second_order.py {otto | rees46} EVENTS.parquet
uv run --no-project --with sequenzo --with pyarrow --with scikit-learn python -I \
  docs/experiments/scripts/path_clusters.py {hm DIR | cdnow FILE}
```

1. **New factors:**
   - **Repeatability of the first product** (H&M product type, Olist
     category): the re-buy rate measured on *other* customers. For H&M
     that's established customers; for Olist, the other half of a random
     split. So it can't leak the outcome.
   - **On sale:** price below 90% of the article's median price.
   - **Acquisition month or week.**
   - **Research before the first purchase** (REES46): visits, days, and
     comparison visits.
2. **"Still alive" models (BG/NBD and MBG/NBD):**
   - Fitted by maximum a posteriori (MAP) on a calibration period, checked
     on a holdout.
   - **CDNOW:** calibrate to Sep 30, 1997, then a 39-week holdout. This is
     the dataset of the original BG/NBD paper.
   - **H&M:** the new-customer cohort, calibrated to Oct 31, 2019, then a
     19-week holdout before the 2020 store closures.
   - **What we compare:** forecast error; ranking of who buys again (AUC)
     vs days since the last purchase and vs a fixed 90-day rule; and when
     P(alive) falls below 50% for customers with different rhythms.
   - BG/NBD lets customers drop out only after a *repeat* purchase. MBG/NBD
     also lets them drop out at the first one, which matters when most
     buy once.
3. **Cox model with time-varying visits** (lifelines):
   - One row per customer-day after the first order, until a second order
     or the end of follow-up: 28 days (REES46) or 14 (OTTO).
   - **Variables:** visits by kind over the previous 7 days, a visit in the
     last 2 days, and items in the first order.
   - 40,000 first-time buyers sampled per dataset.
   - Concept: **survival analysis**, Cox proportional hazards with
     time-varying covariates.
4. **Path clusters** (sequenzo, a Python port of R's TraMineR):
   - **Sequences:** 4,000 repeaters per dataset. Each month of the first
     12 is a state: no order, 1 order, 2+.
   - **Distance:** optimal matching (substitution costs from transition
     rates).
   - **Clustering:** Ward, with the number of clusters chosen by average
     silhouette.
   - **Comparison** with the rule-based families by adjusted Rand index.

## Results

### 1. New first-order factors

**H&M: how repeatable the first product is** (re-buy rate of its type among
established customers):

| Most repeatable types | Re-buy within 180 days | Least repeatable | Re-buy |
|---|---|---|---|
| Trousers | 62% | Hairband | 2% |
| Dress | 60% | Umbrella | 2% |
| T-shirt | 53% | Fine cosmetics | 3% |
| Blouse / top | 49% | Gloves / tie | 4% |

| First order's repeatability | Q1 (least) | Q2 | Q3 | Q4 | Q5 (most) |
|---|---|---|---|---|---|
| One order only | 38.9% | 35.1% | 35.4% | 34.1% | 38.7% |
| Loyal at 12 months | 11.9% | 12.3% | 12.3% | 13.0% | 11.5% |

**Olist, same measure on the other half:**

| First category's repeatability | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| Reorder within 180 days | 1.9% | 1.6% | 1.9% | 2.2% | **2.6%** |

The most repeatable categories are pet shop (3.4%) and bed/bath (3.1%); the
least are "cool stuff" gifts (0.6%) and electronics (1.0%).

**H&M: first order on sale:**

| First order | Share | One order only | Loyal |
|---|---|---|---|
| Full price | 70% | 35.6% | 11.8% |
| Partly on sale | 10% | **33.3%** | **13.9%** |
| Mostly on sale | 21% | **40.6%** | 12.7% |

**When they were acquired:**

| H&M first-order month | Mar | Apr | May | Jun | **Jul** | **Aug** | Sep | Oct | **Nov** |
|---|---|---|---|---|---|---|---|---|---|
| 2nd order within 90 days | 44.1% | 43.3% | 42.5% | 38.4% | 35.0% | 34.8% | 35.8% | 32.5% | 31.0% |
| Share of the first order on sale | 21% | 22% | 24% | 23% | **38%** | **41%** | 30% | 28% | **41%** |

At Olist, Black Friday week 2017 customers reordered at 1.6% vs 2.0% for
other weeks (n = 2,915).

**REES46: research before the first purchase:** 117,146 new buyers, 28.1%
order again within 21 days.
- By visits before the first purchase (1, 2–3, 4–7, 8+): 27.3–28.2%.
- By days from first visit to first purchase: 27.5–28.8%.
- By comparison visits before it: 27.1–28.4%.

No difference.

### 2. "Still alive" models vs a fixed rule

| | CDNOW | H&M |
|---|---|---|
| **BG/NBD parameters** (r, α, a, b) | 0.25, 4.79, 0.71, 2.13 | 0.46, 5.78, 1.67, 7.48 |
| **Published, CDNOW sample** (Fader, Hardie & Lee 2005) | 0.24, 4.41, 0.79, 2.43 | – |
| **Holdout repeat purchases:** actual / BG / MBG | 19,684 / 16,907 / 15,948 | 108,524 / 92,419 / 92,044 |
| **Error per customer (MAE):** MBG / own past rate / cohort mean | **0.79** / 1.04 / 1.17 | **0.73** / 0.87 / 0.96 |

**Who buys again in the holdout** (AUC; 0.5 is random):

| Ranking by | CDNOW | H&M |
|---|---|---|
| Expected purchases (MBG) | **0.765** | **0.741** |
| Days since the last purchase | 0.759 | 0.728 |
| P(alive), MBG | 0.701 | 0.701 |
| Fixed "lapsed after 90 days" rule | 0.673 | 0.675 |
| P(alive), BG/NBD | 0.32 | 0.36 |

**Weeks of silence until P(alive) < 50%** (MBG/NBD, H&M; CDNOW is similar):

| Customer | 1 repeat purchase | 2 | 4 | 8 |
|---|---|---|---|---|
| **One order only** | **8 weeks** (CDNOW: 7) | | | |
| Repeats every 2 weeks | 9 | 9 | 9 | 9 |
| Every 4 weeks | 10 | 11 | 12 | 13 |
| Every 8 weeks | 13 | 16 | 19 | 23 |
| Every 16 weeks | 18 | 25 | 33 | 42 |

### 3. Time to the second order (Cox, time-varying)

Hazard ratio per doubling of visits in the previous 7 days (95% CI):

| | OTTO (40,000 buyers, 10,851 second orders) | REES46 (40,000 buyers, 13,434) |
|---|---|---|
| Shallow visits | 1.08 [1.04, 1.11] | 1.07 [1.03, 1.11] |
| Focused visits (same product twice) | 1.19 [1.14, 1.23] | 1.13 [1.08, 1.18] |
| **Intent:** carts (OTTO), comparing 3+ in a category (REES46) | **1.89** [1.83, 1.97] | **1.38** [1.33, 1.43] |
| **Visited in the last 2 days** | **1.64** [1.55, 1.73] | **1.46** [1.38, 1.55] |
| Items in the first order (per doubling) | 1.17 [1.15, 1.20] | **1.89** [1.81, 1.96] |

### 4. Are the 7 families natural clusters?

| | CDNOW | H&M |
|---|---|---|
| Distinct 12-month sequences among 4,000 repeaters | 1,294 | 1,652 |
| Average silhouette: 2 clusters / 3 / 7 / 10 | **0.51** / 0.10 / 0.05 / 0.08 | **0.43** / −0.08 / −0.03 / 0.00 |
| Adjusted Rand vs the rule families (k = 3–8) | 0.17–0.27 | 0.10–0.25 |

**The two natural clusters:**

| | CDNOW share | H&M share | Active months | Still buying in months 10–12 | Rule families inside |
|---|---|---|---|---|---|
| **Engaged core** | 12% | 16% | 6.2–6.4 | 77–80% | Mostly G steady (61–64%), then F |
| **Light repeaters** | 88% | 84% | 1.9–2.1 | 31–35% | B, C, D, E mixed |

## Conclusion

1. **"Start with a repeatable product" doesn't hold up in fashion; there's
   a small effect in a marketplace.**
   - H&M's first-product repeatability shows no pattern (Q1 and Q5 look
     alike).
   - Olist's most repeatable first categories (pet, home textiles) reorder
     a bit more (2.6% vs 1.6–1.9%).
   - That fits the grocery-first logic: consumables create their own
     reason to return. But the effect is small, and is a weaker test than
     the grocery data the design calls for.
2. **Discounts and sale periods bring in more one-and-done customers.**
   - At H&M, a first order mostly on sale means 40.6% one-and-done, vs
     35.6% at full price. A *partly* on sale first order did best.
   - The months when most new customers bought on sale (Jul, Aug, Nov) have
     the lowest 90-day second-order rates. The decline runs through the
     whole year, though, so season and sales can't be separated here.
   - At Olist, Black Friday customers reorder slightly less.
   - This is consistent with research on deal-acquired customers (Lewis
     2006). Sale-acquired cohorts need a different, earlier
     second-order plan, not just more discounts.
3. **Research before the first purchase doesn't matter** (REES46). What
   counts is what happens *after* it.
4. **"Still alive" models beat the fixed 90-day rule, but only slightly
   beat simple recency.**
   - Our BG/NBD fit reproduces the published CDNOW parameters, which
     validates the setup.
   - **Use MBG/NBD, not BG/NBD, where many buy once.** BG/NBD calls every
     one-order customer "alive", so its P(alive) ranks *worse than
     random* (AUC 0.32–0.36).
   - Ranking who buys again: expected purchases from the model (0.74–0.77)
     ≈ days since the last purchase (0.73–0.76), both clearly above a fixed
     90-day rule (0.67–0.68).
   - Forecasts are 16–24% more accurate per customer than the customer's
     own past rate. In total, both datasets come in 14–19% short, because
     the holdout includes holiday seasons the model doesn't know about.
   - P(alive) is miscalibrated for frequent buyers who go quiet: they come
     back more often than it expects.
   - **What the model adds is a per-customer reactivation moment:**
     - one-order customers look lapsed after about **7–8 weeks**, in line
       with the 60-day cliff;
     - a fortnightly buyer after about 9 weeks;
     - someone buying every 4 months with a long history after about 40
       weeks.

     A single "90 days" is too late for the first group and far too early
     for the last.
5. **The Cox model confirms the landmark results, with one rule: intent
   plus recency.**
   - In both stores, intent visits raise the daily chance of a second
     order most: carts 1.89×, comparisons 1.38× per doubling.
   - A visit in the last 2 days adds 1.5–1.6×.
   - Plain visits add only 7–8%.
   - In electronics (REES46), a multi-item first order is the strongest
     fixed signal (1.89× per doubling).
6. **The 7 families are useful labels, not natural clusters.**
   - The data supports only one clean split: a small **engaged core**
     (12–16% of repeaters: about 6 active months, 80% still buying at the
     end) vs **light repeaters** (84–88%: about 2 active months).
   - Within the light majority, paths form a continuum: silhouette near 0
     beyond 2 clusters, and agreement with our families is low (adjusted
     Rand 0.1–0.27).
   - So keep the families for *decisions* (when to act). Don't read them
     as distinct customer types. The model-based numbers (expected
     purchases, P(alive)) are the better continuous measure.

**What changes in LIFECYCLE / SHOPPER-PROFILE:**
- **Lapsing is per customer:** MBG/NBD P(alive), or expected purchases,
  replaces the fixed 90-day reactivation rule (once a store has a few months
  of data). Until then, use the larger of about 8 weeks and 2–3× the
  customer's own gap between orders (7–8 weeks after a first order).
- **Sale-acquired new customers** get a second-order plan without leaning
  on another discount.
- **Intent and recency since the last order** stay the main signals for
  next-order timing.

**Caveats:**
- **Observational.** The sale, season and repeatability effects are
  associations. Acquisition months also differ in burn-in length and in
  which seasons their windows cover.
- **MAP fits, not full Bayesian posteriors.** Forecasts are stationary, so
  the models know nothing about seasons or holidays.
- **Cox:** proportional hazards weren't formally tested. The sample is
  40,000 buyers, and the windows are short (14–28 days).
- **Clustering:** done on samples of 4,000 repeaters with one distance
  (optimal matching with transition-rate costs) and Ward linkage, which
  assumes Euclidean distances (sequenzo warns about this). Other choices
  could split the light majority differently.
