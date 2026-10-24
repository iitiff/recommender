# 2026-10-24: Shopper price level: per-type bands plus a general level (REES46)

**Status:** done (descriptive, observational)
**Data:** REES46 multi-category store, Oct–Nov 2019 ("© Original Authors";
aggregates only). 1.12M purchases with a price position; 11,705 bulk buyers
removed.

## Why this experiment

`2026-10-24-product-type-and-price.md` concluded that price is "mostly a
property of the moment, not of the person". The evidence was weak: it
correlated one purchase in one category with one purchase in another
(Spearman 0.21).

The owner's hypothesis:
1. People buy in **different price bands per product type**.
2. There is also a **general price level** (income-like), visible only across
   their **collective order history**.

One-vs-one purchases can't detect a weak general signal buried in
per-purchase noise. This experiment tests the hypothesis directly.

## Method

```bash
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/price_level.py events.parquet   # ~2.5 min, <12 GB RAM
```

**Price position** = a product's price percentile within its leaf category
(≥20 products), so "premium" means the same for kettles and phones.

1. **Collective history:** predict each purchase's price position from the
   mean over the shopper's other purchases (or views) in *other* leaf
   categories. Excluding the purchase's own category keeps a single shopping
   mission from predicting itself.
2. **Month to month:** a shopper's mean price position over October views vs
   November views. Computed for all categories, and for **only categories
   not viewed in October**.
3. **Decomposition** (shoppers with ≥5 products viewed in two top-level
   types): a type-level mean splits into
   - a general level: what carries across types, measured as the covariance
     between the two types;
   - a type-specific part;
   - sampling noise (within-type variance / n).
4. **New category:** November purchases in a leaf category the shopper never
   viewed in October, by their October price level (≥10 products viewed).
   This is a real forecast: past to future, into unfamiliar territory.

The decomposition and correlations were smoke-tested on random data, where
they correctly attribute everything to noise.

## Results

**1. More history, clearer level** (Spearman: purchase's price position vs mean of the history in other categories)

| History in other categories | Other purchases | | Products viewed | |
|---|---|---|---|---|
| | n | Spearman | n | Spearman |
| Thin (1 purchase / 1–4 views) | 216,826 | 0.20 | 228,973 | 0.17 |
| Some (2–3 / 5–19) | 166,576 | 0.23 | 321,298 | 0.20 |
| More (4–9 / 20–99) | 81,050 | 0.25 | 306,275 | 0.24 |
| Rich (– / 100+) | | | 55,276 | **0.30** |

**2. Month to month** (Spearman of October vs November mean price position, views)

| Products viewed each month | All categories | Only categories new in November |
|---|---|---|
| 5–19 | 0.41 (n=222,304) | 0.19 (n=174,203) |
| 20–99 | 0.51 (n=80,864) | 0.34 (n=47,856) |
| 100+ | 0.69 (n=2,778) | **0.49** (n=701) |

**3. What a shopper's price position in one product type is made of** (361,831 shoppers)

| Part | Share |
|---|---|
| General level (carries across types) | **14%** |
| Specific to this shopper in this type | **74%** |
| Sampling noise | 12% |

**4. First purchase in a new category, by October price level**

| October price level (tercile) | Purchases | Budget | Mid | Premium |
|---|---|---|---|---|
| Low | 24,363 | **50%** | 34% | 16% |
| Mid | 23,419 | 43% | 36% | 21% |
| High | 20,279 | 36% | 35% | **29%** |

Spearman 0.16 across all of them.

## Conclusion

**The owner's hypothesis holds on both counts; the earlier conclusion was wrong.**

1. **Per-type bands dominate.** About three quarters of a shopper's price
   position in a product type is specific to that type: the premium-phone,
   budget-kettle shopper is common.
2. **A general level is real, and it takes collective history to see it.**
   The correlation grows with evidence: from 0.17 with a few views to 0.30
   with 100+. In categories the shopper had *never browsed*, month to month
   it reaches 0.49 for well-observed shoppers. And it forecasts: shoppers
   with a high October level bought premium in a new category 29% of the
   time vs 16% for a low level (1.8×).
3. **Why it hides in small samples:** each purchase is mostly type-specific
   band plus noise. Averaging many purchases across types cancels both and
   leaves the general level. Concept: measurement reliability (the
   **Spearman–Brown** effect).
4. **What *is* about the moment:** discount chasing
   (`2026-10-24-shopper-traits-stability.md`) and the price band of the
   current comparison. So there are three different price signals:

| Signal | Scope | Evidence |
|---|---|---|
| Price band around the item being viewed (±30%) | this visit | alternatives compared are 2× closer in price than chance |
| Shopper's band in this product type | this type, over time | 74% of type-level price position |
| Shopper's general price level | across types, needs history | 14% of a type's position; forecasts new-category tier (1.8× premium) |

**Design: a hierarchical price profile.**
- Estimate each shopper's band per product type, shrunk toward their general
  level when that type has little data, and the general level shrunk toward
  the store average when history is thin. Concepts: **hierarchical (multilevel)
  model**, **partial pooling**, **empirical Bayes shrinkage**.
- With rich history in a type, its own band wins. In a new type, the general
  level fills in.
- It is exactly the structure that supports category expansion: when we
  introduce a new category, we know which price tier to lead with.

**Guardrails:**
- We model **observed price level**, never income, and never label shoppers
  by it.
- It's a soft ranking prior. It never filters, never hides other tiers
  (premium-level shoppers still buy budget 36% of the time), and never
  changes the price a shopper is charged.
- Using it to price-discriminate would be a different product, with legal
  and trust consequences.

**Caveats:**
- "Type-specific" may include the current mission in that type (one
  ongoing phone search), not only a lasting per-type preference. Two months
  can't separate them; Amazon Reviews (years) can.
- The 100+ products groups are small and unusually active (n=701 for new
  categories).
- One store, electronics-heavy.
- Views count as history. A shopper's own browsing partly reflects what the
  store showed them.
