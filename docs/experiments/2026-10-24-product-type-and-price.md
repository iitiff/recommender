# 2026-10-24: How behaviour differs by product type and price tier

**Status:** done (descriptive, observational). **Corrected in part** by
`2026-10-24-shopper-price-level.md`.

> **Correction (same day):** conclusion 3 ("price is mostly a property of the
> moment, not of the person") was wrong. It rested on correlating one
> purchase with one other purchase, which is too noisy to reveal a general
> level. Averaged over a shopper's collective history, a general price level
> is real: it carries month to month (up to 0.49 in categories never browsed
> before) and forecasts the tier of a first purchase in a new category
> (premium 29% vs 16%). Per-type bands still dominate (74% of a type-level
> price position).

**Data:** REES46 multi-category store, Oct–Nov 2019 ("© Original Authors";
aggregates only). 107M events, 204k products, 1.23M purchases after removing
11,700 bulk buyers (>10 items). 73% of purchases have a readable category.
RetailRocket can't be used here: its prices are hashed and its categories are
anonymous ids.

## Question

The stage deep dive found traits of *shoppers* (momentum, style). Does
behaviour also depend on *what* they're shopping for (product type) and *how
expensive it is* (price tier)? And is price a trait of the shopper, of the
category, or of the moment?

## Method

```bash
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/stream_rees46_events.py events.parquet     # ~15 min, 1.5 GB
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/product_price_behaviour.py events.parquet  # ~4 min, <11 GB RAM
```

- **Product type:** REES46's category path at two levels (`electronics`,
  `electronics.smartphone`).
- **Price tier:** a product's price percentile **within its leaf category**,
  in thirds: budget, mid, premium. A $200 phone is budget and a $200 kettle is
  premium, so only relative position compares across types.
- **Order:** (shopper, product, day), deduplicated.
- **Consideration:** the 30 days before a purchase, so only purchases from
  day 30 on. Counts views of the item, alternatives viewed in the same leaf
  category, visits, and days from first look.
- **Carts:** bought within 7 days.
- **After purchase:** a 30-day horizon, so only purchases before day 31.

## Results

### 1. Product type: how long and how widely people shop

| Type | Purchases | Median price | Alternatives viewed (median / mean) | Visits (median) | Bought in one visit | Days first look → buy (median) | Cart → bought ≤7d |
|---|---|---|---|---|---|---|---|
| Smartphones | 275,169 | $246 | 3 / 5.9 | 3 | 29% | **2.6** | 43% (electronics) |
| Kids' carriages | 2,172 | $104 | **5 / 9.8** | 3 | 33% | 0.7 | 32% (kids) |
| Furniture, living room | 4,070 | $283 | **5 / 9.2** | 2 | 33% | 0.7 | 29% (furniture) |
| Shoes | 9,755 | $83 | **6 / 12.2** | 2 | 39% | 0.2 | **25%** (apparel) |
| Notebooks | 13,897 | $421 | 3 / 7.9 | 2 | 40% | 0.2 | 33% (computers) |
| Kitchen appliances | 51,326 | $180 | 3 / 5.5 | 2 | 41% | 0.09 | 36% (appliances) |
| Irons | 4,129 | $38 | 2 / 3.8 | **1** | **52%** | **0.01** | |
| Personal appliances | 2,795 | $21 | 2 / 4.3 | 2 | 49% | 0.01 | |
| Landline phones | 3,611 | $26 | 2 / 4.7 | 2 | 49% | 0.01 | |

(Cart conversion is by top-level type; the rest by second-level type.)

### 2. Price tier, *within* a type: premium means slower, not wider

| Top-level type | Days first look → buy: budget / mid / premium | Alternatives viewed (median): budget / mid / premium |
|---|---|---|
| Electronics | 0.83 / 1.64 / **3.28** | 2 / 3 / 2 |
| Computers | 0.09 / 0.65 / **0.90** | 2 / 3 / 3 |
| Apparel | 0.05 / 0.25 / 0.15 | 4 / 7 / 7 |
| Appliances | 0.04 / 0.10 / 0.13 | 2 / 3 / 3 |
| Furniture | 0.51 / 0.58 / 0.56 | 3 / 4 / 5 |
| All types | 0.37 / 0.84 / 1.57 | 2 / 3 / 2 |

Cart → bought within 7 days barely moves with tier (budget 37%, mid 39%,
premium 39%) but swings by type (apparel 25% to electronics 43%).

### 3. After a purchase (30 days)

| | Any purchase | Same type again | Same product again |
|---|---|---|---|
| Smartphones | 41% | 35% | 14% |
| Kitchen appliances | 38% | 20% | 7% |
| Irons / personal appliances | 38–39% | 11% | 5–6% |
| Shoes | 45% | 21% | 7% |
| Budget / mid / premium (all types) | 39% / 40% / 45% | 23% / 27% / 34% | 9% / 11% / 14% |

When the next purchase is a *different* type, it is mostly a smartphone
(11–30% of next purchases from any type). That reflects what this store sells
most, not a complement relationship. Real complement pairs are weak, e.g.
components → peripherals 7%.

### 4. Is price a shopper trait?

| Test | Result |
|---|---|
| Same shopper's price percentile, viewing two different types (n=361,835) | Spearman **0.13** |
| Same shopper's price percentile, buying in two different leaf categories (n=81,072) | Spearman **0.21** |
| Alternatives compared before a purchase, within ±20% of the bought price | **35%** vs 19% for random viewed products of the same category |
| Median price gap, alternatives vs bought item | **±34%** vs ±86% for random viewed products |
| Bought item's percentile minus the alternatives' average | −0.01 (52% bought cheaper): no systematic trade-down |

## Conclusion

1. **Product type decides the shape of the decision.** Two different axes
   show up:
   - **Deliberation time**: high-involvement, spec-driven products
     (smartphones, premium electronics) take days and several visits.
     Low-price utility items (irons, personal appliances) are decided in one
     visit, in minutes.
   - **Comparison breadth**: style-driven products (shoes, furniture,
     strollers) get 5–6 alternatives in the median (9–12 on average), even
     when decided the same day.

   Concepts: high vs low **involvement** purchases, **considered** vs
   **impulse** buying.
2. **Price tier slows decisions but doesn't widen them.** Within electronics,
   premium items take 4× longer than budget ones (3.3 vs 0.8 days), with the
   same number of alternatives. Tier hardly matters for cart conversion; type
   does (apparel carts convert least, likely size and fit).
3. **~~Price is mostly a property of the moment, not of the person.~~** (See
   the correction above: a general level exists but needs collective history.)
   - A shopper's price position carries over only weakly from *one* purchase
     to *one* other category (0.13–0.21): the premium-phone buyer often buys a
     budget kettle.
   - Within one decision, shoppers compare in a **price band** about twice
     as tight as chance (±34% vs ±86%).
   - The useful signal is "near the price of what they're looking at", not
     "this is a premium shopper".
4. **Repeat behaviour differs by type, but this store can't show
   replenishment.** It sells durables. Same-type repeat ranges from 11%
   (irons) to 35% (smartphones). The high same-phone re-purchase (14% within
   30 days) is more likely cancellations and re-orders, or buying for family,
   than true repeat demand.

**What this means for the engine:**
- **Product context is a fifth layer** alongside relationship, momentum,
  style and the current visit. Each product type needs its own
  deliberation-time and comparison-breadth profile, *learned per store*
  from its own data (the same logic as timing in LIFECYCLE.md).
- **Recommendations by context:**
  - **High-involvement** (smartphones, premium): keep "recently compared"
    across visits; send re-engagement after days, not hours.
  - **Low-involvement** (irons, small appliances): win within the visit; an
    abandoned-cart nudge must come within hours.
  - **Style-driven** (shoes, furniture): show many visually varied
    alternatives (diversity re-ranking).
- **Price-banded substitutes:** alternatives to the item being viewed should
  sit within roughly ±30% of its price. This is a re-ranking signal around
  an anchor, not a global budget filter.
- **Convention to revisit:** CLAUDE.md says "Price is never a taste feature."
  Superseded by the proposal in `2026-10-24-shopper-price-level.md`: three
  price signals (anchor band, per-type band, general level) as soft ranking
  priors; budget stays a hard filter. Pending the owner's decision.

**Caveats:**
- One store, electronics-heavy (smartphones are 39% of the purchases analyzed), two months.
- 27% of purchases have no category, so they're missing from type tables.
- REES46's sessions are its own ids. Some "visits" may be long tabs.
- Observational: "premium takes longer" may partly be "people who buy premium
  phones are different people".
- Large n makes CIs tiny, so these differences are not noise. But they can
  still be confounded.
- **Replicate before acting:** the REES46 cosmetics store (same authors) would
  test replenishment and a consumables-heavy mix.
