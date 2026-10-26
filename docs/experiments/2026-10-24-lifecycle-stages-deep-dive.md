# 2026-10-24: Lifecycle stages up close: what's inside each stage?

**Status:** done (descriptive, observational)
**Data:** RetailRocket prepared (CC BY-NC-SA 4.0): 1.30M visits analyzed.
REES46 Oct–Nov 2019 ("© Original Authors"; aggregates only): 11.3M visits by
3.2M shoppers, visit-level only (no item data in this pass).

## Question

The five stages used so far (new visitor, returning without an order,
1 order, 2+ orders, bulk) gave a first split. Are they too wide? A stage is
too wide if shoppers inside it **behave differently** (how likely they are to
buy soon) or **need different recommendations** (what they cart next, which
source finds it).

## Method

```bash
python -I docs/experiments/scripts/stage_deep_dive.py retailrocket data/retailrocket
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/stage_deep_dive.py rees46 sessions.parquet   # from visits_by_cohort.py
uv run python docs/experiments/scripts/stage_sources.py data/retailrocket
```

**Lens 1, behaviour (both datasets).** The unit is a visit. The shopper's
state is computed from *earlier* visits only, i.e. what a live system would
know at that moment. The outcome is **ordered within 7 days** (this visit or
later), with "bought this visit" and "came back within 7 days" alongside.
- The first 30 days are burn-in: history only, not analyzed, so fewer
  long-time customers pass as new.
- The last 7 days are excluded so every visit has a full horizon.
- Stages as before, except **"returning, no order" is split into browsers
  (never carted) and abandoners (carted, never bought)**.

**Lens 2, recommendation needs (RetailRocket).** These are the 6,440
next-add-to-cart cases from `2026-10-24-session-sources-retailrocket.md`
(stock filter on), re-segmented with the finer state. For each segment: the
mix of what shoppers carted next, and each source's hit rate@20.

## Results

### 1. The stages' averages (ordered within 7 days)

| Stage | RetailRocket: share of visits | RetailRocket: ordered ≤7d | REES46: share of visits | REES46: ordered ≤7d |
|---|---|---|---|---|
| New (first visit) | 79.0% | 0.7% | 17.0% | 7.6% |
| Browser (returned, never carted) | 18.4% | 1.7% | 48.0% | 9.7% |
| **Abandoner** (carted, never bought) | 1.3% | **11.7%** | 14.3% | **22.2%** |
| 1 order | 0.7% | 12.9% | 12.4% | 28.5% |
| 2 orders | 0.1% | 39.3% | 3.9% | 38.1% |
| 3+ orders | 0.1% | 38.6% | 3.1% | 52.7% |
| Bulk (>10 items) | 0.2% | 82.6% | 1.3% | 79.1% |

The old "returning, no order" stage hid a 2–7× difference: **an abandoner is
nearly as likely to order soon as a first-time buyer**, and far more likely
than a browser.

### 2. Inside each stage: the splits that matter (ordered within 7 days)

| Within stage… | Split | RetailRocket | REES46 |
|---|---|---|---|
| Abandoner | cart <1 day old vs older | **22.0% vs 6.7%** | **30.5% vs 15.9%** |
| Abandoner | bought this visit: 1–2 prior visits vs 16+ | 10.6% vs 0.9% | 11.6% vs 2.9% |
| Browser | back within 1 h vs after >30 days | 3.1% vs 0.9% | 9.0% vs 8.1% |
| Browser | 16+ visits, never carted: bought this visit | 0.1% | 0.8% |
| 1 order | last order <1 day ago vs 1–30 days | 19.1% vs ~11% | 40.8% vs ~24.5% |
| 1 order | open cart <1 day old vs none | 41.8% vs 11.6% | 41.7% vs 27.2% |
| 2 orders | 1–2 prior visits vs 16+ | (n<30) | **62.1% vs 32.1%** |
| 3+ orders | last order <1 day vs >30 days | 55.9% vs 27.4% | 63.5% vs 36.0% |

Three dimensions keep splitting stages 2–3× in both datasets:
- **Momentum:** how fresh the last signal is (open cart, last order, last visit).
- **Shopping style:** few visits per purchase ("decisive") vs many ("researcher").
- **Open cart:** an unfinished cart.

Days since the last order *beyond* the first day barely mattered within
REES46's short window: 1–7, 7–30 and >30 days all sit around 24% for
one-order customers.

### 3. What each finer segment carts next (RetailRocket, lens 2)

| Segment | Cases | Seen this visit | Seen in an earlier visit | Re-buy | New item, browsed category | Best source (hit rate@20) |
|---|---|---|---|---|---|---|
| No history at all (cart is their first event) | 714 | – | – | – | – | trending 4% |
| Any, 1 event before the cart | 2,054 | **82%** | 1% | 1% | 16% | recently viewed 78%; co-visitation only 8% |
| Any, 2–4 events before the cart | 1,642 | 74% | 1% | 1% | 23% | co-visitation **24%** (its best) |
| Browser back within 1 day | 511 | 63% | **25%** | 0% | 11% | recently viewed 81% |
| **Abandoner** | 557 | 43% | 7% | 0% | **45%** | recently viewed only 48%; co-visitation 19% |
| 1 order | 291 | 55% | 3% | **16%** | 22% | recently viewed 71%, co-visitation 18% |
| 1 order, back after >7 days | 61 | 44% | 3% | 13% | **34%** | co-visitation 26% |
| Bulk buyer | 551 | **89%** | 1% | 3% | 6% | recently viewed 86%; co-visitation 4% |
| Exploring 2+ departments this visit | 813 | 77% | 1% | 3% | 19% | co-visitation drops to 7% |

## Conclusion

**Yes, the stages were too wide.** Four concrete changes, each backed by
both datasets or by the recommendation lens:

1. **Abandoners are their own stage.** They're as close to buying as
   first-time buyers. They also **comparison-shop**: half their carts are
   *new* items in categories they browsed (45%), versus a fifth overall.
   They need "alternatives to what's in your cart", not just a reminder of it.
2. **Momentum modifies every stage.** A fresh open cart, an order in the last
   day, or a quick return raises near-term ordering
   compared with the same stage gone cold (about 1.5–3×). "1 order, bought yesterday" and
   "1 order, a month ago" are different moments.
3. **Shopping style is a stable trait.** Decisive shoppers (few visits per
   purchase) convert 4–12× more per visit than researchers in the same stage.
   Researchers need help *deciding* (comparisons, "people chose this"), not
   more options. Window shoppers (16+ visits, never carted) almost never buy,
   and some may be bots; don't spend messages on them.
4. **The current visit decides the immediate need.**
   - After one view, the shopper is on a product page: 82% cart that item, so
     the useful slots are alternatives and complements.
   - After 2–4 views, "viewed together" works best.
   - A shopper exploring several departments defeats co-visitation; they need
     something else, e.g. the best of each department they touched.
   - Quick returners want what they saw last time (a quarter of their carts).
   - Bulk buyers re-cart what they know (89%): a reorder list, not discovery.

**The broader lesson: segments for people, features for models.** Multiply
seven stages by momentum, style and in-visit state and you get more than a
hundred cells, most too small to learn from. So two tracks:
- **For humans** (dashboards, message strategy, the first rule-based
  blender): a small taxonomy of stage plus momentum plus open cart.
- **For models:** the underlying *features* directly (visits, recency,
  orders, days since order, open cart age, visits per order, visit depth,
  departments this visit). A learned blender or ranker then personalizes
  continuously; the industry calls this a "segment of one". That is the
  honest answer to "no one-size-fits-all".

**Caveats:**
- RetailRocket's per-device cookies make 79% of visits look like first
  visits and keep buyer segments small (n=12–125 per cell in lens 2).
  REES46 has stable ids but no item-level lens yet.
- 30-day burn-in reduces but doesn't remove left-censoring.
- REES46's window is 61 days, so "lapsed for months" can't be studied there.
  H&M or Olist would be needed for that (purchases only).
- Everything here is association. "Open cart roughly doubles ordering" means
  shoppers with open carts are closer to buying, not that creating carts
  causes orders.
- "Visits per purchase" is partly mechanical for buy-this-visit (more visits,
  fewer purchases per visit). The 7-day ordering gap (62% vs 32%) is the
  less circular evidence.
