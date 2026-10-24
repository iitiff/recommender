# 2026-10-24: Session candidate sources on RetailRocket, split by journey

**Status:** done
**Model(s):** `recently_viewed@a8b25887`, `covisitation@bfe5197e`, `trending@d50c090f`
vs `most_popular@f290e0b6` (all default params). Random is analytic (below).
**Data:** RetailRocket prepared (CC BY-NC-SA 4.0), split fingerprint
`29d3d0e9df12`. Train: 2,480,490 events up to 2015-09-02 17:49 UTC. Test: the
last ~16 days, 6,440 add-to-carts by 3,950 shoppers. 235,061 items.

## Question and hypotheses

Step 2a of the roadmap: build single-job candidate sources and see **where**
each one works before blending them (2b). Hypotheses, stated before running:

1. Co-visitation beats MostPopular overall on the next add-to-cart.
2. No single source wins every segment. Recently viewed should win when
   shoppers return to something they saw, co-visitation on new items, and
   popularity/trending on shoppers with no history.
3. Filtering to items in stock *at that moment* helps the sources that
   suggest new items.

**Metrics:** Hit rate@20 (equal to Recall@20 here: one target per case) and
MRR@20, paired bootstrap vs MostPopular. Descriptive: new-category share@20,
average list length.

## Method

```bash
uv run recsys evaluate data/retailrocket/catalog data/retailrocket/interactions.csv \
  --split next --test-fraction 0.1 --k 5 --k 20 --group-facet department \
  --models most_popular,trending,recently_viewed,covisitation \
  [--availability data/retailrocket/availability.csv]
```

- **One case per add-to-cart** after the cutoff (the same item carted twice in
  one visit counts once). The context is everything the shopper did *before*
  that moment, including earlier in the same visit. Models are fitted once at
  the cutoff (no retraining during the test window; that's step 3).
- **Segments** are computed from the shopper's own history at that moment:
  - lifecycle stage (`recsys.lifecycle`): an order = a day with a purchase,
    and more than 10 purchased items = bulk buyer;
  - what they carted next, relative to what they'd done before;
  - for buyers, whether the category was one they'd already bought from.
  - "Category" = the top-level department of RetailRocket's category tree
    (25 departments; 21% of items have no category).
- **Stock:** RetailRocket snapshots an `available` flag per item roughly
  weekly. With `--availability`, each case only allows items not marked sold
  out at that time.
- **Random** isn't run: over 235k items its expected hit rate@20 is about
  20 / 235,061 ≈ 0.0001, and below 0.001 with the stock filter.

## Results

**Overall, with stock filter** (95% CIs over the 6,440 cases)

| Model | Hit rate@20 | MRR@20 | Δ Hit rate vs MostPopular | Avg list length | New-category share | Fit | ms/request |
|---|---|---|---|---|---|---|---|
| most_popular | 0.022 [0.018, 0.025] | 0.005 | – | 20.0 | 0.69 | 4 s | 0.8 |
| trending (7-day half-life) | 0.024 [0.020, 0.027] | 0.009 | +0.002 [−0.000, +0.004] | 20.0 | 0.67 | 6 s | 0.9 |
| recently_viewed | **0.630** [0.617, 0.643] | **0.513** | +0.608 ✔ | 5.4 | 0.00 | 0 s | 0.03 |
| covisitation | 0.143 [0.135, 0.152] | 0.048 | +0.121 ✔ | 16.0 | 0.13 | 9 s | 0.3 |

Without the stock filter: most_popular 0.016, trending 0.019, recently_viewed
0.671, covisitation 0.130.

**By what the shopper carted** (hit rate@20, stock filter on)

| Next add-to-cart was… | Cases | most_popular | trending | recently_viewed | covisitation |
|---|---|---|---|---|---|
| No history at all (first tracked event) | 714 (11%) | 0.027 | **0.041** | 0 | 0 |
| An item seen earlier **this visit** | 4,060 (63%) | 0.023 | 0.022 | **0.937** | 0.138 |
| An item seen in an earlier visit | 201 (3%) | 0.030 | 0.025 | **0.881** | 0.219 |
| A re-buy | 73 (1%) | 0.110 | 0.082 | **0.973** | 0.205 |
| A new item in a category they'd browsed | 1,286 (20%) | 0.009 | 0.013 | 0 | **0.234** |
| A new item in a new category | 104 (2%) | 0.019 | **0.048** | 0 | 0 |

**By lifecycle stage** (hit rate@20, stock filter on)

| Stage at the time | Cases | most_popular | trending | recently_viewed | covisitation |
|---|---|---|---|---|---|
| New visitor (first visit) | 3,923 (61%) | 0.020 | 0.024 | 0.577 | 0.133 |
| Returning, no order yet | 1,606 (25%) | 0.023 | 0.021 | 0.658 | 0.198 |
| 1 order | 291 (5%) | 0.024 | 0.024 | 0.711 | 0.175 |
| 2+ orders | 69 (1%) | 0.058 | 0.058 | 0.768 | 0.087 |
| Bulk buyer (>10 items) | 551 (9%) | 0.024 | 0.025 | 0.857 | 0.044 |

Buyers who carted from a category they had never bought from (125 cases) had
already *viewed* that exact item 70% of the time (recently_viewed 0.704).

Full reports, including all CIs and the "vs past orders" table, come from the
command above.

## Conclusion

1. **Hypothesis 1 holds:** co-visitation beats MostPopular by +0.12 hit
   rate@20 [+0.11, +0.13] (14.3% vs 2.2%). In a 235k-item catalog, popularity
   barely works: the top 20 is a tiny slice of what people buy.
2. **Hypothesis 2 holds, and the segments change the story.**
   - **Most "predictable" carts are the page the shopper is already on.** 63%
     of add-to-carts are an item viewed earlier in the same visit, and 61%
     are *the very last item viewed*. Recently viewed "predicts" these almost
     perfectly, but showing a shopper the product they're looking at isn't a
     recommendation that changed anything. This is **repeat consumption**
     inflating session-based metrics, a known trap. Recently viewed is still
     valuable where the shopper has *left*: in a later visit (88%) and for
     re-buys (97%), which is exactly what re-engagement messages need
     (abandoned browse and cart).
   - **The recommender's real job is the other third: items the shopper
     hasn't seen.** There co-visitation finds 23% of carts in categories they
     browsed (vs 1% for popularity), but nothing finds new-category carts
     (trending 5%, n=104).
   - **Cold start is real:** 11% of carts have no history at all, and only
     the non-personalized sources can serve them.
   - **Lifecycle matters, but not as hoped.** Co-visitation works best for
     returning shoppers and first-time buyers (17–20%) and poorly for repeat
     and bulk buyers (4–9%), who mostly re-cart things they already know.
     Repeat customers want their items back, not discovery.
3. **Hypothesis 3 holds:** filtering to in-stock items lifts co-visitation
   (13.0% → 14.3%) and popularity (1.6% → 2.2%), because they stop wasting
   slots on sold-out items. It costs recently_viewed 4 points: 5.9% of carts
   are on items the weekly snapshot marks sold out (the flag is stale), and
   those become impossible hits. Keep the filter (recommending sold-out items
   is a bad experience), and remember it slightly understates accuracy.
4. **Category expansion follows browsing:** buyers who moved into a new
   category had usually looked at the item first. Expansion intent shows up
   in browsing before it shows up in orders, so browsing can trigger the
   "new category" job.

**What this means for 2b (the blender):**
- **Report two headline numbers, not one:** *discovery* hit rate (carts of
  items not seen before, the third where recommendations can add value) and
  *return* hit rate (items seen before). A single blended number is dominated
  by the page-the-shopper-is-on effect.
- **Blend by journey:**
  - recently viewed for returning shoppers, buyers and re-buys;
  - co-visitation for an active visit;
  - trending for cold start;
  - an explicit adjacent-category source, because nothing reaches new
    categories today.
- Weights start as rules by stage and then get learned per shopper.
  Repeat/bulk buyers need re-buy and replenishment more than discovery.

**Caveats:**
- RetailRocket's visitor ids are per-device cookies. Some "new visitors" and
  "no history" cases are returning shoppers on a new device.
- Models are frozen at the cutoff for ~16 days. Daily retraining (step 3)
  will help trending and co-visitation most.
- Default parameters, untuned. Tuning comes after the blender exists, on a
  separate validation window, so the test window isn't overfitted.
- Offline replay measures predicting what shoppers did *without* these
  recommendations. Only an online test measures what recommendations
  *cause*.
