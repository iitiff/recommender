# Shopper profile: understanding a shopper before recommending

What we know about a shopper at the moment of a request, how sure we can
be, and which recommendation decision each piece drives. Every dimension
here was tested on public data (RetailRocket, REES46, Amazon Reviews 2023)
before being kept. This is the spec for the `ShopperState` features in
roadmap step 2b and for the lifecycle messaging in step 4 (docs/LIFECYCLE.md).

## Principles

1. **The current mission first, the person second.** What someone is doing
   now (this visit, an open cart, an order yesterday) moved near-term buying
   1.5–3×. Lasting traits correlate only 0.15–0.45 from one period to the next.
   Traits are priors that tilt decisions; the mission makes them.
2. **A trait must earn its place with two tests.**
   - **Stable:** measured in one period, it still describes the shopper in
     the next (test-retest).
   - **Decision-changing:** knowing it changes what we show or send.

   Several plausible traits failed the first test (see "Rejected").
3. **Computed from events, never stored as truth.** A stored "loyal" or
   "premium" flag goes stale. The profile is recomputed from behaviour at
   request time, with older evidence fading.
4. **Partial pooling when evidence is thin.** A shopper's estimate for a
   product type leans on their overall estimate, and that leans on the store
   average, in proportion to how little we know. Concepts: **hierarchical
   model**, **empirical Bayes shrinkage**.
5. **Learn the numbers per store.** Timing, half-lives, thresholds and
   adjacency differ by an order of magnitude between stores. The engine ships
   the structure; each tenant's data sets the values.
6. **Replicate before acting.** A pattern from one dataset is a hypothesis.
   Category expansion → loyalty looked 2× in RetailRocket and vanished in
   three other stores.

## The six layers

| Layer | Changes | What it captures |
|---|---|---|
| 1. Current visit | every click | what they're looking at right now |
| 2. Momentum | hours to days | how close they are to buying |
| 3. Relationship (stage) | weeks to months | where they are in the lifecycle |
| 4. Style traits | months to years | how they shop |
| 5. Price profile | months to years | which price tier fits them, per product type |
| 6. Product context | per item (store-level) | how this kind of product is bought |

### 1. Current visit

| Signal | Definition | Evidence | Drives |
|---|---|---|---|
| Visit depth | events earlier in this visit | After 1 view, 82% cart *that* item; co-visitation works best at 2–4 views (24% hit rate) (RR) | Product page: alternatives and complements, not the item itself |
| Visit focus | departments touched this visit | Exploring 2+ departments: co-visitation drops from 18% to 7% (RR) | Exploring: best of each department touched |
| Anchor item and its price | the item on screen | Alternatives compared before buying sit within ±34% of the bought price, vs ±86% by chance (REES46) | Price-banded alternatives (soft re-rank) |
| Recently viewed | items touched before, newest first | 63% of carts are an item seen earlier in the visit; quick returners re-cart last visit's items (25%) (RR) | "Continue where you left off", abandoned-browse messages |

### 2. Momentum

| Signal | Definition | Evidence (ordered within 7 days) | Drives |
|---|---|---|---|
| Open cart age | hours since a cart-without-purchase, after the last order | Abandoner with cart <1 day: 22% vs 6.7% older (RR); 30.5% vs 15.9% (REES46) | Abandoned-cart timing; substitutes for the carted item |
| Hours since last order | | 1-order customers <1 day after ordering: 19% vs ~11% (RR); 41% vs 24% (REES46) | Post-purchase add-ons window (complements) |
| Hours since last visit | | Browsers back within 1 h: 3.1% vs 0.9% after >30 days (RR); weak in REES46 | Return-visit triggers (learn per store) |
| Discount moment | a viewed item's price just dropped | Discount buying does **not** persist month to month (−0.06), only within a month (0.38) (REES46) | Price-drop alerts triggered by product + moment, never by a "deal seeker" label |

### 3. Relationship (stage)

Computed by `recsys.lifecycle` (step 2b adds the abandoner stage).

| Stage | Ordered within 7 days (RR / REES46) | Needs |
|---|---|---|
| New (first visit) | 0.7% / 7.6% | trending; the current visit |
| Browser (returned, never carted) | 1.7% / 9.7% | continue browsing; discovery |
| **Abandoner** (carted, never bought) | 11.7% / 22.2% | substitutes for the cart (45% of their carts are new items in browsed categories), reminder |
| 1 order | 12.9% / 28.5% | re-buy (16% of carts), complements, measured expansion |
| 2+ orders | 39% / 38–53% | known items, new arrivals in their categories |
| Bulk (>10 items, likely resellers) | 83% / 79% | reorder lists (89% re-cart known items); keep out of consumer loyalty metrics |

### 4. Style traits

| Trait | Definition | Stability | Evidence it matters | Drives |
|---|---|---|---|---|
| **Brand loyalty** (per category) | same-brand repeat vs the category's brand mix | 0.20 month to month (REES46); same brand 7× chance after 3+ years in grocery, 2–3× in electronics/home (Amazon) | Loyal third bought last month's most-viewed brand 69% vs 44% (REES46) | Same-brand alternatives or not; brand affinity with a time decay |
| **Mainstream vs niche** | mean popularity percentile (within category) of what they view/buy | 0.31 month to month (REES46); 0.18 at 1 year, 0.13 at 3 years (Amazon) | the most stable behavioural trait | Popularity blend weight |
| **Variety seeking** | categories per product viewed | 0.28 month to month (REES46) | | Exploration dial (share of discovery/adjacent slots) |
| **Comparison breadth / decisiveness** | alternatives viewed before buying; visits per purchase | 0.31 month to month (REES46) | Decisive shoppers convert 4–12× more per visit than researchers in the same stage (RR, REES46) | Number of alternatives; comparison aids for researchers |
| **Rating style** | mean stars given | **0.44** at 1 year, 0.39 at 3 years (Amazon) | the most stable trait of all | Mean-center explicit ratings per user (user bias) |

### 5. Price profile (hierarchical)

| Level | Definition | Evidence | Drives |
|---|---|---|---|
| Band per product type | mean price percentile (within leaf category) of what they view/buy in that type | 74% of a type-level price position (REES46); same type a year later 0.20 vs other type 0.09 (Amazon) | Which tier to lead with in a familiar type |
| General price level | the same, pooled across all types | Clearer with more history (0.17 → 0.30); 0.49 month to month in *new* categories when well observed (REES46); 0.18 at 1 year (Amazon) | Which tier to lead with in a type **new** to them: premium 29% vs 16% (REES46) |
| Anchor band | price of the item on screen | see layer 1 | Alternatives within ±30% |

Estimate (partial pooling, with `k` learned per store from the variance split):

```
general(s)    = (n_s · mean_s + k_g · store_mean) / (n_s + k_g)
band(s, type) = (n_st · mean_st + k_t · general(s)) / (n_st + k_t)
```

Guardrails (CLAUDE.md "Price is context, not taste"):
- a soft ranking signal only;
- never a filter beyond the shopper's own budget;
- never a label ("premium shopper") and never income;
- never changes the price charged.

### 6. Product context (per product type, per store)

| Property | Evidence (REES46) | Drives |
|---|---|---|
| Deliberation time | smartphones 2.6 days / 3 visits; irons minutes, 52% bought in one visit; premium electronics 3.3 vs 0.8 days budget | Message timing: days for considered purchases, hours for impulse ones |
| Comparison breadth | shoes 6 alternatives (median), furniture 5, small appliances 2 | How many, and how varied, alternatives (diversity re-ranking) |
| Cart conversion | apparel 25%, electronics 43% | Abandoned-cart priority |
| Adjacency (next type entered) | TV → home audio 2.4×, camera → security 2.1×, pantry → bakery 2.3× (Amazon) | Adjacent-category source; expansion is a mild nudge |
| Time-of-week rhythm | Most categories follow the store's rhythm; a few don't: alcohol late-week afternoons 1.2×, stock-up groceries on the big shopping days 1.1×, furniture/appliances at weekends 1.2× (Instacart, REES46; `2026-10-27-category-timing.md`) | Campaign timing per category, only where distinctive; learn per store **and channel** |
| Repeat cycle | Groceries: 46% shop weekly; a regular weekly shopper orders on day 7 with a 37% chance, then 24%, 15%… Product cycles are longer (milk 11 days, eggs 15, cheese 21) (Instacart; `2026-10-27-replenish-or-explore.md`) | **Rhythm state** vs the shopper's own gap (within cycle / due / overdue / lapsing): replenishment reminders when overdue (≥ 1.25–1.5× their gap), for the basket with the items that are due |
| Exposure to a new category | Depends on how the category is bought (REES46, self-chosen views). Need/spec-driven categories front-load: electronics peaks at the 1st exposure, computers fade to 0.58× by the 10th. Style/browse categories wear in: apparel rises 4.8× to a peak at the 9th, accessories and kids peak at the 6th–7th. Pooled: a peak at the 3rd–4th | Exposure cap per category: about 2–3 for need-driven and high-ticket items, 5–9 for style and browse; tune with impression logs |
| Room to explore | Items new to the shopper: 65% of orders 2–3 → 19% after order 30; a new aisle in 1 of 4 veteran orders; 38% of trials adopted within 3 orders (Instacart) | New-category campaigns early in the lifecycle; measure adoption, not trial |

## Rejected (failed the stability test)

| Candidate | Result | Use instead |
|---|---|---|
| "Deal seeker" | −0.06 month to month | The discount moment (layer 2) |
| Personal send hour, **for non-routine categories** | Electronics store: own usual 3-hour window catches 19.7% of next-month visits vs 18.6% for the store's peak (REES46) | Store-level peak + momentum timing |
| Weekday vs weekend shopper, for non-routine categories | 0.06 month to month (REES46) | Nothing |

**Correction (2026-10-27):** personal timing *does* hold for routine
categories. Grocery shoppers' own usual window and day beat the store's
peak by about 1.3× (32% vs 25%; 23% vs 17%; Instacart,
`2026-10-27-category-timing.md`). Personal routine is a trait for repeat
grocery-type shopping only.
| Category expansion → loyalty | 2× in RetailRocket only; ±1–2 points in Olist, REES46, H&M | A per-store randomized test before any strategy |

## Visit intent after an order

Since the last order: visits, **intent visits** (searches, carts, focused
views, comparing within a category) and days since the last visit. These
predict the next order better than plain visit counts
(`docs/experiments/2026-10-27-loyalty-drivers.md`). Searches are the target
signal; public datasets only allow proxies.

## Not yet known (needs our own logged data or new datasets)

- **Responsiveness:** who acts on recommendations and messages, and which
  explanation types work for whom. This is the uplift question; only
  randomized holdouts on our own traffic can answer it (LIFECYCLE.md
  next-best-action).
- **What they value** (quality, price, durability, looks): review text
  (Amazon Reviews with text).
- **Replenishment cycles:** order data with re-buys (Instacart).
- **Who a message without an incentive persuades:** in a randomized
  *promotional* SMS campaign, the likeliest buyers gained least (+1.0 point
  vs +3.3 on average), and clients who often spend loyalty points gained most
  (+8.4) (`docs/experiments/2026-10-27-x5-sms-uplift.md`).
  - The points result is about response to a promotion.
  - It must not decide who gets a discount: that would be personalized
    pricing, which "Price is context, not taste" excludes.
- **Who an abandoned-cart message persuades:** organic data only shows
  propensity (2+ order shoppers recover 13.6%, new visitors 3.3%; REES46),
  not uplift (`docs/experiments/2026-10-27-cart-abandonment.md`).
- **Lapsing over months:** windows longer than REES46's 61 days, with visits.

## `ShopperState`: what step 2b computes at request time

All fields come from `UserContext.events` (plus store-level tables learned
nightly). Decay half-lives and shrinkage constants are config, learned per
store.

| Group | Fields |
|---|---|
| Visit | `visit_events`, `visit_departments`, `anchor_item`, `anchor_price_pct` |
| Momentum | `hours_since_last_visit`, `open_cart_age_hours`, `hours_since_last_order` |
| Relationship | `stage`, `orders`, `items_bought`, `visits_30d`, `is_bulk` |
| Style | `visits_per_order`, `alternatives_per_purchase`, `mainstream` (decayed), `variety` (decayed), `rating_bias` |
| Brand | decayed `brand_affinity[category][brand]` |
| Price | `price_general`, `price_band[type]` (partially pooled), with evidence counts |
| Store tables | per-type deliberation time, comparison breadth, cart conversion, adjacency lifts, popularity percentiles |

**How it's used, in order of build:**
1. Rules by stage and momentum pick the mix of candidate sources (2b).
2. Traits and the price profile adjust within the mix (soft re-ranking).
3. A learned ranker or contextual bandit takes all fields as features and
   replaces the hand rules once there is logged data to learn from.

**Ethics:** no sensitive attributes are inferred (health, pregnancy,
finances, income); no labels are shown to shoppers or sold; explanations
stay about products ("similar to what you viewed"), never about the person
("because you're a premium shopper").

## Evidence

| Write-up | Dataset | What it established |
|---|---|---|
| `experiments/2026-10-24-session-sources-retailrocket.md` | RR | Current visit dominates carts; source strengths by journey |
| `experiments/2026-10-24-visits-by-cohort.md` | RR, REES46 | Visits climb with loyalty; leading indicator |
| `experiments/2026-10-24-lifecycle-across-datasets.md` | 5 stores | What replicates (and what doesn't) in lifecycles |
| `experiments/2026-10-24-lifecycle-stages-deep-dive.md` | RR, REES46 | Abandoner stage; momentum; style; in-visit needs |
| `experiments/2026-10-24-product-type-and-price.md` | REES46 | Product context; anchor price band |
| `experiments/2026-10-27-category-timing.md` | Instacart, Olist, REES46 | Category rhythms; personal routines in groceries |
| `experiments/2026-10-27-replenish-or-explore.md` | Instacart, REES46 | Rhythm states and reminder timing; product cycles; room to explore; exposure curve |
| `experiments/2026-10-24-shopper-traits-stability.md` | REES46 | Which traits are stable; rejected traits |
| `experiments/2026-10-24-shopper-price-level.md` | REES46 | Hierarchical price profile |
| `experiments/2026-10-24-amazon-traits-over-years.md` | Amazon | Traits over years; brand decay; adjacency |
