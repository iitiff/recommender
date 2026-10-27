# 2026-10-27: Replenish or explore? Shopper rhythm, product cycles, new-category exposure

**Status:** done (descriptive, observational)
**Data:** aggregates only.
- **Instacart** online grocery, 2017: 3.2M prior orders, 32M order lines, 134 aisles. Non-commercial use.
- **REES46** multi-category store, Oct–Nov 2019. "© Original Authors".

## Question

For lifecycle messaging, three decisions:
1. **Replenishment timing:** a weekly shopper hasn't ordered yet. At what
   point do we remind them, and when should we wait because their rhythm
   says they'll come anyway?
2. **Replenish or explore:** when is a shopper open to a new category, and
   when do they just restock?
3. **Exposure:** how many times should we show a new category before moving
   on to the next one?

## Method

```bash
python -I docs/experiments/scripts/replenish_explore.py instacart DIR
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/replenish_explore.py rees46 events.parquet
```

- **Rhythm:**
  - From each shopper's first half of orders (shoppers with 8+ orders): their
    median gap between orders, and its regularity (coefficient of variation
    ≤ 0.5 = regular).
  - On their second half: the chance of ordering on day *d* since the last
    order, given no order yet. Concept: **discrete-time hazard**,
    survival analysis.
  - Also, the share of next orders placed within *k* × their own median gap.
- **Product cycles:** days between purchases of the same product by the
  same shopper, per aisle.
- **Exploration by lifecycle:** the share of an order's items that are new
  to the shopper, and whether it includes a new aisle, by order number.
  **Trial → adoption:** whether a first purchase in a new aisle is repeated
  within the next 3 orders.
- **Exposure curve (REES46):**
  - Covers product types the shopper hadn't bought before viewing them
    (with a week of history before).
  - Each visit in which they view the type is an exposure.
  - Measures the chance of buying in the type at the *k*-th exposure, given
    no purchase at earlier ones (hazard over exposures).
- **Instacart caps the days between orders at 30.**

## Results

**1. Rhythm (120,918 shoppers with 8+ orders)**
- **Cadence:** weekly 46%, every two weeks 31%, monthly 17%, 30+ days 5%.
- **Regular (CV ≤ 0.5):** 35% overall. The 30-day cap makes monthly
  shoppers look more regular than they are.

**Regular weekly shoppers (n = 10,625): chance of ordering on day *d*, if not yet**

| Day since last order | 3 | 5 | 6 | **7** | 8 | 9 | 10 | 12 | **14** | 21 |
|---|---|---|---|---|---|---|---|---|---|---|
| Hazard | 5% | 9% | 16% | **37%** | 24% | 15% | 13% | 10% | **20%** | 13% |
| Ordered by then (cumulative) | | | 36% | 60% | 69% | 74% | 77% | 82% | 87% | 94% |

**Next order within k × the shopper's own median gap**

| k | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| Regular shoppers | 29% | 55% | 66% | 76% | 88% | 95% |
| Irregular shoppers | 36% | 54% | 63% | 71% | 83% | 92% |

**2. Product cycles**
- **Repeat buying:** 40% of shopper-product pairs are bought more than once,
  with a median of 16 days between repeats.
- **Regularity:** only 25% of products a shopper bought 4+ times are bought
  on a regular cycle.

| Aisle | Median days between repeat purchases |
|---|---|
| Milk | 11 |
| Fresh fruits, water & seltzer, refrigerated | 13 |
| Yogurt, soy & lactose-free, granola bars | 14 |
| Eggs, bread | 15 |
| Packaged vegetables & fruits | 16 |
| Chips & pretzels | 17 |
| Fresh vegetables | 19 |
| Frozen produce, crackers | 20 |
| Packaged cheese | 21 |

**3. Room to explore, by order number**

| Order | 1st | 2–3 | 4–10 | 11–30 | 31+ |
|---|---|---|---|---|---|
| Items new to the shopper | 100% | 65% | 43% | 27% | 19% |
| Orders with a new aisle | 100% | 88% | 69% | 44% | 25% |

Trial → adoption: 37.5% of first purchases in a new aisle (3.2M trials) are
bought again within the next 3 orders.

**4. Exposure curve (REES46; 7.6M shopper × new-type pairs, 7.1% bought in the window)**

| k-th visit viewing the type | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|
| Buy at this visit, if not before | 3.6% | 4.4% | **4.6%** | 4.6% | 4.4% | 4.1% | 3.6% | 3.2% |
| Pairs reaching this visit | 7.6M | 2.4M | 1.2M | 733k | 484k | 339k | 185k | 112k |

**By category and price band** (same measure, split by the type's top-level
category and by its median price paid)

| Category | Median price | Buy at 1st exposure | Peak at | Peak vs 1st | 10th (or last) vs peak |
|---|---|---|---|---|---|
| Electronics | $260 | 5.1% | 1st | 1.0× | 0.65× |
| Computers | $180 | 2.6% | 2nd–4th | 1.2× | 0.58× |
| Appliances | $188 | 4.0% | 4th | 1.4× | 0.67× |
| Construction | $90 | 2.2% | 4th | 2.0× | 0.69× |
| Auto | $121 | 2.1% | 4th | 2.1× | 0.80× |
| Furniture | $144 | 1.0% | 5th | 3.3× | 0.92× |
| Sport | $175 | 1.0% | 5th | 3.9× | 0.95× |
| Kids | $38 | 1.4% | 6th | 3.1× | 0.67× |
| Accessories | $35 | 0.9% | 7th | 4.9× | 1.0× |
| Apparel | $83 | 0.8% | 9th | 4.8× | 0.94× |
| Price band: under $25 | | 2.4% | 3rd | 3.6× | 0.90× |
| Price band: $25–100 | | 1.4% | 5th | 2.8× | 0.74× |
| Price band: $100–300 | | 4.2% | 3rd | 1.1× | 0.68× |
| Price band: $300+ | | 3.7% | 2nd | 1.1× | 0.62× |

## Conclusion

1. **Rhythm makes reminders a timing question, not a yes/no.**
   - A regular weekly shopper orders on day 7 with a 37% chance. A reminder
     before then mostly reaches people who were coming anyway: "sure things"
     in uplift terms.
   - The chance falls fast after the usual day (24% on day 8, 15% on day 9).
     **"Overdue" starts around 1.25–1.5× the shopper's own gap.**
   - That is the window where a replenishment reminder has room to work. It
     is the third or so of next orders that would come late, or not at all
     within the month.
   - The day-14 bump is shoppers who skipped a week, so a second reminder
     belongs just before it.
   - These are *when people order*, not *when reminders work*. The exact
     thresholds are for an uplift test to settle.
2. **Replenish the basket, not the item.** Product cycles (11–21 days) are
   longer and less regular than the shopping cycle (7 days). The useful
   reminder is "time for your weekly shop", with the items most likely due
   (days since last bought ≥ that item's usual cycle).
3. **Exploration belongs early in the lifecycle, and stays alive later.**
   - Orders 2–10 are where shoppers build their assortment: 43–65% new
     items, and most orders include a new aisle.
   - Veterans (31+ orders) still try a new aisle in 1 order out of 4.
   - New-category campaigns pay best early. For veterans, they ride on top of
     the replenishment moment rather than replacing it.
   - Success is measured by **adoption** (bought again within 3 orders:
     38% of trials), not by a single trial.
4. **New-category exposure depends on how the category is bought.**
   - **Need- and spec-driven categories front-load.** Electronics,
     computers and $300+ items peak at the 1st–2nd exposure and fade
     fastest (to 0.6× by the 10th). Cap at about 2–3, then move on.
   - **Style and browse categories wear in.** Apparel, accessories, kids,
     furniture and sport start low (about 1%), rise 3–5× by the 5th–9th
     exposure and barely fade. Keep showing them 5–9 times.
   - **In between** (appliances, auto, tools): peak around the 3rd–4th.
   - **Price alone isn't the driver:** apparel ($83) ramps most, while
     computers ($180) fade fastest. The cheap price band ramps because this
     store's cheap categories are accessories and kids' items.
   - Concepts: **wear-in / wear-out**, effective frequency.
   - Pooled across categories, the picture below holds: a peak at the
     3rd–4th exposure.
   - Groceries can't be measured here: REES46 has none, and Instacart has
     no exposures. The routine, low-price end (ramp, then sustain) is a
     hypothesis, not a finding.

   **Pooled across categories: about 3–5 times, then rotate.**
   - The chance of buying peaks at the 3rd–4th exposure and declines slowly
     after about 5. There is no cliff, so interest doesn't vanish after one
     showing, but returns shrink.
   - These are *self-chosen* exposures (the shopper browsed the category).
     *Pushed* impressions convert less and wear out faster.
   - Treat 3–5 as a starting cap, to be tuned with impression logs and a
     frequency test. Concepts: **impression discounting**, frequency capping,
     wear-out, and Duolingo's recovering bandit, where a message regains
     impact after a rest.

**Decision sketch for messaging (to be validated by uplift tests):**

> **Tested in part (2026-10-27, `2026-10-27-x5-sms-uplift.md`):** in a
> randomized grocery promotional SMS campaign, shoppers within their cycle gained as much
> from the message (+3.4 points) as overdue (+3.0) and lapsing (+3.4) ones. So
> "stay quiet within the cycle" does not hold for a generic offer message. It
> may still hold for a replenishment reminder, which needs its own test.

| Shopper state (rhythm vs their own gap) | Share of next orders arriving by then | Message |
|---|---|---|
| Within cycle (< 0.75×) | 29–36% | None for replenishment. A light new-category idea if early in the lifecycle |
| Due (0.75–1.25×) | 55–66% by 1.0–1.25× | Usually none: most are coming anyway |
| **Overdue (1.25–2×)** | 66% → 88% | **Replenishment reminder**: their usual basket, items due first |
| Lapsing (> 2×) | 12–17% still to come | Win-back (timing and offer learned per store) |

**Caveats:**
- **Instacart:** one online grocery, 2017. Gaps are capped at 30 days, and
  "regular" is measured on the first half of each shopper's history.
- **Exposure:** REES46 exposures are the shopper's own views, not our
  impressions, and the window is only two months.
- **Observational:** this shows when people order and buy. Whether a reminder
  or campaign *causes* more orders needs a randomized holdout.
