# 2026-10-28: The second order: same category, or a new one?

**Status:** done (descriptive, observational)
**Follows:**
- `2026-10-24-lifecycle-across-datasets.md` (64–82% of 2nd orders stay in
  the top category; expanding didn't raise the 3rd-order rate);
- `2026-10-27-paths-across-datasets.md` (loyal customers are more frequent,
  not broader).

**Question:** Is it common for a new customer's second order to stay in the
first order's category, before they expand to new categories? And should
the second-order program push expansion, or lead with the first category?

**What is new here:**
1. **A chance baseline.** "Same category" looks high wherever one category
   dominates.
2. **Grocery data.**
3. **Outcomes up to 12 months,** adjusted for the gap to the 2nd order and
   its size.
4. **When customers who reach 6 orders first add a category,** and whether
   customers who stay narrow catch up.

## Data

Aggregates only; cells under 50 customers are suppressed. Cohorts and loaders
are the same as `paths_to_loyalty.py`.

| Dataset | Business | Category | Customers with a 2nd order | Outcome |
|---|---|---|---|---|
| **H&M** | Fashion | Product group (upper body, trousers, shoes, accessories...) | 93,366 | 12 months: 3rd order, loyal (6+ orders and still buying), orders |
| **Olist** | Marketplace | Product category (~70) | 664 | 12 months |
| **X5** | Grocery, new card holders | Product level 2 | 12,049 | 90 days: 3rd order, weekly in days 31–90, orders |
| **Instacart** | Online grocery, shoppers with 10+ orders only | Department (21) | 101,696 | Weekly regular (median gap ≤ 7 days over orders 4–10) |

## Method

```bash
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/second_order_category.py hm TRANSACTIONS_DIR ARTICLES_PARQUET
# ... olist OLIST_DIR | x5 X5_DIR | instacart INSTACART_DIR
```

- **Stays:** the 2nd order buys no category that wasn't in the 1st.
- **Expands:** it buys at least one new category.
- **Repeats:** it buys at least one of the 1st order's categories, possibly
  alongside new ones.
- **Chance:** 2nd orders are shuffled between customers with the same number
  of categories in their 1st and 2nd orders (5 shuffles).
- **Adjusted difference:** expanders minus stayers within strata of the gap
  to the 2nd order and the 2nd order's size, weighted by expanders. A bigger
  2nd order adds categories mechanically; a longer gap means a different
  customer.

## Results

### 1. Staying in category is common, and well above chance

| | H&M | Olist | X5 | Instacart |
|---|---|---|---|---|
| Categories in the 1st order (median) | 1 | 1 | 5 | 5 |
| **2nd order repeats a 1st-order category** | **62%** (chance 48%) | **36%** (chance 7%) | **82%** (chance 73%) | **94%** (chance 84%) |
| 2nd order stays (no new category) | 36% (chance 23%) | 35% (chance 6%) | 20% (chance 12%) | 21% (chance 7%) |
| Share of the 2nd order's categories already bought | 49% | 36% | 52% | 62% |

### 2. Expanding on the 2nd order goes with little or no extra loyalty

Expanders minus stayers, in percentage points (orders as counts):

| Dataset | Outcome | Stayers | Expanders | Raw | **Adjusted** |
|---|---|---|---|---|---|
| H&M | 3rd order within 12 months | 68.6% | 71.1% | +2.5 | **+3.8** |
| H&M | Loyal at 12 months | 18.1% | 19.8% | +1.7 | **+3.0** |
| H&M | Orders in 12 months | 4.59 | 4.69 | +0.11 | +0.34 |
| Olist | 3rd order within 12 months | 11.5% | 9.1% | −2.5 | −5.3 (n = 234 / 430) |
| X5 | Weekly in days 31–90 | 44.4% | 44.5% | +0.1 | +2.4 |
| X5 | Orders in 90 days | 14.9 | 15.1 | +0.2 | +0.9 |
| Instacart | Weekly regular | 41.9% | 40.9% | −1.0 | **0.0** |

### 3. Customers who reach 6 orders expand early, loyal or not

**The order at which the first new category appears** (customers with 6+
orders):

| | Order 2 | Order 3 | Order 4 | Orders 5–6 | Not by order 6 |
|---|---|---|---|---|---|
| H&M loyal (n = 17,945) | 66% | 19% | 7% | 5% | 4% |
| H&M not loyal (n = 7,157) | 64% | 19% | 7% | 5% | 5% |
| Instacart weekly regulars (n = 41,770) | 79% | 12% | 4% | 3% | 3% |
| Instacart others (n = 59,926) | 79% | 12% | 4% | 3% | 3% |

(In X5, nearly every 6-order customer counts as loyal within 90 days, so
there is no comparison group.)

**Breadth after 6 orders, by what the 2nd order did:**

| | Categories in order 1 | Stayed at order 2 → after order 6 | Expanded at order 2 → after order 6 |
|---|---|---|---|
| H&M | 2.1 (stayed) / 1.6 (expanded) | 4.1 | 4.7 |
| X5 | 5.9 / 4.6 | 9.0 | 10.0 |
| Instacart | 4.8 / 4.9 | 7.6 | 10.1 |

## Conclusion

1. **Yes: staying in the same category for the second order is common, and
   more than chance.**
   - In fashion and marketplaces, where a first order holds about one
     category, a third of 2nd orders add nothing new.
   - Most 2nd orders (62% in H&M) come back for something from the first
     category. In Olist, a repeat is 5× chance.
   - In grocery, almost every 2nd basket repeats a department (82–94%) *and*
     adds one. "Same vs new" is the wrong frame there. About half of each
     2nd basket is categories already bought, and grocery breadth grows
     inside the routine.
   - The first category is what brought the customer back. It is the
     strongest single relevance signal for the second-order message.
2. **Expanding on the 2nd order is not where loyalty comes from.**
   - The adjusted differences are small, and differ in sign across
     datasets:
     - +3 points of 12-month loyalty in H&M (18% → 21%);
     - about +2 points of weekly shopping in X5;
     - zero in Instacart;
     - negative in the small Olist sample.
   - Even the H&M +3 points is observational. Customers who expand may
     simply have more needs, so it is an upper bound on what pushing
     expansion could cause.
   - This matches the earlier finding that loyal customers are more
     *frequent*, not broader.
3. **There is no "deepen first, then broaden" phase that separates loyal
   customers.**
   - Among customers who reach 6 orders, loyal and non-loyal ones add their
     first new category at the same point: about two thirds at order 2 in
     H&M and 79% in Instacart.
   - So neither waiting nor rushing distinguishes them.
4. **Staying narrow at order 2 looks like a trait, not a phase.**
   - Stayers stay narrower: 7.6 vs 10.1 departments by order 6 in
     Instacart, and 4.1 vs 4.7 product groups in H&M, though H&M stayers
     started broader.
   - But in H&M, X5 and Instacart they were not less frequent or less
     loyal.
   - Some customers have a small repertoire. Respect it rather than fight
     it.

**For the second-order program (experiment 2.1):**
- **Lead with the first category:** buy again, refills, sizes and colors,
  "complete the look" within the same need.
- **Add one adjacent category as a secondary slot**, not the headline.
- **Run the "next-category vs buy again" arm comparison** already in 2.1,
  against a holdout. These data can't say whether *pushing* expansion
  helps; they only say that customers who expand on their own aren't much
  more loyal.
- **Read out:**
  - the 2nd-order rate in 60–90 days;
  - 12-month loyalty;
  - whether next-category content *lowers* the 2nd-order rate among
    narrow-repertoire customers (the habit risk in 2.13).
- **In grocery, aim the second order at frequency** (the next rung of the
  frequency ladder, `2026-10-27-grocery-paths.md`), with adjacent
  departments riding along in the basket.

**Caveats:**
- **Observational:** expanders and stayers differ in ways the strata don't
  capture.
- **Category granularity shapes every number.** H&M product groups and
  Instacart departments are coarse; Olist categories are narrower; X5
  level 2 is hashed.
- **Instacart includes only shoppers with 10+ orders,** so it describes
  customers who stay, not conversion.
- **X5 follows customers for 90 days only.**
- **Olist has only 664 repeat customers.**
- **H&M "new" customers** had no purchase in the previous 6 months, but may
  have bought before that.
