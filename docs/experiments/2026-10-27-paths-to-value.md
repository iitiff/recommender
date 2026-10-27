# 2026-10-27: The path to loyal *and* valuable: what the first 90 days say (H&M)

**Status:** done (descriptive, observational)
**Follows:** `2026-10-27-paths-to-loyalty.md` (paths to frequency) and `2026-10-27-loyalty-dimensions.md` (loyal ≠ valuable).
**Data:** H&M new-customer cohort (146,884 customers), aggregates only, behavior only.

## Question

"Loyal" now has a value side:
- **True friends:** loyal and valuable.
- **Barnacles:** loyal, low value.
- **Butterflies:** valuable, not frequent.
- **Strangers:** neither.

What does the path into each group look like, and what can we tell by day 90?

## Method

```bash
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/paths_to_value.py TRANSACTIONS_DIR articles.parquet
```

**12-month group:**
- one order;
- otherwise Reinartz & Kumar's grid on repeat customers: loyal = 6+
  orders and still buying in months 10–12; valuable = 12-month spend at or
  above the repeaters' median.

**Day-90 landmark:** only what's known by day 90.
- Orders, and days to the second order.
- Share of items bought on sale.
- Channels used.
- Product groups bought.
- Spend.

**Monthly paths:** for each group, the share active, the share of items
on sale, and spend per month. Spend is indexed to the cohort's average
spend per customer per month.

## Results

**Where new customers end up:** one order 36.4%, strangers 30.6%,
butterflies 20.7%, **true friends 11.1%**, barnacles 1.2%.

### What day 90 tells you

| By day 90 | Share | One order | Strangers | Butterflies | Barnacles | **True friends** |
|---|---|---|---|---|---|---|
| **Orders:** 1 | 57.5% | 63% | 23% | 11% | 0.4% | 2.5% |
| 2 | 22.8% | – | 56% | 31% | 1.5% | 11% |
| 3+ | 19.7% | – | 24% | 37% | 3.0% | **36%** |
| **Second order:** by day 30 | 24.9% | – | 39% | 34% | 2.1% | **24.5%** |
| 31–60 | 10.5% | – | 42% | 33% | 2.7% | 22.3% |
| 61–90 | 7.1% | – | 48% | 33% | 2.1% | 16.9% |
| **Channels:** online only | 59.7% | 44% | 27% | 21% | 0.5% | 7.3% |
| store only | 27.6% | 37% | 38% | 14% | 2.1% | 9.2% |
| both | 12.8% | – | 31% | 34% | 2.3% | **32.7%** |
| **Items bought on sale:** none | 58.3% | 43% | 30% | 19% | 0.7% | 7.1% |
| under half | 24.0% | 13% | 32% | 30% | 2.0% | **22.6%** |
| half or more | 17.7% | **47%** | 30% | 13% | 1.6% | 8.2% |
| **Product groups:** 1 | 37.1% | 57% | 31% | 8% | 0.7% | 2.9% |
| 4+ | 17.8% | 10% | 18% | **42%** | 1.2% | 28.6% |

Rows with "–" contain no one-order customers by construction: they already
placed a second order.

**The most telling day-90 profiles** (second order / channels / items on
sale), for customers who placed a second order by day 30:

| Profile | Share of cohort | True friends | Butterflies | Barnacles |
|---|---|---|---|---|
| Both channels / some on sale | 4.7% | **40.6%** | 35% | 2.2% |
| Both channels / none on sale | 2.4% | 26.6% | 35% | 1.6% |
| Online only / some on sale | 3.3% | 26.1% | 41% | 1.1% |
| Store only / some on sale | 3.0% | 23.6% | 28% | **4.1%** |
| Store only / none on sale | 1.6% | 16.1% | 21% | 3.7% |
| Online only / none on sale | 6.8% | 14.9% | **40%** | 0.6% |

### What each group's first year looks like

| Month after first order | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|
| **True friends:** active | 100% | 49% | 48% | 46% | 44% | 46% | 52% | 57% |
| spend index | 7.5 | 2.6 | 2.5 | 2.3 | 2.4 | 2.2 | 2.5 | 3.6 |
| **Butterflies:** active | 100% | 29% | 27% | 24% | 22% | 20% | 11% | 15% |
| spend index | **9.6** | 1.8 | 1.6 | 1.4 | 1.3 | 1.1 | 0.7 | 1.1 |
| **Barnacles:** active | 100% | 43% | 42% | 38% | 31% | 36% | 50% | 44% |
| items on sale | 33% | 38% | 45% | 45% | 30% | 44% | 45% | 36% |
| **Strangers:** active | 100% | 19% | 17% | 14% | 11% | 11% | 9% | 11% |

Month 1 includes the first order.

## Conclusion

1. **The path to "true friend" shows by day 90.**
   - **Three or more orders by day 90:** 36% end as true friends, vs 2.5%
     with only one.
   - **Using both store and online:** 33% true friends, vs 7–9% with one
     channel.
   - **Some, but not most, items on sale:** 23% true friends, vs 7–8% for
     none or mostly on sale.
   - **The best early profile:** a second order by day 30, both channels,
     some sale buying. 41% of those customers become true friends, almost
     4 times the cohort rate of 11%.
2. **Butterflies have a recognizable path too.**
   - A big first month (spend index 9.6, the highest of any group), then
     fading activity: 29% active in month 2, 11–15% by months 10–12.
   - Typical start: online only, little on sale, a wide first basket (4+
     product groups).
   - The job with them is to serve them well while they buy, rather than
     pushing frequency (Reinartz & Kumar).
3. **Barnacles show early too, but they're rare** (1.2%). They lean to
   store-only shoppers with some sale buying. Their sale share stays high
   all year (30–49% of items), and they stay moderately active (31–50% of
   months).
4. **True friends don't plateau:** they're active in about half of all
   months from month 2 on, and more active again by months 10–12 (57%).
   Their path is steady from early on. This matches the "steady" and
   "early repeat that keeps going" families in the paths write-up.
5. **For lifecycle programs:**
   - **At day 90, tag likely true friends** (3+ orders, or both channels
     with an early second order) and likely butterflies (one big, wide,
     full-price basket, online).
   - **Likely true friends:** inviting single-channel shoppers to try the
     other channel (store or app) is worth testing. Whether *encouraging*
     a second channel creates loyalty is an open causal question, since
     the omnichannel link may be selection (living near a store).
   - **Likely butterflies:** relevance and new arrivals, not frequency
     pushes.
   - **Mostly-on-sale starters** (47% one-and-done): a second-order plan
     without another discount.

**Caveats:**
- **Observational:** early behavior predicts the group; nudging it may not
  change it. The omnichannel link is the main candidate for a test.
- **Some rows are mechanical:** "both channels" needs at least two orders,
  and breadth and spend grow with orders. The profile table compares
  customers who all placed an early second order.
- **No margin data:** full-price share is a proxy.
- **Follow-up overlaps early 2020;** the rise in months 10–12 may partly
  reflect that period.
- **Fashion, one retailer:** grocery paths need the retailer's own data
  (`docs/VALIDATION-PLAN.md` 1.1, 1.16).
