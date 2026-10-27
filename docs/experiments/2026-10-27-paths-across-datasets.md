# 2026-10-27: Paths to loyalty across five stores, by category, and the one-order buyer

**Status:** done (descriptive, observational)
**Follows:** `2026-10-27-paths-to-loyalty.md` (H&M). Same script, now with a loader per dataset.

**Data:** aggregates only, behavior only.

| Dataset | Business | Cohort (new customers) | Follow-up | Categories |
|---|---|---|---|---|
| **H&M** | Fashion, store + online | 146,884; first order after 6+ months without one | 12 months | Product group |
| **CDNOW** | Online music | 23,570; **true first-ever purchases**, Q1 1997 | 12 months | – |
| **Olist** | Marketplace (Brazil) | 19,491; first order Mar–Aug 2017 | 12 months | Product category |
| **Online Retail II** | Gift wholesaler (B2B) | 1,563; first order after 6+ months without one | 12 months | – |
| **X5** | Grocery stores | 12,202; loyalty card issued in the window (true start) | **90 days** | Product level 2 (names hashed) |
| **Instacart** | Online grocery | 101,696 shoppers with 10+ orders | Orders 1–10 | Department |

**Two datasets can't measure conversion:**
- **X5:** its client list was built for a campaign, and only 1.3% of its new
  card holders stop after one order. That's far too low for real new
  customers, so the list was probably limited to active shoppers.
- **Instacart:** includes only shoppers with 4+ orders.

Both are used only for **how routines and category breadth form among
customers who stay**.

## Method

```bash
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/paths_to_loyalty.py {hm|cdnow|x5|olist|onlineretail|instacart} ...
```

The definitions are as in the H&M write-up:
- An order is a customer-day.
- **Loyal:** 6+ orders in the first 12 months, and still buying in months
  10–12.
- **Families:** groups of 12-month activity patterns.

Added for this write-up:
- **Second-order hazard:** the chance of a second order in week *w*, if
  there hasn't been one yet.
- **Late returns:** of customers with no second order by day 90, how many
  come back within the year.
- **Category level:**
  - each order's **main category** (the one with the largest spend);
  - the 12-month outcome by the first order's main category;
  - what the second order buys;
  - which categories are added by order 4;
  - category breadth by order number.

The H&M results reproduce the earlier write-up exactly.

## Results

### 1. Outcomes and the funnel

| | H&M | CDNOW | Online Retail (B2B) | Olist |
|---|---|---|---|---|
| **One order only, 90 days** | 57.5% | 70.8% | 61.4% | 98.6% |
| **One order only, 12 months** | 36.4% | 54.1% | 38.5% | **96.6%** |
| No 2nd order by day 90, but one by 12 months | 36.6% | 23.5% | 37.4% | 2.0% |
| Loyal at 12 months | 12.2% | 6.5% | 10.6% | ~0% |
| Chance of a 2nd order → 3rd → 4th | 64% → 70% → 72% | 46% → 60% → 66% | 62% → 65% → 68% | 3% → 10% → 21% |
| Median days to the 2nd order | 49 | 50 | 59 | 125 |

**The pattern is the same everywhere:** the second order is the biggest
drop. After it, each next order is more likely and comes sooner. In H&M and
CDNOW, the gap between orders shrinks from about 50 days to about 22 by the
seventh order.

### 2. When the second order comes (weekly hazard, %)

| Week | 1 | 2 | 4 | 8 | 12 | 16 | 26 | 39 | 52 |
|---|---|---|---|---|---|---|---|---|---|
| H&M | **11.7** | 5.8 | 4.2 | 3.2 | 2.6 | 1.8 | 1.4 | 0.8 | 0.9 |
| CDNOW | **5.9** | 4.7 | 3.2 | 1.9 | 1.3 | 1.2 | 0.8 | 0.6 | 0.3 |
| Online Retail | 3.8 | **5.7** | 4.4 | 2.7 | 2.6 | 2.0 | 1.6 | 0.9 | 3.1 |
| Olist | 0.2 | 0.2 | 0.15 | 0.08 | 0.09 | 0.04 | 0.06 | 0.03 | 0.03 |

- The chance of a second order is highest in the first weeks.
- It falls **4–5-fold by week 12** in H&M and CDNOW, and about twofold at the wholesaler.
- Then it settles at about 1% a week: a long tail of late returns.
- The wholesaler peaks in week 2 and again at week 52, an annual reorder.

**Loyal at 12 months, by days to the second order**

| Days to 2nd order | 1–14 | 15–30 | 31–60 | 61–120 | 121–365 |
|---|---|---|---|---|---|
| H&M | 25.9% | 27.9% | 25.0% | 17.6% | 5.6% |
| CDNOW | 22.1% | 19.6% | 18.6% | 12.0% | 4.1% |
| Online Retail | 24.3% | 31.9% | 26.4% | 11.1% | 3.6% |

The **60-day cliff replicates** in all three stores where loyalty can form.

### 3. Path families replicate

| Family | H&M | CDNOW | Online Retail | Loyal in it (H&M / CDNOW / OR) |
|---|---|---|---|---|
| A. One and done | 36.4% | 54.1% | 38.5% | 0 |
| B. Quick repeat, then gone (months 1–3) | 12.9% | 12.0% | 12.2% | 0 |
| C. Early repeat, faded by month 9 | 12.3% | 7.1% | 9.3% | 0 |
| D. Slow return (after month 3), then gone | 10.7% | 9.2% | 11.3% | 0 |
| E. Slow start, still buying at the end | 10.1% | 7.3% | 11.5% | 13% / 9% / 7% |
| F. Early repeat, still buying at the end | 10.9% | 6.2% | 10.8% | 46% / 37% / 33% |
| G. Steady (6+ active months) | 6.6% | 4.1% | 6.5% | 88% / 88% / 94% |

- **Share of loyal customers from G / F / E:** H&M 48 / 41 / 11%, CDNOW
  55 / 36 / 10%, Online Retail 58 / 34 / 8%.
- **How many distinct paths:**
  - CDNOW: 1,743 of 4,095 possible 12-month patterns among 10,810
    repeaters, and 19 of them cover half.
  - Online Retail: 395 among 962 repeaters.
  - The long tail is everywhere; the families are stable.

### 4. What the first order tells you about one-and-done

| First order | H&M: one order only | CDNOW | Online Retail | Olist |
|---|---|---|---|---|
| Smallest vs largest basket (items) | 40% vs 33% | **60% vs 40%** | **66% vs 37%** | 97% vs 97% |
| Lowest vs highest spend quintile | 37% vs 33% | **60% vs 46%** | **53% vs 30%** | 96% vs 97% |
| Store vs online (H&M only) | 31% vs 39% | – | – | – |
| 1 vs 2+ categories | 39% vs 31–34% | – | – | 97% vs 94% |

### 5. Category level

**H&M: the entry category barely changes the outcome.**

| First order's main group | Share | One order only | Loyal | 2nd order buys the same group again |
|---|---|---|---|---|
| Garment upper body | 30% | 36.4% | 12.6% | **62.5%** |
| Garment lower body | 26% | 36.2% | 12.1% | 52.0% |
| Garment full body | 16% | 36.3% | 12.6% | 39.9% |
| Swimwear | 15% | 36.3% | 11.4% | 37.2% |
| Underwear | 6% | 35.3% | 12.3% | 33.9% |
| Shoes | 3% | 39.7% | 11.6% | 23.5% |
| Accessories | 3% | 40.1% | 12.1% | 20.3% |

- **About half of second orders (48%) repeat the first main group.**
  Shoes and accessories buyers mostly come back for clothes (repeat rate
  only 20–24%).
- **Building an outfit edges ahead:** upper → lower body 21.1% loyal and
  upper → full body 22.3%, vs upper → upper 18.4%.
- **Added by the 4th order** (among customers reaching it):
  - lower body 38%, upper body 37%, full body 31%;
  - underwear 24%, accessories 23%;
  - swimwear 21%, shoes 11%.
- **Breadth doesn't separate loyal customers:** after the same number of
  orders, loyal customers have bought from about as many groups as the
  rest (4.5 vs 4.4 after order 6). Loyalty is about *frequency*, not
  breadth.

**Olist: every category is one-and-done.** One-order rates run from 94.5%
(furniture and decor) to 98.1% (gifts and gadgets). The few second orders
repeat the category 35% of the time.

**Grocery (among customers who stay): breadth builds order by order.**

| Departments bought by… | Order 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| Instacart (of 21) | 4.9 | 6.7 | 7.8 | 9.1 | 10.8 |

X5 (level 2, hashed) shows the same curve: 4.9–5.2 categories after the
first order, about 10 by the sixth.

| Instacart department | In the first order | By order 10 | Typically first bought at order |
|---|---|---|---|
| Produce | 75% | 95% | 1 |
| Dairy & eggs | 68% | 95% | 1 |
| Beverages / snacks | 46% / 45% | 86% / 88% | 1 |
| Frozen / pantry | 39% / 36% | 81% / 87% | 2 |
| Bakery / deli | 28% / 26% | 70% / 67% | 2 |
| Meat & seafood | 19% | 56% | 2 |
| Breakfast / household | 18% / 15% | 56% / 53% | 3 |
| Personal care | 10% | 44% | 3 |
| International | 8% | 35% | 4 |
| Babies / pets / alcohol | 5% / 2% / 3% | 17% / 7% / 7% | 2–3 |

**Grocery rhythm forms fast.** At X5, 54% of the customers who stay make
their second visit within 7 days (median 7). Later visits come every 4
days.

## Conclusion

1. **The H&M shape is general.** In fashion, music and wholesale (B2B):
   - the second order is the biggest drop;
   - each later order is more likely and sooner;
   - seven path families appear, with similar shares;
   - only three of them hold loyal customers;
   - loyalty falls off a cliff when the second order takes more than about
     60 days.
2. **The one-order share depends mostly on the business model**, from 36%
   (fashion) to 54% (music) to 97% (marketplace).
   - In a marketplace, nothing in the first order (category, basket,
     spend) changes it. Likely, the buyer's relationship is with the
     product search, not the store.
   - That is the first question for a store with "a huge share of
     one-order buyers": is it marketplace-like (search-led, one-off
     needs) or relationship-like?
3. **One-order buyers aren't all lost at day 90.**
   - A quarter to a third of those without a second order by day 90
     return within the year (H&M 37%, CDNOW 24%, wholesale 37%).
   - But those who return after day 120 rarely become loyal (4–6%).
   - So:
     - **Days 1–60** are the window to win a second order. The weekly
       chance falls 2–5-fold by week 12.
     - **After about day 90,** the job becomes cheap, infrequent
       reactivation rather than intensive nurturing.
4. **First-order size predicts one-and-done where baskets vary**
   (CDNOW, wholesale: 60–66% vs 37–40%), but hardly in fashion, and not at
   all in a marketplace. **Channel matters in fashion:** a store first
   order means 31% one-and-done vs 39% online.
5. **At category level, the entry category doesn't decide loyalty;** what
   happens next does.
   - In fashion, shoes and accessories buyers come back for clothes, and
     building an outfit edges ahead.
   - In grocery, staples come first (produce, dairy, beverages, snacks).
     Frozen, pantry, bakery and deli follow by the second order, then
     household, personal care and breakfast by the third.
   - That ordering is a natural **next-category sequence** for
     new-customer messaging. Whether nudging it *causes* loyalty needs a
     holdout test.
6. **Loyal customers are not broader, they are more frequent.** Category
   breadth at a given order number is the same for loyal customers and the
   rest (H&M). Expansion follows from frequency, not the other way round.
   This matches the earlier cross-dataset finding that a new category on
   the 2nd order doesn't raise the 3rd-order rate.

**For a store like Walmart (many one-order buyers):**
- **Tag new customers' paths by family at day 30, 60 and 90.** That gives a
  live view of where each cohort is leaking: never back (A), burst then
  gone (B), or fading (C).
- **Put the effort into the second order in the first 60 days.** Measure
  the incremental second-order rate with a holdout.
- **Split one-order buyers by business mode:** marketplace or one-off
  categories vs the core grocery relationship. Grocery customers who stay
  form a rhythm within 1–2 weeks, so a new grocery customer with no second
  visit after two weeks is already off-pattern.
- **Data needed:** full history from the first purchase, channel, the main
  category of each order, and message exposures with holdouts.

**Caveats:**
- **Observational:** a fast second order predicts loyalty; whether a
  message that speeds it up creates loyalty is untested.
- **X5 and Instacart** are selected (active or 4+ order shoppers). They
  describe customers who stay, not conversion.
- **Online Retail** is small (1,563 new customers) and B2B. **CDNOW** is
  from 1997, with no categories.
- **H&M's** follow-up overlaps 2020 store closures, and its channel coding
  is undocumented.
- **The loyalty definition is a choice.** The families and the cliff don't
  depend on it.
