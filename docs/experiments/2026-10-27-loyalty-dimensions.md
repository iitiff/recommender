# 2026-10-27: What "loyal" means: habit vs attitude, loyal vs valuable

**Status:** done (literature review plus two analyses: one randomized, one descriptive)
**Data:** aggregates only, behavior only.
- **X5 RetailHero:** a randomized promotional SMS campaign with purchase history (grocery).
- **H&M:** the new-customer cohort of `2026-10-27-paths-to-loyalty.md`.

## Question

Our "loyal" tier was **6+ orders in 12 months and still buying at the end**.
Research says loyalty has more dimensions. Which ones matter, and what does
the data say about two of them:
1. **Habit vs attitude** (Liu-Thompkins & Tam 2013): do habitual repeat
   customers respond differently to a message?
2. **Loyal vs valuable** (Reinartz & Kumar 2000, 2002): how much of our
   loyal tier is profitable "true friends", and how much is low-value
   "barnacles"?

## What the literature adds

| Idea | Source | Takeaway |
|---|---|---|
| Attitude × behavior: true, spurious, latent, no loyalty | Dick & Basu, *JAMS* 1994 | Frequent buying can be convenience ("spurious"), and liking without buying can be blocked demand ("latent"). Empirical support is mixed (East et al.'s supermarket test) |
| Loyalty develops in stages: believe → like → intend → act | Oliver, *J. of Marketing* 1999 | Repeat purchase is the last stage; it can be inertia |
| Habit vs attitudinal loyalty | Liu-Thompkins & Tam, *J. of Marketing* 2013 | Attitude helps cross-selling, habit hinders it. A cross-sell message can break a habit; a promotion designed around the habit works |
| Loyal ≠ profitable: true friends, barnacles, butterflies, strangers | Reinartz & Kumar, *J. of Marketing* 2000, *HBR* 2002 | Tenure is only weakly linked to profit |
| Habits beyond buying (promotion, return, low-margin) | Shah, Kumar & Kim, *J. of Marketing Research* 2014; follow-up work | Recurring non-purchase behaviors drive profit |
| Share of wallet | Du, Kamakura & Mela, *J. of Marketing* 2007; Keiningham et al., the "Wallet Allocation Rule" | Spend with us says little about spend elsewhere; share depends on our *rank* among the customer's stores |
| "Polygamous" loyalty, double jeopardy | Ehrenberg; Sharp, *How Brands Grow* | Most buyers split their spend across stores; growth comes mostly from reaching more buyers |
| Value beyond purchases | Kumar et al., *J. of Service Research* 2010 | Referrals, influence and feedback add value |
| Loyalty programs | Meyer-Waarden & Benavent, *JAMS* 2009 | Heavy buyers join first; the effect is small and fades within 6–9 months |

## Method

```bash
uv run --no-project --with pandas python -I \
  docs/experiments/scripts/loyalty_dimensions.py x5 X5_DIR
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/loyalty_dimensions.py hm TRANSACTIONS_DIR
```

**Habit strength (X5):** context stability in the purchase history, for
clients with 5+ visits.
- **Four measures:**
  - share of visits on their usual weekday;
  - share in their usual 3-hour window;
  - share at their usual store;
  - regularity of the gaps between visits.
- **Ranked within visit bands** (5–10, 11–30, 31+), since these shares
  depend on how many visits there are.
- **The habit index** is the mean rank.
- **Uplift** is the SMS group's purchase rate minus the holdout's, by habit
  tercile within each band.

**Loyal × valuable grid (H&M):** among repeat customers.
- **Loyal:** our tier.
- **Valuable:** 12-month spend at or above the median.
- **Second version:** counts only full-price spend, as a margin proxy.
- **Promotion habit:** share of items bought on sale (below 90% of the
  article's median price).

## Results

### 1. Habit vs response to a promotional SMS (X5, 177,363 clients, randomized)

| Visits | Habit low | Habit mid | Habit high |
|---|---|---|---|
| 5–10 (control buys 41%) | **+5.0** [3.5, 6.6] | +4.0 | +3.8 [2.3, 5.4] |
| 11–30 (66%) | +2.1 [1.0, 3.1] | +3.4 | **+4.1** [3.1, 5.2] |
| 31+ (87–91%) | +1.0 [−0.1, 2.2] | +1.3 | +1.7 [0.8, 2.7] |
| **All** | +2.7 | +3.1 | +3.3 |

- Uplift is in points.
- High minus low habit, weighted across bands: +0.9 points.
- The four habit measures barely correlate with each other (0.0–0.2).

### 2. Loyal × valuable (H&M, 93,366 repeat customers)

| Group (by spend) | Share of repeaters | Orders | Spend vs average | Items bought on sale | Single-channel share |
|---|---|---|---|---|---|
| **True friends** (loyal, valuable) | 17.4% | 9.8 | 2.08× | 27% | 0.81 |
| **Barnacles** (loyal, low value) | 1.8% | 6.9 | 0.54× | **37%** | 0.84 |
| **Butterflies** (not loyal, valuable) | **32.6%** | 4.4 | 1.36× | **19%** | 0.88 |
| **Strangers** | 48.2% | 2.9 | 0.38× | 24% | 0.89 |

- **Counting only full-price spend:**
  - Barnacles rise to 2.9% of repeaters (half their items on sale).
  - True friends fall to 16.3%.
  - 7.8% of loyal "true friends" become barnacles.
- **Loyal rate and spend by promotion habit** (quartiles of the share of
  items bought on sale):

  | | Q1 (least) | Q2 | Q3 | Q4 (most) |
  |---|---|---|---|---|
  | Loyal | 4.5% | 22.4% | 28.9% | 21.1% |
  | Full-price spend vs average | 0.89× | 1.47× | 1.10× | **0.54×** |

## Conclusion

1. **Habit doesn't weaken the response to a promotion.**
   - Within the same visit band, habitual and non-habitual clients respond
     about equally (difference +0.9 points). Among mid-frequency shoppers,
     the habitual ones respond *more* (+4.1 vs +2.1).
   - This matches Liu-Thompkins & Tam: habit hurts *cross-selling* (a new
     category), not promotions that fit the routine.
   - So the open test is cross-selling. Does a new-category message work
     less, or even backfire, for habitual grocery shoppers? That is
     experiment 2.13 in the validation plan.
   - The habit measures barely correlate, so "habit" is several things
     (time, day, place, rhythm). Measure them separately.
2. **In fashion, frequency-defined loyalty mostly lines up with value.**
   - 85–91% of loyal repeaters are "true friends".
   - **Barnacles are few (2–3% of repeaters) and are sale buyers:** 37–50%
     of their items on sale, half the value.
   - The bigger blind spot is **butterflies**: a third of repeat
     customers are high-value, full-price buyers who don't buy often enough
     to count as loyal. A loyalty program built on frequency would
     overlook them. Reinartz & Kumar's advice is to make the most of them
     while they buy, not to force frequency.
3. **Promotion habit splits loyalty from value.**
   - Customers who never buy on sale are rarely loyal (4.5%). Part of that
     is mechanical: fewer orders mean fewer chances to hit a sale.
   - The heaviest sale buyers are often loyal (21%) but bring the least
     full-price spend (0.54×).
   - Loyal *and* valuable sit in the middle.
4. **Loyal customers use more channels:** true friends' single-channel
   share is 0.81, vs 0.89 for strangers. This is consistent with the
   omnichannel finding.
5. **What we still can't measure on public data:**
   - share of wallet (needs outside spend);
   - attitude (needs ratings, surveys or preference signals);
   - resilience after shocks (needs enough repeat customers who hit a
     stock-out, late delivery or price change).

   These go to the validation plan.

**A loyalty profile instead of one tier:**

| Dimension | Measure | Source of evidence |
|---|---|---|
| Depth | Frequency, regularity, tenure, expected purchases (MBG/NBD) | Our tier + `2026-10-27-loyalty-models.md` |
| Value | Margin after promotions and returns: true friend / barnacle / butterfly / stranger | This write-up |
| Driver | Habit (time, day, place, rhythm consistency) vs attitude (ratings, reviews, intent searches) | This write-up; Liu-Thompkins & Tam |
| Share of wallet | Estimated from a receipt panel or survey, by category | Du et al.; Keiningham et al. |
| Engagement | Intent visits, lists, reviews, referrals | `2026-10-27-loyalty-drivers.md`; Kumar et al. |
| Resilience | Behavior after a stock-out, late delivery or price change | Dick & Basu (spurious loyalty) |

**Caveats:**
- **The X5 message was a promotion** with undisclosed content, so the
  habit result is about promotions, not cross-selling.
- **Habit is measured from about 4 months of history.**
- **H&M has no margin:** full-price spend is a rough proxy, and value is
  measured over the same 12 months as loyalty.
- **The grid's median split is a choice.**
