# 2026-10-30: Do published growth and messaging findings hold in our datasets?

**Status:** done. Mostly observational; one randomized check (X5).

**Question:** We collected papers that should shape the growth and messaging
plan (`docs/LIFECYCLE.md`, `docs/VALIDATION-PLAN.md`). Which of their
findings show up in the public datasets we have been using? Which don't,
and which can't be tested here?

**Script:** `docs/experiments/scripts/paper_checks.py`. Aggregates only;
cells under 50 customers are suppressed.

## What could and couldn't be tested

| Paper | Claim | Our test | Data |
|---|---|---|---|
| Lewis 2006 (JMR); Anderson & Simester 2004 (Marketing Science) | Deep acquisition discounts lower later value (Lewis); deep discounts *raise* later purchases by new customers (A&S) | First-order discount depth vs second order, 12-month loyalty and spend, adjusted for week, basket and channel | H&M (147k new customers), Olist vouchers (56k) |
| Anderson & Simester 2004 | Deep discounts *lower* established customers' later purchases and make them more deal-seeking | Discount depth in one quarter vs the next 6 months, adjusted for each customer's *earlier* discount share, frequency and spend | H&M (174k), Complete Journey (1.8k) |
| Sahni, Zou & Chintagunta 2017 (Management Science) | Offers work as advertising: most of the lift is not from redeeming the offer | Campaign difference-in-differences, split into coupon products, redeemed lines and other products, with a pre-trend placebo | Complete Journey (15 campaigns) |
| Hohnhold, O'Brien & Tang 2015 (KDD) | Repeated exposure wears off | Campaign lift by the household's 1st, 2nd, 3rd+ campaign | Complete Journey |
| Ascarza 2018 (JMR); Lemmens & Gupta 2020 (Marketing Science) | The highest-risk customers aren't the most persuadable; target by profit impact, not risk | Observed uplift by predicted risk decile; share of incremental purchases and value captured at fixed budgets by ranking rule | **X5 (200k, randomized SMS)** |
| Dai, Milkman & Riis 2014 (Management Science) | People act on goals at temporal landmarks | Do comeback orders after a lapse over-index on Mondays, month starts, early January vs regular orders? | H&M, CDNOW, Online Retail, Olist, Complete Journey |
| Fisher, Gallino & Xu 2019 (JMR) | Faster delivery raises sales (~1.45% per business day) | First-order delivery days vs repeat purchase, within state and month | Olist (55k) |
| Anderson et al. 2020 (WWW, Spotify) | More diverse consumers retain better at equal activity | Retention by category spread (entropy) within strata of activity | H&M (656k), REES46 (183k), Complete Journey |

**Not testable with these data:**
- Sahni, Wheeler & Chintagunta 2018 (name in the subject line);
- Milkman et al. (nudge megastudies);
- Datta, Foubert & Van Heerde 2015 (free trials);
- Schmitt, Skiera & Van den Bulte 2011 (referrals);
- Gallino & Moreno 2014 (in-store availability online);
- Holtz et al. 2020 (recommender experiment).

Gordon et al. 2019 (observational vs experimental ad lift) is mirrored by
our X5 result (`2026-10-27-x5-sms-uplift.md`): a promotional SMS was
credited with about 19× its real lift without a holdout.

## Results

### 1. Discount depth on the first order (new customers)

**H&M,** first order 2019-03-20..06-18 with no purchase in the previous 6
months. Depth = 1 − price ÷ the article's highest earlier price.

| First-order discount | Share | 2nd order ≤ 90 days | Loyal at 12 months | Orders in 12 months | Spend in 12 months (scaled) | Discount depth of later purchases |
|---|---|---|---|---|---|---|
| Full price (< 5% off) | 59% | 42.1% | 11.7% | 2.24 | 0.217 | 12% |
| 5–20% off | 28% | 42.2% | 12.3% | 2.31 | 0.203 | 15% |
| 20–40% off | 10% | 44.2% | 14.0% | 2.59 | 0.209 | 20% |
| 40%+ off | 4% | **46.5%** | **16.3%** | **2.93** | **0.185** | **29%** |

**Adjusted** for first-order week, spend quintile, items and channel, per
10 points deeper:
- 2nd order: +0.36 points [0.15, 0.58];
- loyal at 12 months: +0.57 points [0.42, 0.73];
- 12-month spend: +0.1% [0.0, 0.2];
- discount depth of later purchases: **+2.8 points** [2.7, 2.9].

**Olist:** paying mostly with a voucher went with 2.7% repeat within 180
days vs 2.0% without. Adjusted: +0.5 points [−0.1, +1.2].

### 2. A deep-discount quarter for established customers

**H&M:** 4+ orders in the 6 months before, and bought in the summer-sale
quarter (2019-06-19..09-17). Outcomes over the next 6 months:

| Discount depth that quarter | Earlier depth | Orders after | Spend after | Depth after | Still buying |
|---|---|---|---|---|---|
| Q1 (0–7%) | 9% | 4.54 | 0.52 | 14% | 91% |
| Q2 (7–17%) | 11% | 5.51 | 0.58 | 16% | 94% |
| Q3 (17–32%) | 15% | 5.63 | 0.53 | 21% | 94% |
| Q4 (32%+) | 23% | 5.37 | **0.44** | **29%** | 93% |

**Adjusted** for earlier depth, earlier and same-quarter orders and spend,
per 10 points deeper:
- orders after: +0.5% [0.3, 0.7];
- spend after: +0.25% [0.18, 0.32];
- depth after: **+2.0 points** [1.9, 2.0];
- reference price of items bought after: +0.5% [0.4, 0.6].

**Complete Journey** (1.8k households, Apr–Jun vs Jul–Dec):
- depth after: **+3.1 points** [2.5, 3.7] per 10 points;
- orders and spend: no detectable change (wide intervals);
- shelf price per unit bought: −6.7% [−15, +2].

### 3. Offers as advertising (Complete Journey campaigns)

RESULTS_OFFERS

### 4. Risk vs persuadability vs profit (X5, randomized)

The SMS raised purchases by 3.3 points on average (60.3% → 63.7%).

| Predicted risk of *not* buying without the SMS | D1 (lowest) | D3 | D5 | D7 | D8 | D9 | D10 (highest) |
|---|---|---|---|---|---|---|---|
| Purchase rate without the SMS | 95% | 80% | 65% | 49% | 41% | 34% | 22% |
| **Uplift** | **+1.0** | +2.5 | +3.1 | **+5.1** | **+5.2** | +3.8 | +4.3 |

**Share of all incremental purchases (and of incremental value) captured
at a fixed budget:**

| Budget (share of clients messaged) | Highest risk first | Highest predicted uplift | Highest uplift × value | Random |
|---|---|---|---|---|
| 10% | 13% (13%) | **21%** (15%) | 9% (**19%**) | 10% |
| 20% | 25% (26%) | **32%** (26%) | 18% (**31%**) | 20% |
| 30% | 41% (41%) | **43%** (34%) | 35% (**46%**) | 30% |
| 50% | **66%** (**72%**) | 60% (48%) | 57% (65%) | 50% |

### 5. Fresh start: do comebacks cluster at temporal landmarks?

A comeback is an order 60+ days after the customer's previous one (28+ in
grocery). Each cell is the share of comebacks on the landmark, against the
share of regular repeat orders.

| Dataset (comebacks) | Monday | First 3 days of the month | First 2 weeks of January |
|---|---|---|---|
| H&M (1.87M) | 12.6% vs 12.7% (×1.00) | 9.9% vs 9.9% (×1.00) | 3.0% vs 3.5% (×0.85) |
| CDNOW (15.7k) | 15.8% vs 15.1% (**×1.05**) | 9.9% vs 9.4% (×1.05, n.s.) | 2.6% vs 2.4% (n.s.) |
| Online Retail, B2B (8.0k) | 16.4% vs 15.1% (**×1.08**) | 8.6% vs 8.9% (n.s.) | 1.1% vs 2.6% (×0.41) |
| Olist (1.2k) | 16.2% vs 16.8% (n.s.) | 10.2% vs 10.8% (n.s.) | – |
| Complete Journey (3.9k) | 14.0% vs 13.8% (n.s.) | 10.9% vs 9.6% (**×1.14**) | 0.2% vs 2.5% (×0.06) |

### 6. Delivery speed (Olist)

| First order delivered in | 0–7 days | 8–14 | 15–21 | 22–30 | 31+ |
|---|---|---|---|---|---|
| Repeat within 180 days | 2.32% | 2.02% | 1.89% | 1.73% | 1.26% |

**Within state and month** (controls: order value and freight share):
−0.019 points per extra day [−0.033, −0.006]. That is **about 0.9% more
repeat customers per day faster**, against a 2.0% base. Adding whether
the order was late and how long was promised leaves speed at −0.023 per
day; lateness itself adds nothing (+0.2 points [−0.5, +0.8]).

### 7. Diversity at equal activity

Retention for the most vs least diverse third of customers, within strata
of activity (items and active days):

| Dataset | Retained overall | Most minus least diverse, equal activity | Per unit of entropy (linear, activity controls) |
|---|---|---|---|
| H&M (product groups; 180 → 180 days) | 67% | **+3.9 points** | +8.0 points [7.7, 8.3] |
| REES46 (categories viewed; Oct → Nov purchase) | 10.5% | +1.1 points | +3.4 points [3.1, 3.8] |
| Complete Journey (departments; 90 → 90 days) | 96% | +0.2 points | +3.5 points [0.0, 7.1] |

## Conclusion

| Paper | Holds here? | What we found |
|---|---|---|
| **Lewis 2006** (deep acquisition discounts lower value) | **Not on frequency; partly on value** | Deep-discount starters came back *more* often (loyal +4.6 points raw at 40%+ off; +0.6 per 10 points adjusted). But they spent no more over 12 months (−15% raw, ±0 adjusted) and kept buying on discount (29% vs 12% depth) |
| **Anderson & Simester 2004, new customers** (deep discounts raise later purchases) | **Yes** | Same direction as their catalog experiments |
| **Anderson & Simester 2004, established customers** (deep discounts lower later purchases, raise deal-seeking) | **Deal-seeking: yes. Fewer purchases: no** | After a deep-discount quarter, discount depth rose about 2–3 points per 10 points of depth, even controlling for each customer's earlier discount habit (H&M and Complete Journey). Frequency didn't fall |
| **Sahni et al. 2017** (offers as advertising) | SEE_OFFERS | SEE_OFFERS_DETAIL |
| **Hohnhold et al. 2015** (wear-out) | **Not testable here** | Campaign lifts by count were within noise |
| **Ascarza 2018** (highest risk ≠ most persuadable) | **Partly** | The surest buyers had the smallest uplift (+1 point). But uplift peaked in the high-risk 7th–8th deciles (+5 points) and stayed high at the very top. For a promotional SMS, risk and persuadability went together more than Ascarza found for retention offers |
| **Lemmens & Gupta 2020** (target by profit impact) | **Yes at small budgets** | At 10–30% budgets, uplift × value captured the most incremental *value* (19–46%), while uplift alone captured the most *purchases*. At a 50% budget, simple risk ranking won both: uplift models add most when the budget is tight |
| **Dai et al. 2014** (fresh start) | **No general effect** | No Monday or month-start over-index in H&M, Olist or Complete Journey. Small Monday effects in CDNOW and B2B (week restart), a month-start effect in grocery (paydays, likely). Comebacks *under*-index in early January |
| **Fisher et al. 2019** (delivery speed) | **Yes, similar size** | ~0.9% more repeat customers per day faster within state and month, vs their 1.45% more sales per business day. Speed mattered; being late vs the promise added nothing beyond speed |
| **Anderson et al. 2020** (diversity → retention) | **Yes, modestly** | +3.9 points in H&M and +1.1 in REES46 at equal activity; flat in grocery, where everyone is retained |

**What this changes in our plan:**
1. **Acquisition discounts:** the risk isn't that deep-discount starters
   leave. It's that they become discount shoppers, with lower spend and a
   higher discount share for the next year. Judge first-order offers on
   **12-month margin**, not repeat rate. The proposed first-order
   offer-depth experiment should report margin and the later discount
   share.
2. **Established customers:** a deep-discount season leaves a lasting
   discount habit, beyond their earlier habit. Count that cost in any
   promotion's business case (VALIDATION-PLAN 1.16, 2.11).
3. **Targeting rule (experiment 2.11):**
   - With a tight budget (≤ 30%), rank by predicted uplift × value.
   - With a loose budget, a simple risk ranking does as well.
   - Always exclude the surest buyers.
   - This refines the earlier X5 "skip the likeliest buyers" rule.
4. **Send timing:** don't build fresh-start (Monday or month-start) send
   rules. If anything, test month-start timing for grocery inside the
   micro-randomized trial (experiment 2.17).
5. **Delivery speed is a retention lever** of the same order as a message
   program. Measure it in Phase 0.2 (promised vs actual times) and include
   it in validation 1.6.
6. **Diversity:** a modest positive association, consistent with the
   second-order and signal-screen findings (breadth adds a little). It
   doesn't justify pushing new categories without the 2.1 test.

**Caveats:**
- **Everything except X5 is observational.** Deep-discount buyers, deep
  discount seasons and targeted households differ in ways the controls
  don't capture.
- **H&M reference prices are the article's highest earlier price,** so
  "depth" mixes promotions with ordinary price changes. Spend is in H&M's
  scaled units.
- **X5 has no spend outcome.** Value captured = purchases × the client's
  earlier average basket.
- **Olist vouchers are a payment method,** not necessarily a discount.
  Olist repeat is rare (2%).
- **Complete Journey** has only frequent households and 15 usable
  campaigns.
