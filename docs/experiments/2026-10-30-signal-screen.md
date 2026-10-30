# 2026-10-30: Which behavioral signals predict future value? (screen on 8 datasets)

**Status:** done (predictive screen; not causal)

**Question:** Which customer signals predict value in the next months
*beyond RFM*, without duplicating each other? Signals that pass become
candidates for targeting and for randomized tests. Signals that fail are
dropped before anyone builds them.

**Script:** `docs/experiments/scripts/signal_screen.py`. Aggregates only;
cells under 50 customers are suppressed.

## Data

| Dataset | Business | Customers screened | Windows (observation → outcome) | Outcome | Signal families available |
|---|---|---|---|---|---|
| **H&M** | Fashion | 208k rows (1 in 5 customers, 2 cutoffs) | 90 days → 12 months | Spend | RFM, momentum, breadth, markdown share, channel |
| **CDNOW** | Music, first-ever buyers | 9.5k rows (2 cutoffs) | 90 days → 9 months | Spend | RFM, momentum |
| **Online Retail II** | B2B gifts | 4.8k rows (2 cutoffs) | 90 days → 12 months | Spend net of returns | RFM, momentum, returns, breadth (products) |
| **Olist** | Marketplace | 17k rows (2 cutoffs) | 90 days → 12 months | Spend | RFM, deliveries, reviews, vouchers, installments |
| **Complete Journey** | Grocery, frequent households | 4.6k rows (2 cutoffs) | 90 days → 8 months | Spend | RFM, momentum, departments, discounts, campaigns, coupons |
| **REES46** | Electronics-heavy store | 302k visitors (1 in 10) | 31 days → 30 days | Spend | RFM, sessions, views, carts, categories |
| **OTTO** | General online retailer | 1.08M visitors (sample) | 14 days → 13 days | Items ordered (no prices) | RFM, sessions, views, carts |
| **RetailRocket** | General online store | 240k visitors (1 in 4) | 90 days → 47 days | Items bought (no prices) | Sessions, views, carts, **stock-out views** |

**Not available anywhere:**
- email and push opens, and unsubscribes;
- searches and wishlists;
- support tickets;
- price-drop alerts;
- app vs web (H&M's store vs online is the closest).

## Method

1. **Cutoff and leakage audit.**
   - Features use only data before the cutoff. Olist deliveries and reviews
     count only if dated before it. H&M's markdown reference price is the
     article's highest price *before* the cutoff.
   - Tenure is left-censored at each dataset's start.
   - Monetary RFM overlaps with the target by construction. It is the
     baseline to beat, not a finding.
2. **Univariate screen:** Spearman correlation and mutual information
   against CLTV and against retention, plus coverage.
3. **Redundancy:** features clustered at |Spearman| > 0.7.
4. **Incremental value:** LightGBM on RFM, then RFM plus each family, then
   everything.
   - The holdout is 30% of customers, split by customer, not by row.
   - Metrics: R² on log CLTV, normalized Gini, top-decile CLTV capture, and
     AUC for retained.
   - Lifts come with 95% intervals from a paired bootstrap that resamples
     customers.
5. **Shape:** mean |SHAP| for importance, and SHAP by feature decile for
   thresholds.
6. **Segments:** steps 2–4 within lifecycle stages:
   - prospect: no purchase yet (event data only);
   - new: 1 order;
   - developing: 2–4 orders;
   - established: 5+ orders.
7. **Stability:** agreement of the univariate ranking between two cutoffs.

## Results

### 1. Lift over RFM, by signal family

ΔAUC for retained, with the 95% interval. ΔR² on log CLTV is in brackets
where it matters.

| Family | H&M | CDNOW | Online Retail | Olist | Complete Journey | REES46 | OTTO |
|---|---|---|---|---|---|---|---|
| RFM alone (AUC) | 0.81 | 0.63 | 0.73 | 0.52 | (99% retained) | 0.65 | 0.70 |
| **Momentum** | **+0.013** [0.011, 0.014] (R² +0.03) | **+0.025** [0.011, 0.038] | ±0.01 | ±0.02 | R² +0.013 (CI includes 0) | **+0.09** [0.085, 0.095] | **+0.08** [0.075, 0.083] |
| **Engagement** (sessions, views) | – | – | – | – | – | **+0.10** [0.097, 0.107] | **+0.11** [0.106, 0.112] |
| **Intent** (carts, repeat views) | – | – | – | – | – | +0.06 [0.055, 0.064] | +0.08 [0.079, 0.086] |
| **Lifetime orders** (all history) and breadth | **+0.019** [0.017, 0.020] (R² +0.057) | +0.025 [0.014, 0.037] | +0.015 [0.001, 0.029] | ±0.02 | ±0.02 | +0.08 (breadth viewed) | – |
| **Discounts and promotions** | +0.001 (R² 0) | – | – | ±0.02 | R² ±0.02 (CI includes 0) | – | – |
| **Friction** | – | – | Returns: ±0.01 | Late delivery, reviews: ±0.04 | – | – | – |
| **Channel** (online share) | +0.004 [0.003, 0.005] | – | – | – | – | – | – |

RetailRocket is left out: almost no anonymous visitor bought in the outcome
window, so nothing, RFM included, predicted anything.

### 2. The strongest single signals

| Signal | Evidence |
|---|---|
| **Order count over the customer's whole history** (not just 90 days) | Top feature by SHAP in H&M, CDNOW, Online Retail; Spearman ρ with CLTV ≈ 0.4–0.6. Clusters with tenure |
| **Gap ratio**: days since the last order ÷ the customer's usual gap (3+ orders) | Little linear correlation (ρ ≈ 0), but a clear threshold. Predicted value falls once a customer is **2–3.5× past their usual gap**, in H&M (above 3.75×), CDNOW (above 2.25×), Complete Journey (above 3.5×) and Online Retail (above 1.9×) |
| **Distinct products viewed** (visitors) | Top or near-top by SHAP in REES46 and OTTO; rises steadily past ~10 products |
| **Cart adds** | OTTO ρ 0.26; SHAP rises from the first add. Cart adds and abandoned-cart visits are one cluster (redundant) |
| **Share of visits in the last 7 days** | Momentum for visitors. Peaks when a third to three quarters of recent visits fall in the last week (REES46, OTTO) |
| **Very long visits** | *Lower* value: average visits over ~7–9 minutes have negative SHAP in both REES46 and OTTO, an inverted U |

### 3. Who can be predicted at all

| Stage | Transactions only (RFM + momentum + history) | With visit data (REES46, OTTO) |
|---|---|---|
| **Prospects** (no purchase yet) | – | AUC 0.69 / 0.74 (RFM alone: 0.50) |
| **New** (1 order) | H&M AUC 0.53 → 0.59 with everything; Olist 0.50 | REES46 0.54 → 0.61; OTTO 0.53 → **0.67** |
| **Developing** (2–4) | H&M 0.60 → 0.65; CDNOW 0.55 → 0.57 | 0.59 → 0.65 / 0.58 → 0.64 |
| **Established** (5+) | H&M 0.70 → 0.74; R² 0.35 → 0.43 | – |

In H&M, channel (0.533 → 0.557) and markdown share (→ 0.548) help *new*
customers more than any other family, though the effect is still small.

### 4. Redundancy and stability

**Clusters that recur:**
- order momentum with order slope;
- recency with the gap ratio;
- spend with AOV;
- tenure with lifetime orders;
- cart adds with abandoned-cart visits;
- views with distinct products viewed;
- browse-without-buying share with orders (it mostly says "didn't buy").

**Ranking stability across cutoffs:** Spearman 0.92–0.98, with 9–10 of
the top 10 shared (H&M, CDNOW, Online Retail, Complete Journey). Olist's
ranking is noise (−0.45), because its outcome is too rare.

## Classification

| Signal | Bucket | Reason |
|---|---|---|
| **Gap ratio** (cadence slip, 3+ orders) | **Predictive + actionable** | Replicates in 4 datasets with a clear 2–3.5× threshold. The trigger for per-customer reactivation timing (experiment 2.4) |
| **Cart adds / abandoned carts** | **Predictive + actionable** | Strong in event data. But predictive ≠ uplift: most cart returners come back anyway (`2026-10-27-cart-abandonment.md`), so target by uplift (experiments 2.2, 2.3) |
| **Recent-visit share** (last 7 days) | **Predictive + actionable** | Visitor momentum: a timing signal for "top of mind" and browse follow-ups (experiment 2.10) |
| **Channel mix** (online vs store) | **Predictive + actionable**, small | Helps most for new customers; second-channel invitation (experiment 2.15) |
| **Very long visits** | **Hypothesis** | Negative in two datasets; could be struggle (can't find, comparing). Needs search and navigation logs before any trigger |
| **Lifetime order count, tenure** | **Predictive, not actionable** | The strongest covariate. Compute frequency over the whole history, not just the observation window |
| **Engagement depth** (visits, distinct products viewed) | **Predictive, not actionable** (mostly a symptom) | Biggest lift where visit data exists (+0.10 AUC); makes prospects and new customers targetable. Use as a covariate and targeting input, not a lever |
| **Spend, AOV, recency** | **Predictive, not actionable** (baseline) | RFM |
| **Discount / markdown / coupon share, campaigns received, coupons redeemed** | **Weak or redundant** | H&M markdown share adds nothing. In Complete Journey, campaigns received (ρ 0.33) and coupon use (ρ 0.39) look predictive alone but add nothing beyond spend. Campaigns were targeted at heavy buyers: the selection bias the spec warns about |
| **Returns** (B2B) | **Weak or redundant** | Positively correlated with value (ρ +0.27): bigger buyers return more. A volume symptom, not churn friction |
| **Late deliveries, low reviews** (Olist) | **Inconclusive** | No lift, but only 2.8% of buyers return, so power is low. Matches `2026-10-27-loyalty-drivers.md` (late → low rating, barely lower repurchase) |
| **Stock-out views** (RetailRocket) | **Inconclusive** | Visitors almost never bought in the outcome window |
| **Order momentum ratio, weekly order slope** | **Redundant** | Near-zero correlation; the gap ratio carries the momentum signal for buyers |
| **Category breadth** | **Weak** | Small lift beyond frequency (H&M SHAP 0.008 vs 0.14 for lifetime orders). Matches `2026-10-28-second-order-category.md` |
| **Browse-without-buying share, items viewed 3+ times without buying** | **Redundant / weak** | The first mostly encodes "didn't buy"; the second adds little |

## Conclusion

1. **Use a customer's whole history, not a 90-day window.** Their order
   count over all time was the strongest signal in every transaction
   dataset. That is a one-line change to any RFM feature set.
2. **Cadence slip is the best trigger:** a customer 2–3.5× past *their own*
   usual gap is where predicted value falls. This matches the per-customer
   lapse timing from MBG/NBD (`2026-10-27-loyalty-models.md`), which is
   what experiment 2.4 tests.
3. **Visit data is what makes new customers predictable.**
   - From transactions alone, a new customer's future is close to a coin
     flip (AUC 0.53–0.59).
   - Engagement and recent-visit momentum lift that to 0.61–0.67, and make
     prospects rankable (0.69–0.74).
   - This is the strongest case for the data extract in Phase 0.2: visits
     and searches tied to customers.
4. **Promotion and campaign history looks predictive, but it is not**
   beyond spend. The campaign features show exactly the targeting bias the
   spec warns about. Keep them out of targeting models until randomized
   holdout data exists (Phase 0.1).
5. **Predictive ≠ uplift.** Every "predictive + actionable" signal still
   goes through a randomized test. Several of them (carts, high
   engagement) mark customers who would buy anyway.

**For the retailer's data:**
- Run the same script structure on the retailer's tables (validation
  1.21).
- Add the families the public data lacks: opens and unsubscribes,
  searches, support tickets, out-of-stock encounters, deliveries.
- **Expect:**
  - lifetime frequency and the gap ratio to replicate;
  - visit signals to matter most for new customers;
  - campaign-history features to collapse once spend is in the model.

**Caveats:**
- **Windows are short in event data** (REES46 31 → 30 days, OTTO 14 → 13
  days), and OTTO and RetailRocket have no prices. Value there is items
  bought.
- **Several populations are selected:**
  - Complete Journey covers frequent households only;
  - CDNOW is a single acquisition cohort;
  - Online Retail is B2B.
- **One model configuration,** with no tuning. Lifts are measured against
  that same configuration, so the comparison is fair, but absolute R²
  could be higher.
- **Visitor IDs in event data are cookies or sessions** (OTTO's "session"
  is a shopper's whole month), so prospects and repeat visitors are only
  approximately identified.
- **Predictive screens say nothing about causation or uplift.**
