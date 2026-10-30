# 2026-10-30: Measurement upgrades from industry practice

**Status:** done (public data; no engine change)

**Question:** Five ideas from industry data-science blogs (`docs/READING.md`),
checked on our datasets before they go into the plan:
1. Can a pre-experiment covariate shrink our experiments (Etsy's CUPAC)?
2. Which target should a customer-value model use when most customers buy
   nothing (Google's zero-inflated lognormal)?
3. Does a notification template lose effect when repeated and recover with
   rest (Duolingo's "recovering" bandit)?
4. and 5. Design changes with no data needed: a fractional factorial for
   offer tests (DoorDash), and a program-level cumulative holdback
   (Spotify). Both are now in `docs/VALIDATION-PLAN.md` as 2.18 and 0.4.

**Scripts:**
- `docs/experiments/scripts/cltv_models.py` reuses the signal screen's
  leakage-safe panels.
- `docs/experiments/scripts/duolingo_recovery.py`.

Aggregates only.

## 1. CUPAC: how much smaller can our experiments be?

**Method:**
- An experiment's outcome Y is adjusted by a covariate X measured before
  assignment. This is the control-variates idea behind **CUPED** and
  **CUPAC**.
- The variance falls by the share of Y that X explains (R²). The sample
  size needed for the same detectable effect falls by the same share.
- **Compared:**
  - **CUPED:** the same metric in the 90 days before assignment (spend, or
    orders for retention);
  - **CUPAC:** a cross-fitted LightGBM prediction from every pre-period
    signal in the screen;
  - **both together.**

**Variance reduction** (share of the sample size saved), all customers:

| Dataset | Retained (binary) CUPED → both | Spend CUPED → both | log(1 + spend) CUPED → both |
|---|---|---|---|
| H&M | 5% → **23%** | 37% → **48%** | 33% → **49%** |
| CDNOW | 4% → 6% | 36% → 39% | 13% → 16% |
| Online Retail (B2B) | 3% → 11% | **73% → 73%** | 21% → 28% |
| Complete Journey (grocery) | ~1% (everyone stays) | **73% → 73%** | 53% → 54% |
| REES46 (31-day windows) | 7% → 11% | 16% → 18% | 8% → 13% |
| OTTO (14-day windows) | 10% → 14% | 5% → 7% | 8% → 13% |
| Olist | 0% | 0% | 0% |

**Within lifecycle stages** (both covariates), which matters for a test
aimed at one stage:

| Stage | Retained | Spend |
|---|---|---|
| New (1 order): H&M / REES46 / OTTO | **3% / 3% / 5%** | 4% / 1% / 3% |
| Developing (2–4): H&M / REES46 / OTTO / CDNOW | 5% / 8% / 6% / 2% | 13% / 5% / 3% / 8% |
| Established (5+): H&M / REES46 / CDNOW / Complete Journey | 3% / 19% / 4% / 0% | **44% / 29% / 42% / 72%** |
| Prospects (no purchase): REES46 / OTTO | 3% / 4% | 0% / 2% |

**What it means for the plan's sample sizes:**

| Experiment | Population and outcome | Expected saving |
|---|---|---|
| 2.1 second-order nudge; 2.18 first-order offers | New customers, a binary 2nd order | **3–5%:** not worth the complexity. The factorial design in 2.18 is the bigger lever (four factors in one study) |
| 2.4 reactivation timing | Lapsing repeat customers, binary reactivation | 2–20% |
| 2.5 replenishment; 2.14 share of wallet; 2.16 message budget | Regular customers, spend or orders over 90 days | **40–70%:** sample sizes roughly halve |
| Program holdback (0.4) | Everyone, 12-month spend | 40–70% |

**Lessons:**
- **The earlier value of the same metric does most of the work for spend**
  in the transaction datasets (36–73%).
- **The ML prediction matters for binary outcomes** (H&M retention: 5% →
  23%) and for log spend.
- **Most of the overall gain comes from the gap between lifecycle stages.**
  A test confined to one stage gets the within-stage figure.
- **Etsy's ~27% average sits in the middle of our range.**

## 2. Customer value with many zeros: which target?

**Out-of-fold predictions of outcome-window value from the screen's
signals:**
- **log1p:** regression on log(1 + value), back-transformed naively, and with
  Duan's smearing correction.
- **Raw L2:** least-squares regression on raw value.
- **Zero-inflated lognormal (ZILN) as a hurdle:** P(buy) × exp(μ + σ²/2),
  with μ fitted on buyers' log value.

| Dataset (zero share) | Best ranking (normalized Gini) | Total predicted ÷ actual: log1p naive / raw L2 / ZILN | Decile calibration error: raw L2 / ZILN |
|---|---|---|---|
| H&M (18%) | tie (0.719 for all) | 0.88 / 1.00 / 1.14 | **1%** / 8% |
| Complete Journey (1%) | tie (0.83) | 0.87 / 0.99 / 1.28 | **7%** / 23% |
| Online Retail (23%) | ZILN 0.813 vs 0.80 | 0.56 / 1.00 / 1.32 | **25%** / 31% |
| CDNOW (33%) | ZILN 0.584 vs 0.55–0.57 | 0.40 / 1.00 / 1.13 | 16% / 16% |
| REES46 (92%) | ZILN 0.631 vs 0.60–0.63 | 0.07 / 1.00 / 1.10 | 16% / **6%** |
| OTTO (95%) | ZILN 0.663 vs 0.65 | 0.48 / 1.00 / 0.97 | 10% / **5%** |
| Olist (97%) | none works (Gini 0.11–0.17) | – | – |

**Lessons:**
- **Never back-transform a log model naively for value.** It understated
  total value by 12–93%. Smearing fixed the total in some datasets but
  overshot to 1.3–6.8× in others.
- **When most customers buy nothing** (event data, prospects), the hurdle
  ZILN was best on both ranking and calibration.
- **When most customers buy something,** ranking ties and plain regression
  on raw value is best calibrated. ZILN's single σ over-predicts the total
  by 13–32% there.
- **For the uplift × value targeting rule (2.11):**
  - rank with log1p or ZILN;
  - when a value *level* is needed (budgets, ROI), use raw-value regression,
    or ZILN in mostly-zero populations.

## 3. Notification novelty and recovery (Duolingo)

RESULTS_DUOLINGO

## Changes made to the plan

- **0.4:** a program-level cumulative holdback. 2–5% of customers get no
  lifecycle messages for 6–12 months.
- **0.5:** a pre-experiment covariate for every test, with the savings
  above used for sizing.
- **2.18:** a 2^(4-1) fractional factorial over depth, validity, minimum
  basket and channel, plus a no-offer arm and a holdout. Eight cells of
  about 5k estimate four main effects with the precision of a 20k-per-arm
  two-arm test.

**Caveats:**
- **Variance reduction here is computed without a treatment.** In a real
  test it applies to the comparison between arms, and needs covariates
  measured before assignment.
- **Event datasets have short windows,** and OTTO and RetailRocket values
  are item counts.
- **ZILN is fitted as a hurdle with gradient-boosted parts,** not the
  original single network.
