# Reading list

Papers and industry write-ups behind the lifecycle and messaging plan
(`LIFECYCLE.md`, `VALIDATION-PLAN.md`), grouped by the decision they inform.

- **"Here"** says whether we checked the finding on public data, and where.
  Most of our checks are observational, so "holds" means the pattern
  appears, not that it is causal.
- **Concepts to look up** are in bold.

## Who to message: uplift, not risk or propensity

| Source | Finding | Here |
|---|---|---|
| Ascarza, "Retention futility," *JMR* 2018 | The customers most likely to churn aren't the ones a retention offer moves most; target by **uplift** | Partly: X5's surest buyers moved least (+1 point), high-risk deciles most (+4–5) (`experiments/2026-10-30-paper-checks.md`) |
| Ascarza, Iyengar & Schleicher, *JMR* 2016 | A proactive retention message raised churn (6% → 10%) | – |
| Lemmens & Gupta, "Managing churn to maximize profits," *Marketing Science* 2020 | Target by expected **profit impact** (uplift × value); R package ProfitBoost | Holds at tight budgets on X5 |
| Simester, Timoshenko & Zoumpoulis, *Management Science* 2020 | Randomize customers across *actions*, not policies; any future policy can then be evaluated, more precisely | Adopted for experiment 2.16 |
| Zhao & Harinen (Uber), "Uplift modeling for multiple treatments with cost optimization," 2019 | Multi-treatment uplift with per-treatment cost; in the CausalML library | – |
| ICML 2026, "Large-scale notification dispatch with bundle treatments and multi-outcome uplift optimization" | Uplift per (timing × style) bundle, assigned under platform quotas by Lagrangian relaxation; +0.5% DAU online | – |

## How much to message: volume, fatigue and a per-customer budget

| Source | Finding | Here |
|---|---|---|
| Gupta et al., "Email volume optimization at LinkedIn," *KDD* 2016 | Large volume cuts with little engagement loss | – |
| Gao et al., near-real-time notification optimization (LinkedIn), *KDD* 2018 | One decision across message types under volume limits | – |
| Zhao et al., "Notification volume control and optimization system at Pinterest," *KDD* 2018; Pinterest Engineering, user-state budgets (2020), NEP notification system (2024) | A weekly **budget per person** from predicted benefit; a **PID controller** tunes thresholds to hold volume on target | Design for 2.16 |
| LinkedIn offline reinforcement learning for notifications (arXiv 2202.03867, 2207.03029) | Greedy per-send optimization can hurt long-term engagement | – |
| Zhang, Kumar & Cosguner, *JMR* 2017 | 7 emails a month beat 4 or 10 by 16–32% of lifetime profit | – |
| Kumar, Zhang & Luo, *JMR* 2014 | Opt-out is a cost of every send | – |
| Drèze & Bonfrer, *J. of Interactive Marketing* 2008 | Too-short gaps between sends hurt more than too-long ones | – |
| Hohnhold, O'Brien & Tang, "Focus on the long-term," *KDD* 2015 | Users learn to ignore ads (ad blindness); measuring it led Google to cut mobile ad load 50% | No **wear-out** visible across Complete Journey campaigns months apart |
| Lewis (Yahoo display-ad field experiments) | Wear-out measured without randomization is overstated | – |
| Kronrod & Huber, *IJRM* 2019 | Annoyance from repetition can turn into preference later | – |
| Yancey & Settles (Duolingo), "A sleeping, recovering bandit…," *KDD* 2020 | **Recovering arms**: a template regains effect after a rest; novelty boosts short-term response | `experiments/2026-10-30-measurement-upgrades.md` |

## What to send: offers, content and timing

| Source | Finding | Here |
|---|---|---|
| Sahni, Zou & Chintagunta, *Management Science* 2017 | Offers work as advertising: 90% of the lift is not through redemption | Holds: 93% outside redeemed coupons in Complete Journey (trajectory-matched) |
| Lewis, "Customer acquisition promotions and customer asset value," *JMR* 2006 | Deep acquisition discounts lower repeat buying and value | Not on frequency: H&M deep-discount starters returned more, spent no more, kept buying on discount |
| Anderson & Simester, *Marketing Science* 2004 | Deep discounts raise new customers' later purchases, lower established customers' and raise deal-seeking | New: holds. Established: deal-seeking holds, fewer purchases doesn't |
| Tang, Lin & Sahni (DoorDash), "Business policy experiments using fractional factorial designs," 2023 | Factor a promotion and test a **fractional factorial**: 8 cells instead of 24, a policy with 5% more profit at 67% lower cost | Design for 2.18 |
| 84.51° (Kroger), controlled store test | A digital coupon didn't bring in new customers but won back lapsed ones | – |
| Li, Luo, Lu & Moriguchi, *J. of Marketing* 2021 | Cart reminders after 30–60 minutes reduced purchases; after 1–3 days they increased them | Basis for 2.2 |
| Sahni, Wheeler & Chintagunta, *Marketing Science* 2018 | Name in the subject line: +20% opens; a 2023 *Marketing Letters* replication found no effect | Not testable |
| Milkman et al., megastudies (*Nature* 2021; *PNAS* 2021) | Experts predict poorly which nudge wins; test many at once | Not testable |
| Dai, Milkman & Riis, "The fresh start effect," *Management Science* 2014 | Goal pursuit rises at temporal landmarks | No general effect on comeback timing |

## Experience and service

| Source | Finding | Here |
|---|---|---|
| Fisher, Gallino & Xu, *JMR* 2019 | +1.45% online sales per business day faster delivery | Holds: ~0.9% more repeat customers per day faster (Olist) |
| Gallino & Moreno, *Management Science* 2014 | Showing store availability online shifts sales between channels | Not testable |
| Instacart, item-replacement model | Predict availability without live inventory; candidates cover 95%+ of shoppers' picks | – |

## Loyalty: what it is and how it forms

| Source | Finding | Here |
|---|---|---|
| Reinartz & Kumar, *J. of Marketing* 2000; *HBR* 2002 | Loyal ≠ profitable: true friends, barnacles, butterflies, strangers | `experiments/2026-10-27-loyalty-dimensions.md` |
| Dick & Basu, *JAMS* 1994 | True vs **spurious loyalty** (convenience) vs latent | Validation 1.19 |
| Oliver, *J. of Marketing* 1999 | Loyalty in stages: believe → like → intend → act | – |
| Liu-Thompkins & Tam, *J. of Marketing* 2013 | Habit hinders cross-selling; attitude helps it | Experiment 2.13 |
| Du, Kamakura & Mela, *J. of Marketing* 2007 | **Share of wallet**: spend with us says little about spend elsewhere | Validation 1.18 |
| Ehrenberg; Sharp, *How Brands Grow* | **Double jeopardy**: most buyers split spend across stores | – |
| Meyer-Waarden & Benavent, *JAMS* 2009 | Heavy buyers join loyalty programs first; the effect is small and fades | Validation 1.20 |
| Datta, Foubert & Van Heerde, *JMR* 2015 | Free-trial customers churn faster; 59% lower profit | Not testable |
| Schmitt, Skiera & Van den Bulte, *J. of Marketing* 2011 | Referred customers are more valuable | Not testable |
| Anderson et al. (Spotify), *WWW* 2020; Holtz et al., *EC* 2020 | Diverse listeners retain better; personalization narrows diversity | Holds modestly: +3.9 points (H&M), +1.1 (REES46) at equal activity |

## Customer value and lapse timing

| Source | Finding | Here |
|---|---|---|
| Fader & Hardie 2001 (CDNOW); Fader, Hardie & Lee, *Marketing Science* 2005 | **BG/NBD**: when a customer has likely lapsed, from recency and frequency | MBG/NBD in `experiments/2026-10-27-loyalty-models.md` |
| Wang, Liu & Miao (Google), "A deep probabilistic model for customer lifetime value prediction," 2019 | **Zero-inflated lognormal** (ZILN) loss for value with many zeros and a heavy tail | `experiments/2026-10-30-measurement-upgrades.md` |
| Airbnb, listing lifetime value | Baseline vs incremental vs marketing-induced incremental value | Framing for "value" in uplift × value |

## Measuring it right

| Source | Finding | Here |
|---|---|---|
| Gordon et al. (Facebook), *Marketing Science* 2019 | Observational methods often miss experimental ad lift | X5: an SMS credited with ~19× its real lift without a holdout; Complete Journey: the naive campaign estimate had the wrong sign |
| Lewis & Rao, *QJE* 2015 | Marketing effects are small relative to noise; experiments need large samples | Sample sizes in VALIDATION-PLAN |
| Bernardi et al. (Booking.com), *KDD* 2019 | Better offline model metrics often didn't improve business metrics | Same as our signal screen: predictive ≠ uplift |
| Deng et al. (Microsoft), **CUPED**, *WSDM* 2013; Etsy, **CUPAC** (Code as Craft); Lin & Crespo (Etsy) 2024 | Adjust outcomes with pre-experiment covariates or an ML prediction: Etsy averaged 7% (CUPED) and 27% (CUPAC) variance reduction | `experiments/2026-10-30-measurement-upgrades.md` |
| Athey, Chetty, Imbens & Kang, **surrogate index**, 2019; Netflix, proxy metrics from past experiments, *KDD* 2024 and 2025 | Predict long-term effects from short-term metrics, learned from many past experiments | Experiment 2.17 and later |
| Spotify, cumulative holdbacks and "When proxy metrics break" | Individually positive tests can add up to a negative effect; keep a program-level holdback | Phase 0.1 |
| Klasnja et al., **micro-randomized trials**, *Health Psychology* 2015 (HeartSteps) | Randomize at each decision point with a fixed probability | Experiment 2.17 |
| Saito et al., Open Bandit Dataset and Pipeline, *NeurIPS* 2021 | Off-policy evaluation needs logged propensities and overlap | `experiments/2026-10-27-ope-dry-run.md` |
| Wayfair, geo experiments; Walmart Global Tech, "adopter analysis" | When customer-level tests are impossible: synthetic-control geo designs, difference-in-differences with matching | For store-level and fulfillment changes |
| Pinterest, "Field guide to non-engagement signals" | Survey and quality signals cost short-term retention and pay off later | A guardrail for engagement-trained message models |
