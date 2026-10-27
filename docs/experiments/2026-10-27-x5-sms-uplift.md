# 2026-10-27: Who does a message persuade? Uplift of a randomized SMS campaign (X5 RetailHero)

**Status:** done (randomized experiment, re-analyzed)
**Data:** X5 RetailHero uplift dataset, from a Russian grocery chain.
- 200,039 loyalty-card clients, randomly sent an SMS or not, with whether
  they bought during the campaign.
- Purchase history before the campaign: Nov 21, 2018–Mar 18, 2019.
- License not stated (competition data). Aggregates only; every cell has
  ≥ 50 clients.
- The dataset's age and gender columns are **not used**: the profile is
  built from behavior.

## Question

Every recent write-up ended on the same open question: **who does a message
actually persuade?** Organic data can only show who buys (propensity), not
who buys *because of* the message (uplift). This dataset has a randomized
holdout, so we can test the profile layers directly:
- **Rhythm:** are overdue shoppers more persuadable than those within their
  cycle?
- **Stage and frequency:** are the most loyal shoppers sure things?
- **Price and deal orientation:** do they change who responds?
- **Targeting:** how much better is a model than messaging everyone, or than
  a simple rule?

## Method

```bash
uv run --no-project --with pandas --with scikit-learn python -I \
  docs/experiments/scripts/x5_uplift.py DIR
```

**Features**, all from the purchase history before the campaign:
- visits (transactions);
- days since the last visit;
- median gap between visits;
- **rhythm state** (days since last visit ÷ own median gap; the states in
  LIFECYCLE.md);
- tenure (loyalty card age);
- average basket;
- share of visits that spent loyalty points;
- category breadth (distinct level-2 categories);
- price level: spend-weighted log ratio of the client's unit price to the
  median in the same level-4 category. This is rough, since pack sizes vary.

**Analysis:**
1. **Randomization check:** standardized mean differences (SMD) between
   treated and control.
2. **Uplift by segment:** purchase rate treated minus control, with a 95%
   CI.
3. **T-learner:**
   - One gradient-boosted classifier on treated clients and one on
     control; uplift = the difference of their predictions.
   - Cross-fitted over 5 folds, so each client is scored by models that
     never saw them.
   - Evaluated by observed uplift per predicted decile, and by the share of
     all incremental purchases captured when messaging the top *k*%.
     Concept: **uplift / Qini curve**.
   - Compared with a simple rule: message the least likely buyers first,
     ranked by the control model's predicted propensity.

**What we don't know:**
- The SMS content. The competition calls it a *promotional* campaign, but the
  offer (discount, bonus points or other) isn't disclosed.
- The length of the campaign window.

So the results describe the effect of a **promotional offer**, not of a
message in general.

## Results

**Randomization holds:** every SMD is below 0.01.

**Average effect**

| | Control | SMS | Uplift |
|---|---|---|---|
| Bought during the campaign | 60.3% | 63.7% | **+3.3 points** [2.9, 3.8] |

95% of the treated clients' purchases would have happened without the SMS.
A last-touch report would credit the SMS with 63.7% conversion. Only
1 purchase in 19 among those messaged was caused by it.

**Uplift by segment (points; the 95% CI is about ±0.6–2 points per cell)**

| Layer | Segment (n) | Control | Uplift |
|---|---|---|---|
| **Rhythm** | Within cycle, < 0.75× own gap (92,546) | 63.4% | +3.4 |
| | Due, 0.75–1.25× (33,251) | 61.8% | +3.1 |
| | Overdue, 1.25–2× (28,538) | 61.9% | +3.0 |
| | Lapsing, > 2× (41,884) | 55.3% | +3.4 |
| | 1 visit only (3,820) | 15.9% | +5.6 |
| **Visits** | 1–3 (15,731) | 22.9% | +4.4 |
| | 4–10 (52,872) | 39.8% | +4.3 |
| | 11–30 (91,703) | 66.4% | +3.2 |
| | **31+ (39,733)** | **88.7%** | **+1.3** |
| **Recency** | 0–3 days (76,067) | 78.2% | +2.6 |
| | 8–14 days (48,872) | 45.0% | +4.5 |
| | 15–30 days (26,597) | 34.5% | +3.8 |
| **Tenure** | < 6 months (36,093) | 58.2% | +4.8 |
| | 1–2 years (134,211) | 61.7% | +2.8 |
| **Points** | **Often spends points, > 25% of visits (8,322)** | 40.0% | **+8.4** |
| | Never spends points (79,146) | 50.1% | +3.2 |
| **Basket** | Smallest quintile | 64.9% | +5.4 |
| | Largest quintile | 53.4% | +2.6 |
| **Breadth** | Fewest categories (quintile) | 39.6% | +6.0 |
| | Most categories (quintile) | 76.9% | +1.2 |
| **Price level** | Lowest quintile (cheapest choices) | 61.2% | +4.7 |
| | Quintiles 3–5 | 53.9–62.7% | +2.4 to +3.0 |

**Propensity vs uplift** (deciles of the control model's predicted purchase
chance)

| Propensity decile | 1 (likeliest) | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Control purchase rate | 94.9% | 87.3% | 79.9% | 74.2% | 64.4% | 57.3% | 49.2% | 41.1% | 33.6% | 21.8% |
| Uplift (points) | **+1.0** | +1.8 | +2.6 | +2.0 | +3.7 | +3.1 | **+4.8** | **+5.3** | +3.8 | +4.3 |

**Targeting: share of all incremental purchases captured**

| Message only the top… | 10% | 20% | 30% | 50% | 70% |
|---|---|---|---|---|---|
| By predicted uplift (T-learner) | **22%** | 33% | **46%** | 61% | 77% |
| By lowest propensity first (simple rule) | 13% | 25% | 41% | **65%** | **80%** |

- The top decile by predicted uplift gains +7.3 points, 2.2× the average.
- No decile has a negative uplift, so no "sleeping dogs" are detectable.
- Per 1,000 messages: 33 incremental purchases when messaging everyone, 73
  in the top decile, 51 in the top 30%.

## Conclusion

1. **Attribution overstates the promotion's value about 19-fold here.** The
   SMS lifted purchases by 3.3 points on a 60% base. Any report that counts
   purchases after a message, without a holdout, is mostly measuring
   propensity. This makes a holdout a standing requirement, not a one-off.
2. **The likeliest buyers are the least persuadable.**
   - Uplift falls from +4–5 points in the middle and lower propensity
     deciles to **+1.0 in the top decile**, where 95% buy anyway. These
     are the "sure things".
   - Heavy, broad, recent shoppers (31+ visits, most categories, last 3
     days) gain the least.
   - New, narrow and occasional shoppers gain the most: 1–10 visits,
     tenure under 6 months, fewest categories.
   - Same lesson as the cart work: **high intent ≠ high value**.
3. **Deal orientation is the strongest single signal, because this was a
   promotion.**
   - Clients who often spend points gain **+8.4 points**, 2.5× the
     average. So do those buying the cheapest options (+4.7) and small
     baskets (+5.4).
   - That is close to saying "people who use deals respond to a deal". It
     tells us little about reminders or recommendations that carry no
     incentive; those need their own tests.
   - **It must not decide who gets a lower price.** Sending discounts only
     to deal-responsive shoppers means different shoppers pay different
     prices: personalized pricing. The engine's rule is that price never
     changes the price a shopper is charged. Changing that would be an
     explicit business and ethics decision, not a modeling one.
   - **The economics are unknown.** The outcome is purchases, not margin.
     A discount is also paid to the 95% who would have bought anyway, if
     they redeem it. Net value = margin on the 3.3 points of extra
     purchases minus the discount on *all* redemptions. A lift this size
     on a 60% base can lose money. Without the offer's cost and redemption
     data, this dataset can't say.
   - Repeated offers can also train shoppers to wait for one (the cart
     write-up's warning).
4. **Rhythm state did not change the effect of this SMS.**
   - Within cycle, due, overdue and lapsing all gain about +3 points.
   - This tests the decision sketch in
     `2026-10-27-replenish-or-explore.md`, which proposed staying quiet
     when shoppers are within their cycle.
   - For a generic offer SMS, that sketch is **not supported**: in-cycle
     shoppers responded as much as overdue ones.
   - The sketch may still hold for a *replenishment reminder*, whose
     content depends on what is due. That needs its own test.
   - The decision sketch and LIFECYCLE.md are updated to say so.
5. **Targeting: a model helps at the top; a simple rule is nearly as good
   for broad sends.** For a promotion, targeting decides who is *offered* a
   lower price, so the rule in point 3 applies. The targeting lessons
   transfer to messages without an incentive (skip the sure things), but
   their sizes don't.
   - Messaging the top 10% by predicted uplift captures 22% of incremental
     purchases (2.2×). The top 30% captures 46%.
   - "Message the least likely buyers first" is simpler, needs no
     randomized training data, and captures 65% at half the sends.
   - Next-best-action can start with the propensity rule (skip the sure
     things), then move to an uplift model once our own holdouts exist.
     Rules → propensity → uplift is the roadmap order.

**What this means for the profile and for data requests:**
- **Deal responsiveness is not a targeting trait for discounts** (see
  point 3).
  - At most, it could shape *content* within public prices: for example,
    surfacing items already on sale to everyone, or value packs. That use
    would need its own test.
- **For offers, ask for cost and redemption data:** offer value, who
  redeemed, and margin. Without it, an offer's uplift can't be turned into
  profit.
- **Message logs with holdout flags:** confirmed again as the most valuable
  missing data.

**Caveats:**
- **One campaign, one chain, one channel**, with unknown content and an
  unknown window. The effect of a different message (replenishment,
  new-category) may differ.
- **The outcome is any purchase,** not revenue or margin. An offer can lift
  purchases while cutting margin.
- **The results apply to a promotion:** the content of the SMS is not
  disclosed, but the campaign was promotional.
- **Decile CIs are about ±1.4 points:** small differences between
  neighboring deciles are noise.
- **Segments overlap:** heavy shoppers are both recent and broad. The
  T-learner handles the overlap; the one-way tables don't.
