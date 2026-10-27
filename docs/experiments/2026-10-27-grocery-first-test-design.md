# 2026-10-27: Grocery first? A test design

**Status:** proposed design (to run on a retailer's own data; no public
dataset sells both grocery and general merchandise to the same customers)

## The hypothesis

"Start new customers in grocery, then move them into general merchandise
(GM), because most loyal customers are grocery buyers."

Two claims are mixed together here:
1. **Descriptive:** loyal customers are mostly grocery buyers. Almost
   certainly true.
2. **Causal:** getting a customer to start (or add) grocery *makes* them
   more loyal, including in GM. This is the claim worth money, and the
   first one doesn't show it.

## Why the descriptive fact doesn't prove the causal one

- **Frequency selection.** Grocery is the most frequent need (weekly
  rhythm: Instacart, X5). Anyone who buys grocery from you will look
  frequent, so loyal customers look like grocery buyers by construction.
- **Reverse causality.** Customers who already trust the store add grocery
  later, so loyalty can cause grocery adoption rather than the reverse.
- **Confounders:**
  - distance to a store, and delivery or pickup coverage;
  - household size, and whether they already have a grocery routine
    elsewhere.

  Each drives both grocery adoption and frequency. Some of these must
  never be *inferred* about a shopper (income, family status); a design
  that randomizes doesn't need them.
- **What public data shows on related questions:**
  - In H&M, the entry category barely changes loyalty (one-order rates
    35–40% whatever the first product group).
  - Across four datasets, a new category on the second order didn't raise
    the third-order rate.
  - Loyal customers are more frequent, not broader, at the same order
    count.

  None of that rules grocery-first out. It says "where they start" is not
  automatically a lever.

## Design A: historical data (fast, cheap, still biased)

1. **Event study of grocery adoption** (staggered difference-in-differences):
   - **Who:** GM-only customers who place their first grocery order in
     month *T*, compared with similar GM-only customers who adopt later
     (not-yet-treated controls).
   - **Matching** on the 6 months before *T*:
     - orders and GM categories;
     - tenure, channel, fulfillment eligibility, region.
   - **Outcomes** per month from −6 to +12:
     - **GM orders and GM spend.** Count GM separately, since mixed baskets
       inflate "orders" mechanically.
     - Total orders; visits and intent visits.
   - **Check pre-trends:** if adopters were already rising before *T*,
     the estimate is biased.
   - Concepts: **event study**; **staggered DiD** (Callaway & Sant'Anna
     2021); **propensity-score matching**; **sensitivity analysis** (e.g.
     E-values).
2. **Entry-category cohorts:** new customers whose first order is grocery
   vs GM, matched on the same pre-purchase information (acquisition
   channel, region, fulfillment method). Compare their path families and
   12-month outcomes, using `docs/experiments/scripts/paths_to_loyalty.py`
   with a loader for the store's data.

Both answer "is there a plausible effect, and for whom?" Neither gives a
causal estimate on its own.

## Design B: randomized encouragement (the causal answer)

You can't randomly *make* people buy grocery, but you can randomly *invite*
them. Concept: **randomized encouragement design**, analyzed with
**instrumental variables**.

- **Population:**
  - new or GM-only customers who have never bought grocery;
  - in areas where grocery pickup or delivery is available.
- **Treatment:** an invitation to try grocery.
  - Preferred, because they don't change price: grocery onboarding
    content, a starter list built from their GM purchases, app setup of a
    weekly list, and reminders at their natural shopping time.
  - A fee waiver or discount is a business decision. Our engine's rule is
    that recommendations never change the price a shopper is charged, so
    any offer must be uniform for the randomized group, not targeted by a
    price-sensitivity profile.
- **Control:** business as usual (a holdout).
- **Analysis:**
  - **Intention-to-treat (ITT):** the effect of being invited, the honest
    headline.
  - **Effect of actually adopting grocery,** among people who adopt
    because they were invited: ITT ÷ (difference in grocery take-up
    between arms). Concepts: complier average causal effect (CACE),
    local average treatment effect (LATE).
- **Outcomes:**
  - **12 months:** loyal (6+ orders and still buying), GM orders and GM
    spend, total margin.
  - **Leading indicators at 60–90 days:** second-order rate, visits and
    intent visits (they predict the next order:
    `2026-10-27-loyalty-drivers.md`).
  - **Guardrails:** unsubscribes, fulfillment cost per order, substitution
    complaints.
- **Heterogeneity:** distance to store, fulfillment method, the GM
  categories they started in, early rhythm. Concepts: conditional effects
  (CATE), causal forests.

**Sample size** (two-sided 5%, 80% power, base 12-month loyal rate 10%):

| Invite raises grocery take-up by | Adopting raises loyalty (compliers) by | ITT effect | Per arm |
|---|---|---|---|
| 30 points | 5 points | 1.5 points | **≈ 6,300** |
| 30 points | 3 points | 0.9 points | ≈ 17,400 |

**Decision rule:**
- **Move to grocery-first onboarding as the default** if the ITT on 12-month
  margin is positive and GM spend doesn't fall (grocery isn't just
  replacing GM).
- **Otherwise** keep onboarding based on the customer's own path: the
  second order in 60 days, then intent.

## What's needed from the data

- **Order lines with a grocery/GM flag,** from the account's first purchase.
- **Fulfillment method per order and the eligibility of the customer's
  area.**
- **Acquisition channel.**
- **Visits and searches.**
- **Message exposures, with holdout flags.**
