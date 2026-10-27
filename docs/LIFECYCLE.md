# Lifecycle personalization: re-engagement's real job

Re-engagement isn't "send an email when someone leaves". Its job is to move
each customer one step along the lifecycle, from visitor to first order to
second order to loyal customer, with the least messaging that actually changes
behaviour.

## Stages and the job at each one

| Stage | Goal (next step) | What helps | What to avoid |
|---|---|---|---|
| **Visitor** (browsing, anonymous) | First real intent (cart, sign-up) | On-site recs, session-based "you might also like" | Messaging people who never opted in (you can only email or push the reachable) |
| **Carted, not bought** | First order | Abandoned-cart message with the carted item first | Discounts for people who'd buy anyway |
| **First-time buyer** | **Second order** (often the bottleneck: 2–5% in a marketplace and a general store; much higher in electronics, fashion, B2B) | Post-purchase: replenishment if the product repeats, complements, "how's it going?" | Generic newsletters |
| **Repeat buyer** | Habit and **breadth** | A mix of next-transaction and *adjacent*-category content | Pushing unrelated categories |
| **Loyal** (3+ orders) | Retention | New arrivals in their categories, early access, fewer but better messages | Over-messaging (these are the people most likely to unsubscribe if annoyed) |
| **Lapsing** (past their usual gap) | Win-back | A reason to return, timed from their own purchase rhythm | Treating a 2-day gap and a 60-day gap the same |

## Rhythm: due, overdue, lapsing

Stage says where a customer is in the relationship. **Rhythm** says where
they are in their own buying cycle. The two decide between a replenishment
reminder, a new-category campaign, and silence.

The rhythm is measured against **their own** usual gap (median days between
orders), not a store-wide rule. In groceries (Instacart,
`docs/experiments/2026-10-27-replenish-or-explore.md`), a regular weekly
shopper orders on day 7 with a 37% chance, then 24% on day 8 and 15% on
day 9; 55% of next orders come within 1× their gap, 76% within 1.5×.

| Rhythm state (days since last order ÷ their usual gap) | What it means | Default message |
|---|---|---|
| Within cycle (< 0.75) | Stocked up | No replenishment; new-category ideas if early in the lifecycle |
| Due (0.75–1.25) | Most will order anyway | Usually nothing (sure things) |
| **Overdue** (1.25–2) | Missed their usual slot | **Replenishment reminder**: their usual basket, items whose own cycle has passed first |
| Lapsing (> 2) | Drifting away | Win-back |

**Tested in part:** in a randomized grocery promotional SMS campaign (X5,
`docs/experiments/2026-10-27-x5-sms-uplift.md`), the rhythm state did not
change the message's effect: within cycle +3.4 points, overdue +3.0, lapsing
+3.4. Treat the defaults above as hypotheses for *replenishment reminders*
specifically, not as a rule for every message.

Products have their own, longer cycles (milk 11 days, eggs 15, cheese 21),
so a reminder replenishes the *basket* and lists the items most likely due.

New categories fit early in the lifecycle (orders 2–10: about half the items
are new to the shopper) and still in 1 of 4 orders for veterans. Success is
adoption (bought again within 3 orders: 38% of trials), not a single trial.
The exposure cap depends on how the category is bought: about 2–3 showings
for need-driven and high-ticket categories (electronics, computers), which
convert early and wear out; 5–9 for style and browse categories (apparel,
accessories, furniture), which wear in. Then rotate to the next adjacent
category. All of it is to be tuned with impression logs.

## How the industry names lifecycle stages

There is no single standard, but the common frameworks map onto ours:

| Framework | Stages | In our terms |
|---|---|---|
| E-commerce CRM lifecycle (typical in email/CRM tools) | Subscriber/prospect → first-time buyer → repeat → loyal/VIP → at risk → lapsed → reactivated | Relationship stage + rhythm state |
| RFM segments (Recency, Frequency, Monetary) | e.g. champions, loyal, potential loyalists, new, promising, need attention, about to sleep, at risk, can't lose, hibernating, lost | Stage × rhythm × spend, in fixed bins; our version uses each customer's *own* rhythm instead of store-wide recency bins |
| AARRR ("pirate metrics") | Acquisition → activation → retention → referral → revenue | Funnel metrics: visitor → first order → second order → loyal |
| Grocery retail | Loyalty by share of wallet (primary / secondary / occasional shopper), frequency tiers (weekly, biweekly, monthly), **trip missions** (stock-up, top-up, for tonight, special occasion), household life stage | Cadence + regularity (rhythm), basket size and mission per order; life stage is sensitive (see SHOPPER-PROFILE ethics) |
| Buy-till-you-die models (Pareto/NBD, BG/NBD) | A continuous P(alive) and expected purchases instead of stages | A model-based version of the rhythm state, for when we need a probability rather than bins |

Amazon doesn't publish its internal lifecycle model. Its public touchpoints
show the same jobs: Prime trial → paid → renewal (relationship), Subscribe
& Save and "Buy it again" (replenishment), and cross-category
recommendations (expansion).

## Inside each stage: momentum, style and the current visit

A stage alone is too wide (`docs/experiments/2026-10-24-lifecycle-stages-deep-dive.md`).
In both RetailRocket and REES46, near-term ordering differs 1.5–3× *within* a
stage, and what shoppers cart next changes with what they are doing. And
what they're shopping for changes how they decide
(`docs/experiments/2026-10-24-product-type-and-price.md`). Six layers
describe a shopper at a moment:

| Layer | Changes | Examples | What it changes |
|---|---|---|---|
| **Relationship** (stage) | slowly | new, browser, **abandoner**, 1 order, 2+, bulk | the job: first order, second order, habit |
| **Momentum** | by the hour | open cart <1 day old, ordered in the last day, back within the hour | how close they are to buying: timing and urgency of messages |
| **Style** (traits) | slowly | decisive vs researcher, brand loyalty, mainstream vs niche, variety seeking | help deciding vs more options; same brand or not; popularity blend; exploration dial; whom not to message |
| **Current visit** | by the click | depth (one product page vs several), focus (one department vs exploring), open cart | which source fills the slots right now |
| **Product context** | per item | product type's deliberation time and comparison breadth; price relative to the item being viewed | how long to wait before a nudge; how many and which alternatives (price-banded) |
| **Price profile** | slowly | band per product type, general price level (pooled across their history) | which tier to lead with, especially in a category new to them |

Findings that change behaviour:
- **Abandoners** order almost as soon as first-time buyers, and half their
  carts are *alternatives* to what they carted: show substitutes, not just a
  reminder.
- **Bulk and repeat buyers** mostly re-cart known items: reorder lists over
  discovery.
- **One product page in:** the shopper carts that page 82% of the time, so
  recommendation slots should hold alternatives and complements, not the item.

- **High-involvement products** (smartphones, premium tiers) are decided over
  days and visits; **low-involvement** ones (irons, small appliances) in
  minutes. Message timing follows the product, not just the shopper.
- **Price works at three levels** (`docs/experiments/2026-10-24-shopper-price-level.md`):
  - in the moment, shoppers compare within about ±30% of the item in front
    of them;
  - each shopper has a band per product type (most of their price position);
  - a general price level emerges from their collective history and predicts
    the tier they'll buy in a new category (premium 29% vs 16%).

  Discount chasing, by contrast, is not a lasting trait.
- **Over years** (`docs/experiments/2026-10-24-amazon-traits-over-years.md`):
  - price level, mainstream taste and per-type price bands persist but fade
    slowly (0.15–0.20 a year or more later);
  - brand loyalty lasts for years, strongest for consumables (7× chance after
    3+ years in grocery);
  - shoppers add about 0.4 new product types a year, with mild adjacency
    (TV → home audio 2.4×).

- **Traits are weaker than states** (`docs/experiments/2026-10-24-shopper-traits-stability.md`):
  the most stable traits correlate 0.2–0.3 from one month to the next, while
  momentum moves ordering 1.5–3×. Brand loyalty, mainstream-vs-niche, variety
  and comparison breadth persist. Deal-seeking and weekday/weekend habits
  don't: they belong to the moment. Profiles should decay old evidence and
  let the current visit dominate.

Humans get a handful of segments (stage + momentum + open cart). Models get
the underlying features directly and personalize continuously.

## Two different jobs for a message

1. **Next transaction (exploit):** predict what this customer will buy next in
   categories they already use: the carted item, a replenishment, a close
   alternative or complement. Measured by conversion.
2. **Category expansion (explore):** introduce an *adjacent* category that
   people like them tend to move into next (learned from category transitions
   in purchase histories), to build breadth. Measured by new-category purchases
   and, ultimately, whether the customer keeps coming back.

**Caution from five datasets:** in RetailRocket, customers whose second order
was in a new category placed a third order about twice as often, but Olist,
REES46 and H&M show no such effect. So category expansion is a hypothesis to
test per store with a randomized holdout, not a default.

A recommender trained only on "what will they click next" drifts toward job 1
and keeps customers in their current category. That maximizes the next click,
not long-term value. Job 2 is an explicit choice to trade a little short-term
relevance for breadth: the **explore/exploit** tradeoff.

## Choosing per customer: next-best-action

For each customer, at each moment, the decision is one of:
*no message* · *next-transaction message* · *category-expansion message* ·
*win-back*. Picking well is the "next-best-action" problem. We'll build it in
increasing sophistication, measuring each step:

1. **Rules by stage** (the table above), with frequency caps. A baseline, like MostPopular.
2. **Propensity models:** probability of returning and buying, per stage.
   In a randomized promotional SMS campaign (X5), skipping the likeliest buyers first
   captured 65% of the incremental purchases with half the sends: the top
   propensity decile gained only +1.0 point vs +3.3 on average.
3. **Uplift per action:** the *incremental* effect of each message type vs no
   message, learned from randomized data (Hillstrom, Criteo, X5). Send only where the
   lift is positive; that's how you stop paying for "sure things" and stop
   bothering "sleeping dogs".
4. **Contextual bandit across actions:** keeps exploring which message works
   for whom, logging propensities so offline evaluation stays honest.

## Metrics

| Level | Metric | Why |
|---|---|---|
| North star (loyalty) | **Second-order rate** of first-time buyers within N days; **90-day retained customers** | The bottleneck, and the outcome loyalty is made of |
| Per message | **Incremental** conversions (or new-category purchases) per message vs a randomized holdout | Organic return rates are high (28% of abandoned carts return anyway) |
| Leading indicator | **Incremental return visits within 7 days** per message vs holdout | Visits come long before orders, so experiments can read out faster; never the decision metric on its own |
| Long term | Incremental orders (or revenue) per customer over 90 days | Catches messages that win now but hurt later (e.g. discounts training discount-seeking) |
| Guardrails | Unsubscribes, complaints, messages per customer per week, relevance of expansion picks | Fatigue destroys the channel you're optimizing |

## Paths from first order to loyal

From `docs/experiments/2026-10-27-paths-to-loyalty.md` (H&M, 146,884 new
customers, each followed for a year):
- **Nearly every possible path occurs:** 3,993 of 4,095 monthly patterns.
  So we design for **families** of paths, not individual ones.
- **Seven families; three lead to loyalty:**
  - steady from the start (48% of loyal customers);
  - early repeat that keeps going (41%);
  - slow start that builds (11%).
- **The levers are timing and fading.**
  - A second order within 60 days → 25–28% loyal; after 120 days → 5.6%.
  - 36% of new customers return and then fade (quick burst, early repeat
    that fades, one late return). They are the group to work on.
- **Breadth of the first basket barely matters;** store and mixed-channel
  starts do better.
- **Replicated** in CDNOW (true first-ever buyers) and a B2B wholesaler: the
  same families, the same 60-day cliff
  (`docs/experiments/2026-10-27-paths-across-datasets.md`).
- **The one-order share depends mostly on the business model:** 36% in
  fashion, 54% in music, 97% in a marketplace, where nothing in the first
  order changes it.
- **One-order buyers aren't all lost at day 90.** A quarter to a third
  return within the year, but rarely become loyal. So: win the second
  order in days 1–60, then switch to light reactivation.
- **In grocery, categories enter in a typical order:**
  - staples first: produce, dairy, beverages, snacks;
  - then frozen, pantry, bakery and deli by the second order;
  - then household, personal care and breakfast by the third.

  That order is a candidate next-category sequence. Loyal customers are
  more *frequent*, not broader.

## What else predicts the second order

From `docs/experiments/2026-10-27-loyalty-drivers.md`:
- **Intent visits after the first order predict the second.** At the same
  number of visits, 2+ intent visits (carts, focused views, comparing
  within a category) lift the second-order rate by about 8 points (OTTO).
  A visit in the last 2 days means about 1.5–1.6× the odds. Plain visits
  matter little, so "top of mind" needs intent to count.
- **Fulfillment and ratings:**
  - Lateness drives 1–2 star reviews (78% of orders 8+ days late).
  - In a marketplace with 2% repeat, the effect on reordering can't be
    told apart from zero. Measure it in a higher-repeat business, and
    treat problems as triggers for service recovery.
- **Small returns go with more future business** (B2B), not less.
- **Grocery first** is a causal claim that "loyal customers are grocery
  buyers" can't establish. Test design:
  `docs/experiments/2026-10-27-grocery-first-test-design.md`.

## Visits: the leading signal

Orders are rare and slow; visits arrive early. In REES46 the median number of
visits in a customer's first 30 days climbs with loyalty: **2** (browse only)
→ **5** (one order) → **7** (two) → **10** (three or more). Cart abandoners
visit as much as one-order buyers (they're interested; something blocks them).
A first order typically takes 2 visits over 2 days, and the second order comes
about **2 visits after the first**
(`docs/experiments/2026-10-24-visits-by-cohort.md`).

So:
- **Lifecycle features include visit frequency and recency**, RFM-style
  (recency, frequency, monetary value), not just orders.
- **Messages get a leading metric:** return visit within 7 days, fast to
  measure. Purchases (and the second-order rate) remain the lagging metric
  that decides. Never optimize visits alone; that rewards clickbait.

## Abandoned carts: high intent isn't the same as high value

From `docs/experiments/2026-10-27-cart-abandonment.md` (REES46, RetailRocket, OTTO):
- **Most of the organic outcome is a return, not a purchase.** 61% (REES46),
  24% (RR) and 66% (OTTO) of abandoners come back within 7 days, but only
  6.5%, 4.2% and 9.8% buy a carted item. In REES46, 5.2% buy a
  same-category substitute instead.
- **Which trigger has the bigger audience is store-specific.**
  - A browse message wins if its uplift is at least *(cart abandoners ÷
    browse abandoners)* × the cart message's.
  - That ratio is 0.42 in REES46, 0.85 in RR, and 1.31 in OTTO, where
    shoppers cart often.
  - Only a randomized holdout tells which message actually works better.
- **Timing:**
  - A field experiment (Li et al., *Journal of Marketing* 2021) found
    reminders at 30–60 minutes *reduced* purchases, while reminders at 1–3
    days increased them.
  - Default: no send within the first hour; a first send around the same
    time the next day (OTTO's recovery rate rises again at 21–24 h); one
    follow-up at 48–72 h if the shopper hasn't returned.
- **Suppress** the message once the shopper has returned to the site,
  bought the item or a substitute, or when the item is out of stock (send
  alternatives instead).
- **Content:** the carted item plus 2–3 close alternatives (cart
  substitutes).
- **The experiment:**
  - Arms: a holdout, plus send delays of 1 / 4 / 24 / 72 h.
  - Outcome: any purchase within 7 and 30 days.
  - Size: about 20,000 abandoners per arm to detect a 1-point lift.

## What varies by store (so it must be learned, per tenant)

Across RetailRocket, Olist, REES46, H&M and a B2B wholesaler
(`docs/experiments/2026-10-24-lifecycle-across-datasets.md`):
- the median gap from first to second order ranges from **4 to 74 days**,
- the 30-day second-order rate from **0.7% to 35%**,
- exact re-buys of the same item from **9% to 76%**.

A fixed "send at 1 h and 24 h" rule, or a fixed definition of "lapsing", would
be wrong by an order of magnitude somewhere. Timing, lapse thresholds and
trigger priorities are learned per store from its own data. And before any
loyalty analysis, **screen out resellers/B2B buyers**: in RetailRocket, 1.2% of
buyers bought 29% of the items.

## What this means for the engine

- **Lifecycle stage is a first-class user feature,** derived from events (orders,
  recency, categories bought). It's computed, never stored as truth.
- **Evaluation is split by stage.** A model that's great for loyal customers
  and useless for first-time buyers looks fine on average and fails where it
  matters.
- **A category-novelty metric** (share of recommendations in categories new to
  the user, and hits among them) sits next to recall, so the explore/exploit
  balance is visible.
- **Candidate generators per job:** repeat/replenish, same-category similar,
  and adjacent-category (from category transitions), mixed by a re-ranker
  whose exploration share is a dial.
- **Messages are logged like slates:** `message_sent` carries the action, the
  items, the model and the propensity, so uplift and bandit evaluation work later.

The full shopper model (every dimension, its evidence, stability and the
decision it drives) is in `docs/SHOPPER-PROFILE.md`.

## On-site recommendations: one recommender, many journeys

The same stage logic applies on the site, not just in messages. No single
model serves every journey, so recommendations come from several **candidate
sources** that each do one job (recently viewed, viewed-together, trending;
next: re-buy/replenish, complements, adjacent category), and a **blender**
weights them per shopper from their state: visits, orders, recency, category
breadth, what they're doing in this visit. The blender starts as rules by
stage and becomes learned, so the mix is personal rather than one rule per stage.

First measurements on RetailRocket
(`docs/experiments/2026-10-24-session-sources-retailrocket.md`):
- 63% of add-to-carts are an item already viewed in that visit (usually the
  page the shopper is on), so accuracy must be reported separately for
  *items seen before* and *new items*, or the trivial part dominates.
- Viewed-together finds 23% of new-item carts in browsed categories; nothing
  finds new-category carts yet, which is the gap an adjacent-category source fills.
- Repeat and bulk buyers mostly re-cart what they know; first-time buyers and
  returning browsers respond most to discovery.

See `docs/experiments/2026-10-24-retailrocket-lifecycle.md` for the numbers behind this.
