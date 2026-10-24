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

## Inside each stage: momentum, style and the current visit

A stage alone is too wide (`docs/experiments/2026-10-24-lifecycle-stages-deep-dive.md`).
In both RetailRocket and REES46, near-term ordering differs 1.5–3× *within* a
stage, and what shoppers cart next changes with what they are doing. Four
layers describe a shopper at a moment:

| Layer | Changes | Examples | What it changes |
|---|---|---|---|
| **Relationship** (stage) | slowly | new, browser, **abandoner**, 1 order, 2+, bulk | the job: first order, second order, habit |
| **Momentum** | by the hour | open cart <1 day old, ordered in the last day, back within the hour | how close they are to buying: timing and urgency of messages |
| **Style** | rarely | decisive (few visits per order), researcher (many), window shopper (never carts) | help deciding vs more options; whom not to message |
| **Current visit** | by the click | depth (one product page vs several), focus (one department vs exploring), open cart | which source fills the slots right now |

Findings that change behaviour:
- **Abandoners** order almost as soon as first-time buyers, and half their
  carts are *alternatives* to what they carted: show substitutes, not just a
  reminder.
- **Bulk and repeat buyers** mostly re-cart known items: reorder lists over
  discovery.
- **One product page in:** the shopper carts that page 82% of the time, so
  recommendation slots should hold alternatives and complements, not the item.

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
3. **Uplift per action:** the *incremental* effect of each message type vs no
   message, learned from randomized data (Hillstrom, Criteo). Send only where the
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
