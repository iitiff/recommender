# 2026-10-27: Cart abandonment: who comes back on their own, when, and should we send?

**Status:** done (descriptive, observational; no messages in any of the datasets)
**Data:** aggregates only.
- **REES46** multi-category store (electronics-heavy), carts abandoned Nov 18–23, 2019. "© Original Authors".
- **RetailRocket** (RR) e-commerce, carts abandoned over about 3.3 months of 2015. CC BY-NC-SA 4.0.
- **OTTO** (large German online retailer), carts abandoned Aug 8–21, 2022, after a week of history; a random 25% of shoppers (3.2M, 54M events). **CC BY 4.0** (commercial use allowed). Added as a replication; no categories or prices, so substitutes and cart value aren't measured.

## Question

Is the abandoned-cart message always the most valuable one to send? To answer
that, we need four things:
1. How many abandoners come back and buy **on their own** (no message in any dataset)?
2. **When** do they come back, which sets when a reminder could still add something?
3. **Who** comes back (lifecycle, cart value, category, deliberation)?
4. How the cart trigger compares with other triggers (browse abandonment)
   on audience size and organic buying.

## Method

```bash
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/cart_abandonment.py rees46 events.parquet
python -I docs/experiments/scripts/cart_abandonment.py retailrocket data/retailrocket
python -I docs/experiments/scripts/stream_otto_events.py train.zip otto_events.parquet 4
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/cart_abandonment.py otto otto_events.parquet
```

- **Abandoned cart:**
  - A visit with an add-to-cart and no purchase. A visit ends after 30
    minutes without activity, so the earliest possible "return" is 30
    minutes after the last event.
  - The unit is each shopper's **first** abandoned visit in the window.
- **Follow-up:** the 7 days after the visit's last event.
  - **Return:** the next visit.
  - **Recovery:** buying a carted product.
  - **Substitute:** buying another product in the same leaf category, but
    not the carted one.
- **Timing:** when recoveries happen (share of 7-day recoveries per band of
  hours), and the hourly hazard (chance of recovering in hour *h*, given not
  yet).
- **Segments:**
  - Lifecycle before the visit (new visitor / browser / abandoned before /
    1 order / 2+ orders).
  - Cart value, top-level category, items in the cart.
  - Alternatives compared: products viewed in the visit in the carted leaf
    category.
  - Carted the same product in an earlier visit.
  - Hour of day.
- **Trigger comparison:**
  - Covers every shopper active in the window, and takes each shopper's
    first visit of each kind:
    - cart abandoned;
    - browse abandoned (a product viewed 3+ times in the visit, no cart);
    - plain browse.
  - Measures how often each buys the focal product, or anything, within 7
    days.
- **Data quality (REES46):**
  - Add-to-cart is logged for 99% of purchases only from Nov 8. Before
    that, only about 35–55% of purchases had a same-day cart event.
  - The Nov 15–17 sale has a day with no purchases logged.
  - So abandonments come from Nov 18–23, and their 7 days of follow-up end
    by Nov 30.
- **Data quality (RR, OTTO):** 89% (RR) and 92% (OTTO) of purchases follow an
  add-to-cart.

## Results

**1. What abandoners do in the next 7 days, with no message**

| | REES46 (n=101,719) | RetailRocket (n=21,635) | OTTO (n=433,136) |
|---|---|---|---|
| Come back within 24 h | 34% | 17% | 39% |
| Come back within 7 days | **61%** | 24% | **66%** |
| Buy a carted item within 1 h / 24 h / 7 days | 0.6% / 3.8% / **6.5%** | 1.1% / 3.5% / **4.2%** | 1.1% / 6.6% / **9.8%** |
| Buy a substitute instead (same leaf category) | **5.2%** | 0.5% | not measurable |
| Buy anything within 7 days | 15.3% | 4.9% | 17.3% |
| Come back but buy nothing | **46%** | 19% | **48%** |
| Never come back within 7 days | 39% | **76%** | 34% |

**2. When organic recoveries happen (share of 7-day recoveries)**

| Hours after the visit ended | 0.5–1 | 1–2 | 2–4 | 4–8 | 8–24 | 24–48 | 48–72 | 72–168 | Median |
|---|---|---|---|---|---|---|---|---|---|
| REES46 | 9% | 10% | 10% | 7% | 23% | 15% | 9% | 18% | **18 h** |
| RetailRocket | **26%** | 16% | 14% | 9% | 19% | 8% | 4% | 4% | **2.8 h** |
| OTTO | 11% | 12% | 12% | 11% | 22% | 12% | 7% | 13% | **10.5 h** |

- **Hourly hazard of recovery, RetailRocket:**
  - The chance falls about tenfold within 5 hours.
  - It is near zero after the first day.
- **Hourly hazard of recovery, REES46:**
  - Highest in the first two hours.
  - Then flat at about 0.1% per hour until about 24 hours, with a small
    bump at the same time next day.
  - Then it halves on day 2 and halves again on day 3.
- **Hourly hazard of recovery, OTTO:**
  - Highest in the first two hours (about 1.1% per hour).
  - Falls to about 0.1% per hour by 16 hours.
  - **Rises again at 21–24 hours** (to 0.17%), and again at 45–48 hours:
    shoppers come back at the same time of day.

**3. Who recovers (REES46, bought a carted item within 7 days)**

| Segment | n | Come back 7d | Buy carted item 7d | Buy anything 7d | Median hours to recovery |
|---|---|---|---|---|---|
| New visitor | 21,585 | 39% | 3.3% | 8.0% | 12.6 |
| Browser (visited before, never carted) | 37,201 | 61% | 5.4% | 11.6% | 17.0 |
| Abandoned a cart before | 16,312 | 70% | 6.2% | 13.4% | 19.9 |
| 1 order before | 14,476 | 69% | 8.1% | 20.2% | 19.2 |
| **2+ orders before** | 12,145 | 78% | **13.6%** | **36.5%** | 21.0 |
| **Carted this product before** | 12,580 | 73% | **13.4%** | 23.9% | 23.1 |
| Not carted before | 89,139 | 59% | 5.5% | 14.1% | 16.7 |
| Cart under $25 | 10,063 | 60% | 4.5% | 12.1% | 19.0 |
| Cart $100–300 | 34,779 | 61% | 7.4% | 16.4% | 18.3 |
| Cart $300+ | 24,345 | 60% | 6.1% | 15.1% | **20.6** (45% after 24 h) |
| Compared 5+ alternatives | 20,818 | 69% | 6.0% (+7.4% substitute) | 17.2% | 16.7 |
| Abandoned 6pm–midnight | 25,016 | 60% | 5.9% | 14.3% | **21.0** (9% within 2 h) |
| Abandoned 7–11am | 30,908 | 62% | 6.8% | 16.0% | **8.9** |

- **OTTO replicates it:**
  - 2+ orders recover 20.7% (they buy anything 51% of the time), vs 7.7% for
    browsers and 8.9% for new visitors.
  - Carted this product before: 18.8% vs 9.5%.
  - 3+ items: 17.0% vs 7.2% for one item.
  - Abandoned in the evening (6pm–midnight): median 16.5 h to recovery, vs
    6.3 h for the afternoon.
  - New visitors recover fastest (median 5 h), but half never return.
- **RetailRocket shows the same direction:**
  - Returning browsers recover 7.3%, vs 3.2% for new visitors.
  - Carts with 3+ items recover 8.6%, vs 3.5% for one item.
  - Shoppers who compared 5+ alternatives recover 8.0%, vs 3.6% for 0–1.
- **Category:** differences are small in both stores (REES46: 4.4–7.6%).

**4. Trigger comparison (per 1,000 shoppers active in the window)**

| Trigger | REES46: shoppers | Buy focal 7d | Buy anything 7d | RR: shoppers | Buy focal 7d | Buy anything 7d | OTTO: shoppers | Buy focal 7d | Buy anything 7d |
|---|---|---|---|---|---|---|---|---|---|
| Cart abandoned | **83** | 6.5% | 15.3% | **21** | 4.2% | 4.9% | **195** | 9.8% | 17.3% |
| Browse abandoned (3+ views of a product) | **198** | 1.3% | 6.2% | **24** | 0.8% | 1.6% | **149** | 1.5% | 9.1% |
| Browsed (1–2 views per product) | 830 | 0.5% | 3.7% | 968 | 0.1% | 0.2% | 868 | 0.6% | 4.8% |

A shopper can appear in more than one row.

## Conclusion

1. **Cart abandonment has the highest intent, which isn't the same as the
   highest value.**
   - What matters is the *incremental* purchases a message causes:
     audience × uplift × margin, minus costs (fatigue, unsubscribes,
     discounts that teach shoppers to abandon on purpose).
   - Abandoners have the highest *organic* buying rate (6.5% vs 1.3% for
     browse abandoners, REES46). Part of what a cart message gets credited
     for would have happened anyway: the "sure things", in uplift terms.
   - The audience is small: 2–8% of active shoppers.
   - **Correction:** the earlier re-engagement profile ranked triggers by
     organic purchase rate. That measures intent, not value.
2. **Which trigger has the bigger audience depends on the store.**
   - **Break-even uplift ratio:** a browse message yields more incremental
     orders than a cart message if its uplift is at least *(cart
     abandoners ÷ browse abandoners)* × the cart message's uplift.
   - **REES46: 0.42×.** Browse abandoners are 2.4× as many, so a weaker
     browse message can win.
   - **RR: 0.85×.**
   - **OTTO: 1.31×.** Its shoppers cart often (cart abandoners are 19% of
     active shoppers), so the cart message wins unless the browse message
     works clearly better.
   - Neither uplift is known without a randomized holdout. So "always send
     the cart message first" is an assumption, not a finding. The ratio is
     cheap to compute per store and tells you how much better one message
     has to be.
3. **Most abandoners who come back on their own don't buy.**
   - In REES46, 61% come back within a week but only 6.5% buy the item,
     and 46% return and buy nothing. OTTO is almost the same: 66%, 9.8% and
     48%.
   - Returning shoppers already see their cart on the site. So a message
     adds the most for those who **don't come back**: 39% in REES46, 76%
     in RR, 34% in OTTO.
   - Suppress the message once the shopper has returned.
4. **Substitutes are nearly as common as recoveries (REES46: 5.2% vs 6.5%).**
   - Deliberate shoppers compare: those who compared 5+ alternatives buy a
     substitute more often (7.4%) than the carted item (6.0%).
   - So the message should show **the carted item plus 2–3 close
     alternatives** (cart substitutes: same leaf category, similar price).
   - Suppress once they've bought the item **or a substitute**.
5. **Timing is store-specific and shouldn't be the industry default of
   "1 hour".**
   - **Field-experiment evidence** (Li, Luo, Lu & Moriguchi, *Journal of
     Marketing* 2021, "The Double-Edged Effects of E-Commerce Cart
     Retargeting"):
     - About 40,500 abandoners were randomized to get a cart reminder or
       nothing, at different delays.
     - Reminders sent **30–60 minutes** after abandonment **reduced**
       purchases compared with no reminder.
     - Reminders sent **1–3 days** later **increased** them.
     - Both effects are stronger for carts with more items and higher
       prices.
     - Repeated discounts in reminders can train shoppers to cart and wait
       for a discount.
     - A likely mechanism is reactance: an early nudge feels intrusive and
       interrupts the shopper's deliberation.
     - DOI: 10.1177/0022242920959043. Findings are summarized from the
       abstract and the publisher's press release.
   - **The organic curves say the same about where the room is:**
     - REES46 recoveries are spread over days (median 18 h, 41% after
       24 h).
     - RR recoveries are fast (median 2.8 h, 42% within 2 h), but they also
       stop fast. A 1-hour send there would land on the shoppers about to
       buy anyway.
   - **OTTO adds direct support for "the same time the next day":** its
     recovery hazard rises again at 21–24 hours and 45–48 hours.
   - **Defaults until a test settles it:**
     - No send within the first hour.
     - A first send around **the same time the next day**: the REES46
       hazard bumps then, and evening abandoners recover a median 21 h
       later.
     - A second send at 48–72 h, only if they haven't returned.
     - Deliberating segments (carts over $300, carted before, 2+ orders)
       recover even later (median 20–23 h).
   - The earlier advice ("abandoned-cart nudge within hours", "1 hour +
     24 hours cadence fits") is corrected accordingly.
6. **Who to send to is an uplift question; these segments are hypotheses.**
   - **High propensity, possibly sure things:** 2+ orders (36.5% buy
     something within 7 days anyway) and shoppers re-carting the same
     product.
   - **Low propensity, possibly lost causes:** new visitors (61% never
     return). In practice, new visitors are usually unreachable by email
     anyway.
   - The persuadable middle can't be found from organic data. It needs a
     randomized test analyzed by segment.

**Decision rule for an abandoned-cart message (defaults to be tested):**

| Check | Rule |
|---|---|
| Bought the item or a same-category substitute | **Don't send** (or switch to a post-purchase message) |
| Came back to the site since abandoning | **Don't send** the cart reminder; on-site recs (recently viewed + substitutes) do the job |
| Item out of stock or price changed | Send "similar items" (or the price change, if it dropped) instead |
| Less than 1 hour since abandoning | **Wait** (early sends backfired in a field experiment) |
| Not contactable, or messaged in the last few days | Don't send (frequency cap) |
| Otherwise | Send at about the same time next day: carted item + 2–3 alternatives. One follow-up at 48–72 h if still no return. Stop after that |

**How to find out (the experiment):**
- Randomize at abandonment into: a **holdout** (no message) and send delays
  of **1 h / 4 h / 24 h / 72 h**.
- **Outcomes:**
  - any purchase within 7 days, not only the carted item, since
    substitutes count;
  - purchases within 30 days, to catch messages that only pull sales
    forward;
  - unsubscribes.
- **Sample size:** with a 15% baseline 7-day purchase rate, detecting a
  1-point absolute lift (two-sided 5%, 80% power) takes about **20,000
  abandoners per arm**.
- **Analysis:** uplift by segment (lifecycle, cart value, deliberation),
  then a model of uplift per shopper. Concepts: **conditional average
  treatment effect (CATE)**, T-learner / causal forest.
- **Same holdout for browse abandonment:** settles point 2 directly.

**What this means for data requests:**
- **Message logs with a randomized holdout flag:** send time, channel,
  content, opens and clicks. Without a holdout, no amount of history shows
  uplift.
- **Cart state over time:** adds, removals, quantity changes, price at add.
  REES46 has no removals, so "abandoned" can't tell a parked cart from an
  emptied one.
- **Contactability and identity:** email/push opt-in, and the link between
  anonymous visits and known customers. In RR, 79% of abandoners are new
  visitors.
- **Delivery and stock events,** to suppress messages about unavailable
  items.

**Caveats:**
- **No messages in either dataset:** these are organic baselines. They show
  how much room a message has, not how much it fills.
- **REES46:** one store, six days of abandonments right after a sale.
- **RetailRocket:** cookie IDs undercount returns across devices.
- **OTTO:**
  - Only a week of history before each abandonment, so "new visitor"
    includes returning shoppers first seen that week.
  - A shopper's ID covers their whole period, possibly across devices
    (OTTO doesn't say).
  - No categories or prices.
- **Recovery can include slow checkouts:** a purchase more than 30 minutes
  after the last event (for example, payment off-site) counts as a recovery.
- **The field experiment** is one retailer; its timing results should be
  re-tested per store.
