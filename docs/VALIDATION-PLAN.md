# Validation and experiment plan for a retailer's own data

Everything we learned on public datasets is a **hypothesis** for a real
retailer until it is (1) replicated on its data and (2), where it implies an
action, tested with a randomized holdout. This page consolidates both lists.

**The setting:** a large retailer with grocery and general merchandise (GM),
stores and online, and many one-order customers.

**Every item links to:**
- the public write-up it comes from (`docs/experiments/`);
- the script that reruns it, which needs a small loader for the retailer's
  data.

**Rules carried over from the public work:**
- **Aggregates only.** Suppress cells under 50 customers.
- **Behavior only.** Never infer sensitive attributes (health, pregnancy,
  finances/income, ethnicity, religion), and never label shoppers by price
  or income.
- **Price is context, not taste.** Recommendations never change the price
  a shopper is charged. Offers are uniform within a randomized group,
  never targeted by price sensitivity.
- **Results on real customer data stay in private storage,** never in
  this public repository.

## Phase 0: prerequisites (before any experiment)

| # | What | Why |
|---|---|---|
| 0.1 | **Standing holdouts per message type** (5–10% of eligible customers get nothing), plus message logs with send time, channel, content and the holdout flag | Without a holdout no message's effect can be measured. A promotional SMS (X5) was credited with about 19× its real lift |
| 0.2 | **A data extract:** order lines from each account's first purchase (grocery/GM flag, category, price paid, promotion), channel and fulfillment method per order, visits and searches tied to the customer, cart events (adds, removals), fulfillment events (promised vs actual time, substitutions, damage), returns with reasons, ratings, service contacts, contactability (opt-ins), acquisition channel | Every analysis below. The public datasets lacked searches, removals, messages, fulfillment details and the grocery/GM mix |
| 0.3 | **Local timestamps** (or the store's or customer's time zone) and a documented channel code | Timing results depend on them (REES46 timestamps were UTC; H&M's channel code is undocumented) |

## Phase 1: validations (descriptive, weeks; no customer is contacted)

Each one checks a public finding on the retailer's data. The "public
finding" column is what we expect to see; the "decision" column is what the
answer changes.

| # | Validation | Public finding to check | Script / write-up | Decision it informs |
|---|---|---|---|---|
| 1.1 | **Paths from first order to loyal:** funnel, path families, the second-order cliff | The 2nd order is the biggest drop; 7 path families, 3 lead to loyalty; second order within 60 days → 19–32% loyal, after 120 days → 4–6% (H&M, CDNOW, B2B) | `paths_to_loyalty.py`; paths-to-loyalty, paths-across-datasets | Window and effort for the second-order program |
| 1.2 | **One-order buyers by business mode:** marketplace/one-off vs grocery vs GM | One-order share is mostly the business model (36% fashion → 97% marketplace); 24–37% of 90-day one-order buyers return within a year (H&M, CDNOW, B2B) | `paths_to_loyalty.py` | Separate programs for marketplace-like vs relationship customers |
| 1.3 | **Category paths:** entry category, the 2nd order's category, the order in which grocery departments are adopted | Entry category barely matters; grocery adds staples → frozen, pantry, bakery, deli → household, personal care (Instacart) | `paths_to_loyalty.py` (`category_paths`) | Next-category sequence for new customers |
| 1.4 | **Per-customer lapse timing** (MBG/NBD) vs a fixed 90-day rule | One-order customers look lapsed after 7–8 weeks; repeaters after 9–45 weeks depending on their gap; the model ≈ recency > 90-day rule for ranking (CDNOW, H&M) | `btyd_reactivation.py`; loyalty-models | Reactivation timing per customer |
| 1.5 | **Visits and intent after the first order** (landmark analysis and a Cox model), with **real search queries** | Intent visits (carts 1.9×, comparisons 1.4× per doubling) and a visit in the last 2 days (1.5–1.6×) predict the second order; plain visits ~1.07× (OTTO, REES46) | `visits_after_first_order.py`, `cox_second_order.py` | Intent signals in the shopper profile; who gets a nudge |
| 1.6 | **Experience factors:** fulfillment (promised vs actual time, substitutions), ratings, returns, service contacts → second order | Lateness drives 1–2 star reviews (78%); effect on repeat underpowered in a marketplace (Olist); small returns go with *more* future business (B2B) | `experience_factors.py`; loyalty-drivers | Service-recovery triggers; return policy |
| 1.7 | **First-order factors:** bought on sale, acquisition season, how repeatable the first product is (consumables), grocery vs GM entry | Mostly-on-sale first orders → more one-and-done (41% vs 36%); sale-heavy months have lower 2nd-order rates; repeatability matters little in fashion, a bit in a marketplace | `new_factors.py`; loyalty-models | Plan for sale-acquired cohorts; grocery-first prior |
| 1.8 | **Replenishment rhythm:** hazard by day since the last order, overdue point, product cycles | Weekly shoppers: 37% order on day 7; overdue at about 1.25–1.5× their own gap; product cycles 11–21 days (Instacart) | `replenish_explore.py`; replenish-or-explore | Reminder timing; basket vs item reminders |
| 1.9 | **Category and personal timing** (day, hour, channel), **lead time between online ordering and the fulfillment slot** | Few categories have their own rhythm (alcohol late-week afternoons, stock-up days); personal routines beat the store peak 1.3× in grocery; online ordering isn't simply "store peak minus 2 hours" | `category_timing.py`; category-timing | Send-time rules per category and channel |
| 1.10 | **Cart abandonment baselines:** return, recovery, substitutes, timing; cart vs browse audiences | Most returners don't buy; recovery peaks at the same time next day; break-even uplift ratio 0.42–1.31× by store | `cart_abandonment.py`; cart-abandonment | Cart vs browse message priority; send time |
| 1.11 | **Exposure curves per category** (needs impression logs) | Need-driven categories front-load (cap 2–3), style categories wear in (5–9) (REES46 self-chosen views) | `replenish_explore.py`; replenish-or-explore | Frequency caps per category |
| 1.12 | **Shopper traits stability and price level** | Price band per type is stable; general price level pools across history (REES46, Amazon) | `shopper_traits.py`, `price_level.py` | Which traits enter the shopper profile |
| 1.13 | **Recommender replay:** journey blend vs best single source, tuned on a validation window, confirmed once on test | Structure transfers, weights don't (RetailRocket +4.7 points of hit@20; OTTO needs re-tuning) | `recsys evaluate --split next`; journey-blend write-ups | Ship the blend with store-tuned weights |
| 1.14 | **Engaged core vs light repeaters** (sequence clustering) | Only one natural split: an engaged core of 12–16% vs light repeaters | `path_clusters.py`; loyalty-models | Segment definitions for reporting |
| 1.15 | **Attribute value study:** which customer attributes improve predictions (drop-one-group ablation, freshness, history depth) | Not yet run (proposed) | to build | Which attributes to request from data owners |
| 1.16 | **Loyalty profile:** value grid on *margin* (true friends, barnacles, butterflies, strangers), promotion and return habits | In fashion, barnacles are 2–3% of repeaters and buy on sale; butterflies are a third (H&M) | `loyalty_dimensions.py`, `grocery_paths.py`; loyalty-dimensions, grocery-paths | Who gets rewards vs full-price service; split barnacles into deal-driven vs top-up (share of wallet; about $17 a trip in grocery) |
| 1.17 | **Habit strength** (day, time, store, rhythm, separately) and attitude signals (ratings, reviews, searches) | Habit measures barely correlate; habit didn't weaken promotion response (X5) | `loyalty_dimensions.py` | Inputs for the cross-sell test (2.13) |
| 1.18 | **Share of wallet** by category, from a receipt panel or survey linked to customers | Spend with us says little about spend elsewhere (Du, Kamakura & Mela 2007) | to build | Grow share vs frequency |
| 1.19 | **Resilience after shocks:** behavior after a stock-out, late or failed delivery, substitution, price change, or a competitor opening nearby | Spurious loyalty breaks under shocks (Dick & Basu); not measurable on public data | to build | Service-recovery priorities |
| 1.20 | **Loyalty-program selection:** member trajectories before and after joining vs matched non-members | Heavy buyers join first; the effect is small and fades within 6–9 months (Meyer-Waarden & Benavent 2009) | to build | Whether to credit the program |

## Phase 2: randomized experiments (causal; each has a holdout)

**Sample sizes** are for two-sided 5% significance and 80% power on the
primary outcome. Rule of thumb: per arm ≈ 15.7 × p(1−p) / d², where p is
the base rate and d the lift to detect.

| # | Experiment | Arms | Primary outcome (guardrails) | Size per arm | Hypothesis from |
|---|---|---|---|---|---|
| 2.1 | **Second-order nudge for new customers** in the first 60 days | Holdout; nudge at day 7–14; nudge timed from the customer's own rhythm; next-category content vs "buy again" content | 2nd order within 60/90 days; 12-month loyal (unsubscribes) | ~20k for a 1-point lift on a 15% base | 1.1, 1.3, 1.5 |
| 2.2 | **Abandoned-cart timing** | Holdout; 1 h; 4 h; next day, same hour; 72 h | Any purchase within 7 and 30 days (unsubscribes) | ~20k | 1.10; Li et al. 2021 (early sends hurt) |
| 2.3 | **Cart vs browse abandonment message** | Holdouts for each | Incremental orders per 1,000 eligible customers | Per break-even ratio | 1.10 |
| 2.4 | **Reactivation timing** | Holdout; fixed 90 days; per-customer lapse moment (MBG/NBD) | Reactivated within 30 days; 90-day orders | ~20k | 1.4 |
| 2.5 | **Replenishment reminder timing** for regular grocery shoppers | Holdout; at the due day; at 1.25× their gap; at 1.5× | Orders within 14 days; basket size (complaints, unsubscribes) | ~20k | 1.8; the X5 promotional SMS showed no rhythm effect, so this tests reminders specifically |
| 2.6 | **Grocery first** (randomized encouragement) | Holdout; invitation to try grocery for GM-only customers, preferably non-price | 12-month margin and GM spend (ITT); effect of adopting grocery (CACE) | ~6.3k (30-point take-up, 5-point effect) | `2026-10-27-grocery-first-test-design.md` |
| 2.7 | **New-category frequency caps** | Cap 2–3 vs 5–9 exposures, by category type | Adoption (bought again within 3 orders) | Per category group | 1.11 |
| 2.8 | **Send time** | Store peak; category rhythm; the customer's own usual hour | Opens, visits within 24 h, orders within 7 days | ~20k per arm for opens | 1.9 |
| 2.9 | **Service recovery** after a fulfillment failure or a 1–2 star rating | Holdout; apology or help message with no offer | Second order within 90 days; next rating | Depends on failure volume | 1.6 |
| 2.10 | **"Top of mind" content with no offer** (inspiration, lists) | Holdout; weekly content | Intent visits, then orders within 30 days | ~20k | 1.5: does raising intent visits *cause* orders? |
| 2.11 | **Targeting rule for non-price messages** | Everyone; skip the likeliest buyers (propensity); an uplift model once holdout data exists | Incremental orders per message | Uses 2.1–2.5 data | X5: the propensity rule captured 65% of the lift with half the sends |
| 2.12 | **Recommender online** | Current recs; journey blend; blend + exploration slots | Add-to-cart rate, discovery carts, revenue per visit | Interleaving needs far fewer users | 1.13 |
| 2.13 | **Cross-sell by habit vs attitude** | Holdout; new-category message; routine-reinforcing message; split by habit strength | New-category adoption and the *original* category's purchases (does the message break the routine?) | ~20k per habit group | Liu-Thompkins & Tam 2013; 1.17 |
| 2.14 | **Share-of-wallet growth** for loyal customers with a low estimated share | Holdout; category reminders for categories they buy elsewhere | Category spend share; total margin | Depends on 1.18 | 1.18 |
| 2.15 | **Second-channel invitation** for early single-channel repeat customers | Holdout; invitation to try the other channel (store pickup or app), no price offer | 12-month true-friend rate (loyal and valuable); second-channel use (unsubscribes) | ~20k | `2026-10-27-paths-to-value.md`: both channels by day 90 → 33% true friends vs 7–9%, possibly selection |

**For offers specifically:** measure margin, not just orders: offer cost on
*every* redemption, including customers who would have bought anyway.
Promotions don't target by price sensitivity.

## Suggested order

1. **Phase 0.**
2. **Validations 1.1–1.5 and 1.10** (cheap, and they size Phase 2).
3. **First experiments:** 2.1 (second order), 2.2 (cart timing) and 2.4
   (reactivation). They are high-volume and read out in 1–3 months.
4. **Then** 2.5, 2.8 and 2.9.
5. **Last:** 2.6 (12-month readout) and 2.12 (needs engineering).
6. **Keep** 2.11 running underneath, as holdout data accumulates.

Each readout gets a write-up in private storage, using the public
`docs/experiments/TEMPLATE.md` structure.
