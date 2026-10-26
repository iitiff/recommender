# 2026-10-24: RetailRocket re-engagement profile

**Status:** done (descriptive analysis, no model)
**Data:** RetailRocket (CC BY-NC-SA 4.0; derived statistics shared under the same
license), prepared with `recsys dataset prepare retailrocket`, sessions = 30 min
inactivity gap.

## Question

When a visit ends without a purchase, who comes back, how fast, and do they buy?
That decides which triggers deserve a push/email, what to put in it, and when to send.

## Method

```bash
uv run recsys dataset profile data/retailrocket
```

Verified two ways: a standalone pandas script and the engine's `profile` command
agree exactly.

## Results

2,756,101 events (2,664,312 views, 69,332 add-to-carts, 22,457 purchases),
1,407,580 users, 1,761,675 sessions, 235,061 items. **12.9%** of users ever return.

| After a session that ended like this… | Sessions | Return ≤7d | Return ≤30d | Buy on next visit ≤7d |
|---|---|---|---|---|
| Viewed 1 product, left | 1,374,581 | 12.2% | 15.4% | 0.13% |
| Viewed 3+ products, no cart (abandoned browse) | 140,758 | 28.1% | 32.5% | 0.80% |
| Added to cart, didn't buy (abandoned cart) | 31,992 | 28.1% | 32.1% | **3.99%** |
| Bought something (post-purchase) | 14,297 | 38.4% | 42.9% | **9.09%** |

Time to return: 25% within 0.18 days (~4 h), median 1.15 days, 75% within 8.1 days.

Abandoned carts that convert on the next visit (≤7 days, n=1,276): **82%** bought
an item that was in the abandoned cart.

## Conclusion

- **Trigger priority:** abandoned cart (31× the 1-view purchase rate) >
  post-purchase cross-sell (highest return and repurchase) > abandoned browse
  (6×, but 4× the volume of carts). One-view visitors aren't worth messaging.
- **Content:** for abandoned carts, the carted item first (82%), then 2–3
  recommendations; if the carted item went out of stock, a "similar items" or
  "back in stock" message instead (availability history is in the data).
- **Timing:** returns are front-loaded (a quarter within ~4 h), so the classic
  "1 hour + 24 hours" cadence fits.
- **Caveat that shapes everything next:** these are **organic** return rates.
  28% of abandoned-cart visitors come back with no message at all, so
  "purchases after an email" would credit the email for many of them. Measuring
  a message's real effect needs a randomized holdout (uplift). RetailRocket has
  no messages; Hillstrom and Criteo Uplift do (see docs/DATASETS.md).

**Next:** session-based recommender (co-visitation), evaluated by Recall@20 and
MRR@20 on the next add-to-cart, with an in-stock guardrail.
