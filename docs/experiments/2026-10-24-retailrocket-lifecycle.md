# 2026-10-24: RetailRocket lifecycle: from first order to loyal customer

**Status:** done (descriptive, observational)
**Data:** RetailRocket prepared (CC BY-NC-SA 4.0; derived statistics shared under the
same license). An *order* = a session containing at least one purchase. The
window is 137 days, so late first orders had less time to repeat (right-censoring).

## Questions

1. How concentrated is buying in repeat customers?
2. How often does a first order turn into a second, and how fast?
3. Is the second order "more of the same" or a new category?
4. Do customers who expand into a new category come back more?

## Method

```bash
uv run --no-project --with pandas python -I docs/experiments/scripts/lifecycle.py data/retailrocket
```

## Results

**Lifecycle stages**

| Stage | Users | Share of users | Share of purchased items |
|---|---|---|---|
| Browse only | 1,368,715 | 97.24% | 0% |
| Carted, never bought | 27,146 | 1.93% | 0% |
| 1 order | 11,033 | 0.78% | 63.8% |
| 2 orders | 476 | 0.03% | 7.4% |
| 3+ orders | 210 | 0.01% | **28.8%** |

**First → second order:** 11,719 buyers; **5.9%** placed a second order. Days from
first to second order: 25% 0.2 · median 1.9 · 75% 13.0 · 90% 33.6.

**What the second order contains** (n=686; categories known for 659):
re-bought the same item **32.8%**; same leaf category 49.5%; same top-level
category 76.5%; **new top-level category 23.5%**.

**After two orders, P(third order):**

| Second order was… | n | P(3rd order) | ≈95% CI |
|---|---|---|---|
| In the same top-level category | 504 | 24.8% | 21–29% |
| In a **new** top-level category | 155 | **47.7%** | 40–56% |

## Conclusion

- **Loyal customers are tiny and matter enormously:** 0.04% of users (2+ orders)
  account for ~36% of purchased items. (Some 3+ buyers may be resellers or
  businesses; check before modelling them like consumers.)
- **The 1 → 2 transition is the bottleneck:** only 5.9% of first-time buyers come
  back to buy, and half of those within ~2 days. That's the moment re-engagement
  should be built around.
- **Second orders are mostly "more of the same":** a third re-buy the exact
  item (replenishment), three quarters stay in the same top-level category.
- **Expanders come back about twice as often** (47.7% vs 24.8%). **This is
  correlation, not causation:** customers who were already more engaged may
  both explore more and return more. It's a hypothesis worth a randomized test
  ("send adjacent-category content to some, same-category to others"), not a
  reason to push every customer into new categories.
- Hillstrom (docs/DATASETS.md) randomized *which merchandise category* an email
  promoted, so it can test the cross-category hypothesis with a real holdout.

**Next:** lifecycle-aware evaluation in step 2 (metrics split by stage; a
category-novelty metric), then the lifecycle messaging layer in step 4 (see
docs/LIFECYCLE.md).
