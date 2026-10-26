# 2026-10-24: Do the RetailRocket lifecycle patterns hold in other stores?

**Status:** done (descriptive, observational)
**Question:** The RetailRocket lifecycle analysis found (1) repeat customers buy a
large share of items, (2) few first-time buyers come back, quickly when they do,
(3) second orders are mostly "more of the same", and (4) customers whose second
order is in a new category place a third order about twice as often. Which of
these hold in other stores?

## Data

| Dataset | Store type | Window | License |
|---|---|---|---|
| RetailRocket | General online store | 137 days (2015) | CC BY-NC-SA 4.0 |
| Olist | Brazilian marketplace | 728 days (2016–18) | CC BY-NC-SA 4.0 |
| REES46 (Oct–Nov 2019) | Multi-category, electronics-heavy | 60 days | "© Original Authors" (aggregates only) |
| H&M | Fashion retailer | 733 days (2018–20) | Kaggle competition rules (aggregates only) |
| Online Retail II | UK gift **wholesaler** (B2B) | 738 days (2009–11) | UCI (CC BY 4.0, unverified) |

Only aggregate statistics are reported here; no data is redistributed.

## Method

```bash
python -I docs/experiments/scripts/lifecycle_compare.py <dataset> <path> --horizon 30
# REES46 is streamed first (14.8 GB → purchases only):
python -I docs/experiments/scripts/stream_rees46_purchases.py purchases.csv
```

The same definitions everywhere:
- **Order** = a customer's purchases on one calendar day. Session-based orders
  counted "came back an hour later for one more item" as a repeat purchase,
  and marketplace checkouts split across sellers created several order ids
  seconds apart.
- **Fixed 30-day horizons with right-censoring handled:** "second order within
  30 days" only counts customers whose first order had 30 days of data left.
  The same applies to "third order within 30 days of the second".
- **Expansion** = the 2nd order shares no top-level category with the 1st.
  **Robustness check:** customers whose 1st and 2nd orders each held a single
  item (so "expanded" can't just mean "bought more things").
- 95% Wilson confidence intervals. The 60-day horizon gives the same
  conclusions (REES46 excluded there: its window is only 60 days).

## Results (30-day horizon)

| | RetailRocket | Olist | REES46 | H&M | Online Retail (B2B) |
|---|---|---|---|---|---|
| Buyers | 11,719 | 94,983 | 697,470 | 1,362,281 | 5,881 |
| Buyers with 2+ orders | 4.7% | 2.2% | 30.0% | 67.1%* | 71.2%* |
| Share of items bought by 2+ order buyers | 34.0% (9.2% without resellers†) | 4.8% | 62.7% | 95.1%* | 95.3%* |
| 2nd order within 30 days | 4.3% | 0.7% | 35.1% | 28.1%* | 19.3%* |
| Median days 1st → 2nd order | 4.3 | 74 | 6.2 | 44 | 63 |
| 2nd order: same exact item | 25.4% | 15.1% | 24.5% | 9.2% | 75.9% |
| 2nd order: same top-level category | 76.3% | 40.1%‡ | 81.9% | 63.9% | – |

**P(3rd order within 30 days of the 2nd): stayed in category vs expanded**

| Dataset | Stayed | Expanded | Single-item orders: stayed → expanded |
|---|---|---|---|
| RetailRocket | 26.7% [22–32] (n=322) | **42.3%** [33–52] (n=104) | 7.9% → **30.4%** (n=151 / 46) |
| RetailRocket without resellers† | 11.1% [8–16] (n=243) | **25.6%** [17–36] (n=78) | 7.9% → **20.0%** (n=151 / 40) |
| Olist | 2.0% [1.2–3.2] (n=763) | 2.6% [1.8–3.7] (n=1,114) | 2.3% → 2.3% |
| REES46 | 57.0% [56.6–57.5] (n=53,661) | 54.7% [53.6–55.7] (n=8,933) | 51.5% → 50.8% |
| H&M | 36.1% [36.0–36.2] (n=573,320) | 35.2% [35.0–35.4] (n=324,007) | 34.7% → 36.0% |

\* **Left-censored:** a customer's first order *in the window* isn't their first
order ever. Long-time H&M or wholesale customers look "new", which inflates
repeat rates. RetailRocket, Olist and REES46 have the same issue to a lesser
extent.
† 1.2% of RetailRocket buyers bought more than 10 items each (max 559) and account
for 29% of all purchased items: almost certainly resellers or businesses.
‡ Olist's taxonomy is flat (~70 categories), so "top-level" is a narrower bucket there.

## Conclusion

| Pattern | Holds? |
|---|---|
| Repeat buyers buy a disproportionate share | **Yes, everywhere**, but the size swings enormously, and one store's "loyal" segment can be resellers (RetailRocket: 34% → 9% once removed). **Separate B2B/resellers before modelling loyalty.** |
| Most first-time buyers never return | **Depends on the business:** marketplace (Olist) 2%, general store 5%, electronics store 30%, fashion and B2B much higher (and left-censored). |
| Second orders come quickly | **No:** median 4 days (RetailRocket) to 74 days (Olist). Message timing must be **learned per store**, never hard-coded. |
| Second orders are "more of the same" | **Mostly yes:** 64–82% stay in the top category where hierarchies are comparable. Exact re-buys range from 9% (fashion) to 76% (wholesale replenishment). |
| Expanding to a new category → twice the 3rd-order rate | **No.** Only RetailRocket shows it (and with small samples). Olist, REES46 and H&M (n up to 900k) show differences of ±1–2 points, in both directions. |

**What changes:**
- Category expansion stays a *hypothesis to test per store with a randomized
  holdout*, not a default strategy. A store-specific association is exactly
  what an uplift experiment exists to check.
- Everything with "timing" or "rate" in it must be learned per store (per
  tenant in the Shopify app): the same rule of thumb would be wrong by an order
  of magnitude somewhere.
- Lifecycle analysis needs a reseller/B2B screen first, and every comparison
  needs explicit censoring rules and one order definition.

**The broader lesson:** one dataset produced a striking, plausible,
statistically clear pattern (2×). Four replications showed it isn't general.
Replicate before you build strategy on a pattern.
