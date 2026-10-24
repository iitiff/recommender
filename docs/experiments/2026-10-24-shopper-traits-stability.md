# 2026-10-24: Which shopper traits are real? (stability test, REES46)

**Status:** done (descriptive, observational)
**Data:** REES46 multi-category store, Oct–Nov 2019 ("© Original Authors";
aggregates only). 11,705 bulk buyers (>10 items) removed.

## Question

To understand a shopper before choosing what to recommend, we need to know
which descriptions of them are **traits** (they still describe the person
next month) and which are **states** (they describe what they're doing right
now). A recommender that treats a state as a trait keeps showing yesterday's
person.

## Method

```bash
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/shopper_traits.py events.parquet sessions.parquet   # ~6 min
```

Each trait is computed per shopper in October and again in November, then
compared across shoppers (Spearman rank correlation). Two numbers per trait:
- **Month to month (Oct → Nov):** does the trait carry over? That is the
  stable, trait-like part.
- **Within a month (odd vs even days of October):** how consistent it is
  over a few weeks. This includes the *current shopping mission* (a phone
  hunt spans several days), so it sits above the month-to-month number.
  The gap between the two is roughly the part that belongs to the mission,
  not the person.

The traits:

| Trait | Measured as | Minimum activity |
|---|---|---|
| Brand loyalty | Viewing consecutive products in a category: how much more often they stay with the same brand than that category's brand mix predicts | 3 pairs per month |
| Mainstream vs niche | Mean popularity percentile, within its category, of products they view | 5 products |
| Variety | Distinct categories per product viewed | 10 products |
| Deal sensitivity | Discount they paid vs the product's usual price, minus the discount shoppers typically saw on it | 2 purchases |
| Comparison breadth | Alternatives in the same category viewed in the 7 days before a purchase (median per shopper) | 2 purchases |
| Rhythm | Their usual 3-hour window (from ≥5 October visits): how many November visits fall in it, vs the store's busiest window | 5 + 3 visits |

## Results

| Trait | Shoppers (both months) | Month to month | Within a month | Reads as |
|---|---|---|---|---|
| Mainstream vs niche | 506,909 | **0.31** | 0.34 | **Trait**: almost all of it carries over |
| Brand loyalty | 405,786 | **0.20** | 0.26 | **Trait**, modest |
| Comparison breadth | 19,094 | **0.31** | 0.59 | Half trait, half mission |
| Variety | 256,866 | **0.28** | 0.40 | Mostly trait |
| Deal sensitivity | 19,089 | **−0.06** | 0.38 | **State**: consistent within a month, gone the next |
| Weekend vs weekday share | 281,277 | **0.06** | – | Not a trait |

**Brand loyalty predicts what people buy.** For November purchases in a
category the shopper had browsed in October:

| October brand-loyalty tercile | Purchases | Bought the brand they viewed most in October |
|---|---|---|
| Low | 31,406 | 44% |
| Mid | 42,544 | 53% |
| High | 40,173 | **69%** |

**Rhythm:** a shopper's own usual 3-hour window contained 19.7% of their
November visits. The store's busiest window (17:00–20:00 Moscow time)
contained 18.6%. Personal timing barely beats "evening".

## Conclusion

1. **States outweigh traits.** The strongest trait correlations are
   0.2–0.3 month to month. The states from the stage deep dive (an open cart,
   an order yesterday) moved near-term ordering 1.5–3×. Understanding a
   shopper is mostly understanding **their current mission**, with a few
   traits as soft priors. Concepts: **short-term vs long-term preferences**,
   **interest drift**.
2. **Four traits are worth keeping as priors:**
   - **Brand loyalty**, the most actionable: loyal shoppers buy their
     browsed brand 69% vs 44% of the time. It decides whether alternatives
     stay within the brand.
   - **Mainstream vs niche**, the most stable: it decides how much
     popularity to blend in.
   - **Variety**: it sets the exploration dial.
   - **Comparison breadth**: it decides how many alternatives to show.
3. **Deal-seeking is a moment, not a person.** Someone who bought on
   discount in October was no more likely to do so in November. Price-drop
   messages should be triggered by the *product and the moment* ("the item
   you viewed just dropped"), not by a "deal seeker" label. This fits the
   product/price finding that price is mostly a property of the moment.
4. **Personal send times aren't worth it here.** Nearly everyone shops in
   the evening, and a person's own hour adds about 1 point. Message timing
   should follow momentum (open cart age, days since order), not the clock.
5. **Profiles must forget.** Since even real traits only half carry over,
   old evidence should decay (recency-weighted profiles), and the current
   visit should dominate. This matches the recently-viewed and co-visitation
   results.

**Caveats:**
- Two months only, so "month to month" is one step. Traits might be steadier
  over a year (Amazon Reviews can test that) or drift more.
- Deal sensitivity and comparison breadth need 2+ purchases per month, so
  they cover only 19k shoppers, the more active ones.
- Within-month consistency mixes noise with mission. These numbers rank
  traits; they aren't exact reliability coefficients.
- Mainstream scores are high for everyone (median percentile 0.91) because
  views concentrate on popular items. It's the *differences* between shoppers
  that persist.
- One store and two months, and November includes Black Friday. A store-wide
  sale shifts everyone's discounts, but only matters here if it changes
  *who* buys on discount.
