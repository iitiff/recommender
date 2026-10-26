# 2026-10-24: Site visits by lifecycle cohort

**Status:** done (descriptive, observational)
**Data:** RetailRocket (CC BY-NC-SA 4.0) and REES46 Oct–Nov 2019 ("© Original
Authors"; aggregates only). These are the only datasets in the comparison with
browsing sessions; Olist, H&M and Online Retail II record purchases only.

## Question

How many visits do customers make at each lifecycle stage, how many visits
does a first order take, and how many separate the first and second orders?

## Method

```bash
python -I docs/experiments/scripts/visits_by_cohort.py retailrocket data/retailrocket
python -I docs/experiments/scripts/visits_by_cohort.py rees46 sessions.parquet   # streams 14.8 GB
```

- **Visit** = a session: RetailRocket's 30-minute inactivity sessions; REES46's own `user_session`.
- **Window:** each user is followed for the **first 30 days after their first
  visit**, and only users first seen ≥30 days before the data ends are counted.
  Cohorts are defined by what happened inside that window.
- **Order** = a calendar day with a purchase. **Bulk buyers** (>10 items in
  the window) are separated as likely resellers.

## Results

**Visits in the first 30 days** (median, with 90th percentile in brackets)

| Cohort | REES46 users | REES46 visits | RetailRocket users | RetailRocket visits |
|---|---|---|---|---|
| Browse only | 2,431,554 (80.4%) | 2 (6) | 1,102,937 (97.3%) | 1 (2) |
| Carted, no order | 185,114 (6.1%) | 5 (15) | 21,279 (1.9%) | 1 (3) |
| 1 order | 284,574 (9.4%) | 5 (15) | 8,827 (0.8%) | 2 (5) |
| 2 orders | 70,237 (2.3%) | 7 (20) | 239 (0.02%) | 4 (11) |
| 3+ orders | 42,216 (1.4%) | 10 (25) | 42 (<0.01%) | 8 (77) |
| Bulk buyers (>10 items) | 8,595 (0.3%) | 21 (47) | 97 (0.01%) | 14 (59) |

Share of browse-only users who came back at least once: REES46 55%, RetailRocket 11%.

**Getting to the first and second order** (1-order cohort unless noted)

| | REES46 | RetailRocket |
|---|---|---|
| Visits up to and including the first order | median 2, mean 4.1 | median 1, mean 1.7 |
| Bought on the first visit | 32% | 66% |
| Days from first visit to first order | median 2.0 | median 0 |
| Visits from first to second order (2-order cohort) | median 2, mean 3.8 | median 2, mean 2.7 |

## Conclusion

- **Visits climb step by step with loyalty** (REES46 median: 2 → 5 → 7 → 10).
  Visit frequency is a **leading indicator** that shows engagement before a
  second order confirms it.
- **Cart abandoners are as engaged as buyers** (REES46: median 5 visits, mean
  7.3, vs 5 and 6.8 for one-order customers). Interest isn't the problem;
  something blocks the purchase. They are the strongest re-engagement target.
- **First orders usually take more than one visit** (REES46: median 2 visits,
  2 days). The first visit is mostly research.
- **The second order comes about two visits after the first**, so the practical
  post-purchase goal is one or two more visits.

**Caveats:**
- RetailRocket's visitor ids are per-device cookies. Cross-device shoppers look
  like new visitors, which undercounts visits and likely explains the
  implausible "66% buy on the first visit". REES46's ids are more stable.
- Left-censoring: a first visit in the data may not be the first ever. That's
  likely why REES46's 3+ order customers often buy on their "first" visit.
- Visits and orders grow together (more visits are more chances to buy), so
  this does not show that driving visits *causes* loyalty.

**Implications** (folded into docs/LIFECYCLE.md):
- Visit frequency and recency become lifecycle features: RFM-style, but on
  visits, which arrive well before a second order.
- "Return visit within 7 days" is a fast, leading metric for messages, with
  purchases as the lagging metric that decides. Never optimize visits alone.
