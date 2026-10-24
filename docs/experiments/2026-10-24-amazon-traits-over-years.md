# 2026-10-24: Shopper traits and journeys over years (Amazon Reviews 2023)

**Status:** done (descriptive, observational)
**Data:** Amazon Reviews 2023 (McAuley Lab, UCSD): Electronics, Home &
Kitchen, Grocery & Gourmet Food. License not stated on the dataset card
(treated as research use, aggregates only; see docs/DATASETS.md).
- 124M ratings, 5.95M products, 1996–2023.
- Analysis uses the 110M ratings from the 19.3M shoppers with 2+ ratings.
- 98% of products have a brand; 33% have a price.

## Question

REES46 covered two months. Do the traits we found there (price level,
mainstream taste, brand loyalty, per-type price band) hold over **years**?
And how do shoppers' journeys across categories unfold over years?

**The big caveat, up front:** a rating is evidence of a purchase, but only
of purchases people chose to review.
- Re-buys are almost never reviewed: the file has one rating per
  shopper-product, so **replenishment is invisible** here.
- Prices are from the 2023 crawl, not from the time of purchase.

## Method

```bash
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/stream_amazon.py Electronics DIR          # and the other two (~7 min each, parallel)
uv run --no-project --with pandas --with pyarrow python -I \
  docs/experiments/scripts/amazon_journeys.py DIR OUT_DIR             # ~10 min, <12 GB RAM
```

- **Price level:** mean price percentile within the leaf category (≥20
  priced products).
- **Mainstream:** mean popularity percentile (number of ratings) within the
  leaf category.
- **Rating style:** mean stars given.
- **Year-to-year:** a shopper's year Y vs Y+1 and Y+3, each year with ≥3 ratings.
- **Per-type band:** a shopper's price level in a product type (second-level
  category) with ≥2 priced ratings, compared a year later in the *same* type
  vs a *different* type. One shopping mission rarely spans a calendar year,
  so same-type carry-over is a lasting preference.
- **Brand loyalty:** consecutive purchases in the same leaf category,
  P(same brand) vs chance (Σ brand share² in that category), by the time
  between them.
- **Journeys:** product types entered over tenure (shoppers observed 5+
  years), and the next new type entered after each type, as lift over that
  type's base rate. Merchandising nodes ("Back to School…", promo
  storefronts) are filtered out by requiring ≥1% share.

All sections were smoke-tested on random data: correlations ≈ 0, brand
repeat = chance, lifts ≈ 1.

## Results

### 1. Traits over years (Spearman, same shopper)

| Trait | 1 year apart | 3 years apart |
|---|---|---|
| Rating style (generous vs harsh) | **0.44** (2.50M pairs) | **0.39** (1.14M) |
| Price level | 0.18 (1.12M) | 0.15 (441k) |
| Mainstream vs niche | 0.18 (2.25M) | 0.13 (1.02M) |

### 2. Per-type price band, a year later

| Compared with | Pairs | Spearman |
|---|---|---|
| The **same** product type next year | 600,831 | **0.20** |
| A **different** product type next year | 1,968,013 | 0.09 |

### 3. Brand loyalty over time (next purchase in the same leaf category)

| Time between | Grocery: same brand / chance | Electronics | Home & Kitchen |
|---|---|---|---|
| < 30 days | **22.2%** / 1.3% | 14.0% / 2.3% | 14.2% / 1.0% |
| 30 days – 1 year | 13.4% / 1.1% | 9.4% / 2.2% | 5.7% / 0.9% |
| 1–3 years | 9.6% / 1.0% | 7.7% / 2.3% | 3.8% / 0.9% |
| 3+ years | **6.5%** / 0.9% (7×) | 5.4% / 2.4% (2.3×) | 2.8% / 1.0% (2.8×) |

### 4. Journeys over tenure (4.66M shoppers observed 5+ years)

| End of tenure year | 0 | 1 | 2 | 4 |
|---|---|---|---|---|
| Product types entered (mean) | 1.58 | 1.96 | 2.34 | 3.15 |
| Share of that year's rated purchases in a type new to them | 78% | 53% | 49% | 44% |

**Next new type entered** (share of next entries; lift over base rate):

| After… | Most over-represented next types |
|---|---|
| Television & Video | Home Audio (4.8%, **2.4×**), Car electronics (1.8×), Computers & Accessories (15%, 1.6×) |
| Camera & Photo | Security & Surveillance (2.1×), Computers & Accessories (17%, 1.8×), GPS (1.7×) |
| Computers & Accessories | Headphones (6.9%, 1.7×), Camera & Photo (1.6×), TV & Video (1.6×) |
| Pantry Staples | Breads & Bakery (2.3×), Snacks & Sweets (2.1×), Beverages (1.9×) |
| Home Décor | Wall Art (1.9×), Seasonal Décor (1.7×), Party Supplies (1.6×), Bedding (1.5×) |
| Bath | Storage & Organization (1.5×), Home Décor (1.4×), Bedding (1.4×) |

Another purchase in the same leaf category within a year: Grocery 15%,
Electronics 14%, Home & Kitchen 13%. The same product is never rated twice.

## Conclusion

1. **Traits persist for years, weakly, and fade slowly.** Price level and
   mainstream taste keep about 0.15–0.18 of their rank one to three years
   later. That is real but weak per year. Use them as background priors that
   decay, never as fixed labels.
2. **Rating style is the strongest trait (0.44, still 0.39 three years
   later).** Some people give 5 stars to everything, others rarely do. Any
   model that learns from star ratings must correct for each shopper's own
   baseline. Concepts: **user bias**, **mean-centering** (the baseline
   predictor in collaborative filtering).
3. **The per-type price band is a lasting preference, not just the current
   search.** A year later it carries over twice as strongly within the same
   type (0.20) as across types (0.09). This confirms the hierarchical price
   profile (`2026-10-24-shopper-price-level.md`): a per-type band, plus a
   weaker general level underneath.
4. **Brand loyalty is strong and long-lived, strongest for consumables.**
   Even 3+ years later, the next grocery purchase in a category is the same
   brand 7× more often than chance (2–3× for electronics and home). It
   decays with time, so model brand affinity per category with a time decay
   (a half-life). Concept: **brand affinity with recency decay**. Note that
   the <30-day numbers partly reflect several items of one brand bought
   together.
5. **Journeys expand slowly but steadily, and adjacency is real but mild.**
   - Shoppers add about 0.4 new product types a year.
   - Natural next steps are over-represented 1.5–2.4×: TV → home audio,
     camera → security, pantry → bakery and snacks, décor → wall art.
   - This is what the adjacent-category source in roadmap step 2b needs: it
     learns these transitions per store and weights them by lift.
   - The modest lifts are a warning: expansion is a nudge, not a script.
6. **What reviews can't tell us:** replenishment (re-buys aren't reviewed),
   visits, carts, and discount timing. REES46 and Instacart-style order data
   stay necessary for those.

**Compared with REES46 (two months):** traits that carried over month to
month (price level, mainstream, brand) also carry over years, weaker per
year. The pattern is consistent: understanding someone means a strong
reading of their current mission, plus slowly fading priors.

**Caveats:**
- Reviewers aren't typical shoppers, and reviews aren't all purchases.
- Prices are from 2023 for purchases going back to 1996. Percentiles within
  category soften this but add noise, so price-level correlations are likely
  understated.
- Category trees changed over 27 years, and some nodes are merchandising, not
  product types.
- Three of 33 categories.
