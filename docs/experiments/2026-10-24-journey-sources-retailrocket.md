# 2026-10-24: Journey sources (re-buy, cart substitutes, adjacent category) on RetailRocket

**Status:** done
**Model(s):** `rebuy@8b87ca7f`, `cart_substitutes@d02e1d9b`, `adjacent_category@6b1192e7`
(all defaults) vs the step 2a sources and `most_popular@f290e0b6`.
**Data:** RetailRocket, same split as `2026-10-24-session-sources-retailrocket.md`
(fingerprint `29d3d0e9df12`, 6,440 add-to-carts, stock filter on). The
catalog now has a `type` facet: the second level of the category tree, 136
product types.

## Hypotheses (from docs/SHOPPER-PROFILE.md, stated before running)

1. **Re-buy** finds re-bought items and helps first-order and repeat customers.
2. **Cart substitutes** help abandoners, who comparison-shop (45% of their
   carts are new items in categories they browsed).
3. **Adjacent category** finds some of the new-category carts that no 2a
   source reached.

Each source does one job, so the test is its own segment, not the overall
number. The overall number is reported anyway.

## Method

```bash
uv run recsys evaluate data/retailrocket/catalog data/retailrocket/interactions.csv \
  --split next --test-fraction 0.1 --k 5 --k 20 --group-facet department \
  --availability data/retailrocket/availability.csv \
  --models most_popular,trending,recently_viewed,covisitation,rebuy,cart_substitutes,adjacent_category
```

RetailRocket has no prices or brands, so the substitutes' price band and the
shopper state's brand and price features are inactive here. They need a
dataset with prices (REES46).

## Results (hit rate@20)

| | All | Their target segment | Best 2a source there |
|---|---|---|---|
| `rebuy` (list length 0.6) | 0.007 | **re-buy carts: 0.575** (n=73); 1-order customers: 0.124 | recently_viewed 0.973 / 0.711 |
| `cart_substitutes` (list length 4.5) | 0.051 | **abandoners: 0.153** (n=557); new item in browsed category: 0.131 | covisitation 0.190 / 0.234 |
| `adjacent_category` (list length 17.6) | 0.001 | **new item, new category: 0.019** [0.000, 0.048] (n=104) | trending 0.048 |

Hits per recommended slot (hit rate ÷ list length) for the two sources that
target known intent: cart_substitutes 1.1%, covisitation 0.9%,
recently_viewed 11.7%.

## Conclusion

1. **Re-buy works, and is redundant on this store.**
   - It finds 58% of re-buy carts. Recently viewed already finds 97%,
     because RetailRocket re-buys come within days and the item is still
     among the shopper's latest.
   - Re-buy's value is long gaps: replenishment after weeks, when the item
     has dropped out of "recent". This store can't show that. It needs
     purchase data with repeat cycles (Instacart), and timing (when they'll
     run out) is the next refinement.
2. **Cart substitutes do their job but don't beat co-visitation alone.**
   - For abandoners: 15% vs 19%.
   - It is precise (1.1% of its slots are hits vs 0.9% for co-visitation)
     with short lists, so its value shows up when *blending*. Does it find
     carts co-visitation misses? That's the part 2 question.
   - It can't use price here; on a catalog with prices, the price band
     should sharpen it.
3. **Adjacent category doesn't work as a ranker here:** 1.9% on
   new-category carts vs 4.8% for trending. Three reasons, all expected from
   the analyses:
   - most new-category carts are cold starts with no history (714 of 818);
   - adjacency lifts are mild (1.5–2.5× on Amazon);
   - the exact item someone carts in a new type is hard to guess from
     popularity.

   Its job is **exploration**: putting a few expansion items in front of
   shoppers who are ready for them. Whether that *causes* expansion can only
   be measured with a randomized holdout (LIFECYCLE.md), not by replaying
   what shoppers did without it. Keep it as a small, measured share of
   slots, not a main source.
4. **No single source serves every journey.** The blender (part 2) picks the
   mix per shopper state: recently viewed and re-buy for repeat customers,
   co-visitation plus substitutes for active and abandoning shoppers,
   trending for cold starts, and a dial of adjacent-category slots.

**Caveats:**
- Small segments: re-buy n=73, 2+ orders n=69, new category n=104.
- No price or brand data in RetailRocket.
- Untuned defaults.
- Frozen at the cutoff for ~16 days.
