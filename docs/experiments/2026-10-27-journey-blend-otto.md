# 2026-10-27: Journey blend on OTTO: do the RetailRocket rules transfer?

**Status:** done
**Model(s):**
- `journey_blend@4bf72b7e`: the RetailRocket rules unchanged (recently viewed weight 3, up to 50 items).
- `journey_blend@2f2cddec`: tuned for OTTO (recently viewed weight 6, only the 5 most recent items boosted).
- Compared with every single source and the baselines.

**Data:** OTTO (CC BY 4.0), 1 in 64 shoppers: 201,560 shoppers, 3.39M events, 579k articles (`recsys dataset prepare otto`). Next add-to-cart, no stock filter (OTTO has no availability data).
- **Test window:** split `4fdcc8ca8f9a`, from Aug 26, 2022 12:28 UTC, 24,672 cases.
- **Validation window:** split `9ee1de9048c2`, the ~3 days before the test window (data after it dropped), 20,667 cases.

## Question

The journey blend's rules and its one tuned number (recently viewed weighs 3)
were set on RetailRocket. Do they transfer to a different store, or does each
store need its own tuning? This matters for the Shopify app: the defaults ship
to stores we have no data for yet.

## Method

```bash
uv run recsys dataset prepare otto RAW_DIR data/otto --sample-every 64
uv run recsys evaluate data/otto/catalog data/otto/interactions.csv \
  --split next --test-fraction 0.1 --k 5 --k 20 --baseline recently_viewed \
  --model-config docs/experiments/configs/2026-10-27-journey-blend-otto.yaml
# tuning, validation window only:
#   --model-config docs/experiments/configs/2026-10-27-journey-blend-otto-tuning.yaml \
#   --until 2022-08-26T12:28:39.357+00:00
```

**Protocol (same as RetailRocket):**
1. Run the RetailRocket rules unchanged on the test window (the transfer test).
2. Tune on the **validation window only**. Two knobs: recently viewed's
   weight (3 or 6), and how many recently viewed items the blend boosts (50,
   10 or 5).
3. Run the chosen variant **once** on the test window.

**OTTO has no categories, prices or stock.** Three things stay idle:
- cart substitutes;
- adjacent-category exploration;
- the stock filter.

So on OTTO the blend mixes recently viewed, co-visitation, trending and
re-buy, weighted by journey. The "discovery" segment is every new article
(the category is unknown).

## Results

**1. The RetailRocket rules, unchanged (test window, paired vs recently viewed)**

| | Hit@5 | Hit@20 | MRR@20 | Return | Discovery |
|---|---|---|---|---|---|
| recently_viewed | 0.770 | 0.793 | **0.490** | 0.982 | 0 |
| covisitation | 0.062 | 0.132 | 0.040 | 0.130 | **0.143** |
| blend, RetailRocket rules | 0.736 (**−0.033**) | 0.807 (+0.014) | 0.444 (**−0.046**) | 0.984 | 0.067 |

On RetailRocket the same rules gained at the top (hit@5 +0.015, MRR −0.003,
not significant) and kept 92% of co-visitation's discovery. On OTTO they
**lose at the top** and keep only 47% of discovery.

**Why:**
- OTTO shoppers have long histories (4 weeks), so recently viewed returns
  about 15 items (average list length 14.6), against 5.4 on RetailRocket.
- At weight 3, every one of them outranks co-visitation's best item. That
  leaves co-visitation 5 slots, which hurts discovery.
- An older viewed item that co-visitation also likes gets both
  contributions. It jumps above the item on screen, which hurts the top.

**2. Tuning on the validation window**

| Variant | Hit@5 | Hit@20 | MRR@20 | Discovery |
|---|---|---|---|---|
| recently_viewed alone | 0.787 | 0.809 | 0.512 | 0 |
| A: weight 3, 50 items (RetailRocket) | 0.750 | 0.821 | 0.460 | 0.066 |
| B: weight 6, 50 items | 0.775 | 0.821 | 0.494 | 0.066 |
| C: weight 3, 5 items | 0.790 | 0.816 | 0.480 | 0.106 |
| **D: weight 6, 5 items (chosen)** | **0.790** | 0.816 | **0.503** | **0.106** |
| E: weight 6, 10 items | 0.775 | 0.821 | 0.494 | 0.097 |

**Why D:**
- It's the only variant that doesn't lose at the top: hit@5 is at least
  recently viewed's.
- It has the best discovery.
- It has the smallest MRR loss among those two.
- B and E gain 0.5 point of hit@20 but give up 1.5 points at the top and
  about a third of the discovery.

**3. The chosen variant, run once on the test window (paired vs recently viewed)**

| | Hit@5 | Hit@20 | MRR@20 | Return | Discovery |
|---|---|---|---|---|---|
| blend, RetailRocket rules | −0.033 [−0.036, −0.031] | +0.014 [+0.013, +0.016] | −0.046 [−0.048, −0.043] | 0.984 | 0.067 |
| **blend, tuned for OTTO** | **+0.003** [+0.002, +0.004] | **+0.010** [+0.008, +0.013] | −0.007 [−0.008, −0.006] | 0.965 | **0.124** |

**By next action** (hit rate@20, test)

| Next cart is… | Cases | recently_viewed | covisitation | RetailRocket rules | Tuned for OTTO |
|---|---|---|---|---|---|
| Seen this visit | 18,684 (76%) | 0.998 | 0.114 | 0.998 | 0.990 |
| Seen in an earlier visit | 903 (4%) | 0.650 | 0.334 | **0.684** | 0.451 |
| Re-buy | 335 (1%) | 0.970 | 0.457 | 0.979 | 0.982 |
| New article (discovery) | 4,724 (19%) | 0 | **0.143** | 0.067 | **0.124** |

**By stage** (hit rate@20, test): the tuned blend wins or ties recently
viewed in every stage (new visitor 0.810 vs 0.772, browser 0.825 vs 0.811,
abandoner 0.802 vs 0.800, 1 order 0.796 vs 0.792, 2+ orders 0.794 vs 0.796,
bulk 0.754 vs 0.758).

## Conclusion

1. **The structure transfers; the numbers don't.**
   - Journey weights with rank fusion again beat every single source on
     hit@20 without losing the top, but only after re-tuning on OTTO's own
     validation window.
   - Untuned, the RetailRocket weight hurt the top of the list (hit@5
     −3.3 points).
   - This is the per-store learning LIFECYCLE.md argues for, now shown for
     the recommender too.
   - **For the Shopify app:** ship the structure with conservative
     defaults, and tune the few knobs per store on its own validation
     window before any test. That is roadmap step 2c (learned weights).
2. **History length is the store property that matters here.**
   - OTTO shoppers bring about 15 recently viewed items, RetailRocket
     shoppers about 5.
   - A source's influence in rank fusion grows with the length of its list.
     So "how many items to boost" is a knob as important as the weight.
3. **The tuned version trades one return case for discovery.**
   - Discovery nearly doubles (6.7% → 12.4%, 87% of co-visitation's 14.3%).
   - But re-carting an item from an *earlier* visit drops (68% → 45%),
     because only the 5 most recent items are boosted.
   - A cleaner design would boost **everything from this visit plus the
     last few items from earlier visits**: session-aware recency instead of
     a flat cap. That's a candidate for 2c, not built here.
4. **The gains are smaller than on RetailRocket** (+1.0 vs +4.7 points of
   hit@20).
   - 76% of OTTO's next carts are an article on screen this visit, which
     recently viewed already finds 99.8% of the time.
   - The room left is discovery (19% of carts), where co-visitation at
     14% is the ceiling of what's here.
   - Better discovery needs item content (categories, text, images),
     which OTTO doesn't have.

**Caveats:**
- **Sample:** 1 in 64 shoppers. Co-visitation is learned from that sample
  only, so it is weaker than it would be on all of OTTO.
- **Idle components:** no categories, so cart substitutes and exploration
  slots aren't tested here. No stock filter.
- **Cold start:** essentially absent (26 cases): every OTTO shopper in the
  test window has earlier events.
- **Offline replay:** shoppers saw OTTO's own recommendations, not ours.
