# 2026-10-26: Journey blend on RetailRocket: one list, many journeys

**Status:** done
**Model(s):** `journey_blend@4bf72b7e` (rules by journey, recently viewed weight 3),
`journey_blend@cfcf1b0a` (same + exploration slots) vs the control
`journey_blend@7c4bcb2f` (uniform weights), every single source, and the baselines.
**Data:** RetailRocket (CC BY-NC-SA 4.0), next add-to-cart, stock filter on.
- **Test window:** split `29d3d0e9df12`, the last ~16 days, 6,440 cases.
- **Validation window:** split `6cc88161c2c9`, the ~15 days before the test
  window (data after it dropped), 6,100 cases.

## Question

Can one recommender serve every journey: a returning shopper getting back to
what they saw, an active shopper discovering new items, an abandoner
comparing, a cold start? And do journey-specific weights beat treating
everyone the same?

## Method

`JourneyBlender` (`src/recsys/models/blend.py`):
1. **Journey:** "cold start" (no history) or the lifecycle stage.
2. **Weights per journey** (rules from `docs/SHOPPER-PROFILE.md`).
3. **Weighted reciprocal rank fusion:** each source adds
   weight / (10 + rank) per item. Ranks, not scores, because sources' scores
   aren't comparable.
4. **Exploration:** every 10th slot goes to the adjacent-category source,
   for browsers, abandoners and buyers.

Every source gets the same candidates and exclusions. The blend can only
reorder what the hard filters allow.

```bash
uv run recsys evaluate data/retailrocket/catalog data/retailrocket/interactions.csv \
  --split next --test-fraction 0.1 --k 5 --k 20 --group-facet department \
  --availability data/retailrocket/availability.csv \
  --model-config docs/experiments/configs/2026-10-26-journey-blend.yaml \
  [--until 2015-09-02T17:49:11.563+00:00]   # validation window
  [--baseline recently_viewed]               # paired comparison vs the best source
```

**Avoiding self-deception:**
- The journey rules came from findings on the test window, so they were also
  checked on the earlier validation window.
- The one tuned number (recently viewed's weight) was chosen **on the
  validation window only**, comparing weights 1, 2 and 3
  (`configs/2026-10-26-journey-blend-tuning.yaml`). Then the test window was
  run **once** to confirm.

**Headline metrics:** hit rate@20, split by **need**:
- **return:** the cart is an item seen before;
- **discovery:** a new item;
- **cold start:** no history.

Plus MRR@20 and hit rate@5, because position matters.

## Results

**Tuning on the validation window** (recently viewed's weight in the rules):

| Weight | Hit@5 | Hit@20 | MRR@20 |
|---|---|---|---|
| recently_viewed alone | 0.632 | 0.644 | 0.519 |
| 1 (first version) | 0.545 | 0.691 | 0.450 |
| 2 | 0.631 | 0.693 | 0.506 |
| **3 (chosen)** | **0.641** | **0.693** | **0.517** |

At weight 1 the fusion pushed the item on screen down the list. At 3 the
blend keeps recently viewed's top positions and adds discovery below them.

**Final, both windows** (hit rate@20 unless noted)

| | Test: all | Return | Discovery | Cold | MRR@20 | Validation: all | MRR@20 |
|---|---|---|---|---|---|---|---|
| most_popular | 0.022 | 0.025 | 0.009 | 0.027 | 0.005 | 0.032 | 0.005 |
| trending | 0.024 | 0.023 | 0.016 | 0.041 | 0.009 | 0.033 | 0.006 |
| recently_viewed | 0.630 | **0.935** | 0 | 0 | **0.513** | 0.644 | 0.519 |
| covisitation | 0.143 | 0.143 | **0.216** | 0 | 0.048 | 0.142 | 0.050 |
| blend, uniform weights | 0.656 | 0.907 | 0.192 | 0.041 | 0.394 | 0.668 | 0.402 |
| **blend, journey rules** | **0.677** | **0.935** | 0.198 | **0.041** | 0.510 | **0.693** | 0.517 |
| blend, journey rules + exploration | 0.676 | 0.935 | 0.196 | 0.041 | 0.510 | 0.692 | 0.517 |

**Paired vs recently_viewed, test window:**

| | ΔHit@20 | ΔHit@5 | ΔMRR@20 |
|---|---|---|---|
| journey rules | **+0.047** [+0.042, +0.053] | **+0.015** [+0.011, +0.020] | −0.003 [−0.006, +0.001] |
| uniform weights | +0.027 [+0.020, +0.034] | −0.126 [−0.135, −0.116] | −0.119 [−0.126, −0.112] |

**By journey** (hit rate@20; test / validation)

| Stage | recently_viewed | uniform blend | journey rules |
|---|---|---|---|
| New visitor | 0.577 / 0.599 | 0.621 / 0.650 | **0.638 / 0.657** |
| Browser | 0.755 / 0.695 | 0.773 / 0.734 | **0.792 / 0.747** |
| Abandoner | 0.476 / 0.651 | 0.497 / 0.636 | **0.517 / 0.704** |
| 1 order | 0.711 / 0.653 | 0.711 / 0.639 | **0.732 / 0.689** |
| 2+ orders | 0.768 / 0.783 | 0.739 / 0.729 | **0.783 / 0.783** |
| Bulk buyer | 0.857 / 0.794 | 0.808 / 0.688 | **0.857 / 0.794** |

Cost: 2.6 ms per request (vs 0.7 ms for co-visitation alone), and about a
minute to fit, because the blend fits each source.

## Conclusion

1. **One list now serves both jobs almost as well as the best specialist for
   each.**
   - **Return:** 0.935, the same as recently viewed.
   - **Discovery:** 0.198, 92% of co-visitation's 0.216.
   - **Cold start:** 0.041, the same as trending.

   Overall it beats the best single source by 4.7 points (paired CI excludes
   zero) without losing the top position (MRR unchanged).
2. **Journey rules beat treating everyone the same.**
   - **Overall:** hit rate +2.1 points (test) and +2.5 (validation).
   - **MRR:** 0.394 → 0.510.
   - **Per journey:** the rules win or tie in every stage in both windows.
     The uniform blend *hurt* repeat and bulk buyers (0.808 vs 0.857), who
     mostly re-cart known items: mixing discovery into their list displaced
     what they came for.
3. **Rank fusion needs the specialists' strengths protected.** At weight 1,
   fusing recently viewed with co-visitation cost 6 points of MRR. The fix
   (weight 3) was found on the validation window and held on the test window.
   Position matters as much as presence.
4. **Exploration slots cost almost nothing offline:** −0.1 point, within
   noise. What they *earn* (customers moving into new categories) can't be
   measured by replaying the past. That needs a randomized holdout online,
   which is why the dial exists.

**Caveats:**
- One store, two adjacent windows, frozen models (no daily retraining yet:
  roadmap step 3).
- The rules are hand-set from analyses of this same store. The validation
  window guards against tuning to the test window, not against store-specific
  rules; replicate on another store (REES46) before generalizing.
- No prices or brands in RetailRocket, so the price band and brand affinity
  don't contribute here.
- Offline replay rewards predicting what shoppers did without our
  recommendations. Lift has to be shown online.

**Next:**
- **Learned weights:** replace the hand rules with weights per journey
  learned on a validation window, then a learned ranker over the
  `ShopperState` features (roadmap: learning to rank).
- **Replicate:** run the blend on a REES46 sample, where price and brand exist.
- **Then step 4:** lifecycle messaging. The same shopper state and sources
  decide *who/what/when* to message, with uplift on Hillstrom and Criteo.
