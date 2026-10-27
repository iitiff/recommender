# 2026-10-27: Off-policy evaluation dry run (Open Bandit Dataset)

**Status:** done (method check on public data; no engine change)
**Follows:** LIFECYCLE "Messaging frequency" (point 4: log every send
decision with its probability) and experiments 2.16–2.17 in
`docs/VALIDATION-PLAN.md`.

**Data:** Open Bandit Dataset (ZOZOTOWN, fashion e-commerce, Nov 2019, CC
BY 4.0), "men" campaign.
- ZOZOTOWN ran an A/B test of two policies choosing 3 of 34 items for a
  recommendation slot:
  - **Random** (uniform): 453k impressions;
  - **Bernoulli Thompson Sampling (BTS):** 4.08M impressions.
- Every impression logs the **true probability** that the item was shown in
  that position (the propensity), plus whether it was clicked.
- Aggregates only.

## Why this matters

Before a new send policy goes live (a per-customer budget, a new scoring
rule), we want to know how it *would* have done, using the logs of the
current policy. This is **off-policy evaluation (OPE)**, also called
counterfactual evaluation.

OPE only works if:
1. each logged decision carries the probability with which it was made;
2. the current policy sometimes does what the new one would do (the
   **overlap** or **support** condition).

Public data almost never has both. This dataset does, and because both
policies really ran, each policy's real click rate is known. So we can check
how wrong the estimates are, instead of trusting them.

## Hypothesis

Estimators that use the logged propensities (IPW and doubly robust
variants) recover the other policy's click rate within about 10%. A pure
reward model (DM) is biased toward the policy that produced the logs.

**Decision metric (chosen before running):** relative error |estimate −
truth| / truth. The truth is the evaluation policy's own click rate in its
arm of the A/B test.

## Method

```bash
uv run --no-project --python 3.10 --with "obp==0.5.7" --with "pyyaml>=6" \
  --with "matplotlib<3.9" --with "numpy<2" python -I \
  docs/experiments/scripts/ope_dry_run.py OBD_DIR men 20 100000
# and the same with 10000 logged rounds
```

- **Library:** `obp` (Open Bandit Pipeline; Saito et al., NeurIPS 2021
  Datasets track). It pins old numpy and torch, hence Python 3.10.
- **Two directions:**
  - Random logs → value of BTS: from exploration data, evaluate a smarter
    policy.
  - BTS logs → value of Random: from a concentrated policy, evaluate one
    that does very different things.
- **Evaluation policy:** BTS is reproduced with obp's ZOZOTOWN production
  prior and simulated to get its probability of showing each item in each
  position.
- **Estimators:**
  - **DM (Direct Method):** a click model (logistic regression on user and
    item features, 3-fold cross-fitting) predicts what the new policy's
    choices would get.
  - **IPW (inverse probability weighting):** reweights logged clicks by new
    probability ÷ logged probability.
  - **SNIPW:** self-normalized IPW.
  - **DR (doubly robust):** DM plus an IPW correction of its errors.
  - **SNDR:** self-normalized DR.
  - **Switch-DR:** uses IPW where weights are small and DM where they are
    huge.
- **Uncertainty:** 20 bootstrap samples of 100k logged rounds, and of 10k
  (a smaller send test).

## Results

The true click rates:
- Random: **0.51%**;
- BTS: **0.67%** (+32% over Random).

**Relative error, mean ± sd over 20 bootstrap samples** (lower is better):

| Estimator | Random logs → BTS, 100k | Random logs → BTS, 10k | BTS logs → Random, 100k | BTS logs → Random, 10k |
|---|---|---|---|---|
| DM (reward model only) | 24% ± 3% | 23% ± 12% | 27% ± 7% | 34% ± 16% |
| IPW | 12% ± 6% | 22% ± 16% | **12% ± 7%** | 31% ± 24% |
| SNIPW | **11% ± 6%** | 22% ± 16% | 16% ± 10% | 37% ± 24% |
| DR | 11% ± 7% | 22% ± 16% | 27% ± **31%** | 52% ± **79%** |
| SNDR | 11% ± 7% | 23% ± 16% | 19% ± 16% | **27% ± 26%** |
| Switch-DR | 11% ± 7% | 22% ± 16% | 13% ± 9% | 29% ± 30% |

**Mean estimates, random logs → BTS (100k):**
- DM: 0.51%, which is the logging policy's own rate;
- the weighted estimators: 0.60–0.61%, against a truth of 0.67%.

So they find BTS better, but estimate the lift at about +20% instead of
+32%.

## Conclusion

1. **Logged probabilities are what make this possible.** Every estimator
   that used them beat the reward model alone.
   - DM landed on the *current* policy's click rate (0.51%) and saw no
     lift at all.
   - That is the trap of "predict each customer's response, then add it
     up": the model learned from the old policy's choices and can't tell
     the new choices apart.
   - For messaging, **every send decision must be logged with its
     probability** (LIFECYCLE point 4). A send log without propensities
     only supports this biased kind of estimate.
2. **Exploration logs evaluate new policies well; concentrated logs
   don't.**
   - From uniform-random logs, all weighted estimators agree (about 11%
     error).
   - From BTS logs, which rarely show many items, the weights explode. DR
     swings by ±31% (±79% at 10k), and the estimators disagree.
   - This is the overlap condition in practice. If today's policy sends a
     cart reminder to everyone, no estimator can say what "don't send"
     would do.
   - That is why experiment 2.17 sends at a **fixed probability** (e.g.
     40%) before 2.16 tries a budget.
3. **Precision is coarse: about ±10% with 100k decisions, ±20% with 10k.**
   - OPE ranked a +32% better policy correctly, but underestimated the
     gap.
   - It can't see a +5% improvement. Use it to **screen** candidate
     policies and drop bad ones, then confirm the winner with a randomized
     test.
   - Messaging outcomes (an order within 7 days) are far more common than
     a 0.5% click, which helps precision. Rare outcomes such as opt-outs
     will be as noisy as here.
4. **Use the robust estimators, and report several.**
   - Self-normalized IPW, SNDR and Switch-DR were the steadiest.
   - Plain DR, often recommended as "the best of both", was the most
     fragile with concentrated logs.
   - When estimators disagree, that is the warning sign; don't pick the one
     you like.

**For the retailer's send logs:**
- Phase 0.1 now asks for every send decision, holds included, to be
  logged with its probability.
- The dry run is ready to point at those logs: one row per decision with
  context, action (send / which message / hold), probability and outcome.
  That is the `obp` bandit-feedback format.

**Caveats:**
- **Clicks on a recommendation slot are not messages.** Here there are 34
  items × 3 positions. A send decision is closer to 2–5 actions (hold or
  one of a few messages), and overlap is easier.
- **Part of the error may come from reproducing BTS.** obp uses BTS's
  final prior parameters, while the live policy kept updating during the
  week.
- **The bootstrap resamples one set of logs,** so it shows sampling noise
  but not drift over time.
- **Single campaign ("men").** The "all" campaign doubles the data and
  memory.
- **Not run:** policy learning (training a new policy from logs, `obp`
  off-policy learners) and slate or sequence estimators. Those matter once
  2.17 produces logs.
