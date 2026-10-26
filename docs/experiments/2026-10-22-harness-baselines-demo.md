# 2026-10-22: Evaluation harness check with baselines on demo data

**Status:** done
**Model(s):** `most_popular@63cb5f3b` vs `random@843f5ee4`
**Data:** fictional demo catalog (30 recommendable + 10 quiz-only wines),
synthetic interactions from `examples/demo/generate_interactions.py`
(300 users, 5,179 interactions over ~6 months).

> Synthetic users test the **pipeline**, not models. They're generated from
> simple assumptions with a built-in popularity skew, so these numbers say
> nothing about real people.

## Hypothesis

If the harness is wired correctly:
1. Random's recall@10 matches chance. *(Corrected 2026-10-23: chance is not 10/30. Each user's
   already-rated wines are excluded first, so the exact expectation is the mean of
   min(10, pool)/pool over users: **0.357** temporal, **0.377** leave-last-out.)*
2. MostPopular beats Random on accuracy, because the data has a popularity skew.
3. MostPopular has lower coverage and higher exposure Gini (it shows the same hits to everyone).

**Decision metric:** NDCG@10, paired difference vs MostPopular.

## Method

```bash
uv run recsys evaluate examples/demo examples/demo/interactions.csv               # temporal, last 20%
uv run recsys evaluate examples/demo examples/demo/interactions.csv --split last  # last positive per user
```

## Results

**Temporal split** (data `6e96e1ec7580`, 77 users evaluated):

| Model | Recall@10 | NDCG@10 | Hit rate@10 | Coverage@10 | Gini@10 | Avg popularity@10 |
|---|---|---|---|---|---|---|
| random | 0.330 [0.253, 0.411] | 0.218 [0.164, 0.275] | 0.597 [0.481, 0.701] | 1.00 | 0.09 | 0.061 |
| most_popular | 0.626 [0.548, 0.708] | 0.434 [0.371, 0.503] | 0.883 [0.805, 0.961] | 0.53 | 0.63 | 0.107 |

Paired ΔNDCG@10 (random − most_popular): −0.216 [−0.309, −0.128] ✔

**Leave-last-out** (data `9209d3081d92`, 213 users evaluated):

| Model | Recall@10 | NDCG@10 | Coverage@10 | Gini@10 |
|---|---|---|---|---|
| random | 0.371 [0.315, 0.427] | 0.172 [0.141, 0.207] | 1.00 | 0.06 |
| most_popular | 0.596 [0.531, 0.662] | 0.333 [0.291, 0.377] | 0.60 | 0.59 |

Paired ΔNDCG@10: −0.161 [−0.218, −0.105] ✔

## Conclusion

All three checks pass, so the harness behaves as expected (Random's 0.330 [0.253, 0.411] contains
the exact chance value 0.357). Things learned along the way:

- **Small catalogs make @10 coarse.** With 30 recommendable wines, k=10 is a
  third or more of each user's remaining pool, so even random scores ~0.36 recall. Watch @5 and
  NDCG (which rewards ranking hits *early*) as the main signals until the
  Napa catalog grows.
- **With one held-out item per user, recall@k equals hit rate@k** (see the
  leave-last-out table). That's expected, not a bug.
- **The two splits answer different questions.** The temporal split
  evaluates fewer users (77) but is the realistic "train until today, predict
  next week" setup; leave-last-out evaluates more users (213) with a small
  leak from other users' future data. Prefer temporal once real data exists.
- **The popularity cost is visible.** MostPopular uses only ~55% of the
  catalog with very uneven exposure (Gini ~0.6). A content model should
  beat MostPopular's accuracy *without* that concentration. That's the bar
  for the next experiment.

**Next:** the tag-similarity content model, evaluated against these baselines.

> **Note (2026-10-23):** these Random numbers are from `random@843f5ee4` (v1), whose
> scores depended on the candidate pool. v1 was replaced by `random@94f5bc47` (v2),
> which hashes (seed, user, item). Model ids now include an algorithm version so
> a changed algorithm can never reuse an old id.
