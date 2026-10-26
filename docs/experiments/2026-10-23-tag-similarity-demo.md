# 2026-10-23: Tag similarity vs baselines, and a popularity hybrid (demo data)

**Status:** done
**Model(s):** `tag_similarity` (4 configurations) vs `most_popular@63cb5f3b`, `random@94f5bc47`
**Data:** fictional demo catalog, synthetic interactions (300 users, 5,179 events).
Temporal split `6e96e1ec7580` (77 users), leave-last-out `9209d3081d92` (213 users).

> Synthetic users. They were generated from grape, body and oak preferences
> plus a popularity prior, so they share assumptions with the content model.
> Treat this as a check that the pieces work together and as a demonstration
> of tradeoffs, **not** as evidence about real wine drinkers.

## Hypothesis

1. Pure content similarity beats MostPopular, because the synthetic users have real taste structure.
2. Curated facet weights (grape 0.45, notes 0.20, structure 0.35) beat equal weights.

**Decision metric:** NDCG@10, paired difference vs MostPopular.
**Guardrails:** coverage@10, exposure Gini@10.

## Method

```bash
uv run recsys evaluate examples/demo examples/demo/interactions.csv \
  --model-config examples/demo/models.yaml --k 5 --k 10            # temporal
uv run recsys evaluate ... --split last                             # leave-last-out
```

Configurations (from `examples/demo/models.yaml`):

| id | facet weights | popularity_weight |
|---|---|---|
| `2b051074` | equal across all facets (default) | 0 |
| `a2c96643` | curated: varietal 0.45, note 0.20, numeric 0.35 | 0 |
| `d45e3714` | curated | 0.3 |
| `a8c41196` | curated | 0.6 |

## Results

**Temporal** (ΔNDCG@10 vs MostPopular, paired 95% CI):

| Model | NDCG@10 | ΔNDCG@10 | Coverage@10 | Gini@10 |
|---|---|---|---|---|
| random | 0.248 | −0.186 [−0.264, −0.108] ✔ | 1.00 | 0.07 |
| most_popular | 0.434 | | 0.53 | 0.63 |
| tag, equal weights | 0.364 | −0.070 [−0.173, +0.028] | 1.00 | 0.13 |
| tag, curated | 0.334 | −0.100 [−0.192, −0.004] ✔ | 1.00 | 0.16 |
| tag, curated + 0.3 pop | 0.390 | −0.044 [−0.118, +0.037] | 1.00 | 0.27 |
| tag, curated + 0.6 pop | 0.430 | −0.004 [−0.070, +0.065] | 1.00 | 0.33 |

**Leave-last-out:**

| Model | NDCG@10 | ΔNDCG@10 | Coverage@10 | Gini@10 |
|---|---|---|---|---|
| random | 0.161 | −0.172 [−0.224, −0.119] ✔ | 1.00 | 0.07 |
| most_popular | 0.333 | | 0.60 | 0.59 |
| tag, equal weights | 0.341 | +0.008 [−0.051, +0.067] | 1.00 | 0.11 |
| tag, curated | 0.307 | −0.027 [−0.085, +0.029] | 1.00 | 0.10 |
| tag, curated + 0.3 pop | 0.372 | +0.039 [−0.012, +0.089] | 1.00 | 0.17 |
| tag, curated + 0.6 pop | 0.383 | +0.049 [+0.001, +0.098] ✔ | 1.00 | 0.24 |

## Conclusion

Both hypotheses were **wrong**, and that's the useful part.

1. **Pure content didn't beat popularity.** The synthetic users *discover*
   wines mostly through popularity, so their held-out likes skew popular, and
   offline evaluation rewards popularity. This is **popularity bias in logged
   data**: offline metrics reflect what users were exposed to as much as what
   they'd like. Real logs will have the same bias, driven by whatever the site
   showed them.
2. **Curated weights didn't beat equal weights.** With 30 wines and 77–213
   users, the difference is inside the noise. Hand-tuned weights are a
   hypothesis, not a result.
3. **A content + popularity hybrid gets the best of both:**
   - **Accuracy:** it matches MostPopular (temporal) or beats it (leave-last-out, ✔).
   - **Coverage:** it recommends 100% of the catalog instead of 53–60%.
   - **Concentration:** exposure Gini is roughly half of MostPopular's.

   That's the accuracy-vs-diversity tradeoff in one table.

**Decision:** default the site's model to the hybrid (curated weights,
`popularity_weight` around 0.3–0.6) as a starting point, then re-tune on the
tasting panel and on real logs. Re-run the curated-vs-equal comparison once
there's more data.

**Next:** the Rec Lab, to inspect *why* the hybrid ranks what it ranks, persona by persona.
