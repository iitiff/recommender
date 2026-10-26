# YYYY-MM-DD: <short name>

**Status:** planned | running | done
**Model(s):** `<model_id>` vs `<baseline model_id>`
**Data:** <dataset / snapshot hash, date range, # users, # items, # interactions>

## Hypothesis

<What do we expect, and why? e.g. "Adding numeric-attribute distance to tag
overlap improves NDCG@10, because tags alone treat body 4 and body 5 as
unrelated.">

**Decision metric (chosen before running):** <e.g. NDCG@10 on temporal split>
**Guardrails:** <e.g. catalog coverage must not drop below X; popularity bias not worse>

## Method

<Split, models and params, anything non-default. Enough to reproduce.>

## Results

| Model | Recall@10 | NDCG@10 | Coverage | Avg popularity | Notes |
|---|---|---|---|---|---|
| random | | | | | |
| most_popular | | | | | |
| candidate | | | | | |

<Confidence intervals; plots if useful.>

## Conclusion

<Ship / don't ship / follow-up. What did we learn, including what surprised us?>
