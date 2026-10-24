# Datasets

Public datasets used for learning and offline evaluation. Raw and prepared
files live under `data/` (gitignored) and are **never committed**; the license
of each source governs what you may do with it.

## Prepared (adapter in `src/recsys/datasets/`)

| Dataset | License | What it has | Good for |
|---|---|---|---|
| **RetailRocket** (`retailrocket`) | CC BY-NC-SA 4.0 | 2.76M events (view / add-to-cart / purchase), 1.4M visitors, 235k items, 137 days (May–Sep 2015); item category and **availability over time**; category tree. Names and most properties are hashed. | Session-based recs, catalog churn (stock-outs), re-engagement triggers |

```bash
uv sync --extra datasets
uv run recsys dataset download retailrocket data/raw/retailrocket   # ≈1 GB
uv run recsys dataset prepare retailrocket data/raw/retailrocket data/retailrocket
uv run recsys dataset profile data/retailrocket
```

Prepared layout (the engine's standard one): `catalog/` (`catalog.yaml` +
`items.csv`), `interactions.csv` (`user_id,item_id,kind,value,occurred_at,session_id`),
`availability.csv` (`item_id,valid_from,available`, one row per stock change).

## Candidates (not yet adapted)

| Dataset | What it adds | Notes |
|---|---|---|
| **Hillstrom e-mail** (MineThatData) | 64k customers **randomly** assigned to men's email / women's email / no email, with visit, conversion, spend | Real randomized holdout: the classic dataset for learning **uplift** (incremental effect of a message) |
| **Criteo Uplift** | ~13M users, randomized treatment/control, visit and conversion | Uplift at scale |
| **OTTO** (otto-de/recsys-dataset) | ~12M sessions of clicks, carts, orders | Session-based recs at scale; well-known benchmark |
| **REES46** multi-category store | Views, carts, purchases **with prices** | Revenue metrics |
| **H&M Personalized Fashion** | Transactions + article attributes + images | Content features, seasonality |
| **Instacart** | Grocery orders and reorders | Repeat purchase, basket completion |
| **Amazon Reviews 2023** (McAuley Lab) | Reviews + rich product metadata | Product text for embeddings |

Check each license at the original source before downloading; Hugging Face
re-uploads often omit it. Most are non-commercial or research-only: fine for
learning, not for the commercial product.

## Wine (earlier exploration)

Wine Enthusiast reviews
(critics, not users), WineSensed (CC BY-NC-ND; includes a human taste-similarity
experiment), Vivino scrapes (aggregate ratings only), X-Wines (per-user ratings;
hosted outside Hugging Face, license unchecked).

## Gotchas found so far

- **Datetime resolution:** modern pandas may store datetimes as s/ms/µs, so
  `astype("int64")` doesn't give nanoseconds. Use Timedelta arithmetic
  (`(t - epoch) / pd.Timedelta(seconds=1)`). This made every return gap look
  1000× shorter until fixed.
- **Mixed ISO formats:** timestamps with and without fractional seconds in one
  column need `format="ISO8601"` when parsing.
