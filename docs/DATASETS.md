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

## Used for analysis only (no adapter yet)

Lifecycle comparison in `docs/experiments/2026-10-24-lifecycle-across-datasets.md`.

| Dataset | License | Source used |
|---|---|---|
| **Olist** Brazilian marketplace | CC BY-NC-SA 4.0 | Kaggle `olistbr/brazilian-ecommerce` (HF mirror `bulutttt/olist-raw-data`) |
| **REES46** multi-category store, Oct–Nov 2019 | "© Original Authors" (aggregate statistics only) | Kaggle `mkechinov/...` (HF mirror `kevykibbz/...oct-nov_2019`, 14.8 GB, streamed) |
| **H&M** Personalized Fashion | Kaggle competition rules (aggregate statistics only) | HF mirror `dinhlnd1610/HM-Personalized-Fashion-Recommendations` |
| **Online Retail II** (UK gift wholesaler, B2B) | UCI repository (CC BY 4.0, unverified) | HF mirror `attik/Online-Retail-II-UCI` |

## Candidates (not yet adapted)

| Dataset | What it adds | Notes |
|---|---|---|
| **Hillstrom e-mail** (MineThatData) | 64k customers **randomly** assigned to men's email / women's email / no email, with visit, conversion, spend | Real randomized holdout: the classic dataset for learning **uplift** (incremental effect of a message) |
| **Criteo Uplift** | ~13M users, randomized treatment/control, visit and conversion | Uplift at scale |
| **OTTO** (otto-de/recsys-dataset) | ~12M sessions of clicks, carts, orders | Session-based recs at scale; well-known benchmark |
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
- **pandas + URLs:** `pd.read_csv(url)` downloads the whole file into memory
  before parsing. For big files pass an open stream (`urllib.request.urlopen`)
  with `chunksize`; that's how REES46 (14.8 GB) was processed in flat memory.
- **"Order" definitions:** sessions and marketplace order ids both create fake
  "repeat purchases" minutes apart. Use customer-day for lifecycle work.
- **Censoring:** use fixed horizons and only count customers with enough data
  left (right-censoring); remember that a dataset's first order isn't the
  customer's first ever (left-censoring).
- **Mixed ISO formats:** timestamps with and without fractional seconds in one
  column need `format="ISO8601"` when parsing.
