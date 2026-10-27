# Datasets

Public datasets used for learning and offline evaluation. Raw and prepared
files live under `data/` (gitignored) and are **never committed**; the license
of each source governs what you may do with it.

## Prepared (adapter in `src/recsys/datasets/`)

| Dataset | License | What it has | Good for |
|---|---|---|---|
| **OTTO** (`otto`) | **CC BY 4.0** (commercial use allowed) | 4 weeks (Aug 2022) of clicks / carts / orders from a large German online retailer: ~12.9M shoppers, ~1.8M articles, ~216M events. Articles are anonymized ids (no category, price or brand). `prepare` keeps 1 in `--sample-every` shoppers (default 64: ~200k shoppers, 3.4M events) with complete histories | Session-based recs and lifecycle work on a commercially usable dataset |
| **RetailRocket** (`retailrocket`) | CC BY-NC-SA 4.0 | 2.76M events (view / add-to-cart / purchase), 1.4M visitors, 235k items, 137 days (May–Sep 2015); item category and **availability over time**; category tree. Names and most properties are hashed. | Session-based recs, catalog churn (stock-outs), re-engagement triggers |

```bash
uv sync --extra datasets
uv run recsys dataset download retailrocket data/raw/retailrocket   # ≈1 GB
uv run recsys dataset prepare retailrocket data/raw/retailrocket data/retailrocket
uv run recsys dataset profile data/retailrocket

uv run recsys dataset download otto data/raw/otto                   # ≈2 GB zip
uv run recsys dataset prepare otto data/raw/otto data/otto --sample-every 64
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
| **Instacart** online grocery, 2017 (3.4M orders, order day-of-week and hour; weekday labels undocumented) | Released by Instacart for non-commercial use (aggregate statistics only) | HF mirror `attik/Instacart-Market-Basket-Analysis` |
| **OTTO** German online retailer, 4 weeks of Aug 2022: clicks, carts, orders on anonymized article ids (no categories or prices); an OTTO "session" is one shopper's whole period | **CC BY 4.0** (commercial use allowed; confirmed in the repo README) | HF mirror `hazemessam/otto-recsys` (the official Kaggle zip), streamed with a 25% shopper sample (`docs/experiments/scripts/stream_otto_events.py`); now also has an adapter (above) |
| **X5 RetailHero** uplift (Russian grocery chain): purchase lines Nov 2018–Mar 2019 before an SMS campaign; 200k clients **randomly** sent the SMS or not, with whether they bought | Not stated; competition data from X5 Retail Group (ods.ai). Aggregate statistics only; age and gender columns not used | HF mirror `pytorch-lifestream/retailhero-uplift` (`docs/experiments/scripts/x5_uplift.py`) |
| **Amazon Reviews 2023** (McAuley Lab, UCSD): Electronics, Home & Kitchen, Grocery | Not stated on the dataset card; earlier Amazon releases were "non-commercial research only" (aggregate statistics only) | HF `McAuley-Lab/Amazon-Reviews-2023`: `benchmark/0core/rating_only/*.csv` + `raw/meta_categories/meta_*.jsonl`, streamed (`docs/experiments/scripts/stream_amazon.py`) |

## Candidates (not yet adapted)

| Dataset | What it adds | Notes |
|---|---|---|
| **Hillstrom e-mail** (MineThatData) | 64k customers **randomly** assigned to men's email / women's email / no email, with visit, conversion, spend | Real randomized holdout: the classic dataset for learning **uplift** (incremental effect of a message) |
| **Criteo Uplift** | ~13M users, randomized treatment/control, visit and conversion | Uplift at scale |
| **OTTO** (otto-de/recsys-dataset) | ~12M sessions of clicks, carts, orders | Session-based recs at scale; well-known benchmark |
| **Amazon Reviews 2023**, more categories and review text | Review text (what shoppers value), more categories | Product and review text for embeddings and "what they value" |
| **Amazon-M2** (KDD Cup 2023) | Browsing sessions in 6 countries with title, brand, price; **Apache 2.0** per the NeurIPS 2023 paper | Session recs with a clean license; cross-country comparison. No user ids across sessions |

**More candidates, by the question they answer** (access checked from a
cloud session on 2026-10-27: "HF" means a Hugging Face mirror downloads
directly; Kaggle needs `KAGGLE_USERNAME` / `KAGGLE_KEY` as environment
secrets; others need a free account or an application):

| Dataset | What it adds | Access | License (verify at source) |
|---|---|---|---|
| **dunnhumby The Complete Journey** | 2,500 grocery households over 2 years: transactions, **campaigns, coupons and redemptions**, household demographics (campaigns not randomized) | Free registration | dunnhumby terms, non-commercial |
| **Tmall Repeat Buyers** (Alibaba Tianchi) | New buyers from a Singles' Day sale, labeled by whether they bought again from the merchant within 6 months: **first → second order** | Tianchi account | Research use |
| **OTTO** (otto-de/recsys-dataset; now used, see above) | ~12M sessions of clicks, **carts and orders** over 4 weeks | HF mirror `hazemessam/otto-recsys` | **CC BY 4.0 (commercial use allowed)** |
| **Taobao UserBehavior** (Tianchi) | ~100M events from 1M users over 9 days: views, **carts, favorites**, buys | HF mirror `TGB-Seq/Taobao`; original on Tianchi | Research use |
| **REES46 cosmetics shop** (Oct 2019–Feb 2020) | Five months with **remove-from-cart**, low-priced repeat purchases | Kaggle | "© Original Authors" |
| **YooChoose** (RecSys 2015), **Diginetica** (CIKM Cup 2016) | Classic session benchmarks (clicks and buys; Diginetica adds search queries) | Kaggle / challenge sites | Research use |
| **Dressipi** (RecSys Challenge 2022) | Fashion sessions with purchases and item attributes | Registration | Non-commercial |
| **M5** (Walmart, 2011–16) | Daily unit sales per item and store in 3 states, with **prices, events and SNAP days**: category timing and seasonality; no customers | HF mirror `denephew/M5_Forecasting`; Kaggle | Kaggle competition rules |
| **Amazon ESCI** (Shopping Queries) | Search queries with relevance-labeled products | HF `tasksource/esci` | Apache 2.0 |
| **JD.com** (MSOM 2020 data challenge) | Users with demographics, clicks, orders, **delivery times and promotions** | Application to the challenge | Research use |
| **Hillstrom** (already a candidate) | — | Direct download from minethatdata.com | No formal license |
| **Criteo Uplift** (already a candidate) | — | HF `criteo/criteo-uplift` | CC BY-NC-SA 4.0 |

**For the commercial product:** OTTO (CC BY 4.0) and Amazon-M2 (Apache 2.0)
are the clean bases for defaults that ship; the rest are for learning.

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
