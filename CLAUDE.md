# CLAUDE.md

## What this is

`recsys` is a small, swappable recommender engine: catalog loading, event
logging, recommender models behind one interface, and an offline evaluation
harness. It powers a niche recommendation website (taste quiz → personalized
recs → ratings → better recs) and is the learning ground for a future
multi-tenant product.

The owner is a personalization product manager building ML-engineering skills.
**Explain the "why" behind technical choices**, not just the code: tradeoffs,
alternatives considered, and what would change at larger scale.

## Public / private split (important)

This repository is **public**. It contains only the generic engine plus a
fictional demo catalog (`examples/demo/`).

Never commit to this repo:
- real catalog data, tasting notes, quiz mappings, tuned weights or configs
- experiment results on real user data
- business plans, revenue numbers, affiliate details
- secrets of any kind (they live in `.env` / the host's secret store, never in git)
- model artifacts or databases (`artifacts/`, `data/` are gitignored; they are rebuilt)
- public dataset files, raw or prepared (`data/`; licenses in docs/DATASETS.md)

Those live in the **private site repo**, which installs this package
(`uv pip install -e ../recommender`) and supplies the catalog directory,
configs, branding, deploy files, and real experiment write-ups.

If a change needs niche-specific knowledge, make the engine configurable and
put the niche specifics in the private repo.

## Current focus

The earlier niche-site work is **parked** (its private repo has a resume note).
Current work is the e-commerce track in docs/ROADMAP.md: RetailRocket replay, session-based
recs, then lifecycle messaging (docs/LIFECYCLE.md): move customers from first
order to loyal, measured by uplift and second-order rate.

## Layout

```
src/recsys/
  core/       domain types (Item, Offer, UserContext, ScoredItem, Explanation), IDs
  catalog/    catalog.yaml spec + CSV loader with full validation
  events/     event schema (pydantic) + EventLogger (the one place events are written)
  store/      SQLAlchemy Core tables, engine setup, repo functions (the only SQL)
  models/     Recommender base class + models (baselines, content-based, session and journey sources)
  profile/    taste quiz spec → profile (cold start)
  candidates.py  hard filters (budget, stock, shipping) with funnel counts; stock over time
  sessions.py    split event streams into visits (30-min inactivity gap)
  lifecycle.py   lifecycle stage of a shopper at a moment, derived from events
  shopper.py     shopper profile at request time (docs/SHOPPER-PROFILE.md): ShopperState, ItemFacts
  datasets/   public dataset adapters (raw → engine layout) + re-engagement profile
  eval/       splits, metrics, bootstrap CIs, harness, reports
  lab/        Rec Lab: FastAPI + Jinja2 + HTMX explainability tool (never writes the DB)
  cli.py      `recsys` command
examples/demo/  fictional catalog + generator
docs/           DESIGN.md, ROADMAP.md, LIFECYCLE.md, SHOPPER-PROFILE.md, DATASETS.md,
                VALIDATION-PLAN.md (what to replicate and test on a retailer's data), experiments/
tests/
```

## Commands

```bash
uv sync                                   # install
uv run pytest                             # tests
uv run ruff check . && uv run ruff format --check .
uv run recsys validate-catalog examples/demo
uv run recsys load-catalog examples/demo  # syncs into RECSYS_DATABASE_URL
uv run recsys evaluate examples/demo examples/demo/interactions.csv  # offline eval
uv run recsys evaluate examples/demo examples/demo/interactions.csv --model-config examples/demo/models.yaml
uv run recsys lab examples/demo --interactions examples/demo/interactions.csv  # http://127.0.0.1:8000
uv sync --extra datasets && uv run recsys dataset prepare retailrocket RAW_DIR data/retailrocket
uv run recsys dataset profile data/retailrocket
uv run recsys dataset prepare otto RAW_DIR data/otto --sample-every 64  # CC BY 4.0
uv run recsys evaluate data/retailrocket/catalog data/retailrocket/interactions.csv --split next \
  --test-fraction 0.1 --models most_popular,trending,recently_viewed,covisitation --k 5 --k 20 \
  --group-facet department --availability data/retailrocket/availability.csv  # session eval
uv run recsys evaluate data/retailrocket/catalog data/retailrocket/interactions.csv --split next \
  --test-fraction 0.1 --k 5 --k 20 --group-facet department \
  --availability data/retailrocket/availability.csv \
  --model-config docs/experiments/configs/2026-10-26-journey-blend.yaml  # journey blend
```

## Conventions

- **Models are pure.** They get `TrainingData` / `UserContext` and return
  `ScoredItem`s. No DB, no HTTP. The same code runs in tests, offline eval, and
  serving.
- **New models subclass `Recommender`** and implement `_score_user` (and
  `_score_similar`, `explain`, `score_breakdown` where meaningful; breakdown
  components must sum to the real score). Add them to `MODELS` in
  `tests/test_recommender_contract.py`. Large catalogs: override `_retrieve`
  instead of scoring every item; sources that only know some items set
  `exhaustive = False`.
- **Hard constraints (budget, stock, shipping) are candidate filters**
  (`recsys.candidates`) passed via `candidates=`, never baked into a model.
- **Price is context, not taste.** Taste/content similarity never uses price
  (tag similarity ignores the `price` attribute). Price may be a *soft
  ranking signal* at three levels:
  - distance to the price of the item being viewed (this visit);
  - the shopper's price band in that product type;
  - their general price level, pooled across their history and partially
    pooled toward the type and store averages when evidence is thin.

  Evidence: docs/experiments/2026-10-24-shopper-price-level.md. Price never
  filters beyond the shopper's own budget, never labels shoppers (no income
  inference), and never changes the price a shopper is charged.
- **Hyperparameters live in config** (`load_model_config`), not code; niche
  weights belong in the private repo.
- **Rateable ≠ recommendable.** Items with `recommendable=False` (set by
  `recommendable_when` in `catalog.yaml`) are learned from and can anchor
  explanations but never appear in results; the base class enforces this. Eval
  targets only count recommendable items.
- **All events go through `EventLogger`** and must validate against
  `events/schema.py`. Schema changes bump `SCHEMA_VERSION` and are additive.
- **Every model change gets measured** with the eval harness against the
  Random and MostPopular baselines, and written up in `docs/experiments/`
  (public datasets / demo data) or the private repo (real user data).
- **Every table and event carries `tenant_id`.**
- Domain types are frozen dataclasses; boundary validation uses pydantic.
- Type hints everywhere; ruff for lint/format; match the surrounding style.

## How the owner likes to work

- Lean and shippable: one person, ship fast, avoid premature infrastructure.
- Propose before building anything large; small, reviewable PRs.
- Teach as you go: short "why" notes in PR descriptions and docs, plus pointers
  to the concept's name (so it can be looked up) when introducing something new.
- Be honest about uncertainty and about what metrics can and cannot show.
