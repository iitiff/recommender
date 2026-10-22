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

Those live in the **private site repo**, which installs this package
(`uv pip install -e ../recommender`) and supplies the catalog directory,
configs, branding, deploy files, and real experiment write-ups.

If a change needs niche-specific knowledge, make the engine configurable and
put the niche specifics in the private repo.

## Layout

```
src/recsys/
  core/       domain types (Item, Offer, UserContext, ScoredItem, Explanation), IDs
  catalog/    catalog.yaml spec + CSV loader with full validation
  events/     event schema (pydantic) + EventLogger (the one place events are written)
  store/      SQLAlchemy Core tables, engine setup, repo functions (the only SQL)
  models/     Recommender base class + models (baselines, content-based, ...)
  cli.py      `recsys` command
examples/demo/  fictional catalog + generator
docs/           DESIGN.md, ROADMAP.md, experiments/
tests/
```

## Commands

```bash
uv sync                                   # install
uv run pytest                             # tests
uv run ruff check . && uv run ruff format --check .
uv run recsys validate-catalog examples/demo
uv run recsys load-catalog examples/demo  # syncs into RECSYS_DATABASE_URL
```

## Conventions

- **Models are pure.** They get `TrainingData` / `UserContext` and return
  `ScoredItem`s. No DB, no HTTP. The same code runs in tests, offline eval, and
  serving.
- **New models subclass `Recommender`** and implement `_score_user` (and
  `_score_similar`, `explain` where meaningful). Add them to `MODELS` in
  `tests/test_recommender_contract.py`.
- **Hard constraints (budget, stock, shipping) are candidate filters** passed
  via `candidates=`, never baked into a model.
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
