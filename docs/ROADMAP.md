# Roadmap

Each phase has a learning goal, what gets built, and how we decide it worked.
A phase ships only when its model beats the previous best on the eval harness
(and on online metrics once there is traffic), with a write-up in
`docs/experiments/`.

## Phase 1: Content-based recommendations and the foundations ← now

**Learn:** cold start, content similarity, text embeddings, offline evaluation,
event design.

- [x] Engine scaffold, domain types, `Recommender` interface, contract tests
- [x] Catalog spec + loader + idempotent refresh (items vs offers, soft delete)
- [x] Rateable vs recommendable items (`recommendable_when`): quiz wide, recommend narrow
- [x] Event schema + logger; DB schema with `tenant_id` throughout
- [x] Baselines: Random, MostPopular
- [x] Evaluation harness: temporal and leave-last-out splits, recall/NDCG/hit
      rate@k, coverage, exposure Gini, popularity bias (ARP), bootstrap CIs with
      paired comparisons, markdown/JSON reports, `recsys evaluate`
- [ ] Feed the harness from the site's own DB (feedback + events), not only CSV
- [x] Taste quiz spec → profile (rated items, style affinities, budget); later feedback overrides quiz
- [x] Tag-similarity model (per-facet Jaccard + numeric closeness, max over liked items,
      dislike penalty, quiz affinities, popularity hybrid, cold-start fallback, score breakdowns)
- [x] Candidate filters: budget, in stock, ships to region (with funnel counts)
- [x] Rec Lab: persona/quiz → model → user experience, with an explainability overlay
- [ ] Rec Lab v2: catalog map (scatter of items by style/embedding), "what changed" diff after feedback, live weight sliders
- [ ] Web app: FastAPI + Jinja2 + HTMX; quiz, recs with "because you liked X",
      👍/👎, save, not interested; `/go/{item}` affiliate redirect; impression beacon
- [ ] Text-embedding model (local sentence-transformers) and tag/embedding hybrid
- [ ] Tasting-panel protocol + first experiment write-up
- [ ] Deploy: Alembic migrations, small VPS or PaaS, SQLite + Litestream backups

**Done when:** the content model beats MostPopular on the tasting panel and on
offline eval, and the site is live, logging clean events.

## E-commerce track: replay a real store, then re-engagement ← now

The niche-site work is parked (too static: small curated catalog, slow
explicit feedback). This track uses public e-commerce logs replayed day by day, the
world the Shopify app lives in. Surface #1: "you might also like" on a product
page; then push/email re-engagement.

**Metrics.** Offline primary: Recall@20 / MRR@20 on the next add-to-cart or
purchase in the session. Guardrails: share of recommended items in stock,
coverage, popularity bias, latency. Online (real store): incremental revenue
per visitor vs a randomized holdout. For messages: incremental conversions per
message sent vs holdout (uplift), with unsubscribe and frequency guardrails.

1. - [x] Dataset adapters + RetailRocket (sessions, catalog, availability over
         time) + re-engagement profile (`docs/experiments/2026-10-24-*`)
2. Session-based, journey-aware recommendations (docs/LIFECYCLE.md: "one
   recommender, many journeys")
   - [x] 2a. Candidate sources that each do one job (recently viewed,
         co-visitation, trending), a retrieval path for big catalogs, stock as of
         request time, next-add-to-cart eval split by lifecycle stage and by what
         the shopper did next, category-novelty metric
         (`docs/experiments/2026-10-24-session-sources-retailrocket.md`)
   - [x] 2b part 1. `ShopperState` (recsys.shopper), abandoner stage, re-buy,
         cart-substitute and adjacent-category sources
         (`docs/experiments/2026-10-24-journey-sources-retailrocket.md`)
   - [x] 2b part 2. Journey blend: per-journey weights (rules), weighted
         reciprocal rank fusion, adjacent-category exploration slots, and
         "return" / "discovery" / "cold start" headline metrics
         (`docs/experiments/2026-10-26-journey-blend-retailrocket.md`)
   - [ ] 2c. Learned per-journey weights, then a learned ranker over
         `ShopperState` features; replicate on a REES46 sample (prices, brands)
         (OTTO: the blend's structure transfers but its weights don't; tuned
         per store on a validation window, docs/experiments/2026-10-27-journey-blend-otto.md)
         Spec for the shopper-state features: docs/SHOPPER-PROFILE.md
3. - [ ] Day-by-day replay: nightly retrain → eval gate → model registry
4. - [ ] Lifecycle messaging (docs/LIFECYCLE.md; validations and experiments to run on a
         retailer's own data: docs/VALIDATION-PLAN.md): lifecycle stages; triggers
         (abandoned cart/browse, post-purchase, back-in-stock, win-back); two message
         jobs (next transaction vs category expansion); next-best-action from
         rules → propensity → uplift (Hillstrom / Criteo / X5) → contextual bandit;
         `message_*` events. North star: second-order rate, 90-day retention.
5. - [ ] Rec Lab: live session view and an "Inbox" view (what message, why)

## Phase 2: Collaborative filtering

**Learn:** matrix factorization for implicit feedback (ALS, BPR), negative
sampling, hybrid models for cold start.

- Train/evaluate on a public wine-ratings dataset first (no traffic needed),
  then on our own logs as they grow
- Implicit-feedback MF (e.g. `implicit` library) as a `Recommender`
- Hybrid: content model for new users/items, blend toward CF as data grows
- **Switch criterion:** enough users with ≥5 interactions that CF beats content
  offline with non-overlapping confidence intervals

## Phase 3: Embedding retrieval

**Learn:** two-tower / item2vec embeddings, approximate nearest neighbour
search, the retrieval → ranking split.

- Learned item and user embeddings; ANN index (FAISS or pgvector)
- Retrieval of a few hundred candidates feeding a separate ranking stage
- Matters once catalogs reach many thousands of items

## Phase 4: Learning to rank

**Learn:** feature engineering, gradient-boosted rankers (LightGBM
LambdaMART), position-bias correction using the logged `position`.

- Features: retrieval scores, tag affinities, price fit, popularity, freshness
- Train on logged slates (shown-and-clicked vs shown-and-skipped)
- Counterfactual/unbiased LTR using propensities

## Phase 5: Bandits and experimentation

**Learn:** exploration vs exploitation, A/B testing, interleaving, off-policy
evaluation.

- Experiment assignment (hash of user_id) feeding `experiment` on events
- Team-draft interleaving for low-traffic model comparisons
- A/B tests with power analysis; permanent popularity holdout to measure lift
- Thompson sampling / epsilon-greedy exploration with logged propensities;
  inverse propensity scoring (IPS) for offline policy evaluation. Dry run
  with `obp` on the Open Bandit Dataset: `docs/experiments/2026-10-27-ope-dry-run.md`

## Phase 6: Multi-tenant productization

**Learn:** serving many tenants reliably.

- One model/config per tenant; tenant-aware catalog sync from an external API
- Cold start for tenants with little data (content + popularity priors)
- Scheduled retraining, model registry, low-latency serving (precomputed
  item-to-item tables, caching)
- Per-tenant dashboards proving lift from experiments
- Product details live in the private repo

## Infrastructure track (as needed)

- Alembic migrations (before first production deploy)
- Postgres when concurrent writes or analytics outgrow SQLite
- Export events to Parquet for analysis
- CI: lint + tests on every PR
