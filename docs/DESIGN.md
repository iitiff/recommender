# Design

The decisions behind the engine and why they were made. Update this file when
a decision changes.

## 1. One engine, many catalogs

The engine knows nothing about any niche. A catalog directory supplies:

| File | Purpose |
|---|---|
| `catalog.yaml` | Maps CSV columns onto generic fields: id, title, description, category, **tag facets**, **numeric attributes**, and **bins** |
| `items.csv` | One row per item, hand-curated |
| `offers.csv` | Optional. Where to buy: retailer, variant (e.g. vintage), price, url, stock, shipping regions |

**Why:** a different niche, or an e-commerce store's product fields, is a
different spec, not different code. That is the core of multi-tenancy.

**Tags vs numeric attributes.** Categorical facts (grape, region, flavor
notes) become `Tag(facet, value)`. Scales (body 1-5) stay numeric so models can
use distance ("4 is close to 5"), and are also binned into tags (`body:full`)
so tag-overlap models and explanations can use them.

**Items vs offers.** Item identity (`item_id`) is stable forever, because
events reference it. Price, stock and vintage change weekly and differ by
retailer, so they live in `offers`, which a refresh replaces wholesale. The
e-commerce analogue is product vs variant/inventory.

**Rateable vs recommendable.** The catalog can be wider than what we
recommend. `recommendable_when: {facet: [values]}` in `catalog.yaml` marks
which items may be recommended; everything else is still loaded, rated, used in
the quiz, learned from, and usable as a "because you liked X" anchor, but never
enters a candidate pool. Example: quiz on well-known wines from a whole state
(people recognise more of them, so the quiz extracts more taste signal), but
recommend only within one region. This is a catalog property, not a request
constraint, so the base class enforces it for every model; per-request
constraints still go through `candidates=`. E-commerce analogue: products a
shopper bought or viewed that are no longer sold still describe their taste.

Cross-catalog transfer works through **shared style attributes** (varietal,
body, tannin, oak, flavor notes), not through region tags: a liked full,
oaky Cabernet from one region points to full, oaky Cabernets in another. That
is why the controlled vocabulary for notes and the 1-5 scales matter.

**Refresh is idempotent.** Items are upserted by `item_id` and rewritten only
when their content hash changes (so later, only changed items get re-embedded).
Items that disappear from the CSV are **soft-deleted** (`is_active=false`),
never removed, because historical events point at them.

## 2. Data model

```
tenants        tenant_id, name
items          (tenant_id, item_id), title, description, category, attributes, image_url,
               is_active, recommendable, content_hash, created_at, updated_at
item_tags      (tenant_id, item_id, facet, value)
offers         (tenant_id, offer_id), item_id, retailer, variant, price, currency, url,
               in_stock, ships_to, updated_at
users          (tenant_id, user_id), email?, acquisition (first-touch utm/referrer), created_at
quiz_responses id, tenant_id, user_id, quiz_version, answers, created_at
feedback       (tenant_id, user_id, item_id), rating (+1/-1/null), saved, hidden, updated_at
user_profiles  (tenant_id, user_id), tag_affinities, derived_from, updated_at
events         see below
```

- **`tenant_id` everywhere.** It costs one column now and avoids the most
  painful migration later.
- **Raw facts vs derived state.** `quiz_responses`, `feedback` and `events`
  are truth; `user_profiles` is a rebuildable cache. When the quiz-to-tag
  mapping changes, profiles are recomputed from the raw answers.
- **SQLAlchemy Core + JSON→JSONB variant.** Moving from SQLite to Postgres is a
  connection-string change. `create_all` for now; Alembic before the first
  production deploy.
- **SQLite in WAL mode** so the web app can read while events are written.

## 3. Event schema

One envelope for all events (typed columns), plus a `properties` payload
validated per type (JSON). Columns make common fields cheap to query; JSON
payloads mean a new event type never needs a migration.

| Field | Why |
|---|---|
| `event_id` (UUIDv7) | Dedupe; sorts by time |
| `schema_version` | Lets the schema evolve; changes are additive |
| `event_type`, `tenant_id`, `occurred_at` (UTC, enforced) | Basics |
| `user_id`, `session_id` | Who; sessions enable session-based recs later |
| **`request_id`** | Joins a served slate → impressions → clicks → affiliate click → conversion. This is how an outcome is attributed to the model that caused it. |
| `surface` | Placement (`quiz_results`, `home_recs`, `similar_items`); CTR varies hugely by placement |
| `item_id`, `position` | Position is needed to correct position bias in learning-to-rank |
| **`model_id`** | `name@hash(params)`; makes model comparison and A/B analysis possible |
| `experiment` | `{experiment: variant}` assignments |
| `context` | utm_source/medium/campaign, referrer, user agent: which channel works |
| `properties` | Per-type payload |

Event types: `page_view`, `quiz_completed`, `slate_served` (the full ranked
list, with per-item score, reason and **propensity**), `impression`,
`item_click`, `rating`, `save`, `not_interested`, `affiliate_click` (logged by
a server-side redirect, so ad-blockers can't drop it), `affiliate_conversion`
(imported from network reports, joined on `click_id`).

**Why log the whole slate?** Collaborative filtering and learning-to-rank need
negatives: items that were shown and *not* clicked. **Why propensity now?**
Bandits and off-policy evaluation need the probability the policy showed each
item, and it can't be reconstructed afterwards.

The funnel maps one-to-one onto e-commerce: slate served = widget render,
item click = product view, affiliate click = add to cart, conversion = order.

## 4. Recommender interface

```python
class Recommender(ABC):
    name: ClassVar[str]
    params -> dict                 # hyperparameters; define model_id
    model_id -> "name@hash"
    fit(TrainingData) -> Self
    recommend(UserContext, k, *, candidates=None, explain=True) -> list[ScoredItem]
    similar_items(item_id, k, *, candidates=None) -> list[ScoredItem]
    explain(UserContext, item_id) -> Explanation | None
    save(path) / load(path)
```

- **Template method.** The base class owns candidate filtering, exclusion of
  rated/hidden items, deterministic top-k and explanations; subclasses
  implement only scoring. Models are compared under identical rules.
- **`UserContext`, not `user_id`.** Models never touch the DB, so the same code
  runs in unit tests, offline eval, and serving, and works for anonymous
  sessions.
- **`similar_items` is first-class.** "You might also like" is item-to-item and
  the most common e-commerce recommendation surface.
- **Hard constraints are candidate filters.** Budget, stock and shipping
  region remove items *before* ranking; preferences only reorder. Every model
  obeys constraints identically.
- **Structured explanations.** `Explanation(kind, anchor_item_ids,
  shared_tags)`; templates render text. The `kind` is logged per slate item so
  we can measure which explanation types earn clicks.

## 5. Evaluation (how we know it works)

Evidence in stages, from cheapest to most trustworthy:

1. **Persona tests:** hand-written expectations ("light, high-acid → no
   full-bodied oaky wines in the top 10"). These catch broken models; they
   don't measure quality.
2. **Tasting panel:** friends take the quiz, the model ranks a fixed set of
   bottles per person before a blind tasting, and we compare the predicted
   order to their ratings (NDCG, rank correlation) against random and popular
   orderings, with bootstrap confidence intervals.
3. **Offline replay on logged data:** temporal split; recall@k, NDCG@k,
   catalog coverage, popularity bias, always against Random and MostPopular.
   Full ranking over the whole catalog, never sampled negatives. Only
   recommendable items count as possible hits: a held-out like on a quiz-only
   item can't be recommended, so it's excluded from targets (but still used as
   input). Logged data
   is biased toward the model that produced it, so offline results filter
   ideas; they don't decide.
4. **Online:** per `model_id` and surface: recommendation CTR, share of 👍 on
   recommended items, saves, affiliate clicks per visitor, return visits;
   guardrails on diversity, coverage, latency.
5. **Experiments:** interleaving while traffic is small (far more sensitive
   than A/B splits), A/B tests and a permanent popularity holdout later.

Synthetic users are only for catching bugs, never for comparing models: they
are generated from the same assumptions the model makes.
