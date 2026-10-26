"""Journey-aware blender: one recommender, many journeys (docs/SHOPPER-PROFILE.md).

The blender owns several candidate sources (recently viewed, co-visitation,
trending, re-buy, cart substitutes, adjacent category) and decides, per
request, how much each one counts, from the shopper's state:

1. **Which journey:** "cold start" (no history at all) or the lifecycle stage
   (new visitor, browser, abandoner, 1 order, 2+ orders, bulk buyer).
2. **Weights per journey** (config): e.g. abandoners lean on cart substitutes,
   repeat customers on recently viewed and re-buy, cold starts on trending.
   These are hand rules taken from the analyses. The progression is rules →
   learned weights → a learned ranker (roadmap step 2b and beyond).
3. **Weighted reciprocal rank fusion:** each source contributes
   weight / (rank_constant + rank) for every item it ranks; the sums decide
   the order. Ranks rather than scores, because sources' scores live on
   different scales (a co-visitation confidence vs a trending count).
   Concept: rank aggregation, reciprocal rank fusion (RRF).
4. **Exploration slots:** for journeys listed in `explore_stages`, every
   `explore_every`-th position is filled by the exploration source (adjacent
   category). Exploration is a visible, fixed share of the list (10% by
   default), the explore/exploit dial from LIFECYCLE.md, rather than
   something that wins or loses on score.

Every source receives the same candidates and exclusions, so the blend can
only reorder what the hard filters already allow. The blender's score is
1 / position: the list is built once to `depth` and a top-k is its prefix,
so the same item gets the same score whatever k is asked for.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Container, Sequence
from typing import Any

import numpy as np

from recsys.core.types import (
    Explanation,
    ItemId,
    ScoreBreakdown,
    ScoreComponent,
    ScoredItem,
    TrainingData,
    UserContext,
)
from recsys.models.base import Recommender
from recsys.shopper import shopper_state

COLD = "cold start"

DEFAULT_SOURCES: dict[str, dict[str, Any]] = {
    "recently_viewed": {},
    "covisitation": {},
    "trending": {},
    "rebuy": {},
    "cart_substitutes": {},
    "adjacent_category": {},
}

# Rules by journey, from the analyses (docs/experiments/2026-10-24-*):
# return carts dominate everywhere (recently viewed), abandoners comparison-shop
# (substitutes), repeat customers re-cart what they know (re-buy), and only
# trending can serve a shopper with no history. Recently viewed weighs 3: at
# weight 1 the fusion pushed the item on screen down the list (MRR 0.45 vs
# 0.52), chosen on a validation window (2026-10-26-journey-blend-retailrocket.md).
# Per-store weights belong in config.
DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    COLD: {"trending": 1.0},
    "new visitor": {
        "recently_viewed": 3.0,
        "covisitation": 1.0,
        "cart_substitutes": 0.5,
        "trending": 0.2,
    },
    "browser": {
        "recently_viewed": 3.0,
        "covisitation": 1.0,
        "cart_substitutes": 0.5,
        "trending": 0.2,
    },
    "abandoner": {"recently_viewed": 3.0, "cart_substitutes": 1.0, "covisitation": 0.8},
    "1 order": {
        "recently_viewed": 3.0,
        "covisitation": 0.8,
        "rebuy": 0.6,
        "cart_substitutes": 0.4,
    },
    "2+ orders": {"recently_viewed": 3.0, "rebuy": 0.8, "covisitation": 0.5},
    "bulk buyer": {"recently_viewed": 3.0, "rebuy": 0.8, "covisitation": 0.2},
}
DEFAULT_EXPLORE_STAGES = ("browser", "abandoner", "1 order", "2+ orders")


class JourneyBlender(Recommender):
    name = "journey_blend"
    exhaustive = False
    summary = (
        "Mixes several sources (recently viewed, viewed together, trending, buy again, "
        "cart alternatives, new categories) with weights that depend on where the shopper "
        "is in their journey."
    )
    uses = "Everything the sources use, plus the shopper's lifecycle stage"
    good_at = "Serving new visitors, abandoners and repeat customers differently from one list."
    weak_at = "Hand-set weights: a learned blend should replace them once there is logged data."
    concepts = "two-stage recommendation, reciprocal rank fusion, explore/exploit"

    def __init__(
        self,
        sources: dict[str, dict[str, Any]] | None = None,
        weights: dict[str, dict[str, float]] | None = None,
        default_weights: dict[str, float] | None = None,
        rank_constant: float = 10.0,
        depth: int = 50,
        explore_source: str | None = "adjacent_category",
        explore_every: int = 10,
        explore_stages: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        # `is None`, not `or`: an empty dict is a real choice ("no journey rules").
        self.sources = {
            k: dict(v or {}) for k, v in (DEFAULT_SOURCES if sources is None else sources).items()
        }
        self.weights = {
            k: dict(v) for k, v in (DEFAULT_WEIGHTS if weights is None else weights).items()
        }
        self.default_weights = dict(
            default_weights or {s: 1.0 for s in self.sources if s != explore_source}
        )
        self.rank_constant = rank_constant
        self.depth = depth
        self.explore_source = explore_source
        self.explore_every = explore_every
        self.explore_stages = list(
            DEFAULT_EXPLORE_STAGES if explore_stages is None else explore_stages
        )
        self._children: dict[str, Recommender] = {}

    @property
    def params(self) -> dict[str, Any]:
        return {
            "sources": self.sources,
            "weights": self.weights,
            "default_weights": self.default_weights,
            "rank_constant": self.rank_constant,
            "depth": self.depth,
            "explore_source": self.explore_source,
            "explore_every": self.explore_every,
            "explore_stages": self.explore_stages,
        }

    def _fit(self, data: TrainingData) -> None:
        from recsys.models import REGISTRY  # import here: the registry imports this module

        unknown = [s for s in self.sources if s not in REGISTRY]
        if unknown:
            raise ValueError(f"unknown sources {unknown}; available: {sorted(REGISTRY)}")
        self._children = {s: REGISTRY[s](**p).fit(data) for s, p in self.sources.items()}

    # -- the blend ------------------------------------------------------------

    def journey(self, user: UserContext) -> str:
        """The weights key for this request: cold start, or the lifecycle stage."""
        if not user.events:
            return COLD
        return str(shopper_state(user).stage)

    def _blend(
        self, user: UserContext, candidates: Container[ItemId] | None
    ) -> list[tuple[ItemId, dict[str, float], str | None]]:
        """The full list to `depth`: (item, {source: contribution}, explore source or None)."""
        self._check_fitted()
        journey = self.journey(user)
        weights = self.weights.get(journey, self.default_weights)
        fused: dict[ItemId, dict[str, float]] = defaultdict(dict)
        for source, w in weights.items():
            if w <= 0 or source not in self._children:
                continue
            recs = self._children[source].recommend(
                user, self.depth, candidates=candidates, explain=False
            )
            for rank, rec in enumerate(recs, start=1):
                fused[rec.item_id][source] = w / (self.rank_constant + rank)
        ordered = sorted(fused, key=lambda i: -sum(fused[i].values()))  # stable: ties keep order

        explore: list[ItemId] = []
        if (
            self.explore_source in self._children
            and self.explore_every > 0
            and journey in self.explore_stages
        ):
            explore = [
                r.item_id
                for r in self._children[self.explore_source].recommend(
                    user, self.depth, candidates=candidates, explain=False
                )
            ]
        out: list[tuple[ItemId, dict[str, float], str | None]] = []
        taken: set[ItemId] = set()
        main, extra = iter(ordered), iter(explore)
        while len(out) < self.depth:
            if explore and (len(out) + 1) % self.explore_every == 0:
                item = next((i for i in extra if i not in taken), None)
                if item is not None:
                    out.append((item, {}, self.explore_source))
                    taken.add(item)
                    continue
            item = next((i for i in main if i not in taken), None)
            if item is None:
                break
            out.append((item, fused[item], None))
            taken.add(item)
        return out

    def recommend(
        self,
        user: UserContext,
        k: int = 10,
        *,
        candidates: Container[ItemId] | None = None,
        explain: bool = True,
    ) -> list[ScoredItem]:
        """Overrides the template: each source already applied candidates,
        exclusions and recommendable-only, so the blend only reorders them."""
        blended = self._blend(user, candidates)[:k]
        return [
            ScoredItem(
                item_id,
                1.0 / position,
                self._explain_from(user, item_id, parts, explore) if explain else None,
            )
            for position, (item_id, parts, explore) in enumerate(blended, start=1)
        ]

    def similar_items(
        self,
        item_id: ItemId,
        k: int = 10,
        *,
        candidates: Container[ItemId] | None = None,
    ) -> list[ScoredItem]:
        """Equal-weight rank fusion of the sources that support item-to-item."""
        self._check_fitted()
        fused: dict[ItemId, float] = defaultdict(float)
        for child in self._children.values():
            try:
                recs = child.similar_items(item_id, self.depth, candidates=candidates)
            except NotImplementedError:
                continue
            for rank, rec in enumerate(recs, start=1):
                fused[rec.item_id] += 1.0 / (self.rank_constant + rank)
        ordered = sorted(fused, key=lambda i: -fused[i])[:k]
        return [ScoredItem(i, 1.0 / p) for p, i in enumerate(ordered, start=1)]

    def _score_user(self, user: UserContext, pool: Sequence[ItemId]) -> np.ndarray:
        position = {i: p for p, (i, _, _) in enumerate(self._blend(user, None), start=1)}
        return np.array([1.0 / position[i] if i in position else 0.0 for i in pool])

    # -- explainability -----------------------------------------------------

    def _explain_from(
        self, user: UserContext, item_id: ItemId, parts: dict[str, float], explore: str | None
    ) -> Explanation | None:
        source = explore or max(parts, key=lambda s: parts[s])
        return self._children[source].explain(user, item_id)

    def explain(self, user: UserContext, item_id: ItemId) -> Explanation | None:
        for i, parts, explore in self._blend(user, None):
            if i == item_id:
                return self._explain_from(user, item_id, parts, explore)
        return None

    def score_breakdown(self, user: UserContext, item_id: ItemId) -> ScoreBreakdown | None:
        journey = self.journey(user)
        for position, (i, parts, explore) in enumerate(self._blend(user, None), start=1):
            if i != item_id:
                continue
            total = 1.0 / position
            if explore:
                components = (ScoreComponent(f"exploration slot ({explore})", total),)
            else:
                fused = sum(parts.values())
                components = tuple(
                    ScoreComponent(
                        f"{s} (weight {self.weights.get(journey, self.default_weights)[s]:g})",
                        total * v / fused,
                    )
                    for s, v in sorted(parts.items(), key=lambda p: -p[1])
                )
            return ScoreBreakdown(
                total=total,
                components=components,
                formula=(
                    f"journey '{journey}': order by Σ sources weight / "
                    f"({self.rank_constant:g} + rank); "
                    f"every {self.explore_every}th slot explores; score = 1 / position, "
                    "split by each source's share of the fused score"
                ),
            )
        return None
