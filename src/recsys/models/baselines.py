"""Baselines every real model must beat.

Random sets the floor. MostPopular is the one that matters: well-tuned
popularity baselines beat a surprising number of published models, and a
personalized model that can't beat it isn't worth its complexity.
"""

from __future__ import annotations

import zlib
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from recsys.core.types import (
    Explanation,
    ExplanationKind,
    InteractionKind,
    ItemId,
    ScoreBreakdown,
    ScoreComponent,
    TrainingData,
    UserContext,
)
from recsys.models.base import Recommender


class RandomRecommender(Recommender):
    name = "random"
    version = 2  # v2: per-(user, item) hash, independent of the candidate pool
    personalized = False
    summary = "Gives every eligible item a random score. It never looks at the item or the user."
    uses = "Nothing"
    good_at = "Shows the whole catalog over time (coverage 1.0)."
    weak_at = "No relevance at all. It is the floor every real model must beat."
    concepts = "baseline"

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed

    @property
    def params(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def _draw(self, key: str, item_id: ItemId) -> float:
        # A hash of (seed, user, item), so an item's draw doesn't depend on which
        # other items are in the pool: filters change the list, not the scores.
        return zlib.crc32(f"{self.seed}|{key}|{item_id}".encode()) / 2**32

    def _score_user(self, user: UserContext, pool: Sequence[ItemId]) -> np.ndarray:
        return np.array([self._draw(user.user_id or "", i) for i in pool])

    def _score_similar(self, item_id: ItemId, pool: Sequence[ItemId]) -> np.ndarray:
        return np.array([self._draw(f"similar:{item_id}", i) for i in pool])

    def score_breakdown(self, user: UserContext, item_id: ItemId) -> ScoreBreakdown | None:
        value = self._draw(user.user_id or "", item_id)
        return ScoreBreakdown(
            total=value,
            components=(ScoreComponent("random draw (no signal)", value),),
            formula="score = hash(seed, user, item) scaled to [0, 1)",
        )


DEFAULT_POPULARITY_WEIGHTS: dict[str, float] = {
    InteractionKind.RATING: 1.0,  # likes only; dislikes count 0
    InteractionKind.SAVE: 1.0,
    InteractionKind.AFFILIATE_CLICK: 1.0,
    InteractionKind.CLICK: 0.25,
    # E-commerce funnel: a page view is a weak signal, a cart or purchase a strong one.
    InteractionKind.VIEW: 0.25,
    InteractionKind.ADD_TO_CART: 1.0,
    InteractionKind.PURCHASE: 1.0,
}


class PopularityRecommender(Recommender):
    """Scores items by weighted count of distinct users with a positive signal."""

    name = "most_popular"
    personalized = False
    summary = (
        "Scores each item by how many users liked, saved, carted or bought it. "
        "Everyone sees the same order once hard filters are applied."
    )
    uses = "Everyone else's feedback"
    good_at = "A strong, safe default that works for brand-new visitors."
    weak_at = (
        "Ignores the user's taste and keeps showing the same hits (low coverage, popularity bias)."
    )
    concepts = "popularity baseline, popularity bias"

    def __init__(
        self, weights: dict[str, float] | None = None, retrieval_depth: int = 1000
    ) -> None:
        super().__init__()
        self.weights = dict(weights or DEFAULT_POPULARITY_WEIGHTS)
        # Only the top `retrieval_depth` items are ever scored (plus enough to
        # cover the user's exclusions): the rest of a 235k catalog can't reach
        # a top 20 anyway, unless hard filters remove nearly everything above it.
        self.retrieval_depth = retrieval_depth
        self._popularity: dict[ItemId, float] = {}
        self._ranked: list[ItemId] = []

    @property
    def params(self) -> dict[str, Any]:
        return {"weights": self.weights, "retrieval_depth": self.retrieval_depth}

    def _fit(self, data: TrainingData) -> None:
        # Max weight per (user, item): one enthusiastic user who clicks 50 times
        # shouldn't outweigh 50 users who each liked the item once.
        best: dict[tuple[str, ItemId], float] = defaultdict(float)
        for x in data.interactions:
            if x.kind == InteractionKind.RATING and x.value <= 0:
                continue
            if x.kind == InteractionKind.NOT_INTERESTED:
                continue
            w = self.weights.get(x.kind, 0.0)
            key = (x.user_id, x.item_id)
            best[key] = max(best[key], w)
        popularity: dict[ItemId, float] = defaultdict(float)
        for (_, item_id), w in best.items():
            popularity[item_id] += w
        self._popularity = dict(popularity)
        self._ranked = self._rank(self._popularity)

    def _retrieve(self, user: UserContext, n: int) -> Sequence[ItemId]:
        return self._ranked[: max(n, self.retrieval_depth)]

    def _retrieve_similar(self, item_id: ItemId, n: int) -> Sequence[ItemId]:
        return self._ranked[: max(n, self.retrieval_depth)]

    def _scores(self, pool: Sequence[ItemId]) -> np.ndarray:
        return np.array([self._popularity.get(i, 0.0) for i in pool], dtype=float)

    def _score_user(self, user: UserContext, pool: Sequence[ItemId]) -> np.ndarray:
        return self._scores(pool)

    def _score_similar(self, item_id: ItemId, pool: Sequence[ItemId]) -> np.ndarray:
        return self._scores(pool)

    def explain(self, user: UserContext, item_id: ItemId) -> Explanation | None:
        # No interaction data yet means no popularity: don't claim it.
        if not any(self._popularity.values()):
            return Explanation(ExplanationKind.EXPLORATION)
        return Explanation(ExplanationKind.POPULAR)

    def score_breakdown(self, user: UserContext, item_id: ItemId) -> ScoreBreakdown | None:
        value = self._popularity.get(item_id, 0.0)
        return ScoreBreakdown(
            total=value,
            components=(ScoreComponent("weighted count of users with a positive signal", value),),
            formula=(
                "score = Σ users: max weight of their signals "
                "(like/save/cart/buy 1.0, click/view 0.25)"
            ),
        )
