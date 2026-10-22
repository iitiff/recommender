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
    TrainingData,
    UserContext,
)
from recsys.models.base import Recommender


class RandomRecommender(Recommender):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed

    @property
    def params(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def _rng(self, key: str) -> np.random.Generator:
        # Seeded per user/item so the same request always gets the same answer,
        # which keeps offline evaluation reproducible.
        return np.random.default_rng([self.seed, zlib.crc32(key.encode())])

    def _score_user(self, user: UserContext, pool: Sequence[ItemId]) -> np.ndarray:
        return self._rng(user.user_id or "").random(len(pool))

    def _score_similar(self, item_id: ItemId, pool: Sequence[ItemId]) -> np.ndarray:
        return self._rng(item_id).random(len(pool))


DEFAULT_POPULARITY_WEIGHTS: dict[str, float] = {
    InteractionKind.RATING: 1.0,  # likes only; dislikes count 0
    InteractionKind.SAVE: 1.0,
    InteractionKind.AFFILIATE_CLICK: 1.0,
    InteractionKind.CLICK: 0.25,
}


class PopularityRecommender(Recommender):
    """Scores items by weighted count of distinct users with a positive signal."""

    name = "most_popular"

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        super().__init__()
        self.weights = dict(weights or DEFAULT_POPULARITY_WEIGHTS)
        self._popularity: dict[ItemId, float] = {}

    @property
    def params(self) -> dict[str, Any]:
        return {"weights": self.weights}

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

    def _scores(self, pool: Sequence[ItemId]) -> np.ndarray:
        return np.array([self._popularity.get(i, 0.0) for i in pool], dtype=float)

    def _score_user(self, user: UserContext, pool: Sequence[ItemId]) -> np.ndarray:
        return self._scores(pool)

    def _score_similar(self, item_id: ItemId, pool: Sequence[ItemId]) -> np.ndarray:
        return self._scores(pool)

    def explain(self, user: UserContext, item_id: ItemId) -> Explanation | None:
        return Explanation(ExplanationKind.POPULAR)
