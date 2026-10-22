"""The Recommender interface every model implements.

Template-method design: this base class owns the behaviour that must be
identical across models (candidate filtering, excluding already-rated items,
top-k selection, attaching explanations), and subclasses only implement
*scoring*. That keeps model comparisons fair: no model can look better in
evaluation because it forgot to exclude items the user already rated.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from recsys.core.types import (
    Explanation,
    Item,
    ItemId,
    ScoredItem,
    TrainingData,
    UserContext,
)


class NotFittedError(RuntimeError):
    pass


class Recommender(ABC):
    name: ClassVar[str]

    def __init__(self) -> None:
        self._items: list[Item] = []
        self._index: dict[ItemId, int] = {}
        self._fitted = False

    # -- identity -----------------------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        """Hyperparameters. Override in subclasses; they define `model_id`."""
        return {}

    @property
    def model_id(self) -> str:
        """`name@hash(params)`: logged with every slate so outcomes are attributable."""
        blob = json.dumps(self.params, sort_keys=True, default=str).encode()
        return f"{self.name}@{hashlib.sha1(blob).hexdigest()[:8]}"

    # -- public API ---------------------------------------------------------

    def fit(self, data: TrainingData) -> Self:
        self._items = [i for i in data.items if i.is_active]
        self._index = {item.item_id: n for n, item in enumerate(self._items)}
        self._fit(data)
        self._fitted = True
        return self

    def recommend(
        self,
        user: UserContext,
        k: int = 10,
        *,
        candidates: Collection[ItemId] | None = None,
        explain: bool = True,
    ) -> list[ScoredItem]:
        """Top-k items for a user.

        `candidates` restricts the pool. Hard business constraints (budget,
        in stock, ships to the user's state) are applied by the caller through
        this argument, not inside models, so every model obeys them the same way.
        """
        pool = self._pool(candidates, exclude=user.excluded)
        if not pool:
            return []
        scores = self._score_user(user, pool)
        return [
            ScoredItem(item_id, score, self.explain(user, item_id) if explain else None)
            for item_id, score in _top_k(pool, scores, k)
        ]

    def similar_items(
        self,
        item_id: ItemId,
        k: int = 10,
        *,
        candidates: Collection[ItemId] | None = None,
    ) -> list[ScoredItem]:
        """Item-to-item recommendations ("you might also like")."""
        pool = self._pool(candidates, exclude={item_id})
        if not pool:
            return []
        scores = self._score_similar(item_id, pool)
        return [
            ScoredItem(other, score, self._explain_similar(item_id, other))
            for other, score in _top_k(pool, scores, k)
        ]

    def explain(self, user: UserContext, item_id: ItemId) -> Explanation | None:
        """Why `item_id` suits `user`. Models without a meaningful reason return None."""
        return None

    def save(self, path: Path) -> None:
        """Persist the fitted model. Pickle is fine for artifacts we produce
        ourselves; never load a pickle from an untrusted source."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> Self:
        model = pickle.loads(path.read_bytes())
        if not isinstance(model, cls):
            raise TypeError(f"{path} holds a {type(model).__name__}, not a {cls.__name__}")
        return model

    # -- subclass hooks -----------------------------------------------------

    def _fit(self, data: TrainingData) -> None:  # noqa: B027 - optional hook
        """Learn from `data`. `self._items` and `self._index` are already set."""

    @abstractmethod
    def _score_user(self, user: UserContext, pool: Sequence[ItemId]) -> np.ndarray:
        """Score every item in `pool` for `user`. Higher is better."""

    def _score_similar(self, item_id: ItemId, pool: Sequence[ItemId]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not support similar_items")

    def _explain_similar(self, item_id: ItemId, other: ItemId) -> Explanation | None:
        return None

    # -- helpers ------------------------------------------------------------

    def _pool(
        self, candidates: Collection[ItemId] | None, exclude: Collection[ItemId]
    ) -> list[ItemId]:
        if not self._fitted:
            raise NotFittedError(f"{self.name}: call fit() first")
        if candidates is None:
            ids: Sequence[ItemId] = [item.item_id for item in self._items]
        else:
            allowed = set(candidates)
            ids = [item.item_id for item in self._items if item.item_id in allowed]
        excluded = set(exclude)
        return [i for i in ids if i not in excluded]


def _top_k(pool: Sequence[ItemId], scores: np.ndarray, k: int) -> list[tuple[ItemId, float]]:
    if len(scores) != len(pool):
        raise ValueError(f"model returned {len(scores)} scores for {len(pool)} items")
    # Stable sort: ties keep catalog order, so results are deterministic.
    order = np.argsort(-scores, kind="stable")[:k]
    return [(pool[i], float(scores[i])) for i in order]
