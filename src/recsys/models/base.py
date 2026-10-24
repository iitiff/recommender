"""The Recommender interface every model implements.

Template-method design: this base class owns the behaviour that must be
identical across models (candidate filtering, excluding already-rated items,
top-k selection, attaching explanations), and subclasses only implement
*scoring*. That keeps model comparisons fair: no model can look better in
evaluation because it forgot to exclude items the user already rated.

Two-stage serving: a model can override `_retrieve` to propose a short list
of candidates (its neighbours of what the user just viewed, the top of a
popularity ranking) so it never scores the whole catalog. With 235k items,
scoring everything for every page view is too slow; real systems always
*retrieve* a few hundred candidates, then *rank* them. Hard filters and
exclusions apply to the retrieved list exactly as they do to the full catalog.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from collections.abc import Container, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from recsys.core.types import (
    Explanation,
    Item,
    ItemId,
    ScoreBreakdown,
    ScoredItem,
    TrainingData,
    UserContext,
)


class NotFittedError(RuntimeError):
    pass


class Recommender(ABC):
    name: ClassVar[str]
    # Bump when the algorithm changes in a way that changes results, so
    # model_id changes too. Two different algorithms must never share an id.
    version: ClassVar[int] = 1
    # Does the model react to this user's own likes/quiz answers?
    personalized: ClassVar[bool] = True
    # True = ranks every eligible item. False = a *candidate source* that only
    # returns items it has evidence for (e.g. co-viewed with something the user
    # just looked at), so its lists can be shorter than k, or empty.
    exhaustive: ClassVar[bool] = True
    # Plain-language description for explainability tools (the Rec Lab).
    summary: ClassVar[str] = ""
    uses: ClassVar[str] = ""
    good_at: ClassVar[str] = ""
    weak_at: ClassVar[str] = ""
    concepts: ClassVar[str] = ""  # names to look up

    def __init__(self) -> None:
        self._items: list[Item] = []
        self._index: dict[ItemId, int] = {}
        self._recommendable: list[ItemId] = []
        self._recommendable_set: frozenset[ItemId] = frozenset()
        self._fitted = False

    # -- identity -----------------------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        """Hyperparameters. Override in subclasses; they define `model_id`."""
        return {}

    @property
    def model_id(self) -> str:
        """`name@hash(version, params)`: logged with every slate so outcomes are attributable."""
        identity = self.params if self.version == 1 else {"_version": self.version, **self.params}
        blob = json.dumps(identity, sort_keys=True, default=str).encode()
        return f"{self.name}@{hashlib.sha1(blob).hexdigest()[:8]}"

    # -- public API ---------------------------------------------------------

    def fit(self, data: TrainingData) -> Self:
        self._items = [i for i in data.items if i.is_active]
        self._index = {item.item_id: n for n, item in enumerate(self._items)}
        # Non-recommendable items stay in self._items (models learn from them and
        # use them as explanation anchors) but never enter a candidate pool.
        self._recommendable = [item.item_id for item in self._items if item.recommendable]
        self._recommendable_set = frozenset(self._recommendable)
        self._fit(data)
        self._fitted = True
        return self

    def recommend(
        self,
        user: UserContext,
        k: int = 10,
        *,
        candidates: Container[ItemId] | None = None,
        explain: bool = True,
    ) -> list[ScoredItem]:
        """Top-k items for a user.

        `candidates` restricts the pool. Hard business constraints (budget,
        in stock, ships to the user's state) are applied by the caller through
        this argument, not inside models, so every model obeys them the same way.
        It can be any container with a fast `in` (a set, or a view such as
        "in stock at time t").
        """
        self._check_fitted()
        retrieved = self._retrieve(user, k + len(user.excluded))
        pool = self._pool(candidates, exclude=user.excluded, retrieved=retrieved)
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
        candidates: Container[ItemId] | None = None,
    ) -> list[ScoredItem]:
        """Item-to-item recommendations ("you might also like")."""
        self._check_fitted()
        retrieved = self._retrieve_similar(item_id, k + 1)
        pool = self._pool(candidates, exclude={item_id}, retrieved=retrieved)
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

    def score_breakdown(self, user: UserContext, item_id: ItemId) -> ScoreBreakdown | None:
        """Optional: the components of `item_id`'s score for `user` (they sum to the score).

        Powers explainability tools like the Rec Lab. Models that can't
        decompose their score return None.
        """
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

    def _retrieve(self, user: UserContext, n: int) -> Sequence[ItemId] | None:
        """Candidate items worth scoring for `user` (about `n` are needed).

        None (the default) means the whole catalog. Sources that only know
        about some items return just those.
        """
        return None

    def _retrieve_similar(self, item_id: ItemId, n: int) -> Sequence[ItemId] | None:
        """Like `_retrieve`, for `similar_items`."""
        return None

    @abstractmethod
    def _score_user(self, user: UserContext, pool: Sequence[ItemId]) -> np.ndarray:
        """Score every item in `pool` for `user`. Higher is better."""

    def _score_similar(self, item_id: ItemId, pool: Sequence[ItemId]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not support similar_items")

    def _explain_similar(self, item_id: ItemId, other: ItemId) -> Explanation | None:
        return None

    # -- helpers ------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(f"{self.name}: call fit() first")

    def _rank(self, scores: Mapping[ItemId, float]) -> list[ItemId]:
        """Every recommendable item, best score first; ties keep catalog order.

        User-independent models compute this once at fit time, and `_retrieve`
        then just slices the top of it.
        """
        values = np.array([scores.get(i, 0.0) for i in self._recommendable], dtype=float)
        return [self._recommendable[n] for n in np.argsort(-values, kind="stable")]

    def _pool(
        self,
        candidates: Container[ItemId] | None,
        exclude: Container[ItemId],
        retrieved: Sequence[ItemId] | None = None,
    ) -> list[ItemId]:
        self._check_fitted()
        if retrieved is None:
            ids: Sequence[ItemId] = self._recommendable
        else:  # dedupe, keep the retrieval order (it breaks score ties)
            ok = self._recommendable_set
            ids = [i for i in dict.fromkeys(retrieved) if i in ok]
        if candidates is not None:
            # Lists and tuples have slow `in`; anything else is used as is.
            allowed = set(candidates) if isinstance(candidates, Sequence) else candidates
            ids = [i for i in ids if i in allowed]
        return [i for i in ids if i not in exclude]


def _top_k(pool: Sequence[ItemId], scores: np.ndarray, k: int) -> list[tuple[ItemId, float]]:
    if len(scores) != len(pool):
        raise ValueError(f"model returned {len(scores)} scores for {len(pool)} items")
    # Stable sort: ties keep catalog order, so results are deterministic.
    order = np.argsort(-scores, kind="stable")[:k]
    return [(pool[i], float(scores[i])) for i in order]
