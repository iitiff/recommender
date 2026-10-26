"""Session-based e-commerce candidate sources.

Each model here does one job well and is honest about what it can't do:

- `RecentlyViewedRecommender`: items this shopper already looked at, carted
  or bought, most recent first. Covers "pick up where you left off" and
  re-buys. Knows nothing about items the shopper hasn't touched.
- `CoVisitationRecommender`: items other shoppers looked at in the same
  visits as what this shopper just viewed ("viewed together"). Covers
  continuing *this* shopping trip into items they haven't seen yet.
- `TrendingRecommender`: what is getting attention lately, with older
  activity fading out. Covers brand-new visitors and discovery.

They are *candidate sources* in the two-stage sense: later a blender decides
how much each one should count for a given shopper (a new visitor, a
first-time buyer, a loyal customer). Evaluating them one at a time, split by
lifecycle stage and by what the shopper did next, shows where each one works.

Concepts to look up: session-based recommendation, co-visitation / item-to-item
collaborative filtering, association rules (support, confidence), time decay,
two-stage retrieval and ranking (the OTTO Kaggle competition is a good case
study).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import numpy as np

from recsys.core.types import (
    Explanation,
    ExplanationKind,
    Interaction,
    InteractionKind,
    ItemId,
    ScoreBreakdown,
    ScoreComponent,
    TrainingData,
    UserContext,
)
from recsys.models.base import Recommender
from recsys.sessions import sessionize
from recsys.shopper import DEFAULT_EVENT_WEIGHTS

_VERB = {
    InteractionKind.VIEW: "viewed",
    InteractionKind.CLICK: "clicked",
    InteractionKind.ADD_TO_CART: "carted",
    InteractionKind.PURCHASE: "bought",
}


def _strongest(events: Sequence[Interaction], weights: Mapping[str, float]) -> dict[ItemId, Any]:
    """item -> (weight, kind) of the user's strongest signal on it, over `events`."""
    best: dict[ItemId, Any] = {}
    for x in events:
        w = weights.get(x.kind, 0.0)
        if w > 0 and (x.item_id not in best or w > best[x.item_id][0]):
            best[x.item_id] = (w, x.kind)
    return best


def _recent_items(events: Sequence[Interaction], weights: Mapping[str, float]) -> list[ItemId]:
    """Distinct items with a positive-weight event, most recent first."""
    seen: dict[ItemId, None] = {}
    for x in reversed(events):
        if weights.get(x.kind, 0.0) > 0:
            seen.setdefault(x.item_id, None)
    return list(seen)


class RecentlyViewedRecommender(Recommender):
    """score(item) = weight of the strongest signal on it × decay^(recency rank)."""

    name = "recently_viewed"
    exhaustive = False
    summary = (
        "Shows items this shopper already looked at, carted or bought, newest first; "
        "carted items rank above items only viewed."
    )
    uses = "This shopper's own recent behaviour"
    good_at = "Getting back to an item the shopper was considering, and re-buys."
    weak_at = "Never suggests anything new. Useless for a first page view."
    concepts = "recently viewed, recency, repeat purchase"

    def __init__(
        self,
        decay: float = 0.8,
        max_items: int = 50,
        event_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.decay = decay
        self.max_items = max_items
        self.event_weights = dict(event_weights or DEFAULT_EVENT_WEIGHTS)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "max_items": self.max_items,
            "event_weights": self.event_weights,
        }

    def _user_scores(self, user: UserContext) -> dict[ItemId, float]:
        best = _strongest(user.events, self.event_weights)
        recent = _recent_items(user.events, self.event_weights)[: self.max_items]
        return {item: best[item][0] * self.decay**rank for rank, item in enumerate(recent)}

    def _retrieve(self, user: UserContext, n: int) -> Sequence[ItemId]:
        return list(self._user_scores(user))

    def _retrieve_similar(self, item_id: ItemId, n: int) -> Sequence[ItemId]:
        return []  # nothing to say about an item without a shopper

    def _score_user(self, user: UserContext, pool: Sequence[ItemId]) -> np.ndarray:
        scores = self._user_scores(user)
        return np.array([scores.get(i, 0.0) for i in pool], dtype=float)

    def explain(self, user: UserContext, item_id: ItemId) -> Explanation | None:
        return Explanation(ExplanationKind.RECENTLY_VIEWED, anchor_item_ids=(item_id,))

    def score_breakdown(self, user: UserContext, item_id: ItemId) -> ScoreBreakdown | None:
        best = _strongest(user.events, self.event_weights)
        recent = _recent_items(user.events, self.event_weights)[: self.max_items]
        if item_id not in recent:
            return ScoreBreakdown(0.0, (), "not in this shopper's recent items")
        weight, kind = best[item_id]
        rank = recent.index(item_id)
        value = weight * self.decay**rank
        return ScoreBreakdown(
            total=value,
            components=(
                ScoreComponent(f"you {_VERB.get(kind, kind)} it ({rank} items ago)", value),
            ),
            formula=f"score = signal weight ({weight:g}) × {self.decay:g}^recency rank",
        )


class CoVisitationRecommender(Recommender):
    """Item-to-item co-visitation from sessions, applied to the shopper's latest items.

    Training counts, for every pair of items (a, b) in the same visit, the
    weight of the shopper's strongest signal on b (a cart counts more than a
    view), once per visit. Dividing by the number of visits containing `a`
    gives a weighted *confidence* (association rules): "of the visits that
    included a, how strongly did they also go for b". Only the top
    `max_neighbors` neighbours per item are kept, so memory grows with the
    number of items, not items squared.

    Serving looks at the shopper's latest `max_anchors` items (newest counts
    most: weight × anchor_decay^rank) and sums their neighbours' confidences.
    """

    name = "covisitation"
    exhaustive = False
    summary = (
        "Finds items other shoppers looked at in the same visit as what this shopper "
        "just viewed, weighting the most recent views and carts most."
    )
    uses = "Everyone's visits, plus this shopper's latest items"
    good_at = "Continuing the current shopping trip into items the shopper hasn't seen yet."
    weak_at = "Nothing to go on before the first view; favours items many people browse."
    concepts = "co-visitation, item-to-item collaborative filtering, association rules"
    # Wording, so subclasses that pick different anchors explain themselves.
    _anchor_label = "viewed together with {title}"
    _explanation_kind = ExplanationKind.ALSO_VIEWED

    def __init__(
        self,
        max_neighbors: int = 50,
        max_anchors: int = 10,
        anchor_decay: float = 0.8,
        max_session_items: int = 30,
        event_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.max_neighbors = max_neighbors
        self.max_anchors = max_anchors
        self.anchor_decay = anchor_decay
        self.max_session_items = max_session_items
        self.event_weights = dict(event_weights or DEFAULT_EVENT_WEIGHTS)
        # CSR layout: neighbours of item n are _nbr[_indptr[n]:_indptr[n + 1]], best first.
        self._indptr = np.zeros(1, dtype=np.int64)
        self._nbr = np.zeros(0, dtype=np.int64)
        self._conf = np.zeros(0, dtype=float)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "max_neighbors": self.max_neighbors,
            "max_anchors": self.max_anchors,
            "anchor_decay": self.anchor_decay,
            "max_session_items": self.max_session_items,
            "event_weights": self.event_weights,
        }

    def _fit(self, data: TrainingData) -> None:
        n_items = len(self._items)
        events = [x for x in data.interactions if self.event_weights.get(x.kind, 0.0) > 0]
        if any(x.session_id is None for x in events):
            events = sessionize(events)
        # One row per (visit, item): strongest weight, last position in time.
        events.sort(key=lambda x: x.occurred_at)
        codes: dict[str, int] = {}
        rows_s, rows_i, rows_w, rows_t = [], [], [], []
        for t, x in enumerate(events):
            item = self._index.get(x.item_id)
            if item is None:
                continue
            rows_s.append(codes.setdefault(x.session_id or "", len(codes)))
            rows_i.append(item)
            rows_w.append(self.event_weights[x.kind])
            rows_t.append(t)
        self._indptr = np.zeros(n_items + 1, dtype=np.int64)
        if not rows_s:
            return
        s, i = np.array(rows_s, dtype=np.int64), np.array(rows_i, dtype=np.int64)
        w, t = np.array(rows_w, dtype=float), np.array(rows_t, dtype=np.int64)

        key = s * n_items + i
        order = np.lexsort((t, key))
        key, s, i, w, t = key[order], s[order], i[order], w[order], t[order]
        first = np.r_[True, key[1:] != key[:-1]]
        starts = np.flatnonzero(first)
        w = np.maximum.reduceat(w, starts)
        t = np.maximum.reduceat(t, starts)
        s, i = s[starts], i[starts]

        # Keep each visit's latest `max_session_items` items (bounds the pair count).
        order = np.lexsort((-t, s))
        s, i, w = s[order], i[order], w[order]
        visit_start = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
        sizes = np.diff(np.r_[visit_start, len(s)])
        rank = np.arange(len(s)) - np.repeat(visit_start, sizes)
        keep = rank < self.max_session_items
        s, i, w = s[keep], i[keep], w[keep]
        visit_start = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
        sizes = np.diff(np.r_[visit_start, len(s)])

        visits_with = np.bincount(i, minlength=n_items).astype(float)

        # All ordered pairs within each visit, vectorized: row r pairs with every
        # row of its own visit.
        per_row = np.repeat(sizes, sizes)
        src_row = np.repeat(np.arange(len(s)), per_row)
        offsets = np.repeat(np.repeat(visit_start, sizes), per_row)
        ramp = np.arange(per_row.sum()) - np.repeat(np.cumsum(per_row) - per_row, per_row)
        dst_row = offsets + ramp
        distinct = src_row != dst_row
        src, dst = i[src_row[distinct]], i[dst_row[distinct]]
        weight = w[dst_row[distinct]]

        pair, inverse = np.unique(src * n_items + dst, return_inverse=True)
        total = np.bincount(inverse, weights=weight)
        src, dst = pair // n_items, pair % n_items
        conf = total / visits_with[src]

        order = np.lexsort((dst, -conf, src))  # by source, best first, ties by item order
        src, dst, conf = src[order], dst[order], conf[order]
        group_start = np.r_[0, np.flatnonzero(src[1:] != src[:-1]) + 1]
        group_sizes = np.diff(np.r_[group_start, len(src)])
        rank = np.arange(len(src)) - np.repeat(group_start, group_sizes)
        keep = rank < self.max_neighbors
        src, self._nbr, self._conf = src[keep], dst[keep], conf[keep]
        self._indptr[1:] = np.cumsum(np.bincount(src, minlength=n_items))

    def neighbors(self, item_id: ItemId) -> list[tuple[ItemId, float]]:
        """(item, confidence) pairs for one item, best first."""
        n = self._index.get(item_id)
        if n is None:
            return []
        lo, hi = self._indptr[n], self._indptr[n + 1]
        return [
            (self._items[j].item_id, float(c))
            for j, c in zip(self._nbr[lo:hi], self._conf[lo:hi], strict=True)
        ]

    def _anchors(self, user: UserContext) -> list[tuple[ItemId, float]]:
        best = _strongest(user.events, self.event_weights)
        recent = _recent_items(user.events, self.event_weights)[: self.max_anchors]
        return [(a, best[a][0] * self.anchor_decay**rank) for rank, a in enumerate(recent)]

    def _contributions(self, user: UserContext) -> dict[ItemId, dict[ItemId, float]]:
        """candidate -> {anchor: weight × confidence}."""
        out: dict[ItemId, dict[ItemId, float]] = defaultdict(dict)
        for anchor, weight in self._anchors(user):
            for other, conf in self.neighbors(anchor):
                pair = self._pair_weight(anchor, other)
                if pair > 0:
                    out[other][anchor] = weight * conf * pair
        return out

    def _pair_weight(self, anchor: ItemId, other: ItemId) -> float:
        """Multiplier on a neighbour (0 drops it). Subclasses restrict or re-weight pairs."""
        return 1.0

    def _user_scores(self, user: UserContext) -> dict[ItemId, float]:
        return {item: sum(parts.values()) for item, parts in self._contributions(user).items()}

    def _retrieve(self, user: UserContext, n: int) -> Sequence[ItemId]:
        scores = self._user_scores(user)
        return sorted(scores, key=lambda i: -scores[i])

    def _retrieve_similar(self, item_id: ItemId, n: int) -> Sequence[ItemId]:
        return [
            other for other, _ in self.neighbors(item_id) if self._pair_weight(item_id, other) > 0
        ]

    def _score_user(self, user: UserContext, pool: Sequence[ItemId]) -> np.ndarray:
        scores = self._user_scores(user)
        return np.array([scores.get(i, 0.0) for i in pool], dtype=float)

    def _score_similar(self, item_id: ItemId, pool: Sequence[ItemId]) -> np.ndarray:
        conf = dict(self.neighbors(item_id))
        return np.array(
            [conf.get(i, 0.0) * self._pair_weight(item_id, i) for i in pool], dtype=float
        )

    def explain(self, user: UserContext, item_id: ItemId) -> Explanation | None:
        parts = self._contributions(user).get(item_id)
        if not parts:
            return None
        anchor = max(parts, key=lambda a: parts[a])
        return Explanation(self._explanation_kind, anchor_item_ids=(anchor,))

    def _explain_similar(self, item_id: ItemId, other: ItemId) -> Explanation | None:
        return Explanation(self._explanation_kind, anchor_item_ids=(item_id,))

    def score_breakdown(self, user: UserContext, item_id: ItemId) -> ScoreBreakdown | None:
        parts = self._contributions(user).get(item_id, {})
        ranked = sorted(parts.items(), key=lambda p: -p[1])
        titles = {i.item_id: i.title for i in self._items}
        return ScoreBreakdown(
            total=sum(parts.values()),
            components=tuple(
                ScoreComponent(self._anchor_label.format(title=titles.get(a, a)), v)
                for a, v in ranked
            ),
            formula=(
                "score = Σ anchors a: (anchor weight × "
                f"{self.anchor_decay:g}^rank) × confidence(a → item) × pair weight"
            ),
            anchors=tuple(ranked),
        )


class TrendingRecommender(Recommender):
    """score(item) = Σ shoppers: their strongest signal × 0.5^(age / half-life).

    Like MostPopular (one enthusiastic shopper counts once per item), but old
    activity fades: with a 7-day half-life, a view from two weeks ago counts a
    quarter of a view from today. Catalogs with churn (new arrivals, seasonal
    items, items going out of stock) need this; an all-time count keeps
    recommending last season's hits.
    """

    name = "trending"
    personalized = False
    summary = (
        "Ranks items by recent attention from all shoppers; activity fades by half "
        "every half-life, so new hits rise quickly."
    )
    uses = "Everyone's recent behaviour"
    good_at = "A brand-new visitor's first page, and keeping up with a changing catalog."
    weak_at = "The same list for everyone; ignores what this shopper is looking at."
    concepts = "time decay, half-life, trending, popularity bias"

    def __init__(
        self,
        half_life_days: float = 7.0,
        retrieval_depth: int = 1000,
        event_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.half_life_days = half_life_days
        self.retrieval_depth = retrieval_depth
        self.event_weights = dict(event_weights or DEFAULT_EVENT_WEIGHTS)
        self._scores: dict[ItemId, float] = {}
        self._ranked: list[ItemId] = []
        self.as_of: datetime | None = None

    @property
    def params(self) -> dict[str, Any]:
        return {
            "half_life_days": self.half_life_days,
            "retrieval_depth": self.retrieval_depth,
            "event_weights": self.event_weights,
        }

    def _fit(self, data: TrainingData) -> None:
        events = [x for x in data.interactions if self.event_weights.get(x.kind, 0.0) > 0]
        self._scores = {}
        self.as_of = max((x.occurred_at for x in events), default=None)
        if self.as_of is not None:
            half_life = self.half_life_days * 86400
            best: dict[tuple[str, ItemId], float] = defaultdict(float)
            for x in events:
                age = (self.as_of - x.occurred_at).total_seconds()
                w = self.event_weights[x.kind] * 0.5 ** (age / half_life)
                key = (x.user_id, x.item_id)
                best[key] = max(best[key], w)
            scores: dict[ItemId, float] = defaultdict(float)
            for (_, item), w in best.items():
                scores[item] += w
            self._scores = dict(scores)
        self._ranked = self._rank(self._scores)

    def _retrieve(self, user: UserContext, n: int) -> Sequence[ItemId]:
        return self._ranked[: max(n, self.retrieval_depth)]

    def _retrieve_similar(self, item_id: ItemId, n: int) -> Sequence[ItemId]:
        return self._ranked[: max(n, self.retrieval_depth)]

    def _score_user(self, user: UserContext, pool: Sequence[ItemId]) -> np.ndarray:
        return np.array([self._scores.get(i, 0.0) for i in pool], dtype=float)

    def _score_similar(self, item_id: ItemId, pool: Sequence[ItemId]) -> np.ndarray:
        return self._score_user(UserContext(), pool)

    def explain(self, user: UserContext, item_id: ItemId) -> Explanation | None:
        if not self._scores:
            return Explanation(ExplanationKind.EXPLORATION)
        return Explanation(ExplanationKind.TRENDING)

    def score_breakdown(self, user: UserContext, item_id: ItemId) -> ScoreBreakdown | None:
        value = self._scores.get(item_id, 0.0)
        return ScoreBreakdown(
            total=value,
            components=(ScoreComponent("recent shoppers' signals, faded by age", value),),
            formula=(
                "score = Σ shoppers: strongest signal × 0.5^(age / "
                f"{self.half_life_days:g} days), as of {self.as_of or 'no data'}"
            ),
        )
