"""Content-based recommendations from curated tags and numeric attributes.

How it scores an item c for a user:

    score(c) = max over liked items a of S[a, c]
             − dislike_weight · max over disliked items d of S[d, c]
             + affinity_weight · (quiz tag affinities matched by c)
             + popularity_weight · popularity(c)

where S is an item-item similarity: a weighted sum of per-facet tag overlap
(Jaccard) and closeness on numeric scales (1 − mean normalized distance).

Why "max over liked items" rather than an average: a user who loves both big
Cabernet and light Pinot should get good Cabernets *and* good Pinots, not
medium-bodied compromises that match neither. The max also gives each
recommendation exactly one anchor for "because you liked X".

Design choices:
- S is precomputed in `fit` as one matrix per facet (kept for score
  breakdowns) plus one for numeric attributes. Serving is a lookup and a max.
  That's fine up to a few thousand items; beyond that you switch to
  approximate nearest-neighbour search over embeddings (roadmap Phase 3).
- Facets that are bins of a numeric attribute (e.g. `body:full`) are left out
  of the tag overlap because the numeric scale already covers them; counting
  both would double-weight the same fact.
- With no likes and no quiz affinities there is nothing to personalize from
  (cold start), so the model falls back to popularity and says so.
- `popularity_weight` blends in a popularity prior (a simple hybrid). Pure
  content similarity ignores that some wines are crowd-pleasers; a little
  popularity usually helps accuracy, at some cost in coverage. It's a dial to
  measure, not a default to trust.

Look up: content-based filtering, item-item similarity, Jaccard index.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from recsys.core.types import (
    Explanation,
    ExplanationKind,
    InteractionKind,
    ItemId,
    ScoreBreakdown,
    ScoreComponent,
    Tag,
    TrainingData,
    UserContext,
)
from recsys.models.base import Recommender

NUMERIC = "numeric"


class TagSimilarityRecommender(Recommender):
    name = "tag_similarity"

    def __init__(
        self,
        facet_weights: Mapping[str, float] | None = None,
        numeric_weight: float = 0.35,
        dislike_weight: float = 0.5,
        affinity_weight: float = 0.3,
        popularity_weight: float = 0.0,
        ignore_attributes: Sequence[str] = ("price",),
    ) -> None:
        """
        facet_weights: weight per tag facet, e.g. {"varietal": 0.45, "note": 0.2}.
            Facets not listed get weight 0. None = all non-bin facets share
            (1 - numeric_weight) equally.
        numeric_weight: weight of closeness on numeric attributes (e.g. body 1-5).
        ignore_attributes: numeric attributes that aren't taste. Price is a
            hard constraint (a candidate filter), not a reason two wines are alike.
        Weights are normalized to sum to 1, so S is in [0, 1].
        """
        super().__init__()
        self.facet_weights = dict(facet_weights) if facet_weights is not None else None
        self.numeric_weight = numeric_weight
        self.dislike_weight = dislike_weight
        self.affinity_weight = affinity_weight
        self.popularity_weight = popularity_weight
        self.ignore_attributes = tuple(ignore_attributes)
        self._weights: dict[str, float] = {}
        self._parts: dict[str, np.ndarray] = {}
        self._sim = np.zeros((0, 0))
        self._popularity = np.zeros(0)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "facet_weights": self.facet_weights,
            "numeric_weight": self.numeric_weight,
            "dislike_weight": self.dislike_weight,
            "affinity_weight": self.affinity_weight,
            "popularity_weight": self.popularity_weight,
            "ignore_attributes": list(self.ignore_attributes),
        }

    # -- fit ------------------------------------------------------------------

    def _fit(self, data: TrainingData) -> None:
        items = self._items
        n = len(items)
        all_attrs = {a for item in items for a in item.attributes}
        numeric_attrs = sorted(all_attrs - set(self.ignore_attributes))
        tag_facets = sorted({t.facet for item in items for t in item.tags} - all_attrs)

        if self.facet_weights is None:
            num_w = self.numeric_weight if numeric_attrs else 0.0
            share = (1 - num_w) / len(tag_facets) if tag_facets else 0.0
            raw = {f: share for f in tag_facets}
        else:
            unknown = set(self.facet_weights) - set(tag_facets)
            if unknown:
                raise ValueError(
                    f"facet_weights names unknown facets {sorted(unknown)}; "
                    f"catalog facets are {tag_facets}"
                )
            raw = dict(self.facet_weights)
            num_w = self.numeric_weight if numeric_attrs else 0.0
        raw[NUMERIC] = num_w
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("at least one weight must be positive")
        self._weights = {k: v / total for k, v in raw.items() if v > 0}

        self._parts = {}
        for facet in self._weights:
            if facet == NUMERIC:
                self._parts[facet] = _numeric_similarity(items, numeric_attrs)
            else:
                self._parts[facet] = _jaccard(items, facet)
        self._sim = np.zeros((n, n))
        for facet, w in self._weights.items():
            self._sim += w * self._parts[facet]

        # Cold-start fallback: share of users with a positive signal, scaled to [0, 1].
        fans: dict[ItemId, set[str]] = defaultdict(set)
        for x in data.interactions:
            positive = x.kind in (InteractionKind.SAVE, InteractionKind.AFFILIATE_CLICK) or (
                x.kind == InteractionKind.RATING and x.value > 0
            )
            if positive:
                fans[x.item_id].add(x.user_id)
        counts = np.array([len(fans.get(item.item_id, ())) for item in items], dtype=float)
        self._popularity = counts / counts.max() if counts.max() > 0 else counts

    # -- scoring --------------------------------------------------------------

    def _rows(self, ids: Sequence[ItemId]) -> list[int]:
        return [self._index[i] for i in ids if i in self._index]

    def _is_cold(self, user: UserContext) -> bool:
        return not self._rows(user.liked) and not user.tag_affinities

    def _affinity(self, affinities: Mapping[Tag, float]) -> np.ndarray:
        norm = sum(abs(v) for v in affinities.values()) or 1.0
        return np.array(
            [sum(affinities.get(t, 0.0) for t in item.tags) / norm for item in self._items]
        )

    def _score_user(self, user: UserContext, pool: Sequence[ItemId]) -> np.ndarray:
        cols = self._rows(pool)
        if self._is_cold(user):
            return self._popularity[cols]
        scores = np.zeros(len(cols))
        liked, disliked = self._rows(user.liked), self._rows(user.disliked)
        if liked:
            scores += self._sim[np.ix_(liked, cols)].max(axis=0)
        if disliked:
            scores -= self.dislike_weight * self._sim[np.ix_(disliked, cols)].max(axis=0)
        if user.tag_affinities:
            scores += self.affinity_weight * self._affinity(user.tag_affinities)[cols]
        if self.popularity_weight:
            scores += self.popularity_weight * self._popularity[cols]
        return scores

    def _score_similar(self, item_id: ItemId, pool: Sequence[ItemId]) -> np.ndarray:
        if item_id not in self._index:
            raise ValueError(f"unknown item {item_id!r}")
        return self._sim[self._index[item_id], self._rows(pool)]

    # -- explanations ---------------------------------------------------------

    def _best_anchor(self, ids: Sequence[ItemId], item_id: ItemId) -> tuple[ItemId, float] | None:
        rows = self._rows(ids)
        if not rows or item_id not in self._index:
            return None
        sims = self._sim[rows, self._index[item_id]]
        best = int(np.argmax(sims))
        return self._items[rows[best]].item_id, float(sims[best])

    def _shared_tags(self, a: ItemId, b: ItemId, k: int = 3) -> tuple[Tag, ...]:
        """Tags both items have, most heavily weighted facets first."""
        shared = self._items[self._index[a]].tags & self._items[self._index[b]].tags

        def weight(tag: Tag) -> float:
            # Bin tags (body:full) carry the numeric weight.
            return self._weights.get(tag.facet, self._weights.get(NUMERIC, 0.0))

        return tuple(sorted(shared, key=lambda t: (-weight(t), t))[:k])

    def explain(self, user: UserContext, item_id: ItemId) -> Explanation | None:
        if self._is_cold(user):
            return Explanation(ExplanationKind.POPULAR)
        anchor = self._best_anchor(user.liked, item_id)
        if anchor is not None:
            return Explanation(
                ExplanationKind.BECAUSE_YOU_LIKED,
                anchor_item_ids=(anchor[0],),
                shared_tags=self._shared_tags(anchor[0], item_id),
            )
        item = self._items[self._index[item_id]]
        matched = sorted(
            (t for t in item.tags if user.tag_affinities.get(t, 0) > 0),
            key=lambda t: (-user.tag_affinities[t], t),
        )
        return Explanation(ExplanationKind.MATCHES_YOUR_TASTE, shared_tags=tuple(matched[:3]))

    def _explain_similar(self, item_id: ItemId, other: ItemId) -> Explanation | None:
        return Explanation(
            ExplanationKind.SIMILAR_TO,
            anchor_item_ids=(item_id,),
            shared_tags=self._shared_tags(item_id, other),
        )

    def score_breakdown(self, user: UserContext, item_id: ItemId) -> ScoreBreakdown | None:
        if item_id not in self._index:
            return None
        col = self._index[item_id]
        if self._is_cold(user):
            pop = float(self._popularity[col])
            return ScoreBreakdown(
                total=pop,
                components=(ScoreComponent("popularity (cold-start fallback)", pop),),
                formula="no likes or quiz answers yet → score = relative popularity",
            )

        components: list[ScoreComponent] = []
        anchors: tuple[tuple[ItemId, float], ...] = ()
        liked = self._rows(user.liked)
        if liked:
            sims = self._sim[liked, col]
            order = np.argsort(-sims, kind="stable")
            anchors = tuple((self._items[liked[i]].item_id, float(sims[i])) for i in order)
            best = liked[int(order[0])]
            for facet, w in self._weights.items():
                label = "numeric closeness" if facet == NUMERIC else f"{facet} overlap"
                components.append(
                    ScoreComponent(f"{label} × {w:.2f}", w * float(self._parts[facet][best, col]))
                )
        disliked = self._rows(user.disliked)
        if disliked:
            sims = self._sim[disliked, col]
            worst = int(np.argmax(sims))
            components.append(
                ScoreComponent(
                    f"penalty: similar to disliked {self._items[disliked[worst]].item_id}",
                    -self.dislike_weight * float(sims[worst]),
                )
            )
        if user.tag_affinities:
            components.append(
                ScoreComponent(
                    f"quiz taste match × {self.affinity_weight:.2f}",
                    self.affinity_weight * float(self._affinity(user.tag_affinities)[col]),
                )
            )
        if self.popularity_weight:
            components.append(
                ScoreComponent(
                    f"popularity × {self.popularity_weight:.2f}",
                    self.popularity_weight * float(self._popularity[col]),
                )
            )
        weights = " + ".join(
            f"{w:.2f}·{'numeric' if f == NUMERIC else f}" for f, w in self._weights.items()
        )
        formula = (
            f"max over liked of ({weights}) − {self.dislike_weight}·max sim to disliked"
            f" + {self.affinity_weight}·quiz match + {self.popularity_weight}·popularity"
        )
        return ScoreBreakdown(
            total=sum(c.value for c in components),
            components=tuple(components),
            formula=formula,
            anchors=anchors,
        )


def _jaccard(items: Sequence, facet: str) -> np.ndarray:
    """Pairwise Jaccard overlap of each item's values in one facet (via matrix products)."""
    vocab = sorted({t.value for item in items for t in item.tags if t.facet == facet})
    col = {v: k for k, v in enumerate(vocab)}
    m = np.zeros((len(items), len(vocab)))
    for row, item in enumerate(items):
        for t in item.tags:
            if t.facet == facet:
                m[row, col[t.value]] = 1.0
    inter = m @ m.T
    sizes = m.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _numeric_similarity(items: Sequence, attrs: Sequence[str]) -> np.ndarray:
    """1 − mean normalized |difference| over attributes both items have (0 if none)."""
    n = len(items)
    total, count = np.zeros((n, n)), np.zeros((n, n))
    for attr in attrs:
        x = np.array([item.attributes.get(attr, np.nan) for item in items], dtype=float)
        known = ~np.isnan(x)
        if known.sum() == 0:
            continue
        span = np.nanmax(x) - np.nanmin(x) or 1.0
        diff = np.abs(x[:, None] - x[None, :]) / span
        both = known[:, None] & known[None, :]
        total += np.where(both, diff, 0.0)
        count += both
    sim = 1 - total / np.maximum(count, 1)
    sim[count == 0] = 0.0
    return sim
