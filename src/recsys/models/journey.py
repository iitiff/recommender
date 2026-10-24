"""Candidate sources for later stages of the shopper journey.

Each does one job from docs/SHOPPER-PROFILE.md:

- `RebuyRecommender`: things the shopper bought before, ranked by how often
  buyers of each item buy it again (repeat customers mostly re-cart known
  items).
- `CartSubstitutesRecommender`: alternatives to what is in their open cart:
  same leaf category, viewed in the same visits by other shoppers, near the
  cart item's price. Abandoners comparison-shop: 45% of their carts are new
  items in categories they browsed.
- `AdjacentCategoryRecommender`: popular items in product types that shoppers
  like them tend to move into next (learned from type-to-type transitions),
  which they haven't touched yet. The source for measured category expansion.

Concepts: repurchase propensity, substitutes vs complements, category
transitions (a first-order Markov chain over product types), lift.
"""

from __future__ import annotations

import math
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
from recsys.models.session import CoVisitationRecommender
from recsys.shopper import DEFAULT_EVENT_WEIGHTS, ItemFacts, shopper_state


class RebuyRecommender(Recommender):
    """score(item) = smoothed share of the item's buyers who bought it again.

    Smoothing (empirical Bayes): an item with few buyers leans on its product
    type's re-buy rate, and a type with few buyers on the store's:
        rate(item) = (rebuyers + k·rate(type)) / (buyers + k)
    Items the shopper bought in the current visit are left out: they just
    bought it.
    """

    name = "rebuy"
    exhaustive = False
    summary = (
        "Shows items this shopper bought before, ranked by how often people buy each item again."
    )
    uses = "This shopper's purchases, and everyone's repeat purchases"
    good_at = "Repeat and loyal customers, replenishable products."
    weak_at = "Nothing for shoppers who haven't bought; ignores timing (when they'll run out)."
    concepts = "repurchase propensity, empirical Bayes smoothing"

    def __init__(self, type_facet: str | None = None, prior_strength: float = 10.0) -> None:
        super().__init__()
        self.type_facet = type_facet
        self.prior_strength = prior_strength
        self._rate: dict[ItemId, float] = {}
        self._type_rate: dict[str, float] = {}
        self._global_rate = 0.0
        self._facts = ItemFacts()

    @property
    def params(self) -> dict[str, Any]:
        return {"type_facet": self.type_facet, "prior_strength": self.prior_strength}

    def _fit(self, data: TrainingData) -> None:
        self._facts = ItemFacts.from_items(self._items, type_facet=self.type_facet)
        days: dict[tuple[str, ItemId], set] = defaultdict(set)
        for x in data.interactions:
            if x.kind == InteractionKind.PURCHASE:
                days[(x.user_id, x.item_id)].add(x.occurred_at.date())
        buyers: dict[ItemId, int] = defaultdict(int)
        rebuyers: dict[ItemId, int] = defaultdict(int)
        for (_, item_id), d in days.items():
            buyers[item_id] += 1
            rebuyers[item_id] += len(d) > 1
        k = self.prior_strength
        total = sum(buyers.values())
        self._global_rate = sum(rebuyers.values()) / total if total else 0.0
        t_buyers: dict[str, int] = defaultdict(int)
        t_rebuyers: dict[str, int] = defaultdict(int)
        for item_id, n in buyers.items():
            t = self._facts.type_of.get(item_id, "")
            t_buyers[t] += n
            t_rebuyers[t] += rebuyers[item_id]
        self._type_rate = {
            t: (t_rebuyers[t] + k * self._global_rate) / (n + k) for t, n in t_buyers.items()
        }
        self._rate = {
            item_id: (rebuyers[item_id] + k * self._prior(item_id)) / (n + k)
            for item_id, n in buyers.items()
        }

    def _prior(self, item_id: ItemId) -> float:
        return self._type_rate.get(self._facts.type_of.get(item_id, ""), self._global_rate)

    def rate(self, item_id: ItemId) -> float:
        return self._rate.get(item_id, self._prior(item_id))

    def _bought(self, user: UserContext) -> list[ItemId]:
        just_bought = {x.item_id for x in user.session_events if x.kind == InteractionKind.PURCHASE}
        return [i for i in shopper_state(user).bought if i not in just_bought]

    def _retrieve(self, user: UserContext, n: int) -> Sequence[ItemId]:
        return self._bought(user)  # newest first: recency breaks ties

    def _retrieve_similar(self, item_id: ItemId, n: int) -> Sequence[ItemId]:
        return []

    def _score_user(self, user: UserContext, pool: Sequence[ItemId]) -> np.ndarray:
        return np.array([self.rate(i) for i in pool], dtype=float)

    def explain(self, user: UserContext, item_id: ItemId) -> Explanation | None:
        return Explanation(ExplanationKind.BUY_AGAIN, anchor_item_ids=(item_id,))

    def score_breakdown(self, user: UserContext, item_id: ItemId) -> ScoreBreakdown | None:
        value = self.rate(item_id)
        return ScoreBreakdown(
            total=value,
            components=(ScoreComponent("share of this item's buyers who bought it again", value),),
            formula=(
                f"(rebuyers + {self.prior_strength:g}·type rate) / "
                f"(buyers + {self.prior_strength:g})"
            ),
        )


class CartSubstitutesRecommender(CoVisitationRecommender):
    """Co-visitation restricted to alternatives for the items in the open cart.

    Anchors are the open cart (carted since the last order, not bought),
    newest first. Neighbours must share the anchor's leaf category
    (substitutes, not complements), and when both have prices they are
    weighted by price closeness: exp(-|log(price ratio)| / price_band). With
    price_band 0.3, an item 30% pricier counts about 0.4×, one twice the
    price about 0.1×.
    """

    name = "cart_substitutes"
    summary = (
        "Suggests alternatives to what's in the shopper's cart: same kind of product, "
        "viewed alongside it by other shoppers, at a similar price."
    )
    uses = "This shopper's open cart, everyone's visits, prices"
    good_at = "Comparison shoppers and cart abandoners who are still deciding."
    weak_at = "Nothing to offer without an open cart."
    concepts = "substitutes vs complements, price band"
    _anchor_label = "alternative to {title} in your cart"
    _explanation_kind = ExplanationKind.ALTERNATIVE

    def __init__(self, price_band: float = 0.3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.price_band = price_band
        self._facts = ItemFacts()

    @property
    def params(self) -> dict[str, Any]:
        return {**super().params, "price_band": self.price_band}

    def _fit(self, data: TrainingData) -> None:
        super()._fit(data)
        self._facts = ItemFacts.from_items(self._items)

    def _anchors(self, user: UserContext) -> list[tuple[ItemId, float]]:
        cart = shopper_state(user, event_weights=self.event_weights).open_cart
        return [(a, self.anchor_decay**rank) for rank, a in enumerate(cart[: self.max_anchors])]

    def _pair_weight(self, anchor: ItemId, other: ItemId) -> float:
        leaf = self._facts.leaf_of
        if anchor not in leaf or leaf.get(other) != leaf[anchor]:
            return 0.0
        pa, pb = self._facts.price_of.get(anchor), self._facts.price_of.get(other)
        if pa and pb and self.price_band > 0:
            return math.exp(-abs(math.log(pb / pa)) / self.price_band)
        return 1.0


class AdjacentCategoryRecommender(Recommender):
    """Popular items in product types that shoppers tend to enter next.

    Training: for each shopper, the order in which they first touched each
    product type; count transitions between consecutive new types (A → B).
    lift(A → B) = P(next new type is B | after A) / P(next new type is B).
    Transitions seen fewer than `min_transitions` times are dropped.

    Serving: for the types the shopper is into (decayed affinity, shares
    summing to 1), score each adjacent type they haven't touched by
    Σ affinity(A)·lift(A → B), and its items by that times the item's share
    of the type's recent popularity.
    """

    name = "adjacent_category"
    exhaustive = False
    summary = (
        "Suggests popular items from product types this shopper hasn't tried, chosen "
        "because shoppers into the same types often move on to them."
    )
    uses = "This shopper's product types, everyone's category journeys"
    good_at = "Measured category expansion for shoppers with some history."
    weak_at = "Mild signal (lifts of 1.5–2.5×); nothing for brand-new visitors."
    concepts = "category transitions, Markov chain, lift, explore/exploit"

    def __init__(
        self,
        type_facet: str | None = "type",
        min_transitions: int = 20,
        max_adjacent: int = 10,
        items_per_type: int = 50,
        half_life_days: float = 14.0,
        event_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.type_facet = type_facet
        self.min_transitions = min_transitions
        self.max_adjacent = max_adjacent
        self.items_per_type = items_per_type
        self.half_life_days = half_life_days
        self.event_weights = dict(event_weights or DEFAULT_EVENT_WEIGHTS)
        self._facts = ItemFacts()
        self._adjacent: dict[str, list[tuple[str, float]]] = {}
        self._top_items: dict[str, list[tuple[ItemId, float]]] = {}

    @property
    def params(self) -> dict[str, Any]:
        return {
            "type_facet": self.type_facet,
            "min_transitions": self.min_transitions,
            "max_adjacent": self.max_adjacent,
            "items_per_type": self.items_per_type,
            "half_life_days": self.half_life_days,
            "event_weights": self.event_weights,
        }

    def _fit(self, data: TrainingData) -> None:
        self._facts = ItemFacts.from_items(self._items, type_facet=self.type_facet)
        type_of = self._facts.type_of
        events = sorted(
            (x for x in data.interactions if self.event_weights.get(x.kind, 0.0) > 0),
            key=lambda x: x.occurred_at,
        )
        # Transitions between consecutive *new* types, per shopper.
        last_new: dict[str, str] = {}
        seen: dict[str, set[str]] = defaultdict(set)
        pairs: dict[tuple[str, str], int] = defaultdict(int)
        for x in events:
            t = type_of.get(x.item_id)
            if t is None or t in seen[x.user_id]:
                continue
            seen[x.user_id].add(t)
            if x.user_id in last_new:
                pairs[(last_new[x.user_id], t)] += 1
            last_new[x.user_id] = t
        total = sum(pairs.values())
        into: dict[str, int] = defaultdict(int)
        out_of: dict[str, int] = defaultdict(int)
        for (a, b), n in pairs.items():
            into[b] += n
            out_of[a] += n
        adjacent: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for (a, b), n in pairs.items():
            if n >= self.min_transitions:
                adjacent[a].append((b, (n / out_of[a]) / (into[b] / total)))
        self._adjacent = {
            a: sorted(bs, key=lambda p: (-p[1], p[0]))[: self.max_adjacent]
            for a, bs in adjacent.items()
        }

        # Recent popularity within each type (like TrendingRecommender).
        as_of = events[-1].occurred_at if events else None
        best: dict[tuple[str, ItemId], float] = defaultdict(float)
        for x in events:
            age = (as_of - x.occurred_at).total_seconds() / 86400 if as_of else 0.0
            w = self.event_weights[x.kind] * 0.5 ** (age / self.half_life_days)
            key = (x.user_id, x.item_id)
            best[key] = max(best[key], w)
        pop: dict[ItemId, float] = defaultdict(float)
        for (_, item_id), w in best.items():
            pop[item_id] += w
        by_type: dict[str, list[tuple[ItemId, float]]] = defaultdict(list)
        for item_id, w in pop.items():
            if item_id in type_of:
                by_type[type_of[item_id]].append((item_id, w))
        self._top_items = {}
        for t, rows in by_type.items():
            rows.sort(key=lambda r: (-r[1], r[0]))
            top = rows[: self.items_per_type]
            self._top_items[t] = [(i, w / top[0][1]) for i, w in top]

    def _affinity(self, user: UserContext) -> dict[str, float]:
        aff = shopper_state(
            user, self._facts, half_life_days=self.half_life_days, event_weights=self.event_weights
        ).type_affinity
        total = sum(aff.values())
        return {t: w / total for t, w in aff.items()} if total else {}

    def _contributions(self, user: UserContext) -> dict[ItemId, dict[str, float]]:
        """candidate item -> {source type A: affinity(A) · lift(A → B) · item share}."""
        aff = self._affinity(user)
        out: dict[ItemId, dict[str, float]] = defaultdict(dict)
        for a, wa in aff.items():
            for b, lift in self._adjacent.get(a, []):
                if b in aff:  # already into it: not expansion
                    continue
                for item_id, share in self._top_items.get(b, []):
                    out[item_id][a] = wa * lift * share
        return out

    def _user_scores(self, user: UserContext) -> dict[ItemId, float]:
        return {i: sum(parts.values()) for i, parts in self._contributions(user).items()}

    def _retrieve(self, user: UserContext, n: int) -> Sequence[ItemId]:
        scores = self._user_scores(user)
        return sorted(scores, key=lambda i: -scores[i])

    def _retrieve_similar(self, item_id: ItemId, n: int) -> Sequence[ItemId]:
        return []

    def _score_user(self, user: UserContext, pool: Sequence[ItemId]) -> np.ndarray:
        scores = self._user_scores(user)
        return np.array([scores.get(i, 0.0) for i in pool], dtype=float)

    def explain(self, user: UserContext, item_id: ItemId) -> Explanation | None:
        parts = self._contributions(user).get(item_id)
        if not parts:
            return None
        source = max(parts, key=lambda a: parts[a])
        anchor = next(
            (
                x.item_id
                for x in reversed(user.events)
                if self._facts.type_of.get(x.item_id) == source
            ),
            None,
        )
        target = self._facts.type_of.get(item_id, "")
        return Explanation(
            ExplanationKind.NEW_FOR_YOU,
            anchor_item_ids=(anchor,) if anchor else (),
            shared_tags=(Tag(self.type_facet or "category", target),),
        )

    def score_breakdown(self, user: UserContext, item_id: ItemId) -> ScoreBreakdown | None:
        parts = self._contributions(user).get(item_id, {})
        target = self._facts.type_of.get(item_id, "?")
        lifts: Mapping[str, float] = {
            a: dict(self._adjacent.get(a, [])).get(target, 0.0) for a in parts
        }
        ranked = sorted(parts.items(), key=lambda p: -p[1])
        return ScoreBreakdown(
            total=sum(parts.values()),
            components=tuple(
                ScoreComponent(f"shoppers into {a} move on to {target} ({lifts[a]:.1f}× lift)", v)
                for a, v in ranked
            ),
            formula="score = Σ your types A: affinity(A) × lift(A → type) × item's share of type",
        )
