"""What we know about a shopper at request time (docs/SHOPPER-PROFILE.md).

`shopper_state(user, facts)` turns the events in a `UserContext` into the
layers of the shopper profile: the current visit, momentum, relationship
stage, and the slower traits (type and brand affinity, price profile). It
is computed from behaviour every time, never stored, and older evidence
fades with a half-life.

`ItemFacts` holds what those features need to know about items (product
type, leaf category, brand, price position within the leaf category). It is
built once from the catalog at fit time.

Concepts: recency-weighted profiles (exponential decay), partial pooling /
empirical Bayes shrinkage (the price profile).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime

from recsys.core.types import InteractionKind, Item, ItemId, UserContext
from recsys.lifecycle import BULK_ITEMS, Stage, lifecycle_stage, orders

# How much each behaviour says about intent. Untuned defaults: carting or
# buying says more than a page view. Override in config.
DEFAULT_EVENT_WEIGHTS: dict[str, float] = {
    InteractionKind.VIEW: 1.0,
    InteractionKind.CLICK: 1.0,
    InteractionKind.ADD_TO_CART: 3.0,
    InteractionKind.PURCHASE: 3.0,
}


@dataclass(frozen=True)
class ItemFacts:
    """Store-level facts about items, for shopper features and sources."""

    type_of: Mapping[ItemId, str] = field(default_factory=dict)  # product type
    leaf_of: Mapping[ItemId, str] = field(default_factory=dict)  # leaf category
    brand_of: Mapping[ItemId, str] = field(default_factory=dict)
    price_of: Mapping[ItemId, float] = field(default_factory=dict)
    # Price percentile within the leaf category: "premium" means the same for
    # kettles and phones ("Price is context, not taste" in CLAUDE.md).
    price_pct: Mapping[ItemId, float] = field(default_factory=dict)

    @classmethod
    def from_items(
        cls,
        items: Iterable[Item],
        *,
        type_facet: str | None = None,
        brand_facet: str = "brand",
        price_attribute: str = "price",
        min_priced: int = 20,
    ) -> ItemFacts:
        """`type_facet` names the tag facet used as product type; without one
        (or for items lacking it) the item's category is the type."""
        type_of: dict[ItemId, str] = {}
        leaf_of: dict[ItemId, str] = {}
        brand_of: dict[ItemId, str] = {}
        price_of: dict[ItemId, float] = {}
        for item in items:
            tags = sorted(item.tags)
            typed = [t.value for t in tags if t.facet == type_facet] if type_facet else []
            if typed or item.category:
                type_of[item.item_id] = typed[0] if typed else item.category
            if item.category:
                leaf_of[item.item_id] = item.category
            brands = [t.value for t in tags if t.facet == brand_facet]
            if brands:
                brand_of[item.item_id] = brands[0]
            price = item.attributes.get(price_attribute)
            if price is not None and price > 0:
                price_of[item.item_id] = float(price)
        by_leaf: dict[str, list[tuple[float, ItemId]]] = defaultdict(list)
        for item_id, price in price_of.items():
            if item_id in leaf_of:
                by_leaf[leaf_of[item_id]].append((price, item_id))
        price_pct: dict[ItemId, float] = {}
        for rows in by_leaf.values():
            if len(rows) < min_priced:
                continue
            rows.sort()
            for rank, (_, item_id) in enumerate(rows, start=1):
                price_pct[item_id] = rank / len(rows)
        return cls(type_of, leaf_of, brand_of, price_of, price_pct)


@dataclass(frozen=True)
class ShopperState:
    """The shopper profile at one moment. Fields map to docs/SHOPPER-PROFILE.md."""

    # Relationship
    stage: Stage
    orders: int
    items_bought: int
    # Current visit
    visit_events: int
    visit_types: int
    anchor_item: ItemId | None  # the latest item they touched
    # Momentum
    hours_since_last_visit: float | None
    hours_since_last_order: float | None
    open_cart: tuple[ItemId, ...]  # carted since the last order, not bought; newest first
    open_cart_age_hours: float | None  # of the newest open-cart item
    bought: tuple[ItemId, ...]  # newest first
    # Traits (decayed by `half_life_days`)
    type_affinity: Mapping[str, float] = field(default_factory=dict, hash=False)
    brand_affinity: Mapping[tuple[str, str], float] = field(default_factory=dict, hash=False)
    price_general: float | None = None
    price_band: Mapping[str, float] = field(default_factory=dict, hash=False)
    price_evidence: float = 0.0  # decayed weight behind price_general

    @property
    def is_bulk(self) -> bool:
        return self.items_bought > BULK_ITEMS


def _hours(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds() / 3600


def shopper_state(
    user: UserContext,
    facts: ItemFacts | None = None,
    *,
    half_life_days: float = 90.0,
    event_weights: Mapping[str, float] | None = None,
    pool_general: float = 5.0,
    pool_type: float = 3.0,
    store_price: float = 0.5,
) -> ShopperState:
    """Compute the shopper profile from `user.events` as of `user.now`.

    Price profile (partial pooling; `pool_*` are pseudo-counts, learned per
    store later):
        general     = (Σ w·pct + pool_general·store_price) / (Σ w + pool_general)
        band[type]  = (Σ_type w·pct + pool_type·general)  / (Σ_type w + pool_type)
    so a type with little evidence leans on the general level, and a shopper
    with little evidence leans on the store average.
    """
    facts = facts or ItemFacts()
    weights = event_weights or DEFAULT_EVENT_WEIGHTS
    events = [x for x in user.events if weights.get(x.kind, 0.0) > 0]
    now = user.now or (events[-1].occurred_at if events else None)
    visit = [x for x in events if user.session_id is not None and x.session_id == user.session_id]
    earlier = [x for x in events if user.session_id is None or x.session_id != user.session_id]
    purchases = [x for x in events if x.kind == InteractionKind.PURCHASE]
    last_order = purchases[-1].occurred_at if purchases else None

    bought_set = {x.item_id for x in purchases}
    open_cart: dict[ItemId, datetime] = {}
    for x in events:
        if x.kind == InteractionKind.ADD_TO_CART and (
            last_order is None or x.occurred_at > last_order
        ):
            open_cart[x.item_id] = x.occurred_at
    for item_id in bought_set:
        open_cart.pop(item_id, None)
    cart_newest = sorted(open_cart, key=lambda i: open_cart[i], reverse=True)

    type_aff: dict[str, float] = defaultdict(float)
    brand_aff: dict[tuple[str, str], float] = defaultdict(float)
    price_sum: dict[str, float] = defaultdict(float)
    price_w: dict[str, float] = defaultdict(float)
    for x in events:
        age_days = _hours(now, x.occurred_at) / 24 if now else 0.0
        w = weights[x.kind] * 0.5 ** (max(age_days, 0.0) / half_life_days)
        kind_of = facts.type_of.get(x.item_id)
        if kind_of is not None:
            type_aff[kind_of] += w
            brand = facts.brand_of.get(x.item_id)
            if brand is not None:
                brand_aff[(kind_of, brand)] += w
        pct = facts.price_pct.get(x.item_id)
        if pct is not None:
            key = kind_of or ""
            price_sum[key] += w * pct
            price_w[key] += w

    evidence = sum(price_w.values())
    general = None
    band: dict[str, float] = {}
    if evidence > 0 or facts.price_pct:
        general = (sum(price_sum.values()) + pool_general * store_price) / (evidence + pool_general)
        band = {
            t: (price_sum[t] + pool_type * general) / (price_w[t] + pool_type) for t in price_w if t
        }

    return ShopperState(
        stage=lifecycle_stage(events, user.session_id),
        orders=orders(events),
        items_bought=len(purchases),
        visit_events=len(visit),
        visit_types=len({facts.type_of[x.item_id] for x in visit if x.item_id in facts.type_of}),
        anchor_item=events[-1].item_id if events else None,
        hours_since_last_visit=_hours(
            visit[0].occurred_at if visit else now, earlier[-1].occurred_at
        )
        if earlier and now
        else None,
        hours_since_last_order=_hours(now, last_order) if last_order and now else None,
        open_cart=tuple(cart_newest),
        open_cart_age_hours=_hours(now, open_cart[cart_newest[0]]) if cart_newest and now else None,
        bought=tuple(dict.fromkeys(x.item_id for x in reversed(purchases))),
        type_affinity=dict(type_aff),
        brand_affinity=dict(brand_aff),
        price_general=general,
        price_band=band,
        price_evidence=evidence,
    )
