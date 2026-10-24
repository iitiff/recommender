"""Hard candidate filters: what *may* be recommended, before any model ranks.

Budget, stock and shipping are constraints, not preferences: a $120 bottle is
a bad recommendation for a $20 buyer however well it matches their taste. So
they remove items up front (passed to models via `candidates=`) and every
model obeys them identically.

The result keeps a count at each step, which explainability tools show as a
funnel: most "why wasn't X shown?" questions are answered here, not by a model.
"""

from __future__ import annotations

import csv
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from recsys.core.types import Item, ItemId, Offer


@dataclass(frozen=True)
class CandidateFilter:
    min_price: float | None = None
    max_price: float | None = None
    ship_to: str | None = None  # region code, e.g. "CA"
    require_offer: bool = False  # True once every sellable item has offers
    # Fallback price when an item has no usable offer, e.g. a curated typical price.
    price_attribute: str | None = "price"


@dataclass(frozen=True)
class CandidateResult:
    item_ids: frozenset[ItemId]
    steps: tuple[tuple[str, int], ...]  # (label, items remaining), for funnels
    unknown_price: int  # kept because their price is unknown


def filter_candidates(
    items: Sequence[Item], offers: Sequence[Offer], flt: CandidateFilter
) -> CandidateResult:
    active = [i for i in items if i.is_active]
    steps: list[tuple[str, int]] = [("Catalog", len(active))]
    pool = [i for i in active if i.recommendable]
    steps.append(("Recommendable", len(pool)))

    usable: dict[ItemId, list[Offer]] = defaultdict(list)
    for o in offers:
        if o.in_stock and (flt.ship_to is None or o.ships_to is None or flt.ship_to in o.ships_to):
            usable[o.item_id].append(o)
    has_offers = {o.item_id for o in offers}

    if flt.require_offer or flt.ship_to is not None:
        # Items with offers must have a usable one; items with none at all are
        # dropped only when offers are required.
        pool = [
            i
            for i in pool
            if usable.get(i.item_id) or (i.item_id not in has_offers and not flt.require_offer)
        ]
        steps.append(("Available" + (f" in {flt.ship_to}" if flt.ship_to else ""), len(pool)))

    unknown = 0
    if flt.min_price is not None or flt.max_price is not None:
        kept = []
        for item in pool:
            price = price_of(item, usable.get(item.item_id, []), flt.price_attribute)
            if price is None:
                unknown += 1
                kept.append(item)
            elif (flt.min_price is None or price >= flt.min_price) and (
                flt.max_price is None or price <= flt.max_price
            ):
                kept.append(item)
        pool = kept
        steps.append(("Within budget", len(pool)))

    return CandidateResult(frozenset(i.item_id for i in pool), tuple(steps), unknown)


def price_of(item: Item, offers: Sequence[Offer], attribute: str | None = "price") -> float | None:
    """Cheapest of the given (usable) offers, else the curated price attribute, else None."""
    if offers:
        return min(o.price for o in offers)
    if attribute and attribute in item.attributes:
        return item.attributes[attribute]
    return None


class AvailabilityTimeline:
    """When each item was in or out of stock, for filtering *as of* a moment.

    Offline evaluation must use the stock state at request time, not today's:
    recommending an item that was sold out that day can't have been a hit, and
    filtering by today's stock would leak the future. Items with no record
    before `at` are treated as available (unknown is not the same as sold out).

    CSV: item_id,valid_from,available (one row per change, see
    `recsys dataset prepare`).
    """

    def __init__(self, changes: dict[ItemId, list[tuple[datetime, bool]]]) -> None:
        self._times: dict[ItemId, list[datetime]] = {}
        self._states: dict[ItemId, list[bool]] = {}
        for item_id, rows in changes.items():
            rows = sorted(rows)
            self._times[item_id] = [t for t, _ in rows]
            self._states[item_id] = [s for _, s in rows]

    @classmethod
    def load_csv(cls, path: Path) -> AvailabilityTimeline:
        changes: dict[ItemId, list[tuple[datetime, bool]]] = defaultdict(list)
        with Path(path).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                when = datetime.fromisoformat(row["valid_from"].replace("Z", "+00:00"))
                state = row["available"].strip().lower() in ("true", "1", "yes")
                changes[row["item_id"].strip()].append((when, state))
        return cls(changes)

    def available(self, item_id: ItemId, at: datetime) -> bool | None:
        """True/False as of `at`; None if nothing is known yet."""
        times = self._times.get(item_id)
        if not times:
            return None
        n = bisect_right(times, at)
        return self._states[item_id][n - 1] if n else None

    def at(self, when: datetime) -> InStockAt:
        """A container of items not known to be sold out at `when` (pass as `candidates=`)."""
        return InStockAt(self, when)


@dataclass(frozen=True)
class InStockAt:
    timeline: AvailabilityTimeline
    when: datetime

    def __contains__(self, item_id: object) -> bool:
        return self.timeline.available(str(item_id), self.when) is not False
