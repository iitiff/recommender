"""Lifecycle stage of a shopper at a moment in time, derived from their events.

The stage is *computed* from behaviour every time it's needed, never stored
as a fact: a stored "loyal" flag goes stale the day the customer lapses. See
docs/LIFECYCLE.md for why stages matter and what each one needs.

Definitions (shared with the lifecycle analyses in docs/experiments/):
- an **order** is a calendar day (UTC) with at least one purchase, so a
  shopper who comes back an hour later for one more item hasn't "reordered";
- **bulk buyers** (more than `BULK_ITEMS` purchased items) are likely
  resellers or businesses, kept apart so they don't pass as loyal consumers.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from recsys.core.types import Interaction, InteractionKind

BULK_ITEMS = 10


class Stage(StrEnum):
    NEW_VISITOR = "new visitor"  # nothing before this visit
    RETURNING = "returning, no order"
    FIRST_ORDER = "1 order"
    REPEAT = "2+ orders"
    BULK = "bulk buyer"


def orders(events: Sequence[Interaction]) -> int:
    return len({x.occurred_at.date() for x in events if x.kind == InteractionKind.PURCHASE})


def lifecycle_stage(events: Sequence[Interaction], session_id: str | None = None) -> Stage:
    """Stage given everything the shopper did *before* now (`events`).

    `session_id` is the current visit: events in it don't make someone a
    returning visitor.
    """
    purchased = sum(1 for x in events if x.kind == InteractionKind.PURCHASE)
    if purchased > BULK_ITEMS:
        return Stage.BULK
    n = orders(events)
    if n >= 2:
        return Stage.REPEAT
    if n == 1:
        return Stage.FIRST_ORDER
    if any(x.session_id != session_id or session_id is None for x in events):
        return Stage.RETURNING
    return Stage.NEW_VISITOR
