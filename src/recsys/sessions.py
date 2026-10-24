"""Split each user's event stream into sessions (visits).

Most e-commerce logs record visitors and timestamps but not visits, so a
session is inferred: a new session starts when a user has been inactive for
longer than `gap`. 30 minutes is the common web-analytics convention; it's a
parameter because the right value depends on the store.

Why sessions matter: shopping intent is short-lived. "Viewed three running
shoes in the last ten minutes" predicts the next click far better than a
year-old purchase history, and re-engagement triggers ("left with items in
the cart") are defined per session.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from datetime import timedelta

from recsys.core.types import Interaction

DEFAULT_GAP = timedelta(minutes=30)


def sessionize(
    interactions: Iterable[Interaction], gap: timedelta = DEFAULT_GAP
) -> list[Interaction]:
    """Return the interactions sorted by (user, time) with `session_id` set to
    "<user_id>:<n>", n counting that user's sessions from 0."""
    ordered = sorted(interactions, key=lambda x: (x.user_id, x.occurred_at))
    out: list[Interaction] = []
    user, last, n = None, None, -1
    for x in ordered:
        if x.user_id != user or last is None or x.occurred_at - last > gap:
            n = 0 if x.user_id != user else n + 1
            user = x.user_id
        last = x.occurred_at
        out.append(replace(x, session_id=f"{x.user_id}:{n}"))
    return out
