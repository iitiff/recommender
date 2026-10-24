"""Train/test splits for offline evaluation.

The split decides what the evaluation means, so the rules are explicit:

1. **Never train on the future.** A random split lets the model see a user's
   later behaviour (and other users' later trends) while predicting earlier
   behaviour, which inflates every metric. Splits here are time-based.
2. **Targets are positives the model could actually recommend:** liked,
   saved, or affiliate-clicked items that are recommendable and not already
   excluded by the user's training history (the base Recommender never
   re-recommends rated or hidden items, so those could never be hits).
3. **Inputs can be anything.** Quiz-only (non-recommendable) items still go
   into training data and the user's context; they're just never targets.

Three strategies:
- `temporal_split`: one global cutoff time. The most realistic ("train on
  everything until Monday, predict the rest of the week"), but users who only
  appear after the cutoff are cold-start and are skipped (reported in counts).
- `leave_last_out`: hold out each user's last N positives. Uses data more
  efficiently when it's scarce, at the cost of a small leak: other users'
  later interactions can be in training.
- `next_event_split`: e-commerce, session-aware. Models are trained up to a
  cutoff; after it, every add-to-cart becomes one evaluation *case*: "given
  everything this shopper did up to this moment, including earlier in this
  visit, which item will they cart next?" Each case is tagged with the
  shopper's lifecycle stage and what kind of item the target was, so results
  can be split by journey.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable, Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from recsys.core.types import (
    Interaction,
    InteractionKind,
    Item,
    ItemId,
    Tag,
    TrainingData,
    UserContext,
    UserId,
)
from recsys.lifecycle import lifecycle_stage

# Behaviour a context carries in `events` (explicit feedback has its own fields).
BEHAVIOUR = frozenset(
    {
        InteractionKind.VIEW,
        InteractionKind.CLICK,
        InteractionKind.ADD_TO_CART,
        InteractionKind.PURCHASE,
    }
)


def is_positive(x: Interaction) -> bool:
    """What counts as 'the user wanted this'. Views and clicks alone are too noisy."""
    if x.kind == InteractionKind.RATING:
        return x.value > 0
    return x.kind in (
        InteractionKind.SAVE,
        InteractionKind.AFFILIATE_CLICK,
        InteractionKind.ADD_TO_CART,
        InteractionKind.PURCHASE,
    )


@dataclass(frozen=True)
class Split:
    """Evaluation cases. A case is a user (temporal, leave-last-out) or one
    request in a user's journey (next-event); either way, keyed by a string."""

    train: TrainingData
    contexts: dict[UserId, UserContext]  # what each case looks like at serve time
    targets: dict[UserId, frozenset[ItemId]]  # held-out positives to find
    info: dict[str, Any] = field(default_factory=dict)
    # Optional, per case: hard filters as of request time (e.g. in stock then).
    candidates: dict[UserId, Container[ItemId]] = field(default_factory=dict)
    # Optional, per case: {dimension: value}, e.g. {"stage": "1 order"}.
    segments: dict[UserId, dict[str, str]] = field(default_factory=dict)
    # Optional: item -> category, and per case the categories the user already
    # knows. Enables the category-novelty metric.
    item_groups: Mapping[ItemId, str] = field(default_factory=dict)
    known_groups: dict[UserId, frozenset[str]] = field(default_factory=dict)


def build_context(
    user_id: UserId,
    history: Iterable[Interaction],
    tag_affinities: dict[Tag, float] | None = None,
    *,
    max_events: int = 100,
    session_id: str | None = None,
    now: datetime | None = None,
) -> UserContext:
    """Rebuild a UserContext from interactions, keeping each item's latest state.

    The web app builds contexts from the `feedback` table; this mirrors that
    logic so offline and online users look the same to a model. Behaviour
    (views, carts, purchases) is kept as the latest `max_events` events.
    """
    rating: dict[ItemId, float] = {}
    saved: set[ItemId] = set()
    hidden: set[ItemId] = set()
    ordered = sorted(history, key=lambda x: x.occurred_at)
    for x in ordered:
        if x.kind == InteractionKind.RATING:
            rating[x.item_id] = x.value
        elif x.kind == InteractionKind.SAVE:
            (saved.add if x.value > 0 else saved.discard)(x.item_id)
        elif x.kind == InteractionKind.NOT_INTERESTED:
            hidden.add(x.item_id)
    return UserContext(
        user_id=user_id,
        tag_affinities=tag_affinities or {},
        liked=tuple(sorted(i for i, v in rating.items() if v > 0)),
        disliked=tuple(sorted(i for i, v in rating.items() if v < 0)),
        saved=tuple(sorted(saved)),
        hidden=tuple(sorted(hidden)),
        events=tuple([x for x in ordered if x.kind in BEHAVIOUR][-max_events:])
        if max_events
        else (),
        session_id=session_id,
        now=now,
    )


def _finish(
    items: Sequence[Item],
    train: list[Interaction],
    test: list[Interaction],
    profiles: dict[UserId, dict[Tag, float]] | None,
    info: dict[str, Any],
) -> Split:
    recommendable = {i.item_id for i in items if i.is_active and i.recommendable}
    by_user_train: dict[UserId, list[Interaction]] = defaultdict(list)
    for x in train:
        by_user_train[x.user_id].append(x)
    by_user_test: dict[UserId, list[Interaction]] = defaultdict(list)
    for x in test:
        by_user_test[x.user_id].append(x)

    contexts: dict[UserId, UserContext] = {}
    targets: dict[UserId, frozenset[ItemId]] = {}
    cold = 0
    for user_id in sorted(by_user_test):
        history = by_user_train.get(user_id, [])
        profile = (profiles or {}).get(user_id)
        if not history and not profile:
            cold += 1  # nothing to personalize from; evaluate cold start separately
            continue
        context = build_context(user_id, history, profile)
        wanted = frozenset(
            x.item_id
            for x in by_user_test[user_id]
            if is_positive(x) and x.item_id in recommendable and x.item_id not in context.excluded
        )
        if wanted:
            contexts[user_id] = context
            targets[user_id] = wanted

    info = {
        **info,
        "train_interactions": len(train),
        "test_interactions": len(test),
        "users_evaluated": len(targets),
        "users_skipped_cold": cold,
        "recommendable_items": len(recommendable),
        "fingerprint": fingerprint(items, train, test),
    }
    return Split(TrainingData(items=items, interactions=train), contexts, targets, info)


def temporal_split(
    items: Sequence[Item],
    interactions: Sequence[Interaction],
    *,
    test_fraction: float = 0.2,
    cutoff: datetime | None = None,
    profiles: dict[UserId, dict[Tag, float]] | None = None,
) -> Split:
    """Global time cutoff; by default the latest `test_fraction` of interactions are test."""
    if not interactions:
        raise ValueError("no interactions to split")
    if cutoff is None:
        times = sorted(x.occurred_at for x in interactions)
        cutoff = times[min(len(times) - 1, int(len(times) * (1 - test_fraction)))]
    train = [x for x in interactions if x.occurred_at < cutoff]
    test = [x for x in interactions if x.occurred_at >= cutoff]
    return _finish(
        items, train, test, profiles, {"strategy": "temporal", "cutoff": cutoff.isoformat()}
    )


def leave_last_out(
    items: Sequence[Item],
    interactions: Sequence[Interaction],
    *,
    n: int = 1,
    profiles: dict[UserId, dict[Tag, float]] | None = None,
    keep: Callable[[Interaction], bool] = is_positive,
) -> Split:
    """Hold out each user's last `n` positives on recommendable items, plus
    everything that user did after them (so the user's own future never leaks)."""
    recommendable = {i.item_id for i in items if i.is_active and i.recommendable}
    by_user: dict[UserId, list[Interaction]] = defaultdict(list)
    for x in interactions:
        by_user[x.user_id].append(x)

    train: list[Interaction] = []
    test: list[Interaction] = []
    for history in by_user.values():
        history.sort(key=lambda x: x.occurred_at)
        candidates = [x for x in history if keep(x) and x.item_id in recommendable]
        if len(candidates) <= n:  # too little history to hold anything out
            train.extend(history)
            continue
        boundary = candidates[-n].occurred_at
        train.extend(x for x in history if x.occurred_at < boundary)
        test.extend(x for x in history if x.occurred_at >= boundary)
    return _finish(items, train, test, profiles, {"strategy": "leave_last_out", "n": n})


def groups_from_facet(items: Iterable[Item], facet: str) -> dict[ItemId, str]:
    """item -> its value for a tag facet (e.g. a top-level "department"), for
    category-novelty metrics. Items without that facet are left out."""
    out: dict[ItemId, str] = {}
    for item in items:
        values = sorted(t.value for t in item.tags if t.facet == facet)
        if values:
            out[item.item_id] = values[0]
    return out


def next_action(
    history: Sequence[Interaction],
    session_id: str | None,
    item_id: ItemId,
    groups: Mapping[ItemId, str],
) -> str:
    """What kind of choice the shopper made, relative to what they did before."""
    if not history:
        return "no history (cold start)"
    if any(x.item_id == item_id and x.kind == InteractionKind.PURCHASE for x in history):
        return "re-buy"
    seen = [x for x in history if x.item_id == item_id]
    if any(x.session_id == session_id for x in seen):
        return "seen this visit"
    if seen:
        return "seen in an earlier visit"
    group = groups.get(item_id)
    if group is None:
        return "new item, category unknown"
    if group in {groups.get(x.item_id) for x in history}:
        return "new item, browsed category"
    return "new item, new category"


def vs_past_orders(
    history: Sequence[Interaction], item_id: ItemId, groups: Mapping[ItemId, str]
) -> str:
    """For buyers: is this a category they've bought from? (the expansion question)"""
    bought = {groups.get(x.item_id) for x in history if x.kind == InteractionKind.PURCHASE}
    if not bought:
        return "no past order"
    group = groups.get(item_id)
    if group is None:
        return "category unknown"
    return "category of a past order" if group in bought else "new category vs past orders"


def next_event_split(
    items: Sequence[Item],
    interactions: Sequence[Interaction],
    *,
    test_fraction: float = 0.2,
    cutoff: datetime | None = None,
    target_kind: InteractionKind = InteractionKind.ADD_TO_CART,
    item_groups: Mapping[ItemId, str] | None = None,
    availability: Any = None,
    max_events: int = 100,
) -> Split:
    """Train before `cutoff`; every later `target_kind` event is a case.

    - The context holds the shopper's events strictly before the target,
      including test-period events (live serving sees those too), but models
      are only *fitted* on data before the cutoff. Retraining as time moves on
      is what a day-by-day replay adds.
    - The same item carted twice in one visit counts once.
    - Shoppers with no history at all are kept: they are the cold-start cases,
      and a model that can't serve them should score zero there.
    - `availability` (an AvailabilityTimeline) restricts each case to items in
      stock at request time; the share of targets the timeline itself marks
      as sold out is reported, as a data-quality check.
    """
    if not interactions:
        raise ValueError("no interactions to split")
    ordered = sorted(interactions, key=lambda x: x.occurred_at)
    if cutoff is None:
        cutoff = ordered[min(len(ordered) - 1, int(len(ordered) * (1 - test_fraction)))].occurred_at
    train = [x for x in ordered if x.occurred_at < cutoff]
    test = [x for x in ordered if x.occurred_at >= cutoff]
    groups = dict(item_groups or {})
    recommendable = {i.item_id for i in items if i.is_active and i.recommendable}

    by_user: dict[UserId, list[Interaction]] = defaultdict(list)
    for x in ordered:
        by_user[x.user_id].append(x)
    target_users = sorted({x.user_id for x in test if x.kind == target_kind})

    contexts: dict[UserId, UserContext] = {}
    targets: dict[UserId, frozenset[ItemId]] = {}
    candidates: dict[UserId, Container[ItemId]] = {}
    segments: dict[UserId, dict[str, str]] = {}
    known_groups: dict[UserId, frozenset[str]] = {}
    out_of_stock = 0
    for user_id in target_users:
        history = by_user[user_id]
        times = [x.occurred_at for x in history]
        done: set[tuple[str | None, ItemId]] = set()
        for x in history:
            if x.occurred_at < cutoff or x.kind != target_kind or x.item_id not in recommendable:
                continue
            if (x.session_id, x.item_id) in done:
                continue
            done.add((x.session_id, x.item_id))
            before = history[: bisect_left(times, x.occurred_at)]
            context = build_context(
                user_id, before, max_events=max_events, session_id=x.session_id, now=x.occurred_at
            )
            if x.item_id in context.excluded:
                continue
            case = f"{user_id}#{len(done)}"
            contexts[case] = context
            targets[case] = frozenset({x.item_id})
            segments[case] = {
                "stage": str(lifecycle_stage(before, x.session_id)),
                "next action": next_action(before, x.session_id, x.item_id, groups),
                "vs past orders": vs_past_orders(before, x.item_id, groups),
            }
            if groups:
                known_groups[case] = frozenset(
                    g for g in (groups.get(y.item_id) for y in before) if g is not None
                )
            if availability is not None:
                candidates[case] = availability.at(x.occurred_at)
                out_of_stock += availability.available(x.item_id, x.occurred_at) is False

    info = {
        "strategy": "next_event",
        "unit": "requests",
        "target_kind": str(target_kind),
        "cutoff": cutoff.isoformat(),
        "train_interactions": len(train),
        "test_interactions": len(test),
        "users_evaluated": len(targets),
        "users_skipped_cold": 0,
        "shoppers": len({c.user_id for c in contexts.values()}),
        "recommendable_items": len(recommendable),
        "stock_filter": availability is not None,
        "targets_marked_sold_out": out_of_stock,
        "fingerprint": fingerprint(items, train, test),
    }
    return Split(
        TrainingData(items=items, interactions=train),
        contexts,
        targets,
        info,
        candidates=candidates,
        segments=segments,
        item_groups=groups,
        known_groups=known_groups,
    )


def fingerprint(
    items: Sequence[Item], train: Sequence[Interaction], test: Sequence[Interaction]
) -> str:
    """Short hash of the exact data evaluated, so reports are comparable and reproducible."""
    h = hashlib.sha256()
    for item in sorted(items, key=lambda i: i.item_id):
        h.update(f"{item.item_id}|{item.recommendable}|{item.is_active}\n".encode())
    for label, rows in (("train", train), ("test", test)):
        h.update(label.encode())
        for x in sorted(rows, key=lambda x: (x.occurred_at, x.user_id, x.item_id, x.kind)):
            h.update(f"{x.user_id}|{x.item_id}|{x.kind}|{x.value}|{x.occurred_at}\n".encode())
    return h.hexdigest()[:12]


def train_popularity(train: TrainingData) -> dict[ItemId, float]:
    """Share of training users with a positive interaction on each item (for ARP)."""
    users = {x.user_id for x in train.interactions}
    if not users:
        return {}
    positive: dict[ItemId, set[UserId]] = defaultdict(set)
    for x in train.interactions:
        if is_positive(x):
            positive[x.item_id].add(x.user_id)
    return {item: len(u) / len(users) for item, u in positive.items()}
