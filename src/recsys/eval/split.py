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

Two strategies:
- `temporal_split`: one global cutoff time. The most realistic ("train on
  everything until Monday, predict the rest of the week"), but users who only
  appear after the cutoff are cold-start and are skipped (reported in counts).
- `leave_last_out`: hold out each user's last N positives. Uses data more
  efficiently when it's scarce, at the cost of a small leak: other users'
  later interactions can be in training.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
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


def is_positive(x: Interaction) -> bool:
    """What counts as 'the user wanted this'. Clicks alone are too noisy."""
    if x.kind == InteractionKind.RATING:
        return x.value > 0
    return x.kind in (InteractionKind.SAVE, InteractionKind.AFFILIATE_CLICK)


@dataclass(frozen=True)
class Split:
    train: TrainingData
    contexts: dict[UserId, UserContext]  # what each evaluated user looks like at serve time
    targets: dict[UserId, frozenset[ItemId]]  # held-out positives to find
    info: dict[str, Any] = field(default_factory=dict)


def build_context(
    user_id: UserId,
    history: Iterable[Interaction],
    tag_affinities: dict[Tag, float] | None = None,
) -> UserContext:
    """Rebuild a UserContext from interactions, keeping each item's latest state.

    The web app builds contexts from the `feedback` table; this mirrors that
    logic so offline and online users look the same to a model.
    """
    rating: dict[ItemId, float] = {}
    saved: set[ItemId] = set()
    hidden: set[ItemId] = set()
    for x in sorted(history, key=lambda x: x.occurred_at):
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
