from __future__ import annotations

from datetime import UTC, datetime, timedelta

from recsys.core.types import Interaction, InteractionKind, Item, Tag
from recsys.eval.split import build_context, leave_last_out, temporal_split

T0 = datetime(2026, 1, 1, tzinfo=UTC)
ITEMS = [
    Item("napa-1", "Napa 1"),
    Item("napa-2", "Napa 2"),
    Item("napa-3", "Napa 3"),
    Item("paso-1", "Paso 1", recommendable=False),
]


def x(user, item, kind="rating", value=1, day=0):
    return Interaction(user, item, InteractionKind(kind), value, T0 + timedelta(days=day))


def test_build_context_keeps_latest_state():
    ctx = build_context(
        "u",
        [
            x("u", "napa-1", value=1, day=0),
            x("u", "napa-1", value=-1, day=1),  # changed their mind
            x("u", "napa-2", "save", 1, day=0),
            x("u", "napa-2", "save", 0, day=2),  # un-saved
            x("u", "napa-3", "not_interested", day=3),
        ],
    )
    assert ctx.liked == () and ctx.disliked == ("napa-1",)
    assert ctx.saved == () and ctx.hidden == ("napa-3",)


def test_temporal_split_has_no_future_leakage_and_sane_targets():
    data = [
        x("u1", "paso-1", day=0),  # quiz-only like: input, never a target
        x("u1", "napa-1", day=1),
        x("u1", "napa-1", day=20),  # re-rated after cutoff: already excluded, not a target
        x("u1", "napa-2", day=21),
        x("u1", "paso-1", day=22),  # non-recommendable: not a target
        x("u1", "napa-3", "click", day=23),  # click alone isn't a positive
        x("u2", "napa-2", day=25),  # cold user: no training history
    ]
    split = temporal_split(ITEMS, data, cutoff=T0 + timedelta(days=10))

    assert all(i.occurred_at < T0 + timedelta(days=10) for i in split.train.interactions)
    assert split.targets == {"u1": frozenset({"napa-2"})}
    assert split.contexts["u1"].liked == ("napa-1", "paso-1")
    assert split.info["users_skipped_cold"] == 1
    assert split.info["users_evaluated"] == 1


def test_quiz_profile_makes_cold_user_evaluable():
    data = [x("u1", "napa-1", day=0), x("u2", "napa-2", day=25)]
    profile = {"u2": {Tag("varietal", "zinfandel"): 1.0}}
    split = temporal_split(ITEMS, data, cutoff=T0 + timedelta(days=10), profiles=profile)
    assert "u2" in split.targets
    assert split.contexts["u2"].tag_affinities == profile["u2"]


def test_leave_last_out_holds_out_last_positive_and_later_events():
    data = [
        x("u1", "napa-1", day=0),
        x("u1", "napa-2", day=1),
        x("u1", "napa-3", "click", day=2),  # after the held-out like: test too
        x("u2", "napa-1", day=0),  # only one positive: stays in train
    ]
    split = leave_last_out(ITEMS, data, n=1)
    assert split.targets == {"u1": frozenset({"napa-2"})}
    train_keys = {(i.user_id, i.item_id) for i in split.train.interactions}
    assert ("u1", "napa-1") in train_keys and ("u2", "napa-1") in train_keys
    assert ("u1", "napa-3") not in train_keys


def test_fingerprint_changes_with_data():
    a = temporal_split(ITEMS, [x("u1", "napa-1", day=0), x("u1", "napa-2", day=5)])
    b = temporal_split(ITEMS, [x("u1", "napa-1", day=0), x("u1", "napa-3", day=5)])
    assert a.info["fingerprint"] != b.info["fingerprint"]
