"""The journey blender: weights by journey, rank fusion, exploration slots."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from recsys.core.types import (
    ExplanationKind,
    Interaction,
    InteractionKind,
    Item,
    Tag,
    TrainingData,
    UserContext,
)
from recsys.models import JourneyBlender

T0 = datetime(2026, 3, 10, tzinfo=UTC)
V, C, P = InteractionKind.VIEW, InteractionKind.ADD_TO_CART, InteractionKind.PURCHASE


def ev(user, item, kind=V, hours=0.0, session=None):
    return Interaction(user, item, kind, 1, T0 + timedelta(hours=hours), session or f"{user}:0")


@pytest.fixture
def data() -> TrainingData:
    catalog = [
        Item(f"i{n}", f"Item {n}", category="c", tags=frozenset({Tag("type", f"t{n % 4}")}))
        for n in range(12)
    ]
    history = []
    for n in range(40):  # visits that view i0 then i1/i2; i5 is the trending hit
        history += [ev(f"u{n}", "i0", hours=-50 + n), ev(f"u{n}", f"i{1 + n % 2}", hours=-50 + n)]
        history.append(ev(f"u{n}", "i5", C, hours=-2))
    return TrainingData(catalog, history)


def test_journey_picks_the_weights(data):
    blend = JourneyBlender(sources={"recently_viewed": {}, "trending": {}}).fit(data)
    assert blend.journey(UserContext("x")) == "cold start"
    cold = blend.recommend(UserContext("x"), k=3)
    assert cold[0].item_id == "i5"  # only trending can serve a cold start
    assert cold[0].explanation.kind == ExplanationKind.TRENDING
    shopper = UserContext("s", events=(ev("s", "i9"),), session_id="s:0")
    assert blend.journey(shopper) == "new visitor"
    assert blend.recommend(shopper, k=1)[0].item_id == "i9"  # recently viewed leads


def test_order_is_weighted_reciprocal_rank_fusion(data):
    weights = {"new visitor": {"recently_viewed": 1.0, "covisitation": 0.5}}
    blend = JourneyBlender(
        sources={"recently_viewed": {}, "covisitation": {}},
        weights=weights,
        explore_source=None,
        rank_constant=10,
    ).fit(data)
    shopper = UserContext("s", events=(ev("s", "i1"), ev("s", "i0", hours=0.1)), session_id="s:0")
    expected: dict[str, float] = {}
    for source, w in weights["new visitor"].items():
        for rank, rec in enumerate(blend._children[source].recommend(shopper, 50), start=1):
            expected[rec.item_id] = expected.get(rec.item_id, 0.0) + w / (10 + rank)
    recs = blend.recommend(shopper, k=10)
    assert [r.item_id for r in recs] == sorted(expected, key=lambda i: -expected[i])[:10]
    assert [r.score for r in recs] == [1 / p for p in range(1, len(recs) + 1)]
    top = recs[0].item_id
    breakdown = blend.score_breakdown(shopper, top)
    assert sum(c.value for c in breakdown.components) == pytest.approx(breakdown.total) == 1.0


def test_exploration_slots_only_for_listed_journeys(data):
    weights = {
        "browser": {"trending": 1.0},
        "new visitor": {"trending": 1.0},
    }
    blend = JourneyBlender(
        sources={"trending": {}, "adjacent_category": {"min_transitions": 1}},
        weights=weights,
        explore_every=2,
        explore_stages=["browser"],
    ).fit(data)
    browser = UserContext(
        "s", events=(ev("s", "i0", hours=-30, session="s:0"),), session_id="s:1", now=T0
    )
    assert blend.journey(browser) == "browser"
    recs = blend.recommend(browser, k=4)
    explored = [r for r in recs if r.explanation.kind == ExplanationKind.NEW_FOR_YOU]
    assert [recs.index(r) + 1 for r in explored] == [2, 4]  # every 2nd slot
    new = UserContext("n", events=(ev("n", "i0"),), session_id="n:0")
    assert all(r.explanation.kind != ExplanationKind.NEW_FOR_YOU for r in blend.recommend(new, k=4))


def test_prefix_property(data):
    blend = JourneyBlender().fit(data)
    shopper = UserContext("s", events=(ev("s", "i0"), ev("s", "i3", C)), session_id="s:0")
    assert blend.recommend(shopper, k=3) == blend.recommend(shopper, k=8)[:3]


def test_empty_rules_mean_uniform_weights(data):
    uniform = JourneyBlender(weights={}, explore_stages=[])
    assert uniform.weights == {} and "trending" in uniform.default_weights
    assert "adjacent_category" not in uniform.default_weights  # the explorer isn't fused


def test_unknown_source_is_rejected(data):
    with pytest.raises(ValueError, match="unknown sources"):
        JourneyBlender(sources={"nope": {}}).fit(data)
