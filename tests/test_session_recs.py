"""Session-based sources, lifecycle stages, stock timeline and next-event evaluation."""

from __future__ import annotations

import random
from collections import defaultdict
from datetime import UTC, datetime, timedelta

import pytest

from recsys.candidates import AvailabilityTimeline
from recsys.core.types import (
    ExplanationKind,
    Interaction,
    InteractionKind,
    Item,
    Tag,
    TrainingData,
    UserContext,
)
from recsys.eval import evaluate, to_markdown
from recsys.eval.split import groups_from_facet, next_event_split
from recsys.lifecycle import Stage, lifecycle_stage
from recsys.models import (
    CoVisitationRecommender,
    PopularityRecommender,
    RecentlyViewedRecommender,
    TrendingRecommender,
)
from recsys.models.session import DEFAULT_EVENT_WEIGHTS

T0 = datetime(2026, 3, 1, tzinfo=UTC)
V, C, P = InteractionKind.VIEW, InteractionKind.ADD_TO_CART, InteractionKind.PURCHASE


def ev(user, item, kind=V, minutes=0.0, session=None):
    return Interaction(user, item, kind, 1, T0 + timedelta(minutes=minutes), session or f"{user}:0")


def items(*ids, dept=None):
    return [
        Item(i, f"Item {i}", tags=frozenset({Tag("department", (dept or {}).get(i, "d0"))}))
        for i in ids
    ]


# -- co-visitation -------------------------------------------------------------


def test_covisitation_confidence_counts_each_visit_once():
    data = TrainingData(
        items=items("a", "b", "c", "d"),
        interactions=[
            # visit 1: a, b, then carts c
            ev("u1", "a"), ev("u1", "b", minutes=1), ev("u1", "c", C, minutes=2),
            # visit 2: a, b, a, b (repeats count once)
            ev("u2", "a"), ev("u2", "b", minutes=1), ev("u2", "a", minutes=2),
            ev("u2", "b", minutes=3),
            # visit 3: a, d
            ev("u3", "a"), ev("u3", "d", minutes=1),
        ],
    )  # fmt: skip
    model = CoVisitationRecommender().fit(data)
    # Visits with a: 3. b: view in 2 of them (weight 1 each); c: a cart (weight 3) in 1.
    assert model.neighbors("a") == [
        ("c", pytest.approx(3 / 3)),
        ("b", pytest.approx(2 / 3)),
        ("d", pytest.approx(1 / 3)),
    ]
    assert dict(model.neighbors("c")) == {"a": pytest.approx(1.0), "b": pytest.approx(1.0)}
    assert CoVisitationRecommender(max_neighbors=1).fit(data).neighbors("a") == [("c", 1.0)]


def test_covisitation_matches_a_plain_python_reference():
    rng = random.Random(7)
    catalog = [f"i{n}" for n in range(40)]
    interactions = []
    for v in range(300):
        for step in range(rng.randint(1, 8)):
            kind = rng.choice([V, V, V, C, P])
            interactions.append(ev(f"u{v}", rng.choice(catalog), kind, step, f"u{v}:0"))
    model = CoVisitationRecommender(max_neighbors=1000, max_session_items=1000)
    model.fit(TrainingData(items(*catalog), interactions))

    weights = DEFAULT_EVENT_WEIGHTS
    best: dict[str, dict[str, float]] = defaultdict(dict)
    for x in interactions:
        b = best[x.session_id]
        b[x.item_id] = max(b.get(x.item_id, 0.0), weights[x.kind])
    pair: dict[tuple[str, str], float] = defaultdict(float)
    visits_with: dict[str, int] = defaultdict(int)
    for visit in best.values():
        for a in visit:
            visits_with[a] += 1
            for b, w in visit.items():
                if a != b:
                    pair[a, b] += w
    for a in catalog:
        expected = {b: w / visits_with[a] for (src, b), w in pair.items() if src == a}
        assert dict(model.neighbors(a)) == pytest.approx(expected)


def test_covisitation_weights_recent_anchors_and_explains():
    data = TrainingData(
        items=items("a", "b", "x", "y"),
        interactions=[
            ev("u1", "a"), ev("u1", "x", minutes=1),
            ev("u2", "b"), ev("u2", "y", minutes=1),
        ],
    )  # fmt: skip
    model = CoVisitationRecommender().fit(data)
    shopper = UserContext("s", events=(ev("s", "a"), ev("s", "b", minutes=1)), session_id="s:0")
    recs = model.recommend(shopper, k=4)
    assert [r.item_id for r in recs][:2] == ["y", "x"]  # b is the newer view
    assert recs[0].explanation.kind == ExplanationKind.ALSO_VIEWED
    assert recs[0].explanation.anchor_item_ids == ("b",)
    assert model.recommend(UserContext("new"), k=4) == []  # no evidence, no padding


# -- recently viewed and trending -----------------------------------------------


def test_recently_viewed_ranks_by_recency_and_intent():
    model = RecentlyViewedRecommender(decay=0.8).fit(TrainingData(items("a", "b", "c")))
    shopper = UserContext("s", events=(ev("s", "a"), ev("s", "b", C, 1), ev("s", "c", minutes=2)))
    # c: view (1) at rank 0 = 1.0; b: cart (3) at rank 1 = 2.4; a: 1 × 0.8² = 0.64
    assert [(r.item_id, round(r.score, 2)) for r in model.recommend(shopper, k=5)] == [
        ("b", 2.4),
        ("c", 1.0),
        ("a", 0.64),
    ]


def test_trending_fades_old_activity_unlike_most_popular():
    old = [ev(f"o{n}", "a", minutes=-14 * 24 * 60) for n in range(4)]  # 4 shoppers, 2 weeks ago
    new = [ev(f"n{n}", "b") for n in range(2)]  # 2 shoppers, now
    data = TrainingData(items("a", "b"), [*old, *new])
    trending = TrendingRecommender(half_life_days=7).fit(data)
    assert [r.item_id for r in trending.recommend(UserContext(), k=2)] == ["b", "a"]
    assert trending._scores["a"] == pytest.approx(4 * 0.25)
    popular = PopularityRecommender().fit(data)
    assert [r.item_id for r in popular.recommend(UserContext(), k=2)] == ["a", "b"]


def test_popularity_retrieval_depth_still_respects_filters():
    catalog = [f"i{n}" for n in range(30)]
    data = TrainingData(items(*catalog), [ev(f"u{n}", f"i{n}", C) for n in range(30)])
    model = PopularityRecommender(retrieval_depth=5).fit(data)
    assert len(model.recommend(UserContext(), k=3, candidates={"i20", "i21"})) <= 2


# -- lifecycle, stock ------------------------------------------------------------


def test_lifecycle_stage():
    assert lifecycle_stage([], "s:0") == Stage.NEW_VISITOR
    assert lifecycle_stage([ev("s", "a", session="s:0")], "s:0") == Stage.NEW_VISITOR
    assert lifecycle_stage([ev("s", "a", session="s:0")], "s:1") == Stage.RETURNING
    one = [ev("s", "a", P), ev("s", "b", P, minutes=60)]  # same day: one order
    assert lifecycle_stage(one, "s:9") == Stage.FIRST_ORDER
    two = [*one, ev("s", "c", P, minutes=2 * 24 * 60)]
    assert lifecycle_stage(two, "s:9") == Stage.REPEAT
    bulk = [ev("s", f"i{n}", P, minutes=n) for n in range(11)]
    assert lifecycle_stage(bulk, "s:9") == Stage.BULK


def test_availability_timeline(tmp_path):
    path = tmp_path / "availability.csv"
    path.write_text(
        "item_id,valid_from,available\n"
        "a,2026-03-01T00:00:00+00:00,true\n"
        "a,2026-03-05T00:00:00+00:00,false\n"
    )
    timeline = AvailabilityTimeline.load_csv(path)
    assert timeline.available("a", T0 - timedelta(days=1)) is None  # nothing known yet
    assert timeline.available("a", T0 + timedelta(days=1)) is True
    assert timeline.available("a", T0 + timedelta(days=5)) is False
    in_stock = timeline.at(T0 + timedelta(days=6))
    assert "a" not in in_stock and "unknown-item" in in_stock


# -- next-event evaluation ----------------------------------------------------------


@pytest.fixture
def shop():
    dept = {"a": "home", "b": "home", "c": "home", "z": "garden"}
    catalog = items("a", "b", "c", "z", dept=dept)
    day = 24 * 60
    interactions = [
        # Training period: lots of visits where a and b go together.
        *[ev(f"t{n}", "a", minutes=-day + n) for n in range(5)],
        *[ev(f"t{n}", "b", C, minutes=-day + n + 0.5) for n in range(5)],
        # Test period. Shopper 1, first visit: views a, carts b, carts b again.
        ev("s1", "a", minutes=10, session="s1:0"),
        ev("s1", "b", C, minutes=11, session="s1:0"),
        ev("s1", "b", C, minutes=12, session="s1:0"),  # same visit: not a new case
        ev("s1", "b", P, minutes=13, session="s1:0"),
        # Shopper 1 returns two days later and carts something from a new department.
        ev("s1", "z", C, minutes=2 * day, session="s1:1"),
        # Shopper 2 carts on their very first event (cold start).
        ev("s2", "c", C, minutes=20, session="s2:0"),
    ]
    return catalog, interactions


def test_next_event_split_cases_and_segments(shop):
    catalog, interactions = shop
    split = next_event_split(
        catalog,
        interactions,
        cutoff=T0,
        item_groups=groups_from_facet(catalog, "department"),
    )
    assert len(split.train.interactions) == 10
    cases = sorted(split.targets.items(), key=lambda kv: split.contexts[kv[0]].now)
    assert [sorted(t)[0] for _, t in cases] == ["b", "c", "z"]
    b_case, c_case, z_case = (case for case, _ in cases)
    # The context is exactly what happened before the cart, nothing after.
    assert [x.item_id for x in split.contexts[b_case].events] == ["a"]
    assert split.segments[b_case] == {
        "stage": "new visitor",
        "next action": "new item, browsed category",
        "vs past orders": "no past order",
    }
    assert split.contexts[c_case].events == ()
    assert split.segments[c_case]["stage"] == "new visitor"
    assert split.segments[c_case]["next action"] == "no history (cold start)"
    assert split.segments[z_case] == {
        "stage": "1 order",
        "next action": "new item, new category",
        "vs past orders": "new category vs past orders",
    }
    assert split.known_groups[z_case] == frozenset({"home"})


def test_next_event_evaluation_reports_segments(shop, tmp_path):
    catalog, interactions = shop
    (tmp_path / "stock.csv").write_text(f"item_id,valid_from,available\nz,{T0.isoformat()},false\n")
    split = next_event_split(
        catalog,
        interactions,
        cutoff=T0,
        item_groups=groups_from_facet(catalog, "department"),
        availability=AvailabilityTimeline.load_csv(tmp_path / "stock.csv"),
    )
    assert split.info["targets_marked_sold_out"] == 1  # z was carted while "sold out"
    report = evaluate(
        [PopularityRecommender(), CoVisitationRecommender()], split, ks=(1, 2), n_resamples=50
    )
    covisit = report.results[1]
    # The b case is a hit for co-visitation (a → b), the others can't be.
    assert covisit.metrics["hit_rate@2"].mean == pytest.approx(1 / 3)
    assert covisit.segments["stage"]["new visitor"]["hit_rate@2"].mean == pytest.approx(0.5)
    assert report.segment_counts["stage"] == {"1 order": 1, "new visitor": 2}
    assert "new_category_share@2" in covisit.metrics
    md = to_markdown(report)
    assert "By segment" in md and "MRR@2" in md and "new visitor" in md
