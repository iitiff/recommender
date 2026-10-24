"""Contract tests: every Recommender must pass these.

Add each new model to MODELS and it is checked against the interface's
promises for free.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from recsys.core.types import Interaction, InteractionKind, UserContext
from recsys.models import (
    CoVisitationRecommender,
    NotFittedError,
    PopularityRecommender,
    RandomRecommender,
    RecentlyViewedRecommender,
    Recommender,
    TagSimilarityRecommender,
    TrendingRecommender,
)

MODELS = [
    RandomRecommender,
    PopularityRecommender,
    TagSimilarityRecommender,
    RecentlyViewedRecommender,
    CoVisitationRecommender,
    TrendingRecommender,
]


@pytest.fixture(params=MODELS, ids=lambda cls: cls.name)
def model(request, shop_training) -> Recommender:
    return request.param().fit(shop_training)


def recommendable_ids(data) -> list[str]:
    return [i.item_id for i in data.items if i.recommendable]


@pytest.fixture
def user(demo_training) -> UserContext:
    """Rated a few items and is mid-visit: viewed ten items, carted one."""
    ids = recommendable_ids(demo_training)
    t0 = datetime(2026, 2, 1, tzinfo=UTC)
    events = [
        Interaction("u-test", i, InteractionKind.VIEW, 1, t0 + timedelta(minutes=n), "u-test:0")
        for n, i in enumerate(ids[3:13])
    ]
    events.append(
        Interaction("u-test", ids[5], InteractionKind.ADD_TO_CART, 1, t0 + timedelta(minutes=20))
    )
    return UserContext(
        user_id="u-test",
        liked=(ids[0],),
        disliked=(ids[1],),
        hidden=(ids[2],),
        events=tuple(events),
        session_id="u-test:0",
    )


def test_returns_k_unique_sorted(model, user):
    recs = model.recommend(user, k=10)
    if model.exhaustive:
        assert len(recs) == 10
    else:  # candidate sources only return items they have evidence for
        assert 0 < len(recs) <= 10
    assert len({r.item_id for r in recs}) == len(recs)
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_excludes_rated_and_hidden(model, user, demo_training):
    recs = model.recommend(user, k=100)
    assert not {r.item_id for r in recs} & user.excluded
    if model.exhaustive:
        assert len(recs) == len(recommendable_ids(demo_training)) - len(user.excluded)


def test_respects_candidates(model, user, demo_training):
    allowed = set(recommendable_ids(demo_training)[10:15])
    recs = {r.item_id for r in model.recommend(user, k=10, candidates=allowed)}
    assert recs == allowed if model.exhaustive else recs and recs <= allowed
    # Any container with `in` works (e.g. "in stock at time t"), and lists too.
    assert {r.item_id for r in model.recommend(user, k=10, candidates=sorted(allowed))} == recs


def test_breakdown_adds_up(model, user):
    for rec in model.recommend(user, k=5):
        breakdown = model.score_breakdown(user, rec.item_id)
        if breakdown is None:
            continue
        assert breakdown.total == pytest.approx(rec.score)
        if breakdown.components:
            assert sum(c.value for c in breakdown.components) == pytest.approx(breakdown.total)


def test_never_recommends_non_recommendable(model, demo_training):
    quiz_only = {i.item_id for i in demo_training.items if not i.recommendable}
    assert quiz_only, "demo catalog should include quiz-only items"
    # Even when the user liked quiz-only wines and candidates name them explicitly.
    user = UserContext(user_id="u-quiz", liked=tuple(sorted(quiz_only))[:3])
    recs = model.recommend(user, k=100, candidates=None)
    assert not {r.item_id for r in recs} & quiz_only
    assert not {r.item_id for r in model.recommend(user, k=100, candidates=quiz_only)}
    # A quiz-only wine can still anchor "you might also like" (recommendable results only).
    similar = model.similar_items(sorted(quiz_only)[0], k=5)
    assert len(similar) == 5 or not model.exhaustive
    assert not {r.item_id for r in similar} & quiz_only


def test_deterministic(model, user):
    assert model.recommend(user, k=5) == model.recommend(user, k=5)


def test_explain_flag(model, user):
    assert all(r.explanation is None for r in model.recommend(user, k=5, explain=False))


def test_similar_items_excludes_anchor(model, demo_training):
    anchor = recommendable_ids(demo_training)[3]
    recs = model.similar_items(anchor, k=5)
    assert len(recs) == 5 or (not model.exhaustive and len(recs) <= 5)
    assert anchor not in {r.item_id for r in recs}


def test_unfitted_raises():
    for cls in MODELS:
        with pytest.raises(NotFittedError):
            cls().recommend(UserContext(), k=3)


def test_model_id_tracks_params():
    assert RandomRecommender(seed=1).model_id == RandomRecommender(seed=1).model_id
    assert RandomRecommender(seed=1).model_id != RandomRecommender(seed=2).model_id
    assert RandomRecommender().model_id.startswith("random@")


def test_save_load_round_trip(model, user, tmp_path):
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = type(model).load(path)
    assert loaded.recommend(user, k=5) == model.recommend(user, k=5)


def test_popularity_orders_by_distinct_positive_users(demo_training):
    model = PopularityRecommender().fit(demo_training)
    ids = recommendable_ids(demo_training)
    recs = model.recommend(UserContext(), k=5)
    assert [r.item_id for r in recs] == ids[:5]
    # The item with 10 dislikes has no popularity.
    assert model._popularity.get(ids[6], 0.0) == 0.0


def test_model_id_tracks_algorithm_version():
    class Bumped(RandomRecommender):
        version = 3

    assert Bumped(seed=1).model_id != RandomRecommender(seed=1).model_id


def test_random_scores_do_not_depend_on_the_pool(demo_training):
    model = RandomRecommender().fit(demo_training)
    user = UserContext(user_id="u")
    full = {r.item_id: r.score for r in model.recommend(user, k=100)}
    some = list(full)[:5]
    narrowed = {r.item_id: r.score for r in model.recommend(user, k=100, candidates=some)}
    assert narrowed == {i: full[i] for i in some}
    assert model.score_breakdown(user, some[0]).total == full[some[0]]
