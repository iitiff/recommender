"""Contract tests: every Recommender must pass these.

Add each new model to MODELS and it is checked against the interface's
promises for free.
"""

from __future__ import annotations

import pytest

from recsys.core.types import UserContext
from recsys.models import NotFittedError, PopularityRecommender, RandomRecommender, Recommender

MODELS = [RandomRecommender, PopularityRecommender]


@pytest.fixture(params=MODELS, ids=lambda cls: cls.name)
def model(request, demo_training) -> Recommender:
    return request.param().fit(demo_training)


@pytest.fixture
def user(demo_training) -> UserContext:
    ids = [i.item_id for i in demo_training.items]
    return UserContext(user_id="u-test", liked=(ids[0],), disliked=(ids[1],), hidden=(ids[2],))


def test_returns_k_unique_sorted(model, user):
    recs = model.recommend(user, k=10)
    assert len(recs) == 10
    assert len({r.item_id for r in recs}) == 10
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_excludes_rated_and_hidden(model, user):
    recs = model.recommend(user, k=100)
    assert not {r.item_id for r in recs} & user.excluded
    assert len(recs) == 40 - len(user.excluded)


def test_respects_candidates(model, user, demo_training):
    allowed = {i.item_id for i in demo_training.items[10:15]}
    recs = model.recommend(user, k=10, candidates=allowed)
    assert {r.item_id for r in recs} == allowed


def test_deterministic(model, user):
    assert model.recommend(user, k=5) == model.recommend(user, k=5)


def test_explain_flag(model, user):
    assert all(r.explanation is None for r in model.recommend(user, k=5, explain=False))


def test_similar_items_excludes_anchor(model, demo_training):
    anchor = demo_training.items[3].item_id
    recs = model.similar_items(anchor, k=5)
    assert len(recs) == 5
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
    ids = [i.item_id for i in demo_training.items]
    recs = model.recommend(UserContext(), k=5)
    assert [r.item_id for r in recs] == ids[:5]
    # The item with 10 dislikes has no popularity.
    assert model._popularity.get(ids[6], 0.0) == 0.0
