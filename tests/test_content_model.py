from __future__ import annotations

from datetime import UTC, datetime

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
from recsys.models import TagSimilarityRecommender


def wine(item_id, grape, body, oak, notes, recommendable=True, price=None):
    attrs = {"body": body, "oak": oak}
    if price is not None:
        attrs["price"] = price
    tags = {Tag("varietal", grape)} | {Tag("note", n) for n in notes}
    tags |= {Tag("body", "full" if body >= 4 else "light")}  # a bin tag
    return Item(
        item_id, item_id, tags=frozenset(tags), attributes=attrs, recommendable=recommendable
    )


ITEMS = [
    wine("big-cab", "cabernet", 5, 5, ["blackcurrant", "vanilla"]),
    wine("cab-2", "cabernet", 5, 4, ["blackcurrant", "cedar"]),
    wine("lean-cab", "cabernet", 3, 2, ["red currant", "earth"]),
    wine("pinot", "pinot noir", 2, 2, ["red cherry", "raspberry"]),
    wine("pinot-2", "pinot noir", 2, 3, ["red cherry", "cola"]),
    wine("zin", "zinfandel", 4, 3, ["blackberry", "black pepper"]),
    wine("paso-cab", "cabernet", 5, 5, ["blackcurrant", "vanilla"], recommendable=False),
]
T0 = datetime(2026, 1, 1, tzinfo=UTC)
POPULAR = [Interaction(f"u{i}", "zin", InteractionKind.RATING, 1, T0) for i in range(5)] + [
    Interaction("u9", "pinot", InteractionKind.RATING, 1, T0)
]


@pytest.fixture
def model():
    return TagSimilarityRecommender(
        facet_weights={"varietal": 0.45, "note": 0.2}, numeric_weight=0.35
    ).fit(TrainingData(ITEMS, POPULAR))


def ids(recs):
    return [r.item_id for r in recs]


def test_likes_pull_similar_wines_up(model):
    recs = model.recommend(UserContext("u", liked=("big-cab",)), k=3)
    assert ids(recs)[0] == "cab-2"
    assert ids(recs)[-1] != "pinot"


def test_quiz_only_wine_anchors_recommendable_ones(model):
    recs = model.recommend(UserContext("u", liked=("paso-cab",)), k=2)
    assert ids(recs)[0] in {"big-cab", "cab-2"}
    assert "paso-cab" not in ids(recs)
    exp = recs[0].explanation
    assert exp.kind == ExplanationKind.BECAUSE_YOU_LIKED
    assert exp.anchor_item_ids == ("paso-cab",)
    assert Tag("varietal", "cabernet") in exp.shared_tags


def test_two_tastes_are_both_served_not_averaged(model):
    recs = model.recommend(UserContext("u", liked=("big-cab", "pinot")), k=2)
    assert set(ids(recs)) == {"cab-2", "pinot-2"}


def test_dislike_pushes_similar_wines_down(model):
    base = ids(model.recommend(UserContext("u", liked=("zin",)), k=10))
    after = ids(model.recommend(UserContext("u", liked=("zin",), disliked=("big-cab",)), k=10))
    assert after.index("cab-2") > base.index("cab-2")


def test_cold_start_falls_back_to_popularity(model):
    recs = model.recommend(UserContext("new"), k=2)
    assert ids(recs) == ["zin", "pinot"]
    assert recs[0].explanation.kind == ExplanationKind.POPULAR


def test_quiz_affinities_alone_personalize(model):
    user = UserContext("u", tag_affinities={Tag("varietal", "pinot noir"): 1.0})
    recs = model.recommend(user, k=2)
    assert set(ids(recs)) == {"pinot", "pinot-2"}
    assert recs[0].explanation.kind == ExplanationKind.MATCHES_YOUR_TASTE


@pytest.mark.parametrize(
    "user",
    [
        UserContext("a", liked=("big-cab",), disliked=("pinot",)),
        UserContext("b", liked=("zin", "paso-cab"), tag_affinities={Tag("note", "cola"): 1.0}),
        UserContext("c"),
    ],
)
def test_breakdown_sums_to_the_real_score(user):
    model = TagSimilarityRecommender(popularity_weight=0.2).fit(TrainingData(ITEMS, POPULAR))
    for rec in model.recommend(user, k=5):
        breakdown = model.score_breakdown(user, rec.item_id)
        assert breakdown.total == pytest.approx(rec.score)
        assert sum(c.value for c in breakdown.components) == pytest.approx(rec.score)


def test_breakdown_lists_anchors_best_first(model):
    b = model.score_breakdown(UserContext("u", liked=("pinot", "big-cab")), "cab-2")
    assert [a for a, _ in b.anchors] == ["big-cab", "pinot"]


def test_similar_items_and_explanation(model):
    recs = model.similar_items("big-cab", k=1)
    assert ids(recs) == ["cab-2"]
    assert recs[0].explanation.kind == ExplanationKind.SIMILAR_TO


def test_bin_facets_and_price_are_not_double_counted():
    items = [
        wine("a", "cabernet", 5, 5, ["x"], price=20),
        wine("b", "cabernet", 5, 5, ["x"], price=140),
    ]
    model = TagSimilarityRecommender().fit(TrainingData(items))
    assert "body" not in model._weights  # covered by the numeric scale
    assert model.similar_items("a", k=1)[0].score == pytest.approx(1.0)  # price ignored


def test_unknown_facet_weight_is_an_error():
    with pytest.raises(ValueError, match="unknown facets"):
        TagSimilarityRecommender(facet_weights={"grape": 1.0}).fit(TrainingData(ITEMS))


def test_params_change_model_id():
    assert (
        TagSimilarityRecommender(popularity_weight=0.3).model_id
        != TagSimilarityRecommender().model_id
    )
