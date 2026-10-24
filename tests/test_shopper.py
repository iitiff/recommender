"""Shopper state (docs/SHOPPER-PROFILE.md) and the journey sources built on it."""

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
from recsys.lifecycle import Stage, lifecycle_stage
from recsys.models import (
    AdjacentCategoryRecommender,
    CartSubstitutesRecommender,
    RebuyRecommender,
)
from recsys.shopper import ItemFacts, shopper_state

T0 = datetime(2026, 3, 10, tzinfo=UTC)
V, C, P = InteractionKind.VIEW, InteractionKind.ADD_TO_CART, InteractionKind.PURCHASE


def ev(user, item, kind=V, hours=0.0, session=None):
    return Interaction(user, item, kind, 1, T0 + timedelta(hours=hours), session or f"{user}:0")


def item(item_id, category="phones", type_="electronics", brand=None, price=None):
    tags = {Tag("type", type_)} | ({Tag("brand", brand)} if brand else set())
    return Item(
        item_id,
        f"Item {item_id}",
        category=category,
        tags=frozenset(tags),
        attributes={"price": price} if price else {},
    )


# -- lifecycle and shopper state ------------------------------------------------


def test_abandoner_is_its_own_stage():
    carted_before = [ev("s", "a", C, session="s:0")]
    assert lifecycle_stage(carted_before, "s:1") == Stage.ABANDONER
    assert lifecycle_stage([ev("s", "a", session="s:0")], "s:1") == Stage.BROWSER
    # A cart in the *current* visit doesn't make someone an abandoner yet.
    assert lifecycle_stage(carted_before, "s:0") == Stage.NEW_VISITOR


def test_momentum_and_open_cart():
    events = (
        ev("s", "x", P, hours=-48, session="s:0"),  # last order two days ago
        ev("s", "a", C, hours=-50, session="s:0"),  # carted before that order: not open
        ev("s", "b", C, hours=-5, session="s:1"),
        ev("s", "c", C, hours=-3, session="s:1"),
        ev("s", "c", V, hours=-0.5, session="s:2"),
        ev("s", "d", V, hours=0, session="s:2"),
    )
    state = shopper_state(UserContext("s", events=events, session_id="s:2", now=T0))
    assert state.stage == Stage.FIRST_ORDER
    assert state.open_cart == ("c", "b")  # newest first
    assert state.open_cart_age_hours == pytest.approx(3)
    assert state.hours_since_last_order == pytest.approx(48)
    assert state.hours_since_last_visit == pytest.approx(
        2.5
    )  # s:1 ended 3h ago, s:2 began 0.5h ago
    assert state.visit_events == 2 and state.anchor_item == "d"
    assert state.bought == ("x",)


def test_price_profile_pools_toward_general_then_store():
    phones = [item(f"p{n}", "phones", "electronics", price=100 + n) for n in range(20)]
    kettles = [item(f"k{n}", "kettles", "kitchen", price=20 + n) for n in range(20)]
    facts = ItemFacts.from_items(phones + kettles, type_facet="type")
    assert facts.price_pct["p19"] == 1.0 and facts.price_pct["p0"] == 0.05

    premium_phones = tuple(ev("s", f"p{n}", hours=-n * 0.1) for n in range(15, 20))
    state = shopper_state(UserContext("s", events=premium_phones, now=T0), facts)
    # General level: 5 views averaging 0.9, pooled with 5 pseudo-views at 0.5.
    assert state.price_general == pytest.approx(
        (sum(n / 20 for n in range(16, 21)) + 2.5) / 10, abs=0.01
    )
    assert state.price_band["electronics"] > state.price_general > 0.5
    assert "kitchen" not in state.price_band  # no evidence: use price_general
    newcomer = shopper_state(UserContext("n", now=T0), facts)
    assert newcomer.price_general == 0.5  # store average


def test_affinity_decays_with_age():
    facts = ItemFacts.from_items(
        [item("a", type_="audio"), item("b", type_="video")], type_facet="type"
    )
    events = (ev("s", "a", hours=-24 * 90), ev("s", "b"))  # one view 90 days ago, one now
    aff = shopper_state(
        UserContext("s", events=events, now=T0), facts, half_life_days=90
    ).type_affinity
    assert aff["audio"] == pytest.approx(0.5) and aff["video"] == pytest.approx(1.0)


# -- rebuy --------------------------------------------------------------------------


def test_rebuy_ranks_by_smoothed_repeat_rate():
    catalog = [item("milk", "dairy", "grocery"), item("tv", "tvs", "electronics")]
    history = []
    for n in range(10):  # 10 milk buyers, 8 buy again on another day
        history.append(ev(f"m{n}", "milk", P, hours=-200))
        if n < 8:
            history.append(ev(f"m{n}", "milk", P, hours=-100))
    history += [ev(f"t{n}", "tv", P, hours=-200) for n in range(10)]  # nobody re-buys a TV
    model = RebuyRecommender(type_facet="type", prior_strength=2).fit(
        TrainingData(catalog, history)
    )
    assert model.rate("milk") > 0.5 > model.rate("tv")
    shopper = UserContext("s", events=(ev("s", "milk", P, -30), ev("s", "tv", P, -20)), now=T0)
    recs = model.recommend(shopper, k=5)
    assert [r.item_id for r in recs] == ["milk", "tv"]
    assert recs[0].explanation.kind == ExplanationKind.BUY_AGAIN
    # Something bought in this very visit isn't suggested again.
    just_now = UserContext("s", events=(ev("s", "milk", P, session="s:9"),), session_id="s:9")
    assert model.recommend(just_now, k=5) == []


# -- cart substitutes ---------------------------------------------------------------


def test_substitutes_stay_in_category_and_near_price():
    catalog = [
        item("phone", "phones", price=300),
        item("phone_similar", "phones", price=320),
        item("phone_luxury", "phones", price=1200),
        item("case", "cases", price=20),  # a complement, not a substitute
    ]
    visits = []
    for n in range(10):
        for other in ("phone_similar", "phone_luxury", "case"):
            visits += [ev(f"u{n}", "phone", hours=n), ev(f"u{n}", other, hours=n + 0.1)]
    model = CartSubstitutesRecommender().fit(TrainingData(catalog, visits))
    shopper = UserContext("s", events=(ev("s", "phone", C),), now=T0)
    recs = model.recommend(shopper, k=5)
    assert [r.item_id for r in recs] == ["phone_similar", "phone_luxury"]
    assert recs[0].score > 5 * recs[1].score  # 4× the price is far outside the band
    assert recs[0].explanation.kind == ExplanationKind.ALTERNATIVE
    assert recs[0].explanation.anchor_item_ids == ("phone",)
    assert model.recommend(UserContext("v", events=(ev("v", "phone"),)), k=5) == []  # no cart


# -- adjacent category --------------------------------------------------------------


def test_adjacent_category_learns_transitions_and_skips_known_types():
    catalog = [
        item("tv", type_="tv"),
        item("speaker", type_="audio"),
        item("soundbar", type_="audio"),
        item("toaster", type_="kitchen"),
        item("laptop", type_="computers"),
    ]
    history = []
    for n in range(30):  # TV buyers move on to audio
        history += [ev(f"a{n}", "tv", hours=-100 + n), ev(f"a{n}", "speaker", hours=-90 + n)]
    for n in range(10):
        history += [ev(f"a{n}", "soundbar", hours=-80 + n)]
    for n in range(30):  # everyone else: laptops then kitchen
        history += [ev(f"b{n}", "laptop", hours=-100 + n), ev(f"b{n}", "toaster", hours=-90 + n)]
    model = AdjacentCategoryRecommender(min_transitions=5).fit(TrainingData(catalog, history))
    shopper = UserContext("s", events=(ev("s", "tv"),), now=T0)
    recs = model.recommend(shopper, k=5)
    assert [r.item_id for r in recs] == ["speaker", "soundbar"]  # popular first, audio only
    assert recs[0].explanation.kind == ExplanationKind.NEW_FOR_YOU
    assert recs[0].explanation.shared_tags == (Tag("type", "audio"),)
    # Someone already into audio gets no audio "expansion".
    both = UserContext("s", events=(ev("s", "tv"), ev("s", "speaker")), now=T0)
    assert model.recommend(both, k=5) == []
