from __future__ import annotations

from recsys.candidates import CandidateFilter, filter_candidates
from recsys.core.types import Item, Offer

ITEMS = [
    Item("cheap", "Cheap"),
    Item("pricey", "Pricey"),
    Item("no-offer", "No offer yet", attributes={"price": 45.0}),
    Item("mystery", "Unknown price"),
    Item("quiz-only", "Quiz only", recommendable=False),
    Item("retired", "Retired", is_active=False),
]
OFFERS = [
    Offer("cheap|shop|", "cheap", "shop", 18.0, ships_to=frozenset({"CA"})),
    Offer("pricey|shop|", "pricey", "shop", 120.0),
    Offer("pricey|other|", "pricey", "other", 95.0, in_stock=False),
]


def test_no_filter_keeps_recommendable_active_items():
    result = filter_candidates(ITEMS, OFFERS, CandidateFilter())
    assert result.item_ids == {"cheap", "pricey", "no-offer", "mystery"}
    assert result.steps == (("Catalog", 5), ("Recommendable", 4))


def test_budget_uses_cheapest_in_stock_offer_then_price_attribute():
    result = filter_candidates(ITEMS, OFFERS, CandidateFilter(min_price=30, max_price=100))
    # pricey's $95 offer is out of stock, so its price is $120: out of budget.
    assert result.item_ids == {"no-offer", "mystery"}
    assert result.unknown_price == 1  # mystery kept: price unknown
    assert result.steps[-1] == ("Within budget", 2)


def test_shipping_region():
    result = filter_candidates(ITEMS, OFFERS, CandidateFilter(ship_to="NY"))
    # cheap only ships to CA; pricey's in-stock offer ships anywhere.
    assert "cheap" not in result.item_ids and "pricey" in result.item_ids
    assert result.steps[-1] == ("Available in NY", 3)


def test_require_offer_drops_items_without_offers():
    result = filter_candidates(ITEMS, OFFERS, CandidateFilter(require_offer=True))
    assert result.item_ids == {"cheap", "pricey"}
