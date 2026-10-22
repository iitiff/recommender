"""Domain types shared by every layer.

These are plain frozen dataclasses with no database or web dependencies. Models
receive and return only these types, so a model can be unit-tested, evaluated
offline, and served online through exactly the same code path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

ItemId = str
UserId = str


@dataclass(frozen=True, slots=True, order=True)
class Tag:
    """A (facet, value) pair, e.g. Tag("varietal", "cabernet sauvignon").

    Facets let the quiz ask about one dimension at a time and let explanations
    say *which* dimension matched ("because you like full-bodied wines").
    """

    facet: str
    value: str

    def __str__(self) -> str:
        return f"{self.facet}:{self.value}"

    @classmethod
    def parse(cls, text: str) -> Tag:
        facet, sep, value = text.partition(":")
        if not sep or not facet or not value:
            raise ValueError(f"tag must look like 'facet:value', got {text!r}")
        return cls(facet.strip(), value.strip())


@dataclass(frozen=True, slots=True)
class Item:
    """A catalog item. Identity (`item_id`) is stable forever; events reference it."""

    item_id: ItemId
    title: str
    description: str = ""
    category: str = ""
    tags: frozenset[Tag] = frozenset()
    # Numeric attributes on a known scale (e.g. body 1-5). Kept separate from tags
    # so models can use distance ("4 is close to 5"), which tags cannot express.
    attributes: Mapping[str, float] = field(default_factory=dict, hash=False)
    image_url: str = ""
    is_active: bool = True


@dataclass(frozen=True, slots=True)
class Offer:
    """Where an item can be bought. Volatile: price and stock change often.

    Separate from `Item` so refreshing prices never touches item identity.
    `variant` holds e.g. a wine vintage or a Shopify variant title.
    """

    offer_id: str
    item_id: ItemId
    retailer: str
    price: float
    currency: str = "USD"
    variant: str = ""
    url: str = ""
    in_stock: bool = True
    ships_to: frozenset[str] | None = None  # None = ships anywhere


class InteractionKind(StrEnum):
    RATING = "rating"  # value: +1 like / -1 dislike
    SAVE = "save"  # "want to try": fast intent signal, no tasting needed
    CLICK = "click"
    AFFILIATE_CLICK = "affiliate_click"
    NOT_INTERESTED = "not_interested"


@dataclass(frozen=True, slots=True)
class Interaction:
    user_id: UserId
    item_id: ItemId
    kind: InteractionKind
    value: float
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class UserContext:
    """Everything a model may know about a user at request time.

    Models receive this instead of a user_id so they never query the database.
    The same object can come from the live DB, from an offline eval split, or
    from an anonymous session.
    """

    user_id: UserId | None = None
    # Quiz-derived taste: positive = likes, negative = dislikes.
    tag_affinities: Mapping[Tag, float] = field(default_factory=dict, hash=False)
    liked: tuple[ItemId, ...] = ()
    disliked: tuple[ItemId, ...] = ()
    saved: tuple[ItemId, ...] = ()
    hidden: tuple[ItemId, ...] = ()  # "not interested"

    @property
    def excluded(self) -> frozenset[ItemId]:
        """Items never to recommend again: already rated or explicitly hidden.

        Saved items are *not* excluded; recommending something the user wants to
        try is fine.
        """
        return frozenset(self.liked) | frozenset(self.disliked) | frozenset(self.hidden)


class ExplanationKind(StrEnum):
    BECAUSE_YOU_LIKED = "because_you_liked"  # anchored on a liked item
    MATCHES_YOUR_TASTE = "matches_your_taste"  # anchored on quiz tags
    SIMILAR_TO = "similar_to"  # item-to-item ("you might also like")
    POPULAR = "popular"
    EXPLORATION = "exploration"


@dataclass(frozen=True, slots=True)
class Explanation:
    """Structured reason for a recommendation. Templates render the text.

    Keeping this structured (not a sentence) lets us log `kind` with every
    impression and later measure which explanation types earn more clicks.
    """

    kind: ExplanationKind
    anchor_item_ids: tuple[ItemId, ...] = ()
    shared_tags: tuple[Tag, ...] = ()


@dataclass(frozen=True, slots=True)
class ScoredItem:
    item_id: ItemId
    score: float
    explanation: Explanation | None = None


@dataclass(frozen=True, slots=True)
class TrainingData:
    """A snapshot of what a model may learn from."""

    items: Sequence[Item]
    interactions: Sequence[Interaction] = ()
