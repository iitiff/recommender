"""Event schema: the training data for every future model.

One envelope for all events (typed common fields) plus a `properties` payload
validated per event type. Common fields are real columns so they are cheap to
query and index; per-type payloads are JSON so adding an event type never
needs a table migration.

Rules of thumb baked in here:
- `request_id` ties a served slate to its impressions, clicks, and affiliate
  clicks. Without it you cannot attribute an outcome to the model that caused it.
- `model_id` on every slate makes model comparison and A/B analysis possible.
- `position` and per-item `propensity` cannot be backfilled, so we log them now
  even though only later phases (learning-to-rank, bandits) use them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from recsys.core.ids import new_id
from recsys.core.types import ExplanationKind

SCHEMA_VERSION = 1


class EventType(StrEnum):
    PAGE_VIEW = "page_view"
    QUIZ_COMPLETED = "quiz_completed"
    SLATE_SERVED = "slate_served"  # server-side: the full ranked list we returned
    IMPRESSION = "impression"  # client-side: an item was actually visible
    ITEM_CLICK = "item_click"
    RATING = "rating"
    SAVE = "save"
    NOT_INTERESTED = "not_interested"
    AFFILIATE_CLICK = "affiliate_click"  # logged by the /go redirect, never client-side
    AFFILIATE_CONVERSION = "affiliate_conversion"  # imported from affiliate network reports


class _Props(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PageViewProps(_Props):
    path: str


class QuizCompletedProps(_Props):
    quiz_version: str
    answers: dict[str, Any]


class SlateItem(_Props):
    item_id: str
    position: int = Field(ge=0)
    score: float
    reason: ExplanationKind | None = None
    # Probability this item was shown in this position under the serving policy.
    # Null for deterministic rankers; required for off-policy evaluation later.
    propensity: float | None = Field(default=None, ge=0, le=1)


class SlateServedProps(_Props):
    items: list[SlateItem]
    k: int = Field(ge=1)
    candidate_count: int = Field(ge=0)
    latency_ms: float | None = None


class EmptyProps(_Props):
    pass


class RatingProps(_Props):
    value: int = Field(description="+1 like, -1 dislike")
    previous_value: int | None = None

    @model_validator(mode="after")
    def _check(self) -> RatingProps:
        if self.value not in (-1, 1):
            raise ValueError("rating value must be -1 or +1")
        return self


class SaveProps(_Props):
    saved: bool = True  # False = un-saved


class AffiliateClickProps(_Props):
    retailer: str
    offer_id: str | None = None
    # Passed to the affiliate network as a sub-ID so conversions can be joined back.
    click_id: str


class AffiliateConversionProps(_Props):
    click_id: str
    order_value: float
    commission: float
    currency: str = "USD"


PROPERTY_MODELS: dict[EventType, type[_Props]] = {
    EventType.PAGE_VIEW: PageViewProps,
    EventType.QUIZ_COMPLETED: QuizCompletedProps,
    EventType.SLATE_SERVED: SlateServedProps,
    EventType.IMPRESSION: EmptyProps,
    EventType.ITEM_CLICK: EmptyProps,
    EventType.RATING: RatingProps,
    EventType.SAVE: SaveProps,
    EventType.NOT_INTERESTED: EmptyProps,
    EventType.AFFILIATE_CLICK: AffiliateClickProps,
    EventType.AFFILIATE_CONVERSION: AffiliateConversionProps,
}

# Envelope fields that must be present for each event type.
REQUIRED_FIELDS: dict[EventType, tuple[str, ...]] = {
    EventType.QUIZ_COMPLETED: ("user_id",),
    EventType.SLATE_SERVED: ("request_id", "model_id", "surface"),
    EventType.IMPRESSION: ("item_id", "request_id"),
    EventType.ITEM_CLICK: ("item_id",),
    EventType.RATING: ("user_id", "item_id"),
    EventType.SAVE: ("user_id", "item_id"),
    EventType.NOT_INTERESTED: ("user_id", "item_id"),
    EventType.AFFILIATE_CLICK: ("item_id",),
}


class EventContext(BaseModel):
    """Where the user came from and what they used. Extra keys are allowed."""

    model_config = ConfigDict(extra="allow", frozen=True)

    utm_source: str | None = None
    utm_medium: str | None = None
    utm_campaign: str | None = None
    referrer: str | None = None
    user_agent: str | None = None


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(default_factory=new_id)
    schema_version: int = SCHEMA_VERSION
    event_type: EventType
    tenant_id: str
    occurred_at: datetime = Field(default_factory=_utcnow)
    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    surface: str | None = None
    item_id: str | None = None
    position: int | None = Field(default=None, ge=0)
    model_id: str | None = None
    experiment: dict[str, str] = Field(default_factory=dict)
    context: EventContext = Field(default_factory=EventContext)
    properties: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> Event:
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware (UTC)")
        missing = [f for f in REQUIRED_FIELDS.get(self.event_type, ()) if getattr(self, f) is None]
        if missing:
            raise ValueError(f"{self.event_type} event requires {', '.join(missing)}")
        # Validate and normalize the payload (e.g. enums to strings) so stored
        # JSON always has the documented shape for this event type.
        props = PROPERTY_MODELS[self.event_type].model_validate(self.properties)
        object.__setattr__(self, "properties", props.model_dump(mode="json"))
        return self
