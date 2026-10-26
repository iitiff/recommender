from __future__ import annotations

import uuid
from datetime import datetime

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from recsys.core.ids import uuid7
from recsys.core.types import ExplanationKind
from recsys.events import Event, EventType, SqlEventLogger
from recsys.store import tables


def test_uuid7_is_version_7_and_time_ordered():
    a = uuid7()
    b = uuid7()
    assert a.version == 7 and a.variant == uuid.RFC_4122
    assert a.int >> 80 <= b.int >> 80  # timestamp prefix never goes backwards


def test_rating_event_valid():
    e = Event(
        event_type=EventType.RATING,
        tenant_id="demo",
        user_id="u1",
        item_id="i1",
        properties={"value": 1},
    )
    assert e.properties == {"value": 1, "previous_value": None}
    assert e.occurred_at.tzinfo is not None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"event_type": "rating", "user_id": "u1", "properties": {"value": 1}}, "requires item_id"),
        (
            {"event_type": "rating", "user_id": "u1", "item_id": "i", "properties": {"value": 3}},
            "rating value must be",
        ),
        (
            {"event_type": "item_click", "item_id": "i", "properties": {"surprise": 1}},
            "Extra inputs",
        ),
        ({"event_type": "slate_served", "properties": {}}, "requires request_id"),
        (
            {
                "event_type": "page_view",
                "properties": {"path": "/"},
                "occurred_at": datetime(2026, 1, 1),
            },
            "timezone",
        ),
    ],
)
def test_invalid_events_rejected(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        Event(tenant_id="demo", **kwargs)


def test_slate_served_round_trips_through_sqlite(engine):
    slate = Event(
        event_type=EventType.SLATE_SERVED,
        tenant_id="demo",
        user_id="u1",
        request_id="r1",
        surface="home_recs",
        model_id="most_popular@abc12345",
        context={"utm_source": "facebook", "landing": "/quiz"},
        properties={
            "items": [
                {"item_id": "a", "position": 0, "score": 0.9, "reason": ExplanationKind.POPULAR},
                {"item_id": "b", "position": 1, "score": 0.5},
            ],
            "k": 2,
            "candidate_count": 40,
        },
    )
    click = Event(
        event_type=EventType.AFFILIATE_CLICK,
        tenant_id="demo",
        user_id="u1",
        request_id="r1",
        item_id="a",
        position=0,
        properties={"retailer": "demo-shop", "click_id": "c1"},
    )
    SqlEventLogger(engine).log_many([slate, click])

    with engine.connect() as conn:
        rows = (
            conn.execute(select(tables.events).where(tables.events.c.request_id == "r1"))
            .mappings()
            .all()
        )
    by_type = {r["event_type"]: r for r in rows}
    assert set(by_type) == {"slate_served", "affiliate_click"}
    stored = by_type["slate_served"]
    assert stored["properties"]["items"][0]["reason"] == "popular"
    assert stored["context"] == {"utm_source": "facebook", "landing": "/quiz"}
    assert by_type["affiliate_click"]["position"] == 0
