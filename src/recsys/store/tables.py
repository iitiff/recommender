"""Database schema (SQLAlchemy Core).

Design notes:
- Every table carries `tenant_id`. There is one tenant today; a multi-tenant
  product needs one per store. One extra column now avoids the most painful
  migration later: adding a partition key to a large event log.
- Raw facts (`quiz_responses`, `feedback`, `events`) are the source of truth.
  `user_profiles` is a cache derived from them and can be rebuilt any time.
- Items are soft-deleted (`is_active`), never removed, because historical
  events reference them.
- `events` has no foreign keys on purpose: logging must never fail because
  some other table is in an unexpected state. It is an append-only log.
- `JSONType` maps to JSON on SQLite and JSONB on Postgres.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB

JSONType = JSON().with_variant(JSONB(), "postgresql")
Timestamp = DateTime(timezone=True)

metadata = MetaData()

tenants = Table(
    "tenants",
    metadata,
    Column("tenant_id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("created_at", Timestamp, nullable=False),
)

items = Table(
    "items",
    metadata,
    Column("tenant_id", String, primary_key=True),
    Column("item_id", String, primary_key=True),
    Column("title", String, nullable=False),
    Column("description", Text, nullable=False, default=""),
    Column("category", String, nullable=False, default=""),
    Column("attributes", JSONType, nullable=False, default=dict),
    Column("image_url", String, nullable=False, default=""),
    Column("is_active", Boolean, nullable=False, default=True),
    Column("recommendable", Boolean, nullable=False, default=True),
    # Hash of the content fields; refresh only rewrites (and later re-embeds)
    # items whose content actually changed.
    Column("content_hash", String, nullable=False),
    Column("created_at", Timestamp, nullable=False),
    Column("updated_at", Timestamp, nullable=False),
    ForeignKeyConstraint(["tenant_id"], ["tenants.tenant_id"]),
)

item_tags = Table(
    "item_tags",
    metadata,
    Column("tenant_id", String, primary_key=True),
    Column("item_id", String, primary_key=True),
    Column("facet", String, primary_key=True),
    Column("value", String, primary_key=True),
    ForeignKeyConstraint(
        ["tenant_id", "item_id"], ["items.tenant_id", "items.item_id"], ondelete="CASCADE"
    ),
    Index("ix_item_tags_tag", "tenant_id", "facet", "value"),
)

offers = Table(
    "offers",
    metadata,
    Column("tenant_id", String, primary_key=True),
    Column("offer_id", String, primary_key=True),
    Column("item_id", String, nullable=False),
    Column("retailer", String, nullable=False),
    Column("variant", String, nullable=False, default=""),
    Column("price", Float, nullable=False),
    Column("currency", String, nullable=False, default="USD"),
    Column("url", String, nullable=False, default=""),
    Column("in_stock", Boolean, nullable=False, default=True),
    Column("ships_to", JSONType, nullable=True),  # list of region codes; null = anywhere
    Column("updated_at", Timestamp, nullable=False),
    ForeignKeyConstraint(["tenant_id", "item_id"], ["items.tenant_id", "items.item_id"]),
    Index("ix_offers_item", "tenant_id", "item_id"),
)

users = Table(
    "users",
    metadata,
    Column("tenant_id", String, primary_key=True),
    Column("user_id", String, primary_key=True),  # anonymous UUID from a cookie
    Column("email", String, nullable=True),
    Column("acquisition", JSONType, nullable=False, default=dict),  # first-touch utm/referrer
    Column("created_at", Timestamp, nullable=False),
    ForeignKeyConstraint(["tenant_id"], ["tenants.tenant_id"]),
)

quiz_responses = Table(
    "quiz_responses",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("tenant_id", String, nullable=False),
    Column("user_id", String, nullable=False),
    Column("quiz_version", String, nullable=False),
    Column("answers", JSONType, nullable=False),
    Column("created_at", Timestamp, nullable=False),
    ForeignKeyConstraint(["tenant_id", "user_id"], ["users.tenant_id", "users.user_id"]),
    Index("ix_quiz_user", "tenant_id", "user_id"),
)

# Current feedback state per user-item. The full history lives in `events`.
feedback = Table(
    "feedback",
    metadata,
    Column("tenant_id", String, primary_key=True),
    Column("user_id", String, primary_key=True),
    Column("item_id", String, primary_key=True),
    Column("rating", Integer, nullable=True),  # +1 / -1 / null
    Column("saved", Boolean, nullable=False, default=False),
    Column("hidden", Boolean, nullable=False, default=False),
    Column("updated_at", Timestamp, nullable=False),
    ForeignKeyConstraint(["tenant_id", "user_id"], ["users.tenant_id", "users.user_id"]),
    ForeignKeyConstraint(["tenant_id", "item_id"], ["items.tenant_id", "items.item_id"]),
)

user_profiles = Table(
    "user_profiles",
    metadata,
    Column("tenant_id", String, primary_key=True),
    Column("user_id", String, primary_key=True),
    Column("tag_affinities", JSONType, nullable=False),  # {"facet:value": weight}
    Column("derived_from", JSONType, nullable=False),  # e.g. {"quiz_version": "v1"}
    Column("updated_at", Timestamp, nullable=False),
    ForeignKeyConstraint(["tenant_id", "user_id"], ["users.tenant_id", "users.user_id"]),
)

events = Table(
    "events",
    metadata,
    Column("event_id", String, primary_key=True),
    Column("schema_version", Integer, nullable=False),
    Column("event_type", String, nullable=False),
    Column("tenant_id", String, nullable=False),
    Column("occurred_at", Timestamp, nullable=False),
    Column("user_id", String, nullable=True),
    Column("session_id", String, nullable=True),
    Column("request_id", String, nullable=True),
    Column("surface", String, nullable=True),
    Column("item_id", String, nullable=True),
    Column("position", Integer, nullable=True),
    Column("model_id", String, nullable=True),
    Column("experiment", JSONType, nullable=False),
    Column("context", JSONType, nullable=False),
    Column("properties", JSONType, nullable=False),
    Index("ix_events_time", "tenant_id", "occurred_at"),
    Index("ix_events_user", "tenant_id", "user_id", "occurred_at"),
    Index("ix_events_type", "tenant_id", "event_type", "occurred_at"),
    Index("ix_events_request", "request_id"),
)
