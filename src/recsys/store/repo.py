"""Read/write functions over the tables. The only module that writes SQL."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Connection, Engine, delete, insert, select, update

from recsys.core.types import Item, Offer, Tag
from recsys.store import tables as t

if TYPE_CHECKING:
    from recsys.events.schema import Event


def _now() -> datetime:
    return datetime.now(UTC)


def content_hash(item: Item) -> str:
    payload = {
        "title": item.title,
        "description": item.description,
        "category": item.category,
        "tags": sorted(str(tag) for tag in item.tags),
        "attributes": dict(sorted(item.attributes.items())),
        "image_url": item.image_url,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def ensure_tenant(conn: Connection, tenant_id: str, name: str | None = None) -> None:
    exists = conn.execute(select(t.tenants.c.tenant_id).where(t.tenants.c.tenant_id == tenant_id))
    if exists.first() is None:
        conn.execute(
            insert(t.tenants).values(tenant_id=tenant_id, name=name or tenant_id, created_at=_now())
        )


@dataclass(frozen=True)
class SyncReport:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    deactivated: int = 0
    offers: int = 0

    def __str__(self) -> str:
        return (
            f"items: {self.inserted} inserted, {self.updated} updated, "
            f"{self.unchanged} unchanged, {self.deactivated} deactivated; offers: {self.offers}"
        )


def sync_catalog(
    engine: Engine, tenant_id: str, items: list[Item], offers: list[Offer]
) -> SyncReport:
    """Make the DB match the catalog files, in one transaction.

    Items are upserted by `item_id` and rewritten only when their content hash
    changes. Items missing from the files are deactivated, not deleted. Offers
    are volatile, so they are replaced wholesale on every sync.
    """
    now = _now()
    inserted = updated = unchanged = deactivated = 0
    with engine.begin() as conn:
        ensure_tenant(conn, tenant_id)
        rows = conn.execute(
            select(t.items.c.item_id, t.items.c.content_hash, t.items.c.is_active).where(
                t.items.c.tenant_id == tenant_id
            )
        )
        existing = {r.item_id: (r.content_hash, r.is_active) for r in rows}

        for item in items:
            digest = content_hash(item)
            values = {
                "title": item.title,
                "description": item.description,
                "category": item.category,
                "attributes": dict(item.attributes),
                "image_url": item.image_url,
                "is_active": True,
                "content_hash": digest,
                "updated_at": now,
            }
            if item.item_id not in existing:
                conn.execute(
                    insert(t.items).values(
                        tenant_id=tenant_id, item_id=item.item_id, created_at=now, **values
                    )
                )
                inserted += 1
            elif existing[item.item_id] != (digest, True):
                conn.execute(
                    update(t.items)
                    .where(t.items.c.tenant_id == tenant_id, t.items.c.item_id == item.item_id)
                    .values(**values)
                )
                conn.execute(
                    delete(t.item_tags).where(
                        t.item_tags.c.tenant_id == tenant_id, t.item_tags.c.item_id == item.item_id
                    )
                )
                updated += 1
            else:
                unchanged += 1
                continue
            if item.tags:
                conn.execute(
                    insert(t.item_tags),
                    [
                        {
                            "tenant_id": tenant_id,
                            "item_id": item.item_id,
                            "facet": tag.facet,
                            "value": tag.value,
                        }
                        for tag in item.tags
                    ],
                )

        catalog_ids = {item.item_id for item in items}
        stale = [i for i, (_, active) in existing.items() if active and i not in catalog_ids]
        if stale:
            conn.execute(
                update(t.items)
                .where(t.items.c.tenant_id == tenant_id, t.items.c.item_id.in_(stale))
                .values(is_active=False, updated_at=now)
            )
            deactivated = len(stale)

        conn.execute(delete(t.offers).where(t.offers.c.tenant_id == tenant_id))
        if offers:
            conn.execute(
                insert(t.offers),
                [
                    {
                        "tenant_id": tenant_id,
                        "offer_id": o.offer_id,
                        "item_id": o.item_id,
                        "retailer": o.retailer,
                        "variant": o.variant,
                        "price": o.price,
                        "currency": o.currency,
                        "url": o.url,
                        "in_stock": o.in_stock,
                        "ships_to": sorted(o.ships_to) if o.ships_to is not None else None,
                        "updated_at": now,
                    }
                    for o in offers
                ],
            )

    return SyncReport(inserted, updated, unchanged, deactivated, len(offers))


def load_items(engine: Engine, tenant_id: str, *, active_only: bool = True) -> list[Item]:
    with engine.connect() as conn:
        tag_rows = conn.execute(
            select(t.item_tags.c.item_id, t.item_tags.c.facet, t.item_tags.c.value).where(
                t.item_tags.c.tenant_id == tenant_id
            )
        )
        tags: dict[str, set[Tag]] = defaultdict(set)
        for r in tag_rows:
            tags[r.item_id].add(Tag(r.facet, r.value))

        query = select(t.items).where(t.items.c.tenant_id == tenant_id)
        if active_only:
            query = query.where(t.items.c.is_active.is_(True))
        return [
            Item(
                item_id=r.item_id,
                title=r.title,
                description=r.description,
                category=r.category,
                tags=frozenset(tags.get(r.item_id, ())),
                attributes=dict(r.attributes),
                image_url=r.image_url,
                is_active=r.is_active,
            )
            for r in conn.execute(query.order_by(t.items.c.item_id))
        ]


def load_offers(engine: Engine, tenant_id: str) -> list[Offer]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(t.offers).where(t.offers.c.tenant_id == tenant_id).order_by(t.offers.c.offer_id)
        )
        return [
            Offer(
                offer_id=r.offer_id,
                item_id=r.item_id,
                retailer=r.retailer,
                price=r.price,
                currency=r.currency,
                variant=r.variant,
                url=r.url,
                in_stock=r.in_stock,
                ships_to=frozenset(r.ships_to) if r.ships_to is not None else None,
            )
            for r in rows
        ]


def insert_events(conn: Connection, events: list[Event]) -> None:
    if not events:
        return
    conn.execute(
        insert(t.events),
        [
            {
                **e.model_dump(mode="python", exclude={"context", "event_type"}),
                "event_type": str(e.event_type),
                "context": e.context.model_dump(mode="json", exclude_none=True),
            }
            for e in events
        ],
    )
