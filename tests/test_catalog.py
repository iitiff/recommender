from __future__ import annotations

from pathlib import Path

import pytest

from recsys.catalog import CatalogError, load_catalog
from recsys.core.types import Tag
from recsys.store import repo

SPEC = """
item_id: item_id
title: title
category: null
image_url: null
description: null
tag_facets:
  varietal: {column: varietals, multi: true}
numeric_attributes: [body]
bins:
  body:
    - {label: full, low: 4, high: 5}
"""


def write_catalog(directory: Path, items_csv: str, offers_csv: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "catalog.yaml").write_text(SPEC)
    (directory / "items.csv").write_text(items_csv)
    if offers_csv is not None:
        (directory / "offers.csv").write_text(offers_csv)
    return directory


def test_demo_catalog_loads(demo_catalog):
    assert len(demo_catalog.items) == 40
    assert demo_catalog.offers
    item = demo_catalog.items[0]
    assert any(tag.facet == "varietal" for tag in item.tags)
    assert set(item.attributes) == {"body", "tannin", "acidity", "oak"}


def test_tags_are_normalized_and_bins_become_tags(tmp_path):
    d = write_catalog(
        tmp_path,
        "item_id,title,varietals,body\na,Wine A,Cabernet  Sauvignon; Merlot,5\n",
    )
    (item,) = load_catalog(d).items
    assert item.tags == {
        Tag("varietal", "cabernet sauvignon"),
        Tag("varietal", "merlot"),
        Tag("body", "full"),
    }
    assert item.attributes == {"body": 5.0}


def test_all_problems_reported_with_line_numbers(tmp_path):
    d = write_catalog(
        tmp_path,
        "item_id,title,varietals,body\n"
        "a,Wine A,Merlot,heavy\n"
        "a,Duplicate,Merlot,3\n"
        ",No id,Merlot,3\n",
        "item_id,retailer,variant,price\nzzz,shop,2020,10\na,shop,2020,cheap\n",
    )
    with pytest.raises(CatalogError) as err:
        load_catalog(d)
    problems = "\n".join(err.value.problems)
    assert "line 2: body must be a number" in problems
    assert "line 3: duplicate item_id 'a'" in problems
    assert "line 4: missing item_id" in problems
    assert "offers.csv line 2: unknown item_id 'zzz'" in problems
    assert "offers.csv line 3" in problems


def test_sync_is_idempotent_and_soft_deletes(engine, tmp_path):
    rows = "item_id,title,varietals,body\na,Wine A,Merlot,3\nb,Wine B,Syrah,4\n"
    cat = load_catalog(write_catalog(tmp_path / "v1", rows))

    first = repo.sync_catalog(engine, "t1", cat.items, cat.offers)
    assert (first.inserted, first.updated, first.unchanged) == (2, 0, 0)
    again = repo.sync_catalog(engine, "t1", cat.items, cat.offers)
    assert (again.inserted, again.updated, again.unchanged) == (0, 0, 2)

    # Edit b, drop a: b is updated, a is deactivated (not deleted).
    cat2 = load_catalog(
        write_catalog(tmp_path / "v2", "item_id,title,varietals,body\nb,Wine B,Syrah,5\n")
    )
    report = repo.sync_catalog(engine, "t1", cat2.items, cat2.offers)
    assert (report.updated, report.deactivated) == (1, 1)
    assert [i.item_id for i in repo.load_items(engine, "t1")] == ["b"]
    assert {i.item_id for i in repo.load_items(engine, "t1", active_only=False)} == {"a", "b"}
    (b,) = repo.load_items(engine, "t1")
    assert Tag("body", "full") in b.tags

    # a comes back: reactivated.
    report = repo.sync_catalog(engine, "t1", cat.items, cat.offers)
    assert report.updated == 2
    assert len(repo.load_items(engine, "t1")) == 2


def test_offers_round_trip(engine, demo_catalog):
    repo.sync_catalog(engine, "demo", demo_catalog.items, demo_catalog.offers)
    loaded = repo.load_offers(engine, "demo")
    assert sorted(loaded, key=lambda o: o.offer_id) == sorted(
        demo_catalog.offers, key=lambda o: o.offer_id
    )


def test_tenants_are_isolated(engine, demo_catalog):
    repo.sync_catalog(engine, "store-a", demo_catalog.items[:3], [])
    repo.sync_catalog(engine, "store-b", demo_catalog.items[3:5], [])
    assert len(repo.load_items(engine, "store-a")) == 3
    assert len(repo.load_items(engine, "store-b")) == 2
