from __future__ import annotations

import csv
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from recsys.catalog import load_catalog
from recsys.core.types import Interaction, InteractionKind, Tag
from recsys.datasets import retailrocket
from recsys.datasets.profile import profile
from recsys.eval import load_interactions_csv
from recsys.sessions import sessionize

T0 = 1_433_221_332_000  # ms, 2015-06-02
MIN, HOUR, DAY = 60_000, 3_600_000, 86_400_000


def write(path, header, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    write(
        d / "events.csv",
        ["timestamp", "visitorid", "event", "itemid", "transactionid"],
        [
            (T0, 1, "view", 10, ""),
            (
                T0 + 5 * MIN + 123,
                1,
                "addtocart",
                10,
                "",
            ),  # abandoned cart (ms: mixed ISO formats)...
            (T0 + 2 * HOUR, 1, "view", 11, ""),  # ...new session 2h later
            (T0 + 2 * HOUR + MIN, 1, "transaction", 10, 77),  # bought the carted item
            (T0, 2, "view", 11, ""),
            (T0 + DAY, 2, "view", 12, ""),
        ],
    )
    write(d / "category_tree.csv", ["categoryid", "parentid"], [(1, ""), (2, 1), (3, 2), (4, 1)])
    write(
        d / "item_properties_part1.csv",
        ["timestamp", "itemid", "property", "value"],
        [
            (T0, 10, "categoryid", "3"),
            (T0, 10, "available", "1"),
            (T0 + DAY, 10, "available", "1"),  # no change: dropped
            (T0 + 2 * DAY, 10, "available", "0"),  # went out of stock
            (T0, 11, "categoryid", "4"),
            (T0, 12, "790", "n100.000"),  # hashed property: ignored
            (T0, 99, "categoryid", "2"),  # item never in events: excluded
        ],
    )
    return d


@pytest.fixture
def prepared(raw, tmp_path):
    out = tmp_path / "prepared"
    counts = retailrocket.prepare(raw, out)
    return out, counts


def test_prepare_counts(prepared):
    _, counts = prepared
    assert counts == {
        "items": 3,
        "items_with_category": 2,
        "interactions": 6,
        "sessions": 4,
        "users": 2,
        "availability_changes": 2,
    }


def test_interactions_load_with_kinds_and_sessions(prepared):
    out, _ = prepared
    xs = load_interactions_csv(out / "interactions.csv")
    assert [x.kind for x in xs if x.user_id == "1"] == [
        InteractionKind.VIEW,
        InteractionKind.ADD_TO_CART,
        InteractionKind.VIEW,
        InteractionKind.PURCHASE,
    ]
    assert [x.session_id for x in xs if x.user_id == "1"] == ["1:0", "1:0", "1:1", "1:1"]
    assert {x.session_id for x in xs if x.user_id == "2"} == {"2:0", "2:1"}


def test_pandas_sessions_match_pure_python_sessionize(prepared):
    out, _ = prepared
    xs = load_interactions_csv(out / "interactions.csv")
    resessioned = sessionize(replace(x, session_id=None) for x in xs)
    by_key = {(x.user_id, x.occurred_at, x.kind): x.session_id for x in resessioned}
    assert all(by_key[(x.user_id, x.occurred_at, x.kind)] == x.session_id for x in xs)


def test_catalog_has_category_paths(prepared):
    out, _ = prepared
    items = {i.item_id: i for i in load_catalog(out / "catalog").items}
    assert set(items) == {"10", "11", "12"}
    assert {t for t in items["10"].tags if t.facet == "category"} == {
        Tag("category", "1"),
        Tag("category", "2"),
        Tag("category", "3"),
    }
    assert items["12"].tags == frozenset()  # no category known


def test_availability_keeps_only_flips(prepared):
    out, _ = prepared
    rows = list(csv.DictReader((out / "availability.csv").open()))
    assert [(r["item_id"], r["available"]) for r in rows] == [("10", "true"), ("10", "false")]


def test_reengagement_profile(prepared):
    out, _ = prepared
    p = profile(out)
    cart = next(t for t in p.triggers if "abandoned cart" in t.name)
    assert (cart.sessions, cart.return_7d, cart.buy_next_7d) == (1, 1.0, 1.0)
    assert p.cart_item_share == 1.0 and p.converting_carts == 1
    assert p.returning_users == 1.0  # both users had 2 sessions
    # Gaps: user 1 came back after ~1h55m (0.08 days), user 2 after 1 day.
    assert p.return_gap_days["p25"] == pytest.approx(0.08 + 0.25 * (1 - 0.08), abs=0.01)
    assert p.return_gap_days["p90"] == pytest.approx(0.08 + 0.90 * (1 - 0.08), abs=0.01)
    assert "abandoned cart" in p.to_markdown()


def test_sessionize_gap_boundary():
    t = datetime(2026, 1, 1, tzinfo=UTC)
    xs = [
        Interaction("u", "a", InteractionKind.VIEW, 1, t),
        Interaction("u", "b", InteractionKind.VIEW, 1, t + timedelta(minutes=30)),  # exactly gap
        Interaction("u", "c", InteractionKind.VIEW, 1, t + timedelta(minutes=61)),  # > gap
        Interaction("v", "a", InteractionKind.VIEW, 1, t),
    ]
    assert [x.session_id for x in sessionize(xs)] == ["u:0", "u:0", "u:1", "v:0"]
