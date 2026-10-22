"""Load and validate a catalog directory.

A catalog directory contains:
  catalog.yaml   column mapping (see spec.py)
  items.csv      one row per item, hand-curated
  offers.csv     optional; where to buy (retailer, price, variant, url, ...)

Validation collects every problem with its row number instead of stopping at
the first, because a hand-curated CSV usually has several typos at once.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from recsys.catalog.spec import CatalogSpec
from recsys.core.types import Item, Offer, Tag


class CatalogError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("catalog has problems:\n  " + "\n  ".join(problems))


@dataclass(frozen=True)
class Catalog:
    items: list[Item]
    offers: list[Offer]


def normalize(value: str) -> str:
    """Lowercase and collapse whitespace so 'Cabernet  Sauvignon' == 'cabernet sauvignon'."""
    return " ".join(value.lower().split())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def _parse_bool(text: str) -> bool:
    lowered = text.lower()
    if lowered in ("", "true", "yes", "1", "y"):
        return True
    if lowered in ("false", "no", "0", "n"):
        return False
    raise ValueError(f"expected true/false, got {text!r}")


def parse_items(rows: list[dict[str, str]], spec: CatalogSpec) -> tuple[list[Item], list[str]]:
    items: list[Item] = []
    problems: list[str] = []
    seen: set[str] = set()

    for line, row in enumerate(rows, start=2):  # line 1 is the header
        where = f"items.csv line {line}"
        item_id = row.get(spec.item_id, "")
        title = row.get(spec.title, "")
        if not item_id:
            problems.append(f"{where}: missing {spec.item_id}")
            continue
        if item_id in seen:
            problems.append(f"{where}: duplicate item_id {item_id!r}")
            continue
        seen.add(item_id)
        if not title:
            problems.append(f"{where}: missing {spec.title}")

        tags: set[Tag] = set()
        for facet, fs in spec.tag_facets.items():
            raw = row.get(fs.column, "")
            values = raw.split(fs.separator) if fs.multi else [raw]
            tags.update(Tag(facet, normalize(v)) for v in values if v.strip())

        attributes: dict[str, float] = {}
        for name in spec.numeric_attributes:
            raw = row.get(name, "")
            if raw == "":
                continue
            try:
                attributes[name] = float(raw)
            except ValueError:
                problems.append(f"{where}: {name} must be a number, got {raw!r}")
                continue
            for b in spec.bins.get(name, []):
                if b.low <= attributes[name] <= b.high:
                    tags.add(Tag(name, b.label))

        items.append(
            Item(
                item_id=item_id,
                title=title,
                description=row.get(spec.description, "") if spec.description else "",
                category=normalize(row.get(spec.category, "")) if spec.category else "",
                tags=frozenset(tags),
                attributes=attributes,
                image_url=row.get(spec.image_url, "") if spec.image_url else "",
            )
        )
    return items, problems


def parse_offers(rows: list[dict[str, str]], item_ids: set[str]) -> tuple[list[Offer], list[str]]:
    offers: list[Offer] = []
    problems: list[str] = []
    seen: set[str] = set()

    for line, row in enumerate(rows, start=2):
        where = f"offers.csv line {line}"
        item_id, retailer = row.get("item_id", ""), row.get("retailer", "")
        if item_id not in item_ids:
            problems.append(f"{where}: unknown item_id {item_id!r}")
            continue
        if not retailer:
            problems.append(f"{where}: missing retailer")
            continue
        variant = row.get("variant", "")
        offer_id = f"{item_id}|{retailer}|{variant}"
        if offer_id in seen:
            problems.append(f"{where}: duplicate offer for {item_id}/{retailer}/{variant}")
            continue
        seen.add(offer_id)
        try:
            price = float(row.get("price", ""))
            in_stock = _parse_bool(row.get("in_stock", ""))
        except ValueError as exc:
            problems.append(f"{where}: {exc}")
            continue
        ships = row.get("ships_to", "")
        offers.append(
            Offer(
                offer_id=offer_id,
                item_id=item_id,
                retailer=retailer,
                price=price,
                currency=row.get("currency", "") or "USD",
                variant=variant,
                url=row.get("url", ""),
                in_stock=in_stock,
                ships_to=frozenset(s.strip().upper() for s in ships.split(";") if s.strip())
                if ships
                else None,
            )
        )
    return offers, problems


def load_catalog(directory: Path) -> Catalog:
    directory = Path(directory)
    spec = CatalogSpec.from_yaml(directory / "catalog.yaml")
    items, problems = parse_items(_read_csv(directory / "items.csv"), spec)

    offers: list[Offer] = []
    offers_path = directory / "offers.csv"
    if offers_path.exists():
        offers, offer_problems = parse_offers(_read_csv(offers_path), {i.item_id for i in items})
        problems += offer_problems

    if problems:
        raise CatalogError(problems)
    return Catalog(items=items, offers=offers)
