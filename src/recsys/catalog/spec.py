"""Catalog spec: maps a tenant's CSV columns onto the engine's generic Item.

The engine knows nothing about wine. Each catalog directory has a
`catalog.yaml` saying which columns hold the title, which become tag facets,
and which are numeric scales. A different niche (or a Shopify store's product
fields) is a different spec, not different code.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FacetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    multi: bool = False  # multiple values separated by `separator`
    separator: str = ";"


class Bin(BaseModel):
    """Numeric range [low, high] (inclusive) that becomes a tag value."""

    model_config = ConfigDict(extra="forbid")

    label: str
    low: float
    high: float


class CatalogSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = "item_id"
    title: str = "title"
    description: str | None = "description"
    category: str | None = "category"
    image_url: str | None = "image_url"
    tag_facets: dict[str, FacetSpec] = Field(default_factory=dict)
    numeric_attributes: list[str] = Field(default_factory=list)
    # Numeric attributes also exposed as tags, e.g. body 4-5 -> Tag("body", "full").
    # Tags make scales usable by tag-overlap models and by explanations.
    bins: dict[str, list[Bin]] = Field(default_factory=dict)
    # Which items may be recommended: {facet: [allowed values]}. An item is
    # recommendable when, for every facet listed, it has at least one allowed
    # value. Empty = everything is recommendable. Other items are still loaded
    # so users can rate them and models can learn from them.
    recommendable_when: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> CatalogSpec:
        unknown = set(self.bins) - set(self.numeric_attributes)
        if unknown:
            raise ValueError(f"bins defined for non-numeric attributes: {sorted(unknown)}")
        facets = set(self.tag_facets) | set(self.bins)
        unknown = set(self.recommendable_when) - facets
        if unknown:
            raise ValueError(f"recommendable_when uses unknown facets: {sorted(unknown)}")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> CatalogSpec:
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})
