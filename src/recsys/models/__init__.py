from collections.abc import Callable
from pathlib import Path

import yaml

from recsys.models.base import NotFittedError, Recommender
from recsys.models.baselines import PopularityRecommender, RandomRecommender
from recsys.models.content import TagSimilarityRecommender
from recsys.models.session import (
    CoVisitationRecommender,
    RecentlyViewedRecommender,
    TrendingRecommender,
)

# Name -> factory with default params. The CLI and experiment scripts look
# models up here, so adding a model is one line.
REGISTRY: dict[str, Callable[..., Recommender]] = {
    RandomRecommender.name: RandomRecommender,
    PopularityRecommender.name: PopularityRecommender,
    TagSimilarityRecommender.name: TagSimilarityRecommender,
    RecentlyViewedRecommender.name: RecentlyViewedRecommender,
    CoVisitationRecommender.name: CoVisitationRecommender,
    TrendingRecommender.name: TrendingRecommender,
}


def build_model_entries(entries: list[dict]) -> list[tuple[str, Recommender]]:
    """Build (label, model) pairs from config entries like
    {"label": "Hybrid", "model": "tag_similarity", "params": {...}}."""
    built: list[tuple[str, Recommender]] = []
    for entry in entries:
        name = entry["model"]
        if name not in REGISTRY:
            raise ValueError(f"unknown model {name!r}; available: {sorted(REGISTRY)}")
        model = REGISTRY[name](**(entry.get("params") or {}))
        built.append((entry.get("label") or model.model_id, model))
    return built


def load_model_config(path: Path) -> list[Recommender]:
    """Build models from YAML, so hyperparameters live in config, not code:

    models:
      - label: Hybrid            # optional, shown in the Rec Lab
        model: tag_similarity
        params: {facet_weights: {varietal: 0.45, note: 0.2}, numeric_weight: 0.35}

    Niche-specific weights belong in the private site repo's config.
    """
    spec = yaml.safe_load(Path(path).read_text()) or {}
    return [model for _, model in build_model_entries(spec.get("models", []))]


__all__ = [
    "REGISTRY",
    "CoVisitationRecommender",
    "NotFittedError",
    "PopularityRecommender",
    "RandomRecommender",
    "RecentlyViewedRecommender",
    "Recommender",
    "TagSimilarityRecommender",
    "TrendingRecommender",
    "build_model_entries",
    "load_model_config",
]
