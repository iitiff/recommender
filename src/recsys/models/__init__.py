from collections.abc import Callable
from pathlib import Path

import yaml

from recsys.models.base import NotFittedError, Recommender
from recsys.models.baselines import PopularityRecommender, RandomRecommender
from recsys.models.content import TagSimilarityRecommender

# Name -> factory with default params. The CLI and experiment scripts look
# models up here, so adding a model is one line.
REGISTRY: dict[str, Callable[..., Recommender]] = {
    RandomRecommender.name: RandomRecommender,
    PopularityRecommender.name: PopularityRecommender,
    TagSimilarityRecommender.name: TagSimilarityRecommender,
}


def load_model_config(path: Path) -> list[Recommender]:
    """Build models from YAML, so hyperparameters live in config, not code:

    models:
      - model: tag_similarity
        params: {facet_weights: {varietal: 0.45, note: 0.2}, numeric_weight: 0.35}

    Niche-specific weights belong in the private site repo's config.
    """
    spec = yaml.safe_load(Path(path).read_text()) or {}
    models: list[Recommender] = []
    for entry in spec.get("models", []):
        name = entry["model"]
        if name not in REGISTRY:
            raise ValueError(f"unknown model {name!r}; available: {sorted(REGISTRY)}")
        models.append(REGISTRY[name](**(entry.get("params") or {})))
    return models


__all__ = [
    "REGISTRY",
    "NotFittedError",
    "PopularityRecommender",
    "RandomRecommender",
    "Recommender",
    "TagSimilarityRecommender",
    "load_model_config",
]
