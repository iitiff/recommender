from collections.abc import Callable

from recsys.models.base import NotFittedError, Recommender
from recsys.models.baselines import PopularityRecommender, RandomRecommender

# Name -> factory with default params. The CLI and experiment scripts look
# models up here, so adding a model is one line.
REGISTRY: dict[str, Callable[[], Recommender]] = {
    RandomRecommender.name: RandomRecommender,
    PopularityRecommender.name: PopularityRecommender,
}

__all__ = [
    "REGISTRY",
    "NotFittedError",
    "PopularityRecommender",
    "RandomRecommender",
    "Recommender",
]
