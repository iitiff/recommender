"""Ranking and beyond-accuracy metrics. Pure functions, no I/O.

Accuracy metrics are computed per user and averaged (each user counts once,
however many items they liked). Relevance is binary: an item is either in the
user's held-out positives or not.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence

import numpy as np

from recsys.core.types import ItemId


def recall_at_k(recommended: Sequence[ItemId], relevant: Collection[ItemId], k: int) -> float:
    """Share of the user's relevant items that appear in the top k.

    Note: if a user has more than k relevant items, recall@k can't reach 1.0.
    That's standard, and it's why recall is reported alongside NDCG.
    """
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: Sequence[ItemId], relevant: Collection[ItemId], k: int) -> float:
    """1.0 if at least one relevant item is in the top k. Easy to explain to non-ML folks."""
    return float(any(item in relevant for item in recommended[:k]))


def ndcg_at_k(recommended: Sequence[ItemId], relevant: Collection[ItemId], k: int) -> float:
    """Normalized Discounted Cumulative Gain with binary relevance.

    Like recall, but a hit at position 1 is worth more than a hit at position
    10 (gain discounted by log2(position + 1)), normalized by the best possible
    ordering so 1.0 means "all relevant items, ranked first".
    """
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / np.log2(position + 2)
        for position, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal = sum(1.0 / np.log2(position + 2) for position in range(min(len(relevant), k)))
    return float(dcg / ideal)


def catalog_coverage(
    recommendations: Mapping[str, Sequence[ItemId]], catalog: Collection[ItemId], k: int
) -> float:
    """Share of the recommendable catalog that appears in at least one user's top k.

    Low coverage means the model keeps showing the same few items: bad for
    users with niche taste and for discovering how the long tail performs.
    """
    if not catalog:
        return 0.0
    shown = {item for recs in recommendations.values() for item in recs[:k]}
    return len(shown & set(catalog)) / len(catalog)


def exposure_gini(
    recommendations: Mapping[str, Sequence[ItemId]], catalog: Collection[ItemId], k: int
) -> float:
    """Gini coefficient of how often each catalog item is recommended.

    0 = every item gets equal exposure; 1 = one item gets all of it. Coverage
    says *whether* items are shown at all; Gini says how unevenly.
    """
    counts = Counter(item for recs in recommendations.values() for item in recs[:k])
    x = np.sort(np.array([counts.get(item, 0) for item in catalog], dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float((2 * (ranks * x).sum()) / (n * x.sum()) - (n + 1) / n)


def average_popularity(
    recommended: Sequence[ItemId], popularity: Mapping[ItemId, float], k: int
) -> float:
    """Mean popularity of one user's top k (ARP: average recommendation popularity).

    `popularity` is the share of training users with a positive interaction on
    the item. Comparing this across models shows popularity bias: a model with
    much higher ARP than its accuracy justifies is mostly recommending hits.
    """
    top = recommended[:k]
    if not top:
        return 0.0
    return float(np.mean([popularity.get(item, 0.0) for item in top]))
