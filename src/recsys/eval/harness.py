"""Offline evaluation harness: fit models on a split, score them, compare them.

Every model is evaluated on the same users with the same rules, and every
number comes with a bootstrap confidence interval. With small data (a tasting
panel of 20 people, a young site) differences between models are often noise;
the report says so instead of letting a lucky 0.02 look like progress.

Comparisons use the *paired* bootstrap: resample users, and compute the
difference between two models on the same resampled users. Because both
models face the same easy and hard users, the noise largely cancels, so the
paired interval is much tighter than comparing two separate intervals.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from recsys.core.types import ItemId, UserId
from recsys.eval import metrics as m
from recsys.eval.split import Split, train_popularity
from recsys.models.base import Recommender

# Descriptive metrics: reported, but not compared as "better" or "worse".
DESCRIPTIVE = ("avg_popularity", "new_category_share", "list_length")


@dataclass(frozen=True)
class Estimate:
    """A mean with a 95% bootstrap confidence interval."""

    mean: float
    low: float
    high: float

    @property
    def significant(self) -> bool:
        """For differences: the interval excludes zero."""
        return self.low > 0 or self.high < 0

    def fmt(self, digits: int = 3) -> str:
        return f"{self.mean:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


@dataclass
class ModelResult:
    model_id: str
    metrics: dict[str, Estimate]  # e.g. "ndcg@10"
    aggregates: dict[str, float]  # coverage@k, gini@k (whole-population, no CI)
    fit_seconds: float
    ms_per_user: float
    per_user: dict[str, np.ndarray] = field(repr=False, default_factory=dict)
    # dimension -> segment -> metric -> estimate (when the split has segments)
    segments: dict[str, dict[str, dict[str, Estimate]]] = field(default_factory=dict)


@dataclass
class EvalReport:
    info: dict[str, Any]
    ks: tuple[int, ...]
    results: list[ModelResult]
    baseline: str | None
    # model_id -> metric -> (model - baseline) difference
    comparisons: dict[str, dict[str, Estimate]]
    # dimension -> segment -> number of cases
    segment_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def result(self, model_id: str) -> ModelResult:
        return next(r for r in self.results if r.model_id == model_id)


def bootstrap(values: np.ndarray, *, n_resamples: int, seed: int) -> Estimate:
    if len(values) == 0:
        return Estimate(0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return Estimate(float(values.mean()), float(low), float(high))


def evaluate(
    models: Sequence[Recommender],
    split: Split,
    *,
    ks: Sequence[int] = (5, 10, 20),
    baseline: str | None = "most_popular",
    n_resamples: int = 1000,
    seed: int = 0,
) -> EvalReport:
    """Fit each model on `split.train` and score its top-k for every evaluated user.

    `baseline` is matched against model *names* (e.g. "most_popular"); every
    other model gets a paired comparison against it.
    """
    ks = tuple(sorted(set(ks)))
    max_k = ks[-1]
    users: list[UserId] = sorted(split.targets)
    catalog = [i.item_id for i in split.train.items if i.is_active and i.recommendable]
    popularity = train_popularity(split.train)
    segment_index = _segment_index(split, users)

    results: list[ModelResult] = []
    names: dict[str, str] = {}
    for model in models:
        start = time.perf_counter()
        model.fit(split.train)
        fit_seconds = time.perf_counter() - start

        start = time.perf_counter()
        recs: dict[UserId, list[ItemId]] = {
            u: [
                r.item_id
                for r in model.recommend(
                    split.contexts[u], max_k, candidates=split.candidates.get(u), explain=False
                )
            ]
            for u in users
        }
        ms_per_user = 1000 * (time.perf_counter() - start) / max(len(users), 1)

        per_user: dict[str, np.ndarray] = {}
        aggregates: dict[str, float] = {}
        for k in ks:
            per_user[f"recall@{k}"] = np.array(
                [m.recall_at_k(recs[u], split.targets[u], k) for u in users]
            )
            per_user[f"ndcg@{k}"] = np.array(
                [m.ndcg_at_k(recs[u], split.targets[u], k) for u in users]
            )
            per_user[f"hit_rate@{k}"] = np.array(
                [m.hit_rate_at_k(recs[u], split.targets[u], k) for u in users]
            )
            per_user[f"mrr@{k}"] = np.array(
                [m.mrr_at_k(recs[u], split.targets[u], k) for u in users]
            )
            per_user[f"avg_popularity@{k}"] = np.array(
                [m.average_popularity(recs[u], popularity, k) for u in users]
            )
            per_user[f"list_length@{k}"] = np.array([min(len(recs[u]), k) for u in users], float)
            if split.item_groups:
                per_user[f"new_category_share@{k}"] = np.array(
                    [
                        m.new_group_share(
                            recs[u], split.item_groups, split.known_groups.get(u, frozenset()), k
                        )
                        for u in users
                    ]
                )
            aggregates[f"coverage@{k}"] = m.catalog_coverage(recs, catalog, k)
            aggregates[f"gini@{k}"] = m.exposure_gini(recs, catalog, k)

        results.append(
            ModelResult(
                model_id=model.model_id,
                metrics={
                    name: bootstrap(v, n_resamples=n_resamples, seed=seed)
                    for name, v in per_user.items()
                },
                aggregates=aggregates,
                fit_seconds=fit_seconds,
                ms_per_user=ms_per_user,
                per_user=per_user,
                segments={
                    dim: {
                        value: {
                            name: bootstrap(v[idx], n_resamples=n_resamples, seed=seed)
                            for name, v in per_user.items()
                            if name.endswith(f"@{max_k}")
                        }
                        for value, idx in by_value.items()
                    }
                    for dim, by_value in segment_index.items()
                },
            )
        )
        names[model.model_id] = model.name

    comparisons: dict[str, dict[str, Estimate]] = {}
    base = next((r for r in results if names[r.model_id] == baseline), None)
    if base is not None:
        for r in results:
            if r is base:
                continue
            # Same seed => same resampled users for every metric and model: a paired bootstrap.
            comparisons[r.model_id] = {
                name: bootstrap(
                    r.per_user[name] - base.per_user[name], n_resamples=n_resamples, seed=seed
                )
                for name in r.per_user
                if not name.startswith(DESCRIPTIVE)
            }

    return EvalReport(
        info=dict(split.info),
        ks=ks,
        results=results,
        baseline=base.model_id if base else None,
        comparisons=comparisons,
        segment_counts={
            dim: {value: len(idx) for value, idx in by_value.items()}
            for dim, by_value in segment_index.items()
        },
    )


def _segment_index(split: Split, users: list[UserId]) -> dict[str, dict[str, np.ndarray]]:
    """dimension -> segment value -> positions of its cases in `users`."""
    out: dict[str, dict[str, list[int]]] = {}
    for n, u in enumerate(users):
        for dim, value in split.segments.get(u, {}).items():
            out.setdefault(dim, {}).setdefault(value, []).append(n)
    return {
        dim: {value: np.array(idx) for value, idx in sorted(by_value.items())}
        for dim, by_value in out.items()
    }
