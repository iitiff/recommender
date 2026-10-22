from __future__ import annotations

import math

import numpy as np
import pytest

from recsys.eval import metrics as m
from recsys.eval.harness import bootstrap


def test_recall_at_k():
    recs, relevant = ["a", "b", "c", "d"], {"b", "d", "x"}
    assert m.recall_at_k(recs, relevant, 2) == pytest.approx(1 / 3)
    assert m.recall_at_k(recs, relevant, 4) == pytest.approx(2 / 3)
    assert m.recall_at_k(recs, set(), 4) == 0.0


def test_ndcg_rewards_earlier_hits():
    assert m.ndcg_at_k(["b", "a"], {"b"}, 2) == pytest.approx(1.0)
    assert m.ndcg_at_k(["a", "b"], {"b"}, 2) == pytest.approx(1 / math.log2(3))
    assert m.ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)
    # Ideal DCG is capped at k: 3 relevant items but k=1 and a hit at 1 -> perfect.
    assert m.ndcg_at_k(["a"], {"a", "b", "c"}, 1) == pytest.approx(1.0)


def test_hit_rate():
    assert m.hit_rate_at_k(["a", "b"], {"b"}, 2) == 1.0
    assert m.hit_rate_at_k(["a", "b"], {"b"}, 1) == 0.0


def test_coverage_and_gini():
    recs = {"u1": ["a", "b"], "u2": ["b", "c"]}
    catalog = ["a", "b", "c", "d"]
    assert m.catalog_coverage(recs, catalog, 2) == pytest.approx(0.75)
    assert m.catalog_coverage(recs, catalog, 1) == pytest.approx(0.5)

    equal = {f"u{i}": [item] for i, item in enumerate(catalog)}
    assert m.exposure_gini(equal, catalog, 1) == pytest.approx(0.0)
    concentrated = {f"u{i}": ["a"] for i in range(10)}
    assert m.exposure_gini(concentrated, catalog, 1) == pytest.approx(0.75)  # max is (n-1)/n


def test_average_popularity():
    assert m.average_popularity(["a", "b", "z"], {"a": 0.6, "b": 0.3}, 3) == pytest.approx(0.3)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    values = np.random.default_rng(1).random(200)
    est = bootstrap(values, n_resamples=500, seed=3)
    assert est.low < est.mean < est.high
    assert est == bootstrap(values, n_resamples=500, seed=3)
    constant = bootstrap(np.full(50, 0.4), n_resamples=200, seed=0)
    assert (constant.low, constant.high) == pytest.approx((0.4, 0.4))
    assert constant.significant  # an interval of [0.4, 0.4] excludes zero
