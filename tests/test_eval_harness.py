from __future__ import annotations

import json

from typer.testing import CliRunner

from recsys.cli import app
from recsys.eval import evaluate, load_interactions_csv, temporal_split, to_dict, to_markdown
from recsys.models import PopularityRecommender, RandomRecommender
from tests.conftest import DEMO_DIR


def test_popularity_beats_random_on_demo(demo_catalog):
    """Sanity check of the whole pipeline on synthetic data with a popularity skew.

    (This validates the harness, not the models: see examples/demo/generate_interactions.py.)
    """
    data = load_interactions_csv(DEMO_DIR / "interactions.csv")
    split = temporal_split(demo_catalog.items, data, test_fraction=0.2)
    report = evaluate([RandomRecommender(), PopularityRecommender()], split, ks=(5, 10))

    pop = report.result(PopularityRecommender().model_id)
    rnd = report.result(RandomRecommender().model_id)
    assert pop.metrics["ndcg@10"].mean > rnd.metrics["ndcg@10"].mean
    assert report.comparisons[rnd.model_id]["ndcg@10"].significant
    # Popularity concentrates exposure; random spreads it.
    assert pop.aggregates["gini@10"] > rnd.aggregates["gini@10"]
    assert rnd.aggregates["coverage@10"] == 1.0

    md = to_markdown(report)
    assert "NDCG@10" in md and "Paired difference" in md
    json.dumps(to_dict(report), default=str)


def test_cli_evaluate_writes_reports(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(DEMO_DIR),
            str(DEMO_DIR / "interactions.csv"),
            "--k",
            "5",
            "--split",
            "last",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Recall@5" in result.output
    assert len(list(tmp_path.glob("*.md"))) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cli_rejects_unknown_model(tmp_path):
    result = CliRunner().invoke(
        app,
        ["evaluate", str(DEMO_DIR), str(DEMO_DIR / "interactions.csv"), "--models", "nope"],
    )
    assert result.exit_code == 1
