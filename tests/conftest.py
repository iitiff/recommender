from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from recsys.catalog import Catalog, load_catalog
from recsys.core.types import Interaction, InteractionKind, TrainingData
from recsys.store import init_db, make_engine

DEMO_DIR = Path(__file__).resolve().parents[1] / "examples" / "demo"


@pytest.fixture(scope="session")
def demo_catalog() -> Catalog:
    return load_catalog(DEMO_DIR)


@pytest.fixture
def engine(tmp_path: Path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    return engine


@pytest.fixture(scope="session")
def demo_training(demo_catalog: Catalog) -> TrainingData:
    """Synthetic interactions with a clear popularity order, for interface tests only.

    (Synthetic users are fine for catching bugs, never for comparing models.)
    """
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    ids = [i.item_id for i in demo_catalog.items]
    interactions = []
    # Item 0 liked by 5 users, item 1 by 4, ... item 4 by 1.
    for rank, item_id in enumerate(ids[:5]):
        for u in range(5 - rank):
            interactions.append(
                Interaction(f"u{u}", item_id, InteractionKind.RATING, 1, t0 + timedelta(hours=u))
            )
    # Dislikes must not count as popularity.
    for u in range(10):
        interactions.append(Interaction(f"x{u}", ids[6], InteractionKind.RATING, -1, t0))
    return TrainingData(items=demo_catalog.items, interactions=interactions)
