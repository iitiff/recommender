# recsys

A small, swappable recommender engine, built end to end as a learning project:
catalog loading, a clean event schema, recommender models behind one
interface, and an offline evaluation harness.

- **Design and the reasons behind it:** [docs/DESIGN.md](docs/DESIGN.md)
- **Where it's going:** [docs/ROADMAP.md](docs/ROADMAP.md)
- **Experiments:** [docs/experiments/](docs/experiments/)

## Quickstart

```bash
uv sync
uv run pytest
uv run recsys load-catalog examples/demo   # fictional wine catalog
uv run recsys lab examples/demo --interactions examples/demo/interactions.csv  # Rec Lab
```

The demo catalog is fictional. A real deployment supplies its own catalog
directory (`catalog.yaml` + `items.csv` + optional `offers.csv`).
