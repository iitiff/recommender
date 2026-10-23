"""Command-line entry point: `recsys --help`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from recsys.catalog import CatalogError, load_catalog
from recsys.config import Settings
from recsys.eval import evaluate as run_evaluation
from recsys.eval import leave_last_out, load_interactions_csv, temporal_split, to_markdown
from recsys.eval.report import write_report
from recsys.models import REGISTRY, load_model_config
from recsys.store import init_db, make_engine
from recsys.store.repo import sync_catalog

app = typer.Typer(no_args_is_help=True, help="Recommender engine tools.")


@app.command("init-db")
def init_db_cmd() -> None:
    """Create database tables if they don't exist."""
    settings = Settings()
    init_db(make_engine(settings.database_url))
    typer.echo(f"database ready: {settings.database_url}")


@app.command("validate-catalog")
def validate_catalog(directory: Path) -> None:
    """Check a catalog directory for problems without touching the database."""
    try:
        catalog = load_catalog(directory)
    except CatalogError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"ok: {len(catalog.items)} items, {len(catalog.offers)} offers")


@app.command("load-catalog")
def load_catalog_cmd(directory: Path) -> None:
    """Validate a catalog directory and sync it into the database (idempotent)."""
    settings = Settings()
    try:
        catalog = load_catalog(directory)
    except CatalogError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    engine = make_engine(settings.database_url)
    init_db(engine)
    report = sync_catalog(engine, settings.tenant_id, catalog.items, catalog.offers)
    typer.echo(f"tenant {settings.tenant_id}: {report}")


@app.command()
def evaluate(
    catalog_dir: Path,
    interactions: Path,
    models: Annotated[
        str, typer.Option(help="Comma-separated model names from the registry")
    ] = ",".join(REGISTRY),
    k: Annotated[list[int] | None, typer.Option(help="Cutoffs; repeat for several")] = None,
    split: Annotated[str, typer.Option(help="temporal | last")] = "temporal",
    test_fraction: Annotated[float, typer.Option(help="temporal: share of latest data")] = 0.2,
    last_n: Annotated[int, typer.Option(help="last: positives held out per user")] = 1,
    baseline: Annotated[str, typer.Option(help="Model name others are compared to")] = (
        "most_popular"
    ),
    out: Annotated[Path, typer.Option(help="Where report .md/.json go")] = Path("artifacts/eval"),
    model_config: Annotated[
        Path | None, typer.Option(help="YAML of models + params (overrides --models)")
    ] = None,
) -> None:
    """Fit models on a time-based split of INTERACTIONS and report metrics with CIs."""
    try:
        catalog = load_catalog(catalog_dir)
    except CatalogError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    if model_config is not None:
        built = load_model_config(model_config)
    else:
        names = [n.strip() for n in models.split(",") if n.strip()]
        unknown = [n for n in names if n not in REGISTRY]
        if unknown:
            typer.echo(f"unknown models {unknown}; available: {sorted(REGISTRY)}", err=True)
            raise typer.Exit(1)
        built = [REGISTRY[n]() for n in names]

    data = load_interactions_csv(interactions)
    if split == "temporal":
        s = temporal_split(catalog.items, data, test_fraction=test_fraction)
    elif split == "last":
        s = leave_last_out(catalog.items, data, n=last_n)
    else:
        typer.echo("--split must be 'temporal' or 'last'", err=True)
        raise typer.Exit(1)

    report = run_evaluation(built, s, ks=tuple(k or (5, 10, 20)), baseline=baseline)
    name = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{s.info['fingerprint']}"
    md, js = write_report(report, out, name=name)
    typer.echo(to_markdown(report))
    typer.echo(f"saved {md} and {js}")


@app.command()
def lab(
    catalog_dir: Path,
    interactions: Annotated[
        Path | None, typer.Option(help="Interaction CSV: fits models and fills the scorecard")
    ] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Open the Rec Lab: personas → model → user experience, with a model-view overlay.

    Reads CATALOG_DIR/lab.yaml (models, personas) and CATALOG_DIR/quiz.yaml if present.
    """
    import uvicorn

    from recsys.lab.app import create_app
    from recsys.lab.core import Lab

    try:
        the_lab = Lab(catalog_dir, interactions)
    except (CatalogError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Rec Lab on http://{host}:{port}  (Ctrl+C to stop)")
    uvicorn.run(create_app(the_lab), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
