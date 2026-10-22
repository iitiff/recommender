"""Command-line entry point: `recsys --help`."""

from __future__ import annotations

from pathlib import Path

import typer

from recsys.catalog import CatalogError, load_catalog
from recsys.config import Settings
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


if __name__ == "__main__":
    app()
