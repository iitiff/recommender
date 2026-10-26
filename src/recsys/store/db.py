"""Engine creation and schema setup."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url

from recsys.store.tables import metadata


def make_engine(url: str) -> Engine:
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite" and parsed.database not in (None, "", ":memory:"):
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record) -> None:  # type: ignore[no-untyped-def]
            cur = dbapi_conn.cursor()
            # WAL lets the web app keep serving reads while events are written.
            cur.execute("PRAGMA journal_mode=WAL")
            # SQLite ignores foreign keys unless asked; Postgres always enforces them.
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.close()

    return engine


def init_db(engine: Engine) -> None:
    """Create any missing tables.

    `create_all` is enough while the schema is young. Alembic migrations come in
    before the first production deploy (see docs/ROADMAP.md), when changing a
    table must not lose data.
    """
    metadata.create_all(engine)
