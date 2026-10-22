"""The single choke point for writing events.

Everything that records user behaviour goes through an `EventLogger`, so the
schema is enforced in one place and the storage backend can change (SQLite
today, a queue or warehouse later) without touching the app.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from sqlalchemy import Engine

from recsys.events.schema import Event
from recsys.store import repo


class EventLogger(Protocol):
    def log(self, event: Event) -> None: ...

    def log_many(self, events: Iterable[Event]) -> None: ...


class SqlEventLogger:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def log(self, event: Event) -> None:
        self.log_many([event])

    def log_many(self, events: Iterable[Event]) -> None:
        with self._engine.begin() as conn:
            repo.insert_events(conn, list(events))


class InMemoryEventLogger:
    """For tests and offline simulations."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def log(self, event: Event) -> None:
        self.events.append(event)

    def log_many(self, events: Iterable[Event]) -> None:
        self.events.extend(events)
