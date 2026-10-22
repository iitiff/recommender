"""Load interaction logs for offline evaluation.

CSV format (one row per interaction):
    user_id,item_id,kind,value,occurred_at
    u1,ridge-geyserville,rating,1,2026-03-01T18:22:00Z

`kind` is an InteractionKind (rating, save, click, affiliate_click,
not_interested). Public datasets get converted to this format once; later the
site's own feedback/events tables will feed the same harness.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from recsys.core.types import Interaction, InteractionKind


def _parse_time(text: str) -> datetime:
    ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def load_interactions_csv(path: Path) -> list[Interaction]:
    interactions: list[Interaction] = []
    problems: list[str] = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            try:
                interactions.append(
                    Interaction(
                        user_id=row["user_id"].strip(),
                        item_id=row["item_id"].strip(),
                        kind=InteractionKind(row["kind"].strip()),
                        value=float(row.get("value") or 1),
                        occurred_at=_parse_time(row["occurred_at"].strip()),
                    )
                )
            except (KeyError, ValueError) as exc:
                problems.append(f"line {line}: {exc}")
    if problems:
        raise ValueError(f"{path} has problems:\n  " + "\n  ".join(problems[:20]))
    return interactions


def write_interactions_csv(path: Path, interactions: list[Interaction]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "kind", "value", "occurred_at"])
        for x in interactions:
            value = int(x.value) if float(x.value).is_integer() else x.value
            writer.writerow([x.user_id, x.item_id, x.kind.value, value, x.occurred_at.isoformat()])
