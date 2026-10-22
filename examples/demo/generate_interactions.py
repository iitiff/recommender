"""Generate synthetic interactions for the demo catalog (deterministic).

SYNTHETIC USERS ARE FOR TESTING THE PIPELINE, NOT FOR COMPARING MODELS.
These users are generated from simple assumptions (a favourite grape, a
preferred body, an oak tolerance, plus a popularity prior). Any model built on
the same assumptions will look great here, which proves nothing about real
people. Real comparisons use the tasting panel, public datasets, and real logs.

What the data does exercise:
- a popularity skew (so MostPopular should beat Random, a useful sanity check)
- taste structure (so a content model has something to find)
- quiz-like first ratings on quiz-only (non-recommendable) wines
- likes, dislikes, saves, clicks and affiliate clicks spread over six months

    uv run python examples/demo/generate_interactions.py
"""

from __future__ import annotations

import csv
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
rng = random.Random(7)
START = datetime(2026, 1, 1, tzinfo=UTC)
N_USERS = 300


def load_items() -> list[dict[str, str]]:
    with (HERE / "items.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def match(row: dict[str, str], taste: tuple[set[str], int, int]) -> float:
    """How well a wine fits a synthetic user's taste, 0..1."""
    likes, body_pref, oak_max = taste
    grape = 1.0 if row["varietals"].split(";")[0] in likes else 0.0
    body = 1 - abs(int(row["body"]) - body_pref) / 4
    oak = 1.0 if int(row["oak"]) <= oak_max else 0.0
    return 0.5 * grape + 0.3 * body + 0.2 * oak


def main() -> None:
    items = load_items()
    varietals = sorted({row["varietals"].split(";")[0] for row in items})
    # Popularity prior: a few wines are much more popular than the rest (Zipf-like).
    order = list(range(len(items)))
    rng.shuffle(order)
    popularity = {items[i]["item_id"]: 1 / (rank + 1) ** 0.8 for rank, i in enumerate(order)}
    quiz_items = [row for row in items if row["region"] != "Demo Valley"]

    rows: list[tuple[str, str, str, float, datetime]] = []
    for u in range(N_USERS):
        user = f"demo-u{u:03d}"
        likes = set(rng.sample(varietals, k=rng.choice([1, 1, 2])))
        body_pref = rng.choice([2, 3, 4, 5])
        oak_max = rng.choice([2, 3, 5])

        taste = (likes, body_pref, oak_max)
        t = START + timedelta(days=rng.uniform(0, 150))
        # "Quiz": rate three well-known quiz-only wines first.
        for row in rng.sample(quiz_items, k=3):
            liked = match(row, taste) + rng.gauss(0, 0.1) > 0.55
            rows.append((user, row["item_id"], "rating", 1 if liked else -1, t))
            t += timedelta(minutes=1)

        # Then browse and react over the following weeks.
        weights = [popularity[row["item_id"]] * (0.3 + 2 * match(row, taste)) for row in items]
        n = 3 + int(rng.expovariate(1 / 7))
        seen: set[str] = set()
        for row in rng.choices(items, weights=weights, k=n):
            if row["item_id"] in seen:
                continue
            seen.add(row["item_id"])
            t += timedelta(days=rng.expovariate(1 / 6))
            score = match(row, taste) + rng.gauss(0, 0.15)
            item = row["item_id"]
            rows.append((user, item, "click", 1, t))
            if rng.random() < 0.25 and row["region"] == "Demo Valley":
                rows.append((user, item, "save", 1, t + timedelta(minutes=2)))
            if score > 0.6 and rng.random() < 0.3 and row["region"] == "Demo Valley":
                rows.append((user, item, "affiliate_click", 1, t + timedelta(minutes=3)))
            if rng.random() < 0.7:  # not everyone rates
                rows.append(
                    (user, item, "rating", 1 if score > 0.55 else -1, t + timedelta(days=5))
                )
            elif score < 0.3 and rng.random() < 0.3:
                rows.append((user, item, "not_interested", 1, t + timedelta(minutes=1)))

    rows.sort(key=lambda r: (r[4], r[0], r[1], r[2]))
    with (HERE / "interactions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "kind", "value", "occurred_at"])
        for user, item, kind, value, ts in rows:
            writer.writerow([user, item, kind, value, ts.isoformat(timespec="seconds")])
    print(f"wrote {len(rows)} interactions for {N_USERS} users to {HERE / 'interactions.csv'}")


if __name__ == "__main__":
    main()
