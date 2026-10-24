"""Stream one Amazon Reviews 2023 category into compact local parquet files.

Source: McAuley Lab, UCSD (https://amazon-reviews-2023.github.io/). No
license is stated on the dataset card; earlier Amazon releases were
"non-commercial research only". Data stays local; only aggregates are reported.

Writes OUT/<Category>.ratings.parquet (user, item, rating, t) from the
"0-core rating_only" benchmark file, and OUT/<Category>.items.parquet
(item, cat2, cat3, price, brand) from the item metadata, skipping text and
images. User and item ids are hashed to int64 so the same shopper links
across categories without keeping millions of strings in memory.

Usage:
  uv run --no-project --with pandas --with pyarrow python -I stream_amazon.py Electronics OUT_DIR
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"


def h(values: pd.Series) -> pd.Series:
    return pd.Series(
        pd.util.hash_array(values.to_numpy(dtype=str)).astype("int64"), index=values.index
    )


def ratings(category: str, out: Path) -> int:
    parts, n = [], 0
    stream = urllib.request.urlopen(f"{BASE}/benchmark/0core/rating_only/{category}.csv")
    for chunk in pd.read_csv(stream, chunksize=5_000_000):
        parts.append(
            pd.DataFrame(
                {
                    "user": h(chunk.user_id),
                    "item": h(chunk.parent_asin),
                    "rating": chunk.rating.astype("int8"),
                    "t": (chunk.timestamp // 1000).astype("int64"),
                }
            )
        )
        n += len(chunk)
        print(f"{category} ratings {n:,}", file=sys.stderr, flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(out / f"{category}.ratings.parquet")
    return n


def price(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace("$", "").replace(",", "").split()[0])
    except (ValueError, IndexError):
        return None


def items(category: str, out: Path) -> int:
    rows = []
    stream = urllib.request.urlopen(f"{BASE}/raw/meta_categories/meta_{category}.jsonl")
    for n, line in enumerate(stream, start=1):
        m = json.loads(line)
        cats = m.get("categories") or []
        brand = (m.get("details") or {}).get("Brand") or m.get("store") or ""
        rows.append(
            (
                m["parent_asin"],
                " > ".join(cats[:2]),
                " > ".join(cats[:3]),
                price(m.get("price")),
                str(brand).strip().lower(),
            )
        )
        if n % 500_000 == 0:
            print(f"{category} items {n:,}", file=sys.stderr, flush=True)
    df = pd.DataFrame(rows, columns=["asin", "cat2", "cat3", "price", "brand"])
    df.insert(0, "item", h(df.asin))
    df.drop(columns="asin").to_parquet(out / f"{category}.items.parquet")
    return len(df)


if __name__ == "__main__":
    category, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {"category": category, "ratings": ratings(category, out), "items": items(category, out)}
        )
    )
