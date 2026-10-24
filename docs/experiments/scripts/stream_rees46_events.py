"""Stream REES46 (Oct–Nov 2019, 14.8 GB CSV) into a compact local parquet.

Keeps every event with item-level fields as small integer columns (~3 bytes
per field instead of ~40 characters), so the whole two months fit in memory
for analysis. Strings (category codes, brands) become integer codes with a
JSON sidecar. Output stays local (dataset license: "© Original Authors").

Usage:
  uv run --no-project --with pandas --with pyarrow python -I \
    stream_rees46_events.py OUT.parquet
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

URL = (
    "https://huggingface.co/datasets/kevykibbz/ecommerce-behavior-data-from-multi-category-store_oct-nov_2019"
    "/resolve/main/ecommerce-behavior-data-from-multi-category-store_oct-nov_2019.csv"
)
EPOCH = pd.Timestamp("2019-10-01", tz="UTC")
KINDS = {"view": 0, "cart": 1, "purchase": 2}


def encode(values: pd.Series, table: dict[str, int]) -> pd.Series:
    """Map strings to stable integer codes (-1 = missing), growing `table`."""
    for v in values.dropna().unique():
        table.setdefault(v, len(table))
    return values.map(table).fillna(-1).astype("int32")


def main(out: Path) -> None:
    codes: dict[str, int] = {}
    brands: dict[str, int] = {}
    writer = None
    n = 0
    stream = urllib.request.urlopen(URL)  # a stream, not the URL: see DATASETS.md
    for chunk in pd.read_csv(
        stream, chunksize=4_000_000, dtype={"category_code": str, "brand": str}
    ):
        ts = pd.to_datetime(chunk.event_time.str[:19], utc=True)
        frame = pd.DataFrame(
            {
                "t": ((ts - EPOCH) / pd.Timedelta(seconds=1)).astype("int32"),
                "kind": chunk.event_type.map(KINDS).astype("int8"),
                "product": chunk.product_id.astype("int32"),
                "category_id": chunk.category_id.astype("int64"),
                "code": encode(chunk.category_code, codes),
                "brand": encode(chunk.brand, brands),
                "price": chunk.price.astype("float32"),
                "user": chunk.user_id.astype("int32"),
                "session": pd.util.hash_array(
                    chunk.user_session.fillna("").to_numpy(dtype=str)
                ).astype("int64"),
            }
        )
        table = pa.Table.from_pandas(frame, preserve_index=False)
        writer = writer or pq.ParquetWriter(out, table.schema, compression="zstd")
        writer.write_table(table)
        n += len(frame)
        print(f"events {n:,}", file=sys.stderr, flush=True)
    if writer:
        writer.close()
    out.with_suffix(".json").write_text(json.dumps({"category_codes": codes, "brands": brands}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
