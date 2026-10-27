"""Stream OTTO (otto-de/recsys-dataset, 4 weeks, 11.3 GB JSONL in a zip) into a
compact local parquet, keeping a sample of sessions.

An OTTO "session" is one anonymous shopper's activity over the whole period,
so it plays the role of a user. Events are clicks, carts and orders on
anonymized article ids (no category, price or brand). Sessions are kept when
session % EVERY == 0, which samples shoppers, not events, so each kept
shopper's history is complete. Output stays local (license CC BY 4.0).

Columns: user (int32), t (int32, seconds since 2022-08-01 00:00 German
local time), kind (int8: 0 click, 1 cart, 2 order), product (int32).

Usage:
  python -I stream_otto_events.py train.zip OUT.parquet [EVERY=4]
"""

from __future__ import annotations

import json
import sys
import zipfile

import numpy as np
import pandas as pd

KINDS = {"clicks": 0, "carts": 1, "orders": 2}
T0_MS = 1659304800000  # 2022-07-31 22:00 UTC = 2022-08-01 00:00 CEST
BATCH = 200_000  # sessions per written batch


def main(src: str, out: str, every: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    cols: dict[str, list] = {"user": [], "t": [], "kind": [], "product": []}
    kept = seen = 0

    def flush() -> None:
        nonlocal writer
        if not cols["user"]:
            return
        df = pd.DataFrame(
            {
                "user": np.concatenate(cols["user"]).astype(np.int32),
                "t": np.concatenate(cols["t"]).astype(np.int32),
                "kind": np.concatenate(cols["kind"]).astype(np.int8),
                "product": np.concatenate(cols["product"]).astype(np.int32),
            }
        )
        table = pa.Table.from_pandas(df, preserve_index=False)
        writer = writer or pq.ParquetWriter(out, table.schema)
        writer.write_table(table)
        for v in cols.values():
            v.clear()

    with zipfile.ZipFile(src) as z, z.open(z.namelist()[0]) as f:
        for line in f:
            seen += 1
            # the session id leads each line: {"session":123,...
            sid = int(line[11 : line.index(b",")])
            if sid % every:
                continue
            ev = json.loads(line)["events"]
            n = len(ev)
            cols["user"].append(np.full(n, sid))
            cols["t"].append(np.array([(e["ts"] - T0_MS) // 1000 for e in ev]))
            cols["kind"].append(np.array([KINDS[e["type"]] for e in ev]))
            cols["product"].append(np.array([e["aid"] for e in ev]))
            kept += 1
            if kept % BATCH == 0:
                flush()
                print(f"  {seen:,} sessions read, {kept:,} kept", file=sys.stderr)
    flush()
    if writer:
        writer.close()
    print(f"done: {seen:,} sessions read, {kept:,} kept", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
