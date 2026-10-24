"""Stream the 14.8 GB REES46 Oct–Nov 2019 CSV and keep only purchase events.

Usage: python -I stream_rees46_purchases.py <out.csv>
"""

import sys
import urllib.request

import pandas as pd

URL = (
    "https://huggingface.co/datasets/kevykibbz/ecommerce-behavior-data-from-multi-category-store_oct-nov_2019"
    "/resolve/main/ecommerce-behavior-data-from-multi-category-store_oct-nov_2019.csv"
)
COLS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "user_id",
    "user_session",
]

out = sys.argv[1]
rows = events = 0
first = True
# Pass an open stream, not the URL: given a URL, pandas downloads the whole
# file into memory first (14.8 GB here), while a file-like object is read
# incrementally chunk by chunk.
stream = urllib.request.urlopen(URL)
for chunk in pd.read_csv(stream, usecols=COLS, chunksize=2_000_000, dtype={"category_code": str}):
    events += len(chunk)
    buys = chunk[chunk.event_type == "purchase"].drop(columns="event_type")
    rows += len(buys)
    buys.to_csv(out, mode="w" if first else "a", header=first, index=False)
    first = False
    print(f"events {events:,}  purchases {rows:,}", flush=True)
print("done")
