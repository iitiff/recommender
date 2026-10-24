"""Site visits by lifecycle cohort, from session-level data.

Usage:
  python -I visits_by_cohort.py retailrocket <prepared_dir>
  python -I visits_by_cohort.py rees46 <out_sessions.parquet>   # streams the 14.8 GB CSV first
  python -I visits_by_cohort.py analyze <sessions.parquet> [--days 30]

Each user is followed for their first N days after their first visit, and only
users first seen at least N days before the data ends are included, so a user
who arrived yesterday can't look like a low-visit user. Cohorts are defined by
what happened inside that window. An "order" is a calendar day with a purchase.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REES46_URL = (
    "https://huggingface.co/datasets/kevykibbz/ecommerce-behavior-data-from-multi-category-store_oct-nov_2019"
    "/resolve/main/ecommerce-behavior-data-from-multi-category-store_oct-nov_2019.csv"
)
BULK_ITEMS = 10  # more purchased items than this in the window: likely reseller/business


def sessions_retailrocket(prepared: Path) -> pd.DataFrame:
    ev = pd.read_csv(prepared / "interactions.csv", dtype={"user_id": str})
    ev["ts"] = pd.to_datetime(ev.occurred_at, utc=True, format="ISO8601")
    return (
        ev.assign(v=ev.kind.eq("view"), c=ev.kind.eq("add_to_cart"), b=ev.kind.eq("purchase"))
        .groupby("session_id")
        .agg(
            user_id=("user_id", "first"),
            start=("ts", "min"),
            views=("v", "sum"),
            carts=("c", "sum"),
            buys=("b", "sum"),
        )
        .reset_index(drop=True)
    )


def sessions_rees46(out: Path) -> pd.DataFrame:
    """Stream the CSV, aggregate per (user, session) chunk by chunk, then merge
    partial aggregates of sessions that straddle chunk boundaries."""
    cols = ["event_time", "event_type", "user_id", "user_session"]
    parts, events = [], 0
    stream = urllib.request.urlopen(REES46_URL)  # a stream, not the URL: see DATASETS.md
    for chunk in pd.read_csv(stream, usecols=cols, chunksize=4_000_000):
        events += len(chunk)
        chunk["sid"] = pd.util.hash_array(chunk.user_session.fillna("").to_numpy(dtype=str))
        chunk["ts"] = pd.to_datetime(chunk.event_time.str[:19], utc=True)
        parts.append(
            chunk.assign(
                v=chunk.event_type.eq("view"),
                c=chunk.event_type.eq("cart"),
                b=chunk.event_type.eq("purchase"),
            )
            .groupby(["user_id", "sid"])
            .agg(start=("ts", "min"), views=("v", "sum"), carts=("c", "sum"), buys=("b", "sum"))
            .reset_index()
        )
        print(f"events {events:,}", file=sys.stderr, flush=True)
    s = (
        pd.concat(parts, ignore_index=True)
        .groupby(["user_id", "sid"])
        .agg(
            start=("start", "min"),
            views=("views", "sum"),
            carts=("carts", "sum"),
            buys=("buys", "sum"),
        )
        .reset_index()
        .drop(columns="sid")
    )
    s.to_parquet(out)
    return s


def q(series: pd.Series) -> dict:
    if series.empty:
        return {}
    return {
        "mean": round(float(series.mean()), 2),
        "p50": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.9)),
    }


def analyze(s: pd.DataFrame, days: int) -> dict:
    window = pd.Timedelta(days=days)
    end = s.start.max()
    s = s.sort_values(["user_id", "start"], kind="stable")
    first_visit = s.groupby("user_id").start.transform("min")
    s = s[(first_visit <= end - window) & (s.start < first_visit + window)].copy()
    s["first_visit"] = first_visit[s.index]
    s["n"] = s.groupby("user_id").cumcount()  # visit number, 0-based
    s["day"] = s.start.dt.floor("D")

    users = s.groupby("user_id").agg(
        visits=("n", "size"), carts=("carts", "sum"), items=("buys", "sum")
    )
    buy_sessions = s[s.buys > 0]
    order_days = buy_sessions.groupby("user_id").day.nunique()
    users["orders"] = order_days.reindex(users.index).fillna(0).astype(int)

    cohort = np.select(
        [
            users["items"] > BULK_ITEMS,
            users.orders >= 3,
            users.orders == 2,
            users.orders == 1,
            users.carts > 0,
        ],
        [
            f"bulk buyer (>{BULK_ITEMS} items)",
            "3+ orders",
            "2 orders",
            "1 order",
            "carted, no order",
        ],
        default="browse only",
    )
    users["cohort"] = cohort

    # Visits up to and including the first purchase; visits from 1st to 2nd order.
    first_buy = buy_sessions.groupby("user_id").agg(
        n_first=("n", "min"), day_first=("day", "min"), t_first=("start", "min")
    )
    later = buy_sessions.merge(first_buy, on="user_id")
    later = later[later.day > later.day_first].groupby("user_id").n.min().rename("n_second")
    first_buy = first_buy.join(later)
    users = users.join(first_buy)
    users["visits_to_first_order"] = users.n_first + 1
    users["visits_first_to_second"] = users.n_second - users.n_first
    users["days_to_first_order"] = (
        users.t_first - s.groupby("user_id").first_visit.first()
    ).dt.total_seconds() / 86400

    order = [
        "browse only",
        "carted, no order",
        "1 order",
        "2 orders",
        "3+ orders",
        f"bulk buyer (>{BULK_ITEMS} items)",
    ]
    out = {"window_days": days, "users": int(len(users)), "cohorts": {}}
    for c in order:
        g = users[users.cohort == c]
        if g.empty:
            continue
        row = {
            "users": int(len(g)),
            "share": float(len(g) / len(users)),
            "visits": q(g.visits),
            "returned_2plus_visits": float((g.visits >= 2).mean()),
        }
        if g.orders.sum():
            row["visits_to_first_order"] = q(g.visits_to_first_order.dropna())
            row["bought_on_first_visit"] = float((g.visits_to_first_order == 1).mean())
            row["days_to_first_order_p50"] = float(g.days_to_first_order.median())
        if (g.orders >= 2).any():
            row["visits_first_to_second_order"] = q(g.visits_first_to_second.dropna())
        out["cohorts"][c] = row
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["retailrocket", "rees46", "analyze"])
    ap.add_argument("path", type=Path)
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    if args.mode == "retailrocket":
        sessions = sessions_retailrocket(args.path)
    elif args.mode == "rees46":
        sessions = sessions_rees46(args.path)
    else:
        sessions = pd.read_parquet(args.path)
    print(json.dumps(analyze(sessions, args.days), indent=2))
