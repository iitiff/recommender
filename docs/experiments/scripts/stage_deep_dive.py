"""Lifecycle stages up close: is each stage one kind of shopper, or several?

Unit = a visit. For each visit we compute the shopper's state at the moment
it starts, from earlier visits only (what a live system would know), and
what happened next. If outcomes differ several-fold between sub-groups of one
stage, the stage is too wide to act on.

Usage:
  python -I stage_deep_dive.py retailrocket <prepared_dir>
  python -I stage_deep_dive.py rees46 <sessions.parquet>   # from visits_by_cohort.py

State at visit start (prior visits only):
- stage: new (first visit) / browser (returned, never carted) / abandoner
  (carted, never bought) / 1 order / 2 orders / 3+ orders / bulk (>10 items)
- days since last visit, prior visits, days since last order, open cart
  (a cart-without-purchase visit after the last order)
Outcomes: bought this visit, ordered within 7 days (this visit or later),
returned within 7 days.

Left-censoring: the first 30 days are burn-in (history only, not analyzed),
so fewer old customers pass as "new". Right-censoring: only visits with 7
days of data after them are analyzed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BURN_IN = pd.Timedelta(days=30)
HORIZON = pd.Timedelta(days=7)
BULK_ITEMS = 10
DAY = pd.Timedelta(days=1)


def visits_retailrocket(prepared: Path) -> pd.DataFrame:
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


def visits_rees46(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def state(v: pd.DataFrame) -> pd.DataFrame:
    """Add state-at-visit-start and outcome columns. Vectorized per user."""
    v = v.sort_values(["user_id", "start"], kind="stable").reset_index(drop=True)
    g = v.groupby("user_id", sort=False)
    v["prior_visits"] = g.cumcount()
    v["since_visit"] = (v.start - g.start.shift()) / DAY
    v["bought"] = v.buys > 0
    v["day"] = v.start.dt.floor("D")

    # Orders = distinct days with a purchase, counted over *earlier* visits.
    new_order_day = v.bought & ~v.duplicated(["user_id", "day", "bought"])
    v["prior_orders"] = new_order_day.astype(int).groupby(v.user_id).cumsum() - new_order_day
    v["prior_items"] = g.buys.cumsum() - v.buys
    v["prior_carts"] = g.carts.cumsum() - v.carts
    last_order = v.start.where(v.bought).groupby(v.user_id).ffill()
    v["since_order"] = (v.start - last_order.groupby(v.user_id).shift()) / DAY
    abandoned = v.start.where((v.carts > 0) & ~v.bought).groupby(v.user_id).ffill()
    prev_abandoned = abandoned.groupby(v.user_id).shift()
    prev_order = last_order.groupby(v.user_id).shift()
    v["open_cart"] = prev_abandoned.notna() & (prev_order.isna() | (prev_abandoned > prev_order))
    v["since_cart"] = (v.start - prev_abandoned) / DAY

    # Outcomes
    nxt = g.start.shift(-1)
    v["return_7d"] = (nxt - v.start) <= HORIZON
    next_buy = v.start.where(v.bought)[::-1].groupby(v.user_id[::-1]).ffill()[::-1]
    next_buy_after = next_buy.groupby(v.user_id).shift(-1)
    v["order_7d"] = v.bought | ((next_buy_after - v.start) <= HORIZON)

    v["stage"] = np.select(
        [
            v.prior_items > BULK_ITEMS,
            v.prior_orders >= 3,
            v.prior_orders == 2,
            v.prior_orders == 1,
            v.prior_carts > 0,
            v.prior_visits > 0,
        ],
        ["bulk", "3+ orders", "2 orders", "1 order", "abandoner", "browser"],
        default="new",
    )
    start, end = v.start.min(), v.start.max()
    return v[(v.start >= start + BURN_IN) & (v.start <= end - HORIZON)]


def bucket(x: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(x, [-np.inf, *edges, np.inf], labels=labels, right=True).astype(str)


DIMENSIONS = {
    "days since last visit": lambda v: bucket(
        v.since_visit, [1 / 24, 1, 7, 30], ["<1h", "1h–1d", "1–7d", "7–30d", ">30d"]
    ).where(v.since_visit.notna(), "first visit"),
    "prior visits": lambda v: bucket(
        v.prior_visits, [0, 2, 5, 15], ["0", "1–2", "3–5", "6–15", "16+"]
    ),
    "days since last order": lambda v: bucket(
        v.since_order, [1, 7, 30], ["<1d", "1–7d", "7–30d", ">30d"]
    ).where(v.since_order.notna(), "never ordered"),
    "open cart": lambda v: np.where(
        v.open_cart, np.where(v.since_cart <= 1, "yes, <1d old", "yes, older"), "no"
    ),
}


def rates(v: pd.DataFrame) -> dict:
    return {
        "visits": int(len(v)),
        "buy_this_visit": round(float(v.bought.mean()), 4),
        "order_7d": round(float(v.order_7d.mean()), 4),
        "return_7d": round(float(v.return_7d.mean()), 4),
        "views_per_visit": round(float(v.views.mean()), 2),
    }


def analyze(v: pd.DataFrame) -> dict:
    out: dict = {"visits": int(len(v)), "users": int(v.user_id.nunique()), "stages": {}}
    order = ["new", "browser", "abandoner", "1 order", "2 orders", "3+ orders", "bulk"]
    for s in order:
        sv = v[v.stage == s]
        if sv.empty:
            continue
        row = {"share": round(len(sv) / len(v), 4), **rates(sv), "splits": {}}
        for dim, f in DIMENSIONS.items():
            labels = pd.Series(f(sv), index=sv.index)
            if labels.nunique() < 2:
                continue
            row["splits"][dim] = {
                str(b): rates(sv[labels == b])
                for b in sorted(labels.unique())
                if (labels == b).sum() >= 30
            }
        out["stages"][s] = row
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["retailrocket", "rees46"])
    ap.add_argument("path", type=Path)
    args = ap.parse_args()
    visits = (
        visits_retailrocket(args.path)
        if args.dataset == "retailrocket"
        else visits_rees46(args.path)
    )
    print(json.dumps(analyze(state(visits)), indent=1))
