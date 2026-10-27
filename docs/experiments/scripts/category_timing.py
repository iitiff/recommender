"""When is each category bought? Category timing vs the store's overall rhythm.

Question: do categories have their own time-of-day / day-of-week patterns
(e.g. food around 5-6pm on Fridays), beyond the store's overall traffic
pattern? And on Instacart (groceries): do shoppers have stable personal
routines (day, hour, order cadence)?

Per category, from order-level presence (an order counts once per category):
- tv_hour / tv_dow / tv_week: total variation distance between the
  category's time profile and the store's overall profile (share of the
  category's orders that would have to move to match); minus the same
  distance expected by chance for a category of that size (simulated), so
  small categories don't look distinctive just from noise.
- peak 3-hour window (circular) and peak two days: the category's share in
  them vs the store's share (lift).

Usage (DIR holds the raw files; REES46 from stream_rees46_events.py):
  python -I category_timing.py instacart DIR    # orders, order_products, products, departments
  python -I category_timing.py olist DIR        # olist_*.csv, Brazil local time
  python -I category_timing.py rees46 events.parquet   # UTC, shown as Moscow time
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MIN_ORDERS = 300
BANDS = [
    (0, 7, "0-6"),
    (7, 10, "7-9"),
    (10, 13, "10-12"),
    (13, 16, "13-15"),
    (16, 19, "16-18"),
    (19, 24, "19-23"),
]


def bands(hours: np.ndarray) -> dict[str, float]:
    """Share of orders per band of hours (for lead-time questions: order now, receive later)."""
    return {
        label: round(float(((hours >= lo) & (hours < hi)).mean()), 4) for lo, hi, label in BANDS
    }


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def null_tv(q: np.ndarray, n: int, rng: np.random.Generator, reps: int = 50) -> float:
    """Expected distance of a random sample of n orders from the overall profile."""
    return float(np.mean([tv(rng.multinomial(n, q) / n, q) for _ in range(reps)]))


def profile(orders: pd.DataFrame) -> dict:
    """orders: one row per (order, category) with hour (0-23), dow (0-6 or label index)."""
    rng = np.random.default_rng(0)
    base_orders = orders.drop_duplicates("order")
    all_hour = np.bincount(base_orders.hour, minlength=24) / len(base_orders)
    all_dow = np.bincount(base_orders.dow, minlength=7) / len(base_orders)
    all_week = np.bincount(base_orders.dow * 24 + base_orders.hour, minlength=168) / len(
        base_orders
    )
    out = {
        "orders": int(len(base_orders)),
        "overall_hour": [round(float(x), 4) for x in all_hour],
        "overall_dow": [round(float(x), 4) for x in all_dow],
        "overall_hour_bands": bands(base_orders.hour.to_numpy()),
        "categories": {},
    }
    for cat, g in orders.groupby("category"):
        n = len(g)
        if n < MIN_ORDERS:
            continue
        h = np.bincount(g.hour, minlength=24) / n
        d = np.bincount(g.dow, minlength=7) / n
        w = np.bincount(g.dow * 24 + g.hour, minlength=168) / n
        win = h + np.roll(h, 1) + np.roll(h, -1)
        peak_h = int(win.argmax())
        base_win = all_hour + np.roll(all_hour, 1) + np.roll(all_hour, -1)
        top2 = np.argsort(-d)[:2]
        out["categories"][str(cat)] = {
            "orders": int(n),
            "tv_hour": round(tv(h, all_hour) - null_tv(all_hour, n, rng), 4),
            "tv_dow": round(tv(d, all_dow) - null_tv(all_dow, n, rng), 4),
            "tv_week": round(tv(w, all_week) - null_tv(all_week, n, rng), 4),
            "peak_3h_center": peak_h,
            "peak_3h_share": round(float(win[peak_h]), 4),
            "store_share_same_window": round(float(base_win[peak_h]), 4),
            "peak_days": [int(x) for x in top2],
            "peak_days_share": round(float(d[top2].sum()), 4),
            "store_share_same_days": round(float(all_dow[top2].sum()), 4),
            "hour_bands": bands(g.hour.to_numpy()),
        }
    return out


# -- loaders: one row per (order, category) with order, category, hour, dow ------


def instacart(d: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = pd.read_csv(
        d / "orders.csv",
        usecols=["order_id", "user_id", "eval_set", "order_number", "order_dow",
                 "order_hour_of_day", "days_since_prior_order"],
    )  # fmt: skip
    prior = orders[orders.eval_set == "prior"]
    products = pd.read_csv(d / "products.csv", usecols=["product_id", "department_id"])
    depts = pd.read_csv(d / "departments.csv").set_index("department_id").department
    dept_of = np.zeros(products.product_id.max() + 1, dtype=np.int16)
    dept_of[products.product_id.to_numpy()] = products.department_id.to_numpy()
    op = pd.read_csv(
        d / "order_products__prior.csv",
        usecols=["order_id", "product_id"],
        dtype={"order_id": np.int32, "product_id": np.int32},
    )
    key = np.unique(op.order_id.to_numpy(np.int64) * 64 + dept_of[op.product_id.to_numpy()])
    del op
    oc = pd.DataFrame({"order": key // 64, "dept": key % 64})
    meta = prior.set_index("order_id")[["order_dow", "order_hour_of_day"]]
    oc = oc.join(meta, on="order").dropna()
    rows = pd.DataFrame(
        {
            "order": oc.order.to_numpy(),
            "category": depts.reindex(oc.dept.to_numpy()).to_numpy(),
            # Instacart never documented which day 0 is: keep the raw index.
            "dow": oc.order_dow.astype(int).to_numpy(),
            "hour": oc.order_hour_of_day.astype(int).to_numpy(),
        }
    )
    return rows, prior


def olist(d: Path) -> tuple[pd.DataFrame, None]:
    orders = pd.read_csv(
        d / "olist_orders_dataset.csv", usecols=["order_id", "order_purchase_timestamp"]
    )
    items = pd.read_csv(d / "olist_order_items_dataset.csv", usecols=["order_id", "product_id"])
    products = pd.read_csv(
        d / "olist_products_dataset.csv", usecols=["product_id", "product_category_name"]
    )
    names = pd.read_csv(d / "product_category_name_translation.csv")
    names = names.set_index("product_category_name").product_category_name_english
    x = (
        items.merge(products, on="product_id")
        .merge(orders, on="order_id")
        .drop_duplicates(["order_id", "product_category_name"])
    )
    ts = pd.to_datetime(x.order_purchase_timestamp)
    return pd.DataFrame(
        {
            "order": x.order_id.to_numpy(),
            "category": x.product_category_name.map(names)
            .fillna(x.product_category_name)
            .to_numpy(),
            "dow": ts.dt.day_of_week.to_numpy(),  # 0 = Monday
            "hour": ts.dt.hour.to_numpy(),
        }
    ).dropna(), None


def rees46(path: Path) -> tuple[pd.DataFrame, None]:
    import pyarrow.parquet as pq

    names = json.loads(path.with_suffix(".json").read_text())["category_codes"]
    top = {i: n.split(".")[0] for n, i in names.items()}
    p = pq.read_table(path, columns=["user", "t", "code"], filters=[("kind", "=", 2)]).to_pandas()
    p = p[p.code >= 0]
    ts = (
        pd.Timestamp("2019-10-01", tz="UTC")
        + pd.to_timedelta(p.t, unit="s")
        + pd.Timedelta(hours=3)
    )
    day = ts.dt.floor("D")
    order = p.user.astype(np.int64).astype(str) + "|" + day.astype(str)  # order = customer-day
    rows = pd.DataFrame(
        {
            "order": order.to_numpy(),
            "category": p.code.map(top).to_numpy(),
            "dow": ts.dt.day_of_week.to_numpy(),
            "hour": ts.dt.hour.to_numpy(),
        }
    )
    return rows.drop_duplicates(["order", "category"]), None


# -- personal routines and cadence (Instacart) ------------------------------------


def routines(prior: pd.DataFrame, min_orders: int = 10) -> dict:
    """Learn each shopper's usual 3-hour window, day and gap from their first
    half of orders; test on their second half against store-wide defaults."""
    o = prior.sort_values(["user_id", "order_number"])
    n = o.groupby("user_id").order_number.transform("max")
    o = o[n >= min_orders]
    first = o[o.order_number <= o.groupby("user_id").order_number.transform("max") // 2]
    second = o.drop(first.index)

    def circ_near(h: np.ndarray, center: np.ndarray) -> np.ndarray:
        diff = np.abs(h - center) % 24
        return np.minimum(diff, 24 - diff) <= 1

    hist = pd.crosstab(first.user_id, first.order_hour_of_day).reindex(
        columns=range(24), fill_value=0
    )
    win = hist.to_numpy() + np.roll(hist.to_numpy(), 1, 1) + np.roll(hist.to_numpy(), -1, 1)
    personal_h = pd.Series(win.argmax(1), index=hist.index)
    allh = np.bincount(first.order_hour_of_day, minlength=24)
    global_h = int((allh + np.roll(allh, 1) + np.roll(allh, -1)).argmax())
    personal_d = first.groupby("user_id").order_dow.agg(lambda s: s.value_counts().index[0])
    global_d = int(first.order_dow.value_counts().index[0])

    h2 = second.order_hour_of_day.to_numpy()
    gaps1 = (
        first.dropna(subset=["days_since_prior_order"]).groupby("user_id").days_since_prior_order
    )
    med1 = gaps1.median()
    g2 = second.dropna(subset=["days_since_prior_order"])
    global_med = float(first.days_since_prior_order.median())
    err_personal = (g2.days_since_prior_order - g2.user_id.map(med1)).abs()
    err_global = (g2.days_since_prior_order - global_med).abs()
    gaps = prior.days_since_prior_order.dropna()
    return {
        "shoppers": int(o.user_id.nunique()),
        "later_orders": int(len(second)),
        "in_personal_3h_window": round(
            float(circ_near(h2, second.user_id.map(personal_h).to_numpy()).mean()), 3
        ),
        "in_store_peak_3h_window": round(
            float(circ_near(h2, np.full(len(h2), global_h)).mean()), 3
        ),
        "store_peak_hour_center": global_h,
        "on_personal_usual_day": round(
            float((second.order_dow == second.user_id.map(personal_d)).mean()), 3
        ),
        "on_store_busiest_day": round(float((second.order_dow == global_d).mean()), 3),
        "gap_days_median": float(gaps.median()),
        "gap_share_exactly_7_days": round(float((gaps == 7).mean()), 3),
        "gap_share_30_or_more_(capped)": round(float((gaps >= 30).mean()), 3),
        "next_gap_error_days_personal_median": round(float(err_personal.mean()), 2),
        "next_gap_error_days_store_median": round(float(err_global.mean()), 2),
    }


if __name__ == "__main__":
    mode, path = sys.argv[1], Path(sys.argv[2])
    rows, prior = {"instacart": instacart, "olist": olist, "rees46": rees46}[mode](path)
    result = {"dataset": mode, **profile(rows)}
    if prior is not None:
        result["routines"] = routines(prior)
    print(json.dumps(result, indent=1))
