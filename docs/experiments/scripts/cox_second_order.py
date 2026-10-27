"""Time to the second order with visits that change day by day (Cox model).

Survival analysis instead of fixed windows: each first-time buyer is followed
day by day after their first order until a second order or the end of
follow-up (censored). Concept: Cox proportional hazards with time-varying
covariates (lifelines CoxTimeVaryingFitter), data in counting-process form
(one row per customer-day: start, stop, event, covariates).

Covariates on day k use only days before k (never the day of the order):
  - visits in the previous 7 days, by kind (shallow / focused / research or
    cart, as in visits_after_first_order.py), log2(1 + count);
  - visited yesterday or the day before (recency).
Days since the first order is the model's time axis (the baseline hazard).
Fixed: items in the first order (log2).

rees46: first orders on days 7-16, followed 28 days (before the Nov 15-17 sale).
otto: first orders on days 3-13, followed 14 days.
A random sample of first-time buyers keeps the fit fast.

Usage:
  uv run --no-project --with pandas --with pyarrow --with lifelines python -I \
    cox_second_order.py {rees46|otto} EVENTS.parquet > out.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

GAP, DAY = 1800, 86400
VIEW, CART, BUY = 0, 1, 2
CONFIG = {
    "rees46": {"first_days": (7, 16), "follow": 28},
    "otto": {"first_days": (3, 13), "follow": 14},
}
SAMPLE = 40_000


def r(x: float, n: int = 3) -> float:
    return round(float(x), n)


def load(mode: str, path: str, lo: int, hi: int) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.dataset as ds

    data = ds.dataset(path)
    buys = data.to_table(columns=["user", "t"], filter=ds.field("kind") == BUY).to_pandas()
    first = (buys.t // DAY).groupby(buys.user).min()
    users = first[(first >= lo) & (first <= hi)].index.to_numpy()
    rng = np.random.default_rng(0)
    users = rng.choice(users, size=min(SAMPLE, len(users)), replace=False)
    cols = ["user", "t", "kind", "product"] + (["category_id"] if mode == "rees46" else [])
    e = data.to_table(columns=cols, filter=ds.field("user").isin(pa.array(users))).to_pandas()
    e["t"] = e.t.astype(np.int64)
    e = e.rename(columns={"category_id": "leaf"})
    if "leaf" not in e:
        e["leaf"] = -1
    return e.sort_values(["user", "t"], kind="stable").reset_index(drop=True)


def daily_visits(e: pd.DataFrame, mode: str) -> pd.DataFrame:
    """One row per (user, day) with counts of visits by kind and an order flag."""
    u, t = e.user.to_numpy(), e.t.to_numpy()
    new = np.ones(len(e), bool)
    new[1:] = (u[1:] != u[:-1]) | (t[1:] - t[:-1] > GAP)
    e = e.assign(visit=np.cumsum(new) - 1)
    g = e.groupby("visit")
    v = pd.DataFrame({"user": g.user.first(), "day": g.t.first() // DAY})
    v["carts"] = g.kind.agg(lambda s: int((s == CART).sum()))
    v["buys"] = g.kind.agg(lambda s: int((s == BUY).sum()))
    v["items"] = v.buys
    views = e[e.kind == VIEW]
    focus = views.groupby(["visit", "product"]).size().groupby(level=0).max()
    leaf = views[views.leaf >= 0].groupby(["visit", "leaf"])["product"].nunique()
    v["focused"] = focus.reindex(v.index).fillna(0) >= 2
    v["research"] = leaf.groupby(level=0).max().reindex(v.index).fillna(0) >= 3
    strong = v.carts > 0 if mode == "otto" else v.research
    v["kind"] = np.select([strong, v.focused], ["intent", "focused"], "shallow")
    d = v.groupby(["user", "day", "kind"]).size().unstack(fill_value=0)
    for k in ("shallow", "focused", "intent"):
        if k not in d:
            d[k] = 0
    d["order"] = v.groupby(["user", "day"]).buys.sum().reindex(d.index).fillna(0) > 0
    d["items"] = v.groupby(["user", "day"]).items.sum().reindex(d.index).fillna(0)
    return d.reset_index()


def counting_process(d: pd.DataFrame, follow: int) -> pd.DataFrame:
    """Vectorized: dense user x day arrays relative to the first order (day 0)."""
    first = d[d.order].groupby("user").day.min()
    d = d[d.user.isin(first.index)].copy()
    d["rel"] = d.day - d.user.map(first)
    users = first.index.to_numpy()
    ui = pd.Index(users)
    items = d[d.rel == 0].set_index("user")["items"].reindex(users).fillna(1).to_numpy()
    after = d[(d.rel >= 1) & (d.rel <= follow)]
    row, col = ui.get_indexer(after.user), after.rel.to_numpy()

    def dense(values: np.ndarray) -> np.ndarray:
        a = np.zeros((len(users), follow + 1))
        np.add.at(a, (row, col), values)
        return a

    kinds = {k: dense(after[k].to_numpy(float)) for k in ("shallow", "focused", "intent")}
    order = dense(after.order.to_numpy(float)) > 0
    any_visit = sum(kinds.values()) > 0
    first_event = np.where(order.any(1), order.argmax(1), follow + 1)  # 0 never (rel >= 1)
    frames = []
    for k in range(1, follow + 1):
        alive = first_event >= k
        lo = max(1, k - 7)
        win = {name: a[:, lo:k].sum(1) for name, a in kinds.items()}
        recent = any_visit[:, max(1, k - 2) : k].any(1)
        frames.append(
            pd.DataFrame(
                {
                    "user": users,
                    "start": k - 1,
                    "stop": k,
                    "event": (first_event == k).astype(int),
                    "shallow_7d": np.log2(1 + win["shallow"]),
                    "focused_7d": np.log2(1 + win["focused"]),
                    "intent_7d": np.log2(1 + win["intent"]),
                    "visited_last_2d": recent.astype(int),
                    "first_order_items": np.log2(1 + items),
                }
            )[alive]
        )
    return pd.concat(frames, ignore_index=True)


def fit(cp: pd.DataFrame) -> dict:
    from lifelines import CoxTimeVaryingFitter

    m = CoxTimeVaryingFitter(penalizer=0.0)
    m.fit(cp, id_col="user", start_col="start", stop_col="stop", event_col="event")
    s = m.summary
    return {
        "hazard_ratios": {
            k: {
                "hr": r(s.loc[k, "exp(coef)"]),
                "ci95": [r(s.loc[k, "exp(coef) lower 95%"]), r(s.loc[k, "exp(coef) upper 95%"])],
                "p": float(f"{s.loc[k, 'p']:.2g}"),
            }
            for k in s.index
        },  # fmt: skip
        "log_likelihood_ratio_test_p": float(f"{m.log_likelihood_ratio_test().p_value:.2g}"),
    }


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    cfg = CONFIG[mode]
    e = load(mode, path, *cfg["first_days"])
    cp = counting_process(daily_visits(e, mode), cfg["follow"])
    events = cp.groupby("user").event.max()
    out = {
        "dataset": mode,
        "config": cfg,
        "customers": int(cp.user.nunique()),
        "customer_days": int(len(cp)),
        "second_orders": int(events.sum()),
        "intent_means": "carts" if mode == "otto" else "comparison of 3+ products in a category",
    } | fit(cp)
    print(json.dumps(out, indent=1))
