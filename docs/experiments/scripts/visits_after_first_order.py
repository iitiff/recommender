"""Do visits after a first order predict the second, and do intent visits matter more?

Landmark design (so visits can't be counted after the purchase they predict):
  - first-time buyers: first order day F (an order is a customer-day), with at
    least BURN days of data before it;
  - keep those with no second order in days F+1 .. F+EXPOSURE;
  - exposure: their visits (30 minutes of inactivity ends a visit) in that
    window, split into kinds;
  - outcome: an order in days F+EXPOSURE+1 .. F+EXPOSURE+OUTCOME.

Kinds of visit (no search logs in either dataset, so intent is a proxy):
  - focused: some product viewed 2+ times in the visit (considering an item);
  - research: 3+ different products viewed in one leaf category (comparing,
    the closest thing to a search results page here; REES46 only);
  - cart: an add-to-cart in the visit (OTTO only: REES46 logs carts fully
    only from Nov 8);
  - shallow: everything else (1-2 product views, no focus).

rees46 events.parquet: first orders on days 7-16 (Oct 8-17, 2019), 14-day
  exposure, 14-day outcome, so outcome windows end before the Nov 15-17 sale.
otto otto_events.parquet (25% shopper sample): first orders on days 3-13,
  7-day exposure, 7-day outcome.

Usage:
  uv run --no-project --with pandas --with pyarrow --with statsmodels python -I \
    visits_after_first_order.py {rees46|otto} PATH > out.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

GAP = 1800
DAY = 86400
VIEW, CART, BUY = 0, 1, 2
MIN_CELL = 50
CONFIG = {
    "rees46": {"first_days": (7, 16), "exposure": 14, "outcome": 14},
    "otto": {"first_days": (3, 13), "exposure": 7, "outcome": 7},
}


def r(x: float, n: int = 4) -> float:
    return round(float(x), n)


def load(mode: str, path: str, lo: int, hi: int) -> pd.DataFrame:
    """Events of shoppers whose first order day is in [lo, hi], sorted by user, t."""
    import pyarrow as pa
    import pyarrow.dataset as ds

    has_leaf = mode == "rees46"
    cols = ["user", "t", "kind", "product"] + (["category_id"] if has_leaf else [])
    data = ds.dataset(path)
    buys = data.to_table(columns=["user", "t"], filter=ds.field("kind") == BUY).to_pandas()
    first = (buys.t // DAY).groupby(buys.user).min()
    users = first[(first >= lo) & (first <= hi)].index.to_numpy()
    e = data.to_table(columns=cols, filter=ds.field("user").isin(pa.array(users))).to_pandas()
    e["t"] = e.t.astype(np.int64)
    e = e.rename(columns={"category_id": "leaf"})
    if not has_leaf:
        e["leaf"] = -1
    return e.sort_values(["user", "t"], kind="stable").reset_index(drop=True)


def visits(e: pd.DataFrame) -> pd.DataFrame:
    u, t = e.user.to_numpy(), e.t.to_numpy()
    new = np.ones(len(e), bool)
    new[1:] = (u[1:] != u[:-1]) | (t[1:] - t[:-1] > GAP)
    e = e.assign(visit=np.cumsum(new) - 1)
    g = e.groupby("visit")
    v = pd.DataFrame(
        {
            "user": g.user.first(),
            "day": g.t.first() // DAY,
            "views": g.kind.agg(lambda s: int((s == VIEW).sum())),
            "carts": g.kind.agg(lambda s: int((s == CART).sum())),
            "buys": g.kind.agg(lambda s: int((s == BUY).sum())),
        }
    )
    views = e[e.kind == VIEW]
    v["max_views_one_product"] = (
        views.groupby(["visit", "product"]).size().groupby(level=0).max().reindex(v.index)
    ).fillna(0)
    per_leaf = views[views.leaf >= 0].groupby(["visit", "leaf"])["product"].nunique()
    v["max_products_one_leaf"] = per_leaf.groupby(level=0).max().reindex(v.index).fillna(0)
    return v


def analyse(mode: str, path: str) -> dict:
    cfg = CONFIG[mode]
    lo, hi = cfg["first_days"]
    ex, oc = cfg["exposure"], cfg["outcome"]
    e = load(mode, path, lo, hi)
    v = visits(e)
    order_days = v[v.buys > 0].groupby("user").day.agg(sorted)
    first = order_days.map(lambda ds: ds[0])
    rel = v.day - v.user.map(first)

    # shoppers with no second order during exposure
    def has_order(days: list[int], f: int, a: int, b: int) -> bool:
        return any(f + a <= d <= f + b for d in days)

    keep = [u for u, ds in order_days.items() if not has_order(ds, ds[0], 1, ex)]
    cust = pd.DataFrame(index=pd.Index(keep, name="user"))
    cust["y"] = [int(has_order(order_days[u], first[u], ex + 1, ex + oc)) for u in keep]

    w = v[(rel >= 1) & (rel <= ex) & v.user.isin(keep)].copy()
    focused = w.max_views_one_product >= 2
    research = w.max_products_one_leaf >= 3
    cart = (w.carts > 0) & (mode == "otto")
    w["kind"] = np.select([cart, research, focused], ["cart", "research", "focused"], "shallow")
    counts = w.groupby(["user", "kind"]).size().unstack(fill_value=0)
    for k in ("shallow", "focused", "research", "cart"):
        cust[k] = counts.get(k, pd.Series(dtype=int)).reindex(cust.index, fill_value=0)
    cust["visits"] = cust[["shallow", "focused", "research", "cart"]].sum(axis=1)
    cust["intent"] = cust[["focused", "research", "cart"]].sum(axis=1)
    last = w.groupby("user").day.max().reindex(cust.index)
    landmark = first.reindex(cust.index) + ex
    cust["days_since_last_visit"] = (landmark - last).fillna(ex + 1)

    def table(by: pd.Series) -> dict:
        out = {}
        for k, g in cust.groupby(by, observed=True):
            if len(g) >= MIN_CELL:
                out[str(k)] = {"n": int(len(g)), "share": r(len(g) / len(cust), 3),
                               "second_order": r(g.y.mean())}  # fmt: skip
        return out

    vb = pd.cut(cust.visits, [-1, 0, 1, 3, 7, 1e9], labels=["0", "1", "2-3", "4-7", "8+"])
    ib = pd.cut(cust.intent, [-1, 0, 1, 1e9], labels=["0 intent", "1 intent", "2+ intent"])
    out = {
        "dataset": mode,
        "config": cfg,
        "first_time_buyers": int(len(order_days)),
        "no_2nd_order_during_exposure": int(len(cust)),
        "base_rate_second_order": r(cust.y.mean()),
        "by_visits": table(vb),
        "by_visits_and_intent": table(vb.astype(str) + " visits / " + ib.astype(str)),
        "by_recency": table(
            pd.cut(
                cust.days_since_last_visit,
                [-1, 2, 6, ex, 1e9],
                labels=["0-2 days", "3-6", f"7-{ex}", "no visit"],
            )  # fmt: skip
        ),
        "kind_shares_of_visits": {
            k: r(cust[k].sum() / max(cust.visits.sum(), 1), 3)
            for k in ("shallow", "focused", "research", "cart")
        },
        "logit": logit(cust, mode),
    }
    return out


def logit(cust: pd.DataFrame, mode: str) -> dict:
    """Odds ratio per doubling of each kind of visit (log2(1 + count)), adjusted
    for each other and for recency."""
    import statsmodels.formula.api as smf

    x = cust.copy()
    kinds = ["shallow", "focused"] + (["research"] if mode == "rees46" else ["cart"])
    for k in kinds:
        x[f"l_{k}"] = np.log2(1 + x[k])
    x["recent"] = (x.days_since_last_visit <= 2).astype(int)
    formula = "y ~ " + " + ".join(f"l_{k}" for k in kinds) + " + recent"
    m = smf.logit(formula, data=x).fit(disp=False)
    ci = m.conf_int()
    return {
        k: {"odds_ratio": r(np.exp(m.params[k]), 3),
            "ci95": [r(np.exp(ci.loc[k, 0]), 3), r(np.exp(ci.loc[k, 1]), 3)]}
        for k in m.params.index if k != "Intercept"
    } | {"n": int(m.nobs)}  # fmt: skip


if __name__ == "__main__":
    print(json.dumps(analyse(sys.argv[1], sys.argv[2]), indent=1, default=str))
