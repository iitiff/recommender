"""The path to loyal *and* valuable: what the first 90 days say about where a new
customer ends up, and what each group's first year looks like (H&M).

Cohort and definitions as in paths_to_loyalty.py and loyalty_dimensions.py:
new customers (first order 2019-03-20..06-19 after 6+ months without one),
followed 365 days; an order is a customer-day; an item is "on sale" when its
price is below 90% of the article's median price.

12-month group (Reinartz & Kumar's grid, plus one-order customers):
  one order / strangers (not loyal, low spend) / butterflies (not loyal, high
  spend) / barnacles (loyal, low spend) / true friends (loyal, high spend);
  loyal = 6+ orders and still buying in months 10-12; high spend = at or
  above the repeaters' median 12-month spend.

1. Landmark at day 90: what is known by then (orders, days to the 2nd order,
   share of items on sale, channels used, product groups, spend) and the
   12-month group it leads to. Only information up to day 90 is used.
2. The most common day-90 profiles (2nd-order timing x channels x sale
   share) and the share of each that ends as true friends or barnacles.
3. Month-by-month paths: for each 12-month group, the share active, the
   share of items on sale, and spend per month.

Usage:
  uv run --no-project --with pandas --with pyarrow python -I \
    paths_to_value.py TRANSACTIONS_DIR ARTICLES_PARQUET > out.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

MIN_CELL = 50
GROUPS = ["one order", "strangers", "butterflies", "barnacles", "true friends"]


def r(x: float, n: int = 3) -> float:
    return round(float(x), n)


def load(tx_dir: str, articles: str) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    start = pd.Timestamp("2018-09-20")
    t = ds.dataset(tx_dir).to_table(
        columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    )
    lines = pd.DataFrame(
        {
            "c": pc.dictionary_encode(t["customer_id"]).combine_chunks().indices.to_numpy(),
            "d": (pd.to_datetime(t["t_dat"].to_pandas()) - start).dt.days.to_numpy(np.int16),
            "a": t["article_id"].to_numpy(),
            "price": t["price"].to_numpy().astype(np.float32),
            "online": t["sales_channel_id"].to_numpy() == 2,
        }
    )
    del t
    median = lines.groupby("a").price.median()
    lines["sale"] = lines.price < 0.9 * median.reindex(lines.a).to_numpy()
    first = lines.groupby("c").d.min()
    lines["rel"] = lines.d - first.reindex(lines.c).to_numpy()
    x = lines[lines.c.isin(first[first.between(181, 272)].index) & (lines.rel <= 365)].copy()
    art = pq.read_table(articles, columns=["article_id", "product_group_name"]).to_pandas()
    x["group"] = art.set_index("article_id").product_group_name.reindex(x.a).to_numpy()
    return x


def outcome(x: pd.DataFrame) -> pd.DataFrame:
    orders = x.groupby(["c", "d"]).rel.first().reset_index()
    n = orders.groupby("c").size()
    late = orders[orders.rel > 273].groupby("c").size().reindex(n.index, fill_value=0)
    cust = pd.DataFrame({"orders": n, "spend": x.groupby("c").price.sum()})
    cust["loyal"] = (n >= 6) & (late > 0)
    rep = cust.orders >= 2
    hi = cust.spend >= cust.spend[rep].median()
    cust["group"] = np.select(
        [~rep, cust.loyal & hi, cust.loyal & ~hi, hi],
        ["one order", "true friends", "barnacles", "butterflies"],
        "strangers",
    )
    return cust


def day90(x: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    e = x[x.rel <= 90]
    o = e.groupby(["c", "d"]).agg(rel=("rel", "first"), online=("online", "mean")).reset_index()
    f = pd.DataFrame(index=cust.index)
    n = o.groupby("c").size()
    f["orders_90d"] = pd.cut(n, [0, 1, 2, 1e9], labels=["1", "2", "3+"]).astype(str)
    second = o[o.rel > 0].groupby("c").rel.min().reindex(cust.index)
    timing = ["by day 30", "31-60", "61-90"]
    f["second_order"] = pd.cut(second, [0, 30, 60, 90], labels=timing).astype(object)
    f["second_order"] = f.second_order.fillna("none by day 90")
    online = o.groupby("c").online.agg(lambda s: ((s >= 0.5).any(), (s < 0.5).any()))
    f["channels"] = online.map(
        lambda t: "both" if t[0] and t[1] else ("online only" if t[0] else "store only")
    )
    sale = e.groupby("c").sale.mean()
    f["on_sale"] = pd.cut(sale, [-1, 0, 0.5, 1], labels=["none", "under half",
                                                         "half or more"]).astype(str)  # fmt: skip
    groups = e.groupby("c").group.nunique()
    f["product_groups"] = pd.cut(groups, [0, 1, 3, 100], labels=["1", "2-3", "4+"]).astype(str)
    spend = e.groupby("c").price.sum()
    f["spend_90d"] = pd.qcut(spend.rank(method="first"), 5,
                             labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).astype(str)  # fmt: skip
    return f


def mix(cust: pd.DataFrame, by: pd.Series, top: int | None = None) -> dict:
    out = {}
    groups = sorted(cust.groupby(by), key=lambda kg: -len(kg[1]))
    for k, g in groups[:top] if top else groups:
        if len(g) < MIN_CELL:
            continue
        shares = g.group.value_counts(normalize=True)
        out[str(k)] = {"n": int(len(g)), "share": r(len(g) / len(cust))} | {
            name: r(shares.get(name, 0)) for name in GROUPS
        }
    return out


def monthly(x: pd.DataFrame, cust: pd.DataFrame) -> dict:
    x = x.assign(month=np.minimum(x.rel // 30, 11), grp=cust.group.reindex(x.c).to_numpy())
    out = {}
    for name, g in x.groupby("grp"):
        size = (cust.group == name).sum()
        active = g.groupby("month").c.nunique().reindex(range(12), fill_value=0) / size
        sale = g.groupby("month").sale.mean().reindex(range(12))
        spend = g.groupby("month").price.sum().reindex(range(12), fill_value=0) / size
        out[name] = {
            "customers": int(size),
            "active_share_by_month": [r(v, 2) for v in active],
            "sale_share_by_month": [r(v, 2) for v in sale.fillna(0)],
            # spend per customer in the month / cohort average spend per customer-month
            "spend_index_by_month": [r(v * len(cust) * 12 / x.price.sum(), 2) for v in spend],
        }
    return out


def main(tx_dir: str, articles: str) -> dict:
    x = load(tx_dir, articles)
    cust = outcome(x)
    f = day90(x, cust)
    profile = f.second_order + " / " + f.channels + " / on sale: " + f.on_sale
    return {
        "cohort_customers": int(len(cust)),
        "groups": cust.group.value_counts(normalize=True).round(3).to_dict(),
        "by_day90": {c: mix(cust, f[c]) for c in f.columns},
        "top_day90_profiles": mix(cust, profile, top=15),
        "monthly_paths": monthly(x, cust),
    }


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2]), indent=1, default=str))
