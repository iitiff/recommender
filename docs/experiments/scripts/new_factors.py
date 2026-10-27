"""More first-order factors: is the first product repeatable, was it on sale,
when was the customer acquired, how much did they research first?

hm TRANSACTIONS_DIR ARTICLES (cohort: first order 2019-03-20..06-19 after 6+
  months without one, followed 365 days; loyal = 6+ orders and still buying
  in months 10-12, as in paths_to_loyalty.py):
  - repeatability of the first product: for each product type, the share of
    *established* customers (first seen in the data's first 90 days) who,
    after buying the type in Dec 2018-Jun 2019, bought it again within 180
    days. Computed on other customers, so it can't leak the cohort's outcome.
    The first order's repeatability = spend-weighted mean over its types.
  - bought on sale: a line is discounted when its price is below 90% of the
    article's median price over the whole data; share of the first order.
  - acquisition month: second order within 90 days, by month of the first
    order (Mar-Nov 2019, 6+ months of burn-in, follow-up before Mar 2020).
olist DIR (first orders 2017-01..2018-02, outcome: another order within 180
  days): acquisition week (Black Friday 2017) and month; repeatability of the
  first category, estimated on half of the customers and evaluated on the
  other half (split by a hash of the customer id).
rees46 events.parquet: new users (first seen on day 7+), first purchase on
  days 7-20; research before it (visits, days, comparison visits) vs a second
  order within 21 days (before the Nov 15-17 sale).

Usage:
  uv run --no-project --with pandas --with pyarrow python -I new_factors.py hm DIR ARTICLES
  uv run --no-project --with pandas python -I new_factors.py olist DIR
  uv run --no-project --with pandas --with pyarrow python -I new_factors.py rees46 PARQUET
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

MIN_CELL = 50


def r(x: float, n: int = 3) -> float:
    return round(float(x), n)


def table(df: pd.DataFrame, by: pd.Series, cols: list[str]) -> dict:
    out = {}
    for k, g in df.groupby(by, observed=True):
        if len(g) >= MIN_CELL:
            out[str(k)] = {"n": int(len(g)), "share": r(len(g) / len(df))} | {
                c: r(g[c].mean()) for c in cols
            }
    return out


# -- H&M ------------------------------------------------------------------------------


def hm(tx_dir: str, articles: str) -> dict:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    start = pd.Timestamp("2018-09-20")
    t = ds.dataset(tx_dir).to_table(columns=["t_dat", "customer_id", "article_id", "price"])
    lines = pd.DataFrame(
        {
            "c": pc.dictionary_encode(t["customer_id"]).combine_chunks().indices.to_numpy(),
            "d": (pd.to_datetime(t["t_dat"].to_pandas()) - start).dt.days.to_numpy(np.int16),
            "a": t["article_id"].to_numpy(),
            "price": t["price"].to_numpy().astype(np.float32),
        }
    )
    del t
    art = pq.read_table(articles, columns=["article_id", "product_type_name"]).to_pandas()
    lines["type"] = art.set_index("article_id").product_type_name.reindex(lines.a).to_numpy()
    median = lines.groupby("a").price.median()
    lines["sale"] = lines.price < 0.9 * median.reindex(lines.a).to_numpy()
    first = lines.groupby("c").d.min()
    lines["first"] = first.reindex(lines.c).to_numpy()

    # repeatability per product type, from established customers
    est = lines[(lines["first"] < 90) & lines.type.notna()]
    ct = est.groupby(["c", "type"]).d.agg(list)
    rep = {}
    for (_, typ), days in ct.items():
        days = sorted(set(days))
        w = [x for x in days if 75 <= x <= 272]  # Dec 2018 - Jun 2019
        if not w:
            continue
        rep.setdefault(typ, []).append(any(w[0] < x <= w[0] + 180 for x in days))
    repeat = pd.Series({k: np.mean(v) for k, v in rep.items() if len(v) >= 300})

    cohort = lines[lines["first"].between(181, 272) & (lines.d <= lines["first"] + 365)]
    orders = cohort.groupby(["c", "d"]).size().reset_index()
    orders["rel"] = orders.d - orders.c.map(first)
    n12 = orders.groupby("c").size()
    late = orders[orders.rel > 273].groupby("c").size().reindex(n12.index, fill_value=0)
    second = orders[orders.rel > 0].groupby("c").rel.min().reindex(n12.index)
    cust = pd.DataFrame(
        {
            "loyal": ((n12 >= 6) & (late > 0)).astype(int),
            "one_order": (n12 == 1).astype(int),
            "second_90d": (second <= 90).astype(int),
        }
    )
    f1 = cohort[cohort.d == cohort["first"]]
    f1 = f1.assign(rep=f1.type.map(repeat))
    w = f1.dropna(subset=["rep"])
    cust["first_repeatability"] = (w.rep * w.price).groupby(w.c).sum() / w.price.groupby(w.c).sum()
    cust["first_sale_share"] = f1.groupby("c").sale.mean()
    cols = ["one_order", "second_90d", "loyal"]
    q = pd.qcut(cust.first_repeatability.rank(method="first"), 5,
                labels=["Q1 least", "Q2", "Q3", "Q4", "Q5 most"])  # fmt: skip
    sale = pd.cut(cust.first_sale_share, [-1, 0, 0.5, 1],
                  labels=["full price", "partly on sale", "mostly on sale"])  # fmt: skip

    # acquisition month (Mar-Nov 2019), second order within 90 days
    acq = pd.DataFrame({"first": first[first.between(181, 437)]})
    o_all = lines[lines.c.isin(acq.index)].groupby(["c", "d"]).size().reset_index()
    o_all["rel"] = o_all.d - o_all.c.map(acq["first"])
    sec = o_all[o_all.rel > 0].groupby("c").rel.min().reindex(acq.index)
    acq["second_90d"] = (sec <= 90).astype(int)
    first_sale = lines[lines.c.isin(acq.index) & (lines.d == lines["first"])].groupby("c").sale
    acq["first_sale_share"] = first_sale.mean()
    month = (start + pd.to_timedelta(acq["first"], unit="D")).dt.to_period("M").astype(str)
    acq = acq[month.between("2019-03", "2019-11")]
    month = month.loc[acq.index]
    return {
        "cohort_customers": int(len(cust)),
        "product_types_with_repeatability": int(len(repeat)),
        "most_repeatable_types": {k: r(v) for k, v in repeat.sort_values().tail(10).items()},
        "least_repeatable_types": {k: r(v) for k, v in repeat.sort_values().head(10).items()},
        "by_first_order_repeatability": table(cust, q, cols),
        "by_first_order_sale": table(cust, sale, cols),
        "by_acquisition_month": table(acq, month, ["second_90d", "first_sale_share"]),
    }


# -- Olist ----------------------------------------------------------------------------


def olist(d: str) -> dict:
    d = Path(d)
    orders = pd.read_csv(d / "olist_orders_dataset.csv",
                         usecols=["order_id", "customer_id", "order_purchase_timestamp"],
                         parse_dates=["order_purchase_timestamp"])  # fmt: skip
    cust = pd.read_csv(d / "olist_customers_dataset.csv",
                       usecols=["customer_id", "customer_unique_id"])  # fmt: skip
    items = pd.read_csv(d / "olist_order_items_dataset.csv",
                        usecols=["order_id", "product_id", "price"])  # fmt: skip
    products = pd.read_csv(d / "olist_products_dataset.csv",
                           usecols=["product_id", "product_category_name"])  # fmt: skip
    names = pd.read_csv(d / "product_category_name_translation.csv")
    names = names.set_index("product_category_name").product_category_name_english
    o = orders.merge(cust, on="customer_id")
    o["day"] = o.order_purchase_timestamp.dt.floor("D")
    first_day = o.groupby("customer_unique_id").day.min()
    days = o.groupby("customer_unique_id").day.agg(lambda s: sorted(set(s)))
    x = pd.DataFrame({"first": first_day})
    x = x[(x["first"] >= "2017-01-01") & (x["first"] < "2018-03-01")]
    x["repeat_180d"] = [
        int(any(0 < (dd - f).days <= 180 for dd in days[c])) for c, f in x["first"].items()
    ]
    # main category of the first order (largest spend) and same-category repeat
    li = items.merge(o[["order_id", "customer_unique_id", "day"]], on="order_id")
    li = li.merge(products, on="product_id", how="left")
    li["cat"] = li.product_category_name.map(names).fillna(li.product_category_name)
    first_lines = li[li.day == li.customer_unique_id.map(first_day)]
    main = first_lines.sort_values("price").drop_duplicates("customer_unique_id", keep="last")
    x["cat"] = main.set_index("customer_unique_id").cat.reindex(x.index)
    later = li[li.day > li.customer_unique_id.map(first_day)]
    later = later[(later.day - later.customer_unique_id.map(first_day)).dt.days <= 180]
    same = later.merge(main[["customer_unique_id", "cat"]], on=["customer_unique_id", "cat"])
    x["same_cat_180d"] = x.index.isin(same.customer_unique_id).astype(int)
    half = pd.Series([zlib.crc32(c.encode()) % 2 for c in x.index], index=x.index)
    est = x[half == 0].groupby("cat").agg(n=("repeat_180d", "size"), rate=("repeat_180d", "mean"))
    est = est[est.n >= 300].rate
    ev = x[half == 1].copy()
    ev["cat_repeatability"] = ev.cat.map(est)
    ev = ev.dropna(subset=["cat_repeatability"])
    q = pd.qcut(ev.cat_repeatability.rank(method="first"), 5,
                labels=["Q1 least", "Q2", "Q3", "Q4", "Q5 most"])  # fmt: skip
    week = x["first"].dt.to_period("W-SUN").astype(str)
    bf = np.where(week == "2017-11-20/2017-11-26", "Black Friday week 2017", "other weeks")
    nov = x["first"].dt.to_period("M").astype(str) == "2017-11"
    return {
        "cohort_customers": int(len(x)),
        "repeat_180d": r(x.repeat_180d.mean(), 4),
        "by_black_friday_week": table(x, pd.Series(bf, index=x.index), ["repeat_180d"]),
        "november_2017_vs_other_months": table(
            x, pd.Series(np.where(nov, "Nov 2017", "other months"), index=x.index), ["repeat_180d"]
        ),
        "by_month": table(x, x["first"].dt.to_period("M").astype(str), ["repeat_180d"]),
        "category_repeatability_estimated_on_half": {
            k: r(v, 4) for k, v in est.sort_values().items()
        },  # fmt: skip
        "by_first_category_repeatability_other_half": table(
            ev, q, ["repeat_180d", "same_cat_180d"]
        ),
    }


# -- REES46: research before the first purchase ----------------------------------------


def rees46(path: str) -> dict:
    import pyarrow.dataset as ds

    data = ds.dataset(path)
    e = data.to_table(columns=["user", "t", "kind", "product", "category_id"]).to_pandas()
    e = e.sort_values(["user", "t"], kind="stable")
    e["day"] = e.t // 86400
    seen = e.groupby("user").day.min()
    buys = e[e.kind == 2]
    first_buy = buys.groupby("user").t.min()
    fb_day = first_buy // 86400
    users = fb_day[(fb_day >= 7) & (fb_day <= 20) & (seen.reindex(fb_day.index) >= 7)].index
    e = e[e.user.isin(users)]
    u, t = e.user.to_numpy(), e.t.to_numpy()
    new = np.ones(len(e), bool)
    new[1:] = (u[1:] != u[:-1]) | (t[1:] - t[:-1] > 1800)
    e = e.assign(visit=np.cumsum(new) - 1)
    ft = first_buy.reindex(users)
    before = e[e.t <= e.user.map(ft)]
    visits_before = before.groupby("user").visit.nunique()
    views = before[before.kind == 0]
    per = views.groupby(["visit", "category_id"])["product"].nunique().groupby(level=0).max()
    research = (per >= 3).groupby(before.groupby("visit").user.first().reindex(per.index)).sum()
    days_before = (ft // 86400 - seen.reindex(users)).astype(int)
    later = buys[buys.user.isin(users)]
    later = later[later.day > later.user.map(fb_day)]
    sec = (later.groupby("user").day.min() - fb_day.reindex(users)).reindex(users)
    x = pd.DataFrame(
        {
            "second_21d": (sec <= 21).astype(int),
            "visits_before": visits_before.reindex(users).fillna(1),
            "days_before": days_before,
            "research_visits_before": research.reindex(users).fillna(0),
        }
    )
    cols = ["second_21d"]
    return {
        "new_first_time_buyers": int(len(x)),
        "second_order_21d": r(x.second_21d.mean(), 4),
        "by_visits_before_first_purchase": table(
            x,
            pd.cut(
                x.visits_before, [0, 1, 3, 7, 1e9], labels=["1 (same visit)", "2-3", "4-7", "8+"]
            ),
            cols,
        ),  # fmt: skip
        "by_days_from_first_visit": table(
            x,
            pd.cut(x.days_before, [-1, 0, 3, 14, 1e9], labels=["same day", "1-3", "4-14", "15+"]),
            cols,
        ),  # fmt: skip
        "by_research_visits_before": table(
            x, pd.cut(x.research_visits_before, [-1, 0, 2, 1e9], labels=["0", "1-2", "3+"]), cols
        ),
    }


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    result = {"hm": hm, "olist": olist, "rees46": rees46}[mode](*args)
    print(json.dumps(result, indent=1, default=str))
