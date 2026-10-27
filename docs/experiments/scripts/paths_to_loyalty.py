"""From first order to loyal: how many paths are there, which are common, how they
look at category level, and who stops after one order.

Datasets (each loader returns orders = one row per customer-day):
  hm           H&M fashion, 2 years. Cohort: first order 2019-03-20..06-19 after 6+
               months with none; product group = category; store vs online.
  cdnow        CDNOW music, the cohort of 23,570 *first-ever* buyers of Q1 1997,
               followed to June 1998. No categories or channel.
  x5           X5 grocery. Cohort: loyalty card issued and first order between
               2018-11-22 and 2018-12-16 (true start), followed 90 days only.
               Category = product level 2 (names are hashed in the data).
  olist        Olist marketplace (Brazil). Cohort: first order 2017-03-01..08-31
               (the marketplace was ramping up before); category = product category.
  onlineretail UK gift wholesaler (B2B). Cohort: first order 2010-06-01..11-30
               after 6+ months with none. No categories.
  instacart    Grocery, shoppers with 4+ orders only (survivorship), so not a
               loyalty cohort: how category breadth forms among shoppers who stay.

Per dataset (as far as its fields and follow-up allow):
1. Outcomes: tiers at 12 months (loyal = 6+ orders and still buying in months
   10-12), one-order share at 90 and 365 days, the order funnel.
2. Second order: weekly hazard (chance in week w, if not yet), late returns,
   and the 12-month outcome by days to the second order.
3. Activity paths (12 thirty-day months after the first order): distinct
   patterns, concentration, rule-based families.
4. Step paths for the first three orders: gap, new category or not, channel.
5. First-order traits vs outcome: channel, categories, items, spend quintile.
6. Category level: outcome by the first order's main category (largest spend),
   what the second order buys, which categories are added next.

Behavior only: customer attributes (age, gender, club status, location) and
H&M index groups (Ladieswear/Menswear/Baby) are not used.

Usage:
  uv run --no-project --with pandas --with pyarrow python -I \
    paths_to_loyalty.py hm TRANSACTIONS_DIR ARTICLES_PARQUET > hm.json
  python -I paths_to_loyalty.py cdnow CDNOW_master.txt > cdnow.json
  python -I paths_to_loyalty.py x5 X5_DIR > x5.json
  python -I paths_to_loyalty.py olist OLIST_DIR > olist.json
  python -I paths_to_loyalty.py onlineretail online_retail_II.csv > onlineretail.json
  python -I paths_to_loyalty.py instacart INSTACART_DIR > instacart.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MIN_CELL = 50  # never report a cell with fewer customers
LOYAL_ORDERS = 6
TOP = 15


def r(x: float, n: int = 3) -> float:
    return round(float(x), n)


@dataclass
class Data:
    """orders: c, d (days since the dataset start), items, spend, online (NaN if
    unknown). cats: c, d, cat, spend (one row per order and category), or None."""

    name: str
    orders: pd.DataFrame
    cats: pd.DataFrame | None
    cohort: tuple[int, int]  # first-order day range [lo, hi)
    horizon: int  # days of follow-up after the first order
    notes: dict = field(default_factory=dict)


# -- loaders --------------------------------------------------------------------------


def load_hm(tx_dir: str, articles: str) -> Data:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    start, cohort, horizon = pd.Timestamp("2018-09-20"), (181, 273), 365
    t = ds.dataset(tx_dir).to_table(
        columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    )
    c = pc.dictionary_encode(t["customer_id"]).combine_chunks().indices.to_numpy()
    d = (pd.to_datetime(t["t_dat"].to_pandas()) - start).dt.days.to_numpy().astype(np.int16)
    first = pd.Series(d).groupby(c).min()
    in_cohort = first.between(cohort[0], cohort[1] - 1)
    keep = in_cohort.reindex(c).to_numpy() & (d <= first.reindex(c).to_numpy() + horizon)
    a = pq.read_table(articles, columns=["article_id", "product_group_name"]).to_pandas()
    group = a.set_index("article_id").product_group_name
    lines = pd.DataFrame(
        {
            "c": c[keep],
            "d": d[keep].astype(np.int32),
            "cat": group.reindex(t["article_id"].to_numpy()[keep]).to_numpy(),
            "online": t["sales_channel_id"].to_numpy()[keep] == 2,
            "spend": t["price"].to_numpy()[keep],
        }
    )
    del t
    orders = lines.groupby(["c", "d"]).agg(
        items=("cat", "size"), spend=("spend", "sum"), online=("online", "mean")
    )
    cats = lines.groupby(["c", "d", "cat"]).spend.sum().reset_index()
    return Data(
        "hm",
        orders.reset_index(),
        cats,
        cohort,
        horizon,
        {"channel": "sales_channel_id 2 read as online, 1 as store (undocumented)"},
    )


def load_cdnow(path: str) -> Data:
    x = pd.read_csv(path, sep=r"\s+", header=None, names=["c", "date", "items", "spend"],
                    dtype={"date": str})  # fmt: skip
    x["d"] = (pd.to_datetime(x.date, format="%Y%m%d") - pd.Timestamp("1997-01-01")).dt.days
    orders = x.groupby(["c", "d"]).agg(items=("items", "sum"), spend=("spend", "sum"))
    orders = orders.reset_index()
    orders["online"] = np.nan
    return Data("cdnow", orders, None, (0, 90), 365, {"newness": "true first-ever purchases"})


def load_x5(d: str) -> Data:
    d = Path(d)
    start = pd.Timestamp("2018-11-21")
    clients = pd.read_csv(d / "clients.csv.gz", usecols=["client_id", "first_issue_date"],
                          parse_dates=["first_issue_date"])  # fmt: skip
    new = clients[clients.first_issue_date >= start + pd.Timedelta(days=1)].client_id
    products = pd.read_csv(d / "products.csv.gz", usecols=["product_id", "level_2"])
    level2 = products.set_index("product_id").level_2
    parts = []
    cols = ["client_id", "transaction_datetime", "product_id", "trn_sum_from_iss",
            "product_quantity"]  # fmt: skip
    for chunk in pd.read_csv(d / "purchases.csv.gz", usecols=cols, chunksize=5_000_000):
        chunk = chunk[chunk.client_id.isin(new)]
        parts.append(
            pd.DataFrame(
                {
                    "c": chunk.client_id.to_numpy(),
                    "d": (pd.to_datetime(chunk.transaction_datetime) - start).dt.days.to_numpy(),
                    "cat": ("L2-" + level2.reindex(chunk.product_id).str[:4]).to_numpy(),
                    "spend": chunk.trn_sum_from_iss.to_numpy(),
                    "qty": chunk.product_quantity.to_numpy(),
                }
            )
        )
    lines = pd.concat(parts)
    orders = lines.groupby(["c", "d"]).agg(items=("qty", "size"), spend=("spend", "sum"))
    orders = orders.reset_index()
    orders["online"] = np.nan
    cats = lines.groupby(["c", "d", "cat"]).spend.sum().reset_index()
    return Data(
        "x5",
        orders,
        cats,
        (1, 26),
        90,
        {"newness": "loyalty card issued in the window (true start)",
         "categories": "product level 2, names hashed in the source"},
    )  # fmt: skip


def load_olist(d: str) -> Data:
    d = Path(d)
    start = pd.Timestamp("2016-09-01")
    cust = pd.read_csv(d / "olist_customers_dataset.csv",
                       usecols=["customer_id", "customer_unique_id"])  # fmt: skip
    orders = pd.read_csv(d / "olist_orders_dataset.csv",
                         usecols=["order_id", "customer_id", "order_status",
                                  "order_purchase_timestamp"])  # fmt: skip
    orders = orders[~orders.order_status.isin(["canceled", "unavailable"])]
    items = pd.read_csv(d / "olist_order_items_dataset.csv",
                        usecols=["order_id", "product_id", "price"])  # fmt: skip
    products = pd.read_csv(d / "olist_products_dataset.csv",
                           usecols=["product_id", "product_category_name"])  # fmt: skip
    names = pd.read_csv(d / "product_category_name_translation.csv")
    names = names.set_index("product_category_name").product_category_name_english
    x = items.merge(orders, on="order_id").merge(cust, on="customer_id")
    x = x.merge(products, on="product_id", how="left")
    x["cat"] = x.product_category_name.map(names).fillna(x.product_category_name).fillna("unknown")
    x["c"] = x.customer_unique_id
    x["d"] = (pd.to_datetime(x.order_purchase_timestamp) - start).dt.days
    o = x.groupby(["c", "d"]).agg(items=("cat", "size"), spend=("price", "sum")).reset_index()
    o["online"] = np.nan
    cats = x.groupby(["c", "d", "cat"]).price.sum().rename("spend").reset_index()
    lo = (pd.Timestamp("2017-03-01") - start).days
    hi = (pd.Timestamp("2017-09-01") - start).days
    return Data("olist", o, cats, (lo, hi), 365, {"newness": "marketplace launched 2016"})


def load_onlineretail(path: str) -> Data:
    start = pd.Timestamp("2009-12-01")
    x = pd.read_csv(path, usecols=["Invoice", "Quantity", "InvoiceDate", "Price",
                                   "Customer ID"], dtype={"Invoice": str})  # fmt: skip
    x = x[x["Customer ID"].notna() & ~x.Invoice.str.startswith("C") & (x.Quantity > 0)]
    x["c"] = x["Customer ID"].astype(int)
    x["d"] = (pd.to_datetime(x.InvoiceDate) - start).dt.days
    x["spend"] = x.Quantity * x.Price
    o = x.groupby(["c", "d"]).agg(items=("Quantity", "size"), spend=("spend", "sum"))
    o = o.reset_index()
    o["online"] = np.nan
    lo = (pd.Timestamp("2010-06-01") - start).days
    hi = (pd.Timestamp("2010-12-01") - start).days
    return Data("onlineretail", o, None, (lo, hi), 365, {"business": "wholesale (B2B)"})


# -- shared preparation -----------------------------------------------------------------


def cohort_orders(data: Data) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    o = data.orders.sort_values(["c", "d"]).copy()
    o["first"] = o.groupby("c").d.transform("min")
    o = o[o["first"].between(data.cohort[0], data.cohort[1] - 1)]
    o = o[o.d <= o["first"] + data.horizon].copy()
    o["k"] = o.groupby("c").cumcount() + 1
    cats = None
    if data.cats is not None:
        cats = data.cats.merge(o[["c", "d", "k"]], on=["c", "d"])
        cats = cats[cats.cat.notna()]
    return o, cats


def customers(o: pd.DataFrame, horizon: int) -> pd.DataFrame:
    g = o.groupby("c")
    cust = pd.DataFrame({"orders": g.size()})
    rel = o.d - o["first"]
    cust["orders_90d"] = o[rel <= 90].groupby("c").size().reindex(cust.index, fill_value=0)
    second = o[o.k == 2].set_index("c").d - o[o.k == 1].set_index("c").d
    cust["days_to_2nd"] = second.reindex(cust.index)
    if horizon >= 365:
        late = o[rel > 273].groupby("c").size()
        cust["active_last_quarter"] = late.reindex(cust.index, fill_value=0) > 0
        cust["loyal"] = (cust.orders >= LOYAL_ORDERS) & cust.active_last_quarter
        cust["tier"] = np.select(
            [cust.orders == 1, cust.orders <= 3, cust.loyal],
            ["one order", "2-3 orders", "loyal (6+, still buying)"],
            "4+ orders, not loyal",
        )
    return cust


def rate_table(cust: pd.DataFrame, by: pd.Series, top: int | None = None) -> dict:
    out = {}
    groups = list(cust.groupby(by))
    if top:
        groups = sorted(groups, key=lambda kg: -len(kg[1]))[:top]
    for k, g in groups:
        if len(g) < MIN_CELL:
            continue
        row = {
            "n": int(len(g)),
            "share": r(len(g) / len(cust)),
            "one_order": r((g.orders == 1).mean()),
            "second_by_90d": r((g.days_to_2nd <= 90).mean()),
            "mean_orders": r(g.orders.mean(), 2),
        }
        if "loyal" in g:
            row["loyal"] = r(g.loyal.mean())
        out[str(k)] = row
    return out


# -- 1-2. outcomes, funnel, second order ---------------------------------------------------


def outcomes(o: pd.DataFrame, cust: pd.DataFrame, horizon: int) -> dict:
    gap = o.d - o.groupby("c").d.shift()
    funnel = {}
    for n in range(1, 8):
        reached = (cust.orders >= n).sum()
        if reached < MIN_CELL:
            break
        funnel[f"{n}->{n + 1}"] = {
            "reached_n": int(reached),
            "p_next": r((cust.orders >= n + 1).sum() / reached),
            "median_days_between": float(gap[o.k == n + 1].median()),
        }
    out = {
        "cohort_customers": int(len(cust)),
        "one_order_90d": r((cust.orders_90d == 1).mean()),
        "funnel": funnel,
        "median_days_to_2nd": float(cust.days_to_2nd.median()),
    }
    if horizon >= 365:
        out["one_order_365d"] = r((cust.orders == 1).mean())
        out["tiers"] = cust.tier.value_counts(normalize=True).round(3).to_dict()
        none90 = cust.days_to_2nd.isna() | (cust.days_to_2nd > 90)
        out["no_2nd_by_90d_returns_by_365d"] = r(cust.days_to_2nd[none90].notna().mean())
    return out


def second_order(cust: pd.DataFrame, horizon: int) -> dict:
    """Weekly hazard of the second order, and the outcome by days to it."""
    weeks = min(52, horizon // 7)
    w = np.ceil(cust.days_to_2nd / 7)
    hazard = {}
    for k in (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 26, 39, 52):
        if k > weeks:
            break
        at_risk = (w.isna() | (w >= k)).sum()
        hazard[f"week_{k}"] = r(100 * (w == k).sum() / at_risk, 2)
    edges = [0, 14, 30, 60, 120, 365] if horizon >= 365 else [0, 7, 14, 30, 60, 90]
    labels = [f"{a + 1}-{b}" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    bucket = pd.cut(cust.days_to_2nd, edges, labels=labels).astype(object)
    return {
        "weekly_hazard_pct": hazard,
        "by_days_to_2nd": rate_table(cust, bucket.fillna("no 2nd order")),
    }


# -- 3. activity paths and families ----------------------------------------------------------


def activity(o: pd.DataFrame, cust: pd.DataFrame) -> tuple[pd.Series, dict]:
    rep = o[o.k > 1]
    month = np.minimum((rep.d - rep["first"] - 1) // 30, 11)
    bits = (
        pd.Series(np.left_shift(1, month.to_numpy()), index=rep.c.to_numpy())
        .groupby(level=0)
        .agg(lambda s: np.bitwise_or.reduce(s.to_numpy()))
    )
    pat = bits.reindex(cust.index, fill_value=0)
    strings = pat.map(lambda b: "".join("x" if b >> i & 1 else "." for i in range(12)))
    repeaters = strings.value_counts().drop("." * 12, errors="ignore")
    cum = repeaters.cumsum() / repeaters.sum()
    top = {
        s: {"n": int(n), "share_of_repeaters": r(n / repeaters.sum()),
            "loyal": r(cust.loyal[strings == s].mean())}
        for s, n in repeaters.head(TOP).items()
    }  # fmt: skip
    return pat, {
        "repeaters": int(repeaters.sum()),
        "distinct_patterns_among_repeaters": int(len(repeaters)),
        "possible_patterns": 2**12 - 1,
        "patterns_covering_50pct_of_repeaters": int((cum < 0.5).sum() + 1),
        "patterns_covering_80pct_of_repeaters": int((cum < 0.8).sum() + 1),
        "top_patterns": top,
    }


def families(pat: pd.Series) -> pd.Series:
    """Rule-based families from the 12-month pattern (bit i = month i+1)."""
    p = [int(x) for x in pat.to_numpy()]
    active = np.array([bin(x).count("1") for x in p])
    first_rep = np.array([(x & -x).bit_length() if x else 0 for x in p])
    last_rep = np.array([x.bit_length() for x in p])
    return pd.Series(
        np.select(
            [
                np.array(p) == 0,
                (last_rep <= 3) & (active <= 3),
                active >= 6,
                (last_rep >= 10) & (first_rep <= 3),
                (last_rep >= 10) & (first_rep > 3),
                first_rep > 3,
            ],
            [
                "A one and done",
                "B quick repeat, then gone (months 1-3 only)",
                "G steady (6+ active months)",
                "F early repeat, still buying in months 10-12",
                "E slow start (after month 3), still buying in months 10-12",
                "D slow return (after month 3), then gone",
            ],
            "C early repeat, faded by month 9",
        ),
        index=pat.index,
    )


# -- 4-5. step paths and first-order traits ---------------------------------------------------


def new_category_flags(cats: pd.DataFrame) -> pd.Series:
    """(c, k) -> True when order k buys a category the customer hadn't bought before."""
    first_k = cats.groupby(["c", "cat"]).k.transform("min")
    return (first_k == cats.k).groupby([cats.c, cats.k]).any()


def early_paths(o: pd.DataFrame, cats: pd.DataFrame | None, cust: pd.DataFrame) -> dict:
    o = o.copy()
    o["gap"] = o.d - o.groupby("c").d.shift()
    step = pd.cut(o.gap, [0, 30, 90, 400], labels=["fast", "medium", "slow"]).astype(str)
    has_channel = o.online.notna().any()
    if cats is not None:
        new = new_category_flags(cats).reindex(pd.MultiIndex.from_frame(o[["c", "k"]]))
        step = step + "/" + np.where(new.fillna(False).to_numpy(), "new", "same")
    if has_channel:
        step = step + "/" + np.where(o.online >= 0.5, "online", "store")
    o["step"] = step
    o["start"] = np.where(o.online >= 0.5, "online", "store") if has_channel else "first"
    first = o[o.k == 1].set_index("c").start
    s2, s3 = (o[o.k == k].set_index("c").step for k in (2, 3))
    three = cust[cust.orders >= 3].index
    path = first.reindex(three) + " > " + s2.reindex(three) + " > " + s3.reindex(three)
    counts = path.value_counts()
    per_step = 3 * (2 if cats is not None else 1) * (2 if has_channel else 1)
    out = {
        "customers_with_3plus_orders": int(len(three)),
        "distinct_paths_3_orders": int(len(counts)),
        "possible_paths_3_orders": (2 if has_channel else 1) * per_step**2,
        "paths_covering_50pct": int(((counts.cumsum() / counts.sum()) < 0.5).sum() + 1),
    }
    if "loyal" in cust:
        loyal3 = cust.loyal.reindex(three)
        out["top_paths_3_orders"] = {
            p: {"n": int(n), "share": r(n / len(three)), "loyal": r(loyal3[path == p].mean())}
            for p, n in counts.head(12).items()
        }
    two = cust[cust.orders >= 2].index
    out["step_2"] = rate_table(cust.loc[two], first.reindex(two) + " > " + s2.reindex(two))
    parts = s2.reindex(two).str.split("/", expand=True)
    names = ["gap_to_2nd"] + (["2nd_adds_new_category"] if cats is not None else [])
    names += ["2nd_channel"] if has_channel else []
    for i, name in enumerate(names):
        out[name] = rate_table(cust.loc[two], parts[i])
    return out


def first_order_traits(o: pd.DataFrame, cats: pd.DataFrame | None, cust: pd.DataFrame) -> dict:
    f = o[o.k == 1].set_index("c").reindex(cust.index)
    out = {
        "items": rate_table(
            cust, pd.cut(f["items"], [0, 1, 3, 6, 1e9], labels=["1", "2-3", "4-6", "7+"])
        ),
        "spend_quintile": rate_table(
            cust, pd.qcut(f.spend.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        ),
    }
    if f.online.notna().any():
        out["channel"] = rate_table(
            cust, pd.Series(np.where(f.online >= 0.5, "online", "store"), index=cust.index)
        )
    if cats is not None:
        n = cats[cats.k == 1].groupby("c").cat.nunique().reindex(cust.index, fill_value=0)
        out["categories"] = rate_table(cust, n.clip(upper=4))
    return out


# -- 6. category level -------------------------------------------------------------------------


def category_paths(cats: pd.DataFrame, cust: pd.DataFrame) -> dict:
    main = cats.sort_values("spend", ascending=False).drop_duplicates(["c", "k"])
    main = main.set_index(["c", "k"]).cat
    first = main.xs(1, level="k").reindex(cust.index)
    second = main.xs(2, level="k") if (main.index.get_level_values("k") == 2).any() else None
    out = {"by_first_main_category": rate_table(cust, first, top=TOP)}
    if second is None:
        return out
    two = second.index
    bought2 = cats[cats.k == 2].groupby("c").cat.agg(set)
    same = pd.Series([first.get(c) in bought2.get(c, set()) for c in two], index=two, name="same")
    out["second_order_includes_first_main_category"] = r(same.mean())
    out["second_order_same_category_by_first"] = {
        str(k): r(g.mean()) for k, g in same.groupby(first.reindex(two)) if len(g) >= MIN_CELL
    }
    trans = first.reindex(two) + " -> " + second
    counts = trans.value_counts()
    out["top_transitions"] = {
        t: {
            "n": int(n),
            "share_of_2nd_orders": r(n / len(two)),
            **({"loyal": r(cust.loyal.reindex(two)[trans == t].mean())} if "loyal" in cust else {}),
        }
        for t, n in counts.head(TOP).items()
    }
    # categories added at orders 2-4, among customers who reached order 4
    k_first = cats.groupby(["c", "cat"]).k.min()
    added = k_first[(k_first > 1) & (k_first <= 4)]
    reached4 = cust[cust.orders >= 4].index
    added = added[added.index.get_level_values("c").isin(reached4)]
    tops = added.index.get_level_values("cat").value_counts()
    out["added_by_order_4"] = {
        "customers_reaching_order_4": int(len(reached4)),
        "top_added_categories": {k: r(v / len(reached4)) for k, v in tops.head(TOP).items()},
    }
    # breadth by order number: loyal (or 6+ orders) vs the rest
    distinct = (
        cats.assign(first_k=cats.groupby(["c", "cat"]).k.transform("min"))
        .query("first_k == k")
        .groupby(["c", "k"])
        .size()
        .groupby(level=0)
        .cumsum()
    )
    distinct = distinct.reset_index(name="n")
    flag = cust.loyal if "loyal" in cust else cust.orders >= LOYAL_ORDERS
    breadth = {}
    for k in range(1, 7):
        rows = distinct[distinct.k <= k].groupby("c").n.max()
        base = cust[cust.orders >= k].index
        rows = rows.reindex(base).fillna(0)
        breadth[f"after_order_{k}"] = {
            "loyal": r(rows[flag.reindex(base)].mean(), 2),
            "others": r(rows[~flag.reindex(base)].mean(), 2),
        }
    out["distinct_categories_by_order"] = breadth
    return out


# -- Instacart: how breadth forms among shoppers who stay ------------------------------


def instacart(d: str) -> dict:
    d = Path(d)
    orders = pd.read_csv(d / "orders.csv", usecols=["order_id", "user_id", "eval_set",
                                                    "order_number"])  # fmt: skip
    orders = orders[orders.eval_set == "prior"]
    products = pd.read_csv(d / "products.csv", usecols=["product_id", "department_id"])
    depts = pd.read_csv(d / "departments.csv").set_index("department_id").department
    dept_of = np.zeros(products.product_id.max() + 1, dtype=np.int16)
    dept_of[products.product_id.to_numpy()] = products.department_id.to_numpy()
    op = pd.read_csv(d / "order_products__prior.csv", usecols=["order_id", "product_id"],
                     dtype={"order_id": np.int32, "product_id": np.int32})  # fmt: skip
    key = np.unique(op.order_id.to_numpy(np.int64) * 64 + dept_of[op.product_id.to_numpy()])
    del op
    od = pd.DataFrame({"order_id": key // 64, "dept": key % 64})
    od = od.merge(orders[["order_id", "user_id", "order_number"]], on="order_id")
    n_orders = orders.groupby("user_id").order_number.max()
    users = n_orders[n_orders >= 10].index
    od = od[od.user_id.isin(users) & (od.order_number <= 10)]
    first = od.groupby(["user_id", "dept"]).order_number.min().reset_index()
    first["name"] = first.dept.map(depts)
    per = first.groupby("name").order_number
    adoption = pd.DataFrame(
        {
            "in_order_1": per.apply(lambda s: (s == 1).sum()) / len(users),
            "by_order_10": per.size() / len(users),
            "median_order_first_bought": per.median(),
        }
    ).sort_values("in_order_1", ascending=False)
    breadth = {
        f"after_order_{k}": r(first[first.order_number <= k].groupby("user_id").size()
                              .reindex(users, fill_value=0).mean(), 2)
        for k in (1, 2, 3, 5, 10)
    }  # fmt: skip
    return {
        "shoppers_with_10plus_orders": int(len(users)),
        "distinct_departments_by_order": breadth,
        "adoption": {
            k: {c: r(v, 3) for c, v in row.items()} for k, row in adoption.round(3).iterrows()
        },  # fmt: skip
        "caveat": "only shoppers with 4+ orders are in the data (survivorship)",
    }


# -- main --------------------------------------------------------------------------------------


def analyse(data: Data) -> dict:
    o, cats = cohort_orders(data)
    cust = customers(o, data.horizon)
    out = {
        "dataset": data.name,
        "cohort_days": list(data.cohort),
        "horizon_days": data.horizon,
        "notes": data.notes,
        "outcomes": outcomes(o, cust, data.horizon),
        "second_order": second_order(cust, data.horizon),
        "early_paths": early_paths(o, cats, cust),
        "first_order": first_order_traits(o, cats, cust),
    }
    if data.horizon >= 365:
        pat, out["activity_paths"] = activity(o, cust)
        out["families"] = rate_table(cust, families(pat))
    if cats is not None:
        out["category_paths"] = category_paths(cats, cust)
    return out


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "instacart":
        result = instacart(*args)
    else:
        loader = {"hm": load_hm, "cdnow": load_cdnow, "x5": load_x5, "olist": load_olist,
                  "onlineretail": load_onlineretail}[mode]  # fmt: skip
        result = analyse(loader(*args))
    print(json.dumps(result, indent=1, default=str))
