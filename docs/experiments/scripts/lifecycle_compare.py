"""Compare customer-lifecycle patterns across e-commerce datasets.

Usage: python -I lifecycle_compare.py <dataset> <path> [--horizon 60]
  dataset: retailrocket (prepared dir) | olist (dir) | hm (dir) | rees46 (purchases.csv)
           | onlineretail (csv)

Every dataset is converted to order lines (user_id, order_id, ts, item_id,
top_cat, leaf_cat), then the same metrics are computed:
  1. concentration: share of purchased items bought by repeat customers
  2. 1st→2nd order within H days (only first orders with H days of data left)
  3. what the 2nd order contains (same item / leaf / top category, or new top)
  4. P(3rd order within H days of the 2nd) for "expanded into a new top
     category" vs "stayed", with a robustness check on single-item orders

An "order" is always a customer's purchases on one calendar day (UTC), so
datasets with sessions, order ids or neither are compared on equal terms.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd


# -- converters -------------------------------------------------------------------
def retailrocket(path: Path) -> pd.DataFrame:
    ev = pd.read_csv(path / "interactions.csv", dtype={"user_id": str, "item_id": str})
    ev = ev[ev.kind == "purchase"]
    items = pd.read_csv(
        path / "catalog" / "items.csv",
        dtype={"item_id": str, "category": str, "category_path": str},
    )
    items = items.set_index("item_id")
    return pd.DataFrame(
        {
            "user_id": ev.user_id,
            "order_id": ev.session_id,
            "ts": pd.to_datetime(ev.occurred_at, utc=True, format="ISO8601"),
            "item_id": ev.item_id,
            "top_cat": ev.item_id.map(items.category_path.str.split(";").str[0]),
            "leaf_cat": ev.item_id.map(items.category),
        }
    )


def olist(path: Path) -> pd.DataFrame:
    orders = pd.read_csv(path / "olist_orders_dataset.csv")
    orders = orders[~orders.order_status.isin(["canceled", "unavailable"])]
    customers = pd.read_csv(path / "olist_customers_dataset.csv")
    items = pd.read_csv(path / "olist_order_items_dataset.csv", usecols=["order_id", "product_id"])
    products = pd.read_csv(
        path / "olist_products_dataset.csv", usecols=["product_id", "product_category_name"]
    )
    df = (
        orders.merge(customers[["customer_id", "customer_unique_id"]], on="customer_id")
        .merge(items, on="order_id")
        .merge(products, on="product_id", how="left")
    )
    ts = pd.to_datetime(df.order_purchase_timestamp, utc=True)
    return pd.DataFrame(
        {
            "user_id": df.customer_unique_id,
            # Same-day orders count as one: buying from two sellers at once creates
            # two order ids seconds apart, which isn't a repeat purchase.
            "order_id": df.customer_unique_id + "|" + ts.dt.strftime("%Y-%m-%d"),
            "ts": ts,
            "item_id": df.product_id,
            "top_cat": df.product_category_name,  # flat taxonomy: top = leaf
            "leaf_cat": df.product_category_name,
        }
    )


def hm(path: Path) -> pd.DataFrame:
    parts = sorted((path / "transactions").glob("*.parquet"))
    tx = pd.concat(
        [pd.read_parquet(p, columns=["t_dat", "customer_id", "article_id"]) for p in parts],
        ignore_index=True,
    )
    tx["customer_id"] = tx.customer_id.astype("category")
    art = pd.read_parquet(
        next((path / "articles").glob("*.parquet")),
        columns=["article_id", "product_group_name", "product_type_name"],
    ).set_index("article_id")
    ts = pd.to_datetime(tx.t_dat, utc=True)
    return pd.DataFrame(
        {
            "user_id": tx.customer_id.cat.codes,
            # One order = one customer's purchases on one day (H&M has no order ids).
            "order_id": tx.customer_id.cat.codes.astype("int64") * 100_000
            + (ts - ts.min()).dt.days,
            "ts": ts,
            "item_id": tx.article_id,
            "top_cat": tx.article_id.map(art.product_group_name).astype("category"),
            "leaf_cat": tx.article_id.map(art.product_type_name).astype("category"),
        }
    )


def rees46(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"category_code": str, "user_session": str})
    return pd.DataFrame(
        {
            "user_id": df.user_id,
            "order_id": df.user_id.astype(str) + "|" + df.user_session.fillna(""),
            "ts": pd.to_datetime(df.event_time.str.replace(" UTC", "", regex=False), utc=True),
            "item_id": df.product_id,
            "top_cat": df.category_code.str.split(".").str[0],  # ~30% of products lack a code
            "leaf_cat": df.category_id,
        }
    )


def onlineretail(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Invoice": str, "StockCode": str})
    df = df[df["Customer ID"].notna() & (df.Quantity > 0) & ~df.Invoice.str.startswith("C")]
    return pd.DataFrame(
        {
            "user_id": df["Customer ID"].astype("int64"),
            "order_id": df.Invoice,
            "ts": pd.to_datetime(df.InvoiceDate, utc=True),
            "item_id": df.StockCode,
            "top_cat": pd.NA,  # no categories in this dataset
            "leaf_cat": pd.NA,
        }
    )


CONVERTERS = {f.__name__: f for f in (retailrocket, olist, hm, rees46, onlineretail)}


# -- metrics ------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def overlap_users(lines: pd.DataFrame, col: str) -> tuple[pd.Index, pd.Index]:
    """Users whose rank-0 and rank-1 orders share a value of `col`, and users for
    whom `col` is known in both orders. Vectorized: no per-order Python sets."""
    known = lines.dropna(subset=[col])[["user_id", "rank", col]].drop_duplicates()
    ranks_known = known.groupby("user_id")["rank"].nunique()
    both_known = ranks_known.index[ranks_known == 2]
    shared = known.groupby(["user_id", col], observed=True)["rank"].nunique()
    overlap = shared[shared == 2].index.get_level_values("user_id").unique()
    return overlap, both_known


def analyze(lines: pd.DataFrame, horizon_days: int) -> dict:
    # One definition of "order" across datasets: a customer's purchases on one
    # calendar day. Session-based orders count "came back an hour later for one
    # more item" as a repeat purchase, which isn't loyalty.
    lines = lines.assign(
        order_id=lines.user_id.astype(str) + "|" + lines.ts.dt.strftime("%Y-%m-%d")
    )
    h = pd.Timedelta(days=horizon_days)
    end, start = lines.ts.max(), lines.ts.min()
    orders = lines.groupby(["user_id", "order_id"], observed=True).agg(
        ts=("ts", "min"), n_lines=("item_id", "size")
    )
    orders = orders.reset_index().sort_values(["user_id", "ts", "order_id"])
    orders["rank"] = orders.groupby("user_id").cumcount()
    n_orders = orders.groupby("user_id").size()
    out: dict = {
        "window_days": int((end - start).days),
        "buyers": int(len(n_orders)),
        "orders": int(len(orders)),
        "buyers_2plus": float((n_orders >= 2).mean()),
    }
    lines_per_user = lines.groupby("user_id", observed=True).size()
    out["item_share_2plus"] = float(
        lines_per_user[n_orders[n_orders >= 2].index].sum() / len(lines)
    )
    out["item_share_3plus"] = float(
        lines_per_user[n_orders[n_orders >= 3].index].sum() / len(lines)
    )

    o = orders.set_index(["user_id", "rank"]).ts.unstack()  # columns: rank 0, 1, 2, ...
    first, second = o[0], o.get(1)
    third = o.get(2)
    eligible = first <= end - h
    out["second_order_rate_h"] = float(((second - first) <= h)[eligible].mean())
    gaps = (second - first).dropna().dt.total_seconds() / 86400
    out["days_1_to_2"] = {
        f"p{int(q * 100)}": round(float(gaps.quantile(q)), 1) for q in (0.25, 0.5, 0.75)
    }

    # What the second order contains.
    r = orders[orders["rank"] <= 1][["user_id", "order_id", "rank", "n_lines"]]
    l2 = lines.merge(r, on=["user_id", "order_id"])
    two = r.groupby("user_id")["rank"].nunique()
    two = two.index[two == 2]
    comp = {"n_second_orders": int(len(two))}
    for col in ("item_id", "leaf_cat", "top_cat"):
        ov, known = overlap_users(l2, col)
        known = known.intersection(two)
        comp[f"same_{col}"] = (
            float(len(ov.intersection(known)) / len(known)) if len(known) else None
        )
        comp[f"known_{col}"] = int(len(known))
    out["second_order"] = comp

    # Expansion vs staying, and P(3rd order within H of the 2nd).
    if comp["known_top_cat"]:
        ov, known = overlap_users(l2, "top_cat")
        sec_ok = second.dropna()
        sec_ok = sec_ok[sec_ok <= end - h].index
        base = known.intersection(sec_ok)
        third_soon = (
            ((third - second) <= h) if third is not None else pd.Series(False, index=second.index)
        )
        single = r.pivot(index="user_id", columns="rank", values="n_lines")
        single = single.index[(single[0] == 1) & (single[1] == 1)]

        def group(users: pd.Index) -> dict:
            stayed = users.intersection(ov)
            expanded = users.difference(ov)
            res = {}
            for name, g in (("stayed", stayed), ("expanded", expanded)):
                k = int(third_soon.reindex(g).fillna(False).sum())
                lo, hi = wilson(k, len(g))
                res[name] = {
                    "n": int(len(g)),
                    "p_third": k / len(g) if len(g) else None,
                    "ci": [lo, hi],
                }
            return res

        out["expansion"] = {
            "all": group(base),
            "single_item_orders": group(base.intersection(single)),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(CONVERTERS))
    ap.add_argument("path", type=Path)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument(
        "--max-items-per-user",
        type=int,
        default=None,
        help="Drop users who bought more items than this (e.g. resellers) before analysing",
    )
    args = ap.parse_args()
    lines = CONVERTERS[args.dataset](args.path)
    if args.max_items_per_user is not None:
        per_user = lines.groupby("user_id", observed=True).size()
        lines = lines[lines.user_id.isin(per_user.index[per_user <= args.max_items_per_user])]
    result = analyze(lines, args.horizon)
    print(
        json.dumps(
            {"dataset": args.dataset, "horizon_days": args.horizon, **result}, indent=2, default=str
        )
    )
