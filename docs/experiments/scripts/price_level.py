"""Is there a general shopper price level, visible only across many orders? (REES46)

Price position = a product's price percentile within its leaf category
(≥20 products), so a premium kettle and a premium phone both score high.

1. Order history: predict each purchase's price position from the
   shopper's *other* purchases in *other* leaf categories; Spearman by how
   many other purchases there are. Same with views as the history.
2. Month to month: a shopper's mean price position over October views vs
   November views, all categories and only categories not viewed in October.
3. Decomposition (views): for shoppers with ≥5 products in two top-level
   types, the share of variance in a type-level mean that is the shopper's
   general level (covariance across types), type-specific, and noise.
4. New category: November purchases in a leaf category the shopper never
   viewed in October, by the shopper's October price level tercile.

Usage:
  uv run --no-project --with pandas --with pyarrow python -I price_level.py events.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DAY = 86_400
OCT_END = 31 * DAY
BULK_ITEMS = 10
MIN_PRODUCTS = 20


def spearman(a, b) -> float:  # noqa: ANN001
    # Positional: a Series keeps its index, and pandas would align on it.
    return float(pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank()))


def load_products(path: Path) -> pd.DataFrame:
    names = json.loads(path.with_suffix(".json").read_text())["category_codes"]
    top_of = {i: n.split(".")[0] for n, i in names.items()}
    cols = pq.read_table(path, columns=["product", "category_id", "code", "price"]).to_pandas()
    p = cols.groupby("product").agg(
        category_id=("category_id", "first"), code=("code", "first"), price=("price", "median")
    )
    del cols
    p = p[p.price > 0]
    size = p.groupby("category_id").price.transform("size")
    p["pct"] = p.groupby("category_id").price.rank(pct=True).where(size >= MIN_PRODUCTS)
    p["cat"] = p.category_id.astype("category").cat.codes.astype("int32")
    p["top"] = p.code.map(top_of).fillna("(none)")
    return p.dropna(subset=["pct"])


def attach(df: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    idx = products.index.get_indexer(df["product"])
    df = df[idx >= 0].copy()
    idx = idx[idx >= 0]
    for c in ("pct", "cat", "top"):
        df[c] = products[c].to_numpy()[idx]
    return df


def history_curve(p: pd.DataFrame, history: pd.DataFrame, buckets) -> dict:  # noqa: ANN001
    """Spearman(purchase price position, mean over the shopper's history in other leaf categories).

    Excluding the purchase's own leaf category keeps one shopping mission
    from "predicting" itself. History spans the whole two months.
    """
    # Per shopper and category: sum and count of the history's price positions.
    h = history.groupby(["user", "cat"]).pct.agg(["sum", "count"]).reset_index()
    tot = h.groupby("user")[["sum", "count"]].sum()
    x = p.merge(h, on=["user", "cat"], how="left").fillna({"sum": 0.0, "count": 0})
    x = x.join(tot, on="user", rsuffix="_all")
    n_other = x.count_all - x["count"]
    x["other_mean"] = (x.sum_all - x["sum"]) / n_other
    x["n_other"] = n_other
    x = x[x.n_other > 0]
    out = {}
    for lo, hi in buckets:
        g = x[(x.n_other >= lo) & (x.n_other <= hi)]
        if len(g) >= 500:
            out[f"{lo}-{hi}"] = {
                "purchases": int(len(g)),
                "spearman": round(spearman(g.pct, g.other_mean), 3),
            }
    return out


def main(path: Path) -> None:
    products = load_products(path)
    buys = pq.read_table(
        path, columns=["user", "product", "t"], filters=[("kind", "=", 2)]
    ).to_pandas()
    buys["day"] = buys.t // DAY
    buys = buys.sort_values("t").drop_duplicates(["user", "product", "day"])
    items = buys.groupby("user").size()
    bulk = items.index[items > BULK_ITEMS].to_numpy()
    buys = attach(buys[~np.isin(buys.user.to_numpy(), bulk)], products).drop_duplicates(
        ["user", "product"]
    )

    v = pq.read_table(
        path, columns=["user", "product", "t"], filters=[("kind", "=", 0)]
    ).to_pandas()
    v = v[~np.isin(v.user.to_numpy(), bulk)].sort_values("t", kind="stable")
    v = attach(v[~v.duplicated(["user", "product"])], products)

    out: dict = {"bulk_buyers_removed": int(len(bulk)), "purchases": int(len(buys))}

    # 1. Order history (other purchases in other leaf categories) and view history.
    out["history_from_other_purchases"] = history_curve(
        buys, buys[["user", "cat", "pct"]], [(1, 1), (2, 3), (4, 9)]
    )
    buyers = v[v.user.isin(buys.user.unique())]
    out["history_from_views_in_other_categories"] = history_curve(
        buys, buyers[["user", "cat", "pct"]], [(1, 4), (5, 19), (20, 99), (100, 10**9)]
    )

    # 2. Month to month, views.
    oct_, nov = v[v.t < OCT_END], v[v.t >= OCT_END]
    o = oct_.groupby("user").pct.agg(["mean", "count"])
    res = {}
    for name, nv in {
        "all categories": nov,
        "only categories not viewed in October": nov[
            ~pd.MultiIndex.from_frame(nov[["user", "cat"]]).isin(
                pd.MultiIndex.from_frame(oct_[["user", "cat"]].drop_duplicates())
            )
        ],
    }.items():
        n = nv.groupby("user").pct.agg(["mean", "count"])
        both = o.join(n, lsuffix="_o", rsuffix="_n", how="inner")
        rows = {}
        for lo, hi in [(5, 19), (20, 99), (100, 10**9)]:
            g = both[
                (both.count_o >= lo)
                & (both.count_n >= lo)
                & (both.count_o <= hi)
                & (both.count_n <= hi)
            ]
            if len(g) >= 500:
                rows[f"{lo}-{hi} products each month"] = {
                    "shoppers": int(len(g)),
                    "spearman": round(spearman(g.mean_o, g.mean_n), 3),
                }
        res[name] = rows
    out["month_to_month_views"] = res

    # 3. Decomposition across top-level types (views, whole period).
    st = (
        v[v.top != "(none)"]
        .groupby(["user", "top"])
        .pct.agg(["mean", "var", "count"])
        .reset_index()
    )
    st = st[st["count"] >= 5].sort_values(["user", "count"], ascending=[True, False])
    st["rank"] = st.groupby("user").cumcount()
    a = st[st["rank"] == 0].set_index("user")
    b = st[st["rank"] == 1].set_index("user")
    ab = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    general = float(np.cov(ab.mean_a, ab.mean_b)[0, 1])
    total = float((ab.mean_a.var() + ab.mean_b.var()) / 2)
    noise = float(((ab.var_a / ab.count_a).mean() + (ab.var_b / ab.count_b).mean()) / 2)
    out["decomposition_views"] = {
        "shoppers": int(len(ab)),
        "share_general_level": round(general / total, 3),
        "share_type_specific": round((total - general - noise) / total, 3),
        "share_noise": round(noise / total, 3),
        "type_mean_correlation": round(float(ab.mean_a.corr(ab.mean_b)), 3),
    }

    # 4. New category: Nov purchases in leaf categories not viewed in Oct.
    level = o[o["count"] >= 10]["mean"]
    seen = pd.MultiIndex.from_frame(oct_[["user", "cat"]].drop_duplicates())
    nb = buys[(buys.t >= OCT_END) & buys.user.isin(level.index)]
    nb = nb[~pd.MultiIndex.from_frame(nb[["user", "cat"]]).isin(seen)]
    terc = pd.qcut(level.rank(method="first"), 3, labels=["low", "mid", "high"])
    nb = nb.assign(level=terc.reindex(nb.user).to_numpy())
    nb["tier"] = pd.cut(nb.pct, [0, 1 / 3, 2 / 3, 1], labels=["budget", "mid", "premium"])
    out["new_category_purchase_tier_by_oct_level"] = {
        str(lv): {
            "purchases": int(len(g)),
            **{
                str(k): round(float(val), 3)
                for k, val in g.tier.value_counts(normalize=True).sort_index().items()
            },
        }
        for lv, g in nb.groupby("level", observed=True)
    }
    out["new_category_spearman"] = round(spearman(nb.pct, level.reindex(nb.user).to_numpy()), 3)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
