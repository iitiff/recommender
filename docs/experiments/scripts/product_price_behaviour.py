"""How shopping behaviour differs by product type and price tier (REES46).

Input: the compact event parquet from stream_rees46_events.py (+ its .json).

- Product type: REES46 category_code, top level ("electronics") and second
  level ("electronics.smartphone"). Events without a code are left out of
  type tables (reported).
- Price tier: a product's price percentile *within its leaf category*
  (category_id, ≥20 products), in thirds: budget / mid / premium. A $200
  phone is budget and a $200 kettle is premium, so absolute prices aren't
  comparable across types; relative position is.
- Order = a purchase of a product on a day (user, product, day), deduplicated.
- Bulk buyers (>10 purchased items, likely resellers) are left out entirely.

Analyses:
A. Consideration before a purchase (30-day lookback, so only purchases from
   day 30 on): views of the bought product, alternatives viewed in the same
   leaf category, visits, days from first look to purchase.
B. Cart → purchase of the same product within 7 days.
C. After a purchase (30-day horizon, so only purchases before day 31): buy
   again in the same type, re-buy the same product, and what the next
   purchase is.
D. Price as a shopper trait: is a shopper's price tier consistent across
   categories? Do they compare within a price band? Do they buy cheaper or
   pricier than what they browse?

Usage:
  uv run --no-project --with pandas --with pyarrow python -I \
    product_price_behaviour.py events.parquet > out.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DAY = 86_400
LOOKBACK = 30 * DAY
MIN_PRODUCTS = 20  # products needed in a leaf category to rank prices
TIERS = ["budget", "mid", "premium"]
BULK_ITEMS = 10


def load(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Events (t, kind, product, user, cat) and a product table.

    Reads only the columns each step needs: all nine columns of 110M events
    don't fit in 15 GB alongside the intermediate copies pandas makes.
    """
    names = json.loads(path.with_suffix(".json").read_text())["category_codes"]
    code_name = np.array(["?"] * (max(names.values()) + 2), dtype=object)
    for name, i in names.items():
        code_name[i] = name
    code_name[-1] = ""  # index -1 = missing

    # Products: leaf category, type and median price; price percentile within the leaf.
    cols = pq.read_table(path, columns=["product", "category_id", "code", "price"]).to_pandas()
    products = cols.groupby("product").agg(
        category_id=("category_id", "first"), code=("code", "first"), price=("price", "median")
    )
    del cols
    products = products[products.price > 0]
    size = products.groupby("category_id").price.transform("size")
    products["pct"] = (
        products.groupby("category_id").price.rank(pct=True).where(size >= MIN_PRODUCTS)
    )
    products["tier"] = pd.cut(products.pct, [0, 1 / 3, 2 / 3, 1], labels=TIERS).astype(str)
    name = pd.Series(code_name[products.code.to_numpy()], index=products.index)
    products["top"] = name.str.split(".").str[0].replace("", "(no category)")
    products["type"] = name.str.split(".").str[:2].str.join(".").replace("", "(no category)")
    products["cat"] = products.category_id.astype("category").cat.codes.astype("int16")

    ev = pq.read_table(path, columns=["t", "kind", "product", "user"]).to_pandas()
    ev["cat"] = category_of(ev["product"], products)
    return ev[ev.cat >= 0], products


def category_of(product: pd.Series, products: pd.DataFrame) -> np.ndarray:
    idx = products.index.get_indexer(product)
    return np.where(idx >= 0, products.cat.to_numpy()[idx], -1).astype("int16")


def purchases(ev: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    p = ev.loc[ev.kind == 2, ["user", "product", "cat", "t"]].copy()
    p["day"] = p.t // DAY
    p = p.sort_values("t").drop_duplicates(["user", "product", "day"]).reset_index(drop=True)
    p["price"] = products.price.reindex(p["product"]).to_numpy()  # product's median price
    return p


def key(user: pd.Series, cat: pd.Series) -> np.ndarray:
    return user.to_numpy(dtype=np.int64) * 10_000 + cat.to_numpy(dtype=np.int64)


def views_before(path: Path, products: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Views (with visit ids) in the 30 days before each purchase in `b`, same leaf category."""
    v = pq.read_table(
        path, columns=["user", "product", "t", "session"], filters=[("kind", "=", 0)]
    ).to_pandas()
    v["cat"] = category_of(v["product"], products)
    v = v[np.isin(key(v.user, v.cat), np.unique(key(b.user, b.cat)))]
    j = v.merge(
        b[["buy", "user", "cat", "product", "t"]], on=["user", "cat"], suffixes=("", "_buy")
    )
    return j[(j.t <= j.t_buy) & (j.t > j.t_buy - LOOKBACK)]


def consideration(b: pd.DataFrame, j: pd.DataFrame) -> pd.DataFrame:
    """One row per purchase (from day 30 on) with what happened in the 30 days before."""
    same = j["product"] == j.product_buy
    agg = pd.DataFrame(
        {
            "views_of_item": same.groupby(j.buy).sum(),
            "alternatives": j["product"].where(~same).groupby(j.buy).nunique(),
            "visits": j.session.groupby(j.buy).nunique(),
            "days": (j.t_buy - j.t).groupby(j.buy).max() / DAY,
        }
    )
    out = b.join(agg, on="buy")
    out[["views_of_item", "alternatives", "visits"]] = out[
        ["views_of_item", "alternatives", "visits"]
    ].fillna(0)
    out["days"] = out.days.fillna(0)
    return out


def cart_conversion(ev: pd.DataFrame, buys: pd.DataFrame) -> pd.DataFrame:
    end = ev.t.max()
    c = ev.loc[(ev.kind == 1) & (ev.t <= end - 7 * DAY), ["user", "product", "t"]].copy()
    c["day"] = c.t // DAY
    c = c.sort_values("t").drop_duplicates(["user", "product", "day"])
    j = c.merge(
        buys[["user", "product", "t"]], on=["user", "product"], how="left", suffixes=("", "_buy")
    )
    j["bought"] = (j.t_buy >= j.t) & (j.t_buy <= j.t + 7 * DAY)
    return j.groupby(["user", "product", "t"], as_index=False).bought.any()


def after_purchase(buys: pd.DataFrame, products: pd.DataFrame, end: int) -> pd.DataFrame:
    """For each purchase with 30 days of data after it: what the shopper bought next."""
    b = (
        buys.join(products[["type", "top"]], on="product")
        .sort_values(["user", "t"])
        .reset_index(drop=True)
    )
    b["i"] = np.arange(len(b))
    later = b.merge(b, on="user", suffixes=("", "_n"))
    later = later[(later.day_n > later.day) & (later.t_n <= later.t + 30 * DAY)]
    first_next = later.sort_values("t_n").drop_duplicates("i")
    base = b[b.t <= end - 30 * DAY].copy()
    base["same_type_30d"] = base.i.isin(later.loc[later.type_n == later.type, "i"])
    base["same_item_30d"] = base.i.isin(later.loc[later.product_n == later["product"], "i"])
    base["any_30d"] = base.i.isin(later.i)
    nxt = first_next.set_index("i")
    base["next_type"] = base.i.map(nxt.type_n)
    base["next_top"] = base.i.map(nxt.top_n)
    return base


def summarize(frame: pd.DataFrame, by: str, cols: dict, min_n: int = 300) -> list[dict]:
    rows = []
    for value, g in frame.groupby(by, observed=True):
        if len(g) < min_n:
            continue
        rows.append(
            {by: value, "n": int(len(g)), **{k: round(float(f(g)), 4) for k, f in cols.items()}}
        )
    return sorted(rows, key=lambda r: -r["n"])


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def price_trait(ev: pd.DataFrame, products: pd.DataFrame, cons: pd.DataFrame) -> dict:
    out: dict = {}
    # 1. Viewing: a shopper's mean price percentile in their two most-viewed top-level types.
    views = ev[ev.kind == 0]
    pairs = np.unique(
        views.user.to_numpy(np.int64) * 1_000_000_000 + views["product"].to_numpy(np.int64)
    )
    v = pd.DataFrame({"user": pairs // 1_000_000_000, "product": pairs % 1_000_000_000})
    v = v.join(products[["pct", "top"]], on="product").dropna(subset=["pct"])
    v = v[v.top != "(no category)"]
    per = v.groupby(["user", "top"]).pct.agg(["mean", "size"]).reset_index()
    per = per[per["size"] >= 5].sort_values(["user", "size"], ascending=[True, False])
    per["rank"] = per.groupby("user").cumcount()
    two = per[per["rank"] < 2].pivot(index="user", columns="rank", values="mean").dropna()
    out["viewing_tier_correlation_across_types"] = {
        "users": int(len(two)),
        "spearman": round(spearman(two[0], two[1]), 3),
    }
    # 2. Buying: price percentile of purchases in two different leaf categories.
    b = (
        cons[["user", "product", "cat", "t"]]
        .join(products["pct"], on="product")
        .dropna(subset=["pct"])
    )
    b = b.sort_values("t").drop_duplicates(["user", "cat"])
    b["n"] = b.groupby("user").cumcount()
    pair = b[b.n < 2].pivot(index="user", columns="n", values="pct").dropna()
    out["buying_tier_correlation_across_categories"] = {
        "users": int(len(pair)),
        "spearman": round(spearman(pair[0], pair[1]), 3),
    }
    return out


def price_band(
    ev: pd.DataFrame,
    products: pd.DataFrame,
    cons: pd.DataFrame,
    viewed: pd.DataFrame,
    seed: int = 0,
) -> dict:
    """Alternatives viewed before a purchase vs random products of the same leaf category."""
    b = cons[cons.alternatives > 0].reset_index(drop=True)
    j = viewed[viewed.buy.isin(b.buy) & (viewed["product"] != viewed.product_buy)]
    j = j.merge(b[["buy", "price"]], on="buy")
    alt = products[["price", "pct"]].rename(columns={"price": "price_alt", "pct": "pct_alt"})
    j = j.drop_duplicates(["buy", "product"]).join(alt, on="product")
    j = j[(j.price_alt > 0) & (j.price > 0)]
    ratio = np.abs(np.log(j.price_alt / j.price))

    # Baseline: a random *viewed* product (view-weighted, so as popular as what
    # shoppers actually browse) from the same leaf category as the purchase.
    rng = np.random.default_rng(seed)
    sample = ev.loc[ev.kind == 0, ["product", "cat"]].sample(
        min(5_000_000, int((ev.kind == 0).sum())), random_state=seed
    )
    pool = sample.join(products["price"], on="product").dropna().sort_values("cat")
    starts = pool.cat.searchsorted(b.cat.to_numpy(), "left")
    ends = pool.cat.searchsorted(b.cat.to_numpy(), "right")
    ok = ends - starts > 1
    pick = starts[ok] + (rng.random(ok.sum()) * (ends[ok] - starts[ok])).astype(int)
    base = np.abs(np.log(pool.price.to_numpy()[pick] / b.price.to_numpy()[ok]))

    buy_pct = b.set_index("buy").pct
    alt_pct = j.groupby("buy").pct_alt.mean()
    gap = (buy_pct - alt_pct).dropna()
    return {
        "purchases_with_alternatives": int(len(b)),
        "alternatives": int(len(j)),
        "median_abs_log_price_ratio": round(float(ratio.median()), 3),
        "share_within_20pct": round(float((ratio <= np.log(1.2)).mean()), 3),
        "baseline_random_same_category_median": round(float(np.median(base)), 3),
        "baseline_share_within_20pct": round(float((base <= np.log(1.2)).mean()), 3),
        "bought_minus_browsed_percentile_mean": round(float(gap.mean()), 3),
        "share_bought_cheaper_than_browsed": round(float((gap < 0).mean()), 3),
    }


def main(path: Path) -> None:
    ev, products = load(path)
    end = int(ev.t.max())
    buys = purchases(ev, products)
    items = buys.groupby("user").size()
    bulk = items.index[items > BULK_ITEMS]
    ev = ev[~ev.user.isin(bulk)]
    buys = buys[~buys.user.isin(bulk)]
    b = buys[buys.t >= LOOKBACK].reset_index(drop=True).rename_axis("buy").reset_index()
    viewed = views_before(path, products, b)
    viewed = viewed[~viewed.user.isin(bulk)]
    cons = consideration(b, viewed).join(products[["top", "type", "tier", "pct"]], on="product")
    carts = cart_conversion(ev, buys).join(products[["top", "type", "tier"]], on="product")
    after = after_purchase(buys, products, end).join(products[["tier"]], on="product")

    consider_cols = {
        "median_price": lambda g: g.price.median(),
        "views_of_item_median": lambda g: g.views_of_item.median(),
        "alternatives_median": lambda g: g.alternatives.median(),
        "alternatives_mean": lambda g: g.alternatives.mean(),
        "visits_median": lambda g: g.visits.median(),
        "share_one_visit": lambda g: (g.visits <= 1).mean(),
        "days_median": lambda g: g.days.median(),
        "share_over_1_day": lambda g: (g.days > 1).mean(),
    }
    after_cols = {
        "any_purchase_30d": lambda g: g.any_30d.mean(),
        "same_type_30d": lambda g: g.same_type_30d.mean(),
        "same_item_30d": lambda g: g.same_item_30d.mean(),
    }
    top_next = {}
    for t, g in after.dropna(subset=["next_type"]).groupby("type"):
        if len(g) >= 300:
            top_next[t] = {
                k: round(v, 3) for k, v in g.next_type.value_counts(normalize=True).head(4).items()
            }

    result = {
        "events": int(len(ev)),
        "products": int(len(products)),
        "products_ranked_by_price": int(products.pct.notna().sum()),
        "purchases": int(len(buys)),
        "bulk_buyers_removed": int(len(bulk)),
        "share_purchases_with_type": round(float((cons.top != "(no category)").mean()), 3),
        "consideration_by_top": summarize(cons, "top", consider_cols),
        "consideration_by_type": summarize(cons, "type", consider_cols, min_n=2000),
        "consideration_by_tier": summarize(cons, "tier", consider_cols),
        "consideration_by_top_and_tier": summarize(
            cons.assign(top_tier=cons.top + " / " + cons.tier),
            "top_tier",
            consider_cols,
            min_n=1000,
        ),
        "cart_conversion_by_top": summarize(carts, "top", {"bought_7d": lambda g: g.bought.mean()}),
        "cart_conversion_by_tier": summarize(
            carts, "tier", {"bought_7d": lambda g: g.bought.mean()}
        ),
        "after_purchase_by_top": summarize(after, "top", after_cols),
        "after_purchase_by_type": summarize(after, "type", after_cols, min_n=1000),
        "after_purchase_by_tier": summarize(after, "tier", after_cols),
        "next_purchase_type": top_next,
        "price_trait": price_trait(ev, products, cons),
        "price_band": price_band(ev, products, cons, viewed),
    }
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
