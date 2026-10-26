"""Which shopper traits are real (stable over time) on REES46?

For each candidate trait, compute it per shopper in October and again in
November, and correlate (Spearman) across shoppers with enough activity in
both months. A trait worth modelling should carry over. As a noise ceiling,
the same trait is also computed on odd vs even days of October: if Oct→Nov
is close to that ceiling, the trait is stable and only measurement noise
limits it; if it's much lower, the trait drifts.

Traits:
- brand loyalty: when a shopper looks at consecutive products in the same
  category, how much more often they stay with the same brand than the
  category's brand mix would predict
- mainstream vs niche: average popularity percentile (within category) of
  the products they view
- variety: distinct categories per product viewed
- deal sensitivity: how much below a product's usual price they pay,
  compared with the price shoppers typically saw for that product
- comparison breadth: alternatives viewed in the purchase's category in the
  7 days before buying
- rhythm (from visits): does a shopper's own usual hour predict when they
  come back better than the store's busiest hour does?

Usage:
  uv run --no-project --with pandas --with pyarrow python -I \
    shopper_traits.py events.parquet sessions.parquet > traits.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DAY = 86_400
OCT_END = 31 * DAY  # t is seconds since 2019-10-01 UTC
BULK_ITEMS = 10
MIN_PRODUCTS = 20
PERIODS = {
    "oct": lambda t: t < OCT_END,
    "nov": lambda t: t >= OCT_END,
    "oct_odd": lambda t: (t < OCT_END) & ((t // DAY) % 2 == 1),
    "oct_even": lambda t: (t < OCT_END) & ((t // DAY) % 2 == 0),
}


def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def stability(trait) -> dict:  # noqa: ANN001 - callable(period mask) -> Series per user
    s = {name: trait(name) for name in PERIODS}
    both = pd.concat([s["oct"], s["nov"]], axis=1, keys=["a", "b"]).dropna()
    half = pd.concat([s["oct_odd"], s["oct_even"]], axis=1, keys=["a", "b"]).dropna()
    q = s["oct"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "users_oct_and_nov": int(len(both)),
        "spearman_oct_vs_nov": round(spearman(both.a, both.b), 3),
        "users_odd_and_even_days": int(len(half)),
        "spearman_odd_vs_even_days": round(spearman(half.a, half.b), 3),
        "oct_quantiles": {f"p{int(k * 100)}": round(float(v), 4) for k, v in q.items()},
        "_oct": s["oct"],
    }


def load_products(path: Path) -> pd.DataFrame:
    cols = pq.read_table(path, columns=["product", "category_id", "brand", "price"]).to_pandas()
    p = cols.groupby("product").agg(
        category_id=("category_id", "first"), brand=("brand", "first"), price=("price", "median")
    )
    del cols
    p["cat"] = p.category_id.astype("category").cat.codes.astype("int32")
    return p


def index_of(product: np.ndarray, products: pd.DataFrame, column: str, fill) -> np.ndarray:  # noqa: ANN001
    idx = products.index.get_indexer(product)
    return np.where(idx >= 0, products[column].to_numpy()[np.maximum(idx, 0)], fill)


def first_views(path: Path, products: pd.DataFrame, bulk: np.ndarray) -> pd.DataFrame:
    """Each (shopper, product) once, at its first view; with category and brand."""
    v = pq.read_table(
        path, columns=["user", "product", "t"], filters=[("kind", "=", 0)]
    ).to_pandas()
    v = v[~np.isin(v.user.to_numpy(), bulk)]
    v = v.sort_values("t", kind="stable")
    v = v[~v.duplicated(["user", "product"])]
    v["cat"] = index_of(v["product"].to_numpy(), products, "cat", -1).astype("int32")
    v["brand"] = index_of(v["product"].to_numpy(), products, "brand", -1).astype("int32")
    return v[v.cat >= 0].reset_index(drop=True)


def brand_loyalty(v: pd.DataFrame, period: str) -> pd.Series:
    known = v[v.brand >= 0]
    # Chance of two random views in a category sharing a brand: Σ share².
    shares = known.groupby(["cat", "brand"]).size() / known.groupby("cat").size()
    base = (shares**2).groupby(level="cat").sum()
    x = known[PERIODS[period](known.t.to_numpy())].sort_values(["user", "cat", "t"])
    pair = (x.user.to_numpy()[1:] == x.user.to_numpy()[:-1]) & (
        x.cat.to_numpy()[1:] == x.cat.to_numpy()[:-1]
    )
    same = x.brand.to_numpy()[1:] == x.brand.to_numpy()[:-1]
    excess = same.astype(float) - base.reindex(x.cat.to_numpy()[1:]).to_numpy()
    df = pd.DataFrame({"user": x.user.to_numpy()[1:][pair], "excess": excess[pair]})
    g = df.groupby("user").excess
    return g.mean()[g.size() >= 3]


def mainstream(v: pd.DataFrame, pop_pct: pd.Series, period: str) -> pd.Series:
    x = v[PERIODS[period](v.t.to_numpy())]
    pct = pop_pct.reindex(x["product"].to_numpy()).to_numpy()
    g = pd.Series(pct, index=x.user.to_numpy()).dropna().groupby(level=0)
    return g.mean()[g.size() >= 5]


def variety(v: pd.DataFrame, period: str) -> pd.Series:
    x = v[PERIODS[period](v.t.to_numpy())]
    g = x.groupby("user")
    n = g.size()
    return (g.cat.nunique() / n)[n >= 10]


def purchases(path: Path, products: pd.DataFrame) -> pd.DataFrame:
    p = pq.read_table(
        path, columns=["user", "product", "t", "price"], filters=[("kind", "=", 2)]
    ).to_pandas()
    p["day"] = p.t // DAY
    p = p.sort_values("t").drop_duplicates(["user", "product", "day"]).reset_index(drop=True)
    p["cat"] = index_of(p["product"].to_numpy(), products, "cat", -1).astype("int32")
    return p


def deal_excess(path: Path, products: pd.DataFrame, p: pd.DataFrame) -> pd.Series:
    """Per purchase: discount vs the product's usual price, minus the discount
    shoppers typically saw on that product's views."""
    v = pq.read_table(path, columns=["product", "price"], filters=[("kind", "=", 0)]).to_pandas()
    ref = products.price
    depth_view = 1 - v.price / ref.reindex(v["product"].to_numpy()).to_numpy()
    typical = depth_view.groupby(v["product"].to_numpy()).mean()
    del v
    paid = 1 - p.price / ref.reindex(p["product"].to_numpy()).to_numpy()
    return paid - typical.reindex(p["product"].to_numpy()).to_numpy()


def per_user_purchases(
    p: pd.DataFrame, values: pd.Series, period: str, how: str = "mean"
) -> pd.Series:
    mask = PERIODS[period](p.t.to_numpy())
    g = pd.Series(values.to_numpy()[mask], index=p.user.to_numpy()[mask]).dropna().groupby(level=0)
    return getattr(g, how)()[g.size() >= 2]


def alternatives_before(v: pd.DataFrame, p: pd.DataFrame) -> pd.Series:
    """Other products in the purchase's category the shopper first viewed in the 7 days before."""
    b = p[["user", "cat", "product", "t"]].reset_index().rename(columns={"index": "buy"})
    k = v.user.to_numpy(np.int64) * 100_000 + v.cat.to_numpy(np.int64)
    kb = b.user.to_numpy(np.int64) * 100_000 + b.cat.to_numpy(np.int64)
    vv = v[np.isin(k, np.unique(kb))]
    j = vv.merge(b, on=["user", "cat"], suffixes=("", "_buy"))
    j = j[(j.t <= j.t_buy) & (j.t > j.t_buy - 7 * DAY) & (j["product"] != j.product_buy)]
    return j.groupby("buy").size().reindex(b.buy, fill_value=0)


def rhythm(sessions: Path, bulk: np.ndarray) -> dict:
    s = pq.read_table(sessions, columns=["user_id", "start"]).to_pandas()
    s = s[~np.isin(s.user_id.to_numpy(), bulk)]
    local = s.start + pd.Timedelta(
        hours=3
    )  # REES46 is a Russian store: Moscow time for readability
    s["hour"] = local.dt.hour
    s["weekend"] = local.dt.day_of_week >= 5
    oct_, nov = (
        s[s.start < pd.Timestamp("2019-11-01", tz="UTC")],
        s[s.start >= pd.Timestamp("2019-11-01", tz="UTC")],
    )
    counts = oct_.groupby("user_id").size()
    regular = counts.index[counts >= 5]
    o = oct_[oct_.user_id.isin(regular)]
    # Personal usual hour: the 3-hour window (circular) with most October visits.
    table = pd.crosstab(o.user_id, o.hour).reindex(columns=range(24), fill_value=0)
    hist = table.to_numpy()
    window = hist + np.roll(hist, 1, axis=1) + np.roll(hist, -1, axis=1)
    personal = pd.Series(window.argmax(axis=1), index=table.index)
    glob = np.bincount(oct_.hour, minlength=24)
    glob_hour = int((glob + np.roll(glob, 1) + np.roll(glob, -1)).argmax())
    n = nov[nov.user_id.isin(personal.index)]
    n_counts = n.groupby("user_id").size()
    n = n[n.user_id.isin(n_counts.index[n_counts >= 3])]

    def near(h: np.ndarray, target: np.ndarray) -> np.ndarray:
        d = np.abs(h - target) % 24
        return np.minimum(d, 24 - d) <= 1  # the same 3-hour window

    hit_personal = near(n.hour.to_numpy(), personal.reindex(n.user_id).to_numpy())
    hit_global = near(n.hour.to_numpy(), np.full(len(n), glob_hour))
    wk_o = o.groupby("user_id").weekend.mean()
    wk_n = n.groupby("user_id").weekend.mean()
    both = pd.concat([wk_o, wk_n], axis=1, keys=["a", "b"]).dropna()
    return {
        "shoppers": int(n.user_id.nunique()),
        "nov_visits": int(len(n)),
        "store_busiest_3h_window_center_msk": glob_hour,
        "nov_visits_in_personal_3h_window": round(float(hit_personal.mean()), 3),
        "nov_visits_in_store_busiest_3h_window": round(float(hit_global.mean()), 3),
        "weekend_share_spearman_oct_vs_nov": round(spearman(both.a, both.b), 3),
    }


def main(events: Path, sessions: Path) -> None:
    products = load_products(events)
    p = purchases(events, products)
    items = p.groupby("user").size()
    bulk = items.index[items > BULK_ITEMS].to_numpy()
    p = p[~np.isin(p.user.to_numpy(), bulk)].reset_index(drop=True)
    v = first_views(events, products, bulk)

    viewers = v.groupby("product").size()
    products["viewers"] = viewers.reindex(products.index, fill_value=0)
    size = products.groupby("cat").viewers.transform("size")
    pop_pct = products.groupby("cat").viewers.rank(pct=True).where(size >= MIN_PRODUCTS)

    out: dict = {"bulk_buyers_removed": int(len(bulk)), "traits": {}}
    out["traits"]["brand_loyalty"] = stability(lambda per: brand_loyalty(v, per))
    out["traits"]["mainstream"] = stability(lambda per: mainstream(v, pop_pct, per))
    out["traits"]["variety"] = stability(lambda per: variety(v, per))
    excess = deal_excess(events, products, p)
    out["traits"]["deal_sensitivity"] = stability(lambda per: per_user_purchases(p, excess, per))
    alts = alternatives_before(v, p)
    out["traits"]["comparison_breadth"] = stability(
        lambda per: per_user_purchases(p, alts.astype(float), per, how="median")
    )

    # Does Oct brand loyalty predict sticking with the brand viewed most in Oct?
    loyal = out["traits"]["brand_loyalty"]["_oct"]
    ov = v[(v.t < OCT_END) & (v.brand >= 0)]
    counts = ov.groupby(["user", "cat", "brand"]).size().rename("n").reset_index()
    counts = counts.sort_values(["user", "cat", "n"], ascending=[True, True, False])
    top = counts.drop_duplicates(["user", "cat"]).set_index(["user", "cat"]).brand.rename("top")
    nb = p[p.t >= OCT_END].copy()
    nb["brand"] = index_of(nb["product"].to_numpy(), products, "brand", -1)
    nb = nb[nb.brand >= 0].join(top, on=["user", "cat"]).dropna(subset=["top"])
    nb["tercile"] = (
        pd.qcut(loyal.rank(method="first"), 3, labels=["low", "mid", "high"])
        .reindex(nb.user)
        .to_numpy()
    )
    nb = nb.dropna(subset=["tercile"])
    out["brand_prediction"] = {
        str(t): {
            "purchases": int(len(g)),
            "bought_oct_top_brand": round(float((g.brand == g.top).mean()), 3),
        }
        for t, g in nb.groupby("tercile", observed=True)
    }
    out["rhythm"] = rhythm(sessions, bulk)
    for t in out["traits"].values():
        t.pop("_oct")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
