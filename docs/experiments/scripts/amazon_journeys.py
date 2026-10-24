"""Shopper traits and journeys over years (Amazon Reviews 2023, 3 categories).

A review is treated as evidence of a purchase (only purchases people chose
to review, so a biased subset). Input: parquet files from stream_amazon.py.

1. Year-to-year stability of a shopper's price level, mainstream taste
   (item popularity) and rating style: year Y vs Y+1 and Y vs Y+3.
2. Per-type price band: is a shopper's price position in a product type a
   lasting preference? Same type a year apart vs a different type a year apart.
3. Brand loyalty over time: consecutive purchases in the same leaf category,
   P(same brand) vs chance, by the time between them.
4. Journeys: breadth growth with tenure, next new category entered (lift
   over its base rate), and repeat purchases in the same leaf category
   (consumables vs durables).

Usage:
  uv run --no-project --with pandas --with pyarrow python -I \
    amazon_journeys.py DIR OUT_DIR [SECTION ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DOMAINS = ["Grocery_and_Gourmet_Food", "Electronics", "Home_and_Kitchen"]
MIN_ITEMS = 20
YEAR = 365 * 86_400


def spearman(a, b) -> float:  # noqa: ANN001
    return float(pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank()))


def load(d: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = pd.concat(
        [
            pd.read_parquet(d / f"{c}.ratings.parquet").assign(domain=np.int8(i))
            for i, c in enumerate(DOMAINS)
        ],
        ignore_index=True,
    )
    items = pd.concat(
        [pd.read_parquet(d / f"{c}.items.parquet") for c in DOMAINS], ignore_index=True
    )
    items = items.drop_duplicates("item").set_index("item")
    for col in ("cat2", "cat3", "brand"):
        items[col] = items[col].replace("", np.nan).astype("category")
    items["n"] = r.groupby("item").size().reindex(items.index, fill_value=0)
    priced = items.price.where(items.price > 0)
    size = priced.groupby(items.cat3, observed=True).transform("count")
    items["price_pct"] = (
        priced.groupby(items.cat3, observed=True).rank(pct=True).where(size >= MIN_ITEMS)
    )
    csize = items.groupby("cat3", observed=True).n.transform("size")
    items["pop_pct"] = (
        items.groupby("cat3", observed=True).n.rank(pct=True).where(csize >= MIN_ITEMS)
    )

    # Every test needs 2+ purchases per shopper; popularity above used everyone.
    per_user = r.groupby("user").size()
    r = r[r.user.isin(per_user.index[per_user >= 2])]
    idx = items.index.get_indexer(r["item"])
    keep = idx >= 0
    idx = idx[keep]
    # Compact columns: ~35 bytes per rating instead of ~60 (tens of millions of rows).
    c = pd.DataFrame(
        {
            "user": pd.factorize(r.user.to_numpy()[keep])[0].astype("int32"),
            "item": idx.astype("int32"),
            "rating": r.rating.to_numpy()[keep],
            "t": r.t.to_numpy()[keep].astype("int32"),
            "domain": r.domain.to_numpy()[keep],
        }
    )
    del r
    for col in ("price_pct", "pop_pct"):
        c[col] = items[col].to_numpy(dtype="float32")[idx]
    for col in ("cat2", "cat3", "brand"):
        c[col] = items[col].cat.codes.to_numpy()[idx].astype("int32")  # -1 = missing
    c["year"] = pd.to_datetime(c.t, unit="s").dt.year.astype("int16")
    return c, items


def year_pairs(r: pd.DataFrame) -> dict:
    uy = r.groupby(["user", "year"]).agg(
        price=("price_pct", "mean"),
        n_price=("price_pct", "count"),
        pop=("pop_pct", "mean"),
        n_pop=("pop_pct", "count"),
        rating=("rating", "mean"),
        n=("rating", "size"),
    )
    uy = uy[uy.n >= 3].reset_index()
    out = {}
    for gap in (1, 3):
        later = uy.assign(year=uy.year - gap)
        j = uy.merge(later, on=["user", "year"], suffixes=("", "_later"))
        row = {}
        for trait, n in (("price", "n_price"), ("pop", "n_pop"), ("rating", "n")):
            g = j[(j[n] >= 3) & (j[f"{n}_later"] >= 3)].dropna(subset=[trait, f"{trait}_later"])
            row[trait] = {
                "pairs": int(len(g)),
                "spearman": round(spearman(g[trait], g[f"{trait}_later"]), 3),
            }
        out[f"{gap} year apart"] = row
    return out


def type_band(r: pd.DataFrame, seed: int = 0) -> dict:
    x = r.loc[r.cat2 >= 0, ["user", "year", "cat2", "price_pct"]].dropna(subset=["price_pct"])
    ut = x.groupby(["user", "year", "cat2"]).price_pct.agg(["mean", "count"])
    ut = ut[ut["count"] >= 2].reset_index()
    nxt = ut.assign(year=ut.year - 1)
    j = ut.merge(nxt, on=["user", "year"], suffixes=("", "_next"))
    same = j[j.cat2 == j.cat2_next]
    diff = j[j.cat2 != j.cat2_next]
    if len(diff) > 3_000_000:
        diff = diff.sample(3_000_000, random_state=seed)
    return {
        "same_type_next_year": {
            "pairs": int(len(same)),
            "spearman": round(spearman(same["mean"], same.mean_next), 3),
        },
        "other_type_next_year": {
            "pairs": int(len(diff)),
            "spearman": round(spearman(diff["mean"], diff.mean_next), 3),
        },
    }


def brand_loyalty(r: pd.DataFrame) -> dict:
    x = r.loc[(r.cat3 >= 0) & (r.brand >= 0), ["user", "cat3", "brand", "t", "domain"]]
    x = x.sort_values(["user", "cat3", "t"])
    shares = x.groupby(["cat3", "brand"]).size() / x.groupby("cat3").size()
    base = (shares**2).groupby(level="cat3").sum()
    u, c, b, t = (x[k].to_numpy() for k in ("user", "cat3", "brand", "t"))
    pair = (u[1:] == u[:-1]) & (c[1:] == c[:-1])
    df = pd.DataFrame(
        {
            "same": (b[1:] == b[:-1])[pair],
            "base": base.reindex(c[1:][pair]).to_numpy(),
            "gap_days": (t[1:] - t[:-1])[pair] / 86_400,
            "domain": x.domain.to_numpy()[1:][pair],
        }
    )
    df["gap"] = pd.cut(
        df.gap_days, [-1, 30, 365, 3 * 365, 1e9], labels=["<30d", "30d-1y", "1-3y", "3y+"]
    )
    out = {}
    for (dom, gap), g in df.groupby(["domain", "gap"], observed=True):
        if len(g) >= 1000:
            out.setdefault(DOMAINS[dom], {})[str(gap)] = {
                "pairs": int(len(g)),
                "same_brand": round(float(g.same.mean()), 3),
                "chance": round(float(g.base.mean()), 3),
            }
    return out


def breadth(r: pd.DataFrame) -> dict:
    """Types entered by tenure year, for shoppers observed at least 5 years."""
    x = r.loc[r.cat2 >= 0, ["user", "t", "cat2"]].sort_values(["user", "t"])
    first_t = x.groupby("user").t.transform("min")
    x["tenure"] = ((x.t - first_t) // YEAR).astype("int16")
    del first_t
    span = x.groupby("user").tenure.max()
    x = x[x.user.isin(span.index[span >= 5])]
    new = ~x.duplicated(["user", "cat2"])  # first time each shopper enters each type
    e = x[new]
    return {
        "shoppers_observed_5y+": int(e.user.nunique()),
        "types_entered_by_end_of_tenure_year": {
            f"year {k}": round(float((e.tenure <= k).groupby(e.user).sum().mean()), 2)
            for k in (0, 1, 2, 4)
        },
        "share_of_purchases_in_a_new_type": {
            f"year {k}": round(float(new[x.tenure == k].mean()), 3) for k in (0, 1, 2, 4)
        },
    }


def cross_domain(r: pd.DataFrame) -> dict:
    """First domain → ever reviews in another of the three (shoppers with 5+ reviews)."""
    x = r[["user", "t", "domain"]].sort_values("t")
    first_dom = x.drop_duplicates("user").set_index("user").domain
    doms = x.groupby("user").domain.nunique()
    multi = x.groupby("user").size() >= 5
    return {
        DOMAINS[d]: round(float((doms[multi & (first_dom == d)] > 1).mean()), 3) for d in range(3)
    }


def next_type(r: pd.DataFrame, items: pd.DataFrame) -> dict:
    """Next new type entered after each type: lift over its base rate among next entries."""
    x = r.loc[r.cat2 >= 0, ["user", "t", "cat2"]].sort_values(["user", "t"])
    ent = x[~x.duplicated(["user", "cat2"])].reset_index(drop=True)
    del x
    same_user = ent.user.eq(ent.user.shift()).to_numpy()
    a = ent.cat2.to_numpy()[:-1][same_user[1:]]
    b = ent.cat2.to_numpy()[1:][same_user[1:]]
    del ent
    trans = pd.DataFrame({"a": a, "b": b})
    base = trans.b.value_counts(normalize=True)
    big = set(trans.b.value_counts().index[trans.b.value_counts() >= 2000])
    names = items.cat2.cat.categories
    adj = {}
    for a_ in trans.a.value_counts().head(12).index:
        g = trans.b[trans.a == a_].value_counts(normalize=True)
        # A meaningful share: lift alone favours tiny merchandising nodes
        # ("Back to School…", promo storefronts) that Amazon mixes into the tree.
        g = g[g.index.isin(big) & (g >= 0.01)]
        lift = (g / base.reindex(g.index)).sort_values(ascending=False).head(4)
        adj[str(names[a_])] = {
            "n": int((trans.a == a_).sum()),
            "next": {
                str(names[b_]): {"share": round(float(g[b_]), 3), "lift": round(float(v), 1)}
                for b_, v in lift.items()
            },
        }
    return adj


def repeat(r: pd.DataFrame) -> dict:
    """Another purchase in the same leaf category within a year (consumables vs durables).

    Numpy sorts instead of DataFrame sorts/duplicated: 110M rows don't fit twice.
    """
    keep = (r.cat3 >= 0).to_numpy()
    u, c, t = r.user.to_numpy()[keep], r.cat3.to_numpy()[keep], r.t.to_numpy()[keep]
    item, dom = r["item"].to_numpy()[keep], r.domain.to_numpy()[keep]
    order = np.lexsort((t, c, u))
    u, c, t, item, dom = u[order], c[order], t[order], item[order], dom[order]
    del order
    has_next = np.r_[(u[1:] == u[:-1]) & (c[1:] == c[:-1]) & (t[1:] - t[:-1] <= YEAR), False]
    out = {}
    for d in range(3):
        m = dom == d
        key = np.sort((u[m].astype(np.int64) << 23) | item[m].astype(np.int64))
        out[DOMAINS[d]] = {
            "purchases": int(m.sum()),
            "share": round(float(has_next[m].mean()), 3),
            "same_item_reviewed_again": round(float((key[1:] == key[:-1]).mean()), 4),
        }
    return out


def log(step: str) -> None:
    rss = int(Path("/proc/self/statm").read_text().split()[1]) * 4096 / 1e9
    print(f"{step}: {rss:.1f} GB", file=sys.stderr, flush=True)


def main(d: Path, out_dir: Path) -> None:
    """Each section is saved as soon as it finishes, so a late failure keeps the rest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    r, items = load(d)
    log("loaded")
    sections = {
        "summary": lambda: {
            "ratings_from_shoppers_with_2plus": int(len(r)),
            "shoppers_with_2plus": int(r.user.nunique()),
            "items": int(len(items)),
            "share_items_priced": round(float(items.price_pct.notna().mean()), 3),
            "share_items_with_brand": round(float(items.brand.notna().mean()), 3),
            "years": [int(r.year.min()), int(r.year.max())],
        },
        "stability_years": lambda: year_pairs(r),
        "type_band": lambda: type_band(r),
        "brand_loyalty": lambda: brand_loyalty(r),
        "breadth": lambda: breadth(r),
        "cross_domain": lambda: cross_domain(r),
        "next_type": lambda: next_type(r, items),
        "repeat": lambda: repeat(r),
    }
    only = set(sys.argv[3:])
    for name, run in sections.items():
        if only and name not in only:
            continue
        (out_dir / f"{name}.json").write_text(json.dumps(run(), indent=1))
        log(name)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
