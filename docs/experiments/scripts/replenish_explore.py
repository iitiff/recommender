"""Replenish or explore? Shopper rhythm, product cycles and new-category exposure.

instacart DIR (groceries; orders.csv, order_products__prior.csv, products.csv, aisles.csv):
  1. Rhythm: each shopper's median gap and its regularity (CV), from the
     first half of their orders; cadence classes.
  2. When does a regular shopper become "overdue"? For regular shoppers,
     the chance of ordering on each day since the last order, given no
     order yet (discrete-time hazard), by day relative to their own median.
  3. Product cycles: days between purchases of the same product by the
     same shopper, per aisle; how regular they are.
  4. Room to explore by lifecycle: share of orders with a product or an aisle
     new to the shopper, by order number; and whether a first purchase in a
     new aisle is repeated within the next 3 orders (trial -> adoption).

rees46 events.parquet (from stream_rees46_events.py):
  5. Exposure curve: for product types a shopper hadn't bought before, the
     chance of buying in the type at the k-th visit in which they view it,
     given no purchase in earlier visits (discrete-time hazard over exposures);
     overall, by the type's price band (median price paid) and by top level.

Instacart caps days between orders at 30, so day counts beyond that are
lumped together.

Usage:
  python -I replenish_explore.py instacart DIR > out.json
  uv run --no-project --with pandas --with pyarrow python -I \
    replenish_explore.py rees46 events.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def r(x: float, n: int = 3) -> float:
    return round(float(x), n)


# -- Instacart ------------------------------------------------------------------


def load_instacart(d: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    orders = pd.read_csv(
        d / "orders.csv",
        usecols=["order_id", "user_id", "eval_set", "order_number", "days_since_prior_order"],
    )
    orders = orders[orders.eval_set == "prior"].sort_values(["user_id", "order_number"])
    orders["day"] = (
        orders.days_since_prior_order.fillna(0).groupby(orders.user_id).cumsum().astype(np.int32)
    )
    products = pd.read_csv(d / "products.csv", usecols=["product_id", "aisle_id"])
    aisles = pd.read_csv(d / "aisles.csv").set_index("aisle_id").aisle
    op = pd.read_csv(
        d / "order_products__prior.csv",
        usecols=["order_id", "product_id"],
        dtype={"order_id": np.int32, "product_id": np.int32},
    )
    aisle_of = np.zeros(products.product_id.max() + 1, dtype=np.int16)
    aisle_of[products.product_id.to_numpy()] = products.aisle_id.to_numpy()
    meta = orders.set_index("order_id")[["user_id", "order_number", "day"]]
    idx = meta.index.get_indexer(op.order_id)
    lines = pd.DataFrame(
        {
            "user": meta.user_id.to_numpy()[idx].astype(np.int32),
            "n": meta.order_number.to_numpy()[idx].astype(np.int16),
            "day": meta.day.to_numpy()[idx],
            "product": op.product_id.to_numpy(),
            "aisle": aisle_of[op.product_id.to_numpy()],
        }
    )
    return orders, lines, aisles


def rhythm(orders: pd.DataFrame) -> dict:
    o = orders[orders.groupby("user_id").order_number.transform("max") >= 8]
    half = o.groupby("user_id").order_number.transform("max") // 2
    first, second = o[o.order_number <= half], o[o.order_number > half]
    g1 = first.dropna(subset=["days_since_prior_order"]).groupby("user_id").days_since_prior_order
    med, cv = g1.median(), g1.std() / g1.mean()
    cadence = pd.cut(
        med, [-1, 8, 16, 29, 31], labels=["weekly", "every 2 weeks", "monthly", "30+ (capped)"]
    )
    regular = cv <= 0.5
    out = {
        "shoppers_8plus_orders": int(len(med)),
        "cadence_share": {
            str(k): r(v) for k, v in cadence.value_counts(normalize=True).sort_index().items()
        },
        "regular_share_cv_le_0.5": r(regular.mean()),
        "regular_share_by_cadence": {
            str(k): r(regular[cadence == k].mean())
            for k in cadence.cat.categories
            if (cadence == k).any()
        },
    }

    # Hazard by day since the last order, for regular weekly shoppers (median 6-8 days).
    weekly = med.index[(med >= 6) & (med <= 8) & regular]
    gaps = second[second.user_id.isin(weekly)].days_since_prior_order.dropna().astype(int)
    counts = np.bincount(gaps.clip(upper=30), minlength=31)
    at_risk = counts[::-1].cumsum()[::-1]
    out["weekly_regular_shoppers"] = int(len(weekly))
    out["weekly_hazard_by_day"] = {
        d: r(counts[d] / at_risk[d]) for d in range(1, 22) if at_risk[d] >= 100
    }
    out["weekly_cumulative_by_day"] = {
        d: r(counts[: d + 1].sum() / counts.sum()) for d in (6, 7, 8, 9, 10, 12, 14, 21)
    }

    # Relative to each regular shopper's own median: share of next orders placed by k × median.
    reg = med.index[regular & (med >= 3) & (med <= 20)]
    s = second[second.user_id.isin(reg)].dropna(subset=["days_since_prior_order"])
    ratio = s.days_since_prior_order / s.user_id.map(med)
    out["regular_next_order_by_multiple_of_own_median"] = {
        str(k): r((ratio <= k).mean()) for k in (0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
    }
    out["irregular_next_order_by_multiple_of_own_median"] = {}
    irr = med.index[~regular & (med >= 3) & (med <= 20)]
    si = second[second.user_id.isin(irr)].dropna(subset=["days_since_prior_order"])
    ri = si.days_since_prior_order / si.user_id.map(med)
    out["irregular_next_order_by_multiple_of_own_median"] = {
        str(k): r((ri <= k).mean()) for k in (0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
    }
    return out


def product_cycles(lines: pd.DataFrame, aisles: pd.Series, top: int = 15) -> dict:
    x = lines.sort_values(["user", "product", "day"], kind="stable")
    u, p, d = (x[c].to_numpy() for c in ("user", "product", "day"))
    same = (u[1:] == u[:-1]) & (p[1:] == p[:-1])
    gap = (d[1:] - d[:-1])[same]
    aisle = x.aisle.to_numpy()[1:][same]
    pairs = pd.DataFrame({"aisle": aisle, "gap": gap, "user": u[1:][same], "product": p[1:][same]})
    pairs = pairs[pairs.gap > 0]  # same-day duplicates aren't a cycle
    rebuy = x.groupby(["user", "product"]).size()
    by_aisle = pairs.groupby("aisle").gap.agg(["median", "size"])
    by_aisle = by_aisle[by_aisle["size"] >= 20_000].sort_values("size", ascending=False).head(top)
    stats = pairs.groupby(["user", "product"]).gap.agg(["mean", "std", "size"])
    stats = stats[stats["size"] >= 3]
    regular = (stats["std"] / stats["mean"] <= 0.5).mean()
    return {
        "user_products": int(len(rebuy)),
        "share_bought_more_than_once": r((rebuy > 1).mean()),
        "repeat_interval_median_days": r(np.median(pairs.gap), 1),
        "user_products_with_4plus_purchases_regular_cv_le_0.5": r(regular),
        "top_aisles_by_repeat_volume": {
            str(aisles.get(a, a)): {"repeats": int(row["size"]), "median_days": r(row["median"], 1)}
            for a, row in by_aisle.iterrows()
        },
    }


def exploration(lines: pd.DataFrame) -> dict:
    x = lines.sort_values(["user", "n"], kind="stable")
    x["new_product"] = ~x.duplicated(["user", "product"])
    x["new_aisle"] = ~x.duplicated(["user", "aisle"])
    per_order = x.groupby(["user", "n"]).agg(
        new_product=("new_product", "mean"), new_aisle=("new_aisle", "any")
    )
    per_order = per_order.reset_index()
    bucket = pd.cut(
        per_order.n, [0, 1, 3, 10, 30, 1000], labels=["1st", "2-3", "4-10", "11-30", "31+"]
    )
    out = {
        str(b): {
            "orders": int((bucket == b).sum()),
            "share_of_items_new_to_shopper": r(per_order.new_product[bucket == b].mean()),
            "orders_with_a_new_aisle": r(per_order.new_aisle[bucket == b].mean()),
        }
        for b in bucket.cat.categories
    }
    # Trial -> adoption: a first purchase in an aisle (from order 2 on), bought
    # again within the shopper's next 3 orders (only if they placed 3 more).
    last = x.groupby("user").n.transform("max")
    trials = x[x.new_aisle & (x.n >= 2) & (x.n <= last - 3)][
        ["user", "aisle", "n"]
    ].drop_duplicates()
    later = (
        x[["user", "aisle", "n"]]
        .drop_duplicates()
        .merge(trials, on=["user", "aisle"], suffixes=("", "_t"))
    )
    adopted = later[(later.n > later.n_t) & (later.n <= later.n_t + 3)][
        ["user", "aisle", "n_t"]
    ].drop_duplicates()
    return {
        "by_order_number": out,
        "new_aisle_trials": int(len(trials)),
        "trial_repeated_within_next_3_orders": r(len(adopted) / max(len(trials), 1)),
    }


# -- REES46: exposure curve -------------------------------------------------------


PRICE_BANDS = [0, 25, 100, 300, 1e9]
PRICE_LABELS = ["under $25", "$25-100", "$100-300", "$300+"]


def curve(k_max: pd.Series, bought: pd.Series, max_k: int = 10, min_reach: int = 1000) -> dict:
    """Hazard over exposures: P(buy at the k-th viewing visit | not bought before)."""
    hazard, reach = {}, {}
    for k in range(1, max_k + 1):
        reached = k_max >= k
        if reached.sum() < min_reach:
            break
        reach[k] = int(reached.sum())
        hazard[k] = r((bought & (k_max == k)).sum() / reached.sum(), 4)
    peak = max(hazard, key=hazard.get) if hazard else None
    return {
        "pairs": int(len(k_max)),
        "bought_within_window": r(bought.mean(), 4),
        "hazard": hazard,
        "reach": reach,
        "peak_k": peak,
        "peak_vs_first": r(hazard[peak] / hazard[1], 2) if peak and hazard[1] else None,
        "last_k_vs_peak": r(hazard[max(hazard)] / hazard[peak], 2)
        if peak and hazard[peak]
        else None,
        "eventually_buy_given_reached_k": {
            k: r((bought & (k_max >= k)).sum() / (k_max >= k).sum(), 4) for k in reach
        },
    }


def exposure_curve(path: Path, max_k: int = 10) -> dict:
    import pyarrow.parquet as pq

    names = json.loads(path.with_suffix(".json").read_text())["category_codes"]
    type_of_code = {i: ".".join(n.split(".")[:2]) for n, i in names.items()}
    ev = pq.read_table(path, columns=["user", "t", "kind", "code", "session", "price"]).to_pandas()
    ev = ev[ev.code >= 0]
    codes = pd.Series(type_of_code)
    type_cat = pd.Categorical(codes.reindex(ev.code.to_numpy()).to_numpy())
    ev["type"] = type_cat.codes.astype(np.int16)
    ev = ev.drop(columns="code")
    # A type's typical price: the median price paid in it.
    type_price = ev[ev.kind == 2].groupby("type").price.median()
    buys = ev[ev.kind == 2].groupby(["user", "type"]).t.min().rename("t_buy")
    views = ev[ev.kind == 0].drop(columns=["kind", "price"])
    del ev
    # One row per (shopper, type, visit): the visit's first view of that type.
    v = views.groupby(["user", "type", "session"]).t.min().reset_index()
    del views
    v = v.join(buys, on=["user", "type"])
    first_view = v.groupby(["user", "type"]).t.transform("min")
    # "New" type: not bought before the first time they viewed it; and allow a
    # week of history before it so that earlier purchases are visible.
    v = v[(first_view >= 7 * 86_400) & ~(v.t_buy < first_view)]
    v = v[v.t_buy.isna() | (v.t <= v.t_buy)]  # exposures up to (and including) the buying visit
    v = v.sort_values(["user", "type", "t"])
    v["k"] = v.groupby(["user", "type"]).cumcount() + 1
    k_max = v.groupby(["user", "type"]).k.max()
    bought = v.groupby(["user", "type"]).t_buy.first().notna()

    pair_type = k_max.index.get_level_values("type")
    band = pd.cut(
        type_price.reindex(pair_type).to_numpy(), PRICE_BANDS, labels=PRICE_LABELS, right=False
    )
    top = pd.Series(np.asarray(type_cat.categories)).str.split(".").str[0]
    top_of_pair = top.reindex(pair_type).to_numpy()
    out = {"all": curve(k_max, bought, max_k), "by_price_band": {}, "by_top_level": {}}
    for label in PRICE_LABELS:
        m = np.asarray(band == label)
        if m.sum() >= 20_000:
            out["by_price_band"][label] = curve(k_max[m], bought[m], max_k)
    for name in pd.unique(top_of_pair):
        m = top_of_pair == name
        if m.sum() >= 20_000:
            out["by_top_level"][str(name)] = curve(k_max[m], bought[m], max_k) | {
                "median_price": r(type_price.reindex(pair_type[m]).median(), 0)
            }
    return out


if __name__ == "__main__":
    mode, path = sys.argv[1], Path(sys.argv[2])
    if mode == "instacart":
        orders, lines, aisles = load_instacart(path)
        result = {
            "rhythm": rhythm(orders),
            "product_cycles": product_cycles(lines, aisles),
            "exploration": exploration(lines),
        }
    else:
        result = {"exposure": exposure_curve(path)}
    print(json.dumps(result, indent=1))
