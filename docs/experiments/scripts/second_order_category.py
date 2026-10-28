"""Does the second order stay in the first order's category, and does it matter?

Questions:
1. How often does the 2nd order stay inside the categories of the 1st (no new
   category), or at least repeat one? Compared with chance: 2nd orders shuffled
   between customers with the same number of categories in each order.
2. Stayers vs expanders: do customers whose 2nd order adds a new category end up
   more loyal? Raw, and adjusted for the gap to the 2nd order and the 2nd order's
   size (stratified: the average within-stratum difference, weighted by
   expanders).
3. When do customers expand? The order at which the first new category appears,
   for loyal customers vs the rest, and breadth by order 6 for stayers vs
   expanders at order 2 (do stayers catch up?).

Datasets reuse the loaders in paths_to_loyalty.py (same cohorts and definitions):
  hm      H&M fashion; category = product group; 12-month follow-up.
  olist   Olist marketplace; category = product category; 12 months.
  x5      X5 grocery, new card holders; category = product level 2; 90 days.
  instacart  Grocery, shoppers with 10+ orders (survivorship); category =
          department; outcome = weekly regular (median gap over orders 4-10 <= 7 days).

Aggregates only; cells under 50 customers are suppressed.

Usage:
  uv run --no-project --with pandas --with pyarrow python -I \
    second_order_category.py hm TRANSACTIONS_DIR ARTICLES_PARQUET > hm.json
  ... olist OLIST_DIR | x5 X5_DIR | instacart INSTACART_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths_to_loyalty as ptl  # noqa: E402

MIN_CELL = ptl.MIN_CELL
r = ptl.r


def order_sets(cats: pd.DataFrame) -> pd.DataFrame:
    """(c, k) -> frozenset of categories, for orders 1..6."""
    x = cats[cats.k <= 6]
    return x.groupby(["c", "k"]).cat.agg(frozenset).unstack("k")


def stay_rates(s1: pd.Series, s2: pd.Series, seed: int = 0, reps: int = 5) -> dict:
    """Observed vs chance: 2nd order inside the 1st's categories / repeats one."""
    within = np.array([b <= a for a, b in zip(s1, s2, strict=True)])
    repeat = np.array([bool(a & b) for a, b in zip(s1, s2, strict=True)])
    share_old = np.array([len(a & b) / len(b) for a, b in zip(s1, s2, strict=True)])
    strata = pd.DataFrame({"n1": s1.map(len).clip(upper=5), "n2": s2.map(len).clip(upper=5)})
    rng = np.random.default_rng(seed)
    w_null, r_null = [], []
    s2v = s2.to_numpy()
    for _ in range(reps):
        perm = np.arange(len(s2v))
        for _, idx in strata.groupby(["n1", "n2"]).indices.items():
            perm[idx] = rng.permutation(idx)
        shuffled = s2v[perm]
        w_null.append(np.mean([b <= a for a, b in zip(s1, shuffled, strict=True)]))
        r_null.append(np.mean([bool(a & b) for a, b in zip(s1, shuffled, strict=True)]))
    return {
        "customers_with_2nd_order": int(len(s1)),
        "median_categories_1st_order": float(s1.map(len).median()),
        "2nd_order_no_new_category": r(within.mean()),
        "chance_no_new_category": r(np.mean(w_null)),
        "2nd_order_repeats_a_category": r(repeat.mean()),
        "chance_repeats_a_category": r(np.mean(r_null)),
        "share_of_2nd_order_categories_already_bought": r(share_old.mean()),
    }


def adjusted_gap(outcome: pd.Series, expanded: pd.Series, strata: pd.DataFrame) -> dict:
    """Expanders minus stayers, raw and within strata (weighted by expanders)."""
    df = pd.DataFrame({"y": outcome.astype(float), "e": expanded}).join(strata)
    raw = df[df.e].y.mean() - df[~df.e].y.mean()
    diffs, weights = [], []
    for _, g in df.groupby(list(strata.columns), observed=True):
        ne, ns = int(g.e.sum()), int((~g.e).sum())
        if ne >= MIN_CELL and ns >= MIN_CELL:
            diffs.append(g[g.e].y.mean() - g[~g.e].y.mean())
            weights.append(ne)
    adj = float(np.average(diffs, weights=weights)) if diffs else float("nan")
    return {
        "stayers": r(df[~df.e].y.mean()),
        "expanders": r(df[df.e].y.mean()),
        "n_stayers": int((~df.e).sum()),
        "n_expanders": int(df.e.sum()),
        "raw_difference": r(raw),
        "adjusted_difference": r(adj),
        "share_of_expanders_in_compared_strata": r(sum(weights) / max(df.e.sum(), 1)),
    }


def expansion_timing(sets: pd.DataFrame, flag: pd.Series) -> dict:
    """First order (2..6) that adds a new category, among customers with 6+ orders."""
    six = sets.dropna(subset=[6]).index.intersection(flag.index)
    first_new = pd.Series(7, index=six)
    seen = sets.loc[six, 1].copy()
    for k in range(2, 7):
        new = pd.Series([bool(b - a) for a, b in zip(seen, sets.loc[six, k], strict=True)],
                        index=six)  # fmt: skip
        first_new[(first_new == 7) & new] = k
        seen = pd.Series([a | b for a, b in zip(seen, sets.loc[six, k], strict=True)], index=six)
    out = {}
    for name, idx in (("loyal", six[flag.reindex(six)]), ("others", six[~flag.reindex(six)])):
        if len(idx) < MIN_CELL:
            continue
        v = first_new[idx].replace(7, "none by 6").astype(str).value_counts(normalize=True)
        out[name] = {"n": int(len(idx)), **{f"order_{k}": r(p) for k, p in v.sort_index().items()}}
    return out


def breadth_catch_up(sets: pd.DataFrame, expanded: pd.Series) -> dict:
    """Distinct categories after order 6, by what order 2 did."""
    six = sets.dropna(subset=[6]).index.intersection(expanded.index)
    breadth = pd.Series(
        [len(frozenset().union(*row)) for row in sets.loc[six, list(range(1, 7))].to_numpy()],
        index=six,
    )
    b1 = sets.loc[six, 1].map(len)
    out = {}
    for name, e in (("stayed_at_2", False), ("expanded_at_2", True)):
        idx = six[expanded.reindex(six) == e]
        if len(idx) >= MIN_CELL:
            out[name] = {"n": int(len(idx)), "categories_order_1": r(b1[idx].mean(), 2),
                         "categories_after_order_6": r(breadth[idx].mean(), 2)}  # fmt: skip
    return out


def analyse(name: str, sets: pd.DataFrame, outcomes: dict[str, pd.Series],
            strata: pd.DataFrame, flag: pd.Series | None) -> dict:  # fmt: skip
    two = sets.dropna(subset=[2]).index
    s1, s2 = sets.loc[two, 1], sets.loc[two, 2]
    expanded = pd.Series([bool(b - a) for a, b in zip(s1, s2, strict=True)], index=two)
    out = {"dataset": name, "stay_vs_chance": stay_rates(s1, s2)}
    out["expanders_minus_stayers"] = {
        k: adjusted_gap(y.reindex(two), expanded, strata.reindex(two)) for k, y in outcomes.items()
    }
    if 6 in sets.columns and flag is not None:
        out["first_new_category_order"] = expansion_timing(sets, flag)
        out["breadth_after_6_orders"] = breadth_catch_up(sets, expanded)
    return out


def from_loader(data: ptl.Data) -> dict:
    o, cats = ptl.cohort_orders(data)
    cust = ptl.customers(o, data.horizon)
    sets = order_sets(cats)
    two = cust[cust.orders >= 2].index
    o2 = o[o.k == 2].set_index("c").reindex(two)
    gap = pd.cut(cust.days_to_2nd.reindex(two), [-1, 7, 30, 90, 1e9],
                 labels=["0-7", "8-30", "31-90", "91+"])  # fmt: skip
    size = pd.cut(o2["items"], [0, 1, 3, 1e9], labels=["1", "2-3", "4+"])
    strata = pd.DataFrame({"gap": gap, "size": size}, index=two)
    third = cust.orders.reindex(two) >= 3
    outcomes = {"3rd_order": third}
    if "loyal" in cust:
        outcomes["loyal_12m"] = cust.loyal.reindex(two)
        outcomes["orders_12m"] = cust.orders.reindex(two)
        flag = cust.loyal
    else:  # X5: 90 days
        late = o[(o.d - o["first"]) > 30].groupby("c").size()
        outcomes["weekly_days_31_90"] = late.reindex(two, fill_value=0) >= 9
        outcomes["orders_90d"] = cust.orders.reindex(two)
        flag = cust.orders >= ptl.LOYAL_ORDERS
    return analyse(data.name, sets, outcomes, strata, flag)


def instacart(d: str) -> dict:
    d = Path(d)
    cols = ["order_id", "user_id", "eval_set", "order_number", "days_since_prior_order"]
    orders = pd.read_csv(d / "orders.csv", usecols=cols)
    orders = orders[orders.eval_set == "prior"]
    n_orders = orders.groupby("user_id").order_number.max()
    users = n_orders[n_orders >= 10].index
    orders = orders[orders.user_id.isin(users) & (orders.order_number <= 10)]
    products = pd.read_csv(d / "products.csv", usecols=["product_id", "department_id"])
    depts = pd.read_csv(d / "departments.csv").set_index("department_id").department
    dept_of = np.zeros(products.product_id.max() + 1, dtype=np.int16)
    dept_of[products.product_id.to_numpy()] = products.department_id.to_numpy()
    op = pd.read_csv(d / "order_products__prior.csv", usecols=["order_id", "product_id"],
                     dtype={"order_id": np.int32, "product_id": np.int32})  # fmt: skip
    op = op[op.order_id.isin(orders.order_id)]
    lines = op.groupby("order_id").size().rename("items")
    key = np.unique(op.order_id.to_numpy(np.int64) * 64 + dept_of[op.product_id.to_numpy()])
    del op
    od = pd.DataFrame({"order_id": key // 64, "cat": pd.Series(key % 64).map(depts).to_numpy()})
    od = od.merge(orders[["order_id", "user_id", "order_number"]], on="order_id")
    od = od.rename(columns={"user_id": "c", "order_number": "k"})
    sets = order_sets(od)
    o = orders.set_index(["user_id", "order_number"])
    gap2 = o.days_since_prior_order.xs(2, level="order_number")
    regular = (
        orders[orders.order_number.between(4, 10)]
        .groupby("user_id")
        .days_since_prior_order.median()
        <= 7
    )
    size2 = lines.reindex(o.order_id.xs(2, level="order_number")).to_numpy()
    strata = pd.DataFrame(
        {
            "gap": pd.cut(gap2, [-1, 7, 14, 29, 30], labels=["0-7", "8-14", "15-29", "30+"]),
            "size": pd.qcut(pd.Series(size2, index=gap2.index).rank(method="first"), 4,
                            labels=["Q1", "Q2", "Q3", "Q4"]),
        }
    )  # fmt: skip
    out = analyse("instacart", sets, {"weekly_regular": regular}, strata, regular)
    out["caveat"] = "shoppers with 10+ orders only; gaps capped at 30 days"
    return out


if __name__ == "__main__":
    kind, *paths = sys.argv[1:]
    if kind == "instacart":
        res = instacart(paths[0])
    else:
        loader = {"hm": ptl.load_hm, "olist": ptl.load_olist, "x5": ptl.load_x5}[kind]
        res = from_loader(loader(*paths))
    print(json.dumps(res, indent=1, default=str))
