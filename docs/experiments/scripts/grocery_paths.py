"""Paths to loyal and valuable in grocery: Complete Journey, X5 new card holders, Instacart.

cj DIR  84.51° Complete Journey (CC0 R package `completejourney`): one year (2017)
  of every purchase by 2,469 households who shop the chain frequently; baskets
  with stores, loyalty-card and coupon discounts, departments including
  general merchandise (DRUG GM, COSMETICS, GARDEN CENTER = "GM").
  1. Value grid for the year: loyal = active in at least the median number of
     weeks and active in the last 13 weeks; valuable = spend at or above the
     median. Each group's discount share, coupon redemptions, GM share,
     departments, and share of trips at their main store.
  2. Grocery vs GM: within households that shopped the same number of weeks
     in Q1, does a larger GM share in Q1 go with more (or fewer) active weeks
     in Q2-Q4? And GM adopters (first GM purchase in Q2, none in Q1) vs
     households with no GM in Q1-Q2: change in active weeks from Q1 to Q3
     (difference-in-differences, matched on Q1 activity).
  3. Quarter-to-quarter paths: the activity state each quarter (active weeks
     0 / 1-3 / 4-8 / 9-13) and what the quarter before an upgrade (light ->
     regular) looks like vs households that stayed light.
x5 DIR  X5 RetailHero: loyalty card issued and first order 2018-11-22..12-16,
  followed 90 days. What day 30 says about days 31-90: visits by day 30,
  stores, product categories, share of items below the product's usual price.
instacart DIR  Shoppers with 10+ orders (survivorship: 4+ in the data): what
  orders 1-3 say about becoming a weekly regular (median gap <= 7 days over
  orders 4-10).

Usage:
  uv run --no-project --with pandas --with pyreadr python -I grocery_paths.py cj DIR
  uv run --no-project --with pandas python -I grocery_paths.py x5 DIR
  uv run --no-project --with pandas python -I grocery_paths.py instacart DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_CELL = 30
GM = {"DRUG GM", "COSMETICS", "GARDEN CENTER"}


def r(x: float, n: int = 3) -> float:
    return round(float(x), n)


def rates(df: pd.DataFrame, by: pd.Series, cols: list[str]) -> dict:
    out = {}
    for k, g in df.groupby(by, observed=True):
        if len(g) >= MIN_CELL:
            out[str(k)] = {"n": int(len(g)), "share": r(len(g) / len(df))} | {
                c: r(g[c].mean()) for c in cols
            }
    return out


# -- Complete Journey -----------------------------------------------------------------


def cj(d: str) -> dict:
    import pyreadr

    d = Path(d)
    t = pyreadr.read_r(d / "transactions.rds")[None]
    products = next(iter(pyreadr.read_r(d / "products.rda").values()))
    red = next(iter(pyreadr.read_r(d / "coupon_redemptions.rda").values()))
    dept = products.set_index("product_id").department
    t["dept"] = dept.reindex(t.product_id).to_numpy()
    t["gm"] = t.dept.isin(GM)
    t["quarter"] = np.minimum((t.week - 1) // 13, 3)
    t["disc"] = t.retail_disc.abs() + t.coupon_disc.abs()
    h = t.groupby("household_id")
    cust = pd.DataFrame(
        {
            "active_weeks": h.week.nunique(),
            "trips": h.basket_id.nunique(),
            "spend": h.sales_value.sum(),
            "discount_share": h.disc.sum() / (h.sales_value.sum() + h.disc.sum()),
            "gm_share": t[t.gm].groupby("household_id").sales_value.sum() / h.sales_value.sum(),
            "departments": h.dept.nunique(),
        }
    )
    cust["gm_share"] = cust.gm_share.fillna(0)
    trips = t.drop_duplicates("basket_id")
    cust["main_store_share"] = trips.groupby("household_id").store_id.agg(
        lambda s: s.value_counts().iloc[0] / len(s)
    )
    cust["coupons_redeemed"] = red.groupby("household_id").size().reindex(cust.index).fillna(0)
    late = t[t.week >= 41].groupby("household_id").size().reindex(cust.index).fillna(0) > 0
    loyal = (cust.active_weeks >= cust.active_weeks.median()) & late
    hi = cust.spend >= cust.spend.median()
    names = ["true friends", "barnacles", "butterflies"]
    cust["group"] = np.select([loyal & hi, loyal & ~hi, hi], names, "strangers")
    grid = {}
    for k, g in cust.groupby("group"):
        grid[k] = {"share": r(len(g) / len(cust))} | {
            c: r(g[c].median() if c in ("spend", "trips", "active_weeks") else g[c].mean())
            for c in ["active_weeks", "trips", "spend", "discount_share", "coupons_redeemed",
                      "gm_share", "departments", "main_store_share"]
        }  # fmt: skip
    disc_q = pd.qcut(cust.discount_share.rank(method="first"), 4,
                     labels=["Q1 least", "Q2", "Q3", "Q4 most"])  # fmt: skip
    cust["true_friend"] = (cust.group == "true friends").astype(int)
    cust["barnacle"] = (cust.group == "barnacles").astype(int)

    # 2. grocery vs GM
    q = t.groupby(["household_id", "quarter"]).week.nunique().unstack(fill_value=0)
    q = q.reindex(columns=range(4), fill_value=0)
    gmq = t[t.gm].groupby(["household_id", "quarter"]).sales_value.sum().unstack(fill_value=0)
    gmq = gmq.reindex(index=q.index, columns=range(4), fill_value=0)
    spq = t.groupby(["household_id", "quarter"]).sales_value.sum().unstack(fill_value=0)
    spq = spq.reindex(columns=range(4), fill_value=0)
    x = pd.DataFrame({"q1_weeks": q[0], "later_weeks": q[[1, 2, 3]].sum(axis=1)})
    x["q1_gm_share"] = (gmq[0] / spq[0].where(spq[0] > 0)).fillna(0)
    x = x[x.q1_weeks > 0]
    x["band"] = pd.cut(x.q1_weeks, [0, 3, 8, 13], labels=["1-3", "4-8", "9-13"]).astype(str)
    x["gm_tercile"] = x.groupby("band").q1_gm_share.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    )
    gm_vs_later = {
        f"Q1 weeks {b} / GM share {g}": {"n": int(len(s)), "mean_gm_share": r(s.q1_gm_share.mean()),
                                         "later_active_weeks_of_39": r(s.later_weeks.mean(), 1)}
        for (b, g), s in x.groupby(["band", "gm_tercile"])
    }  # fmt: skip
    adopters = q.index[(gmq[0] == 0) & (gmq[1] > 0)]
    never = q.index[(gmq[0] == 0) & (gmq[1] == 0)]
    did = {}
    for name, idx in (("GM adopters in Q2", adopters), ("no GM in Q1-Q2", never)):
        s = q.loc[idx]
        s = s[s[0] > 0]
        did[name] = {"n": int(len(s)), "q1_weeks": r(s[0].mean(), 2), "q3_weeks": r(s[2].mean(), 2),
                     "change": r((s[2] - s[0]).mean(), 2)}  # fmt: skip
    # matched on Q1 weeks: average change within Q1-weeks strata
    a, b = q.loc[adopters], q.loc[never]
    diffs, w = [], []
    for k in range(1, 14):
        sa, sb = a[a[0] == k], b[b[0] == k]
        if len(sa) >= 5 and len(sb) >= 5:
            diffs.append((sa[2] - sa[0]).mean() - (sb[2] - sb[0]).mean())
            w.append(len(sa))
    did["matched_difference_in_change"] = r(np.average(diffs, weights=w), 2) if w else None

    # 3. quarter-to-quarter states
    levels = ["0", "light 1-3", "mid 4-8", "regular 9-13"]
    state = q.apply(lambda c: pd.cut(c, [-1, 0, 3, 8, 13], labels=levels))
    trans = []
    for k in range(3):
        trans.append(pd.DataFrame({"from": state[k].astype(str), "to": state[k + 1].astype(str)}))
    tr = pd.concat(trans)
    matrix = pd.crosstab(tr["from"], tr["to"], normalize="index").round(3)
    # what the quarter before light -> regular looks like vs light -> light
    feats = []
    for k in range(3):
        tk = t[t.quarter == k]
        hk = tk.groupby("household_id")
        f = pd.DataFrame(
            {
                "departments": hk.dept.nunique(),
                "discount_share": hk.disc.sum() / (hk.sales_value.sum() + hk.disc.sum()),
                "gm_share": tk[tk.gm].groupby("household_id").sales_value.sum()
                / hk.sales_value.sum(),
                "basket": hk.sales_value.sum() / hk.basket_id.nunique(),
            }
        ).fillna({"gm_share": 0})
        f["from"], f["to"] = (
            state[k].reindex(f.index).astype(str),
            state[k + 1].reindex(f.index).astype(str),
        )
        feats.append(f)
    ff = pd.concat(feats)
    up = {}
    for frm in ("light 1-3", "mid 4-8"):
        s = ff[ff["from"] == frm]
        up[frm] = {
            to: {"n": int(len(g)), **{c: r(g[c].mean()) for c in
                                       ("departments", "discount_share", "gm_share", "basket")}}
            for to, g in s.groupby("to") if len(g) >= MIN_CELL
        }  # fmt: skip
    return {
        "households": int(len(cust)),
        "grid": grid,
        "by_discount_share_quartile": rates(cust, disc_q, ["true_friend", "barnacle"]),
        "q1_gm_share_vs_later_activity": gm_vs_later,
        "gm_adoption": did,
        "quarter_transition_matrix": json.loads(matrix.to_json()),
        "quarter_before_moving_from": up,
    }


# -- X5: new card holders, day-30 landmark ---------------------------------------------


def x5(d: str) -> dict:
    d = Path(d)
    start = pd.Timestamp("2018-11-21")
    clients = pd.read_csv(d / "clients.csv.gz", usecols=["client_id", "first_issue_date"],
                          parse_dates=["first_issue_date"])  # fmt: skip
    new = clients[clients.first_issue_date >= start + pd.Timedelta(days=1)].client_id
    products = pd.read_csv(d / "products.csv.gz", usecols=["product_id", "level_2"])
    level2 = products.set_index("product_id").level_2
    cols = ["client_id", "transaction_id", "transaction_datetime", "store_id", "product_id",
            "product_quantity", "trn_sum_from_iss"]  # fmt: skip
    parts = []
    for chunk in pd.read_csv(d / "purchases.csv.gz", usecols=cols, chunksize=5_000_000):
        parts.append(chunk[chunk.client_id.isin(new)])
    x = pd.concat(parts)
    x["day"] = (pd.to_datetime(x.transaction_datetime) - start).dt.days
    first = x.groupby("client_id").day.min()
    cohort = first[(first >= 1) & (first <= 25)].index
    x = x[x.client_id.isin(cohort)]
    x["rel"] = x.day - x.client_id.map(first)
    x = x[x.rel <= 90]
    unit = x.trn_sum_from_iss / x.product_quantity.where(x.product_quantity > 0)
    usual = unit.groupby(x.product_id).transform("median")
    x["below_usual"] = unit < 0.9 * usual
    x["cat"] = level2.reindex(x.product_id).to_numpy()
    e, late = x[x.rel <= 30], x[x.rel > 30]
    ge = e.groupby("client_id")
    f = pd.DataFrame(
        {
            "visits_30d": ge.transaction_id.nunique(),
            "stores_30d": ge.store_id.nunique(),
            "categories_30d": ge.cat.nunique(),
            "below_usual_30d": ge.below_usual.mean(),
            "spend_30d": ge.trn_sum_from_iss.sum(),
        }
    )
    f["visits_31_90"] = late.groupby("client_id").transaction_id.nunique().reindex(f.index)
    f["visits_31_90"] = f.visits_31_90.fillna(0)
    f["spend_31_90"] = late.groupby("client_id").trn_sum_from_iss.sum().reindex(f.index).fillna(0)
    f["regular_31_90"] = (f.visits_31_90 >= 9).astype(int)  # about weekly or more
    f["lapsed_31_90"] = (f.visits_31_90 == 0).astype(int)
    f["high_value_31_90"] = (f.spend_31_90 >= f.spend_31_90.median()).astype(int)
    cols_out = ["regular_31_90", "lapsed_31_90", "high_value_31_90"]
    return {
        "new_card_holders": int(len(f)),
        "rates": {c: r(f[c].mean()) for c in cols_out},
        "by_visits_30d": rates(f, pd.cut(f.visits_30d, [0, 1, 3, 7, 1e9],
                                         labels=["1", "2-3", "4-7", "8+"]), cols_out),
        "by_stores_30d": rates(f, f.stores_30d.clip(upper=3), cols_out),
        "by_categories_30d_tercile": rates(
            f, pd.qcut(f.categories_30d.rank(method="first"), 3, labels=["low", "mid", "high"]),
            cols_out),
        "by_below_usual_price_30d": rates(f, pd.cut(f.below_usual_30d, [-1, 0.1, 0.3, 1],
                                                     labels=["<10%", "10-30%", "30%+"]), cols_out),
    }  # fmt: skip


# -- Instacart: path to weekly regular among shoppers who stay --------------------------


def instacart(d: str) -> dict:
    d = Path(d)
    use = ["order_id", "user_id", "eval_set", "order_number", "days_since_prior_order"]
    orders = pd.read_csv(d / "orders.csv", usecols=use)
    orders = orders[orders.eval_set == "prior"]
    n = orders.groupby("user_id").order_number.max()
    users = n[n >= 10].index
    o = orders[orders.user_id.isin(users) & (orders.order_number <= 10)]
    gaps = o.pivot(index="user_id", columns="order_number", values="days_since_prior_order")
    weekly = (gaps[[4, 5, 6, 7, 8, 9, 10]].median(axis=1) <= 7).astype(int)
    products = pd.read_csv(d / "products.csv", usecols=["product_id", "department_id"])
    dtypes = {"order_id": np.int32, "product_id": np.int32, "reordered": np.int8}
    op = pd.read_csv(d / "order_products__prior.csv", usecols=list(dtypes), dtype=dtypes)
    first3 = o[o.order_number <= 3][["order_id", "user_id", "order_number"]]
    op = op.merge(first3, on="order_id")
    op = op.merge(products, on="product_id")
    g = op.groupby("user_id")
    f = pd.DataFrame({"weekly_regular": weekly})
    f["gap_2"] = gaps[2]
    f["gap_3"] = gaps[3]
    f["basket_1to3"] = g.size() / 3
    f["departments_1to3"] = g.department_id.nunique()
    f["reorder_share_2to3"] = op[op.order_number > 1].groupby("user_id").reordered.mean()
    cols = ["weekly_regular"]
    return {
        "shoppers_10plus_orders": int(len(f)),
        "weekly_regular_rate": r(f.weekly_regular.mean()),
        "by_gap_to_2nd": rates(f, pd.cut(f.gap_2, [-1, 7, 14, 29, 30],
                                         labels=["0-7", "8-14", "15-29", "30+ (capped)"]), cols),
        "by_gap_to_3rd": rates(f, pd.cut(f.gap_3, [-1, 7, 14, 29, 30],
                                         labels=["0-7", "8-14", "15-29", "30+ (capped)"]), cols),
        "by_basket_size_quintile": rates(f, pd.qcut(f.basket_1to3.rank(method="first"), 5,
                                                    labels=["Q1", "Q2", "Q3", "Q4", "Q5"]), cols),
        "by_departments_tercile": rates(f, pd.qcut(f.departments_1to3.rank(method="first"), 3,
                                                   labels=["low", "mid", "high"]), cols),
        "by_reorder_share_2to3": rates(f, pd.cut(f.reorder_share_2to3, [-0.01, 0.25, 0.5, 0.75,
                                                                         1.0]), cols),
    }  # fmt: skip


if __name__ == "__main__":
    mode, arg = sys.argv[1], sys.argv[2]
    print(json.dumps({"cj": cj, "x5": x5, "instacart": instacart}[mode](arg), indent=1,
                     default=str))  # fmt: skip
