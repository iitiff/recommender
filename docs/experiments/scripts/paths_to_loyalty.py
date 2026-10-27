"""From first order to loyal: how many paths are there, and which are common? (H&M)

Cohort: customers whose first order in the data falls between 2019-03-20 and
2019-06-19, i.e. after at least 6 months with no purchase (burn-in; the data
starts 2018-09-20), each followed for 365 days after their first order. An
order is a customer-day. Behavior only: the customer file (age, club status)
is not used, nor index groups such as Ladieswear/Menswear/Baby, which would
label shoppers rather than describe what they bought.

1. Outcome at 12 months (tiers) and the order-by-order funnel (chance of an
   (n+1)-th order given n, and the median days between them).
2. Time to the second order and the 12-month outcome by it.
3. Activity paths: which of the 12 thirty-day months after the first order
   have an order (a 12-bit pattern). How many distinct patterns, how
   concentrated, the most common ones, and rule-based path families with
   their outcomes.
4. Early paths: the first three orders as steps, each described by its gap
   (fast <= 30 days, medium 31-90, slow > 90), whether it adds a product group
   new to the customer (new / same), and its channel. Count of distinct paths,
   the most common, and the share of each that ends loyal.
5. First-order traits and the 12-month outcome: channel, product groups, items.

Channels: sales_channel_id is undocumented; 2 is commonly read as online and
1 as store. Shown as "online (2)" / "store (1)" with that caveat.

Usage:
  uv run --no-project --with pandas --with pyarrow python -I \
    paths_to_loyalty.py hm_transactions_dir articles.parquet > out.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

START = pd.Timestamp("2018-09-20")
COHORT = (181, 273)  # days since START: 2019-03-20 .. 2019-06-19
HORIZON = 365
MIN_CELL = 50


def r(x: float, n: int = 3) -> float:
    return round(float(x), n)


def load(tx_dir: str, articles: str) -> pd.DataFrame:
    """Cohort orders: one row per (customer, day) with channel, groups, items, spend."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    t = ds.dataset(tx_dir).to_table(
        columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    )
    c = pc.dictionary_encode(t["customer_id"]).combine_chunks().indices.to_numpy()
    d = (pd.to_datetime(t["t_dat"].to_pandas()) - START).dt.days.to_numpy().astype(np.int16)
    first = pd.Series(d).groupby(c).min()
    in_cohort = first.between(COHORT[0], COHORT[1] - 1)
    keep = in_cohort.reindex(c).to_numpy() & (d <= first.reindex(c).to_numpy() + HORIZON)

    a = pq.read_table(articles, columns=["article_id", "product_group_name"]).to_pandas()
    groups = sorted(a.product_group_name.unique())
    pg = a.set_index("article_id").product_group_name.map({g: i for i, g in enumerate(groups)})
    lines = pd.DataFrame(
        {
            "c": c[keep],
            "d": d[keep],
            "bit": np.left_shift(1, pg.reindex(t["article_id"].to_numpy()[keep]).to_numpy()),
            "online": t["sales_channel_id"].to_numpy()[keep] == 2,
            "price": t["price"].to_numpy()[keep],
        }
    )
    del t
    o = lines.groupby(["c", "d"]).agg(
        bits=("bit", lambda s: np.bitwise_or.reduce(s.to_numpy())),
        online=("online", "mean"),
        items=("bit", "size"),
        spend=("price", "sum"),
    )
    o = o.reset_index().sort_values(["c", "d"])
    o["k"] = o.groupby("c").cumcount() + 1  # order number
    o["first"] = o.groupby("c").d.transform("min")
    o.attrs["groups"] = groups
    return o


# -- outcomes -----------------------------------------------------------------------


def outcomes(o: pd.DataFrame) -> pd.DataFrame:
    g = o.groupby("c")
    cust = pd.DataFrame({"orders": g.size(), "first": g["first"].first()})
    late = o[o.d - o["first"] > 273].groupby("c").size()
    cust["active_last_quarter"] = late.reindex(cust.index, fill_value=0) > 0
    cust["tier"] = np.select(
        [
            cust.orders == 1,
            cust.orders <= 3,
            (cust.orders >= 6) & cust.active_last_quarter,
            cust.orders >= 4,
        ],
        ["one order", "2-3 orders", "loyal (6+, still buying)", "4+ orders, not loyal"],
        "4+ orders, not loyal",
    )
    cust["loyal"] = cust.tier == "loyal (6+, still buying)"
    return cust


def rate_table(cust: pd.DataFrame, by: pd.Series) -> dict:
    out = {}
    for k, g in cust.groupby(by):
        if len(g) < MIN_CELL:
            continue
        out[str(k)] = {
            "n": int(len(g)),
            "share": r(len(g) / len(cust)),
            "loyal": r(g.loyal.mean()),
            "one_order": r((g.orders == 1).mean()),
        }
    return out


def funnel(o: pd.DataFrame, cust: pd.DataFrame) -> dict:
    out = {}
    gap = o.d - o.groupby("c").d.shift()
    for n in range(1, 8):
        reached = (cust.orders >= n).sum()
        nxt = (cust.orders >= n + 1).sum()
        out[f"{n}->{n + 1}"] = {
            "reached_n": int(reached),
            "p_next": r(nxt / reached),
            "median_days_between": float(gap[o.k == n + 1].median()),
        }
    return out


# -- paths --------------------------------------------------------------------------


def activity(o: pd.DataFrame, cust: pd.DataFrame) -> tuple[pd.Series, dict]:
    """12-bit pattern of months (30-day periods after the first order) with an order."""
    rep = o[o.k > 1]
    month = np.minimum((rep.d - rep["first"] - 1) // 30, 11)  # day 1-30 -> month 0
    bits = (
        pd.Series(np.left_shift(1, month.to_numpy()), index=rep.c.to_numpy())
        .groupby(level=0)
        .agg(lambda s: np.bitwise_or.reduce(s.to_numpy()))
    )
    pat = bits.reindex(cust.index, fill_value=0)
    strings = pat.map(lambda b: "".join("x" if b >> i & 1 else "." for i in range(12)))
    counts = strings.value_counts()
    repeaters = counts.drop("." * 12, errors="ignore")
    top = {}
    for s, n in repeaters.head(15).items():
        top[s] = {"n": int(n), "share_of_repeaters": r(n / repeaters.sum()),
                  "loyal": r(cust.loyal[strings == s].mean())}  # fmt: skip
    cum = repeaters.cumsum() / repeaters.sum()
    summary = {
        "distinct_patterns_among_repeaters": int(len(repeaters)),
        "possible_patterns": 2**12 - 1,
        "patterns_covering_50pct_of_repeaters": int((cum < 0.5).sum() + 1),
        "patterns_covering_80pct_of_repeaters": int((cum < 0.8).sum() + 1),
        "singleton_patterns_share_of_repeaters": r(
            repeaters[repeaters == 1].sum() / repeaters.sum()
        ),
        "top_patterns": top,
    }
    return pat, summary


def families(pat: pd.Series) -> pd.Series:
    """Rule-based families from the 12-month pattern (bit i = month i+1)."""
    p = [int(x) for x in pat.to_numpy()]
    active = np.array([bin(x).count("1") for x in p])
    first_rep = np.array([(x & -x).bit_length() if x else 0 for x in p])  # month of 1st repeat
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


def early_paths(o: pd.DataFrame, cust: pd.DataFrame) -> dict:
    """Steps 1->2 and 2->3: gap, new product group or not, channel of the new order."""
    o = o.copy()
    o["gap"] = o.d - o.groupby("c").d.shift()
    seen = o.groupby("c").bits.transform(lambda s: np.bitwise_or.accumulate(s.to_numpy()))
    before = seen.groupby(o.c).shift().fillna(0).astype(np.int64)
    o["new_group"] = (o.bits & ~before) != 0
    o["step"] = (
        pd.cut(o.gap, [0, 30, 90, 400], labels=["fast", "medium", "slow"]).astype(str)
        + "/"
        + np.where(o.new_group, "new", "same")
        + "/"
        + np.where(o.online >= 0.5, "online", "store")
    )
    o["start"] = np.where(o.online >= 0.5, "online", "store")
    first = o[o.k == 1].set_index("c").start
    s2 = o[o.k == 2].set_index("c").step
    s3 = o[o.k == 3].set_index("c").step
    three = cust[cust.orders >= 3].index
    path = first.reindex(three) + " > " + s2.reindex(three) + " > " + s3.reindex(three)
    counts = path.value_counts()
    loyal3 = cust.loyal.reindex(three)
    two = cust[cust.orders >= 2].index
    path2 = first.reindex(two) + " > " + s2.reindex(two)
    out = {
        "customers_with_3plus_orders": int(len(three)),
        "distinct_paths_3_orders": int(len(counts)),
        "possible_paths_3_orders": 2 * 12 * 12,
        "paths_covering_50pct": int(((counts.cumsum() / counts.sum()) < 0.5).sum() + 1),
        "top_paths_3_orders": {
            p: {"n": int(n), "share": r(n / len(three)), "loyal": r(loyal3[path == p].mean())}
            for p, n in counts.head(12).items()
        },
        "step_2_alone": rate_table(cust.loc[two], path2),
    }
    # each element of the second step on its own (customers with 2+ orders)
    st = s2.reindex(two).str.split("/", expand=True)
    for i, name in enumerate(["gap_to_2nd", "2nd_adds_new_group", "2nd_channel"]):
        out[name] = rate_table(cust.loc[two], st[i])
    return out


def first_order_traits(o: pd.DataFrame, cust: pd.DataFrame) -> dict:
    f = o[o.k == 1].set_index("c").reindex(cust.index)
    n_groups = f.bits.map(lambda b: bin(int(b)).count("1"))
    return {
        "channel": rate_table(
            cust, pd.Series(np.where(f.online >= 0.5, "online", "store"), index=cust.index)
        ),  # fmt: skip
        "product_groups_in_first_order": rate_table(cust, n_groups.clip(upper=4)),
        "items_in_first_order": rate_table(
            cust, pd.cut(f["items"], [0, 1, 3, 6, 1e9], labels=["1", "2-3", "4-6", "7+"])
        ),
    }


def main(tx_dir: str, articles: str) -> dict:
    o = load(tx_dir, articles)
    cust = outcomes(o)
    pat, act = activity(o, cust)
    gap2 = (o[o.k == 2].set_index("c").d - o[o.k == 1].set_index("c").d).reindex(cust.index)
    return {
        "cohort_customers": int(len(cust)),
        "tiers": cust.tier.value_counts(normalize=True).round(3).to_dict(),
        "orders_in_12_months": cust.orders.clip(upper=12)
        .value_counts(normalize=True)
        .sort_index()
        .round(3)
        .to_dict(),  # fmt: skip
        "funnel": funnel(o, cust),
        "by_days_to_2nd_order": rate_table(
            cust,
            pd.cut(
                gap2,
                [0, 14, 30, 60, 120, 365],
                labels=["1-14", "15-30", "31-60", "61-120", "121-365"],
            )
            .astype(object)
            .fillna("no 2nd order"),
        ),
        "activity_paths": act,
        "families": rate_table(cust, families(pat)),
        "early_paths": early_paths(o, cust),
        "first_order": first_order_traits(o, cust),
        "product_groups": o.attrs["groups"],
    }


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2]), indent=1, default=str))
