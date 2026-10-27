"""Loyalty beyond repeat purchase: habit vs response to a message, and loyal vs valuable.

x5 DIR (randomized SMS campaign + purchase history; see x5_uplift.py):
  Habit strength from each client's transactions before the campaign
  (Liu-Thompkins & Tam 2013: habit = repetition in a stable context):
    - share of visits on their most common weekday;
    - share in their most common 3-hour window;
    - share at their most common store;
    - regularity of gaps between visit days (1 / (1 + coefficient of variation)).
  Consistency shares depend on how many visits there are, so each is ranked
  within bands of visit count and the habit index is the mean rank. Clients
  with 5+ visits only. Uplift (treated minus control purchase rate) by habit
  tercile within visit bands.

hm TRANSACTIONS_DIR (the new-customer cohort of paths_to_loyalty.py):
  Reinartz & Kumar's (2002) loyal x valuable grid among repeat customers:
    loyal = 6+ orders and still buying in months 10-12; valuable = 12-month
    spend at or above the repeaters' median. True friends (loyal, valuable),
    barnacles (loyal, low value), butterflies (not loyal, valuable),
    strangers. Value is spend (no margin in the data); a second version
    counts only full-price spend (lines not on sale), since sales carry less
    margin. Promotion habit = share of lines bought on sale.

Usage:
  uv run --no-project --with pandas python -I loyalty_dimensions.py x5 DIR > x5.json
  uv run --no-project --with pandas --with pyarrow python -I loyalty_dimensions.py hm DIR > hm.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_CELL = 50


def r(x: float, n: int = 4) -> float:
    return round(float(x), n)


def effect(g: pd.DataFrame) -> dict:
    t, c = g[g.treated == 1].target, g[g.treated == 0].target
    if min(len(t), len(c)) < MIN_CELL:
        return {"n": int(len(g)), "suppressed": True}
    u = t.mean() - c.mean()
    se = np.sqrt(t.var() / len(t) + c.var() / len(c))
    return {"n": int(len(g)), "control_rate": r(c.mean()), "uplift": r(u),
            "ci95": [r(u - 1.96 * se), r(u + 1.96 * se)]}  # fmt: skip


# -- X5: habit strength and uplift -----------------------------------------------------


def x5(d: str) -> dict:
    d = Path(d)
    train = pd.read_csv(d / "uplift_train.csv.gz").set_index("client_id")
    parts = []
    cols = ["client_id", "transaction_id", "transaction_datetime", "store_id"]
    for chunk in pd.read_csv(d / "purchases.csv.gz", usecols=cols, chunksize=5_000_000):
        chunk = chunk[chunk.client_id.isin(train.index)].drop_duplicates("transaction_id")
        parts.append(chunk)
    tx = pd.concat(parts).drop_duplicates("transaction_id")
    ts = pd.to_datetime(tx.transaction_datetime)
    tx = tx.assign(dow=ts.dt.dayofweek, hour=ts.dt.hour, day=ts.dt.normalize())
    g = tx.groupby("client_id")
    n = g.size()
    keep = n[n >= 5].index
    tx = tx[tx.client_id.isin(keep)]
    g = tx.groupby("client_id")
    f = pd.DataFrame({"visits": g.size()})
    f["dow_share"] = g.dow.agg(lambda s: s.value_counts().iloc[0] / len(s))
    f["store_share"] = g.store_id.agg(lambda s: s.value_counts().iloc[0] / len(s))
    hours = pd.crosstab(tx.client_id, tx.hour).reindex(columns=range(24), fill_value=0)
    h = hours.to_numpy()
    win = h + np.roll(h, 1, 1) + np.roll(h, -1, 1)
    f["hour_share"] = pd.Series(win.max(1) / h.sum(1), index=hours.index)
    days = tx.drop_duplicates(["client_id", "day"]).sort_values(["client_id", "day"])
    gaps = days.groupby("client_id").day.diff().dt.days.dropna()
    cv = gaps.groupby(days.loc[gaps.index, "client_id"]).agg(lambda s: s.std() / s.mean())
    f["regularity"] = 1 / (1 + cv.reindex(f.index).fillna(cv.median()))
    f["band"] = pd.cut(f.visits, [4, 10, 30, 1e9], labels=["5-10", "11-30", "31+"]).astype(str)
    parts_rank = [f.groupby("band")[c].rank(pct=True)
                  for c in ("dow_share", "hour_share", "store_share", "regularity")]  # fmt: skip
    f["habit"] = pd.concat(parts_rank, axis=1).mean(axis=1)
    f["habit_tercile"] = f.groupby("band").habit.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    )
    f = f.join(train[["treatment_flg", "target"]]).rename(columns={"treatment_flg": "treated"})
    out = {
        "clients_with_5plus_visits": int(len(f)),
        "habit_component_correlations": {
            f"{k[0]} ~ {k[1]}": r(v, 3)
            for k, v in f[["dow_share", "hour_share", "store_share", "regularity"]]
            .corr()
            .stack()
            .items()
            if k[0] < k[1]
        },  # fmt: skip
        "overall_by_habit": {k: effect(g) for k, g in f.groupby("habit_tercile")},
        "by_band_and_habit": {
            f"{b} visits / habit {h}": effect(g)
            for (b, h), g in f.groupby(["band", "habit_tercile"])
        },
    }
    # stratified difference: high minus low habit uplift, weighted by band size
    diffs, weights = [], []
    for _, g in f.groupby("band"):
        hi, lo = effect(g[g.habit_tercile == "high"]), effect(g[g.habit_tercile == "low"])
        if "uplift" in hi and "uplift" in lo:
            diffs.append(hi["uplift"] - lo["uplift"])
            weights.append(len(g))
    out["high_minus_low_habit_uplift_within_bands"] = r(np.average(diffs, weights=weights))
    return out


# -- H&M: true friends, barnacles, butterflies, strangers -------------------------------


def hm(tx_dir: str) -> dict:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    start = pd.Timestamp("2018-09-20")
    t = ds.dataset(tx_dir).to_table(
        columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    )
    lines = pd.DataFrame(
        {
            "c": pc.dictionary_encode(t["customer_id"]).combine_chunks().indices.to_numpy(),
            "d": (pd.to_datetime(t["t_dat"].to_pandas()) - start).dt.days.to_numpy(np.int16),
            "a": t["article_id"].to_numpy(),
            "price": t["price"].to_numpy().astype(np.float32),
            "online": t["sales_channel_id"].to_numpy() == 2,
        }
    )
    del t
    median = lines.groupby("a").price.median()
    lines["sale"] = lines.price < 0.9 * median.reindex(lines.a).to_numpy()
    first = lines.groupby("c").d.min()
    lines["rel"] = lines.d - first.reindex(lines.c).to_numpy()
    cohort = first[first.between(181, 272)].index
    x = lines[lines.c.isin(cohort) & (lines.rel <= 365)]
    orders = x.groupby(["c", "d"]).agg(rel=("rel", "first"), online=("online", "mean"))
    orders = orders.reset_index()
    g = orders.groupby("c")
    cust = pd.DataFrame({"orders": g.size()})
    late = orders[orders.rel > 273].groupby("c").size().reindex(cust.index, fill_value=0)
    cust["loyal"] = (cust.orders >= 6) & (late > 0)
    lg = x.groupby("c")
    cust["spend"] = lg.price.sum()
    cust["full_price_spend"] = x[~x.sale].groupby("c").price.sum().reindex(cust.index).fillna(0)
    cust["sale_share"] = lg.sale.mean()
    cust["channel_consistency"] = g.online.agg(lambda s: max((s >= 0.5).mean(), (s < 0.5).mean()))
    rep = cust[cust.orders >= 2].copy()

    def grid(value: str) -> pd.Series:
        hi = rep[value] >= rep[value].median()
        return pd.Series(
            np.select([rep.loyal & hi, rep.loyal & ~hi, ~rep.loyal & hi],
                      ["true friends", "barnacles", "butterflies"], "strangers"),
            index=rep.index,
        )  # fmt: skip

    def describe(labels: pd.Series) -> dict:
        out = {}
        for k, s in rep.groupby(labels):
            out[k] = {
                "share_of_repeaters": r(len(s) / len(rep), 3),
                "mean_orders": r(s.orders.mean(), 2),
                "mean_spend_index": r(s.spend.mean() / rep.spend.mean(), 2),
                "sale_share_of_lines": r(s.sale_share.mean(), 3),
                "channel_consistency": r(s.channel_consistency.mean(), 3),
            }
        return out

    by_spend, by_full = grid("spend"), grid("full_price_spend")
    loyal = rep[rep.loyal]
    moved = (by_spend[loyal.index] == "true friends") & (by_full[loyal.index] == "barnacles")
    sale_q = pd.qcut(rep.sale_share.rank(method="first"), 4, labels=["Q1 least", "Q2", "Q3",
                                                                    "Q4 most"])  # fmt: skip
    return {
        "cohort_customers": int(len(cust)),
        "repeaters": int(len(rep)),
        "loyal_share_of_repeaters": r(rep.loyal.mean(), 3),
        "grid_by_spend": describe(by_spend),
        "grid_by_full_price_spend": describe(by_full),
        "loyal_true_friends_by_spend_who_are_barnacles_by_full_price": r(
            moved.sum() / (by_spend[loyal.index] == "true friends").sum(), 3
        ),
        "loyal_and_value_by_sale_share_quartile": {
            k: {
                "n": int(len(s)),
                "loyal": r(s.loyal.mean(), 3),
                "spend_index": r(s.spend.mean() / rep.spend.mean(), 2),
                "full_price_spend_index": r(
                    s.full_price_spend.mean() / rep.full_price_spend.mean(), 2
                ),
            }
            for k, s in rep.groupby(sale_q, observed=True)
        },  # fmt: skip
    }


if __name__ == "__main__":
    mode, arg = sys.argv[1], sys.argv[2]
    print(json.dumps({"x5": x5, "hm": hm}[mode](arg), indent=1, default=str))
