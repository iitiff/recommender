"""Who does a message persuade? Uplift of a randomized SMS campaign by shopper profile.

X5 RetailHero (grocery chain): purchase histories before an SMS campaign, and
for 200k clients whether they were randomly sent the SMS (treatment_flg) and
whether they bought during the campaign (target).

1. Randomization check: standardized mean differences of pre-campaign
   features between treated and control.
2. Average effect: purchase rate treated minus control, with a 95% CI.
3. Uplift by segment (profile layers from docs/SHOPPER-PROFILE.md, all from
   purchase history; age and gender are not used):
   rhythm state (days since last visit / own median gap), visits, recency,
   tenure, basket size, points redemption, category breadth, price level.
4. Model: T-learner (one gradient-boosted classifier for treated, one for
   control), cross-fitted over 5 folds so every client is scored by models
   that never saw them. Observed uplift by predicted-uplift decile, the
   share of all incremental purchases captured by targeting the top k%, and
   uplift by predicted propensity (the control model) to test whether the
   most likely buyers are the most persuadable; plus the share captured by
   the simple rule "message the least likely buyers first".

Usage:
  uv run --no-project --with pandas --with scikit-learn python -I \
    x5_uplift.py DIR > out.json      # DIR holds the *.csv.gz files
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 5_000_000
MIN_CELL = 50  # never report a cell with fewer clients


def r(x: float, n: int = 4) -> float:
    return round(float(x), n)


# -- features from purchase history ------------------------------------------------


def load(d: Path) -> pd.DataFrame:
    train = pd.read_csv(d / "uplift_train.csv.gz")
    ids = pd.Index(train.client_id)
    clients = pd.read_csv(
        d / "clients.csv.gz",
        usecols=["client_id", "first_issue_date", "first_redeem_date"],
        parse_dates=["first_issue_date", "first_redeem_date"],
    ).set_index("client_id")
    products = pd.read_csv(d / "products.csv.gz", usecols=["product_id", "level_2", "level_4"])
    products = products.set_index("product_id")

    txns, cats, prices = [], [], []
    cols = [
        "client_id", "transaction_id", "transaction_datetime", "purchase_sum",
        "regular_points_spent", "product_id", "product_quantity", "trn_sum_from_iss",
    ]  # fmt: skip
    for chunk in pd.read_csv(d / "purchases.csv.gz", usecols=cols, chunksize=CHUNK):
        chunk = chunk[chunk.client_id.isin(ids)]
        c = ids.get_indexer(chunk.client_id)
        t = chunk.drop_duplicates("transaction_id")
        txns.append(
            pd.DataFrame(
                {
                    "c": ids.get_indexer(t.client_id),
                    "txn": t.transaction_id.to_numpy(),
                    "t": pd.to_datetime(t.transaction_datetime).to_numpy(),
                    "basket": t.purchase_sum.to_numpy(np.float32),
                    "points_spent": (t.regular_points_spent < 0).to_numpy(),
                }
            )
        )
        lv = products.reindex(chunk.product_id)
        cats.append(
            pd.DataFrame({"c": c, "l2": lv.level_2.to_numpy()}).drop_duplicates()
        )  # fmt: skip
        p = pd.DataFrame(
            {
                "c": c,
                "l4": lv.level_4.to_numpy(),
                "spend": chunk.trn_sum_from_iss.to_numpy(np.float32),
                "qty": chunk.product_quantity.to_numpy(np.float32),
            }
        )
        prices.append(p.groupby(["c", "l4"], sort=False)[["spend", "qty"]].sum().reset_index())
        print(f"  read {len(chunk):,} lines", file=sys.stderr)

    tx = pd.concat(txns).drop_duplicates("txn").sort_values(["c", "t"])
    end = tx.t.max()
    f = pd.DataFrame(index=pd.RangeIndex(len(ids)))
    g = tx.groupby("c")
    f["visits"] = g.size()
    f["recency_days"] = (end - g.t.max()).dt.total_seconds() / 86400
    gaps = tx.t.diff().dt.total_seconds() / 86400
    gaps[tx.c.ne(tx.c.shift())] = np.nan
    f["median_gap_days"] = gaps.groupby(tx.c).median()
    f["basket"] = g.basket.mean()
    f["points_spent_share"] = g.points_spent.mean()
    cl = clients.reindex(ids)
    f["tenure_days"] = ((end - cl.first_issue_date).dt.total_seconds() / 86400).to_numpy()
    f["redeemed_ever"] = cl.first_redeem_date.notna().to_numpy() & (
        cl.first_redeem_date.to_numpy() <= np.datetime64(end)
    )
    f["categories"] = pd.concat(cats).drop_duplicates().groupby("c").size()
    # price level: spend-weighted log ratio of the client's unit price to the
    # median client unit price in the same level-4 category (rough: pack sizes vary)
    pr = pd.concat(prices).groupby(["c", "l4"])[["spend", "qty"]].sum().reset_index()
    pr = pr[(pr.qty > 0) & (pr.spend > 0)]
    pr["unit"] = pr.spend / pr.qty
    pr["ratio"] = np.log(pr.unit / pr.groupby("l4").unit.transform("median"))
    w = pr.spend * pr.ratio
    f["price_level"] = w.groupby(pr.c).sum() / pr.spend.groupby(pr.c).sum()
    f["treated"] = train.treatment_flg.to_numpy()
    f["target"] = train.target.to_numpy()
    f.attrs["history"] = [str(tx.t.min()), str(end)]
    return f


# -- analysis -------------------------------------------------------------------------


def effect(x: pd.DataFrame) -> dict:
    t, c = x[x.treated == 1].target, x[x.treated == 0].target
    if min(len(t), len(c)) < MIN_CELL:
        return {"n": int(len(x)), "suppressed": True}
    u = t.mean() - c.mean()
    se = np.sqrt(t.var() / len(t) + c.var() / len(c))
    return {
        "n": int(len(x)),
        "control_rate": r(c.mean()),
        "treated_rate": r(t.mean()),
        "uplift": r(u),
        "ci95": [r(u - 1.96 * se), r(u + 1.96 * se)],
    }


def segments(f: pd.DataFrame) -> dict[str, pd.Series]:
    ratio = f.recency_days / f.median_gap_days.clip(lower=1)
    rhythm = pd.Series(
        np.select(
            [f.visits <= 1, ratio < 0.75, ratio < 1.25, ratio < 2],
            ["1 visit only", "within cycle (<0.75x gap)", "due (0.75-1.25x)", "overdue (1.25-2x)"],
            "lapsing (>2x)",
        ),
        index=f.index,
    )
    q = lambda s, k=5: pd.qcut(s.rank(method="first"), k, labels=[f"Q{i + 1}" for i in range(k)])  # noqa: E731
    return {
        "rhythm": rhythm,
        "visits": pd.cut(f.visits, [0, 3, 10, 30, 1e9], labels=["1-3", "4-10", "11-30", "31+"]),
        "recency": pd.cut(
            f.recency_days,
            [-1, 3, 7, 14, 30, 1e9],
            labels=["0-3d", "4-7d", "8-14d", "15-30d", "31d+"],
        ),
        "tenure": pd.cut(
            f.tenure_days, [-1e9, 180, 365, 730, 1e9], labels=["<6 mo", "6-12 mo", "1-2 y", "2 y+"]
        ),
        "basket_quintile": q(f.basket),
        "points": pd.cut(
            f.points_spent_share,
            [-1, 0, 0.25, 1],
            labels=["never spends points", "some", "often (>25%)"],
        ),
        "categories_quintile": q(f.categories.fillna(0)),
        "price_level_quintile": q(f.price_level.fillna(0)),
    }


def balance(f: pd.DataFrame, cols: list[str]) -> dict:
    t, c = f[f.treated == 1], f[f.treated == 0]
    out = {}
    for col in cols:
        a, b = t[col].astype(float), c[col].astype(float)
        sd = np.sqrt((a.var() + b.var()) / 2)
        out[col] = r((a.mean() - b.mean()) / sd if sd > 0 else 0, 3)
    return out


def t_learner(f: pd.DataFrame, cols: list[str], folds: int = 5) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import KFold

    x = f[cols].astype(float).to_numpy()
    y, w = f.target.to_numpy(), f.treated.to_numpy()
    p1, p0 = np.zeros(len(f)), np.zeros(len(f))
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(x):
        for arm, out in ((1, p1), (0, p0)):
            m = tr[w[tr] == arm]
            model = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, min_samples_leaf=200, random_state=0
            )
            model.fit(x[m], y[m])
            out[te] = model.predict_proba(x[te])[:, 1]
    return pd.DataFrame({"p_treated": p1, "p_control": p0, "uplift_hat": p1 - p0}, index=f.index)


def targeting(f: pd.DataFrame, score: pd.Series) -> dict:
    """Observed uplift per decile of a score, and incremental purchases captured
    by messaging the top k% (vs everyone), estimated within each slice."""
    order = score.rank(method="first", ascending=False)
    dec = np.ceil(order / len(f) * 10).astype(int)
    by_decile = {f"D{d}": effect(f[dec == d]) for d in range(1, 11)}
    total = effect(f)["uplift"] * len(f)
    captured = {}
    for k in (10, 20, 30, 50, 70):
        top = f[order <= len(f) * k / 100]
        captured[f"top_{k}pct"] = r(effect(top)["uplift"] * len(top) / total, 3)
    return {"by_decile": by_decile, "share_of_incremental_purchases": captured}


def main(d: Path) -> dict:
    f = load(d)
    cols = ["visits", "recency_days", "median_gap_days", "basket", "points_spent_share",
            "tenure_days", "redeemed_ever", "categories", "price_level"]  # fmt: skip
    out = {
        "clients": int(len(f)),
        "history": f.attrs["history"],
        "treated_share": r(f.treated.mean()),
        "balance_smd": balance(f.fillna(0), cols),
        "average_effect": effect(f),
        "segments": {},
    }
    for name, seg in segments(f).items():
        out["segments"][name] = {
            str(k): effect(f[seg == k]) for k in pd.unique(seg.dropna()) if (seg == k).sum()
        }
    scores = t_learner(f, cols)
    out["model"] = {
        "by_predicted_uplift": targeting(f, scores.uplift_hat),
        "by_predicted_propensity": targeting(f, scores.p_control)["by_decile"],
        # the simple rule: skip the likeliest buyers, message the rest
        "by_lowest_propensity_first": targeting(f, -scores.p_control)[
            "share_of_incremental_purchases"
        ],
    }
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1))
