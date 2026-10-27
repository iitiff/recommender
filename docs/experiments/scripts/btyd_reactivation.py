"""Is a customer still "alive"? Buy-till-you-die models vs a fixed recency rule.

Fits BG/NBD and MBG/NBD (pymc-marketing, MAP) on a calibration period and
checks them on a holdout period:
  - total and per-frequency repeat purchases, predicted vs actual;
  - per-customer error vs two naive forecasts (own past rate; cohort mean);
  - ranking: who buys again in the holdout, by P(alive) vs days since the
    last purchase vs a fixed "lapsed after 90 days" rule (AUC);
  - when P(alive) falls below 50% for customers with different histories,
    i.e. a per-customer reactivation moment instead of one cutoff.

BG/NBD lets customers drop out only right after a repeat purchase, so a
one-order customer is always "alive". MBG/NBD lets them drop out at the
first purchase too, which matters where most customers buy once.

Time unit: weeks. A purchase is a customer-day.
  cdnow CDNOW_master.txt: the 23,570 first-ever buyers of Q1 1997; calibration
    to 1997-09-30 (39 weeks after the start), holdout to 1998-06-30. This is the
    dataset of Fader, Hardie & Lee (2005), "Counting your customers the easy way".
  hm TRANSACTIONS_DIR: new customers of 2019-03-20..06-19 (as in
    paths_to_loyalty.py); calibration to 2019-10-31, holdout to 2020-03-14
    (before store closures).

Usage:
  uv run --no-project --with pymc-marketing --with pyarrow --with scikit-learn python -I \
    btyd_reactivation.py {cdnow FILE | hm DIR} > out.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

WEEK = 7


def r(x: float, n: int = 3) -> float:
    return round(float(x), n)


def load_cdnow(path: str) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    x = pd.read_csv(path, sep=r"\s+", header=None, names=["c", "date", "n", "v"],
                    dtype={"date": str})  # fmt: skip
    x["day"] = pd.to_datetime(x.date, format="%Y%m%d")
    return x[["c", "day"]].drop_duplicates(), pd.Timestamp("1997-09-30"), pd.Timestamp("1998-06-30")


def load_hm(tx_dir: str) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    t = ds.dataset(tx_dir).to_table(columns=["t_dat", "customer_id"])
    x = pd.DataFrame(
        {
            "c": pc.dictionary_encode(t["customer_id"]).combine_chunks().indices.to_numpy(),
            "day": pd.to_datetime(t["t_dat"].to_pandas()),
        }
    ).drop_duplicates()
    first = x.groupby("c").day.transform("min")
    x = x[(first >= "2019-03-20") & (first < "2019-06-20")]
    return x, pd.Timestamp("2019-10-31"), pd.Timestamp("2020-03-14")


def rfm(x: pd.DataFrame, cal_end: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cal = x[x.day <= cal_end]
    g = cal.groupby("c").day
    first, last = g.min(), g.max()
    out = pd.DataFrame(
        {
            "customer_id": first.index,
            "frequency": (g.nunique() - 1).to_numpy(),
            "recency": ((last - first).dt.days / WEEK).to_numpy(),
            "T": ((cal_end - first).dt.days / WEEK).to_numpy(),
        }
    )
    hold = x[(x.day > cal_end) & (x.day <= end)].groupby("c").day.nunique()
    out["holdout"] = hold.reindex(first.index, fill_value=0).to_numpy()
    out["days_since_last"] = (cal_end - last).dt.days.to_numpy()
    return out


def fit(model_cls, data: pd.DataFrame):  # type: ignore[no-untyped-def]
    m = model_cls()
    m.fit(data[["customer_id", "frequency", "recency", "T"]], method="map", progressbar=False)
    return m


def mean(da) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Point estimate from a MAP fit's (chain, draw, customer) output."""
    dims = [d for d in da.dims if d in ("chain", "draw")]
    return np.asarray(da.mean(dims))


def params(m) -> dict:  # type: ignore[no-untyped-def]
    post = m.fit_result
    return {k: r(float(post[k].mean()), 3) for k in ("r", "alpha", "a", "b") if k in post.data_vars}


def analyse(data: pd.DataFrame, holdout_weeks: float) -> dict:
    from pymc_marketing.clv import BetaGeoModel, ModifiedBetaGeoModel
    from sklearn.metrics import roc_auc_score

    rows = data[["customer_id", "frequency", "recency", "T"]]
    out = {"customers": int(len(data)), "holdout_weeks": r(holdout_weeks, 1)}
    bought = (data.holdout > 0).astype(int)
    naive_rate = data.frequency / data["T"].clip(lower=1) * holdout_weeks
    out["naive_mae"] = {
        "own_past_rate": r((naive_rate - data.holdout).abs().mean()),
        "cohort_mean": r((data.holdout.mean() - data.holdout).abs().mean()),
    }
    out["ranking_auc"] = {
        "days_since_last_purchase": r(roc_auc_score(bought, -data.days_since_last)),
        "lapsed_after_90_days_rule": r(roc_auc_score(bought, (data.days_since_last <= 90))),
    }
    for name, cls in (("bg_nbd", BetaGeoModel), ("mbg_nbd", ModifiedBetaGeoModel)):
        m = fit(cls, data)
        pred = mean(m.expected_purchases(rows, future_t=holdout_weeks))
        alive = mean(m.expected_probability_alive(rows))
        by_freq = {}
        for k, g in data.assign(pred=pred).groupby(data.frequency.clip(upper=7)):
            by_freq[str(k) + ("+" if k == 7 else "")] = {
                "n": int(len(g)), "actual": r(g.holdout.mean()), "predicted": r(g.pred.mean())
            }  # fmt: skip
        out[name] = {
            "params": params(m),
            "holdout_total_actual": int(data.holdout.sum()),
            "holdout_total_predicted": r(pred.sum(), 0),
            "mae": r(np.abs(pred - data.holdout).mean()),
            "by_calibration_frequency": by_freq,
            "auc_p_alive": r(roc_auc_score(bought, alive)),
            "auc_expected_purchases": r(roc_auc_score(bought, pred)),
            "p_alive_calibration": {
                str(k): {"n": int(len(g)), "bought_in_holdout": r(g.bought.mean())}
                for k, g in pd.DataFrame({"p": alive, "bought": bought}).groupby(
                    pd.cut(alive, [0, 0.25, 0.5, 0.75, 0.9, 1.0001]), observed=True
                )
            },
            "weeks_of_silence_until_p_alive_below_50pct": silence(m, data),
        }
    return out


def silence(m, data: pd.DataFrame) -> dict:  # type: ignore[no-untyped-def]
    """Weeks after the last purchase until P(alive) drops below 0.5, for
    customers with x repeat purchases spaced `gap` weeks apart (x = 0: weeks
    after the first and only purchase)."""

    def weeks_until(x: int, t_x: float) -> int | str:
        weeks = np.arange(0, 157)
        grid = pd.DataFrame({"customer_id": np.arange(len(weeks)), "frequency": x,
                             "recency": t_x, "T": t_x + weeks})  # fmt: skip
        below = np.flatnonzero(mean(m.expected_probability_alive(grid)) < 0.5)
        return int(weeks[below[0]]) if len(below) else ">156"

    out: dict = {"one_order_customers": weeks_until(0, 0.0)}
    for gap in (2, 4, 8, 16):
        out[f"gap_{gap}_weeks"] = {f"{x}_repeats": weeks_until(x, x * gap) for x in (1, 2, 4, 8)}
    return out


if __name__ == "__main__":
    mode, arg = sys.argv[1], sys.argv[2]
    x, cal_end, end = {"cdnow": load_cdnow, "hm": load_hm}[mode](arg)
    data = rfm(x, cal_end, end)
    result = {"dataset": mode, "calibration_end": str(cal_end.date()), "end": str(end.date())}
    result |= analyse(data, (end - cal_end).days / WEEK)
    print(json.dumps(result, indent=1, default=str))
