"""Does the first experience predict coming back? Delivery, rating, returns.

olist DIR (marketplace, Brazil; orders, items, payments, reviews, customers):
  Cohort: customers whose first order was placed 2017-01-01..2018-02-28.
  Outcome: another order (on a later day) within 180 days of the first.
  First-order experience:
    - fulfillment: delivered or not (canceled/unavailable/never delivered);
      days delivered vs the promised date (estimated_delivery_date);
      days from purchase to delivery;
    - the review score the customer gave the first order;
    - price paid: order value, freight share of the total, installments,
      payment type.
  One-way rates, then a logistic regression with all of them plus month and
  the customer's state as adjustment controls (state is only a control:
  delivery times depend on distance; it is not reported as a segment).

onlineretail FILE (UK gift wholesaler, B2B; cancellations are invoices "C..."):
  Cohort: first order 2010-03-01..2010-11-30 after 3+ months without one,
  followed 365 days. Exposure: cancellations/returns in the first 90 days
  (none / some), as a share of the value bought then. Outcome: orders in days
  91-365 and loyal (6+ orders in 12 months, still buying in months 10-12),
  within strata of how many orders they placed in the first 90 days (more
  orders give more chances to return something).

Usage:
  uv run --no-project --with pandas --with statsmodels python -I \
    experience_factors.py olist DIR > olist.json
  python -I experience_factors.py onlineretail online_retail_II.csv > or.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_CELL = 50
# Adjustment control only (delivery time depends on distance from the mostly
# Sao Paulo sellers): SP, the rest of the Southeast, and the other regions.
REGION = {"SP": "SP"} | {s: "Southeast (not SP)" for s in ("RJ", "MG", "ES")} | {
    s: "South" for s in ("PR", "RS", "SC")} | {s: "Center-West" for s in ("DF", "GO", "MT", "MS")
    } | {s: "Northeast" for s in ("AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE")} | {
    s: "North" for s in ("AC", "AP", "AM", "PA", "RO", "RR", "TO")}  # fmt: skip


def r(x: float, n: int = 3) -> float:
    return round(float(x), n)


def rates(y: pd.Series, by: pd.Series) -> dict:
    out = {}
    for k, g in y.groupby(by, observed=True):
        if len(g) >= MIN_CELL:
            out[str(k)] = {"n": int(len(g)), "share": r(len(g) / len(y)), "rate": r(g.mean(), 4)}
    return out


# -- Olist --------------------------------------------------------------------------


def olist(d: str) -> dict:
    d = Path(d)
    ts = ["order_purchase_timestamp", "order_delivered_customer_date",
          "order_estimated_delivery_date"]  # fmt: skip
    orders = pd.read_csv(d / "olist_orders_dataset.csv", parse_dates=ts)
    cust = pd.read_csv(d / "olist_customers_dataset.csv",
                       usecols=["customer_id", "customer_unique_id", "customer_state"])  # fmt: skip
    o = orders.merge(cust, on="customer_id")
    o["day"] = o.order_purchase_timestamp.dt.floor("D")
    first_day = o.groupby("customer_unique_id").day.transform("min")
    first = o[o.day == first_day].sort_values("order_purchase_timestamp")
    first = first.drop_duplicates("customer_unique_id").set_index("order_id")
    later = o[o.day > first_day].groupby("customer_unique_id").day.min()

    start, end = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-03-01")
    first = first[(first.day >= start) & (first.day < end)]
    nxt = later.reindex(first.customer_unique_id).to_numpy()
    y = pd.Series(
        ((nxt - first.day.to_numpy()) / np.timedelta64(1, "D") <= 180), index=first.index
    ).astype(int)

    items = pd.read_csv(d / "olist_order_items_dataset.csv",
                        usecols=["order_id", "price", "freight_value"])  # fmt: skip
    it = items.groupby("order_id").agg(price=("price", "sum"), freight=("freight_value", "sum"),
                                       n_items=("price", "size"))  # fmt: skip
    pay = pd.read_csv(d / "olist_order_payments_dataset.csv")
    pay_main = pay.sort_values("payment_value").drop_duplicates("order_id", keep="last")
    pay_main = pay_main.set_index("order_id")
    inst = pay.groupby("order_id").payment_installments.max()
    rev = pd.read_csv(d / "olist_order_reviews_dataset.csv",
                      usecols=["order_id", "review_score", "review_creation_date"])  # fmt: skip
    rev = rev.sort_values("review_creation_date").drop_duplicates("order_id", keep="last")
    score = rev.set_index("order_id").review_score

    f = pd.DataFrame(index=first.index)
    delivered = first.order_status.eq("delivered") & first.order_delivered_customer_date.notna()
    f["fulfilled"] = np.where(delivered, "delivered", "not delivered (" + first.order_status + ")")
    late_days = (
        first.order_delivered_customer_date.dt.floor("D") - first.order_estimated_delivery_date
    ).dt.days
    f["vs_promise"] = pd.cut(
        late_days,
        [-1000, -15, -8, -1, 0, 3, 7, 1000],
        labels=["15+ days early", "8-14 early", "1-7 early", "on the day", "1-3 late", "4-7 late",
                "8+ late"],
    ).astype(object).where(delivered)  # fmt: skip
    lead = (first.order_delivered_customer_date - first.order_purchase_timestamp).dt.days
    days = ["0-7", "8-14", "15-21", "22-30", "31+"]
    f["delivery_days"] = pd.cut(lead, [-1, 7, 14, 21, 30, 1000], labels=days).astype(object)
    f["review"] = score.reindex(f.index).fillna(0).astype(int).astype(str).replace("0", "none")
    it = it.reindex(f.index)
    total = it.price + it.freight
    f["value_q"] = pd.qcut(total.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    f["freight_share_q"] = pd.qcut((it.freight / total).rank(method="first"), 5,
                                   labels=["Q1", "Q2", "Q3", "Q4", "Q5"])  # fmt: skip
    f["installments"] = pd.cut(inst.reindex(f.index), [-1, 1, 4, 9, 100],
                               labels=["1", "2-4", "5-9", "10+"]).astype(object)  # fmt: skip
    f["payment"] = pay_main.payment_type.reindex(f.index)
    f["items"] = pd.cut(it.n_items, [0, 1, 1000], labels=["1", "2+"]).astype(object)

    out = {
        "cohort_customers": int(len(f)),
        "repeat_within_180d": r(y.mean(), 4),
        "one_way": {
            c: rates(y, f[c])
            for c in [
                "fulfilled",
                "vs_promise",
                "delivery_days",
                "review",
                "value_q",
                "freight_share_q",
                "installments",
                "payment",
                "items",
            ]
        },  # fmt: skip
    }
    # review score by delivery vs promise (the mechanism: late -> low score)
    dv = f[delivered]
    out["review_by_vs_promise"] = {
        str(k): {
            "n": int(len(g)),
            "mean_score": r(pd.to_numeric(g.review, errors="coerce").mean(), 2),
            "share_1_or_2": r(g.review.isin(["1", "2"]).mean()),
        }
        for k, g in dv.groupby("vs_promise")
        if len(g) >= MIN_CELL
    }
    out["logit"] = logit(y[delivered], f[delivered], first[delivered])
    return out


def logit(y: pd.Series, f: pd.DataFrame, first: pd.DataFrame) -> dict:
    """Odds ratios (95% CI) for repeat purchase, with and without the review score."""
    import statsmodels.formula.api as smf

    x = f.copy()
    x["y"] = y
    x["region"] = first.customer_state.map(REGION).fillna("other")
    x["delivery_days"] = f.delivery_days
    x["month"] = first.day.dt.to_period("M").astype(str)
    x = x.dropna(subset=["vs_promise", "installments", "payment"])
    base = ("y ~ C(vs_promise, Treatment('1-7 early')) + C(value_q) + C(freight_share_q)"
            " + C(installments) + C(payment) + C(items) + C(region) + C(month)")  # fmt: skip
    out = {}
    for name, formula in (
        ("without_review", base),
        ("with_review", base + " + C(review, Treatment('5'))"),
        ("with_delivery_days", base + " + C(delivery_days, Treatment('8-14'))"),
    ):
        m = smf.logit(formula, data=x).fit(disp=False, maxiter=200)
        ci = m.conf_int()
        keep = [
            k for k in m.params.index if k.startswith(("C(vs_promise", "C(review", "C(delivery"))
        ]
        out[name] = {
            k.split("[T.")[-1].rstrip("]") + (" (review)" if "review" in k else ""): {
                "odds_ratio": r(np.exp(m.params[k]), 2),
                "converged": bool(m.mle_retvals["converged"]),
                "ci95": [r(np.exp(ci.loc[k, 0]), 2), r(np.exp(ci.loc[k, 1]), 2)],
            }
            for k in keep
        }
        out[name]["n"] = int(m.nobs)
    return out


# -- Online Retail II: returns ----------------------------------------------------------


def onlineretail(path: str) -> dict:
    start = pd.Timestamp("2009-12-01")
    x = pd.read_csv(path, usecols=["Invoice", "Quantity", "InvoiceDate", "Price", "Customer ID"],
                    dtype={"Invoice": str})  # fmt: skip
    x = x[x["Customer ID"].notna()].copy()
    x["c"] = x["Customer ID"].astype(int)
    x["d"] = (pd.to_datetime(x.InvoiceDate) - start).dt.days
    x["value"] = x.Quantity * x.Price
    x["is_return"] = x.Invoice.str.startswith("C") | (x.Quantity < 0)
    buys = x[~x.is_return]
    orders = buys.groupby(["c", "d"]).value.sum().reset_index()
    first = orders.groupby("c").d.min()
    lo = (pd.Timestamp("2010-03-01") - start).days
    hi = (pd.Timestamp("2010-12-01") - start).days
    cohort = first[(first >= lo) & (first < hi)]
    o = orders[orders.c.isin(cohort.index)].copy()
    o["rel"] = o.d - o.c.map(cohort)
    o = o[o.rel <= 365]
    cust = pd.DataFrame(index=cohort.index)
    cust["orders_90d"] = o[o.rel <= 90].groupby("c").size()
    cust["orders_after_90d"] = o[o.rel > 90].groupby("c").size().reindex(cust.index, fill_value=0)
    n12 = o.groupby("c").size()
    late = o[o.rel > 273].groupby("c").size().reindex(cust.index, fill_value=0)
    cust["loyal"] = ((n12 >= 6) & (late > 0)).astype(int)
    cust["any_after_90d"] = (cust.orders_after_90d > 0).astype(int)
    bought90 = o[o.rel <= 90].groupby("c").value.sum()
    ret = x[x.is_return & x.c.isin(cohort.index)].copy()
    ret["rel"] = ret.d - ret.c.map(cohort)
    returned90 = ret[(ret.rel >= 0) & (ret.rel <= 90)].groupby("c").value.sum().abs()
    share = (returned90.reindex(cust.index, fill_value=0) / bought90.reindex(cust.index)).fillna(0)
    labels = ["none", "under 5% of value", "5%+"]
    cust["returns_90d"] = pd.cut(share, [-1, 0, 0.05, 1e9], labels=labels).astype(str)
    cust["orders_90d_band"] = pd.cut(cust.orders_90d, [0, 1, 3, 1000],
                                     labels=["1", "2-3", "4+"]).astype(str)  # fmt: skip
    out = {"cohort_customers": int(len(cust)),
           "any_return_90d": r((cust.returns_90d != "none").mean())}  # fmt: skip
    for outcome in ("any_after_90d", "loyal"):
        out[outcome] = {
            "all": rates(cust[outcome], cust.returns_90d),
            "by_orders_in_first_90d": {
                band: rates(g[outcome], g.returns_90d)
                for band, g in cust.groupby("orders_90d_band")
            },
        }
    return out


if __name__ == "__main__":
    mode, arg = sys.argv[1], sys.argv[2]
    result = {"olist": olist, "onlineretail": onlineretail}[mode](arg)
    print(json.dumps(result, indent=1, default=str))
