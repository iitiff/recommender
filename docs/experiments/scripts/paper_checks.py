"""Check published growth and messaging findings against the public datasets.

Each check is the closest test the data allows. Most are observational (the
papers' own designs were randomized or quasi-experimental), so they test
whether the pattern appears, not whether it is causal.

  acquisition   Lewis 2006; Anderson & Simester 2004 (new customers): first-order
                discount depth vs second order and 12-month value.
                H&M markdowns, Olist vouchers.
  established   Anderson & Simester 2004 (established customers): a deep-discount
                quarter vs later frequency, price paid and discount share,
                controlling for each customer's earlier discount share.
                H&M, Complete Journey.
  offers        Sahni, Zou & Chintagunta 2017 (offers work as advertising) and
                wear-out (Hohnhold et al. 2015): campaign difference-in-differences
                in Complete Journey, split into coupon products, redeemed
                coupons and everything else, and by the household's campaign count.
  futility      Ascarza 2018; Lemmens & Gupta 2020: uplift by churn risk, and
                incremental purchases and value at a fixed budget by ranking rule.
                X5 randomized SMS (the only randomized dataset we hold).
  freshstart    Dai, Milkman & Riis 2014: do comeback orders after a lapse
                over-index on Mondays, month starts and January, vs regular orders?
                H&M, CDNOW, Online Retail, Olist, Complete Journey.
  delivery      Fisher, Gallino & Xu 2019: delivery days of the first order vs
                repeat purchase, within state and month. Olist.
  diversity     Anderson et al. 2020 (Spotify): spread of purchases across
                categories at equal activity vs retention. H&M, Complete Journey,
                REES46.

Aggregates only; cells under 50 customers are suppressed.

Usage (see each function for its arguments):
  uv run --no-project --with pandas --with pyarrow --with scikit-learn \
    --with statsmodels --with pyreadr python -I paper_checks.py CHECK DATASET PATHS... > out.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_CELL = 50
SEED = 0


def r(x: float, n: int = 4) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def rate(y: pd.Series) -> dict | None:
    n = int(y.notna().sum())
    if n < MIN_CELL:
        return None
    p = float(y.mean())
    se = np.sqrt(p * (1 - p) / n) if 0 <= p <= 1 else float(y.std() / np.sqrt(n))
    return {"n": n, "mean": r(p), "ci": [r(p - 1.96 * se), r(p + 1.96 * se)]}


def mean_ci(y: pd.Series) -> dict | None:
    y = y.dropna()
    if len(y) < MIN_CELL:
        return None
    se = y.std() / np.sqrt(len(y))
    return {
        "n": int(len(y)),
        "mean": r(y.mean()),
        "ci": [r(y.mean() - 1.96 * se), r(y.mean() + 1.96 * se)],
    }


def ols(y: pd.Series, x: pd.DataFrame, fe: list[pd.Series] | None = None) -> dict:
    """Linear model with dummy fixed effects and heteroskedasticity-robust (HC1) SEs.
    Returns coefficient and 95% CI for every column of x."""
    import statsmodels.api as sm

    parts = [x.astype(float)]
    for f in fe or []:
        parts.append(pd.get_dummies(f.astype(str), prefix=f.name or "fe", drop_first=True,
                                    dtype=float))  # fmt: skip
    X = sm.add_constant(pd.concat(parts, axis=1))
    ok = y.notna() & X.notna().all(axis=1)
    m = sm.OLS(y[ok].astype(float), X[ok]).fit(cov_type="HC1")
    ci = m.conf_int()
    return {
        "n": int(ok.sum()),
        **{c: {"coef": r(m.params[c], 5), "ci": [r(ci.loc[c, 0], 5), r(ci.loc[c, 1], 5)]}
           for c in x.columns},
    }  # fmt: skip


# -- loaders: orders as one row per customer-day ----------------------------------------


def hm_lines(tx_dir: str, articles: str) -> pd.DataFrame:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    t = ds.dataset(tx_dir).to_table(
        columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    )
    c = pc.dictionary_encode(t["customer_id"]).combine_chunks().indices.to_numpy()
    day = (pd.to_datetime(t["t_dat"].to_pandas()) - pd.Timestamp("2018-09-20")).dt.days
    a = pq.read_table(articles, columns=["article_id", "product_group_name"]).to_pandas()
    lines = pd.DataFrame(
        {
            "c": c.astype(np.int32),
            "day": day.to_numpy().astype(np.int16),
            "article": t["article_id"].to_numpy().astype(np.int64),
            "price": t["price"].to_numpy().astype(np.float32),
            "online": t["sales_channel_id"].to_numpy() == 2,
        }
    )
    del t
    group = a.set_index("article_id").product_group_name.astype("category")
    lines["cat"] = group.reindex(lines.article).to_numpy()
    return lines


def cj_tables(d: str) -> dict[str, pd.DataFrame]:
    import pyreadr

    d = Path(d)
    tx = pyreadr.read_r(str(d / "transactions.rds"))[None]
    one = {k: next(iter(pyreadr.read_r(str(d / f"{k}.rda")).values()))
           for k in ["products", "campaigns", "campaign_descriptions", "coupon_redemptions",
                     "coupons"]}  # fmt: skip
    tx["c"] = tx.household_id.astype(str)
    tx["day"] = (pd.to_datetime(tx.transaction_timestamp) - pd.Timestamp("2017-01-01")).dt.days
    tx["week"] = tx.day // 7
    tx["cat"] = tx.product_id.map(one["products"].set_index("product_id").department)
    return {"tx": tx, **one}


def orders_of(name: str, paths: list[str]) -> pd.DataFrame:
    """c, date (Timestamp, day), spend for the fresh-start check."""
    if name == "hm":
        x = hm_lines(*paths)
        o = x.groupby(["c", "day"]).price.sum().rename("spend").reset_index()
        o["date"] = pd.Timestamp("2018-09-20") + pd.to_timedelta(o.day.astype(int), "D")
        return o[["c", "date", "spend"]]
    if name == "cdnow":
        x = pd.read_csv(paths[0], sep=r"\s+", header=None, names=["c", "date", "items", "spend"],
                        dtype={"date": str})  # fmt: skip
        x["date"] = pd.to_datetime(x.date, format="%Y%m%d")
        return x.groupby(["c", "date"]).spend.sum().reset_index()
    if name == "onlineretail":
        x = pd.read_csv(paths[0], usecols=["Invoice", "Quantity", "InvoiceDate", "Price",
                                           "Customer ID"], dtype={"Invoice": str})  # fmt: skip
        x = x[x["Customer ID"].notna() & ~x.Invoice.str.startswith("C") & (x.Quantity > 0)]
        x["c"] = x["Customer ID"].astype(int)
        x["date"] = pd.to_datetime(x.InvoiceDate).dt.floor("D")
        x["spend"] = x.Quantity * x.Price
        return x.groupby(["c", "date"]).spend.sum().reset_index()
    if name == "olist":
        d = Path(paths[0])
        cols = ["customer_id", "order_status", "order_purchase_timestamp"]
        o = pd.read_csv(d / "olist_orders_dataset.csv", usecols=cols)
        cust = pd.read_csv(d / "olist_customers_dataset.csv",
                           usecols=["customer_id", "customer_unique_id"])  # fmt: skip
        o = o[~o.order_status.isin(["canceled", "unavailable"])].merge(cust, on="customer_id")
        o["c"] = o.customer_unique_id
        o["date"] = pd.to_datetime(o.order_purchase_timestamp).dt.floor("D")
        o["spend"] = 1.0
        return o.groupby(["c", "date"]).spend.sum().reset_index()
    if name == "cj":
        tx = cj_tables(paths[0])["tx"]
        tx["date"] = pd.Timestamp("2017-01-01") + pd.to_timedelta(tx.day, "D")
        return tx.groupby(["c", "date"]).sales_value.sum().rename("spend").reset_index()
    raise ValueError(name)


# -- 1. acquisition discount depth (new customers) --------------------------------------


def acquisition_hm(tx_dir: str, articles: str) -> dict:
    lines = hm_lines(tx_dir, articles)
    lo, hi, H = 181, 273, 365
    first = lines.groupby("c").day.min()
    cohort = first[(first >= lo) & (first < hi)].index  # none in the prior 6 months
    ref = lines[lines.day < hi].groupby("article").price.max()  # pre-outcome reference price
    x = lines[lines.c.isin(cohort)].copy()
    x["first"] = first.reindex(x.c).to_numpy()
    x = x[x.day <= x["first"] + H]
    x["depth"] = (1 - x.price / ref.reindex(x.article).to_numpy()).clip(lower=0)
    f1 = x[x.day == x["first"]]
    g1 = f1.groupby("c")
    cust = pd.DataFrame(
        {
            "depth": g1.depth.mean(),
            "spend": g1.price.sum(),
            "items": g1.size(),
            "online": g1.online.mean() >= 0.5,
            "week": (g1.day.first() // 7).astype(int),
        }
    )
    later = x[x.day > x["first"]]
    rel = later.day - later["first"]
    days = later.groupby("c").day.nunique()
    cust["second_90d"] = later[rel <= 90].groupby("c").size().reindex(cust.index).notna()
    cust["orders_12m"] = days.reindex(cust.index, fill_value=0)
    cust["spend_12m"] = later.groupby("c").price.sum().reindex(cust.index, fill_value=0.0)
    late = later[rel > 273].groupby("c").size().reindex(cust.index).notna()
    cust["loyal_12m"] = (cust.orders_12m >= 5) & late  # 6+ orders incl. the first
    cust["later_depth"] = later.groupby("c").depth.mean().reindex(cust.index)
    cust["band"] = pd.cut(cust.depth, [-0.01, 0.05, 0.2, 0.4, 1.0],
                          labels=["full price (<5% off)", "5-20% off", "20-40% off",
                                  "40%+ off"])  # fmt: skip
    out = {"cohort": int(len(cust)), "by_first_order_depth": {}}
    for b, g in cust.groupby("band", observed=True):
        out["by_first_order_depth"][str(b)] = {
            "share": r(len(g) / len(cust)),
            "second_order_90d": rate(g.second_90d.astype(float)),
            "loyal_12m": rate(g.loyal_12m.astype(float)),
            "orders_12m_after_first": mean_ci(g.orders_12m),
            "spend_12m_after_first": mean_ci(g.spend_12m),
            "depth_of_later_purchases": mean_ci(g.later_depth),
        }
    # adjusted: depth (per 10 points off) with first-order week, spend, items and channel
    xq = pd.DataFrame({"depth_per_10pts": cust.depth * 10})
    fe = [cust.week.rename("week"),
          pd.qcut(cust.spend.rank(method="first"), 5, labels=False).rename("spendq"),
          cust["items"].clip(upper=6).rename("items"), cust.online.rename("online")]  # fmt: skip
    out["adjusted_per_10pts_deeper"] = {
        "second_order_90d": ols(cust.second_90d.astype(float), xq, fe),
        "loyal_12m": ols(cust.loyal_12m.astype(float), xq, fe),
        "log1p_spend_12m": ols(np.log1p(cust.spend_12m), xq, fe),
        "later_depth": ols(cust.later_depth, xq, fe),
    }
    out["notes"] = {"cohort": "first order 2019-03-20..06-18, none in the prior 6 months",
                    "depth": "1 - price / article's highest price before 2019-06-19",
                    "controls": "first-order week, spend quintile, items, channel"}  # fmt: skip
    return out


def acquisition_olist(d: str) -> dict:
    d = Path(d)
    o = pd.read_csv(d / "olist_orders_dataset.csv")
    cust = pd.read_csv(d / "olist_customers_dataset.csv",
                       usecols=["customer_id", "customer_unique_id", "customer_state"])  # fmt: skip
    o = o[~o.order_status.isin(["canceled", "unavailable"])].merge(cust, on="customer_id")
    o["day"] = pd.to_datetime(o.order_purchase_timestamp).dt.floor("D")
    first_day = o.groupby("customer_unique_id").day.transform("min")
    first = o[o.day == first_day].drop_duplicates("customer_unique_id").set_index("order_id")
    nxt = o[o.day > first_day].groupby("customer_unique_id").day.min()
    first = first[(first.day >= "2017-01-01") & (first.day < "2018-03-01")]
    gap = nxt.reindex(first.customer_unique_id).to_numpy() - first.day.to_numpy()
    y = pd.Series(gap / np.timedelta64(1, "D") <= 180, index=first.index).astype(float)
    pay = pd.read_csv(d / "olist_order_payments_dataset.csv")
    pay["voucher"] = np.where(pay.payment_type == "voucher", pay.payment_value, 0.0)
    p = pay.groupby("order_id").agg(paid=("payment_value", "sum"), voucher=("voucher", "sum"))
    p = p.reindex(first.index)
    share = (p.voucher / p.paid).fillna(0)
    band = pd.cut(share, [-0.01, 0, 0.25, 0.75, 1.0],
                  labels=["no voucher", "voucher <25%", "25-75%", "75%+ of the order"])  # fmt: skip
    out = {"cohort": int(len(first)), "repeat_180d_by_voucher_share": {
        str(b): rate(y[band == b]) for b in band.cat.categories}}  # fmt: skip
    fe = [first.customer_state.rename("state"), first.day.dt.to_period("M").rename("month"),
          pd.qcut(p.paid.rank(method="first"), 5, labels=False).rename("valueq")]  # fmt: skip
    out["adjusted_any_voucher"] = ols(
        y, pd.DataFrame({"any_voucher": (share > 0).astype(float)}), fe
    )
    out["notes"] = {"voucher": "Olist vouchers are a payment method (gift cards, credits, promos)",
                    "controls": "state, month, order value quintile"}  # fmt: skip
    return out


# -- 2. discount depth for established customers ---------------------------------------


def established(
    base: pd.DataFrame, treat: pd.DataFrame, after: pd.DataFrame, min_base: int
) -> dict:
    """Each frame: c, day, spend, depth (line level), ref (reference price paid level).
    Established = min_base+ orders in the base period and 1+ in the treatment period."""

    def per(x: pd.DataFrame, p: str) -> pd.DataFrame:
        g = x.groupby("c")
        return pd.DataFrame({f"{p}_orders": g.day.nunique(), f"{p}_spend": g.spend.sum(),
                             f"{p}_depth": g.depth.mean(), f"{p}_ref": g.ref.mean()})  # fmt: skip

    b, t, a = per(base, "b"), per(treat, "t"), per(after, "a")
    f = b[b.b_orders >= min_base].join(t, how="inner").join(a, how="left")
    f[["a_orders", "a_spend"]] = f[["a_orders", "a_spend"]].fillna(0)
    q = pd.qcut(
        f.t_depth.rank(method="first"), 4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (deepest)"]
    )
    out = {"customers": int(len(f)), "by_treatment_depth_quartile": {}}
    for k, g in f.groupby(q, observed=True):
        out["by_treatment_depth_quartile"][str(k)] = {
            "depth_range": [r(g.t_depth.min(), 3), r(g.t_depth.max(), 3)],
            "base_depth": r(g.b_depth.mean(), 3),
            "orders_after": mean_ci(g.a_orders),
            "spend_after": mean_ci(g.a_spend),
            "depth_after": mean_ci(g.a_depth),
            "reference_price_after": mean_ci(g.a_ref),
            "active_after": rate((g.a_orders > 0).astype(float)),
        }
    controls = pd.DataFrame({
        "t_depth_per_10pts": f.t_depth * 10, "b_depth_per_10pts": f.b_depth * 10,
        "log_b_orders": np.log(f.b_orders), "log_b_spend": np.log1p(f.b_spend),
        "log_t_orders": np.log(f.t_orders), "log_t_spend": np.log1p(f.t_spend),
    })  # fmt: skip
    out["adjusted"] = {
        k: {"t_depth_per_10pts": ols(y, controls)["t_depth_per_10pts"]}
        for k, y in {
            "log1p_orders_after": np.log1p(f.a_orders),
            "log1p_spend_after": np.log1p(f.a_spend),
            "depth_after": f.a_depth,
            "log_reference_price_after": np.log(f.a_ref),
        }.items()
    }
    return out


def established_hm(tx_dir: str, articles: str) -> dict:
    x = hm_lines(tx_dir, articles)
    ref = x[x.day < 365].groupby("article").price.max()
    x["ref"] = ref.reindex(x.article).to_numpy()
    x["depth"] = (1 - x.price / x.ref).clip(lower=0)
    x["spend"] = x.price
    base, treat, after = (
        x[x.day.between(92, 272)],
        x[x.day.between(273, 364)],
        x[x.day.between(365, 546)],
    )
    out = established(base, treat, after, min_base=4)
    out["notes"] = {"base": "2018-12-21..2019-06-18",
                    "treatment": "2019-06-19..09-17 (summer sale)",
                    "after": "2019-09-18..2020-03-17",
                    "reference": "article's highest price before 2019-09-18"}  # fmt: skip
    return out


def established_cj(d: str) -> dict:
    tx = cj_tables(d)["tx"]
    disc = tx.retail_disc + tx.coupon_disc + tx.coupon_match_disc
    tx["depth"] = (disc / (tx.sales_value + disc)).where(tx.sales_value + disc > 0)
    tx["spend"] = tx.sales_value
    tx["ref"] = (tx.sales_value + disc) / tx.quantity.where(tx.quantity > 0)
    base, treat, after = tx[tx.day < 90], tx[tx.day.between(90, 180)], tx[tx.day > 180]
    out = established(base, treat, after, min_base=4)
    out["notes"] = {"base": "Jan-Mar 2017", "treatment": "Apr-Jun", "after": "Jul-Dec",
                    "depth": "(loyalty + coupon discounts) / shelf value",
                    "reference": "shelf price per unit (mix-dependent)"}  # fmt: skip
    return out


# -- 3. offers as advertising, and wear-out ---------------------------------------------


def offers_cj(d: str, reps: int = 200) -> dict:
    t = cj_tables(d)
    tx, camp, desc, coupons = t["tx"], t["campaigns"], t["campaign_descriptions"], t["coupons"]
    desc["s"] = (pd.to_datetime(desc.start_date) - pd.Timestamp("2017-01-01")).dt.days // 7
    desc["e"] = (pd.to_datetime(desc.end_date) - pd.Timestamp("2017-01-01")).dt.days // 7
    camp["c"] = camp.household_id.astype(str)
    seq = camp.merge(desc[["campaign_id", "s"]], on="campaign_id").sort_values(["c", "s"])
    seq["k"] = seq.groupby("c").cumcount() + 1
    tx["redeemed"] = tx.coupon_disc > 0
    hh = tx.c.unique()
    rng = np.random.default_rng(SEED)
    rows, per_type, by_k = [], {}, {}
    for _, cp in desc.iterrows():
        s, e, cid = int(cp.s), int(cp.e), cp.campaign_id
        if s < 8 or e > 52:
            continue
        prods = set(coupons.loc[coupons.campaign_id == cid, "product_id"])
        win = tx[(tx.week >= s - 8) & (tx.week <= e)]
        post = win.week >= s
        cov = win.product_id.isin(prods)
        parts = {
            "total": win.sales_value,
            "coupon_products": win.sales_value.where(cov, 0.0),
            "other_products": win.sales_value.where(~cov, 0.0),
            "redeemed_lines": win.sales_value.where(win.redeemed, 0.0),
        }
        treated = set(camp.loc[camp.campaign_id == cid, "c"])
        weeks_pre, weeks_post = 8, e - s + 1
        res = {}
        frame = pd.DataFrame(index=pd.Index(hh, name="c"))
        for name, v in parts.items():
            pre = v[~post].groupby(win.c[~post]).sum().reindex(hh, fill_value=0) / weeks_pre
            po = v[post].groupby(win.c[post]).sum().reindex(hh, fill_value=0) / weeks_post
            frame[name] = (po - pre).to_numpy()
        frame["treated"] = frame.index.isin(treated)
        if frame.treated.sum() < MIN_CELL:
            continue
        if s >= 16:  # placebo: the 8 weeks before vs the 8 before those, same split
            pw = tx[(tx.week >= s - 16) & (tx.week < s)]
            late_pre = pw.week >= s - 8
            a1 = pw.sales_value[late_pre].groupby(pw.c[late_pre]).sum().reindex(hh, fill_value=0)
            a0 = pw.sales_value[~late_pre].groupby(pw.c[~late_pre]).sum().reindex(hh, fill_value=0)
            frame["placebo"] = ((a1 - a0) / 8).to_numpy()
        else:
            frame["placebo"] = np.nan
        for name in parts:
            dd = frame.loc[frame.treated, name].mean() - frame.loc[~frame.treated, name].mean()
            res[name] = dd
        res["campaign"], res["type"], res["n_treated"] = (
            int(cid),
            cp.campaign_type,
            int(frame.treated.sum()),
        )
        rows.append(res)
        frame["campaign"] = cid
        per_type.setdefault(cp.campaign_type, []).append(frame)
        kk = seq[seq.campaign_id == cid].set_index("c").k.clip(upper=3)
        frame["k"] = kk.reindex(frame.index)
        by_k.setdefault("all", []).append(frame)

    def pooled(frames: list[pd.DataFrame], key: str | None = None) -> dict:
        allf = pd.concat(frames)

        def est(f: pd.DataFrame) -> dict:
            out = {}
            for name in ["total", "coupon_products", "other_products", "redeemed_lines",
                         "placebo"]:  # fmt: skip
                gs = [g for _, g in f.groupby("campaign") if g[name].notna().any()]
                parts = [g.loc[g.treated, name].mean() - g.loc[~g.treated, name].mean() for g in gs]
                w = [g.treated.sum() for g in gs]
                out[name] = float(np.average(parts, weights=w)) if gs else float("nan")
            return out

        point = est(allf)
        boot = []
        camps = allf.campaign.unique()
        for _ in range(reps):  # resample campaigns (clusters)
            pick = rng.choice(camps, len(camps), replace=True)
            boot.append(est(pd.concat([allf[allf.campaign == c].assign(campaign=i)
                                       for i, c in enumerate(pick)])))  # fmt: skip
        out = {}
        for name, v in point.items():
            b = [x[name] for x in boot]
            out[name] = {"lift_per_week": r(v, 3),
                         "ci": [r(np.percentile(b, 2.5), 3),
                                r(np.percentile(b, 97.5), 3)]}  # fmt: skip
        if point["total"] != 0:
            out["share_of_lift_outside_redeemed_lines"] = r(
                1 - point["redeemed_lines"] / point["total"], 3
            )
            out["share_of_lift_outside_coupon_products"] = r(
                point["other_products"] / point["total"], 3
            )
        out["campaigns"] = int(len(camps))
        out["treated_households"] = int(allf.treated.sum())
        return out

    out = {"per_campaign": [{k: (r(v, 3) if isinstance(v, float) else v) for k, v in x.items()}
                            for x in rows]}  # fmt: skip
    out["pooled"] = pooled([f for fs in per_type.values() for f in fs])
    out["by_type"] = {k: pooled(v) for k, v in per_type.items()}
    # wear-out: the household's 1st, 2nd, 3rd+ campaign of the year (vs the same controls)
    allf = pd.concat(by_k["all"])
    out["by_campaign_count"] = {}
    for k in (1, 2, 3):
        sub = allf[~allf.treated | (allf.k == k)]
        n = int((sub.treated).sum())
        if n >= MIN_CELL:
            res = pooled([g for _, g in sub.groupby("campaign")])
            out["by_campaign_count"][f"{k}{'+' if k == 3 else ''}"] = res
    out["notes"] = {
        "design": "per campaign: change in weekly spend (campaign weeks vs 8 weeks before), "
        "targeted households minus all other households; pooled over campaigns "
        "weighted by targeted households; 95% CI from resampling campaigns",
        "selection": "campaigns were targeted, not randomized; household-level differences "
        "cancel in the before/after change, trends do not",
        "overlap": "campaigns overlap in time, so effects of concurrent campaigns mix",
        "placebo": "same contrast for the 8 weeks before vs the 8 before those (should be ~0)",
    }
    return out


# -- 4. retention futility and profit-based targeting (X5, randomized) ------------------


def futility_x5(d: str) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import x5_uplift as x5  # noqa: E402

    f = x5.load(Path(d))
    cols = ["visits", "recency_days", "median_gap_days", "basket", "points_spent_share",
            "tenure_days", "redeemed_ever", "categories", "price_level"]  # fmt: skip
    s = x5.t_learner(f, cols)
    f["risk"] = 1 - s.p_control  # chance of NOT buying without the message
    f["uplift_hat"] = s.uplift_hat
    f["value"] = f.basket.fillna(0)  # spend per visit before the campaign
    out = {"clients": int(len(f)), "average_effect": x5.effect(f)}
    dec = pd.qcut(f.risk.rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
    out["uplift_by_risk_decile (D10 = highest risk)"] = {
        str(k): x5.effect(g) for k, g in f.groupby(dec, observed=True)
    }

    def inc(g: pd.DataFrame, col: str) -> float:
        t, c = g[g.treated == 1], g[g.treated == 0]
        return float((t[col].mean() - c[col].mean()) * len(g))

    f["value_bought"] = f.value * f.target
    total_p, total_v = inc(f, "target"), inc(f, "value_bought")
    rules = {
        "highest risk first": f.risk,
        "highest predicted uplift first": f.uplift_hat,
        "highest uplift x value first": f.uplift_hat * f.value,
        "lowest propensity first": -s.p_control,
    }
    out["captured_at_budget"] = {}
    for k in (10, 20, 30, 50):
        row = {}
        for name, score in rules.items():
            top = f[score.rank(method="first", ascending=False) <= len(f) * k / 100]
            row[name] = {"incremental_purchases_share": r(inc(top, "target") / total_p, 3),
                         "incremental_value_share":
                             r(inc(top, "value_bought") / total_v, 3)}  # fmt: skip
        row["random (expected)"] = {"incremental_purchases_share": k / 100,
                                    "incremental_value_share": k / 100}  # fmt: skip
        out["captured_at_budget"][f"top_{k}pct"] = row
    out["notes"] = {
        "risk": "1 - cross-fitted P(buy | no SMS)",
        "value": "average basket before the campaign; the outcome has no spend, so "
        "value captured = purchases x prior basket (an approximation)",
    }
    return out


# -- 5. fresh start: comebacks at temporal landmarks ------------------------------------


def freshstart(name: str, *paths: str) -> dict:
    o = orders_of(name, list(paths)).sort_values(["c", "date"])
    o["gap"] = o.groupby("c").date.diff().dt.days
    lapse = 28 if name == "cj" else 60
    o = o[o.gap.notna()]
    o["comeback"] = o.gap >= lapse
    o["monday"] = o.date.dt.dayofweek == 0
    o["month_start"] = o.date.dt.day <= 3
    o["january_first_2w"] = (o.date.dt.month == 1) & (o.date.dt.day <= 14)
    out = {"lapse_days": lapse, "repeat_orders": int(len(o)), "comebacks": int(o.comeback.sum())}
    for lm in ["monday", "month_start", "january_first_2w"]:
        cb, rg = o.loc[o.comeback, lm], o.loc[~o.comeback, lm]
        if len(cb) < MIN_CELL or rg.sum() < MIN_CELL:
            continue
        p1, p0 = cb.mean(), rg.mean()
        se = np.sqrt(p1 * (1 - p1) / len(cb) + p0 * (1 - p0) / len(rg))
        out[lm] = {"comeback_share": r(p1), "regular_share": r(p0), "ratio": r(p1 / p0, 3),
                   "diff_ci": [r(p1 - p0 - 1.96 * se), r(p1 - p0 + 1.96 * se)]}  # fmt: skip
    wd = o.date.dt.dayofweek
    out["weekday_share"] = {
        day: {
            "comeback": r((wd[o.comeback] == i).mean()),
            "regular": r((wd[~o.comeback] == i).mean()),
        }
        for i, day in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    }
    out["notes"] = {"comeback": f"an order {lapse}+ days after the customer's previous one",
                    "caution": "month starts are also paydays; the comparison with regular "
                               "orders nets out what affects all orders alike"}  # fmt: skip
    return out


# -- 6. delivery speed (Olist) ------------------------------------------------------------


def delivery_olist(d: str) -> dict:
    d = Path(d)
    ts = ["order_purchase_timestamp", "order_delivered_customer_date",
          "order_estimated_delivery_date"]  # fmt: skip
    o = pd.read_csv(d / "olist_orders_dataset.csv", parse_dates=ts)
    cust = pd.read_csv(d / "olist_customers_dataset.csv",
                       usecols=["customer_id", "customer_unique_id", "customer_state"])  # fmt: skip
    o = o.merge(cust, on="customer_id")
    o["day"] = o.order_purchase_timestamp.dt.floor("D")
    first_day = o.groupby("customer_unique_id").day.transform("min")
    first = o[o.day == first_day].sort_values("order_purchase_timestamp")
    first = first.drop_duplicates("customer_unique_id").set_index("order_id")
    nxt = o[o.day > first_day].groupby("customer_unique_id").day.min()
    first = first[(first.day >= "2017-01-01") & (first.day < "2018-03-01")]
    first = first[first.order_status.eq("delivered") & first.order_delivered_customer_date.notna()]
    gap = nxt.reindex(first.customer_unique_id).to_numpy() - first.day.to_numpy()
    y = pd.Series(gap / np.timedelta64(1, "D") <= 180, index=first.index).astype(float)
    days = (first.order_delivered_customer_date - first.order_purchase_timestamp).dt.days.clip(
        0, 60
    )
    late = first.order_delivered_customer_date.dt.floor("D") > first.order_estimated_delivery_date
    promised = (first.order_estimated_delivery_date - first.day).dt.days.clip(0, 90)
    items = pd.read_csv(d / "olist_order_items_dataset.csv",
                        usecols=["order_id", "price", "freight_value"])  # fmt: skip
    it = items.groupby("order_id")[["price", "freight_value"]].sum().reindex(first.index)
    fe = [first.customer_state.rename("state"), first.day.dt.to_period("M").rename("month"),
          pd.qcut(it.price.rank(method="first"), 5, labels=False).rename("valueq"),
          pd.qcut((it.freight_value / (it.price + it.freight_value)).rank(method="first"), 5,
                  labels=False).rename("freightq")]  # fmt: skip
    base = y.mean()
    m1 = ols(y, pd.DataFrame({"delivery_days": days}), fe)
    m2 = ols(y, pd.DataFrame({"delivery_days": days, "late": late.astype(float),
                              "promised_days": promised}), fe)  # fmt: skip
    per_day = m1["delivery_days"]["coef"]
    return {
        "cohort": int(len(first)),
        "repeat_180d": r(base),
        "by_delivery_days": {
            k: rate(y[b == k])
            for b in [
                pd.cut(
                    days, [-1, 7, 14, 21, 30, 60], labels=["0-7", "8-14", "15-21", "22-30", "31+"]
                )
            ]
            for k in b.cat.categories
        },  # fmt: skip
        "within_state_month": m1,
        "within_state_month_with_promise_and_late": m2,
        "relative_change_per_day_faster": r(-per_day / base, 4) if per_day is not None else None,
        "notes": {
            "controls": "state, purchase month, value and freight-share quintiles",
            "paper": "Fisher et al.: +1.45% online sales per business day faster",
        },
    }


# -- 7. diversity at equal activity --------------------------------------------------------


def diversity_core(lines: pd.DataFrame, C: int, W: int, H: int) -> dict:
    """lines: c, day, cat (one row per item or event). Active in [C-W, C);
    retained = any purchase in [C, C+H)."""
    obs = lines[(lines.day >= C - W) & (lines.day < C) & lines.cat.notna()]
    buy = lines[(lines.day >= C) & (lines.day < C + H)]
    cnt = obs.groupby(["c", "cat"], observed=True).size()
    tot = cnt.groupby(level="c").sum()
    p = cnt / tot.reindex(cnt.index.get_level_values("c")).to_numpy()
    ent = (-(p * np.log(p))).groupby(level="c").sum()
    f = pd.DataFrame({"items": tot, "entropy": ent,
                      "categories": cnt.groupby(level="c").size(),
                      "days": obs.groupby("c").day.nunique()})  # fmt: skip
    f["retained"] = f.index.isin(buy.c.unique())
    f = f[f["items"] >= 2]
    strata = pd.qcut(f["items"].rank(method="first"), 5, labels=False).astype(str) + "/" + \
        f.days.clip(upper=6).astype(str)  # fmt: skip
    f["div_tercile"] = f.groupby(strata).entropy.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if len(s) >= 3 else np.nan
    )
    diffs, w = [], []
    for _, g in f.groupby(strata):
        hi, lo = g[g.div_tercile == 2], g[g.div_tercile == 0]
        if len(hi) >= MIN_CELL and len(lo) >= MIN_CELL:
            diffs.append(hi.retained.mean() - lo.retained.mean())
            w.append(len(g))
    out = {
        "customers": int(len(f)),
        "retained": r(f.retained.mean()),
        "retained_by_diversity_tercile_raw": {
            str(k): rate(g.retained.astype(float)) for k, g in f.groupby("div_tercile")
        },  # fmt: skip
        "most_minus_least_diverse_at_equal_activity": r(np.average(diffs, weights=w))
        if diffs
        else None,
        "strata_compared": len(diffs),
    }
    x = pd.DataFrame({"entropy": f.entropy})
    out["linear_with_activity_controls"] = ols(
        f.retained.astype(float), x, [pd.Series(strata, name="activity")]
    )["entropy"]
    return out


def diversity(name: str, *paths: str) -> dict:
    if name == "hm":
        x = hm_lines(*paths)
        out = diversity_core(x[["c", "day", "cat"]], C=365, W=180, H=180)
        out["notes"] = {
            "category": "product group",
            "windows": "180 days -> 180 days at 2019-09-20",
        }
        return out
    if name == "cj":
        tx = cj_tables(paths[0])["tx"]
        out = diversity_core(tx[["c", "day", "cat"]], C=181, W=90, H=90)
        out["notes"] = {"category": "department", "windows": "90 days -> 90 days at 2017-07-01"}
        return out
    if name == "rees46":
        import pyarrow.parquet as pq

        t = pq.read_table(paths[0], columns=["user", "t", "kind", "code"])
        u = t["user"].to_numpy()
        keep = u % 10 == 0
        e = pd.DataFrame({"c": u[keep], "day": (t["t"].to_numpy()[keep] + 3 * 3600) // 86400,
                          "kind": t["kind"].to_numpy()[keep],
                          "cat": t["code"].to_numpy()[keep]})  # fmt: skip
        del t, u
        e["cat"] = e.cat.where(e.cat >= 0)
        views = e[e.kind == 0]
        buys = e[e.kind == 2][["c", "day"]].assign(cat=1)
        lines = pd.concat([views[(views.day < 31)], buys[buys.day >= 31]])
        out = diversity_core(lines[["c", "day", "cat"]], C=31, W=31, H=30)
        out["notes"] = {"category": "category of viewed products",
                        "windows": "Oct views -> Nov purchase",
                        "sample": "1 in 10 users"}  # fmt: skip
        return out
    raise ValueError(name)


# -- main -----------------------------------------------------------------------------------

CHECKS = {
    ("acquisition", "hm"): acquisition_hm,
    ("acquisition", "olist"): acquisition_olist,
    ("established", "hm"): established_hm,
    ("established", "cj"): established_cj,
    ("offers", "cj"): offers_cj,
    ("futility", "x5"): futility_x5,
    ("delivery", "olist"): delivery_olist,
}

if __name__ == "__main__":
    check, name, *paths = sys.argv[1:]
    if check == "freshstart":
        res = freshstart(name, *paths)
    elif check == "diversity":
        res = diversity(name, *paths)
    else:
        res = CHECKS[(check, name)](*paths)
    print(json.dumps(res, indent=1, default=str))
