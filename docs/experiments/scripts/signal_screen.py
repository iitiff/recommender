"""Signal vs future value (CLTV) screen on public datasets.

Which behavioral signals predict a customer's value in the next months beyond
RFM, without redundancy? A *screening* analysis: it finds predictive signals,
not levers. Signals that pass move on to randomized tests.

Definitions:
- Cutoff C splits an observation window [C - W, C), where every feature is
  computed, from an outcome window [C, C + H), where CLTV is measured. No
  feature uses anything at or after C (delivery dates, reviews and reference
  prices included).
- CLTV = net spend in the outcome window (returns subtracted where the data
  has them); where there are no prices (OTTO, RetailRocket), items bought.
  Targets: log1p(CLTV) and retained (any purchase in the outcome window).
- Population: customers with a purchase or a visit in the observation window.
- Lifecycle stage from orders before C: prospect (0, event data only), new (1),
  developing (2-4), established (5+).

Steps (per dataset): univariate screen (Spearman, mutual information,
coverage); redundancy clusters (|Spearman| > 0.7); incremental value of each
signal family over RFM with LightGBM on a holdout split by customer (R2 on
log CLTV, normalized Gini, top-decile CLTV capture, AUC for retained); mean
|SHAP| and SHAP by feature decile (thresholds); the same by lifecycle stage;
stability of the univariate ranking across cutoffs.

Datasets (aggregates only; cells under 50 customers are not reported):
  hm          TRANSACTIONS_DIR ARTICLES_PARQUET   fashion, 2 cutoffs, H 12 months
  cdnow       CDNOW_master.txt                    music, 2 cutoffs, H 9 months
  onlineretail online_retail_II.csv               B2B gifts with returns, 2 cutoffs
  olist       OLIST_DIR                           marketplace: deliveries, reviews
  cj          COMPLETEJOURNEY_DIR                 grocery: campaigns, coupons
  rees46      events.parquet                      sessions, views, carts (31 + 30 days)
  otto        events.parquet                      clicks, carts, orders (14 + 13 days)
  retailrocket DATA_DIR                           views, carts, stock-outs (90 + 47 days)

Usage:
  uv run --no-project --with pandas --with pyarrow --with scikit-learn \
    --with lightgbm --with scipy --with pyreadr python -I signal_screen.py hm ... > hm.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400
GAP = 1800  # seconds of inactivity that end a visit
MIN_CELL = 50
SEED = 0
VIEW, CART, BUY = 0, 1, 2


def r(x: float, n: int = 3) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), n)


# -- features ------------------------------------------------------------------------


def weekly_slope(c: pd.Series, day: pd.Series, C: float, W: float) -> pd.Series:
    """Least-squares slope of weekly counts over the window's full weeks."""
    nw = int(W // 7)
    if nw < 2:
        return pd.Series(dtype=float)
    w = ((C - day) // 7).astype(int)
    keep = w < nw
    t = nw - 1 - w[keep]
    denom = nw * (nw**2 - 1) / 12
    return ((t - (nw - 1) / 2) / denom).groupby(c[keep].to_numpy()).sum()


def order_features(o: pd.DataFrame, C: float, W: float) -> tuple[pd.DataFrame, pd.Series]:
    """o: c, day (int), spend, items: one row per customer-day with a purchase."""
    pre = o[o.day < C]
    g = pre.groupby("c")
    n_all = g.size()
    f = pd.DataFrame(index=n_all.index)
    f["recency_days"] = C - g.day.max()
    f["tenure_days"] = C - g.day.min()
    obs = pre[pre.day >= C - W]
    go = obs.groupby("c")
    f["orders"] = go.size().reindex(f.index, fill_value=0)
    f["spend"] = go.spend.sum().reindex(f.index, fill_value=0.0)
    f["aov"] = (f.spend / f.orders).where(f.orders > 0)
    last = obs[obs.day >= C - W / 3].groupby("c").size().reindex(f.index, fill_value=0)
    f["order_momentum"] = (last + 1) / ((f.orders - last) / 2 + 1)
    f["order_slope"] = weekly_slope(obs.c, obs.day, C, W).reindex(f.index, fill_value=0.0)
    gaps = pre.day.groupby(pre.c).diff()
    med = gaps.groupby(pre.c).median()
    f["gap_ratio"] = (f.recency_days / med.reindex(f.index)).where(n_all >= 3)
    f["lifetime_orders"] = n_all
    stage = pd.Series(
        np.select([n_all >= 5, n_all >= 2], ["established", "developing"], "new"), index=f.index
    )
    return f, stage


def targets(o: pd.DataFrame, C: float, H: float, value: str = "spend") -> pd.Series:
    post = o[(o.day >= C) & (o.day < C + H)]
    return post.groupby("c")[value].sum()


def sessionize(e: pd.DataFrame) -> pd.DataFrame:
    """e sorted by (c, t). Adds sid (visit id)."""
    c, t = e.c.to_numpy(), e.t.to_numpy()
    new = np.r_[True, (c[1:] != c[:-1]) | (np.diff(t) > GAP)]
    return e.assign(sid=np.cumsum(new))


def event_features(e: pd.DataFrame, C: float, W: float) -> pd.DataFrame:
    """e: c, t (seconds), kind, product, sorted by (c, t). C, W in days."""
    x = sessionize(e[(e.t >= (C - W) * DAY) & (e.t < C * DAY)])
    x = x.assign(view=x.kind == VIEW, cart=x.kind == CART, buy=x.kind == BUY)
    s = x.groupby("sid").agg(
        c=("c", "first"),
        start=("t", "min"),
        end=("t", "max"),
        view=("view", "max"),
        cart=("cart", "max"),
        buy=("buy", "max"),
    )
    gs = s.groupby("c")
    f = pd.DataFrame(index=gs.size().index)
    f["sessions"] = gs.size()
    f["avg_session_sec"] = (s.end - s.start).groupby(s.c).mean()
    recent = s[s.start >= (C - 7) * DAY].groupby("c").size().reindex(f.index, fill_value=0)
    f["session_momentum"] = (recent + 1) / (f.sessions * 7 / W + 1)
    f["session_slope"] = weekly_slope(s.c, s.start / DAY, C, W).reindex(f.index, fill_value=0.0)
    f["sessions_last7_share"] = recent / f.sessions
    v = x[x.kind == VIEW]
    f["views"] = v.groupby("c").size().reindex(f.index, fill_value=0)
    f["distinct_products_viewed"] = (
        v.groupby("c")["product"].nunique().reindex(f.index, fill_value=0)
    )
    f["cart_adds"] = x[x.kind == CART].groupby("c").size().reindex(f.index, fill_value=0)
    f["cart_abandon_sessions"] = (s.cart & ~s.buy).groupby(s.c).sum().reindex(f.index)
    f["browse_no_buy_share"] = (s.view & ~s.buy).groupby(s.c).mean().reindex(f.index)
    pv = v.groupby(["c", "product"]).size()
    bought = set(zip(x.c[x.kind == BUY], x["product"][x.kind == BUY], strict=True))
    hot = pv[pv >= 3]
    not_bought = [k not in bought for k in hot.index]
    f["viewed_3x_not_bought"] = (
        hot[not_bought].groupby(level="c").size().reindex(f.index, fill_value=0)
    )
    return f


def panel_from(parts: list[tuple[str, pd.DataFrame, pd.Series, pd.Series]]) -> pd.DataFrame:
    """parts: (cutoff label, features, stage, cltv) -> one panel."""
    rows = []
    for label, f, stage, cltv in parts:
        p = f.copy()
        p["stage"] = stage.reindex(p.index).fillna("prospect")
        p["cltv"] = cltv.reindex(p.index).fillna(0.0).clip(lower=0)
        p["retained"] = cltv.reindex(p.index).fillna(0.0) > 0
        p["cutoff"] = label
        p["customer"] = p.index
        rows.append(p.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


# -- screening --------------------------------------------------------------------------


def gini(a: np.ndarray, p: np.ndarray) -> float:
    a = a[np.argsort(-p, kind="stable")]
    n = len(a)
    return float(np.cumsum(a).sum() / a.sum() / n - (n + 1) / (2 * n))


def lgbm(kind: str):
    import lightgbm as lgb

    args = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                verbose=-1)  # fmt: skip
    return lgb.LGBMRegressor(**args) if kind == "reg" else lgb.LGBMClassifier(**args)


def fit_eval(cols: list[str], tr: pd.DataFrame, te: pd.DataFrame) -> tuple[dict, object, dict]:
    from sklearn.metrics import r2_score, roc_auc_score

    reg = lgbm("reg").fit(tr[cols], np.log1p(tr.cltv))
    p = reg.predict(te[cols])
    y = te.cltv.to_numpy()
    top = np.argsort(-p, kind="stable")[: max(1, len(p) // 10)]
    out = {
        "r2_log_cltv": r(r2_score(np.log1p(y), p)),
        "norm_gini": r(gini(y, p) / gini(y, y)) if y.sum() > 0 else None,
        "top_decile_capture": r(y[top].sum() / y.sum()) if y.sum() > 0 else None,
    }
    preds = {"log": p}
    counts = [tr.retained.sum(), (~tr.retained).sum(), te.retained.sum(), (~te.retained).sum()]
    if min(counts) >= MIN_CELL:
        clf = lgbm("clf").fit(tr[cols], tr.retained)
        preds["prob"] = clf.predict_proba(te[cols])[:, 1]
        out["auc_retained"] = r(roc_auc_score(te.retained, preds["prob"]))
    return out, reg, preds


def lift_ci(te: pd.DataFrame, base: dict, new: dict, reps: int = 200) -> dict:
    """Paired bootstrap (resampling customers) of the lift over RFM: 95% intervals."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(SEED)
    cust = te.customer.to_numpy()
    groups = pd.Series(np.arange(len(te))).groupby(pd.factorize(cust)[0]).agg(list).tolist()
    ly, ret = np.log1p(te.cltv.to_numpy()), te.retained.to_numpy()
    d_r2, d_auc = [], []
    for _ in range(reps):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        yy = ly[idx]
        sst = ((yy - yy.mean()) ** 2).sum()
        d_r2.append(
            (((yy - base["log"][idx]) ** 2).sum() - ((yy - new["log"][idx]) ** 2).sum()) / sst
        )
        if "prob" in base and "prob" in new and len(np.unique(ret[idx])) == 2:
            d_auc.append(roc_auc_score(ret[idx], new["prob"][idx])
                         - roc_auc_score(ret[idx], base["prob"][idx]))  # fmt: skip
    out = {"delta_r2_ci": [r(np.percentile(d_r2, 2.5)), r(np.percentile(d_r2, 97.5))]}
    if d_auc:
        out["delta_auc_ci"] = [r(np.percentile(d_auc, 2.5)), r(np.percentile(d_auc, 97.5))]
    return out


def incremental(
    fam: dict[str, list[str]], tr: pd.DataFrame, te: pd.DataFrame, ci: bool = True
) -> tuple[dict, object]:
    base = fam["RFM"]
    m, _, p0 = fit_eval(base, tr, te)
    out = {"RFM": m}
    every = [c for cols in fam.values() for c in cols]
    runs = [(f"RFM + {k}", base + v) for k, v in fam.items() if k != "RFM"] + [("All", every)]
    model = None
    for name, cols in runs:
        m, model, p1 = fit_eval(cols, tr, te)
        out[name] = m | (lift_ci(te, p0, p1) if ci else {})
    return out, model


def shap_summary(model, te: pd.DataFrame, feats: list[str]) -> dict:
    x = te[feats].sample(min(len(te), 20_000), random_state=SEED)
    contrib = model.booster_.predict(x, pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=feats).sort_values(ascending=False)
    out = {"mean_abs_shap": {k: r(v, 4) for k, v in imp.items()}, "by_decile": {}}
    for f in imp.index[:10]:
        v, s = x[f].to_numpy(dtype=float), contrib[:, feats.index(f)]
        ok = np.isfinite(v)
        rows = []
        if ok.sum() >= MIN_CELL:
            q = pd.qcut(pd.Series(v[ok]).rank(method="first"), 10, labels=False)
            for b in range(10):
                m = (q == b).to_numpy()
                rows.append([r(v[ok][m].min(), 2), r(v[ok][m].max(), 2), r(s[ok][m].mean(), 3)])
        if (~ok).sum() >= MIN_CELL:
            rows.append(["missing", None, r(s[~ok].mean(), 3)])
        out["by_decile"][f] = rows
    return out


def redundancy(p: pd.DataFrame, feats: list[str]) -> list[list[str]]:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    rho = p[feats].sample(min(len(p), 50_000), random_state=SEED).corr("spearman").abs()
    rho = rho.fillna(0.0).to_numpy()
    np.fill_diagonal(rho, 1.0)
    labels = fcluster(linkage(squareform(1 - rho, checks=False), "average"), 0.3, "distance")
    groups = pd.Series(feats).groupby(labels).agg(list)
    return [g for g in groups if len(g) > 1]


def univariate(p: pd.DataFrame, feats: list[str], fam_of: dict[str, str]) -> dict:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    s = p.sample(min(len(p), 200_000), random_state=SEED)
    m = s.sample(min(len(s), 20_000), random_state=SEED)
    xm = m[feats].astype(float).fillna(-1.0)
    mi_c = mutual_info_regression(xm, np.log1p(m.cltv), random_state=SEED)
    mi_r = mutual_info_classif(xm, m.retained, random_state=SEED)
    out = {}
    for i, f in enumerate(feats):
        if s[f].notna().sum() < MIN_CELL:
            out[f] = {"family": fam_of[f], "coverage": r(s[f].notna().mean()), "suppressed": True}
            continue
        out[f] = {
            "family": fam_of[f],
            "coverage": r(s[f].notna().mean()),
            "spearman_cltv": r(s[f].corr(s.cltv, method="spearman")),
            "spearman_retained": r(s[f].corr(s.retained.astype(float), method="spearman")),
            "mi_cltv": r(mi_c[i], 4),
            "mi_retained": r(mi_r[i], 4),
        }
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1].get("spearman_cltv") or 0)))


def screen(p: pd.DataFrame, fam: dict[str, list[str]], notes: dict) -> dict:
    fam = {k: v for k, v in fam.items() if v}
    feats = [c for cols in fam.values() for c in cols]
    fam_of = {c: k for k, cols in fam.items() for c in cols}
    p[feats] = p[feats].astype(float)
    out = {
        "notes": notes,
        "population": {
            "rows": int(len(p)),
            "customers": int(p.customer.nunique()),
            "cutoffs": sorted(p.cutoff.unique().tolist()),
            "retained": r(p.retained.mean()),
            "cltv_mean": r(p.cltv.mean(), 2),
            "cltv_median": r(p.cltv.median(), 2),
            "cltv_top10pct_share": r(p.cltv.nlargest(len(p) // 10).sum() / p.cltv.sum()),
            "stages": {k: int(v) for k, v in p.stage.value_counts().items()},
        },
        "families": fam,
    }
    out["univariate"] = univariate(p, feats, fam_of)
    out["redundant_clusters"] = redundancy(p, feats)
    rng = np.random.default_rng(SEED)
    cust = p.customer.unique()
    test_c = set(rng.choice(cust, size=int(0.3 * len(cust)), replace=False))
    is_te = p.customer.isin(test_c)
    tr, te = p[~is_te], p[is_te]
    out["incremental"], model = incremental(fam, tr, te)
    out["shap"] = shap_summary(model, te, feats)
    out["by_stage"] = {}
    for st in p.stage.unique():
        trs, tes = tr[tr.stage == st], te[te.stage == st]
        if len(tes) < 1000 or tes.retained.sum() < MIN_CELL:
            continue
        sub = p[p.stage == st]
        out["by_stage"][st] = {
            "n": int(len(sub)),
            "retained": r(sub.retained.mean()),
            "spearman_cltv": {
                f: r(sub[f].corr(sub.cltv, method="spearman"))
                for f in feats
                if sub[f].nunique() > 1 and sub[f].notna().sum() >= MIN_CELL
            },
            "incremental": incremental(fam, trs, tes, ci=False)[0],
        }
    if p.cutoff.nunique() > 1:
        per = {
            cut: pd.Series(
                {
                    f: abs(g[f].corr(g.cltv, method="spearman"))
                    for f in feats
                    if g[f].notna().sum() >= MIN_CELL
                }
            )
            for cut, g in p.groupby("cutoff")
        }
        a, b = list(per.values())[:2]
        out["stability"] = {
            "rank_agreement_spearman": r(a.corr(b, method="spearman")),
            "top10_overlap": len(set(a.nlargest(10).index) & set(b.nlargest(10).index)),
            "per_cutoff_spearman_cltv": {
                k: {f: r(x) for f, x in v.items()} for k, v in per.items()
            },
        }
    return out


# -- datasets ----------------------------------------------------------------------------


def day_of(ts: pd.Series, epoch: str) -> pd.Series:
    return (pd.to_datetime(ts) - pd.Timestamp(epoch)).dt.days


def hm(tx_dir: str, articles: str, every: int = 5) -> dict:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    epoch, W, H = "2018-09-20", 90, 365
    t = ds.dataset(tx_dir).to_table(
        columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]
    )
    c = pc.dictionary_encode(t["customer_id"]).combine_chunks().indices.to_numpy()
    keep = c % every == 0
    a = pq.read_table(articles, columns=["article_id", "product_group_name"]).to_pandas()
    lines = pd.DataFrame(
        {
            "c": c[keep],
            "day": day_of(t["t_dat"].to_pandas()[keep], epoch).to_numpy(),
            "article": t["article_id"].to_numpy()[keep],
            "price": t["price"].to_numpy()[keep],
            "online": t["sales_channel_id"].to_numpy()[keep] == 2,
        }
    )
    del t
    lines["group"] = a.set_index("article_id").product_group_name.reindex(lines.article).to_numpy()
    orders = lines.groupby(["c", "day"]).agg(spend=("price", "sum"), items=("price", "size"))
    orders = orders.reset_index()
    parts = []
    for label in ("2019-06-01", "2019-09-01"):
        C = (pd.Timestamp(label) - pd.Timestamp(epoch)).days
        f, stage = order_features(orders, C, W)
        obs = lines[(lines.day >= C - W) & (lines.day < C)]
        f = f.loc[f.index.isin(obs.c.unique())]
        ref = lines[lines.day < C].groupby("article").price.max()  # pre-cutoff reference
        obs = obs.assign(markdown=obs.price < 0.9 * ref.reindex(obs.article).to_numpy())
        g = obs.groupby("c")
        f["markdown_share"] = g.markdown.mean()
        f["online_share"] = g.online.mean()
        f["distinct_groups"] = g.group.nunique()
        parts.append((label, f, stage, targets(orders, C, H)))
    fam = {
        "RFM": ["recency_days", "orders", "spend", "aov", "tenure_days"],
        "Momentum": ["order_momentum", "order_slope", "gap_ratio"],
        "Breadth & lifecycle": ["distinct_groups", "lifetime_orders"],
        "Response (discounts)": ["markdown_share"],
        "Channel": ["online_share"],
    }
    notes = {"sample": f"1 in {every} customers", "W": W, "H": H,
             "unavailable": "sessions, opens, carts, returns, deliveries, tickets, reviews",
             "tenure": "left-censored at 2018-09-20"}  # fmt: skip
    return screen(panel_from(parts), fam, notes)


def cdnow(path: str) -> dict:
    W, H = 90, 270
    x = pd.read_csv(path, sep=r"\s+", header=None, names=["c", "date", "items", "spend"],
                    dtype={"date": str})  # fmt: skip
    x["day"] = (pd.to_datetime(x.date, format="%Y%m%d") - pd.Timestamp("1997-01-01")).dt.days
    orders = x.groupby(["c", "day"]).agg(spend=("spend", "sum"), items=("items", "sum"))
    orders = orders.reset_index()
    parts = []
    for label, C in (("1997-07-01", 181), ("1997-10-01", 273)):
        f, stage = order_features(orders, C, W)
        f = f[f.orders > 0]
        parts.append((label, f, stage, targets(orders, C, H)))
    fam = {
        "RFM": ["recency_days", "orders", "spend", "aov", "tenure_days"],
        "Momentum": ["order_momentum", "order_slope", "gap_ratio"],
        "Lifecycle": ["lifetime_orders"],
    }
    notes = {"W": W, "H": H, "cohort": "first-ever buyers of Q1 1997", "unavailable": "all but RFM"}
    return screen(panel_from(parts), fam, notes)


def onlineretail(path: str) -> dict:
    W, H, epoch = 90, 365, "2009-12-01"
    cols = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"]
    x = pd.read_csv(path, usecols=cols, dtype={"Invoice": str, "StockCode": str})
    x = x[x["Customer ID"].notna() & (x.Price > 0)]
    x["c"] = x["Customer ID"].astype(int)
    x["day"] = day_of(x.InvoiceDate, epoch)
    x["value"] = x.Quantity * x.Price
    ret = x.Invoice.str.startswith("C") | (x.Quantity < 0)
    buys, rets = x[~ret], x[ret]
    orders = buys.groupby(["c", "day"]).agg(spend=("value", "sum"), items=("value", "size"))
    orders = orders.reset_index()
    rday = rets.groupby(["c", "day"]).value.sum().abs().rename("returned").reset_index()
    parts = []
    for label in ("2010-09-01", "2010-12-01"):
        C = (pd.Timestamp(label) - pd.Timestamp(epoch)).days
        f, stage = order_features(orders, C, W)
        f = f[f.orders > 0]
        ro = rday[(rday.day >= C - W) & (rday.day < C)].groupby("c")
        f["returns"] = ro.size().reindex(f.index, fill_value=0)
        f["return_rate"] = ro.returned.sum().reindex(f.index, fill_value=0) / f.spend
        ob = buys[(buys.day >= C - W) & (buys.day < C)]
        f["distinct_products"] = ob.groupby("c").StockCode.nunique()
        net = targets(orders, C, H).sub(
            rday[(rday.day >= C) & (rday.day < C + H)].groupby("c").returned.sum(), fill_value=0
        )
        parts.append((label, f, stage, net))
    fam = {
        "RFM": ["recency_days", "orders", "spend", "aov", "tenure_days"],
        "Momentum": ["order_momentum", "order_slope", "gap_ratio"],
        "Friction (returns)": ["returns", "return_rate"],
        "Breadth & lifecycle": ["distinct_products", "lifetime_orders"],
    }
    notes = {"W": W, "H": H, "business": "wholesale (B2B)", "cltv": "net of returns",
             "breadth": "distinct products (no categories)"}  # fmt: skip
    return screen(panel_from(parts), fam, notes)


def olist(d: str) -> dict:
    W, H, epoch = 90, 365, "2016-09-01"
    d = Path(d)
    cust = pd.read_csv(d / "olist_customers_dataset.csv",
                       usecols=["customer_id", "customer_unique_id"])  # fmt: skip
    o = pd.read_csv(d / "olist_orders_dataset.csv").merge(cust, on="customer_id")
    o = o[~o.order_status.isin(["canceled", "unavailable"])]
    o["c"] = o.customer_unique_id
    o["day"] = day_of(o.order_purchase_timestamp, epoch)
    o["delivered"] = day_of(o.order_delivered_customer_date, epoch)
    o["estimated"] = day_of(o.order_estimated_delivery_date, epoch)
    items = pd.read_csv(d / "olist_order_items_dataset.csv")
    products = pd.read_csv(d / "olist_products_dataset.csv",
                           usecols=["product_id", "product_category_name"])  # fmt: skip
    items = items.merge(products, on="product_id", how="left")
    o = o.merge(items.groupby("order_id").agg(value=("price", "sum")), on="order_id")
    pay = pd.read_csv(d / "olist_order_payments_dataset.csv")
    pay["voucher"] = np.where(pay.payment_type == "voucher", pay.payment_value, 0.0)
    pay = pay.groupby("order_id").agg(paid=("payment_value", "sum"), voucher=("voucher", "sum"),
                                      installments=("payment_installments", "max"))  # fmt: skip
    o = o.merge(pay, on="order_id", how="left")
    rev = pd.read_csv(d / "olist_order_reviews_dataset.csv",
                      usecols=["order_id", "review_score", "review_creation_date"])  # fmt: skip
    rev["review_day"] = day_of(rev.review_creation_date, epoch)
    rev = rev.sort_values("review_day").drop_duplicates("order_id", keep="last")
    o = o.merge(rev[["order_id", "review_score", "review_day"]], on="order_id", how="left")
    lines = items.merge(o[["order_id", "c", "day"]], on="order_id")
    orders = o.groupby(["c", "day"]).agg(spend=("value", "sum"), items=("value", "size"))
    orders = orders.reset_index()
    parts = []
    for label in ("2017-05-01", "2017-08-01"):
        C = (pd.Timestamp(label) - pd.Timestamp(epoch)).days
        f, stage = order_features(orders, C, W)
        f = f[f.orders > 0]
        ob = o[(o.day >= C - W) & (o.day < C)]
        done = ob[ob.delivered < C]  # known at the cutoff
        f["late_deliveries"] = (done.delivered > done.estimated).groupby(done.c).sum()
        f["avg_delivery_days"] = (done.delivered - done.day).groupby(done.c).mean()
        seen = ob[ob.review_day < C]
        f["avg_review"] = seen.groupby("c").review_score.mean()
        f["low_reviews"] = (seen.review_score <= 2).groupby(seen.c).sum()
        f[["late_deliveries", "low_reviews"]] = f[["late_deliveries", "low_reviews"]].fillna(0)
        go = ob.groupby("c")
        f["voucher_share"] = go.voucher.sum() / go.paid.sum()
        f["installments"] = go.installments.mean()
        lo = lines[(lines.day >= C - W) & (lines.day < C)]
        f["distinct_categories"] = lo.groupby("c").product_category_name.nunique()
        parts.append((label, f, stage, targets(orders, C, H)))
    fam = {
        "RFM": ["recency_days", "orders", "spend", "aov", "tenure_days"],
        "Momentum": ["order_momentum", "order_slope", "gap_ratio"],
        "Friction": ["late_deliveries", "avg_delivery_days", "avg_review", "low_reviews"],
        "Breadth & lifecycle": ["distinct_categories", "lifetime_orders"],
        "Response (discounts)": ["voucher_share"],
        "Payment": ["installments"],
    }
    notes = {"W": W, "H": H, "leakage": "deliveries and reviews only if dated before the cutoff",
             "business": "marketplace; ~3% of buyers ever return"}  # fmt: skip
    return screen(panel_from(parts), fam, notes)


def cj(d: str) -> dict:
    import pyreadr

    W, H, epoch = 90, 240, "2017-01-01"
    d = Path(d)
    tx = pyreadr.read_r(str(d / "transactions.rds"))[None]
    prod = next(iter(pyreadr.read_r(str(d / "products.rda")).values()))
    camp = next(iter(pyreadr.read_r(str(d / "campaigns.rda")).values()))
    desc = next(iter(pyreadr.read_r(str(d / "campaign_descriptions.rda")).values()))
    red = next(iter(pyreadr.read_r(str(d / "coupon_redemptions.rda")).values()))
    tx["c"] = tx.household_id.astype(str)
    tx["day"] = day_of(tx.transaction_timestamp, epoch)
    tx["dept"] = tx.product_id.map(prod.set_index("product_id").department)
    orders = tx.groupby(["c", "day"]).agg(
        spend=("sales_value", "sum"), items=("sales_value", "size"),
        retail_disc=("retail_disc", "sum"), coupon_disc=("coupon_disc", "sum"),
    ).reset_index()  # fmt: skip
    desc["s"] = day_of(desc.start_date, epoch)
    desc["e"] = day_of(desc.end_date, epoch)
    camp = camp.merge(desc[["campaign_id", "s", "e"]], on="campaign_id")
    camp["c"] = camp.household_id.astype(str)
    red["c"] = red.household_id.astype(str)
    red["day"] = day_of(red.redemption_date, epoch)
    parts = []
    for label in ("2017-04-01", "2017-05-01"):
        C = (pd.Timestamp(label) - pd.Timestamp(epoch)).days
        f, stage = order_features(orders, C, W)
        f = f[f.orders > 0]
        ob = orders[(orders.day >= C - W) & (orders.day < C)].groupby("c")
        f["loyalty_discount_share"] = ob.retail_disc.sum() / (ob.spend.sum() + ob.retail_disc.sum())
        f["coupon_order_share"] = ob.coupon_disc.apply(lambda s: (s > 0).mean())
        rc = camp[(camp.s < C) & (camp.e >= C - W)].groupby("c").campaign_id.nunique()
        f["campaigns_received"] = rc.reindex(f.index, fill_value=0)
        rr = red[(red.day >= C - W) & (red.day < C)].groupby("c")
        f["coupons_redeemed"] = rr.size().reindex(f.index, fill_value=0)
        used = rr.campaign_id.nunique().reindex(f.index, fill_value=0)
        f["redemption_rate"] = (used / f.campaigns_received).where(f.campaigns_received > 0)
        lt = tx[(tx.day >= C - W) & (tx.day < C)].groupby("c")
        f["distinct_departments"] = lt.dept.nunique()
        f["stores_used"] = lt.store_id.nunique()
        parts.append((label, f, stage, targets(orders, C, H)))
    fam = {
        "RFM": ["recency_days", "orders", "spend", "aov", "tenure_days"],
        "Momentum": ["order_momentum", "order_slope", "gap_ratio"],
        "Breadth & lifecycle": ["distinct_departments", "lifetime_orders"],
        "Response (campaigns, coupons)": ["loyalty_discount_share", "coupon_order_share",
                                          "campaigns_received", "coupons_redeemed",
                                          "redemption_rate"],
        "Convenience": ["stores_used"],
    }  # fmt: skip
    notes = {"W": W, "H": H, "population": "2,469 frequent households (no new customers)",
             "tenure": "left-censored at 2017-01-01",
             "campaigns": "targeted, not random (selection bias)"}  # fmt: skip
    return screen(panel_from(parts), fam, notes)


def events_panel(e: pd.DataFrame, label: str, C: float, W: float, H: float,
                 cats: pd.Series | None = None) -> tuple:  # fmt: skip
    """e: c, t (s), kind, product, value (purchase value), sorted by (c, t)."""
    e = e.assign(day=e.t // DAY)
    buys = e[e.kind == BUY]
    orders = buys.groupby(["c", "day"]).agg(spend=("value", "sum"), items=("value", "size"))
    orders = orders.reset_index()
    fo, stage = order_features(orders, C, W)
    fe = event_features(e, C, W)
    f = fe.join(fo, how="left")
    f[["orders", "spend", "lifetime_orders"]] = f[["orders", "spend", "lifetime_orders"]].fillna(0)
    return label, f, stage, targets(orders, C, H)


def rees46(path: str, every: int = 10) -> dict:
    import pyarrow.parquet as pq

    W, H, shift = 31, 30, 3 * 3600  # Moscow time
    t = pq.read_table(path, columns=["user", "t", "kind", "product", "code", "price"])
    user = t["user"].to_numpy()
    keep = user % every == 0
    e = pd.DataFrame({k: t[k].to_numpy()[keep] for k in ["user", "t", "kind", "product", "code",
                                                       "price"]})  # fmt: skip
    del t, user
    e = e.rename(columns={"user": "c"})
    e["t"] = e.t.astype(np.int64) + shift
    e["value"] = e.price
    e = e.sort_values(["c", "t"], kind="stable").reset_index(drop=True)
    label, f, stage, cltv = events_panel(e, "2019-11-01", 31, W, H)
    x = e[(e.t >= 0) & (e.t < 31 * DAY)]
    f["distinct_categories_viewed"] = x[x.kind == VIEW].groupby("c").code.nunique()
    f["distinct_categories_bought"] = x[x.kind == BUY].groupby("c").code.nunique()
    fam = {
        "RFM": ["recency_days", "orders", "spend", "aov"],
        "Momentum": ["order_momentum", "session_momentum", "session_slope",
                     "sessions_last7_share"],
        "Engagement": ["sessions", "avg_session_sec", "views", "distinct_products_viewed"],
        "Intent": ["cart_adds", "cart_abandon_sessions", "viewed_3x_not_bought",
                   "browse_no_buy_share"],
        "Breadth": ["distinct_categories_viewed", "distinct_categories_bought"],
    }  # fmt: skip
    notes = {"sample": f"1 in {every} users", "W": W, "H": H,
             "history": "starts 2019-10-01, so tenure and lifecycle are not measurable",
             "unavailable": "searches, wishlist, opens, returns, deliveries"}  # fmt: skip
    return screen(panel_from([(label, f, stage, cltv)]), fam, notes)


def otto(path: str, every: int = 4) -> dict:
    W, H = 14, 13
    e = pd.read_parquet(path)
    e = e[e.user % every == 0].rename(columns={"user": "c"})
    e["t"] = e.t.astype(np.int64)
    e["value"] = 1.0  # no prices: value = items ordered
    e = e.sort_values(["c", "t"], kind="stable").reset_index(drop=True)
    label, f, stage, cltv = events_panel(e, "day 14", 14, W, H)
    fam = {
        "RFM": ["recency_days", "orders", "spend"],
        "Momentum": ["session_momentum", "session_slope", "sessions_last7_share"],
        "Engagement": ["sessions", "avg_session_sec", "views", "distinct_products_viewed"],
        "Intent": ["cart_adds", "cart_abandon_sessions", "viewed_3x_not_bought",
                   "browse_no_buy_share"],
    }  # fmt: skip
    notes = {"sample": f"1 in {every} of the stored 25% sample", "W": W, "H": H,
             "cltv": "items ordered (no prices)", "spend": "items ordered"}  # fmt: skip
    return screen(panel_from([(label, f, stage, cltv)]), fam, notes)


def retailrocket(d: str, every: int = 4) -> dict:
    W, H = 90, 47
    d = Path(d)
    x = pd.read_csv(d / "interactions.csv", usecols=["user_id", "item_id", "kind", "occurred_at"])
    x = x[x.user_id % every == 0]
    ts = pd.to_datetime(x.occurred_at, utc=True, format="ISO8601")
    t0 = ts.min().normalize()
    e = pd.DataFrame({
        "c": x.user_id.to_numpy(),
        "t": (ts - t0).dt.total_seconds().astype(np.int64).to_numpy(),
        "kind": x.kind.map({"view": VIEW, "add_to_cart": CART, "purchase": BUY}).to_numpy(),
        "product": x.item_id.to_numpy(),
    })  # fmt: skip
    e["value"] = 1.0
    e = e.sort_values(["c", "t"], kind="stable").reset_index(drop=True)
    label, f, stage, cltv = events_panel(e, "day 90", 90, W, H)
    av = pd.read_csv(d / "availability.csv")
    av["t"] = (
        (pd.to_datetime(av.valid_from, utc=True, format="ISO8601") - t0)
        .dt.total_seconds()
        .astype(np.int64)
    )
    av = av.sort_values("t")
    v = e[(e.kind == VIEW) & (e.t < 90 * DAY)].sort_values("t")
    v = pd.merge_asof(v, av[["item_id", "t", "available"]].rename(columns={"item_id": "product"}),
                      on="t", by="product")  # fmt: skip
    v["oos"] = v.available.astype(str).str.lower().eq("false")
    v = v[v.t >= 0]
    f["oos_views"] = v.groupby("c").oos.sum().reindex(f.index, fill_value=0)
    f["oos_view_share"] = f.oos_views / f.views.where(f.views > 0)
    fam = {
        "RFM": ["recency_days", "orders", "spend"],
        "Momentum": ["session_momentum", "session_slope", "sessions_last7_share"],
        "Engagement": ["sessions", "avg_session_sec", "views", "distinct_products_viewed"],
        "Intent": ["cart_adds", "cart_abandon_sessions", "viewed_3x_not_bought",
                   "browse_no_buy_share"],
        "Friction (stock-outs)": ["oos_views", "oos_view_share"],
    }  # fmt: skip
    notes = {"sample": f"1 in {every} visitors", "W": W, "H": H,
             "cltv": "items bought (prices hashed)",
             "resellers": "1.2% of buyers buy 10+ items; not removed"}  # fmt: skip
    return screen(panel_from([(label, f, stage, cltv)]), fam, notes)


if __name__ == "__main__":
    kind, *paths = sys.argv[1:]
    fn = {"hm": hm, "cdnow": cdnow, "onlineretail": onlineretail, "olist": olist, "cj": cj,
          "rees46": rees46, "otto": otto, "retailrocket": retailrocket}[kind]  # fmt: skip
    print(json.dumps(fn(*paths), indent=1, default=str))
