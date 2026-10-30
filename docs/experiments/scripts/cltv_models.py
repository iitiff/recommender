"""Two uses of the signal-screen panels: shorter experiments and better CLTV targets.

1. CUPAC sizing (Etsy, "predicted control variates"). An experiment's outcome Y
   is adjusted by a covariate X measured before assignment: Y - theta * X. The
   variance falls by corr(Y, X)^2, and so does the sample size needed for the
   same detectable effect. Compared:
     CUPED: X = the same metric in the observation window (spend for spend,
            orders for retention);
     CUPAC: X = a cross-fitted LightGBM prediction of Y from every pre-period
            signal in the screen.
   Overall and by lifecycle stage (new customers have little history).

2. CLTV target: value is zero-inflated (many customers buy nothing) and
   heavy-tailed. Compared on held-out folds:
     log1p regression, back-transformed naively (expm1) and with Duan smearing;
     raw-value regression (L2);
     zero-inflated lognormal, as a hurdle model: P(buy) x exp(mu + sigma^2 / 2),
     with mu fitted on buyers' log value (the ZILN idea of Wang et al. 2019 with
     gradient-boosted parts instead of one network).
   Metrics: normalized Gini and top-decile capture (ranking), total predicted /
   actual and mean decile calibration error (level).

Usage (same datasets and paths as signal_screen.py):
  uv run --no-project --with pandas --with pyarrow --with scikit-learn \
    --with lightgbm --with scipy --with pyreadr python -I cltv_models.py hm ... > out.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signal_screen as ss  # noqa: E402

MAX_ROWS = 300_000
FOLDS = 3
r = ss.r


def capture(p: pd.DataFrame, fam: dict, notes: dict) -> tuple:
    return p, fam, notes


def folds(p: pd.DataFrame) -> np.ndarray:
    cust = pd.factorize(p.customer)[0]
    return np.random.default_rng(ss.SEED).permutation(cust.max() + 1)[cust] % FOLDS


def crossfit(p: pd.DataFrame, feats: list[str], y: np.ndarray, kind: str, mask=None) -> np.ndarray:
    """Out-of-fold predictions; train only on rows in mask (if given)."""
    out = np.full(len(p), np.nan)
    f = folds(p)
    for k in range(FOLDS):
        tr = (f != k) & (mask if mask is not None else True)
        te = f == k
        m = ss.lgbm(kind).fit(p.loc[tr, feats], y[tr])
        out[te] = (
            m.predict_proba(p.loc[te, feats])[:, 1]
            if kind == "clf"
            else m.predict(p.loc[te, feats])
        )
    return out


def var_reduction(y: np.ndarray, x: np.ndarray) -> float | None:
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < ss.MIN_CELL or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None
    return r(np.corrcoef(y[ok], x[ok])[0, 1] ** 2, 3)


def var_reduction_both(y: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> float | None:
    """R^2 of y on both covariates (CUPED and CUPAC together)."""
    ok = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    if ok.sum() < ss.MIN_CELL or np.std(y[ok]) == 0:
        return None
    X = np.column_stack([np.ones(ok.sum()), x1[ok], x2[ok]])
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    return r(1 - np.var(y[ok] - X @ beta) / np.var(y[ok]), 3)


def cupac(p: pd.DataFrame, feats: list[str]) -> dict:
    ys = {
        "spend": p.cltv.to_numpy(float),
        "log1p_spend": np.log1p(p.cltv.to_numpy(float)),
        "retained": p.retained.to_numpy(float),
    }
    pre = {
        "spend": p.spend.to_numpy(float),
        "log1p_spend": np.log1p(p.spend.to_numpy(float)),
        "retained": p.orders.to_numpy(float),
    }
    out = {}
    for name, y in ys.items():
        pred = crossfit(p, feats, y, "clf" if name == "retained" else "reg")
        row = {"overall": {"cuped": var_reduction(y, pre[name]),
                           "cupac": var_reduction(y, pred),
                           "both": var_reduction_both(y, pre[name], pred)}}  # fmt: skip
        for st in p.stage.unique():
            m = (p.stage == st).to_numpy()
            if m.sum() >= 1000:
                row[st] = {"n": int(m.sum()), "cuped": var_reduction(y[m], pre[name][m]),
                           "cupac": var_reduction(y[m], pred[m]),
                           "both": var_reduction_both(y[m], pre[name][m], pred[m])}  # fmt: skip
        out[name] = row
    out["sample_size_factor"] = "n_needed = n_unadjusted x (1 - variance reduction)"
    return out


def value_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    ok = np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    top = np.argsort(-pred, kind="stable")[: len(pred) // 10]
    dec = pd.qcut(pd.Series(pred).rank(method="first"), 10, labels=False).to_numpy()
    errs = []
    for d in range(10):
        a, b = y[dec == d].mean(), pred[dec == d].mean()
        if a > 0:
            errs.append(abs(b - a) / a)
    return {
        "norm_gini": r(ss.gini(y, pred) / ss.gini(y, y)),
        "top_decile_capture": r(y[top].sum() / y.sum()),
        "spearman": r(pd.Series(y).corr(pd.Series(pred), method="spearman")),
        "total_predicted_over_actual": r(pred.sum() / y.sum()),
        "mean_decile_calibration_error": r(np.mean(errs)),
    }


def targets(p: pd.DataFrame, feats: list[str]) -> dict:
    y = p.cltv.to_numpy(float)
    ly = np.log1p(y)
    f = folds(p)
    lp = crossfit(p, feats, ly, "reg")
    # Duan smearing on the training residuals of each fold
    smear = np.full(len(p), np.nan)
    for k in range(FOLDS):
        resid = ly[f != k] - lp[f != k]
        smear[f == k] = np.exp(lp[f == k]) * np.mean(np.exp(resid)) - 1
    raw = crossfit(p, feats, y, "reg")
    pos = y > 0
    prob = crossfit(p, feats, pos.astype(float), "clf")
    mu = crossfit(p, feats, np.where(pos, np.log(np.where(pos, y, 1.0)), np.nan), "reg", mask=pos)
    sigma2 = np.full(len(p), np.nan)
    for k in range(FOLDS):
        tr = (f != k) & pos
        sigma2[f == k] = np.var(np.log(y[tr]) - mu[tr])
    ziln = prob * np.exp(mu + sigma2 / 2)
    return {
        "zero_share": r(1 - pos.mean()),
        "log1p_naive": value_metrics(y, np.expm1(lp)),
        "log1p_smearing": value_metrics(y, smear),
        "raw_l2": value_metrics(y, raw),
        "zero_inflated_lognormal": value_metrics(y, ziln),
    }


def main(name: str, *paths: str) -> dict:
    ss.FINISH = capture
    fn = {"hm": ss.hm, "cdnow": ss.cdnow, "onlineretail": ss.onlineretail, "olist": ss.olist,
          "cj": ss.cj, "rees46": ss.rees46, "otto": ss.otto}[name]  # fmt: skip
    p, fam, notes = fn(*paths)
    if len(p) > MAX_ROWS:
        keep = p.customer.drop_duplicates().sample(MAX_ROWS, random_state=ss.SEED)
        p = p[p.customer.isin(keep)].reset_index(drop=True)
    feats = [c for cols in fam.values() for c in cols]
    p[feats] = p[feats].astype(float)
    if "spend" not in p:
        p["spend"] = 0.0
    return {
        "dataset": name,
        "rows": int(len(p)),
        "notes": notes,
        "cupac": cupac(p, feats),
        "cltv_targets": targets(p, feats),
    }


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:]), indent=1, default=str))
