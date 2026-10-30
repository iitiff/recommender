"""Novelty and recovery of notification templates (Duolingo, Yancey & Settles 2020 data).

Duolingo practice reminders: for each notification, the template sent, the
templates eligible, the user's template history for about 30 days, and whether a
lesson was completed within 2 hours. The paper's "recovering" bandit assumes a
template's effect drops right after it was used and recovers with time. Here:

1. Conversion by days since the *same* template was last sent to this user
   (never in the window, < 1 day, ..., 14-30 days), adjusted for the template,
   the user's notification count in the window, days since any notification,
   recent volume and UI language (linear probability model, robust SEs).
2. Conversion by recent volume (notifications in the last 7 days), adjusted the
   same way: a fatigue check, confounded by engagement (see the notes).

The logging policy chose templates from eligibility and history, both in the
data, so adjusting for history features removes much of the selection; user
traits the policy did not use (engagement) still differ.

Data: CC BY-NC 4.0 (Harvard Dataverse doi:10.7910/DVN/23ZWVI). Aggregates only.

Usage:
  uv run --no-project --with pandas --with pyarrow --with statsmodels python -I \
    duolingo_recovery.py PARQUET_DIR [MAX_ROWS=3000000] > out.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MIN_CELL = 50


def r(x: float, n: int = 4) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def load(d: str, max_rows: int) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    files = sorted(p for p in Path(d).rglob("part-*.parquet") if not p.name.startswith("."))
    tables, n = [], 0
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g)
            tables.append(t)
            n += t.num_rows
            if n >= max_rows:
                break
        if n >= max_rows:
            break
    t = pa.concat_tables(tables).slice(0, max_rows)
    sel = t["selected_template"].to_numpy(zero_copy_only=False)
    h = t["history"].combine_chunks()
    parent = pc.list_parent_indices(h).to_numpy()
    flat = pc.list_flatten(h)
    ht = flat.field("template").to_numpy(zero_copy_only=False)
    hd = flat.field("n_days").to_numpy(zero_copy_only=False)
    df = pd.DataFrame({
        "converted": t["session_end_completed"].to_numpy(zero_copy_only=False).astype(float),
        "template": sel,
        "lang": t["ui_language"].to_numpy(zero_copy_only=False),
        "eligible": pc.list_value_length(t["eligible_templates"]).to_numpy(),
        "hist_n": pc.list_value_length(h).fill_null(0).to_numpy(),
    })  # fmt: skip
    same = ht == sel[parent]
    df["days_since_same"] = pd.Series(hd[same]).groupby(parent[same]).min().reindex(df.index)
    df["days_since_any"] = pd.Series(hd).groupby(parent).min().reindex(df.index)
    df["sent_7d"] = pd.Series(hd <= 7).groupby(parent).sum().reindex(df.index).fillna(0)
    df["same_7d"] = pd.Series(hd[same] <= 7).groupby(parent[same]).sum().reindex(df.index).fillna(0)
    return df


def ols(y: pd.Series, x: pd.DataFrame, fe: list[pd.Series]) -> dict:
    import statsmodels.api as sm

    parts = [x.astype(float)] + [
        pd.get_dummies(f.astype(str), prefix=f.name, drop_first=True, dtype=float) for f in fe
    ]
    X = sm.add_constant(pd.concat(parts, axis=1))
    m = sm.OLS(y.astype(float), X).fit(cov_type="HC1")
    ci = m.conf_int()
    return {c: {"coef": r(m.params[c], 5), "ci": [r(ci.loc[c, 0], 5), r(ci.loc[c, 1], 5)]}
            for c in x.columns}  # fmt: skip


def main(d: str, max_rows: int = 3_000_000) -> dict:
    df = load(d, max_rows)
    bins = [0, 1, 2, 3, 5, 7, 14, 31]
    labels = ["<1d", "1-2d", "2-3d", "3-5d", "5-7d", "7-14d", "14-30d"]
    df["recency"] = pd.cut(df.days_since_same, bins, labels=labels, right=False).astype(object)
    df["recency"] = df.recency.fillna("not sent in window")
    controls = [
        df.template.rename("template"),
        df.lang.rename("lang"),
        df.hist_n.clip(upper=40).floordiv(5).rename("hist_n"),
        pd.cut(df.days_since_any, [0, 0.9, 1.1, 2, 3, 7, 31], right=False)
        .astype(str)
        .rename("gap"),
        df.eligible.rename("eligible"),
    ]
    out = {
        "notifications": int(len(df)),
        "conversion": r(df.converted.mean()),
        "raw_by_recency_of_same_template": {
            k: {"n": int(len(g)), "conversion": r(g.converted.mean())}
            for k, g in df.groupby("recency")
            if len(g) >= MIN_CELL
        },
    }
    # adjusted: dummies for each recency band vs "not sent in window" (the novel case)
    dummies = pd.get_dummies(df.recency, dtype=float).drop(columns=["not sent in window"])
    out["adjusted_vs_not_sent_in_window"] = ols(
        df.converted, dummies, controls + [df.sent_7d.clip(upper=10).rename("sent_7d")]
    )
    # fatigue: recent volume, holding the same-template recency and the rest fixed
    vol = pd.get_dummies(df.sent_7d.clip(upper=8).astype(int).astype(str).rename("v"),
                         prefix="sent7d", dtype=float)  # fmt: skip
    vol = vol.drop(columns=[c for c in vol.columns if c.endswith("_7")])  # 7 = daily, reference
    out["adjusted_by_notifications_last_7d_vs_7"] = ols(
        df.converted, vol, controls + [df.recency.rename("recency")]
    )
    out["raw_by_notifications_last_7d"] = {
        str(k): {"n": int(len(g)), "conversion": r(g.converted.mean())}
        for k, g in df.groupby(df.sent_7d.clip(upper=8).astype(int))
        if len(g) >= MIN_CELL
    }
    out["by_template_novel_vs_repeat_within_7d"] = {
        t: {
            "novel": r(g.loc[g.days_since_same.isna(), "converted"].mean()),
            "repeat_within_7d": r(g.loc[g.days_since_same < 7, "converted"].mean()),
            "n": int(len(g)),
        }
        for t, g in df.groupby("template")
        if len(g) >= 1000
    }
    out["notes"] = {
        "sample": f"first {len(df):,} rows of training part 1 (of 87.7M training rows)",
        "window": "history covers about the previous 30 days",
        "fatigue_caveat": "reminders go to learners who haven't practiced; recent volume also "
        "tracks how many days a learner skipped, so the volume pattern is not causal",
    }
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(main(a[0], int(a[1]) if len(a) > 1 else 3_000_000), indent=1, default=str))
