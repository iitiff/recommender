"""Cart abandonment: who comes back on their own, when, and who never does.

An abandoned cart is a visit (30 minutes of inactivity ends a visit) with an
add-to-cart and no purchase. The unit is a shopper's first abandoned visit in
the window; everything is measured from the end of that visit (its last
event), over the next 7 days, with no reminders in the data:
  - return: the next visit;
  - recovery: buying a carted product (organic, i.e. without a reminder);
  - substitute: buying another product in the same leaf category instead;
  - other: buying anything else.
Timing: hourly hazard of recovery since abandonment and the share of 7-day
recoveries that happen in each band of hours.
Segments: lifecycle before the visit, cart value, top-level category, items
in the cart, how many alternatives they compared in the visit, whether they
had carted the same product before, hour of day.
Trigger comparison: for every shopper active in the window, their first
visit of each kind (cart abandoned, browse abandoned = a product viewed 3+
times without carting, plain browse), and how often they buy the focal
product or anything within 7 days on their own.

REES46 logs add-to-cart for every purchase only from 2019-11-08 (day 38), and
the 2019-11-15..17 sale has a day with no purchases logged (day 45), so
abandonments come from days 48-53 (Nov 18-23, Moscow time) and their 7 days
of follow-up end by day 60.

Usage:
  uv run --no-project --with pandas --with pyarrow python -I \
    cart_abandonment.py rees46 events.parquet > out.json
  python -I cart_abandonment.py retailrocket data/retailrocket > out.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

GAP = 1800  # seconds of inactivity that end a visit
DAY = 86400
FOLLOW = 7 * DAY
VIEW, CART, BUY = 0, 1, 2
BANDS_H = [0.5, 1, 2, 4, 8, 24, 48, 72, 168]  # hours after the visit ended
VALUE_BANDS = [0, 25, 100, 300, 1e12]
VALUE_LABELS = ["under $25", "$25-100", "$100-300", "$300+"]


def r(x: float, n: int = 3) -> float:
    return round(float(x), n)


# -- loaders: events sorted by (user, t) with user, t (s), kind, product, leaf,
#    top (int code), price (float, NaN if unknown); names of the top codes; and
#    the abandonment window ----------------------------------------------------


def load_rees46(path: Path) -> tuple[pd.DataFrame, dict[int, str], tuple[int, int]]:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    names = json.loads(path.with_suffix(".json").read_text())["category_codes"]
    tops = sorted({n.split(".")[0] for n in names})
    top_of = np.full(max(names.values()) + 2, len(tops), dtype=np.int16)  # code -1 -> unknown
    for n, i in names.items():
        top_of[i] = tops.index(n.split(".")[0])
    shift = 3 * 3600  # Moscow time, so hours read as local
    lo, hi = 48 * DAY, 54 * DAY
    # only shoppers active in the window (with their full history), to fit in memory
    ut = pq.read_table(path, columns=["user", "t"]).to_pandas()
    t = ut.t.to_numpy()
    active = np.unique(ut.user.to_numpy()[(t + shift >= lo) & (t + shift < hi)])
    del ut, t
    table = ds.dataset(path).to_table(
        columns=["t", "kind", "product", "category_id", "code", "price", "user"],
        filter=ds.field("user").isin(pa.array(active)),
    )
    user = table["user"].to_numpy()
    t = table["t"].to_numpy().astype(np.int32) + shift
    order = np.lexsort((t, user))
    leaf, _ = pd.factorize(table["category_id"].to_numpy()[order])
    e = pd.DataFrame(
        {
            "user": user[order],
            "t": t[order].astype(np.int64),
            "kind": table["kind"].to_numpy()[order],
            "product": table["product"].to_numpy()[order],
            "leaf": leaf.astype(np.int32),
            "top": top_of[table["code"].to_numpy()[order]],
            "price": table["price"].to_numpy()[order],
        }
    )
    names_top = {i: n for i, n in enumerate(tops)} | {len(tops): "unknown"}
    return e, names_top, (lo, hi)


def load_retailrocket(d: Path) -> tuple[pd.DataFrame, dict[int, str], tuple[int, int]]:
    e = pd.read_csv(d / "interactions.csv", usecols=["user_id", "item_id", "kind", "occurred_at"])
    items = pd.read_csv(d / "catalog/items.csv", usecols=["item_id", "department", "type"])
    ts = pd.to_datetime(e.occurred_at, format="ISO8601")
    t0 = ts.min().floor("D")
    out = pd.DataFrame(
        {
            "user": e.user_id.to_numpy(),
            "t": ((ts - t0).dt.total_seconds()).astype(np.int64).to_numpy(),
            "kind": e.kind.map({"view": VIEW, "add_to_cart": CART, "purchase": BUY}).to_numpy(),
            "product": e.item_id.to_numpy(),
        }
    )
    meta = items.set_index("item_id")
    out["leaf"] = out["product"].map(meta.type).fillna(-1).astype(np.int64)
    out["top"] = out["product"].map(meta.department).fillna(-1).astype(np.int64)
    out["price"] = np.nan
    out = out.sort_values(["user", "t"], kind="stable").reset_index(drop=True)
    days = int(out.t.max() // DAY)
    names = {int(x): f"department {x}" for x in out.top.unique()}
    return out, names, (30 * DAY, (days - 7) * DAY)  # a month of history, 7 days follow-up


# -- visits -----------------------------------------------------------------------


def visits(e: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    u, t, k = e.user.to_numpy(), e.t.to_numpy(), e.kind.to_numpy()
    new = np.ones(len(e), bool)
    new[1:] = (u[1:] != u[:-1]) | (t[1:] - t[:-1] > GAP)
    vid = np.cumsum(new) - 1
    first = np.flatnonzero(new)
    last = np.r_[first[1:] - 1, len(e) - 1]
    n = len(first)
    v = pd.DataFrame(
        {
            "user": u[first],
            "start": t[first],
            "end": t[last],
            "views": np.bincount(vid, k == VIEW, n).astype(np.int32),
            "carts": np.bincount(vid, k == CART, n).astype(np.int32),
            "buys": np.bincount(vid, k == BUY, n).astype(np.int32),
        }
    )
    same_user_next = np.r_[v.user.to_numpy()[1:] == v.user.to_numpy()[:-1], False]
    v["next_start"] = np.where(same_user_next, np.r_[v.start.to_numpy()[1:], 0], -1)
    # history before each visit, within the user
    g = v.groupby("user", sort=False)
    v["prior_visits"] = g.cumcount()
    for col, flag in (("prior_buy_visits", v.buys > 0), ("prior_cart_visits", v.carts > 0)):
        f = flag.astype(np.int32)
        v[col] = f.groupby(v.user, sort=False).cumsum() - f
    return vid, v


def lifecycle(v: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                v.prior_visits == 0,
                v.prior_buy_visits >= 2,
                v.prior_buy_visits == 1,
                v.prior_cart_visits > 0,
            ],
            ["new visitor", "2+ orders", "1 order", "abandoned before"],
            "browser",
        ),
        index=v.index,
    )


# -- follow-up --------------------------------------------------------------------


def follow_up(e: pd.DataFrame, focal: pd.DataFrame, sub_leaf: bool = True) -> pd.DataFrame:
    """focal: one row per (case, user, end, product, leaf). Returns per case: hours
    to the first purchase of a focal product, of a same-leaf substitute, of
    anything, within 7 days of `end`."""
    cases = focal.drop_duplicates("case")[["case", "user", "end"]]
    buys = e.loc[e.kind == BUY, ["user", "t", "product", "leaf"]]
    b = buys.merge(cases, on="user")
    b = b[(b.t > b.end) & (b.t <= b.end + FOLLOW)]
    b["h"] = (b.t - b.end) / 3600
    fp = focal[["case", "product"]].drop_duplicates()
    fl = focal[["case", "leaf"]].drop_duplicates()
    b = b.merge(fp.assign(is_focal=True), on=["case", "product"], how="left")
    b = b.merge(fl.assign(same_leaf=True), on=["case", "leaf"], how="left")
    b["is_focal"] = b.is_focal.fillna(False).astype(bool)
    b["same_leaf"] = b.same_leaf.fillna(False).astype(bool) & ~b.is_focal
    out = cases.set_index("case")[[]]
    out["h_focal"] = b[b.is_focal].groupby("case").h.min()
    out["h_sub"] = b[b.same_leaf].groupby("case").h.min()
    out["h_any"] = b.groupby("case").h.min()
    return out


def summarise(c: pd.DataFrame) -> dict:
    """c: per case with h_return, h_focal, h_sub, h_any (hours; NaN = none in 7 days)."""
    n = len(c)
    rec = c.h_focal.dropna()
    bought_focal = c.h_focal.notna()
    out = {
        "n": int(n),
        "returned_24h": r((c.h_return <= 24).mean()),
        "returned_7d": r(c.h_return.notna().mean()),
        "bought_cart_item_1h": r((c.h_focal <= 1).mean()),
        "bought_cart_item_24h": r((c.h_focal <= 24).mean()),
        "bought_cart_item_7d": r(bought_focal.mean()),
        "bought_substitute_only_7d": r((c.h_sub.notna() & ~bought_focal).mean()),
        "bought_anything_7d": r(c.h_any.notna().mean()),
        "returned_no_purchase_7d": r((c.h_return.notna() & c.h_any.isna()).mean()),
        "never_returned_7d": r(c.h_return.isna().mean()),
    }
    if len(rec) >= 30:
        out["median_hours_to_recovery"] = r(rec.median(), 1)
        out["recoveries_within_2h_share"] = r((rec <= 2).mean())
        out["recoveries_after_24h_share"] = r((rec > 24).mean())
    return out


def timing(c: pd.DataFrame) -> dict:
    """Share of 7-day recoveries / returns in each band of hours; hourly hazard."""
    out = {}
    for col in ("h_focal", "h_return"):
        h = c[col].dropna().to_numpy()
        lo, shares = 0.0, {}
        for hi in BANDS_H:
            shares[f"{lo:g}-{hi:g}h"] = r(((h > lo) & (h <= hi)).mean())
            lo = hi
        out[col.replace("h_", "") + "_by_band"] = shares
    h = c.h_focal.to_numpy()
    hours = np.floor(np.nan_to_num(h, nan=1e9)).astype(np.int64)
    at_risk = np.array([(hours >= i).sum() for i in range(72)])
    events = np.bincount(hours[hours < 72], minlength=72)
    out["recovery_hazard_first_72h_pct"] = [r(100 * x, 2) for x in events / np.maximum(at_risk, 1)]
    return out


# -- main analysis ----------------------------------------------------------------


def analyse(e: pd.DataFrame, top_names: dict[int, str], window: tuple[int, int]) -> dict:
    vid, v = visits(e)
    v["stage"] = lifecycle(v)
    lo, hi = window
    in_win = (v.end >= lo) & (v.end < hi)
    v["h_return"] = np.where(
        (v.next_start > 0) & (v.next_start - v.end <= FOLLOW), (v.next_start - v.end) / 3600, np.nan
    )

    # --- cart abandonment cases: first abandoned visit per shopper in the window
    ab = v[in_win & (v.carts > 0) & (v.buys == 0)]
    ab = ab[~ab.user.duplicated()]
    case_of_visit = pd.Series(np.arange(len(ab)), index=ab.index)
    ev_case = case_of_visit.reindex(vid).to_numpy()
    sel = ~np.isnan(ev_case)
    ev = e.loc[sel].assign(case=ev_case[sel].astype(np.int64))
    carted = ev[ev.kind == CART]
    focal = (
        carted.groupby(["case", "product"])
        .agg(user=("user", "first"), leaf=("leaf", "first"), top=("top", "first"),
             price=("price", "max"))
        .reset_index()
    )  # fmt: skip
    focal["end"] = ab.end.to_numpy()[focal.case]
    c = follow_up(e, focal)
    c["h_return"] = ab.h_return.to_numpy()
    c["stage"] = ab.stage.to_numpy()
    c["items"] = focal.groupby("case").size().clip(upper=3).map({1: "1", 2: "2", 3: "3+"})
    value = focal.groupby("case").price.sum(min_count=1)
    c["value"] = pd.cut(value, VALUE_BANDS, labels=VALUE_LABELS, right=False).astype(str)
    main = focal.sort_values("price", ascending=False).drop_duplicates("case").set_index("case")
    c["top"] = main.top.map(top_names)
    # alternatives compared: distinct products viewed in the visit in a carted leaf
    views = ev[ev.kind == VIEW][["case", "product", "leaf"]].drop_duplicates()
    alt = views.merge(focal[["case", "leaf"]].drop_duplicates(), on=["case", "leaf"])
    n_alt = alt.groupby("case")["product"].nunique().reindex(c.index, fill_value=0)
    c["compared"] = pd.cut(n_alt, [-1, 1, 4, 1e9], labels=["0-1 products", "2-4", "5+"]).astype(str)
    # carted the same product in an earlier visit
    prior = e.loc[e.kind == CART, ["user", "product", "t"]].merge(
        focal[["case", "user", "product"]], on=["user", "product"]
    )
    start = ab.start.to_numpy()
    recart = prior[prior.t < start[prior.case]].case.unique()
    c["carted_before"] = np.where(c.index.isin(recart), "yes", "no")
    hour = (ab.end.to_numpy() % DAY) // 3600
    c["hour"] = pd.cut(hour, [-1, 6, 11, 17, 23], labels=["0-6", "7-11", "12-17", "18-23"]).astype(
        str
    )

    out = {
        "window_days": [lo // DAY, hi // DAY],
        "cases": summarise(c),
        "timing": timing(c),
        "segments": {},
    }
    for seg in ["stage", "value", "top", "items", "compared", "carted_before", "hour"]:
        if c[seg].isin(["nan", "None"]).all():
            continue
        groups = {
            str(k): summarise(g) for k, g in c.groupby(seg) if len(g) >= 500 and str(k) != "nan"
        }
        out["segments"][seg] = groups

    # --- trigger comparison: shoppers active in the window, first visit of each kind
    w = v[in_win & (v.buys == 0)].copy()
    keep = (e.kind.to_numpy() == VIEW) & (in_win & (v.buys == 0)).to_numpy()[vid]
    vv = pd.DataFrame({"visit": vid[keep], "product": e["product"].to_numpy()[keep]})
    top_view = vv.groupby(["visit", "product"]).size().rename("n").reset_index()
    top_view = top_view.sort_values("n", ascending=False).drop_duplicates("visit")
    top_view = top_view.set_index("visit")
    w["max_views"] = top_view.n.reindex(w.index, fill_value=0)
    kinds = {
        "cart abandoned": w.carts > 0,
        "browse abandoned (a product viewed 3+ times)": (w.carts == 0) & (w.max_views >= 3),
        "browsed (1-2 views per product)": (w.carts == 0) & (w.max_views.between(1, 2)),
    }
    active = v.loc[in_win, "user"].nunique()
    comp = {"active_shoppers": int(active)}
    leaf_of = e.drop_duplicates("product").set_index("product").leaf
    for name, mask in kinds.items():
        x = w[mask]
        x = x[~x.user.duplicated()]
        if name == "cart abandoned":
            f = focal[["case", "user", "end", "product", "leaf"]]
            res = c
        else:
            prod = top_view["product"].reindex(x.index).to_numpy()
            f = pd.DataFrame(
                {
                    "case": np.arange(len(x)),
                    "user": x.user.to_numpy(),
                    "end": x.end.to_numpy(),
                    "product": prod,
                    "leaf": leaf_of.reindex(prod).to_numpy(),
                }
            )
            res = follow_up(e, f)
            res["h_return"] = x.h_return.to_numpy()
        comp[name] = {
            "shoppers_per_1000_active": r(1000 * len(x) / active, 1),
            "bought_focal_7d": r(res.h_focal.notna().mean()),
            "bought_anything_7d": r(res.h_any.notna().mean()),
            "returned_7d": r(res.h_return.notna().mean()),
        }
    out["trigger_comparison"] = comp
    return out


if __name__ == "__main__":
    mode, path = sys.argv[1], Path(sys.argv[2])
    events, top_names, window = {"rees46": load_rees46, "retailrocket": load_retailrocket}[mode](
        path
    )
    cart = events[events.kind == CART][["user", "product", "t"]]
    buy = events[events.kind == BUY][["user", "product", "t"]]
    m = buy.merge(cart.groupby(["user", "product"]).t.min().rename("tc"), on=["user", "product"],
                  how="left")  # fmt: skip
    coverage = r((m.tc <= m.t).mean())
    result = {
        "dataset": mode,
        "purchases_with_prior_cart": coverage,
        **analyse(events, top_names, window),
    }
    print(json.dumps(result, indent=1))
