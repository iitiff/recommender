"""Lifecycle view of RetailRocket: how buyers become repeat buyers, and whether
repeat purchases stay in the same category or expand to new ones.

Usage: python -I lifecycle.py <prepared_dir>
An "order" = a session containing at least one purchase.
"""

import sys
from pathlib import Path

import pandas as pd

d = Path(sys.argv[1])
ev = pd.read_csv(d / "interactions.csv", dtype={"user_id": str, "item_id": str})
ev["t"] = pd.to_datetime(ev.occurred_at, utc=True, format="ISO8601")
items = pd.read_csv(d / "catalog" / "items.csv", dtype={"item_id": str, "category_path": str})
top = items.set_index("item_id").category_path.dropna().str.split(";").str[0]
leaf = items.set_index("item_id").category.dropna().astype(str)

# -- lifecycle stages -----------------------------------------------------------
per_user = ev.groupby("user_id").agg(
    carts=("kind", lambda k: (k == "add_to_cart").sum()),
    buys=("kind", lambda k: (k == "purchase").sum()),
)
orders = ev[ev.kind == "purchase"].groupby(["user_id", "session_id"]).t.min().reset_index()
n_orders = orders.groupby("user_id").size()
per_user["orders"] = n_orders.reindex(per_user.index).fillna(0).astype(int)
stage = pd.cut(per_user.orders, [-1, 0, 1, 2, 10**9], labels=["0", "1", "2", "3+"]).astype(str)
stage[(per_user.orders == 0) & (per_user.carts > 0)] = "0 (carted)"
stage[(per_user.orders == 0) & (per_user.carts == 0)] = "0 (browse only)"
total_buys = per_user.buys.sum()
print("== Lifecycle stages")
for s in ["0 (browse only)", "0 (carted)", "1", "2", "3+"]:
    m = stage == s
    print(f"{s:<16} users {m.sum():>9,} ({m.mean():6.2%})   share of purchased items {per_user.buys[m].sum() / total_buys:6.1%}")

# -- first → second order ----------------------------------------------------------
orders = orders.sort_values(["user_id", "t"])
orders["n"] = orders.groupby("user_id").cumcount()
first = orders[orders.n == 0].set_index("user_id")
second = orders[orders.n == 1].set_index("user_id")
buyers = len(first)
print(f"\n== First → second order\nbuyers: {buyers:,}; made a 2nd order: {len(second) / buyers:.1%}")
gap = (second.t - first.loc[second.index].t).dt.total_seconds() / 86400
print("days from 1st to 2nd order: " + " · ".join(f"p{int(q*100)} {gap.quantile(q):.1f}" for q in (0.25, 0.5, 0.75, 0.9)))

bought = ev[ev.kind == "purchase"][["user_id", "session_id", "item_id"]]
cats_top = bought.assign(c=bought.item_id.map(top)).dropna().groupby("session_id").c.apply(set)
cats_leaf = bought.assign(c=bought.item_id.map(leaf)).dropna().groupby("session_id").c.apply(set)
items_in = bought.groupby("session_id").item_id.apply(set)


def compare(a_sess, b_sess):
    same_item = same_leaf = same_top = known = 0
    for a, b in zip(a_sess, b_sess, strict=True):
        if items_in.get(a, set()) & items_in.get(b, set()):
            same_item += 1
        if a in cats_leaf.index and b in cats_leaf.index and a in cats_top.index and b in cats_top.index:
            known += 1
            same_leaf += bool(cats_leaf[a] & cats_leaf[b])
            same_top += bool(cats_top[a] & cats_top[b])
    return same_item, same_leaf, same_top, known


pairs = second.join(first, rsuffix="_1")
si, sl, st, k = compare(pairs.session_id_1, pairs.session_id)
print(f"2nd order vs 1st (n={len(pairs):,}; categories known for {k:,}):")
print(f"  re-bought the same item {si / len(pairs):.1%}")
print(f"  same leaf category {sl / k:.1%} | same top-level category {st / k:.1%} | new top-level category {1 - st / k:.1%}")

# -- does expanding to a new category go with a 3rd order? (observational) --------
third = set(orders[orders.n == 2].user_id)
rows = []
for u, r in pairs.iterrows():
    a, b = r.session_id_1, r.session_id
    if a in cats_top.index and b in cats_top.index:
        rows.append((not (cats_top[a] & cats_top[b]), u in third))
df = pd.DataFrame(rows, columns=["expanded", "third"])
print("\n== After two orders: P(3rd order)")
for flag, label in ((False, "both orders in the same top-level category"), (True, "2nd order in a NEW top-level category")):
    g = df[df.expanded == flag]
    print(f"  {label:<44} n={len(g):>5,}  P(3rd order) {g.third.mean():.1%}")
