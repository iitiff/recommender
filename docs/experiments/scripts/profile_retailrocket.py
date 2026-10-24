"""Profile RetailRocket events for re-engagement (who comes back, after what, and buys what).

Usage: python -I profile_retailrocket.py <events.csv>
Sessions = a visitor's events with gaps under 30 minutes (a common web-analytics convention).
"""

import sys

import pandas as pd

GAP_MS = 30 * 60 * 1000
DAY_MS = 24 * 3600 * 1000

ev = pd.read_csv(sys.argv[1])
ev = ev.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
print("== Events")
print(ev["event"].value_counts().to_string())
start, end = (
    pd.to_datetime(ev.timestamp.min(), unit="ms"),
    pd.to_datetime(ev.timestamp.max(), unit="ms"),
)
print(f"range: {start:%Y-%m-%d} → {end:%Y-%m-%d} ({(end - start).days} days)")
print(f"visitors: {ev.visitorid.nunique():,}   items: {ev.itemid.nunique():,}")

new_session = (ev.visitorid != ev.visitorid.shift()) | (ev.timestamp.diff() > GAP_MS)
ev["session"] = new_session.cumsum()
s = ev.groupby("session").agg(
    visitor=("visitorid", "first"),
    start=("timestamp", "min"),
    end=("timestamp", "max"),
    n=("event", "size"),
    views=("event", lambda e: (e == "view").sum()),
    carts=("event", lambda e: (e == "addtocart").sum()),
    buys=("event", lambda e: (e == "transaction").sum()),
)
print("\n== Sessions")
print(f"sessions: {len(s):,}   events/session median: {s.n.median():.0f}")
spv = s.groupby("visitor").size()
print(f"visitors with 2+ sessions (returning): {(spv > 1).mean():.1%}")

s = s.sort_values(["visitor", "start"])
s["next_start"] = s.groupby("visitor").start.shift(-1)
s["gap_days"] = (s.next_start - s.end) / DAY_MS
s["next_buys"] = s.groupby("visitor").buys.shift(-1)


def funnel(name, mask):
    g = s[mask]
    if len(g) == 0:
        return
    ret7 = (g.gap_days <= 7).mean()
    ret30 = (g.gap_days <= 30).mean()
    buy_next = ((g.gap_days <= 7) & (g.next_buys > 0)).mean()
    print(
        f"{name:<38} n={len(g):>9,}  return ≤7d {ret7:6.1%}  ≤30d {ret30:6.1%}  "
        f"buy on next visit ≤7d {buy_next:6.2%}"
    )


print("\n== Re-engagement triggers (what happens after a session ends without buying)")
no_buy = s.buys == 0
funnel("all non-buying sessions", no_buy)
funnel("browse only, 1 view", no_buy & (s.carts == 0) & (s.views == 1))
funnel("browse only, 3+ views (abandoned browse)", no_buy & (s.carts == 0) & (s.views >= 3))
funnel("added to cart, no purchase (abandoned cart)", no_buy & (s.carts > 0))
funnel("sessions that bought (post-purchase)", s.buys > 0)

ret = s[s.gap_days.notna()]
print("\n== Time to return (returning sessions)")
print(ret.gap_days.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(2).to_string())

# For abandoned carts that convert on the next visit within 7 days: same item, or something else?
cart_items = ev[ev.event == "addtocart"].groupby("session").itemid.apply(set)
buy_items = ev[ev.event == "transaction"].groupby("session").itemid.apply(set)
s["next_session"] = s.reset_index().groupby("visitor")["session"].shift(-1).values
ab = s[no_buy & (s.carts > 0) & (s.gap_days <= 7) & (s.next_buys > 0)]
same = sum(
    bool(cart_items.get(i, set()) & buy_items.get(int(n), set()))
    for i, n in zip(ab.index, ab.next_session, strict=True)
)
print("\n== Abandoned carts that convert on the next visit (≤7 days)")
print(f"n={len(ab):,}; bought an item that was in the abandoned cart: {same / max(len(ab), 1):.0%}")
