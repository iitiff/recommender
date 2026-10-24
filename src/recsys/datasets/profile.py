"""Re-engagement profile of a prepared, sessionized e-commerce dataset.

Answers "after a visit ends without a purchase, who comes back, how fast, and
do they buy?". This decides which triggers are worth a push or email
(abandoned cart vs abandoned browse vs post-purchase) and when to send.

These are *organic* return rates: they show who comes back on their own,
not who a message would persuade. Measuring persuasion needs a randomized
holdout (uplift); see docs/ROADMAP.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recsys.datasets.retailrocket import _pandas

DAY = 86_400.0


@dataclass(frozen=True)
class TriggerRow:
    name: str
    sessions: int
    return_7d: float
    return_30d: float
    buy_next_7d: float


@dataclass(frozen=True)
class Profile:
    counts: dict[str, int]
    returning_users: float
    triggers: list[TriggerRow]
    return_gap_days: dict[str, float]  # quantiles of time to next visit
    cart_item_share: float | None  # converting abandoned carts that bought a carted item
    converting_carts: int

    def to_markdown(self) -> str:
        c = self.counts
        lines = [
            f"- {c['interactions']:,} events ({c['view']:,} views, "
            f"{c['add_to_cart']:,} add-to-carts, {c['purchase']:,} purchases), "
            f"{c['users']:,} users, {c['sessions']:,} sessions, {c['items']:,} items",
            f"- Users with 2+ sessions (returning): {self.returning_users:.1%}",
            "",
            "| After a session that ended like this… | Sessions | Return ≤7d | Return ≤30d "
            "| Buy on next visit ≤7d |",
            "|---|---|---|---|---|",
        ]
        for t in self.triggers:
            lines.append(
                f"| {t.name} | {t.sessions:,} | {t.return_7d:.1%} | {t.return_30d:.1%} "
                f"| {t.buy_next_7d:.2%} |"
            )
        q = self.return_gap_days
        lines += [
            "",
            f"Time to return (days): 25% {q['p25']:.2f} · median {q['p50']:.2f} · "
            f"75% {q['p75']:.2f} · 90% {q['p90']:.2f}",
        ]
        if self.cart_item_share is not None:
            lines.append(
                f"Abandoned carts converting on the next visit (≤7d, n={self.converting_carts:,}): "
                f"{self.cart_item_share:.0%} bought an item that was in the cart."
            )
        return "\n".join(lines) + "\n"


def profile(prepared_dir: Path) -> Profile:
    pd = _pandas()
    ev = pd.read_csv(
        Path(prepared_dir) / "interactions.csv", dtype={"user_id": str, "item_id": str}
    )
    when = pd.to_datetime(ev.occurred_at, utc=True, format="ISO8601")
    # Seconds since epoch via Timedelta arithmetic, which is correct whatever the
    # datetime resolution (pandas may use s/ms/us/ns; astype("int64") is not).
    ev["ts"] = (when - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)
    kinds = ev.kind.value_counts()
    is_view, is_cart, is_buy = (ev.kind == k for k in ("view", "add_to_cart", "purchase"))
    s = (
        ev.assign(v=is_view, c=is_cart, b=is_buy)
        .groupby("session_id")
        .agg(
            user=("user_id", "first"),
            start=("ts", "min"),
            end=("ts", "max"),
            views=("v", "sum"),
            carts=("c", "sum"),
            buys=("b", "sum"),
        )
        .sort_values(["user", "start"])
    )
    nxt = s.groupby("user").shift(-1)
    s["gap_days"] = (nxt.start - s.end) / DAY
    s["next_buys"] = nxt.buys
    s["next_session"] = s.index.to_series().groupby(s.user).shift(-1)

    def row(name: str, mask) -> TriggerRow:
        g = s[mask]
        n = max(len(g), 1)
        return TriggerRow(
            name,
            len(g),
            float((g.gap_days <= 7).sum() / n),
            float((g.gap_days <= 30).sum() / n),
            float(((g.gap_days <= 7) & (g.next_buys > 0)).sum() / n),
        )

    no_buy = s.buys == 0
    triggers = [
        row("Viewed 1 product, left", no_buy & (s.carts == 0) & (s.views == 1)),
        row(
            "Viewed 3+ products, no cart (abandoned browse)",
            no_buy & (s.carts == 0) & (s.views >= 3),
        ),
        row("Added to cart, didn't buy (abandoned cart)", no_buy & (s.carts > 0)),
        row("Bought something (post-purchase)", s.buys > 0),
    ]
    gaps = s.gap_days.dropna()
    quantiles = {
        f"p{int(q * 100)}": float(gaps.quantile(q)) if len(gaps) else 0.0
        for q in (0.25, 0.5, 0.75, 0.9)
    }

    carted = ev[is_cart].groupby("session_id").item_id.apply(set)
    bought = ev[is_buy].groupby("session_id").item_id.apply(set)
    conv = s[no_buy & (s.carts > 0) & (s.gap_days <= 7) & (s.next_buys > 0)]
    same = sum(
        bool(carted.get(i, set()) & bought.get(n, set()))
        for i, n in zip(conv.index, conv.next_session, strict=True)
    )

    counts = {
        "interactions": len(ev),
        "view": int(kinds.get("view", 0)),
        "add_to_cart": int(kinds.get("add_to_cart", 0)),
        "purchase": int(kinds.get("purchase", 0)),
        "users": int(ev.user_id.nunique()),
        "sessions": len(s),
        "items": int(ev.item_id.nunique()),
    }
    returning = float((s.groupby("user").size() > 1).mean()) if len(s) else 0.0
    return Profile(
        counts, returning, triggers, quantiles, same / len(conv) if len(conv) else None, len(conv)
    )
