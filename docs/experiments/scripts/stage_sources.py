"""Which candidate source works for which finer journey segment (RetailRocket).

Re-segments the next-add-to-cart cases of `recsys evaluate --split next` with
finer state, computed from each shopper's full history before the cart:
- journey: relationship (new / browser / abandoner / 1 order / 2+ orders /
  bulk) × momentum (time since their previous visit)
- visit depth: events earlier in this visit
- visit focus: departments browsed earlier in this visit
- open cart: a cart-without-purchase visit since their last order

Usage: uv run python docs/experiments/scripts/stage_sources.py data/retailrocket
"""

from __future__ import annotations

import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from recsys.candidates import AvailabilityTimeline
from recsys.catalog import load_catalog
from recsys.core.types import InteractionKind
from recsys.eval import evaluate, load_interactions_csv, to_markdown
from recsys.eval.split import groups_from_facet, next_event_split
from recsys.lifecycle import BULK_ITEMS, orders
from recsys.models import (
    CoVisitationRecommender,
    PopularityRecommender,
    RecentlyViewedRecommender,
    TrendingRecommender,
)

P, C = InteractionKind.PURCHASE, InteractionKind.ADD_TO_CART


def momentum(hours: float) -> str:
    if hours < 1:
        return "<1h"
    if hours < 24:
        return "1h–1d"
    if hours < 24 * 7:
        return "1–7d"
    return ">7d"


def describe(before, session_id, now, groups) -> dict[str, str]:
    earlier = [x for x in before if x.session_id != session_id]
    visit = [x for x in before if x.session_id == session_id]
    purchased = sum(1 for x in before if x.kind == P)
    n_orders = orders(before)
    if purchased > BULK_ITEMS:
        rel = "bulk"
    elif n_orders >= 2:
        rel = "2+ orders"
    elif n_orders == 1:
        rel = "1 order"
    elif any(x.kind == C for x in earlier):
        rel = "abandoner"
    elif earlier:
        rel = "browser"
    else:
        rel = "new"
    journey = rel
    if earlier:
        visit_start = visit[0].occurred_at if visit else now
        gap = (visit_start - earlier[-1].occurred_at).total_seconds() / 3600
        journey = f"{rel}, back after {momentum(gap)}"
    last_order = max((x.occurred_at for x in earlier if x.kind == P), default=None)
    open_cart = any(
        x.kind == C and (last_order is None or x.occurred_at > last_order) for x in earlier
    )
    depth = len(visit)
    focus = len({groups.get(x.item_id) for x in visit if x.item_id in groups})
    return {
        "journey": journey,
        "relationship": rel,
        "visit depth": "0 (cart is first event)"
        if depth == 0
        else "1"
        if depth == 1
        else "2–4"
        if depth < 5
        else "5+",
        "visit focus": "no category info"
        if not visit
        else f"{min(focus, 2)}{'+' if focus >= 2 else ''} departments",
        "open cart": "yes" if open_cart else "no",
    }


def main(prepared: Path) -> None:
    catalog = load_catalog(prepared / "catalog")
    data = load_interactions_csv(prepared / "interactions.csv")
    groups = groups_from_facet(catalog.items, "department")
    split = next_event_split(
        catalog.items,
        data,
        test_fraction=0.1,
        item_groups=groups,
        availability=AvailabilityTimeline.load_csv(prepared / "availability.csv"),
    )
    by_user = defaultdict(list)
    for x in sorted(data, key=lambda x: x.occurred_at):
        by_user[x.user_id].append(x)
    times = {u: [x.occurred_at for x in xs] for u, xs in by_user.items()}
    segments = {}
    for case, ctx in split.contexts.items():
        history = by_user[ctx.user_id]
        before = history[: bisect_left(times[ctx.user_id], ctx.now)]
        seg = describe(before, ctx.session_id, ctx.now, groups)
        seg["next action"] = split.segments[case]["next action"]
        segments[case] = seg

    # What shoppers carted next, by segment (no model involved).
    for dim in ("journey", "visit depth", "visit focus", "open cart"):
        mix = defaultdict(Counter)
        for seg in segments.values():
            mix[seg[dim]][seg["next action"]] += 1
        print(f"\n### Next action mix by {dim}\n")
        actions = [
            "no history (cold start)",
            "seen this visit",
            "seen in an earlier visit",
            "re-buy",
            "new item, browsed category",
            "new item, new category",
        ]
        print(f"| {dim} | cases | " + " | ".join(actions) + " |")
        print("|---|---|" + "---|" * len(actions))
        for value, counts in sorted(mix.items(), key=lambda kv: -sum(kv[1].values())):
            n = sum(counts.values())
            print(
                f"| {value} | {n:,} | " + " | ".join(f"{counts[a] / n:.0%}" for a in actions) + " |"
            )

    models = [
        PopularityRecommender(),
        TrendingRecommender(),
        RecentlyViewedRecommender(),
        CoVisitationRecommender(),
    ]
    report = evaluate(models, replace(split, segments=segments), ks=(20,), n_resamples=300)
    print("\n" + to_markdown(report, k=20))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
