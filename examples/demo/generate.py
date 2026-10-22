"""Generate the fictional demo catalog (deterministic).

Every producer and wine here is invented. The demo exists so the public engine
runs, tests, and demos without any real curated data, which lives in the
private site repo.

    uv run python examples/demo/generate.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

OUT = Path(__file__).parent
rng = random.Random(42)

# The niche we "sell" (recommendable) vs a neighbouring region whose wines are
# only used to learn taste (quiz, ratings), mirroring a real site that quizzes on
# a broad region but recommends within one.
REGIONS = {
    "Demo Valley": ["North Bench", "Cedar Hills", "River Flats", "Fog Gap", "Stone Ridge"],
    "Far Coast": ["Sea Bluff", "Sandy Hills"],
}
PRODUCERS = [
    "Alder Creek",
    "Bramble Hill",
    "Copperline",
    "Driftwood",
    "Elkhorn",
    "Foxglove",
    "Granite Bay",
    "Hollow Oak",
    "Ironwood",
    "Juniper Row",
    "Kestrel",
    "Larkspur",
]
# varietal -> (body, tannin, acidity, oak) centre, plus typical notes
VARIETALS = {
    "Cabernet Sauvignon": (
        (5, 5, 3, 4),
        ["blackcurrant", "cedar", "graphite", "mocha", "black cherry"],
    ),
    "Merlot": ((4, 3, 3, 3), ["plum", "black cherry", "cocoa", "bay leaf"]),
    "Pinot Noir": ((2, 2, 4, 2), ["red cherry", "raspberry", "forest floor", "rose"]),
    "Zinfandel": ((4, 3, 3, 3), ["blackberry jam", "black pepper", "raspberry", "clove"]),
    "Syrah": ((4, 4, 3, 3), ["blackberry", "black pepper", "smoked meat", "violet"]),
    "Petite Sirah": ((5, 5, 3, 3), ["blueberry", "black pepper", "dark chocolate"]),
    "Cabernet Franc": ((3, 3, 4, 2), ["red currant", "bell pepper", "graphite", "violet"]),
}
PAIRINGS = {
    "Cabernet Sauvignon": ["steak", "lamb", "aged cheddar"],
    "Merlot": ["roast chicken", "mushroom risotto", "pork loin"],
    "Pinot Noir": ["salmon", "duck", "mushroom risotto"],
    "Zinfandel": ["barbecue", "pizza", "burgers"],
    "Syrah": ["lamb", "barbecue", "sausages"],
    "Petite Sirah": ["barbecue", "short ribs", "aged cheddar"],
    "Cabernet Franc": ["roast chicken", "goat cheese", "pork loin"],
}
STYLE_WORDS = {
    "body": {1: "very light", 2: "light", 3: "medium-bodied", 4: "full", 5: "very full"},
}


def clamp(v: int) -> int:
    return max(1, min(5, v))


def main() -> None:
    items, offers = [], []
    names = list(VARIETALS)
    for n in range(40):
        producer = PRODUCERS[n % len(PRODUCERS)]
        varietal = names[rng.randrange(len(names))]
        (body, tannin, acidity, oak), note_pool = VARIETALS[varietal]
        body, tannin = (
            clamp(body + rng.choice([-1, 0, 0, 1])),
            clamp(tannin + rng.choice([-1, 0, 1])),
        )
        acidity, oak = clamp(acidity + rng.choice([-1, 0, 1])), clamp(oak + rng.choice([-1, 0, 1]))
        blend = [varietal]
        if varietal == "Cabernet Sauvignon" and rng.random() < 0.4:
            blend.append(rng.choice(["Merlot", "Cabernet Franc"]))
        notes = rng.sample(note_pool, k=min(3, len(note_pool)))
        if oak >= 4:
            notes.append("vanilla")
        region = "Demo Valley" if n < 30 else "Far Coast"
        subregion = rng.choice(REGIONS[region])
        item_id = f"demo-{producer.lower().replace(' ', '-')}-{n:02d}"
        items.append(
            {
                "item_id": item_id,
                "title": f"{producer} {subregion} {varietal}",
                "producer": producer,
                "wine_type": "red",
                "varietals": ";".join(blend),
                "region": region,
                "subregion": subregion,
                "body": body,
                "tannin": tannin,
                "acidity": acidity,
                "oak": oak,
                "flavor_notes": ";".join(notes),
                "food_pairings": ";".join(rng.sample(PAIRINGS[varietal], k=2)),
                "description": (
                    f"A {STYLE_WORDS['body'][body]} {varietal} from the fictional "
                    f"{subregion} district, showing {', '.join(notes[:-1])} and {notes[-1]}."
                ),
                "image_url": "",
            }
        )
        if region != "Demo Valley":
            continue  # quiz-only wines: rated, never sold here, so no offers
        base = rng.choice([18, 24, 32, 45, 60, 85, 120, 140])
        for vintage in rng.sample(["2019", "2020", "2021", "2022"], k=rng.choice([1, 1, 2])):
            offers.append(
                {
                    "item_id": item_id,
                    "retailer": rng.choice(["demo-shop", "demo-cellar"]),
                    "variant": vintage,
                    "price": round(base * rng.uniform(0.9, 1.1), 2),
                    "currency": "USD",
                    "url": "",
                    "in_stock": "true" if rng.random() > 0.1 else "false",
                    "ships_to": "" if rng.random() > 0.3 else "CA;OR;WA",
                }
            )

    for name, rows in (("items.csv", items), ("offers.csv", offers)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {len(items)} items, {len(offers)} offers to {OUT}")


if __name__ == "__main__":
    main()
