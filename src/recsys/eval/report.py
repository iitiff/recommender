"""Render an EvalReport as markdown (for humans and experiment write-ups) and JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from recsys.eval.harness import EvalReport

# Segments in journey order rather than alphabetical.
SEGMENT_ORDER = (
    "return",
    "discovery",
    "cold start",
    "new visitor",
    "browser",
    "abandoner",
    "1 order",
    "2+ orders",
    "bulk buyer",
    "no history (cold start)",
    "seen this visit",
    "seen in an earlier visit",
    "re-buy",
    "new item, browsed category",
    "new item, new category",
    "new item, category unknown",
    "no past order",
    "category of a past order",
    "new category vs past orders",
    "category unknown",
)


def _order(value: str) -> tuple[int, str]:
    return (SEGMENT_ORDER.index(value) if value in SEGMENT_ORDER else len(SEGMENT_ORDER), value)


def _small(x: float) -> str:
    """Popularity shares on big stores are tiny; don't round them to 0.000."""
    return f"{x:.3f}" if x == 0 or x >= 0.001 else f"{x:.1e}"


def to_markdown(report: EvalReport, *, k: int | None = None) -> str:
    k = k or (10 if 10 in report.ks else report.ks[-1])
    info = report.info
    unit = info.get("unit", "users")
    novelty = any(f"new_category_share@{k}" in r.metrics for r in report.results)
    lines = [
        f"**Data:** `{info.get('fingerprint')}` | strategy: {info.get('strategy')} | "
        f"{unit} evaluated: {info.get('users_evaluated')} "
        f"(skipped cold: {info.get('users_skipped_cold')}) | "
        f"train/test interactions: {info.get('train_interactions')}/"
        f"{info.get('test_interactions')} | recommendable items: "
        f"{info.get('recommendable_items')}",
        "",
        f"| Model | Recall@{k} | NDCG@{k} | Hit rate@{k} | MRR@{k} | Coverage@{k} | Gini@{k} "
        f"| Avg popularity@{k} | Avg list length |"
        + (f" New-category share@{k} |" if novelty else "")
        + f" ms/{unit[:-1] if unit.endswith('s') else unit} |",
        "|---|---|---|---|---|---|---|---|---|---|" + ("---|" if novelty else ""),
    ]
    for r in report.results:
        mt = r.metrics
        lines.append(
            f"| `{r.model_id}` | {mt[f'recall@{k}'].fmt()} | {mt[f'ndcg@{k}'].fmt()} "
            f"| {mt[f'hit_rate@{k}'].fmt()} | {mt[f'mrr@{k}'].fmt()} "
            f"| {r.aggregates[f'coverage@{k}']:.2f} "
            f"| {r.aggregates[f'gini@{k}']:.2f} | {_small(mt[f'avg_popularity@{k}'].mean)} "
            f"| {mt[f'list_length@{k}'].mean:.1f} |"
            + (f" {mt[f'new_category_share@{k}'].mean:.2f} |" if novelty else "")
            + f" {r.ms_per_user:.2f} |"
        )
    lines.append("")
    lines.append(f"Brackets are 95% bootstrap confidence intervals over {unit}.")

    if report.comparisons:
        lines += [
            "",
            f"**Paired difference vs `{report.baseline}`** (model minus baseline; "
            "✔ = interval excludes zero)",
            "",
            f"| Model | ΔRecall@{k} | ΔNDCG@{k} | ΔHit rate@{k} | ΔMRR@{k} |",
            "|---|---|---|---|---|",
        ]
        for model_id, diffs in report.comparisons.items():
            cells = []
            for name in (f"recall@{k}", f"ndcg@{k}", f"hit_rate@{k}", f"mrr@{k}"):
                d = diffs[name]
                cells.append(
                    f"{d.mean:+.3f} [{d.low:+.3f}, {d.high:+.3f}]{' ✔' if d.significant else ''}"
                )
            lines.append(f"| `{model_id}` | " + " | ".join(cells) + " |")

    lines += _segment_tables(report)
    return "\n".join(lines) + "\n"


def _segment_tables(report: EvalReport) -> list[str]:
    """One table per segment dimension: hit rate at the largest k, per model."""
    if not report.segment_counts:
        return []
    k = report.ks[-1]
    unit = report.info.get("unit", "users")
    lines = [
        "",
        f"**By segment** (Hit rate@{k} [95% CI]; columns are models in the order above)",
    ]
    names = [r.model_id.split("@")[0] for r in report.results]

    def label(model_id: str) -> str:
        """The bare name, unless several configurations of one model are compared."""
        name = model_id.split("@")[0]
        return model_id if names.count(name) > 1 else name

    for dim, counts in report.segment_counts.items():
        total = sum(counts.values())
        lines += [
            "",
            f"| {dim} | {unit} |" + "".join(f" `{label(r.model_id)}` |" for r in report.results),
            "|---|---|" + "---|" * len(report.results),
        ]
        for value in sorted(counts, key=_order):
            cells = []
            for r in report.results:
                e = r.segments.get(dim, {}).get(value, {}).get(f"hit_rate@{k}")
                cells.append(f" {e.fmt()} |" if e else " – |")
            lines.append(
                f"| {value} | {counts[value]:,} ({counts[value] / total:.0%}) |" + "".join(cells)
            )
    return lines


def to_dict(report: EvalReport) -> dict[str, Any]:
    return {
        "info": report.info,
        "ks": list(report.ks),
        "baseline": report.baseline,
        "results": [
            {
                "model_id": r.model_id,
                "metrics": {name: asdict(e) for name, e in r.metrics.items()},
                "aggregates": r.aggregates,
                "fit_seconds": r.fit_seconds,
                "ms_per_user": r.ms_per_user,
                "segments": {
                    dim: {
                        value: {name: asdict(e) for name, e in metrics.items()}
                        for value, metrics in by_value.items()
                    }
                    for dim, by_value in r.segments.items()
                },
            }
            for r in report.results
        ],
        "comparisons": {
            model_id: {name: asdict(e) for name, e in diffs.items()}
            for model_id, diffs in report.comparisons.items()
        },
        "segment_counts": report.segment_counts,
    }


def write_report(report: EvalReport, out_dir: Path, *, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / f"{name}.md", out_dir / f"{name}.json"
    md.write_text(to_markdown(report))
    js.write_text(json.dumps(to_dict(report), indent=2, default=str))
    return md, js
