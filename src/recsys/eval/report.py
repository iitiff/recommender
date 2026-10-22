"""Render an EvalReport as markdown (for humans and experiment write-ups) and JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from recsys.eval.harness import EvalReport


def to_markdown(report: EvalReport, *, k: int | None = None) -> str:
    k = k or (10 if 10 in report.ks else report.ks[-1])
    info = report.info
    lines = [
        f"**Data:** `{info.get('fingerprint')}` | strategy: {info.get('strategy')} | "
        f"users evaluated: {info.get('users_evaluated')} "
        f"(skipped cold: {info.get('users_skipped_cold')}) | "
        f"train/test interactions: {info.get('train_interactions')}/"
        f"{info.get('test_interactions')} | recommendable items: "
        f"{info.get('recommendable_items')}",
        "",
        f"| Model | Recall@{k} | NDCG@{k} | Hit rate@{k} | Coverage@{k} | Gini@{k} "
        f"| Avg popularity@{k} | ms/user |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report.results:
        mt = r.metrics
        lines.append(
            f"| `{r.model_id}` | {mt[f'recall@{k}'].fmt()} | {mt[f'ndcg@{k}'].fmt()} "
            f"| {mt[f'hit_rate@{k}'].fmt()} | {r.aggregates[f'coverage@{k}']:.2f} "
            f"| {r.aggregates[f'gini@{k}']:.2f} | {mt[f'avg_popularity@{k}'].mean:.3f} "
            f"| {r.ms_per_user:.2f} |"
        )
    lines.append("")
    lines.append("Brackets are 95% bootstrap confidence intervals over users.")

    if report.comparisons:
        lines += [
            "",
            f"**Paired difference vs `{report.baseline}`** (model minus baseline; "
            "✔ = interval excludes zero)",
            "",
            f"| Model | ΔRecall@{k} | ΔNDCG@{k} | ΔHit rate@{k} |",
            "|---|---|---|---|",
        ]
        for model_id, diffs in report.comparisons.items():
            cells = []
            for name in (f"recall@{k}", f"ndcg@{k}", f"hit_rate@{k}"):
                d = diffs[name]
                cells.append(
                    f"{d.mean:+.3f} [{d.low:+.3f}, {d.high:+.3f}]{' ✔' if d.significant else ''}"
                )
            lines.append(f"| `{model_id}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


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
            }
            for r in report.results
        ],
        "comparisons": {
            model_id: {name: asdict(e) for name, e in diffs.items()}
            for model_id, diffs in report.comparisons.items()
        },
    }


def write_report(report: EvalReport, out_dir: Path, *, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / f"{name}.md", out_dir / f"{name}.json"
    md.write_text(to_markdown(report))
    js.write_text(json.dumps(to_dict(report), indent=2, default=str))
    return md, js
