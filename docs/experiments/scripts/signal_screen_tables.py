"""Turn signal_screen.py outputs (one JSON per dataset) into CSV tables.

Writes, into OUT_DIR:
  population.csv          rows, customers, retention, CLTV spread, stage counts
  univariate_screen.csv   feature x target: Spearman, mutual information, coverage
  incremental_lift.csv    RFM vs each family vs all, overall and by lifecycle stage
  redundancy.csv          feature clusters with |Spearman| > 0.7
  shap_importance.csv     mean |SHAP| per feature (all-signal model)
  shap_thresholds.csv     mean SHAP by feature decile (top 10 features)
  stability.csv           |Spearman| with CLTV per cutoff, and rank agreement

All values are aggregates; suppressed cells stay empty.

Usage:
  python -I signal_screen_tables.py OUT_DIR hm.json cdnow.json ...
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def write(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main(out: Path, files: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pop, uni, inc, red, imp, thr, stab = [], [], [], [], [], [], []
    for f in files:
        name = Path(f).stem
        d = json.loads(Path(f).read_text())
        p = d["population"]
        pop.append([name, p["rows"], p["customers"], ";".join(p["cutoffs"]), p["retained"],
                    p["cltv_mean"], p["cltv_median"], p["cltv_top10pct_share"],
                    ";".join(f"{k}={v}" for k, v in p["stages"].items()),
                    d["notes"].get("W"), d["notes"].get("H")])  # fmt: skip
        for feat, u in d["univariate"].items():
            uni.append([name, feat, u["family"], u["coverage"], u.get("spearman_cltv"),
                        u.get("spearman_retained"), u.get("mi_cltv"), u.get("mi_retained"),
                        bool(u.get("suppressed"))])  # fmt: skip
        stages = [("all", d["incremental"])] + [
            (s, v["incremental"]) for s, v in d["by_stage"].items()
        ]
        for seg, models in stages:
            for model, m in models.items():
                cis = [
                    *(m.get("delta_r2_ci") or [None] * 2),
                    *(m.get("delta_auc_ci") or [None] * 2),
                ]
                inc.append([name, seg, model, m.get("r2_log_cltv"), m.get("norm_gini"),
                            m.get("top_decile_capture"), m.get("auc_retained"), *cis])  # fmt: skip
        for i, cluster in enumerate(d["redundant_clusters"], 1):
            red.append([name, i, ";".join(cluster)])
        for feat, v in d["shap"]["mean_abs_shap"].items():
            imp.append([name, feat, v])
        for feat, rows in d["shap"]["by_decile"].items():
            for k, (lo, hi, s) in enumerate(rows, 1):
                thr.append([name, feat, "missing" if lo == "missing" else k,
                            None if lo == "missing" else lo, hi, s])  # fmt: skip
        st = d.get("stability")
        if st:
            for cut, per in st["per_cutoff_spearman_cltv"].items():
                for feat, v in per.items():
                    stab.append([name, cut, feat, v, st["rank_agreement_spearman"],
                                 st["top10_overlap"]])  # fmt: skip
    write(out / "population.csv", ["dataset", "rows", "customers", "cutoffs", "retained",
          "cltv_mean", "cltv_median", "cltv_top10pct_share", "stages", "obs_days",
          "outcome_days"], pop)  # fmt: skip
    write(out / "univariate_screen.csv", ["dataset", "feature", "family", "coverage",
          "spearman_cltv", "spearman_retained", "mi_cltv", "mi_retained", "suppressed"],
          uni)  # fmt: skip
    write(out / "incremental_lift.csv", ["dataset", "segment", "model", "r2_log_cltv",
          "norm_gini", "top_decile_capture", "auc_retained", "delta_r2_lo", "delta_r2_hi",
          "delta_auc_lo", "delta_auc_hi"], inc)  # fmt: skip
    write(out / "redundancy.csv", ["dataset", "cluster", "features"], red)
    write(out / "shap_importance.csv", ["dataset", "feature", "mean_abs_shap"], imp)
    write(out / "shap_thresholds.csv", ["dataset", "feature", "decile", "feature_min",
          "feature_max", "mean_shap"], thr)  # fmt: skip
    write(out / "stability.csv", ["dataset", "cutoff", "feature", "abs_spearman_cltv",
          "rank_agreement", "top10_overlap"], stab)  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
