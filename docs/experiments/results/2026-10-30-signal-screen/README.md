# Signal screen results (2026-10-30)

Aggregate tables behind `docs/experiments/2026-10-30-signal-screen.md`:
which behavioral signals predict future customer value (CLTV) beyond RFM,
on 8 public datasets.

## Summary

| Signal | Verdict | Why |
|---|---|---|
| Order count over the whole history | Strongest covariate | Top SHAP feature in H&M, CDNOW, Online Retail; +0.057 R² over 90-day RFM in H&M |
| Gap ratio (days since the last order ÷ the customer's usual gap) | Best trigger | Value falls past ~2–3.5× the customer's own gap, in 4 datasets |
| Visit engagement and recent-visit share | Biggest lift where available | +0.10–0.11 AUC; new customers 0.53–0.59 → 0.61–0.67; prospects 0.69–0.74 |
| Cart adds | Actionable, target by uplift | +0.06–0.08 AUC; redundant with abandoned-cart visits |
| Discount, coupon and campaign history | Drop for now | Correlates alone, adds nothing beyond spend (campaigns were targeted) |
| Returns (B2B) | Not a churn signal | Goes with higher value (ρ +0.27) |
| Late deliveries, reviews, stock-outs | Inconclusive | Outcome too rare in the public data |

## Files

| File | Contents |
|---|---|
| `population.csv` | Rows, customers, cutoffs, retention, CLTV spread, lifecycle stage counts, window lengths |
| `univariate_screen.csv` | Per feature: family, coverage, Spearman and mutual information with CLTV and retention |
| `incremental_lift.csv` | R² on log CLTV, normalized Gini, top-decile capture and AUC for RFM, RFM + each family and all signals. Overall rows have 95% bootstrap intervals for the lift over RFM; per-stage rows don't |
| `redundancy.csv` | Feature clusters with \|Spearman\| > 0.7 |
| `shap_importance.csv` | Mean \|SHAP\| per feature in the all-signal model |
| `shap_thresholds.csv` | Mean SHAP by feature decile (top 10 features per dataset), for trigger thresholds |
| `stability.csv` | \|Spearman\| with CLTV at each cutoff, and rank agreement between cutoffs |

## Rules

- **Aggregates only.** No customer-level rows. Cells under 50 customers are
  suppressed (`suppressed = True`, empty values).
- **The datasets themselves are not in the repo.** Their licenses are in
  `docs/DATASETS.md`.
- **Monetary units are each dataset's own.** H&M prices are scaled; OTTO and
  RetailRocket count items.

## Regenerate

```bash
uv run --no-project --with pandas --with pyarrow --with scikit-learn --with lightgbm \
  --with scipy --with pyreadr python -I docs/experiments/scripts/signal_screen.py \
  DATASET PATHS... > DATASET.json   # one per dataset, see the script's docstring
python -I docs/experiments/scripts/signal_screen_tables.py \
  docs/experiments/results/2026-10-30-signal-screen *.json
```
