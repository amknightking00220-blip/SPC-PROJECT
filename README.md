# SPC-Sentinel

A live Statistical Process Control (SPC) dashboard for a metal rod manufacturing dataset — built in Python/Streamlit, extending an earlier Power BI mini-project into a full data analyst portfolio piece.

## What it does

- Simulates live production data and auto-refreshes the dashboard as new batches arrive
- Computes real SPC control charts (X-bar, R, P, NP) with industry-standard formulas and flags out-of-control batches
- Runs hypothesis tests (chi-square, ANOVA, two-proportion z-test) to check whether observed differences across Machine/Operator/Shift are statistically real or just noise
- Predicts a single rod's defect risk from its own early in-process checkpoint readings, before the rod is finished, using a Random Forest that retrains on every refresh

## Tech stack

| Layer | Tool |
|---|---|
| App / UI | Streamlit |
| Data handling | pandas, numpy |
| Statistics | scipy.stats |
| ML | scikit-learn (RandomForestClassifier) |
| Charts | Plotly |

## Dataset

Rod-level CSV, tab-separated: `BatchID, Operator, Machine, Shift, Diameter_mm, Defective`
600 rods, 120 batches, 5 rods/batch. Diameter target = 10.0mm. Overall defect rate ≈ 26.3%.

## Project structure

```
spc_sentinel_app.py   ← Streamlit app (4 pages, wires everything below together)
spc.py                ← Control-chart math: batch_summary, xbar_r_limits, p_np_limits, flag_violations
stats.py               ← Hypothesis tests: chi_square_defect_association, anova_diameter_by_group, two_proportion_ztest
data.py                ← Dataset loader + live batch simulator: load_data, generate_new_batch, append_live_batch
rod_checkpoints.py     ← Generates simulated per-rod early checkpoint readings: generate_checkpoint_dataset
rod_predictor.py       ← Progressive per-rod defect prediction: train_progressive_checkpoint_predictors, predict_rod_risk
```

## App pages

1. **Live Overview** — KPI cards (total rods, batches, defects, defect rate, out-of-control batch count) + the most recently added batch
2. **Control Charts** — X-bar, R, P, NP charts, with out-of-control batches marked and each chart's exact formula/limits shown with live numbers
3. **Hypothesis Tests** — chi-square tests (defect rate vs. Machine/Operator/Shift), ANOVA (diameter vs. Machine/Operator), and an interactive two-proportion z-test between any two chosen categories
4. **Early Defect Risk** — retrains a Random Forest on the live dataset each refresh; shows Stage 1 (1 checkpoint) vs Stage 2 (2 checkpoints) accuracy/precision/recall, feature importances, and a live predictor you can feed readings into

## Process / methodology notes

- **Live data**: `append_live_batch()` generates new batches calibrated from the real dataset's own statistics (mean, std, defect rate), not arbitrary numbers, and appends them to a growing in-memory dataframe (`st.session_state`) on every Streamlit rerun.
- **Control limits**: standard Shewhart SPC formulas for subgroup size n=5 (A2 = 0.577, D3 = 0, D4 = 2.114; P/NP limits from the binomial 3-sigma formula).
- **Hypothesis tests**: standard significance threshold α = 0.05.
- **Leakage prevention**: `rod_predictor.py` explicitly asserts that no feature beyond the current checkpoint stage (including the final diameter or label) is present in the training features before fitting.
- **Checkpoint data is simulated**: Checkpoint 1/2 diameter readings are synthetic, calibrated from a real relationship in the data (defective rods average ~2.7x further from the 10.0mm target than good rods at ~0.048mm vs ~0.018mm). Checkpoint 3 (final) is the real, unaltered diameter from the dataset. This demonstrates the early-prediction concept; it has not been validated against real in-process sensor data, which was never recorded for this dataset.
- **Model retraining**: by design, the Early Defect Risk model retrains from scratch on every refresh against the current live dataset, rather than training once and scoring new rods against a frozen model — prioritizing freshness over metric stability.

## Key findings

- Defect rate shows **no significant association** with Machine, Operator, or Shift (all chi-square p > 0.3)
- Rod diameter **does significantly differ** by Machine (p ≈ 0.02) and Operator (p < 0.0001) — machines/operators are inconsistent with each other, but not enough to push defect rates up
- Early defect prediction recall improves from **~0.59 (Checkpoint 1 only) to ~0.66 (Checkpoints 1+2)** on the static dataset, confirming that more in-process information meaningfully improves early detection
- A bug in the original Power BI dashboard's KPI cards (showing 3 total defects / 3 out-of-control subgroups instead of the real 158 / 17) was identified and corrected in this version

## Running locally

```bash
pip install streamlit streamlit-autorefresh plotly pandas numpy scipy scikit-learn
streamlit run spc_sentinel_app.py
```

Requires `metal rods spc project.csv` (tab-separated, with spaces in the filename) in the same directory.

## Scope notes

Deliberately excludes Docker/Kubernetes, orchestrators, and live A/B testing infrastructure — a single-file Streamlit app was chosen over a multi-service architecture to prioritize depth on the statistics/ML over deployment complexity. An Isolation Forest anomaly-detection module (`ml.py`) was built and validated but is not wired into the current app scope.
