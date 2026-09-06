# Project Blueprint (v2 — Simplified): SPC-Sentinel
### One Streamlit App. Live Data. Automation. Testing. Early-Warning ML.

---

## 0. The Pitch

> **SPC-Sentinel** is a live Statistical Process Control dashboard for manufacturing quality. It auto-updates as new batches arrive, flags out-of-control processes automatically, tests whether observed differences (by machine/operator/shift) are statistically real, and predicts — from just the first couple of rods in a batch — whether that batch is likely to end up defective, before production finishes.

That last sentence is your strongest talking point. Lead with it.

---

## 1. Already Done (Phase 1 — Power BI)

- ✅ 600-rod / 120-batch dataset, machine/operator/shift dimensions
- ✅ X-bar, R, P, NP control charts
- ✅ SQL root-cause breakdown
- ✅ Static Power BI dashboard

Everything below is the "engineering layer" on top — but kept simple, no software architecture required.

---

## 2. The Whole App, in Five Files

```
spc_app/
├── app.py            ← Streamlit app + pages (sidebar nav)
├── spc.py            ← control-chart math (plain functions)
├── stats.py           ← chi-square / ANOVA hypothesis tests
├── ml.py              ← anomaly detection + early defect prediction
├── data.py            ← loads dataset, generates a new "live" batch
└── tests/
    └── test_spc.py    ← ~8 simple assert-based tests
```

No orchestrator, no Docker, no multi-service setup, no background processes. One app, one deployment, one thing to keep running. This is deliberate — a lean, fully-working project beats a sprawling half-working one, every time.

---

## 3. The Five Ingredients

### ① Live Data
`data.py` has one function, `generate_new_batch()`, that creates a new realistic batch row (with an occasional small drift toward more defects, so the demo has something interesting to show). Add `streamlit-autorefresh` (one line) so the app calls this automatically every ~15–30 seconds — new data appears with zero manual steps.

### ② Automation
The moment a new batch appears, control limits, chart flags, and hypothesis-test results all recompute — because the app just re-runs top to bottom on every refresh. No scheduler needed to make this true. This is a fair, honest use of the word "automated": nothing needs a person to click anything.

### ③ Testing
`tests/test_spc.py`, plain `pytest` asserts:
- control limits match known values on a fixed sample batch
- defect rate calculation is correct
- a synthetic clearly-bad batch actually gets flagged
- generated data has no missing/negative values
- defect-prediction model accuracy stays above a floor (e.g. 75%)
- chi-square test returns a sane p-value on known data
- (optional) early-prediction model doesn't use any "future" columns — a quick assert on the feature list itself

This is closer to "sanity-checking my numbers," which you already do instinctively — just written down as code.

### ④ Better UI
Reuse the multi-theme CSS approach from your Nassau Candy dashboard. Sidebar navigation between pages:

```
📊 Live Overview        — KPI cards, latest batch, auto-refreshing
📈 Control Charts       — X-bar / R / P / NP, live-updating
🔬 Hypothesis Tests     — chi-square / ANOVA results, plain-language verdicts
🤖 Early Defect Risk    — the centerpiece (see below)
🔍 Root Cause           — your existing SQL breakdown, interactive filters
```

### ⑤ ML — two models, different jobs

**Model A — Anomaly Detection (Isolation Forest).** Same pattern as Nassau Candy. Flags a batch's diameter/variation as unusual even before it crosses a formal control limit — an early drift signal layered on top of the classic SPC charts.

**Model B — Early Defect Prediction (the centerpiece).** This is the one worth building carefully.

---

## 4. Early Defect Prediction — Done Properly

**The idea:** predict whether a batch will end up defective using *only the first 1–2 rods measured in that batch* — before the rest are even produced. This is prevention, not just detection.

**Train/test setup:**
- **One row per batch.** Features = early signals only (rod 1 diameter, rod 2 diameter, machine, operator, shift). Label = whether the *full* batch ended up defective.
- **Split:** hold out ~20% of batches the model never trains on, to check it generalizes rather than memorizes.
- **Avoid data leakage** — the single most important thing to get right. Do NOT include: total defects, final defect rate, later rods' readings, or anything only knowable after the batch finished. If any of these sneak into the features, the model will look great on paper and be useless in practice. Explicitly list your feature set and double-check none of it "sees the future" before training.
- **Metric that matters most: recall on the "will be defective" class**, not raw accuracy. Missing a genuinely bad batch is expensive (wasted material); a false alarm just costs a double-check. State this trade-off explicitly — it's a judgment call, and naming it is what makes the model choice defensible.

**What the page shows:**
- Enter/simulate the first 1–2 rod readings for a new batch → live predicted risk %
- Feature importances underneath, in plain language ("Machine M3 + Afternoon shift are the biggest risk drivers") — turns a bare prediction into a recommendation
- Honest reporting of train/test accuracy *and* recall, not just the flattering number
- A short note on what features were used and why leakage was avoided — this one paragraph does more for credibility than anything else on the page

---

## 5. Hypothesis Testing — Applied to What You Already Have

No new data needed — these run directly on your existing SQL breakdown:

- **Chi-square test of independence** — is defect rate actually associated with machine/operator/shift, or could the differences (e.g. M3: 0.24 vs M1: 0.29) just be noise?
- **One-way ANOVA** — does average rod diameter genuinely differ across M1/M2/M3?
- **Two-proportion z-test** — is Operator B's defect rate significantly different from Operator A's, specifically?

This upgrades "M3 looks worse" into "M3 is statistically significantly worse (p = 0.03)" — a stronger, more defensible claim, and it's just `scipy.stats` on data you've already got.

---

## 6. What's Intentionally Left Out (and why)

- **True A/B testing** — needs live randomized assignment and a running experiment, which needs the orchestrator/engineering machinery we cut. Rather than fake it on historical data (which isn't a real A/B test and would be shaky to defend), it's cleanly out of scope for this version. Mention it as "future work" if asked.
- **Docker / multi-service deployment** — one Streamlit app deployed to Streamlit Community Cloud tells the same story with a fraction of the moving parts.
- **A scheduler/orchestrator process** — auto-refresh on the app itself covers "automated" honestly, without a second process to maintain.

Naming these as deliberate scope decisions (rather than omissions) is itself a good interview answer — it shows judgment, not just a longer todo list.

---

## 7. Resume Bullets This Earns You

| Built | Bullet |
|---|---|
| Live auto-refreshing app | "Built a live-updating Streamlit dashboard for manufacturing quality monitoring" |
| Automated SPC recompute | "Automated control-limit calculation and violation flagging on every new data point" |
| Hypothesis testing | "Applied chi-square and ANOVA tests to validate root-cause hypotheses across machine/operator/shift" |
| Early defect prediction | "Built an early-warning classifier predicting batch defect risk from the first rods produced, with explicit data-leakage checks and recall-focused evaluation" |
| pytest suite | "Wrote automated tests validating statistical calculations and model performance floors" |
| Power BI (existing) | "Delivered executive BI reporting with SQL-driven root-cause analysis" |

---

## 8. Suggested Order

1. `spc.py` — port control-chart math, check it matches your Power BI numbers
2. `data.py` — static dataset loader + `generate_new_batch()`
3. `stats.py` — chi-square / ANOVA on existing data (quick, satisfying win)
4. `ml.py` — Model A (anomaly, reuse Nassau Candy code) then Model B (early prediction — take your time on leakage-checking here)
5. `app.py` — wire the five pages together, add autorefresh, reuse your CSS theme
6. `tests/test_spc.py` — last, once functions exist to test

Steps 1, 2, 3 are all comfortably "data analyst" work — pandas and scipy, nothing new to learn. Step 4's Model B is the one place worth slowing down, because getting the train/test split and leakage-checking right is what makes the whole "prevention" story true instead of just claimed.
