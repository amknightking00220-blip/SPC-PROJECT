"""
app.py — SPC-Sentinel Streamlit dashboard.

Ties together spc.py, stats.py, data.py, ml.py, rod_checkpoints.py, and
rod_predictor.py into one live, multi-page app.

Being built one page at a time. Currently done:
    - Live Overview

Still to come:
    - Control Charts
    - Hypothesis Tests
    - Early Defect Risk
    - Root Cause Explorer
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import data
import spc
import stats
import rod_checkpoints
import rod_predictor

# streamlit-autorefresh gives us a one-line way to re-run the app on a timer,
# which is what makes "live" actually true without a background process.
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="SPC-Sentinel", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# SESSION STATE — this is what makes the data "live" instead of resetting to
# the original 600 rows on every single rerun. See the chat explanation for
# why this is necessary in Streamlit specifically.
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = data.load_data()

if "last_drift_event" not in st.session_state:
    st.session_state.last_drift_event = False

# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
PAGES = [
    "📊 Live Overview",
    "📈 Control Charts",
    "🔬 Hypothesis Tests",
    "🤖 Early Defect Risk",
]
page = st.sidebar.radio("Navigate", PAGES)

st.sidebar.markdown("---")
autorefresh_on = st.sidebar.checkbox("Live auto-refresh (new batch every 20s)", value=True)
if autorefresh_on:
    # count is just an internal tick counter streamlit needs; we don't use it
    st_autorefresh(interval=20_000, key="live_refresh_tick")

manual_batch = st.sidebar.button("➕ Add one batch now")

# ---------------------------------------------------------------------------
# THE ONE PLACE NEW DATA GETS ADDED
# Runs on every autorefresh tick (if enabled) AND on manual button click.
# ---------------------------------------------------------------------------
if autorefresh_on or manual_batch:
    st.session_state.df, drifted = data.append_live_batch(st.session_state.df)
    st.session_state.last_drift_event = drifted

# Recompute SPC numbers fresh every rerun — this is what "automated" means
# here: nothing is cached from before, it's always based on the current df.
report = spc.full_spc_report(st.session_state.df)


# ---------------------------------------------------------------------------
# PAGE: LIVE OVERVIEW
# ---------------------------------------------------------------------------
def render_live_overview():
    st.title("📊 Live Overview")

    if st.session_state.last_drift_event:
        st.warning("⚠️ The most recently added batch was a simulated **drift event** "
                    "(elevated defect probability + wider diameter spread).")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Rods", report["total_rods"])
    col2.metric("Total Batches", report["total_batches"])
    col3.metric("Total Defects", report["total_defects"])
    col4.metric("Defect Rate", f"{report['overall_defect_rate']*100:.1f}%")
    col5.metric("Out-of-Control Batches", report["out_of_control_batches"])

    st.markdown("---")
    st.subheader("Most Recent Batch")

    latest_batch_id = st.session_state.df["BatchID"].max()
    latest_rows = st.session_state.df[st.session_state.df["BatchID"] == latest_batch_id]
    st.dataframe(latest_rows, width="stretch", hide_index=True)

    st.caption(
        f"Batch {latest_batch_id} — "
        f"{latest_rows['Defective'].sum()} of {len(latest_rows)} rods defective. "
        "This page auto-refreshes and adds one new simulated batch every ~20 seconds "
        "when live auto-refresh is on (toggle in the sidebar), or instantly via "
        "the 'Add one batch now' button."
    )


# ---------------------------------------------------------------------------
# HELPER: builds one control chart (used by all 4 chart types below).
# Out-of-control batches are drawn as red markers so they jump out visually,
# same idea as red flags in the old Power BI charts.
# ---------------------------------------------------------------------------
def make_control_chart(batches, y_col, flag_col, center, ucl, lcl, title, y_label):
    colors = ["red" if flagged else "steelblue" for flagged in batches[flag_col]]

    fig = go.Figure()

    # connecting line (plain, so the eye can follow the trend)
    fig.add_trace(go.Scatter(
        x=batches["BatchID"], y=batches[y_col],
        mode="lines", line=dict(color="lightblue", width=1),
        showlegend=False, hoverinfo="skip",
    ))
    # the actual data points, colored by whether they violated a limit
    fig.add_trace(go.Scatter(
        x=batches["BatchID"], y=batches[y_col],
        mode="markers", marker=dict(color=colors, size=6),
        name="Batch value",
        hovertemplate="Batch %{x}<br>Value: %{y:.4f}<extra></extra>",
    ))

    fig.add_hline(y=center, line_dash="dash", line_color="gray",
                  annotation_text="Center", annotation_position="right")
    fig.add_hline(y=ucl, line_dash="dot", line_color="orange",
                  annotation_text="UCL", annotation_position="right")
    fig.add_hline(y=lcl, line_dash="dot", line_color="orange",
                  annotation_text="LCL", annotation_position="right")

    fig.update_layout(
        title=title, xaxis_title="BatchID", yaxis_title=y_label,
        height=340, margin=dict(t=40, b=30, l=10, r=10),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# PAGE: CONTROL CHARTS
# ---------------------------------------------------------------------------
def render_control_charts():
    st.title("📈 Control Charts")
    st.caption(
        "All four charts recompute their control limits from the *current* live "
        "dataset every refresh — so as new batches come in, the center line and "
        "UCL/LCL can shift slightly, and newly out-of-control batches (red dots) "
        "appear automatically."
    )

    batches = report["batches"]
    limits = report["limits"]

    # ----- X-bar chart: average diameter per batch -----
    st.plotly_chart(
        make_control_chart(
            batches, "Diameter_Mean", "Xbar_OutOfControl",
            limits["xbar_center"], limits["xbar_ucl"], limits["xbar_lcl"],
            "X-bar Chart — Average Diameter per Batch", "Diameter (mm)",
        ),
        width="stretch",
    )
    st.caption(
        f"Formula: Center = grand mean of batch averages = {limits['xbar_center']:.4f} mm. "
        f"UCL/LCL = Center ± A2 × R-bar, with A2 = 0.577 (the standard constant for "
        f"subgroup size n=5) and R-bar = {limits['r_center']:.4f} → "
        f"UCL = {limits['xbar_ucl']:.4f}, LCL = {limits['xbar_lcl']:.4f}."
    )

    # ----- R chart: within-batch range -----
    st.plotly_chart(
        make_control_chart(
            batches, "Diameter_Range", "R_OutOfControl",
            limits["r_center"], limits["r_ucl"], limits["r_lcl"],
            "R Chart — Variation Within Each Batch", "Diameter Range (mm)",
        ),
        width="stretch",
    )
    st.caption(
        f"Formula: Center = R-bar (mean range across batches) = {limits['r_center']:.4f} mm. "
        f"UCL = D4 × R-bar, LCL = D3 × R-bar, with D4 = 2.114 and D3 = 0 "
        f"(standard constants for n=5) → UCL = {limits['r_ucl']:.4f}, LCL = {limits['r_lcl']:.4f}."
    )

    # ----- P chart: defect proportion per batch -----
    st.plotly_chart(
        make_control_chart(
            batches, "Defect_Rate", "P_OutOfControl",
            limits["p_center"], limits["p_ucl"], limits["p_lcl"],
            "P Chart — Defect Proportion per Batch", "Defect Rate",
        ),
        width="stretch",
    )
    st.caption(
        f"Formula: Center = p-bar (overall defect rate across all rods) = {limits['p_center']:.4f}. "
        f"UCL/LCL = p-bar ± 3 × sqrt(p-bar × (1 − p-bar) / n), with n = 5 rods/batch → "
        f"UCL = {limits['p_ucl']:.4f}, LCL = {limits['p_lcl']:.4f} (clipped at 0 if negative)."
    )

    # ----- NP chart: defect count per batch -----
    st.plotly_chart(
        make_control_chart(
            batches, "Total_Defects", "NP_OutOfControl",
            limits["np_center"], limits["np_ucl"], limits["np_lcl"],
            "NP Chart — Defect Count per Batch", "Defective Rods",
        ),
        width="stretch",
    )
    st.caption(
        f"Formula: Center = n × p-bar = {limits['np_center']:.4f} defects/batch. "
        f"UCL/LCL = Center ± 3 × sqrt(np-bar × (1 − p-bar)) → "
        f"UCL = {limits['np_ucl']:.4f}, LCL = {limits['np_lcl']:.4f} (clipped at 0 if negative)."
    )

    st.markdown("---")
    st.caption(
        f"{report['out_of_control_batches']} of {report['total_batches']} batches are "
        "currently out-of-control on at least one chart (row highlighted red on any chart above)."
    )


# ---------------------------------------------------------------------------
# HELPER: renders one test result as a small card — p-value + plain verdict.
# Colored green if significant, gray if not, so the eye can scan quickly.
# ---------------------------------------------------------------------------
def render_test_result(result: dict):
    st.markdown(f"**{result['test']}**")
    if result["p_value"] < stats.ALPHA:
        st.success(result["verdict"])
    else:
        st.info(result["verdict"])


# ---------------------------------------------------------------------------
# PAGE: HYPOTHESIS TESTS
# ---------------------------------------------------------------------------
def render_hypothesis_tests():
    st.title("🔬 Hypothesis Tests")
    st.caption(
        "All p-values below recompute on the current live dataset every refresh. "
        "As more batches come in, a result can occasionally flip from significant "
        "to not (or vice versa) — that's expected with a growing sample, not a bug."
    )

    df = st.session_state.df

    st.subheader("Is defect rate actually associated with Machine / Operator / Shift?")
    st.caption(
        "Chi-square test of independence — checks whether defect rate differences "
        "across categories are a real pattern or could just be random noise."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        render_test_result(stats.chi_square_defect_association(df, "Machine"))
    with c2:
        render_test_result(stats.chi_square_defect_association(df, "Operator"))
    with c3:
        render_test_result(stats.chi_square_defect_association(df, "Shift"))

    st.markdown("---")
    st.subheader("Does average rod diameter genuinely differ by Machine / Operator?")
    st.caption(
        "One-way ANOVA — checks whether the diameter differences between categories "
        "are bigger than you'd expect from normal random variation alone."
    )
    c1, c2 = st.columns(2)
    with c1:
        result = stats.anova_diameter_by_group(df, "Machine")
        render_test_result(result)
        st.caption("Group means (mm): " + ", ".join(
            f"{k}: {v:.4f}" for k, v in result["group_means"].items()
        ))
    with c2:
        result = stats.anova_diameter_by_group(df, "Operator")
        render_test_result(result)
        st.caption("Group means (mm): " + ", ".join(
            f"{k}: {v:.4f}" for k, v in result["group_means"].items()
        ))

    st.markdown("---")
    st.subheader("Compare two specific categories directly")
    st.caption(
        "Two-proportion z-test — e.g. is Operator B's defect rate significantly "
        "different from Operator A's, specifically (not just 'some difference exists somewhere')."
    )

    group_col = st.selectbox("Compare within:", ["Machine", "Operator", "Shift"])
    categories = sorted(df[group_col].unique())

    c1, c2 = st.columns(2)
    category_a = c1.selectbox("Category A", categories, index=0, key="cat_a")
    category_b = c2.selectbox("Category B", categories, index=min(1, len(categories) - 1), key="cat_b")

    if category_a == category_b:
        st.warning("Pick two different categories to compare.")
    else:
        result = stats.two_proportion_ztest(df, group_col, category_a, category_b)
        render_test_result(result)
        st.caption(
            f"Defect rate — {category_a}: {result['rate_a']*100:.1f}%   "
            f"{category_b}: {result['rate_b']*100:.1f}%"
        )


# ---------------------------------------------------------------------------
# PAGE: EARLY DEFECT RISK
# ---------------------------------------------------------------------------
def render_early_defect_risk():
    st.title("🤖 Early Defect Risk")
    st.caption(
        "⚠️ Honesty note: Checkpoint 1 and 2 readings are SIMULATED (calibrated from "
        "a real relationship in the data — defective rods sit ~2.7x further from the "
        "10.0mm target on average than good rods). This demonstrates the concept on "
        "simulated in-process checkpoints; it hasn't been validated against real sensor "
        "readings, because those were never recorded for this dataset."
    )
    st.caption(
        "This page **retrains the Random Forest from scratch on every refresh**, using "
        "the current live dataset — the honest, fully-live approach, at the cost of the "
        "model (and its reported recall/precision) shifting slightly refresh to refresh "
        "as the training data grows."
    )

    df = st.session_state.df

    with st.spinner("Retraining model on current live data..."):
        checkpoint_df = rod_checkpoints.generate_checkpoint_dataset(df)
        stages = rod_predictor.train_progressive_checkpoint_predictors(checkpoint_df)

    st.subheader("Model performance at each stage")
    metrics_rows = []
    for n, result in stages.items():
        m = result["metrics"]
        metrics_rows.append({
            "Stage": f"Checkpoint {'1' if n == 1 else '1 + 2'}",
            "Accuracy": f"{m['accuracy']:.3f}",
            "Precision": f"{m['precision']:.3f}",
            "Recall": f"{m['recall']:.3f}",
            "F1": f"{m['f1']:.3f}",
            "Trained on": f"{m['train_size']} rods",
        })
    st.dataframe(pd.DataFrame(metrics_rows), hide_index=True, width="stretch")
    st.caption(
        "Recall matters most here: missing a genuinely bad rod costs wasted material, "
        "while a false alarm just costs a double-check."
    )

    st.markdown("---")
    st.subheader("What drives risk the most (Stage 2 model)")
    fi = stages[2]["metrics"]["feature_importances"].head(6)
    fig = go.Figure(go.Bar(
        x=fi.values[::-1], y=fi.index[::-1], orientation="h",
        marker=dict(color="steelblue"),
    ))
    fig.update_layout(
        height=280, margin=dict(t=20, b=20, l=10, r=10),
        xaxis_title="Importance",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Try it: predict one rod's risk live")
    c1, c2, c3 = st.columns(3)
    machine = c1.selectbox("Machine", sorted(df["Machine"].unique()))
    operator = c2.selectbox("Operator", sorted(df["Operator"].unique()))
    shift = c3.selectbox("Shift", sorted(df["Shift"].unique()))

    checkpoint1 = st.number_input(
        "Checkpoint 1 diameter reading (mm)", value=10.000, step=0.001, format="%.3f",
    )
    result1 = rod_predictor.predict_rod_risk(stages, [checkpoint1], machine, operator, shift)
    st.metric("Predicted risk after Checkpoint 1", f"{result1['predicted_risk']*100:.1f}%")
    st.caption(f"This stage's model recall on held-out test data: {result1['recall_at_this_stage']:.3f}")

    have_cp2 = st.checkbox("I also have a Checkpoint 2 reading for this rod")
    if have_cp2:
        checkpoint2 = st.number_input(
            "Checkpoint 2 diameter reading (mm)", value=10.000, step=0.001, format="%.3f",
        )
        result2 = rod_predictor.predict_rod_risk(
            stages, [checkpoint1, checkpoint2], machine, operator, shift
        )
        st.metric("Predicted risk after Checkpoint 1 + 2", f"{result2['predicted_risk']*100:.1f}%")
        st.caption(f"This stage's model recall on held-out test data: {result2['recall_at_this_stage']:.3f}")


# ---------------------------------------------------------------------------
# ROUTER — pages not built yet just show a placeholder for now
# ---------------------------------------------------------------------------
if page == "📊 Live Overview":
    render_live_overview()
elif page == "📈 Control Charts":
    render_control_charts()
elif page == "🔬 Hypothesis Tests":
    render_hypothesis_tests()
elif page == "🤖 Early Defect Risk":
    render_early_defect_risk()
else:
    st.title(page)
    st.info("This page hasn't been built yet — coming in the next step.")
