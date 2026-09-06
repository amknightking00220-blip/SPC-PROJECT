"""
ml.py — Anomaly detection layer for the rod manufacturing dataset.

Uses Isolation Forest to flag unusual batches based on their diameter
and defect characteristics — independent of the classical SPC control
limits in spc.py. The interesting part isn't the model itself, it's
comparing its flags against the SPC flags: do they agree? Does ML catch
anything SPC misses, or vice versa?

Early defect prediction lives elsewhere now: see rod_checkpoints.py
(synthetic checkpoint data) and rod_predictor.py (the prediction model)
for the per-rod, in-process early-warning system. That approach replaced
an earlier batch-level attempt that lived in this file — the checkpoint
version performed meaningfully better (59% vs 37% recall at the earliest
stage), so it's the one carried forward.

Depends on spc.py's batch_summary() to get one row per batch.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from spc import batch_summary, xbar_r_limits, p_np_limits, flag_violations

ANOMALY_FEATURES = ["Diameter_Mean", "Diameter_Range", "Defect_Rate"]
CONTAMINATION = 0.15  # expected fraction of batches that are "unusual" — roughly matches the ~14% SPC flags


def prepare_features(batch_df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Scales the anomaly-detection features so no single feature (e.g. Defect_Rate,
    which ranges 0-1) dominates just because of its raw scale.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(batch_df[ANOMALY_FEATURES])
    return X, scaler


def train_anomaly_detector(batch_df: pd.DataFrame, contamination: float = CONTAMINATION) -> tuple[IsolationForest, StandardScaler]:
    """
    Fits an Isolation Forest on batch-level features. Returns both the
    trained model and the scaler (needed together to score new batches later).
    """
    X, scaler = prepare_features(batch_df)

    model = IsolationForest(
        contamination=contamination,
        random_state=42,   # fixed seed so results are reproducible, not different every run
        n_estimators=100,
    )
    model.fit(X)
    return model, scaler


def score_anomalies(model: IsolationForest, scaler: StandardScaler, batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores every batch in batch_df. Adds two columns:
        Anomaly_Score  — lower (more negative) = more unusual
        Is_Anomaly     — True if the model flags this batch as an outlier
    """
    out = batch_df.copy()
    X = scaler.transform(out[ANOMALY_FEATURES])

    out["Anomaly_Score"] = model.decision_function(X)
    out["Is_Anomaly"] = model.predict(X) == -1  # sklearn convention: -1 = outlier, 1 = normal
    return out


def compare_ml_vs_spc(scored_df: pd.DataFrame) -> dict:
    """
    Compares ML anomaly flags against classical SPC out-of-control flags.
    This comparison is the actual point of this file — not the model alone.
    """
    both = scored_df[scored_df["Is_Anomaly"] & scored_df["Any_OutOfControl"]]
    ml_only = scored_df[scored_df["Is_Anomaly"] & ~scored_df["Any_OutOfControl"]]
    spc_only = scored_df[~scored_df["Is_Anomaly"] & scored_df["Any_OutOfControl"]]
    neither = scored_df[~scored_df["Is_Anomaly"] & ~scored_df["Any_OutOfControl"]]

    return {
        "total_batches": len(scored_df),
        "flagged_by_both": len(both),
        "flagged_by_ml_only": len(ml_only),
        "flagged_by_spc_only": len(spc_only),
        "flagged_by_neither": len(neither),
        "ml_only_batches": ml_only[["BatchID", "Machine", "Operator", "Shift", "Diameter_Mean", "Defect_Rate"]],
        "spc_only_batches": spc_only[["BatchID", "Machine", "Operator", "Shift", "Diameter_Mean", "Defect_Rate"]],
    }


def full_anomaly_report(df: pd.DataFrame) -> dict:
    """
    Convenience wrapper: runs SPC flagging + anomaly detection + comparison
    on raw rod-level data, in one call. This is what app.py will use.
    """
    batches = batch_summary(df)
    xr_limits = xbar_r_limits(batches)
    pnp_limits = p_np_limits(batches)
    limits = {**xr_limits, **pnp_limits}
    flagged = flag_violations(batches, limits)

    model, scaler = train_anomaly_detector(flagged)
    scored = score_anomalies(model, scaler, flagged)
    comparison = compare_ml_vs_spc(scored)

    return {
        "scored_batches": scored,
        "comparison": comparison,
        "model": model,
        "scaler": scaler,
    }



if __name__ == "__main__":
    # Quick manual check when you run: python ml.py
    df = pd.read_csv("metal_rods_spc_project.csv", sep="\t")
    report = full_anomaly_report(df)

    c = report["comparison"]
    print("=" * 70)
    print("ML ANOMALY DETECTION vs CLASSICAL SPC — COMPARISON")
    print("=" * 70)
    print(f"Total batches:              {c['total_batches']}")
    print(f"Flagged by BOTH:            {c['flagged_by_both']}")
    print(f"Flagged by ML only:         {c['flagged_by_ml_only']}")
    print(f"Flagged by SPC only:        {c['flagged_by_spc_only']}")
    print(f"Flagged by neither:         {c['flagged_by_neither']}")

    print()
    print("Batches ML flagged that SPC did NOT catch:")
    print("-" * 70)
    print(c["ml_only_batches"].to_string(index=False))

    print()
    print("Batches SPC flagged that ML did NOT catch:")
    print("-" * 70)
    print(c["spc_only_batches"].to_string(index=False))
