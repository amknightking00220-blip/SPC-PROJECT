"""
spc.py — Statistical Process Control calculations for the rod manufacturing dataset.

No classes, no framework magic — just functions that take a dataframe and
return numbers, so they're easy to read, easy to test, and easy to reuse
in the Streamlit app later.

Expects the raw CSV with one row per ROD (5 rods per batch):
    BatchID, Operator, Machine, Shift, Diameter_mm, Defective
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Standard SPC control-chart constants for subgroup size n = 5
# (from the standard SPC constants table — these don't change, they're
# defined by statistical theory for a given subgroup size)
# ---------------------------------------------------------------------------
A2 = 0.577   # X-bar chart limit multiplier
D3 = 0.0     # R chart lower limit multiplier
D4 = 2.114   # R chart upper limit multiplier


def batch_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse rod-level data into one row per batch:
    mean diameter, range (max-min), defect count, defect rate, machine/operator/shift.

    This is the table every other function in this file works from.
    """
    grouped = df.groupby("BatchID").agg(
        Operator=("Operator", "first"),
        Machine=("Machine", "first"),
        Shift=("Shift", "first"),
        Diameter_Mean=("Diameter_mm", "mean"),
        Diameter_Range=("Diameter_mm", lambda x: x.max() - x.min()),
        Sample_Size=("Diameter_mm", "count"),
        Total_Defects=("Defective", "sum"),
    ).reset_index()

    grouped["Defect_Rate"] = grouped["Total_Defects"] / grouped["Sample_Size"]
    return grouped.sort_values("BatchID").reset_index(drop=True)


def xbar_r_limits(batch_df: pd.DataFrame) -> dict:
    """
    Control limits for the X-bar chart (average diameter per batch)
    and the R chart (within-batch range), using the standard formulas
    for subgroup size n=5.
    """
    xbar_bar = batch_df["Diameter_Mean"].mean()   # grand mean
    r_bar = batch_df["Diameter_Range"].mean()      # mean range

    return {
        "xbar_center": xbar_bar,
        "xbar_ucl": xbar_bar + A2 * r_bar,
        "xbar_lcl": xbar_bar - A2 * r_bar,
        "r_center": r_bar,
        "r_ucl": D4 * r_bar,
        "r_lcl": D3 * r_bar,
    }


def p_np_limits(batch_df: pd.DataFrame) -> dict:
    """
    Control limits for the P chart (defect proportion per batch)
    and the NP chart (defect count per batch), assuming constant
    subgroup size n (5 rods per batch here).
    """
    n = batch_df["Sample_Size"].iloc[0]  # constant subgroup size
    p_bar = batch_df["Total_Defects"].sum() / batch_df["Sample_Size"].sum()
    np_bar = n * p_bar

    p_std_error = np.sqrt(p_bar * (1 - p_bar) / n)
    np_std_error = np.sqrt(np_bar * (1 - p_bar))

    return {
        "p_center": p_bar,
        "p_ucl": p_bar + 3 * p_std_error,
        "p_lcl": max(0.0, p_bar - 3 * p_std_error),
        "np_center": np_bar,
        "np_ucl": np_bar + 3 * np_std_error,
        "np_lcl": max(0.0, np_bar - 3 * np_std_error),
    }


def flag_violations(batch_df: pd.DataFrame, limits: dict) -> pd.DataFrame:
    """
    Adds boolean flag columns to batch_df marking which batches breach
    each control chart's limits. Returns the same dataframe with extra columns.
    """
    out = batch_df.copy()

    out["Xbar_OutOfControl"] = (
        (out["Diameter_Mean"] > limits["xbar_ucl"]) |
        (out["Diameter_Mean"] < limits["xbar_lcl"])
    )
    out["R_OutOfControl"] = (
        (out["Diameter_Range"] > limits["r_ucl"]) |
        (out["Diameter_Range"] < limits["r_lcl"])
    )
    out["P_OutOfControl"] = (
        (out["Defect_Rate"] > limits["p_ucl"]) |
        (out["Defect_Rate"] < limits["p_lcl"])
    )
    out["NP_OutOfControl"] = (
        (out["Total_Defects"] > limits["np_ucl"]) |
        (out["Total_Defects"] < limits["np_lcl"])
    )
    out["Any_OutOfControl"] = (
        out["Xbar_OutOfControl"] | out["R_OutOfControl"] |
        out["P_OutOfControl"] | out["NP_OutOfControl"]
    )
    return out


def full_spc_report(df: pd.DataFrame) -> dict:
    """
    Convenience wrapper: runs the whole pipeline on raw rod-level data
    and returns everything you'd want to display on a dashboard.
    """
    batches = batch_summary(df)
    xr_limits = xbar_r_limits(batches)
    pnp_limits = p_np_limits(batches)
    all_limits = {**xr_limits, **pnp_limits}
    flagged = flag_violations(batches, all_limits)

    return {
        "batches": flagged,
        "limits": all_limits,
        "total_rods": len(df),
        "total_batches": len(batches),
        "total_defects": int(batches["Total_Defects"].sum()),
        "overall_defect_rate": batches["Total_Defects"].sum() / batches["Sample_Size"].sum(),
        "out_of_control_batches": int(flagged["Any_OutOfControl"].sum()),
    }


if __name__ == "__main__":
    # Quick manual check when you run: python spc.py
    df = pd.read_csv("metal_rods_spc_project.csv", sep="\t")
    report = full_spc_report(df)

    print("=" * 60)
    print("SPC SUMMARY")
    print("=" * 60)
    print(f"Total rods:              {report['total_rods']}")
    print(f"Total batches:           {report['total_batches']}")
    print(f"Total defective rods:    {report['total_defects']}")
    print(f"Overall defect rate:     {report['overall_defect_rate']:.4f}")
    print(f"Out-of-control batches:  {report['out_of_control_batches']}")
    print()
    print("CONTROL LIMITS")
    print("-" * 60)
    for k, v in report["limits"].items():
        print(f"  {k:15s}: {v:.5f}")
    print()
    print("FLAGGED (OUT-OF-CONTROL) BATCHES")
    print("-" * 60)
    flagged_only = report["batches"][report["batches"]["Any_OutOfControl"]]
    print(flagged_only[[
        "BatchID", "Machine", "Operator", "Shift",
        "Diameter_Mean", "Diameter_Range", "Total_Defects", "Defect_Rate",
        "Xbar_OutOfControl", "R_OutOfControl", "P_OutOfControl", "NP_OutOfControl"
    ]].to_string(index=False))
