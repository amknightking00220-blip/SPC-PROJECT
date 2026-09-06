"""
stats.py — Hypothesis testing on the rod manufacturing dataset.

Answers questions like: "Machine M3 has a lower defect rate than M1 in the
data — but is that a real effect, or could it just be noise?"

All functions return a plain dict with the test statistic, p-value, and a
ready-to-display plain-language verdict, so the Streamlit app can show
these directly without re-interpreting anything.

Expects the raw rod-level dataframe:
    BatchID, Operator, Machine, Shift, Diameter_mm, Defective
"""

import pandas as pd
import numpy as np
from scipy import stats

ALPHA = 0.05  # standard significance threshold


def _verdict(p_value: float) -> str:
    if p_value < ALPHA:
        return f"Statistically significant (p = {p_value:.4f} < {ALPHA}) — the difference is likely real."
    return f"Not statistically significant (p = {p_value:.4f} >= {ALPHA}) — the difference could just be noise."


def chi_square_defect_association(df: pd.DataFrame, group_col: str) -> dict:
    """
    Is defect rate actually associated with the given group column
    (Machine, Operator, or Shift)? Or could the observed differences
    just be random chance?

    Builds a contingency table: group category x [defective, ok] counts.
    """
    contingency = pd.crosstab(df[group_col], df["Defective"])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    return {
        "test": f"Chi-square test of independence: Defective ~ {group_col}",
        "chi2_statistic": chi2,
        "degrees_of_freedom": dof,
        "p_value": p_value,
        "contingency_table": contingency,
        "verdict": _verdict(p_value),
    }


def anova_diameter_by_group(df: pd.DataFrame, group_col: str) -> dict:
    """
    Does average rod diameter genuinely differ across group categories
    (e.g. across M1/M2/M3), or is the variation just normal noise?

    One-way ANOVA across all categories in group_col at once.
    """
    groups = [
        sub_df["Diameter_mm"].values
        for _, sub_df in df.groupby(group_col)
    ]
    f_statistic, p_value = stats.f_oneway(*groups)

    group_means = df.groupby(group_col)["Diameter_mm"].mean()

    return {
        "test": f"One-way ANOVA: Diameter_mm ~ {group_col}",
        "f_statistic": f_statistic,
        "p_value": p_value,
        "group_means": group_means,
        "verdict": _verdict(p_value),
    }


def two_proportion_ztest(df: pd.DataFrame, group_col: str, category_a: str, category_b: str) -> dict:
    """
    Is the defect rate for category_a significantly different from
    category_b within group_col? e.g. Operator "A" vs Operator "B".
    """
    group_a = df[df[group_col] == category_a]["Defective"]
    group_b = df[df[group_col] == category_b]["Defective"]

    n_a, n_b = len(group_a), len(group_b)
    x_a, x_b = group_a.sum(), group_b.sum()
    p_a, p_b = x_a / n_a, x_b / n_b

    # pooled proportion under the null hypothesis (no real difference)
    p_pool = (x_a + x_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))

    z_statistic = (p_a - p_b) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))  # two-tailed

    return {
        "test": f"Two-proportion z-test: {group_col} '{category_a}' vs '{category_b}'",
        "rate_a": p_a,
        "rate_b": p_b,
        "z_statistic": z_statistic,
        "p_value": p_value,
        "verdict": _verdict(p_value),
    }


def run_all_tests(df: pd.DataFrame) -> dict:
    """
    Convenience wrapper: runs the standard battery of tests for this
    dataset (Machine, Operator, Shift) and returns everything at once.
    """
    results = {
        "chi_square_machine": chi_square_defect_association(df, "Machine"),
        "chi_square_operator": chi_square_defect_association(df, "Operator"),
        "chi_square_shift": chi_square_defect_association(df, "Shift"),
        "anova_diameter_machine": anova_diameter_by_group(df, "Machine"),
        "anova_diameter_operator": anova_diameter_by_group(df, "Operator"),
    }
    return results


if __name__ == "__main__":
    # Quick manual check when you run: python stats.py
    df = pd.read_csv("metal_rods_spc_project.csv", sep="\t")
    results = run_all_tests(df)

    print("=" * 70)
    print("HYPOTHESIS TEST RESULTS")
    print("=" * 70)

    for key, r in results.items():
        print()
        print(r["test"])
        print("-" * 70)
        print(f"p-value: {r['p_value']:.4f}")
        print(r["verdict"])

    print()
    print("=" * 70)
    print("EXAMPLE: two-proportion z-test, Operator B vs Operator A")
    print("=" * 70)
    r = two_proportion_ztest(df, "Operator", "B", "A")
    print(f"Defect rate — Operator B: {r['rate_a']:.3f}   Operator A: {r['rate_b']:.3f}")
    print(f"p-value: {r['p_value']:.4f}")
    print(r["verdict"])
