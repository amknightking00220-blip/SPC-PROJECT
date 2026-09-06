"""
rod_checkpoints.py — Simulates in-process checkpoint readings for each rod.

IMPORTANT — what this file is and isn't:
- The original CSV (metal_rods_spc_project.csv) is NEVER modified. This file
  reads it, but only ADDS new synthetic columns in memory, for backend/
  coding use in this checkpoint-prediction feature specifically.
- Checkpoint 3 (final) = the REAL diameter from your CSV. Never altered.
- Checkpoint 1 (early) and Checkpoint 2 (mid) are SIMULATED — they did not
  come from real sensor readings, because your original data only ever
  recorded one final measurement per rod. This is stated explicitly here
  and should be stated the same way in any write-up: this feature is a
  demonstrated concept on simulated checkpoints, not something validated
  against real historical in-process data (which was never recorded).

The simulation is calibrated using a REAL relationship found in your data:
defective rods sit about 2.7x further from the 10.0mm target on average
than good rods (0.048mm vs 0.018mm) — so the synthetic drift uses that
real ratio, rather than an arbitrary made-up number.
"""

import pandas as pd
import numpy as np

TARGET_DIAMETER = 10.0
RECOVERY_ALLOWED = True   # a rod can drift early and still finish within spec, or vice versa


def generate_checkpoint_dataset(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Takes the real rod-level dataframe and adds two SIMULATED early-stage
    diameter columns (Checkpoint1_Diameter, Checkpoint2_Diameter), while
    keeping the real final diameter and defect label untouched.

    Returns a NEW dataframe — df itself (and the original CSV) is never modified.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    final_diameter = out["Diameter_mm"].values
    is_defective = out["Defective"].values

    # Real-data-calibrated drift: defective rods get a bigger typical
    # early-stage offset from target, but with enough spread that some
    # overlap with good rods happens (that's what allows "recovery").
    base_drift_scale = np.where(is_defective == 1, 0.048, 0.018)
    early_drift = rng.normal(loc=0, scale=base_drift_scale * 2.2, size=len(out))

    # Checkpoint 1 (early): far from final, full drift + sensor noise
    checkpoint1 = final_diameter + early_drift + rng.normal(0, 0.010, size=len(out))

    # Checkpoint 2 (mid): drift has partly settled back toward final
    # (process self-correcting, like real machines re-calibrating mid-run)
    settle_factor = 0.45
    checkpoint2 = final_diameter + early_drift * settle_factor + rng.normal(0, 0.008, size=len(out))

    out["Checkpoint1_Diameter"] = checkpoint1
    out["Checkpoint2_Diameter"] = checkpoint2
    out["Checkpoint3_Diameter"] = final_diameter  # real value, just renamed for consistency

    return out


if __name__ == "__main__":
    # Quick manual check when you run: python rod_checkpoints.py
    df = pd.read_csv("metal_rods_spc_project.csv", sep="\t")
    checkpoints = generate_checkpoint_dataset(df)

    print("=" * 70)
    print("SYNTHETIC CHECKPOINT DATA — SANITY CHECK")
    print("=" * 70)
    print(checkpoints[[
        "BatchID", "Defective", "Checkpoint1_Diameter", "Checkpoint2_Diameter", "Checkpoint3_Diameter"
    ]].head(10).to_string(index=False))

    print()
    print("Does early drift actually correlate with the real outcome?")
    print("-" * 70)
    checkpoints["Checkpoint1_Offset"] = (checkpoints["Checkpoint1_Diameter"] - TARGET_DIAMETER).abs()
    print(checkpoints.groupby("Defective")["Checkpoint1_Offset"].mean())

    print()
    print("How many 'recovery' cases exist (early looked bad, final was fine, or vice versa)?")
    print("-" * 70)
    early_looks_bad = checkpoints["Checkpoint1_Offset"] > checkpoints["Checkpoint1_Offset"].median()
    final_is_defective = checkpoints["Defective"] == 1
    recovered = ((early_looks_bad) & (~final_is_defective)).sum()
    surprised_late = ((~early_looks_bad) & (final_is_defective)).sum()
    print(f"Early looked risky but finished OK (recovered): {recovered} rods")
    print(f"Early looked fine but finished defective (surprised late): {surprised_late} rods")
