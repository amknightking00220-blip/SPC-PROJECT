"""
data.py — Loads the real dataset and generates new, realistic "live" batches.

The simulator's numbers (mean diameter, std, defect rate) are pulled directly
from your actual CSV, not guessed — so a generated batch looks statistically
like a real one, not an arbitrary random batch.

This is what makes the Streamlit app feel "live": every time the app
refreshes, it can call generate_new_batch() and have a new row appear.
"""

import pandas as pd
import numpy as np

CSV_PATH = "metal rods spc project.csv"

# ---------------------------------------------------------------------------
# Parameters learned from the real dataset (computed once, hardcoded here so
# the simulator doesn't silently drift if the source file changes later —
# see recompute_parameters() below if you want them refreshed from data)
# ---------------------------------------------------------------------------
DIAMETER_MEAN = 9.999833
DIAMETER_STD = 0.039024
BASE_DEFECT_RATE = 0.2633   # per-rod probability of being defective, normal conditions
DRIFT_DEFECT_RATE = 0.55     # elevated probability during a simulated "drift" event
RODS_PER_BATCH = 5

MACHINES = ["M1", "M2", "M3"]
OPERATORS = ["A", "B", "C"]
SHIFTS = ["Morning", "Afternoon", "Night"]


def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    """Loads the original tab-separated dataset."""
    return pd.read_csv(path, sep="\t")


def recompute_parameters(df: pd.DataFrame) -> dict:
    """
    Optional: recompute the simulator's parameters directly from a given
    dataframe, instead of the hardcoded constants above. Useful if you
    later load a different or updated dataset.
    """
    return {
        "diameter_mean": df["Diameter_mm"].mean(),
        "diameter_std": df["Diameter_mm"].std(),
        "base_defect_rate": df["Defective"].mean(),
    }


def generate_new_batch(batch_id: int, drift: bool = False, rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Generates one new batch (5 rod rows) with the same schema as the
    original dataset.

    drift=True simulates a process going out of control — elevated defect
    probability and slightly wider diameter spread — useful for demoing
    that the control charts and alerts actually catch it.
    """
    if rng is None:
        rng = np.random.default_rng()

    machine = rng.choice(MACHINES)
    operator = rng.choice(OPERATORS)
    shift = rng.choice(SHIFTS)

    diameter_std = DIAMETER_STD * (1.6 if drift else 1.0)
    defect_rate = DRIFT_DEFECT_RATE if drift else BASE_DEFECT_RATE

    diameters = rng.normal(loc=DIAMETER_MEAN, scale=diameter_std, size=RODS_PER_BATCH)
    defects = rng.binomial(n=1, p=defect_rate, size=RODS_PER_BATCH)

    return pd.DataFrame({
        "BatchID": [batch_id] * RODS_PER_BATCH,
        "Operator": [operator] * RODS_PER_BATCH,
        "Machine": [machine] * RODS_PER_BATCH,
        "Shift": [shift] * RODS_PER_BATCH,
        "Diameter_mm": diameters,
        "Defective": defects,
    })


def append_live_batch(df: pd.DataFrame, drift_probability: float = 0.08, rng: np.random.Generator = None) -> pd.DataFrame:
    """
    Convenience function for the app: takes the current dataframe, generates
    ONE new batch (with a small random chance of simulating a drift event),
    appends it, and returns the updated dataframe.

    This is the single function app.py will call on every refresh.
    """
    if rng is None:
        rng = np.random.default_rng()

    next_batch_id = df["BatchID"].max() + 1
    is_drift_event = rng.random() < drift_probability

    new_batch = generate_new_batch(next_batch_id, drift=is_drift_event, rng=rng)
    updated_df = pd.concat([df, new_batch], ignore_index=True)

    return updated_df, is_drift_event


if __name__ == "__main__":
    # Quick manual check when you run: python data.py
    df = load_data()
    print(f"Loaded {len(df)} rods across {df['BatchID'].nunique()} batches")
    print()

    print("Generating 3 new normal batches + 1 drift batch...")
    rng = np.random.default_rng(seed=42)  # seeded for reproducible demo output

    for i in range(3):
        df, drifted = append_live_batch(df, drift_probability=0.0, rng=rng)
    df, drifted = append_live_batch(df, drift_probability=1.0, rng=rng)  # force a drift batch

    print(f"Now {len(df)} rods across {df['BatchID'].nunique()} batches")
    print()
    print("Last 5 new batches added:")
    print(df.tail(20).to_string(index=False))
