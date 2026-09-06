"""
rod_predictor.py — Predicts whether a SINGLE rod will finish defective,
using only its own in-process checkpoint readings (not sibling rods).

Depends on rod_checkpoints.py for the simulated Checkpoint1/2 columns.
See rod_checkpoints.py's docstring for the important honesty note about
those columns being simulated, not real sensor data.

Same progressive idea as ml.py's batch-level Model B, but the "steps"
here are real physical stages of ONE rod's own formation:
    Stage 1 — after Checkpoint 1 only (rod barely started)
    Stage 2 — after Checkpoint 1 + 2 (rod mid-process)

Ground truth (the label) is always the REAL Defective column from your
original CSV — only the early-stage features are simulated, never the answer.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from rod_checkpoints import generate_checkpoint_dataset

MAX_CHECKPOINTS = 2   # Checkpoint 3 is the final/real diameter — using it wouldn't be "early" anymore


def build_checkpoint_features(checkpoint_df: pd.DataFrame, n_checkpoints: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Builds features using only the first n_checkpoints readings for each rod,
    plus Machine/Operator/Shift (known before the rod is even made).

    Returns (X, y) where y is the REAL Defective label.
    """
    feature_cols = [f"Checkpoint{i}_Diameter" for i in range(1, n_checkpoints + 1)]
    features_df = checkpoint_df[feature_cols + ["Machine", "Operator", "Shift"]].copy()

    X = pd.get_dummies(features_df, columns=["Machine", "Operator", "Shift"])
    y = checkpoint_df["Defective"]

    return X, y


def assert_no_checkpoint_leakage(X: pd.DataFrame, n_checkpoints: int) -> bool:
    """
    Confirms no feature references a checkpoint beyond n_checkpoints
    (e.g. the final Checkpoint3 reading sneaking into an "early" model).
    """
    leaky = [f"Checkpoint{i}" for i in range(n_checkpoints + 1, 4)]
    offending = [col for col in X.columns for pat in leaky if pat in col]
    if offending:
        raise ValueError(f"Data leakage detected — these columns see the future: {offending}")
    return True


def train_checkpoint_predictor(X: pd.DataFrame, y: pd.Series, n_checkpoints: int,
                                test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Trains a Random Forest to predict rod-level defect outcome from early
    checkpoint readings, with a proper train/test split.
    """
    assert_no_checkpoint_leakage(X, n_checkpoints)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_importances": pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False),
    }

    return {"model": model, "metrics": metrics, "feature_columns": list(X.columns)}


def train_progressive_checkpoint_predictors(checkpoint_df: pd.DataFrame, max_checkpoints: int = MAX_CHECKPOINTS) -> dict:
    """
    Trains one model per checkpoint stage (1, then 1+2), so you can compare
    how prediction quality improves as the rod progresses through production.
    """
    stages = {}
    for n in range(1, max_checkpoints + 1):
        X, y = build_checkpoint_features(checkpoint_df, n_checkpoints=n)
        stages[n] = train_checkpoint_predictor(X, y, n_checkpoints=n)
    return stages


def predict_rod_risk(stages: dict, checkpoint_readings: list, machine: str, operator: str, shift: str) -> dict:
    """
    Given however many checkpoint readings exist so far for ONE rod
    currently being produced, returns the live predicted risk of it
    finishing defective.
    """
    n_seen = len(checkpoint_readings)
    if n_seen not in stages:
        raise ValueError(f"No trained model for {n_seen} checkpoints (available: {list(stages.keys())})")

    row = {"Machine": machine, "Operator": operator, "Shift": shift}
    for i, reading in enumerate(checkpoint_readings, start=1):
        row[f"Checkpoint{i}_Diameter"] = reading

    row_df = pd.DataFrame([row])
    row_encoded = pd.get_dummies(row_df, columns=["Machine", "Operator", "Shift"])
    row_aligned = row_encoded.reindex(columns=stages[n_seen]["feature_columns"], fill_value=0)

    probability = stages[n_seen]["model"].predict_proba(row_aligned)[0][1]
    return {"checkpoints_seen": n_seen, "predicted_risk": probability, "recall_at_this_stage": stages[n_seen]["metrics"]["recall"]}


if __name__ == "__main__":
    # Quick manual check when you run: python rod_predictor.py
    df = pd.read_csv("metal_rods_spc_project.csv", sep="\t")
    checkpoint_df = generate_checkpoint_dataset(df)

    print("=" * 70)
    print("SINGLE-ROD CHECKPOINT PREDICTION — PROGRESSIVE RESULTS")
    print("=" * 70)

    stages = train_progressive_checkpoint_predictors(checkpoint_df)

    print(f"{'Checkpoints':<14}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    print("-" * 62)
    for n, result in stages.items():
        m = result["metrics"]
        print(f"{n:<14}{m['accuracy']:<12.3f}{m['precision']:<12.3f}{m['recall']:<12.3f}{m['f1']:<12.3f}")

    print()
    print("Feature importances at Stage 2 (Checkpoint 1 + 2):")
    print("-" * 62)
    print(stages[2]["metrics"]["feature_importances"].head(6).to_string())

    print()
    print("Example: watching ONE rod's risk update as it moves through production")
    print("-" * 62)
    # a rod drifting low early — simulate its checkpoint sequence
    example_checkpoints = [9.93, 9.96]

    for n in range(1, 3):
        result = predict_rod_risk(
            stages,
            checkpoint_readings=example_checkpoints[:n],
            machine="M3", operator="B", shift="Afternoon",
        )
        print(f"After checkpoint {n}: predicted risk = {result['predicted_risk']*100:.1f}%  "
              f"(this stage's model recall: {result['recall_at_this_stage']:.2f})")
