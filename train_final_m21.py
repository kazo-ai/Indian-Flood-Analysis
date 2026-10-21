"""
Train the final M21 Extra Trees pipeline on the complete processed dataset.

Run from the project root:
    python train_final_m21.py

This refits the already-selected model on all available rows after CV.
Do not use in-sample scores from this fit as a replacement for CV metrics.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config as C
import models as M

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "flood_dataset.csv"
MODEL_DIR = ROOT / "outputs" / "models"
MODEL_PATH = MODEL_DIR / "m21_extra_trees.joblib"
METADATA_PATH = MODEL_DIR / "m21_extra_trees_metadata.json"
IMPORTANCE_PATH = MODEL_DIR / "m21_extra_trees_feature_importance.csv"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = [C.LABEL_COL, "event", "block"]
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    numeric_features = [col for col in C.NUMERIC_FEATURES if col in df.columns]
    missing_numeric = [col for col in C.NUMERIC_FEATURES if col not in df.columns]
    if missing_numeric:
        raise ValueError(f"Configured numeric features missing from dataset: {missing_numeric}")

    landcover_features = sorted(
        col for col in df.columns
        if col.startswith("lc_") and col != "lc_name"
    )
    features = numeric_features + landcover_features

    if not features:
        raise ValueError("No model features were selected.")
    forbidden = {C.LABEL_COL, "lon", "lat", "event", "block", "lc_name"}
    leaked = sorted(set(features).intersection(forbidden))
    if leaked:
        raise ValueError(f"Unexpected target/metadata columns in features: {leaked}")

    X = df[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        raise ValueError(f"Features entirely missing: {all_missing}")

    y = pd.to_numeric(df[C.LABEL_COL], errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"Expected binary labels {{0, 1}}, found {sorted(y.unique())}")

    print(f"Dataset: {DATA_PATH.relative_to(ROOT)}")
    print(f"Rows: {len(df):,}")
    print(f"Features: {len(features)}")
    print(f"Events: {df['event'].nunique()}")
    print(f"Label counts:\n{y.value_counts().sort_index().to_string()}")
    print("\nFeature columns:")
    for col in features:
        print(f"  - {col}")

    # Match the full configuration used by the completed model experiments.
    model = M.make_extra_trees(seed=C.SEED, n_estimators=400)
    print("\nFitting M21 Extra Trees on the complete dataset...")
    model.fit(X, y)

    # Reload/prediction smoke test before saving final artifacts.
    probabilities = model.predict_proba(X.iloc[:10])
    if probabilities.shape != (min(10, len(X)), 2):
        raise RuntimeError(f"Unexpected predict_proba shape: {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Smoke-test predictions contain non-finite values.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Smoke-test class probabilities do not sum to 1.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Confirm that the saved artifact can be loaded and used.
    loaded_model = joblib.load(MODEL_PATH)
    reloaded_probabilities = loaded_model.predict_proba(X.iloc[:10])
    if not np.allclose(probabilities, reloaded_probabilities, atol=1e-8):
        raise RuntimeError("Saved/reloaded model predictions differ from the fitted model.")

    classifier = model.named_steps["classifier"]
    importance = pd.DataFrame({
        "feature": features,
        "importance": classifier.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    metadata = {
        "model_id": "M21",
        "model_name": "Extra Trees",
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "dataset_path": str(DATA_PATH.relative_to(ROOT)),
        "target_column": C.LABEL_COL,
        "feature_columns": features,
        "feature_count": len(features),
        "training_rows": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "event_counts": {str(k): int(v) for k, v in df["event"].value_counts().sort_index().items()},
        "seed": int(C.SEED),
        "parameters": {
            "n_estimators": 400,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "imputer": "median",
        },
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_note": (
            "This artifact was refit on all dataset rows after cross-validation. "
            "Use the previously recorded spatial/block and event-held-out CV metrics "
            "for performance estimates; training-set scores are not generalization estimates."
        ),
        "label_note": (
            "Target is Sentinel-1-derived new-open-water detection (weak label), "
            "not independently verified flood extent or an operational forecast."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSUCCESS")
    print(f"Saved model:      {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved metadata:   {METADATA_PATH.relative_to(ROOT)}")
    print(f"Saved importance: {IMPORTANCE_PATH.relative_to(ROOT)}")
    print("Reload smoke test: passed")
    print("\nTop 10 feature importances (descriptive, not causal):")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
