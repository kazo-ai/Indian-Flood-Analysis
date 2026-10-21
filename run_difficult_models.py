import sys
import time
import traceback
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import models as M
from sklearn.base import clone

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "difficult_models"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "training_module", ROOT / "03_train_models.py"
)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)

def save(df, path):
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def main():
    requested = sys.argv[1:] or ["H4", "M24"]
    df, X, y, features = T.load_dataset()

    baselines, hybrids = M.get_models(fast=True)
    all_models = {**baselines, **hybrids}

    aliases = {
        "H4": "H4 CNN-BiLSTM",
        "M24": "M24 TabNet",
    }

    for requested_name in requested:
        name = aliases.get(requested_name.upper(), requested_name)
        if name not in all_models:
            raise ValueError(f"Model unavailable: {name}")

        model_dir = OUT / name.replace(" ", "_").replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        results_path = model_dir / "results_per_fold.csv"
        predictions_path = model_dir / "oof_predictions.csv"
        failures_path = model_dir / "failures.csv"

        results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
        oof = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
        failures = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()

        print(f"\n{'='*65}\n{name}\nOutput: {model_dir}\nConfiguration: FAST")

        for cv, n_folds in [("block", 3), ("event", 4)]:
            splits = T.make_splits(df, X, y, cv, n_folds)

            for fold, (train_idx, test_idx) in enumerate(splits):
                if (
                    not results.empty
                    and {"cv", "fold", "status"}.issubset(results.columns)
                    and (
                        (results["cv"] == cv)
                        & (pd.to_numeric(results["fold"], errors="coerce") == fold)
                        & (results["status"] == "ok")
                    ).any()
                ):
                    print(f"Skipping completed {cv} fold {fold}")
                    continue

                print(f"\n{cv.upper()} fold {fold + 1}/{len(splits)}")
                started = time.time()

                try:
                    X_train = X.iloc[train_idx].copy()
                    X_test = X.iloc[test_idx].copy()

                    model = clone(all_models[name])
                    model.fit(X_train, y.iloc[train_idx])
                    prob = T.get_probabilities(model, X_test)
                    metrics = T.calculate_metrics(y.iloc[test_idx], prob)

                    row = {
                        "cv": cv, "model": name, "fold": fold,
                        "status": "ok",
                        "train_rows": len(train_idx),
                        "test_rows": len(test_idx),
                        "elapsed_seconds": round(time.time() - started, 2),
                        **metrics,
                    }
                    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                    save(results, results_path)

                    pred = pd.DataFrame({
                        "cv": cv, "model": name, "fold": fold,
                        "row_index": test_idx,
                        "y_true": y.iloc[test_idx].to_numpy(),
                        "y_prob": prob,
                        "y_pred": (prob >= 0.5).astype(int),
                        "event": df.iloc[test_idx]["event"].to_numpy(),
                        "block": df.iloc[test_idx]["block"].to_numpy(),
                    })
                    oof = pd.concat([oof, pred], ignore_index=True)
                    save(oof, predictions_path)

                    print(
                        f"AUC={metrics['AUC']:.4f} "
                        f"F1={metrics['F1']:.4f} "
                        f"IoU={metrics['IoU']:.4f}"
                    )

                except Exception as exc:
                    failure = {
                        "cv": cv, "model": name, "fold": fold,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    failures = pd.concat([failures, pd.DataFrame([failure])], ignore_index=True)
                    save(failures, failures_path)
                    print("FAILED:", repr(exc))
                    print("Failure saved; continuing.")

        if not results.empty:
            successful = results[results["status"] == "ok"]
            if not successful.empty:
                metrics = ["AUC", "F1", "IoU", "Precision", "Recall", "Accuracy"]
                summary = successful.groupby("cv")[metrics].agg(["mean", "std"])
                summary.columns = [f"{m}_{s}" for m, s in summary.columns]
                save(summary.reset_index(), model_dir / "summary.csv")

        print(f"\nSaved results for {name} in {model_dir}")

if __name__ == "__main__":
    main()
