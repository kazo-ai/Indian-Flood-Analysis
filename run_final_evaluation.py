import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

import models as M
import importlib.util

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "final_evaluation"
OUT.mkdir(parents=True, exist_ok=True)

RESULTS = OUT / "results_per_fold.csv"
OOF = OUT / "oof_predictions.csv"
FAILURES = OUT / "failures.csv"

# Import existing training utilities without running its main().
spec = importlib.util.spec_from_file_location(
    "training_module", ROOT / "03_train_models.py"
)
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)


def atomic_save(df, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def load_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main():
    fast = "--full" not in sys.argv
    requested = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    print("Configuration:", "fast" if fast else "full")
    print("Output:", OUT)

    df, X, y, features = T.load_dataset()
    print(
        f"Dataset: {len(df):,} rows | {len(features)} features | "
        f"{df['block'].nunique()} blocks | {df['event'].nunique()} events"
    )

    baselines, hybrids = M.get_models(fast=fast)
    all_models = {**baselines, **hybrids}
    baseline_names = set(baselines)

    # H4 CNN-BiLSTM is excluded from the main evaluation
    # because its full configuration stalled.
    excluded_models = {"H4 CNN-BiLSTM", "M24 TabNet"}
    all_models = {
        name: model
        for name, model in all_models.items()
        if name not in excluded_models
    }
    print("Excluded models:", sorted(excluded_models))
    if requested:
        lookup = {name.casefold(): name for name in all_models}
        selected = {}
        for item in requested:
            key = item.casefold()
            if key not in lookup:
                raise ValueError(f"Unknown model: {item}. Available: {list(all_models)}")
            selected[lookup[key]] = all_models[lookup[key]]
        all_models = selected
    print(f"Available models: {len(all_models)}")

    protocols = [
        ("block", 3),
        ("event", 4),
    ]

    results = load_csv(RESULTS)
    oof = load_csv(OOF)
    failures = load_csv(FAILURES)

    for cv, n_folds in protocols:
        splits = T.make_splits(df, X, y, cv, n_folds)

        # Validate splits are identical across all models.
        print(f"\n{cv.upper()} CV: {len(splits)} folds")

        for model_name, base_model in all_models.items():
            for fold, (train_idx, test_idx) in enumerate(splits):
                done = (
                    not results.empty
                    and (
                        (results["cv"] == cv)
                        & (results["model"] == model_name)
                        & (pd.to_numeric(results["fold"], errors="coerce") == fold)
                        & (results["status"] == "ok")
                    ).any()
                )
                if done:
                    continue

                print(
                    f"\n[{cv} | {model_name} | fold {fold + 1}/{len(splits)}]"
                )
                started = time.time()

                try:
                    X_train = X.iloc[train_idx].copy()
                    X_test = X.iloc[test_idx].copy()

                    # Existing graph models require coordinates.
                    if model_name in {
                        "M17 GNN + XGBoost",
                        "M18 Graph Attention Network",
                        "M19 GraphSAGE + XGBoost",
                    }:
                        coords = ["lon", "lat"]
                        if not all(c in df.columns for c in coords):
                            raise ValueError("Graph model requires lon and lat.")
                        X_train = np.column_stack([
                            X_train.to_numpy(),
                            df.iloc[train_idx][coords].to_numpy(),
                        ])
                        X_test = np.column_stack([
                            X_test.to_numpy(),
                            df.iloc[test_idx][coords].to_numpy(),
                        ])

                    y_train = y.iloc[train_idx]
                    y_test = y.iloc[test_idx]

                    model = clone(base_model)
                    model.fit(X_train, y_train)
                    prob = T.get_probabilities(model, X_test)

                    if len(prob) != len(test_idx):
                        raise ValueError("Prediction count mismatch.")

                    metrics = T.calculate_metrics(y_test, prob)
                    row = {
                        "cv": cv,
                        "model": model_name,
                        "fold": fold,
                        "kind": "baseline" if model_name in baseline_names else "hybrid",
                        "status": "ok",
                        "train_rows": len(train_idx),
                        "test_rows": len(test_idx),
                        "train_events": ",".join(sorted(df.iloc[train_idx]["event"].astype(str).unique())),
                        "test_events": ",".join(sorted(df.iloc[test_idx]["event"].astype(str).unique())),
                        "train_blocks": df.iloc[train_idx]["block"].nunique(),
                        "test_blocks": df.iloc[test_idx]["block"].nunique(),
                        "elapsed_seconds": round(time.time() - started, 2),
                        **metrics,
                    }

                    pred = pd.DataFrame({
                        "cv": cv,
                        "model": model_name,
                        "fold": fold,
                        "row_index": test_idx,
                        "y_true": np.asarray(y_test, dtype=int),
                        "y_prob": prob,
                        "y_pred": (prob >= 0.5).astype(int),
                        "event": df.iloc[test_idx]["event"].to_numpy(),
                        "block": df.iloc[test_idx]["block"].to_numpy(),
                    })
                    for c in ("lon", "lat"):
                        if c in df.columns:
                            pred[c] = df.iloc[test_idx][c].to_numpy()

                    oof = pd.concat([oof, pred], ignore_index=True)
                    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

                    # Persist every successful fold immediately.
                    atomic_save(results, RESULTS)
                    atomic_save(oof, OOF)

                    print(
                        f"AUC={metrics['AUC']:.4f} | "
                        f"F1={metrics['F1']:.4f} | "
                        f"IoU={metrics['IoU']:.4f} | "
                        f"{row['elapsed_seconds']:.1f}s"
                    )

                except Exception as exc:
                    error = {
                        "cv": cv,
                        "model": model_name,
                        "fold": fold,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "timestamp": pd.Timestamp.now().isoformat(),
                    }
                    failures = pd.concat(
                        [failures, pd.DataFrame([error])],
                        ignore_index=True,
                    )
                    results = pd.concat([
                        results,
                        pd.DataFrame([{
                            "cv": cv,
                            "model": model_name,
                            "fold": fold,
                            "kind": "baseline" if model_name in baseline_names else "hybrid",
                            "status": "failed",
                            "error": repr(exc),
                        }]),
                    ], ignore_index=True)

                    atomic_save(results, RESULTS)
                    atomic_save(failures, FAILURES)
                    print("FAILED:", repr(exc))
                    print("Failure saved; continuing to next fold.")

    # Summarize only successful folds.
    if not results.empty:
        ok = results[results["status"] == "ok"].copy()
        if not ok.empty:
            metrics = [
                "AUC", "F1", "IoU", "Precision", "Recall", "Accuracy"
            ]
            summary = (
                ok.groupby(["cv", "model", "kind"])[metrics]
                .agg(["mean", "std", "count"])
            )
            summary.columns = [
                f"{metric}_{stat}" for metric, stat in summary.columns
            ]
            summary = summary.reset_index()
            atomic_save(summary, OUT / "leaderboard.csv")

        expected = len(all_models) * sum(n for _, n in protocols)
        print(f"\nExpected fold evaluations: {expected}")
        print("Successful folds:", int((results["status"] == "ok").sum()))
        print("Failed folds:", int((results["status"] == "failed").sum()))
        print("Results:", RESULTS)
        print("OOF predictions:", OOF)
        print("Leaderboard:", OUT / "leaderboard.csv")


if __name__ == "__main__":
    main()
