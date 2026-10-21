import argparse
import os
import tempfile
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    LeaveOneGroupOut,
)
import config as C
import models as M
warnings.filterwarnings("ignore", category=UserWarning)
# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "flood_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASTER_RESULTS_PATH = OUTPUT_DIR / "results_per_fold_master.csv"
MASTER_SUMMARY_PATH = OUTPUT_DIR / "summary_master.csv"
MASTER_OOF_PATH = OUTPUT_DIR / "oof_predictions_master.csv"
# ============================================================
# SAFE CSV WRITING
# ============================================================
def atomic_csv_write(df, path):
    """
    Write a CSV atomically to reduce the chance of corrupting
    a previous results file if the process is interrupted.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=path.stem + "_",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    try:
        df.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
def load_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()
# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================
def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )
    df = pd.read_csv(DATASET_PATH)
    if C.LABEL_COL not in df.columns:
        raise ValueError(
            f"Missing target column: {C.LABEL_COL}"
        )
    if "block" not in df.columns:
        raise ValueError("Dataset is missing the 'block' column.")
    if "event" not in df.columns:
        raise ValueError("Dataset is missing the 'event' column.")
    # Use the numeric predictors defined in config.py and the
    # one-hot land-cover features already present in the dataset.
    numeric_features = [
        col for col in C.NUMERIC_FEATURES
        if col in df.columns
    ]
    landcover_features = sorted([
        col for col in df.columns
        if col.startswith("lc_") and col != "lc_name"
    ])
    features = numeric_features + landcover_features
    if not features:
        raise ValueError("No usable features were found.")
    missing_features = [
        col for col in C.NUMERIC_FEATURES
        if col not in df.columns
    ]
    if missing_features:
        print(
            "WARNING: Configured numeric features absent from "
            f"dataset: {missing_features}"
        )
    # Missing values are handled inside each training fold.
    # This prevents fitting the imputer using validation samples.
    X = df[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        raise ValueError(
            "These selected features contain no usable values: "
            f"{all_missing}"
        )
    y = pd.to_numeric(
        df[C.LABEL_COL],
        errors="raise",
    ).astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(
            "The target must contain binary labels 0 and 1."
        )
    if y.nunique() != 2:
        raise ValueError(
            "Both target classes must be present."
        )
    # Remove rows with missing group identifiers.
    valid = df["block"].notna() & df["event"].notna()
    if not valid.all():
        dropped = int((~valid).sum())
        print(f"Dropping {dropped} rows without group identifiers.")
        df = df.loc[valid].reset_index(drop=True)
        X = X.loc[valid].reset_index(drop=True)
        y = y.loc[valid].reset_index(drop=True)
    return df, X, y, features
# ============================================================
# CROSS-VALIDATION
# ============================================================
def make_splits(df, X, y, cv, folds):
    if cv == "block":
        groups = df["block"].astype(str).to_numpy()
        n_groups = pd.Series(groups).nunique()
        if n_groups < folds:
            raise ValueError(
                f"Requested {folds} folds, but only "
                f"{n_groups} unique spatial blocks exist."
            )
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=C.SEED,
        )
        splits = list(
            splitter.split(X, y, groups=groups)
        )
    elif cv == "event":
        groups = df["event"].astype(str).to_numpy()
        splitter = LeaveOneGroupOut()
        splits = list(
            splitter.split(X, y, groups=groups)
        )
    else:
        raise ValueError(f"Unknown CV strategy: {cv}")
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)
        if overlap:
            raise RuntimeError(
                f"Group leakage detected in fold {fold}: "
                f"{list(overlap)[:5]}"
            )
        if len(np.unique(y.iloc[test_idx])) < 2:
            raise ValueError(
                f"Fold {fold} contains only one target class. "
                "AUC cannot be calculated reliably."
            )
    return splits
# ============================================================
# MODEL SELECTION
# ============================================================
def get_available_models(fast=False):
    baselines, hybrids = M.get_models(fast=fast)
    # Preserve the names used by the existing training script.
    return {
        **baselines,
        **hybrids,
    }
def select_models(all_models, requested):
    if not requested:
        return all_models
    # Allow both:
    # --only LR RF
    # --only LR,RF
    requested_names = []
    exact_names = {name.casefold(): name for name in all_models}

    for item in requested:
        item = item.strip()

        # Prefer an exact match, even when the model name contains commas.
        if item.casefold() in exact_names:
            requested_names.append(item)
        else:
            # Still support shorthand such as --only LR,RF.
            requested_names.extend(
                part.strip()
                for part in item.split(",")
                if part.strip()
            )
    lookup = {
        name.casefold(): name
        for name in all_models
    }

    # Canonical model IDs mapped to existing training names.
    model_id_aliases = {
        "M01": "LR",
        "M02": "RF",
        "M03": "XGB",
        "M04": "LGBM",
        "M05": "MLP",
        "M06": "H1 Stacking (RF+XGB+LGBM -> LR)",
        "M07": "H2 Soft-Voting (RF+XGB+MLP)",
        "M08": "H3 GA-XGBoost",
        "M09": "H6 RF-feature-selection + LGBM",
        "M10": "M10 CatBoost Stacking",
        "M11": "H5 MLP-embedding + XGBoost",
        "M12": "M12 Autoencoder + XGBoost",
        "M13": "M13 Transformer + XGBoost",
        "M14": "M14 FT-Transformer",
        "M15": "M15 CNN + Attention",
        "M16": "M16 DNN Ensemble",
        "M17": "M17 GNN + XGBoost",
        "M18": "M18 Graph Attention Network",
        "M19": "M19 GraphSAGE + XGBoost",
        "M20": "M20 RBF-SVM",
        "M21": "M21 Extra Trees",
        "M22": "M22 Standalone CatBoost",
        "M23": "M23 Explainable Boosting Machine",
        "M24": "M24 TabNet",
        "M25": "M25 Calibrated Ensemble",
    }

    for model_id, legacy_name in model_id_aliases.items():
        if legacy_name in all_models:
            lookup[model_id.casefold()] = legacy_name
    selected = {}
    for requested_name in requested_names:
        actual_name = lookup.get(requested_name.casefold())
        if actual_name is None:
            available = "\n".join(
                f"  {name}" for name in all_models
            )
            raise ValueError(
                f"\nUnknown model name: {requested_name}\n\n"
                "Available exact model names:\n"
                f"{available}\n\n"
                "Use the exact model name shown above."
            )
        selected[actual_name] = all_models[actual_name]
    return selected
# ============================================================
# MODEL PREDICTIONS
# ============================================================
def get_probabilities(model, X):
    """
    Return continuous scores for ROC AUC.
    predict_proba is preferred. If unavailable, use the
    decision function and map scores to a 0-1 range.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        probabilities = np.asarray(probabilities)
        if probabilities.ndim == 2:
            probabilities = probabilities[:, 1]
        return np.clip(
            probabilities.astype(float),
            0.0,
            1.0,
        )
    if hasattr(model, "decision_function"):
        scores = np.asarray(
            model.decision_function(X),
            dtype=float,
        )
        # Numerically stable sigmoid.
        scores = np.clip(scores, -50, 50)
        return 1.0 / (1.0 + np.exp(-scores))
    return np.asarray(
        model.predict(X),
        dtype=float,
    )
# ============================================================
# METRICS
# ============================================================
def calculate_metrics(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (
        probabilities >= 0.5
    ).astype(int)
    auc = roc_auc_score(
        y_true,
        probabilities,
    )
    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )
    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )
    accuracy = accuracy_score(
        y_true,
        predictions,
    )
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()
    union = tp + fp + fn
    iou = tp / union if union else 0.0
    return {
        "AUC": float(auc),
        "F1": float(f1),
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "Accuracy": float(accuracy),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
# ============================================================
# MASTER RESULTS
# ============================================================
def load_master_results():
    master = load_csv_if_exists(MASTER_RESULTS_PATH)
    if not master.empty:
        if "cv" not in master.columns:
            master["cv"] = "block"
        return master
    # Migrate the results from the original training script
    # if the master file has not yet been created.
    legacy_path = OUTPUT_DIR / "results_per_fold_block.csv"
    legacy = load_csv_if_exists(legacy_path)
    if not legacy.empty and "model" in legacy.columns:
        if "cv" not in legacy.columns:
            legacy["cv"] = "block"
        print(
            "Initializing the master results table from "
            "outputs/results_per_fold_block.csv"
        )
        return legacy
    return pd.DataFrame()
def update_master_results(new_results):
    master = load_master_results()
    combined = pd.concat(
        [master, pd.DataFrame(new_results)],
        ignore_index=True,
        sort=False,
    )
    if combined.empty:
        return combined
    # A repeated run replaces that model's results for the
    # corresponding CV strategy and fold.
    keys = ["cv", "model", "fold"]
    combined = combined.drop_duplicates(
        subset=keys,
        keep="last",
    )
    combined = combined.sort_values(
        ["cv", "model", "fold"]
    ).reset_index(drop=True)
    atomic_csv_write(
        combined,
        MASTER_RESULTS_PATH,
    )
    return combined
def create_summary(master):
    if master.empty:
        return pd.DataFrame()
    metric_columns = [
        col for col in [
            "AUC",
            "F1",
            "IoU",
            "Precision",
            "Recall",
            "Accuracy",
        ]
        if col in master.columns
    ]
    summary = (
        master.groupby(
            ["cv", "kind", "model"],
            dropna=False,
        )[metric_columns]
        .mean()
        .reset_index()
    )
    if "AUC" in summary.columns:
        summary = summary.sort_values(
            ["cv", "AUC"],
            ascending=[True, False],
        )
    atomic_csv_write(
        summary,
        MASTER_SUMMARY_PATH,
    )
    return summary
# ============================================================
# MASTER OUT-OF-FOLD PREDICTIONS
# ============================================================
def load_master_oof():
    master = load_csv_if_exists(MASTER_OOF_PATH)
    if not master.empty:
        return master
    # Migrate the previous script's wide-format OOF predictions
    # into long format if possible.
    legacy_path = OUTPUT_DIR / "oof_predictions_block.csv"
    legacy = load_csv_if_exists(legacy_path)
    if legacy.empty:
        return pd.DataFrame()
    target_column = next(
        (
            col for col in ["y_true", "y", C.LABEL_COL]
            if col in legacy.columns
        ),
        None,
    )
    if target_column is None:
        print(
            "Legacy OOF file found, but its target column "
            "could not be identified. Keeping the original "
            "file unchanged."
        )
        return pd.DataFrame()
    migrated = []
    reserved = {
        target_column,
        "row_index",
        "index",
        "fold",
        "event",
        "block",
        "lon",
        "lat",
        "cv",
        "kind",
    }
    for col in legacy.columns:
        if col in reserved:
            continue
        probabilities = pd.to_numeric(
            legacy[col],
            errors="coerce",
        )
        if probabilities.isna().all():
            continue
        actual = pd.to_numeric(
            legacy[target_column],
            errors="coerce",
        )
        valid = probabilities.notna() & actual.notna()
        if not valid.any():
            continue
        migrated.append(
            pd.DataFrame({
                "cv": "block",
                "model": col,
                "fold": -1,
                "row_index": np.arange(len(legacy))[valid],
                "y_true": actual[valid].astype(int).to_numpy(),
                "y_prob": probabilities[valid].to_numpy(),
                "y_pred": (
                    probabilities[valid].to_numpy() >= 0.5
                ).astype(int),
                "kind": "legacy",
            })
        )
    if not migrated:
        return pd.DataFrame()
    return pd.concat(
        migrated,
        ignore_index=True,
        sort=False,
    )
def update_master_oof(new_predictions):
    master = load_master_oof()

    if isinstance(new_predictions, list):
        new_df = (
            pd.concat(
                new_predictions,
                ignore_index=True,
                sort=False,
            )
            if new_predictions
            else pd.DataFrame()
        )
    else:
        new_df = pd.DataFrame(new_predictions)

    combined = pd.concat(
        [master, new_df],
        ignore_index=True,
        sort=False,
    )

    if combined.empty:
        return combined

    keys = ["cv", "model", "fold", "row_index"]

    combined = combined.drop_duplicates(
        subset=keys,
        keep="last",
    )

    atomic_csv_write(combined, MASTER_OOF_PATH)

    return combined


# ============================================================
# COMPARISON FIGURE
# ============================================================
def create_comparison_plot(summary, cv):
    if summary.empty:
        return
    subset = summary[
        summary["cv"].astype(str) == cv
    ].copy()
    if subset.empty:
        return
    subset = subset.sort_values(
        "AUC",
        ascending=True,
    )
    height = max(
        5,
        0.42 * len(subset),
    )
    fig, ax = plt.subplots(
        figsize=(11, height)
    )
    y_positions = np.arange(len(subset))
    ax.barh(
        y_positions - 0.18,
        subset["AUC"],
        height=0.34,
        label="AUC",
    )
    ax.barh(
        y_positions + 0.18,
        subset["F1"],
        height=0.34,
        label="F1",
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(subset["model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title(
        f"Model Comparison — {cv} Cross-Validation"
    )
    ax.grid(
        axis="x",
        alpha=0.25,
    )
    ax.legend()
    fig.tight_layout()
    output_path = (
        OUTPUT_DIR /
        f"model_comparison_{cv}_master.png"
    )
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)
# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Flood susceptibility model training with "
            "spatial-block or event-based cross-validation."
        )
    )
    parser.add_argument(
        "--cv",
        choices=["block", "event"],
        default="block",
        help="Cross-validation strategy.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of spatial-block folds.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=(
            "Exact model names. Example: "
            '--only LR RF or --only "H1 Stacking (RF+XGB+LGBM -> LR)"'
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the existing fast model configurations.",
    )
    args = parser.parse_args()
    started = time.time()
    print("=" * 68)
    print("FLOOD SUSCEPTIBILITY — MODEL TRAINING")
    print("=" * 68)
    df, X, y, features = load_dataset()
    print(f"Dataset rows:       {len(df):,}")
    print(f"Features:           {len(features)}")
    print(f"Unique blocks:      {df['block'].nunique()}")
    print(f"Events:             {df['event'].nunique()}")
    print(f"Flooded proportion: {y.mean():.3f}")
    print(f"\nCross-validation: {args.cv}")
    splits = make_splits(
        df,
        X,
        y,
        args.cv,
        args.folds,
    )
    print(f"Number of folds: {len(splits)}")
    
    baselines, hybrids = M.get_models(fast=args.fast)

    all_models = {
        **baselines,
        **hybrids,
    }

    baseline_names = set(baselines.keys())

    selected_models = select_models(
        all_models,
        args.only,
    )

    # Skip models whose folds for this CV strategy are already complete.
    existing_results = load_master_results()
    skipped_models = []

    if not existing_results.empty and "cv" in existing_results.columns:
        cv_results = existing_results[
            existing_results["cv"].astype(str).eq(args.cv)
        ].copy()

        expected_folds = set(range(len(splits)))

        for model_name in list(selected_models):
            model_rows = cv_results[
                cv_results["model"].astype(str).eq(model_name)
            ]

            if not model_rows.empty:
                saved_folds = set(
                    pd.to_numeric(
                        model_rows["fold"], errors="coerce"
                    ).dropna().astype(int)
                )

                if expected_folds.issubset(saved_folds):
                    skipped_models.append(model_name)
                    del selected_models[model_name]

    if skipped_models:
        print("\nSkipping models with completed folds:")
        for name in skipped_models:
            print(f"  - {name}")

    print("\nModels selected:")
    for name in selected_models:
        print(f"  - {name}")
    print("\nStarting training...\n")
    run_results = []
    run_predictions = []
    for model_name, base_model in selected_models.items():
        print(f"\n{'-' * 68}")
        print(f"MODEL: {model_name}")
        print(f"{'-' * 68}")
        model_started = time.time()
        for fold, (train_idx, test_idx) in enumerate(splits):
            fold_started = time.time()
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            graph_model_names = {
                "M17 GNN + XGBoost",
                "M18 Graph Attention Network",
                "M19 GraphSAGE + XGBoost",
            }
            if model_name in graph_model_names:
                coordinate_cols = ["lon", "lat"]
                missing_coords = [
                    col for col in coordinate_cols if col not in df.columns
                ]
                if missing_coords:
                    raise ValueError(
                        f"Graph model requires coordinates: {missing_coords}"
                    )
                X_train = np.column_stack([
                    X_train.to_numpy(),
                    df.iloc[train_idx][coordinate_cols].to_numpy(),
                ])
                X_test = np.column_stack([
                    X_test.to_numpy(),
                    df.iloc[test_idx][coordinate_cols].to_numpy(),
                ])

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            train_events = sorted(
                df.iloc[train_idx]["event"].unique()
            )
            test_events = sorted(
                df.iloc[test_idx]["event"].unique()
            )
            print(
                f"  Fold {fold + 1}/{len(splits)} | "
                f"train={len(train_idx):,} | "
                f"test={len(test_idx):,}"
            )
            try:
                model = clone(base_model)
                model.fit(
                    X_train,
                    y_train,
                )
                probabilities = get_probabilities(
                    model,
                    X_test,
                )
                if len(probabilities) != len(test_idx):
                    raise ValueError(
                        "Prediction count does not match "
                        "the validation sample count."
                    )
                metrics = calculate_metrics(
                    y_test,
                    probabilities,
                )
                row = {
                    "cv": args.cv,
                    "model": model_name,
                    "fold": fold,
                    "kind": (
                             "baseline"
                        if model_name in baseline_names
                         else "hybrid"
                            ),
                    "train_rows": len(train_idx),
                    "test_rows": len(test_idx),
                    "train_events": ",".join(train_events),
                    "test_events": ",".join(test_events),
                    "train_blocks": df.iloc[
                        train_idx
                    ]["block"].nunique(),
                    "test_blocks": df.iloc[
                        test_idx
                    ]["block"].nunique(),
                    **metrics,
                    "elapsed_seconds": round(
                        time.time() - fold_started,
                        2,
                    ),
                }
                run_results.append(row)
                # Save each completed fold immediately.
                master = update_master_results(
                    run_results
                )
                # Record predictions only for the held-out fold.
                pred_df = pd.DataFrame({
                    "cv": args.cv,
                    "model": model_name,
                    "fold": fold,
                    "row_index": test_idx,
                    "y_true": np.asarray(
                        y_test,
                        dtype=int,
                    ),
                    "y_prob": probabilities,
                    "y_pred": (
                        probabilities >= 0.5
                    ).astype(int),
                    "event": df.iloc[
                        test_idx
                    ]["event"].to_numpy(),
                    "block": df.iloc[
                        test_idx
                    ]["block"].to_numpy(),
                })
                for coordinate in ["lon", "lat"]:
                    if coordinate in df.columns:
                        pred_df[coordinate] = (
                            df.iloc[test_idx][
                                coordinate
                            ].to_numpy()
                        )
                run_predictions.append(pred_df)
                update_master_oof(
                    run_predictions
                )
                print(
                    f"    AUC={metrics['AUC']:.4f} | "
                    f"F1={metrics['F1']:.4f} | "
                    f"IoU={metrics['IoU']:.4f} | "
                    f"Accuracy={metrics['Accuracy']:.4f}"
                )
            except Exception as exc:
                print(
                    f"    ERROR in fold {fold + 1}: "
                    f"{type(exc).__name__}: {exc}"
                )
                # Keep completed folds; continue to the next fold.
                continue
        # Rebuild the master table after each model.
        master = load_master_results()
        if not master.empty:
            summary = create_summary(master)
            create_comparison_plot(
                summary,
                args.cv,
            )
        print(
            f"\nFinished {model_name} in "
            f"{time.time() - model_started:.1f}s"
        )
    # Save current-run files separately, without overwriting
    # the original results_per_fold_block.csv.
    if run_results:
        current_results = pd.DataFrame(run_results)
        atomic_csv_write(
            current_results,
            OUTPUT_DIR /
            f"results_per_fold_{args.cv}_latest.csv",
        )
        current_summary = (
            current_results.groupby(
                ["cv", "kind", "model"],
                dropna=False,
            )[
                [
                    "AUC",
                    "F1",
                    "IoU",
                    "Precision",
                    "Recall",
                    "Accuracy",
                ]
            ]
            .mean()
            .reset_index()
            .sort_values(
                "AUC",
                ascending=False,
            )
        )
        atomic_csv_write(
            current_summary,
            OUTPUT_DIR /
            f"summary_{args.cv}_latest.csv",
        )
    if run_predictions:
        current_oof = pd.concat(
            run_predictions,
            ignore_index=True,
        )
        atomic_csv_write(
            current_oof,
            OUTPUT_DIR /
            f"oof_predictions_{args.cv}_latest.csv",
        )
    master = load_master_results()
    if not master.empty:
        summary = create_summary(master)
        create_comparison_plot(
            summary,
            args.cv,
        )
        current_summary = summary[
            summary["cv"].astype(str) == args.cv
        ].copy()
        if not current_summary.empty:
            print("\n" + "=" * 88)
            print("CUMULATIVE MODEL RESULTS")
            print("=" * 88)
            display_columns = [
                col for col in [
                    "kind",
                    "model",
                    "AUC",
                    "F1",
                    "IoU",
                    "Precision",
                    "Recall",
                    "Accuracy",
                ]
                if col in current_summary.columns
            ]
            print(
                current_summary[
                    display_columns
                ].round(4).to_string(index=False)
            )
    print("\n" + "=" * 68)
    print("TRAINING RUN COMPLETE")
    print("=" * 68)
    print(f"Elapsed time: {time.time() - started:.1f}s")
    print("\nPersistent files:")
    print(f"  {MASTER_RESULTS_PATH}")
    print(f"  {MASTER_SUMMARY_PATH}")
    print(f"  {MASTER_OOF_PATH}")
    print("\nCurrent-run files:")
    print(
        OUTPUT_DIR /
        f"results_per_fold_{args.cv}_latest.csv"
    )
    print(
        OUTPUT_DIR /
        f"summary_{args.cv}_latest.csv"
    )
    print(
        OUTPUT_DIR /
        f"oof_predictions_{args.cv}_latest.csv"
    )
if __name__ == "__main__":
    main()