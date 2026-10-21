import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

import config as C
import models as M


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "processed" / "flood_dataset.csv"
OUT = ROOT / "outputs" / "interpretability"
OUT.mkdir(parents=True, exist_ok=True)

SEED = C.SEED

NUMERIC_FEATURES = [
    "elevation",
    "slope",
    "curvature",
    "twi",
    "hand",
    "dist_river",
    "rain_event_total",
    "rain_max_1day",
    "rain_max_3day",
    "rain_antecedent_30d",
    "ndvi_pre",
    "clay",
    "sand",
]

LANDCOVER_FEATURES = [
    "lc_bare",
    "lc_built",
    "lc_crop",
    "lc_grass",
    "lc_shrub",
    "lc_tree",
    "lc_water",
    "lc_wetland",
]

FEATURES = NUMERIC_FEATURES + LANDCOVER_FEATURES

GROUPS = {
    "Terrain": ["elevation", "slope", "curvature"],
    "Hydrology": ["twi", "hand", "dist_river"],
    "Rainfall": [
        "rain_event_total",
        "rain_max_1day",
        "rain_max_3day",
        "rain_antecedent_30d",
    ],
    "Vegetation": ["ndvi_pre"],
    "Soil": ["clay", "sand"],
    "Land cover": LANDCOVER_FEATURES,
}


def load_data():
    df = pd.read_csv(DATASET)

    required = FEATURES + [C.LABEL_COL, "block", "event"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Match the training script: retain rows with valid grouping identifiers.
    valid = df["block"].notna() & df["event"].notna()
    df = df.loc[valid].reset_index(drop=True)

    X = df[FEATURES].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if X.columns[X.isna().all()].tolist():
        raise ValueError("At least one selected feature is entirely missing.")

    y = pd.to_numeric(df[C.LABEL_COL], errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("Expected binary target labels 0 and 1.")

    return df, X, y


def make_splits(df, X, y, cv, folds):
    if cv == "block":
        groups = df["block"].astype(str).to_numpy()
        if len(np.unique(groups)) < folds:
            raise ValueError("Number of unique blocks is smaller than folds.")

        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=SEED,
        )
        splits = list(splitter.split(X, y, groups=groups))

    elif cv == "event":
        groups = df["event"].astype(str).to_numpy()
        splitter = LeaveOneGroupOut()
        splits = list(splitter.split(X, y, groups=groups))

    else:
        raise ValueError(f"Unknown CV strategy: {cv}")

    # Reproduce the original training script's leakage and AUC checks.
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        if train_groups.intersection(test_groups):
            raise RuntimeError(f"Group leakage in fold {fold}.")

        if y.iloc[test_idx].nunique() < 2:
            raise ValueError(f"Fold {fold} contains only one target class.")

    return splits


def score_auc(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, probabilities)


def run_cv(df, X, y, cv, folds, repeats):
    splits = make_splits(df, X, y, cv, folds)

    # This is the same model factory/configuration as the original M21.
    base_model = M.make_extra_trees(seed=SEED, n_estimators=400)

    records = []

    print(f"\n{'=' * 65}")
    print(f"CV strategy: {cv} | folds: {len(splits)}")
    print(f"Features: {len(FEATURES)} | permutation repeats: {repeats}")
    print(f"{'=' * 65}")

    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\nFold {fold + 1}/{len(splits)}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = M.make_extra_trees(seed=SEED, n_estimators=400)
        model.fit(X_train, y_train)

        baseline_auc = score_auc(model, X_test, y_test)
        print(
            f"Train rows: {len(train_idx):,} | "
            f"Test rows: {len(test_idx):,} | "
            f"Held-out AUC: {baseline_auc:.4f}"
        )

        # Permutation is performed only on the held-out fold.
        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=repeats,
            random_state=SEED + fold,
            n_jobs=1,
        )

        for i, feature in enumerate(FEATURES):
            records.append({
                "cv": cv,
                "fold": fold,
                "feature": feature,
                "baseline_auc": baseline_auc,
                "importance_mean": float(result.importances_mean[i]),
                "importance_std_within_fold": float(result.importances_std[i]),
                "test_rows": len(test_idx),
                "test_events": ",".join(
                    sorted(df.iloc[test_idx]["event"].astype(str).unique())
                ),
            })

        top = np.argsort(result.importances_mean)[::-1][:5]
        print("Top 5 features in this fold:")
        for i in top:
            print(
                f"  {FEATURES[i]:25s} "
                f"{result.importances_mean[i]:+.5f}"
            )

    per_fold = pd.DataFrame(records)
    per_fold_path = OUT / f"m21_permutation_importance_{cv}_per_fold.csv"
    per_fold.to_csv(per_fold_path, index=False)

    # Each fold contributes one mean importance estimate per feature.
    summary = (
        per_fold.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_sd_across_folds=("importance_mean", "std"),
            folds=("fold", "nunique"),
            mean_baseline_auc=("baseline_auc", "mean"),
        )
    )
    summary["importance_sd_across_folds"] = (
        summary["importance_sd_across_folds"].fillna(0.0)
    )
    summary["cv"] = cv
    summary = summary.sort_values("importance_mean", ascending=False)

    summary_path = OUT / f"m21_permutation_importance_{cv}_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Publication-style individual-feature figure.
    plot_df = summary.sort_values("importance_mean", ascending=True)
    fig_height = max(6, 0.34 * len(plot_df))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.barh(
        plot_df["feature"],
        plot_df["importance_mean"],
        xerr=plot_df["importance_sd_across_folds"],
        capsize=2,
    )
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Decrease in held-out ROC AUC after permutation")
    ax.set_ylabel("Feature")
    ax.set_title(f"M21 Held-out Permutation Importance — {cv.title()} CV")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    figure_path = OUT / f"m21_permutation_importance_{cv}.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Group-level sums are not valid for permutation importance.
    # Instead, jointly shuffle every feature in each group using a shared row
    # permutation, then measure the held-out AUC decrease.
    group_records = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        model = M.make_extra_trees(seed=SEED, n_estimators=400)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx]
        baseline_auc = score_auc(model, X_test, y_test)
        rng = np.random.default_rng(SEED + 1000 + fold)

        for group_name, group_features in GROUPS.items():
            decreases = []

            for _ in range(repeats):
                X_permuted = X_test.copy()
                permutation = rng.permutation(len(X_test))

                # Joint permutation preserves the relationships among
                # variables within the group, but breaks their association
                # with the outcome and the other feature groups.
                X_permuted.loc[:, group_features] = (
                    X_test.iloc[permutation][group_features].to_numpy()
                )

                permuted_auc = score_auc(model, X_permuted, y_test)
                decreases.append(baseline_auc - permuted_auc)

            group_records.append({
                "cv": cv,
                "fold": fold,
                "feature_group": group_name,
                "baseline_auc": baseline_auc,
                "importance_mean": float(np.mean(decreases)),
                "importance_sd_within_fold": float(np.std(decreases, ddof=1))
                    if len(decreases) > 1 else 0.0,
            })

    group_df = pd.DataFrame(group_records)
    group_df.to_csv(
        OUT / f"m21_group_permutation_importance_{cv}_per_fold.csv",
        index=False,
    )

    group_summary = (
        group_df.groupby("feature_group", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_sd_across_folds=("importance_mean", "std"),
            folds=("fold", "nunique"),
        )
    )
    group_summary["importance_sd_across_folds"] = (
        group_summary["importance_sd_across_folds"].fillna(0.0)
    )
    group_summary["cv"] = cv
    group_summary = group_summary.sort_values(
        "importance_mean",
        ascending=False,
    )
    group_summary.to_csv(
        OUT / f"m21_group_permutation_importance_{cv}_summary.csv",
        index=False,
    )

    group_plot = group_summary.sort_values(
        "importance_mean",
        ascending=True,
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(
        group_plot["feature_group"],
        group_plot["importance_mean"],
        xerr=group_plot["importance_sd_across_folds"],
        capsize=3,
    )
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Decrease in held-out ROC AUC after joint group permutation")
    ax.set_ylabel("Feature group")
    ax.set_title(f"M21 Group Permutation Importance — {cv.title()} CV")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        OUT / f"m21_group_permutation_importance_{cv}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nSaved:")
    print(f"  {per_fold_path}")
    print(f"  {summary_path}")
    print(f"  {figure_path}")
    print(f"  {OUT / f'm21_group_permutation_importance_{cv}_summary.csv'}")
    print(f"  {OUT / f'm21_group_permutation_importance_{cv}.png'}")

    print("\nTop 10 individual features:")
    print(summary.head(10).to_string(index=False))

    print("\nFeature-group results:")
    print(group_summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Held-out permutation importance for M21 Extra Trees."
    )
    parser.add_argument(
        "--cv",
        choices=["block", "event", "both"],
        default="both",
        help="CV strategy to evaluate.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Spatial-block folds; original experiment uses 3.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Permutation repeats per feature and fold.",
    )
    args = parser.parse_args()

    if args.repeats < 2:
        parser.error("--repeats must be at least 2.")

    df, X, y = load_data()
    print(f"Dataset rows: {len(df):,}")
    print(f"Feature count: {len(FEATURES)}")
    print(f"Events: {sorted(df['event'].astype(str).unique())}")
    print(f"Spatial blocks: {df['block'].nunique()}")

    strategies = ["block", "event"] if args.cv == "both" else [args.cv]

    for cv in strategies:
        run_cv(df, X, y, cv, args.folds, args.repeats)

    metadata = {
        "model": "M21 Extra Trees",
        "n_estimators": 400,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "random_state": SEED,
        "features": FEATURES,
        "primary_metric": "ROC AUC",
        "importance_definition": (
            "Held-out ROC AUC decrease after feature permutation; "
            "group permutation jointly shuffles all columns in a group."
        ),
        "caveat": (
            "Permutation importance measures predictive reliance, not causality. "
            "Correlated predictors can share or mask importance."
        ),
    }
    with open(OUT / "m21_permutation_importance_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
