from pathlib import Path
import pandas as pd

base = Path("outputs/final_evaluation")
difficult = Path("outputs/difficult_models")
out = Path("outputs/final_evaluation")

main = pd.read_csv(base / "results_per_fold.csv")

# Keep only successfully completed main-evaluation folds.
main = main[main["status"] == "ok"].copy()

frames = [main]

for name, folder in [
    ("H4 CNN-BiLSTM", "H4_CNN-BiLSTM"),
    ("M24 TabNet", "M24_TabNet"),
]:
    path = difficult / folder / "results_per_fold.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    frames.append(df)

all_results = pd.concat(frames, ignore_index=True)

metrics = [
    "AUC", "F1", "IoU", "Precision",
    "Recall", "Accuracy",
]

summary = (
    all_results
    .groupby(["cv", "model"], as_index=False)
    .agg({
        **{
            metric: ["mean", "std", "count"]
            for metric in metrics
        }
    })
)

summary.columns = [
    "_".join(str(x) for x in col if x).rstrip("_")
    if isinstance(col, tuple) else col
    for col in summary.columns
]

# Preserve the run configuration as a qualification.
summary["configuration"] = summary["model"].map(
    lambda m: "FAST configuration; provisional"
    if m in {"H4 CNN-BiLSTM", "M24 TabNet"}
    else "Main evaluation configuration"
)

for cv in ["block", "event"]:
    part = summary[summary["cv"] == cv].copy()

    part["AUC_rank"] = part["AUC_mean"].rank(
        ascending=False, method="min"
    )
    part["F1_rank"] = part["F1_mean"].rank(
        ascending=False, method="min"
    )
    part["IoU_rank"] = part["IoU_mean"].rank(
        ascending=False, method="min"
    )

    part = part.sort_values(
        ["AUC_mean", "F1_mean", "IoU_mean"],
        ascending=False,
    )

    path = out / f"combined_{cv}_leaderboard.csv"
    part.to_csv(path, index=False)

    print(f"\n{'=' * 85}")
    print(f"{cv.upper()} CV LEADERBOARD")
    print("=" * 85)

    cols = [
        "model",
        "AUC_mean", "AUC_std",
        "F1_mean", "F1_std",
        "IoU_mean", "IoU_std",
        "Accuracy_mean",
        "AUC_rank", "F1_rank", "IoU_rank",
        "configuration",
    ]

    print(part[cols].to_string(index=False, float_format="%.4f"))
    print(f"\nSaved: {path}")

print("\nFOLD COMPLETENESS")
print(all_results.groupby(["cv", "model"])["fold"].nunique().to_string())

print("\nTOTAL SUCCESSFUL FOLDS:", len(all_results))
print("UNIQUE MODELS:", all_results["model"].nunique())
