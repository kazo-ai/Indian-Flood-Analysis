import pandas as pd
from pathlib import Path

OUT = Path("outputs")
df = pd.read_csv(OUT / "results_per_fold_master.csv")

metrics = ["AUC", "F1", "IoU", "Precision", "Recall", "Accuracy"]
df["cv"] = df["cv"].astype(str).str.lower().str.strip()
df["model"] = df["model"].astype(str).str.strip()

df = df[df["cv"].isin(["block", "event"])].copy()

# Summarize each model and evaluation method.
summary = (
    df.groupby(["model", "cv"])[metrics]
    .agg(["mean", "std"])
)

# Flatten the MultiIndex before pivoting.
summary.columns = [
    f"{metric}_{stat}" for metric, stat in summary.columns
]
summary = summary.reset_index()

# Pivot to one row per model.
leaderboard = summary.pivot(
    index="model",
    columns="cv",
    values=[c for c in summary.columns if c not in ["model", "cv"]]
)

leaderboard.columns = [
    f"{metric}_{cv}" for metric, cv in leaderboard.columns
]
leaderboard = leaderboard.reset_index()

# Performance drops: block-CV mean minus event-CV mean.
for metric in metrics:
    block_col = f"{metric}_block_mean"
    event_col = f"{metric}_event_mean"

    if block_col in leaderboard.columns and event_col in leaderboard.columns:
        leaderboard[f"{metric}_drop"] = (
            leaderboard[block_col] - leaderboard[event_col]
        )

# Sort by event-held-out AUC where available.
if "AUC_event_mean" in leaderboard.columns:
    leaderboard = leaderboard.sort_values(
        "AUC_event_mean", ascending=False, na_position="last"
    )

out_file = OUT / "spatial_vs_event_leaderboard.csv"
leaderboard.to_csv(out_file, index=False)

show = [
    "model",
    "AUC_block_mean", "AUC_block_std",
    "AUC_event_mean", "AUC_event_std", "AUC_drop",
    "F1_block_mean", "F1_event_mean", "F1_drop",
    "IoU_block_mean", "IoU_event_mean", "IoU_drop",
]
show = [c for c in show if c in leaderboard.columns]

print("\n=== BLOCK CV vs EVENT CV ===")
print(leaderboard[show].round(4).to_string(index=False))

print(f"\nSaved full leaderboard to: {out_file}")

print("\n=== COMPLETENESS ===")
for cv, expected in [("block", 3), ("event", 4)]:
    counts = df[df["cv"] == cv].groupby("model")["fold"].nunique()
    incomplete = counts[counts != expected]
    print(f"{cv}: {len(counts)} models; expected {expected} folds each")
    if len(incomplete):
        print("Incomplete:\n", incomplete.to_string())
    else:
        print("All recorded models have the expected fold counts.")
