#!/usr/bin/env python3
"""
Generate publication-style PNG figures for the flood susceptibility project.

Run from the project root:
    python generate_research_figures.py

Inputs are existing outputs only; this script does not train models.
Figures are written to outputs/figures/.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score, f1_score, jaccard_score,
    precision_score, recall_score, accuracy_score, brier_score_loss
)
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Use corrected evaluation outputs; avoid potentially stale master files.
FOLD_PATH = OUT / "results_per_fold_master.csv"
OOF_PATH = OUT / "oof_predictions_master.csv"
LEADER_PATH = OUT / "spatial_vs_event_leaderboard.csv"
DATA_PATH = ROOT / "data" / "processed" / "flood_dataset.csv"

METRICS = ["AUC", "F1", "IoU", "Precision", "Recall", "Accuracy"]
TOP_N = 15
# Consistent publication-style typography.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 15,
    "savefig.facecolor": "white",
})

FINALISTS = [
    "M21 Extra Trees",
    "RF",
    "M22 Standalone CatBoost",
    "H2 Soft-Voting (RF+XGB+MLP)",
    "M16 DNN Ensemble",
    "M25 Calibrated Ensemble",
    "LR",
]

def savefig(name, dpi=300):
    """Save publication-quality PNG and vector PDF without changing chart content."""
    path = FIG / name
    fig = plt.gcf()

    # Improve readability consistently across existing figures.
    fig.set_dpi(dpi)
    fig.tight_layout(pad=1.4)

    # High-resolution raster output.
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.18,
        facecolor="white",
    )

    # Vector PDF for crisp text and lines in reports/publications.
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.18,
        facecolor="white",
    )

    plt.close(fig)
    print(f"Saved {path.relative_to(ROOT)}")
    print(f"Saved {pdf_path.relative_to(ROOT)}")

def load_inputs():
    missing = [p for p in [FOLD_PATH, OOF_PATH, LEADER_PATH, DATA_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required project files:\n" + "\n".join(map(str, missing)))
    folds = pd.read_csv(FOLD_PATH)
    oof = pd.read_csv(OOF_PATH)
    leader = pd.read_csv(LEADER_PATH)
    data = pd.read_csv(DATA_PATH)
    folds["cv"] = folds["cv"].astype(str).str.lower().str.strip()
    oof["cv"] = oof["cv"].astype(str).str.lower().str.strip()
    folds["model"] = folds["model"].astype(str).str.strip()
    oof["model"] = oof["model"].astype(str).str.strip()
    return folds, oof, leader, data

def numeric_metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    result = {}
    if len(np.unique(y)) == 2:
        result["AUC"] = roc_auc_score(y, p)
    else:
        result["AUC"] = np.nan
    result["F1"] = f1_score(y, pred, zero_division=0)
    result["IoU"] = jaccard_score(y, pred, zero_division=0)
    result["Precision"] = precision_score(y, pred, zero_division=0)
    result["Recall"] = recall_score(y, pred, zero_division=0)
    result["Accuracy"] = accuracy_score(y, pred)
    return result

def block_event_leaderboard(leader):
    # 01: event-CV leaderboard
    d = leader.dropna(subset=["AUC_mean_event"]).sort_values("AUC_mean_event", ascending=False)
    d = d.head(TOP_N).sort_values("AUC_mean_event", ascending=True)
    plt.figure(figsize=(11, max(6, 0.38 * len(d))))
    plt.barh(d["model"], d["AUC_mean_event"], xerr=d.get("AUC_std_event"))
    plt.xlabel("Mean event-held-out AUC (error bars: SD across events)")
    plt.title("Model leaderboard: unseen-event generalization")
    plt.xlim(max(0, float(d["AUC_mean_event"].min()) - 0.08), 1.0)
    savefig("01_event_auc_leaderboard.png")

    # 02: spatial/block vs event AUC
    d = leader.dropna(subset=["AUC_mean_block", "AUC_mean_event"])
    plt.figure(figsize=(9, 7))
    plt.scatter(d["AUC_mean_block"], d["AUC_mean_event"])
    for _, r in d.iterrows():
        plt.annotate(r["model"], (r["AUC_mean_block"], r["AUC_mean_event"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")
    lo = min(d["AUC_mean_block"].min(), d["AUC_mean_event"].min()) - 0.015
    hi = max(d["AUC_mean_block"].max(), d["AUC_mean_event"].max()) + 0.015
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlabel("Block/spatial CV mean AUC")
    plt.ylabel("Event-held-out CV mean AUC")
    plt.title("Spatial validation versus unseen-event validation")
    savefig("02_spatial_vs_event_auc.png")

    # 03: metrics heatmap for all models, block and event
    metrics_to_plot = [
        ("AUC_mean_block", "Block AUC"),
        ("AUC_mean_event", "Event AUC"),
        ("F1_mean_block", "Block F1"),
        ("F1_mean_event", "Event F1"),
        ("IoU_mean_block", "Block IoU"),
        ("IoU_mean_event", "Event IoU"),
    ]
    cols = [c for c, _ in metrics_to_plot if c in leader.columns]
    labels = [lab for c, lab in metrics_to_plot if c in leader.columns]
    d = leader.sort_values("AUC_mean_event", ascending=False, na_position="last")
    matrix = d[cols].to_numpy(dtype=float)
    plt.figure(figsize=(11, max(7, 0.35 * len(d))))
    im = plt.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(im, label="Score")
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(range(len(d)), d["model"], fontsize=8)
    plt.title("Model performance across evaluation strategies")
    savefig("03_model_metrics_heatmap.png")

    # 04: finalist metric comparison (event means)
    cols = [f"{m}_mean_event" for m in ["AUC", "F1", "IoU", "Precision", "Recall", "Accuracy"]]
    cols = [c for c in cols if c in leader.columns]
    d = leader[leader["model"].isin(FINALISTS)].copy()
    d = d.sort_values("AUC_mean_event", ascending=False)
    if not d.empty and cols:
        # Standardize nothing: all metrics are 0-1 and directly comparable as scores.
        x = np.arange(len(d))
        width = 0.8 / len(cols)
        plt.figure(figsize=(14, 7))
        for i, col in enumerate(cols):
            plt.bar(x + (i - (len(cols)-1)/2)*width, d[col], width, label=col.replace("_mean_event", ""))
        plt.xticks(x, d["model"], rotation=30, ha="right")
        plt.ylabel("Mean event-CV score")
        plt.ylim(0, 1)
        plt.title("Finalist comparison across event-CV metrics")
        plt.legend(ncol=3)
        savefig("04_finalist_metrics.png")

    # 05: metric rankings
    rank_cols = [f"{m}_mean_event" for m in ["AUC", "F1", "IoU"] if f"{m}_mean_event" in leader.columns]
    if rank_cols:
        d = leader[["model"] + rank_cols].copy()
        rank_names = []
        for c in rank_cols:
            rcol = c.replace("_mean_event", "_rank")
            d[rcol] = d[c].rank(ascending=False, method="min")
            rank_names.append(rcol)
        d = d.sort_values(rank_names[0])
        plt.figure(figsize=(10, max(7, 0.35 * len(d))))
        im = plt.imshow(d[rank_names].to_numpy(), aspect="auto", interpolation="nearest")
        plt.colorbar(im, label="Rank (lower is better)")
        plt.xticks(range(len(rank_names)), [c.replace("_rank", "") for c in rank_names])
        plt.yticks(range(len(d)), d["model"], fontsize=8)
        plt.title("Event-CV model ranks by AUC, F1 and IoU")
        savefig("05_metric_ranks.png")

def event_level_figures(oof):
    # Event-level scores from event-CV OOF predictions, not fold-average approximations.
    needed = {"y_true", "y_prob", "model", "event", "cv"}
    if not needed.issubset(oof.columns):
        print("Skipping event-level figures; OOF columns missing:", needed - set(oof.columns))
        return
    d = oof[(oof["cv"] == "event") & oof["event"].notna()].copy()
    d["y_true"] = pd.to_numeric(d["y_true"], errors="coerce")
    d["y_prob"] = pd.to_numeric(d["y_prob"], errors="coerce")
    d = d.dropna(subset=["y_true", "y_prob", "event"])
    if d.empty:
        print("No event-CV OOF rows available; skipping event figures.")
        return

    records = []
    for (model, event), g in d.groupby(["model", "event"]):
        if len(g) and g["y_true"].nunique() == 2:
            m = numeric_metrics(g["y_true"], g["y_prob"])
            records.append({"model": model, "event": event, "n": len(g), **m})
    em = pd.DataFrame(records)
    if em.empty:
        print("Could not compute event-level metrics from OOF predictions.")
        return

    # 06: event AUC heatmap
    pivot = em.pivot(index="model", columns="event", values="AUC")
    if "AUC" in []:
        pass
    if "M21 Extra Trees" in pivot.index:
        pivot = pivot.loc[pd.concat([
            em.groupby("model")["AUC"].mean().sort_values(ascending=False).head(TOP_N),
            em.groupby("model")["AUC"].mean().sort_values(ascending=False).loc[
                em.groupby("model")["AUC"].mean().sort_values(ascending=False).index.isin(["H4 CNN-BiLSTM"])
            ]
        ]).index.unique()] if len(pivot) else pivot
    else:
        pivot = pivot.loc[em.groupby("model")["AUC"].mean().sort_values(ascending=False).head(TOP_N).index]
    plt.figure(figsize=(10, max(7, 0.34 * len(pivot))))
    im = plt.imshow(pivot.to_numpy(), aspect="auto", interpolation="nearest", vmin=0.5, vmax=1)
    plt.colorbar(im, label="Held-out event AUC")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=8)
    plt.title("Model performance by held-out flood event")
    savefig("06_event_auc_heatmap.png")

    # 07: finalist metrics per event, separate panels to avoid mixing scales
    finalists = [m for m in FINALISTS if m in em["model"].unique()]
    if finalists:
        sub = em[em["model"].isin(finalists)]
        events = list(sub["event"].drop_duplicates())
        fig, axes = plt.subplots(1, len(events), figsize=(5 * len(events), 5), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, event in zip(axes, events):
            e = sub[sub["event"] == event].sort_values("AUC", ascending=False)
            ax.barh(e["model"], e["AUC"])
            ax.set_title(str(event))
            ax.set_xlim(0.5, 1.0)
            ax.set_xlabel("AUC")
        fig.suptitle("Finalist AUC on each held-out event")
        fig.tight_layout()
        savefig("07_finalists_by_event.png")

    # 08: event mean AUC with between-event SD
    stats = em.groupby("model")["AUC"].agg(["mean", "std"]).sort_values("mean", ascending=True).tail(TOP_N)
    plt.figure(figsize=(10, max(6, 0.38 * len(stats))))
    plt.barh(stats.index, stats["mean"], xerr=stats["std"])
    plt.xlabel("AUC across held-out events (error bars: between-event SD)")
    plt.title("Event generalization and variability")
    savefig("08_event_auc_variability.png")

    # 09: per-model generalization gap from the master leaderboard
    if LEADER_PATH.exists():
        lead = pd.read_csv(LEADER_PATH)
        if {"AUC_mean_block", "AUC_mean_event"}.issubset(lead.columns):
            lead = lead.dropna(subset=["AUC_mean_block", "AUC_mean_event"]).copy()
            lead["gap"] = lead["AUC_mean_block"] - lead["AUC_mean_event"]
            lead = lead.sort_values("gap")
            plt.figure(figsize=(10, max(6, 0.36 * len(lead))))
            plt.barh(lead["model"], lead["gap"])
            plt.axvline(0, linestyle="--", linewidth=1)
            plt.xlabel("Block-CV mean AUC minus event-CV mean AUC")
            plt.title("Generalization gap (positive means lower event AUC)")
            savefig("09_generalization_gap.png")

    # 10: event metric profiles for leading models
    chosen = [m for m in FINALISTS if m in em["model"].unique()]
    sub = em[em["model"].isin(chosen)]
    if not sub.empty:
        metrics = ["AUC", "F1", "IoU"]
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 11), sharex=True)
        for ax, metric in zip(axes, metrics):
            for model, g in sub.groupby("model"):
                g = g.sort_values("event")
                ax.plot(g["event"], g[metric], marker="o", label=model)
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25)
        axes[0].set_title("Finalist metric profiles across held-out events")
        axes[-1].tick_params(axis="x", rotation=20)
        axes[0].legend(ncol=2, fontsize=8)
        savefig("10_finalist_event_profiles.png")

def probability_figures(oof):
    required = {"cv", "model", "y_true", "y_prob", "y_pred"}
    if not required.issubset(oof.columns):
        print("Skipping probability figures; OOF columns missing:", required - set(oof.columns))
        return
    # Event OOF predictions are the main out-of-domain analysis.
    d = oof[oof["cv"].astype(str).str.lower().eq("event")].copy()
    for c in ["y_true", "y_prob", "y_pred"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["y_true", "y_prob", "y_pred"])
    if d.empty:
        print("No event OOF predictions available; skipping probability figures.")
        return

    # 11: confusion matrices, aggregated over event-held-out OOF predictions.
    models = [m for m in FINALISTS if m in d["model"].unique()]
    if models:
        ncols = 3
        nrows = int(np.ceil(len(models) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, model in zip(axes, models):
            g = d[d["model"] == model]
            cm = confusion_matrix(g["y_true"].astype(int), g["y_pred"].astype(int), labels=[0, 1])
            im = ax.imshow(cm, interpolation="nearest")
            ax.set_title(model, fontsize=9)
            ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
            ax.set_yticks([0, 1], ["True 0", "True 1"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        for ax in axes[len(models):]:
            ax.axis("off")
        fig.suptitle("Event-CV confusion matrices at threshold 0.5")
        savefig("11_event_confusion_matrices.png")

    # 12: ROC curves. Pooling is descriptive across held-out event predictions.
    plt.figure(figsize=(9, 7))
    plotted = 0
    for model in models:
        g = d[d["model"] == model]
        if g["y_true"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(g["y_true"].astype(int), g["y_prob"])
        score = roc_auc_score(g["y_true"].astype(int), g["y_prob"])
        plt.plot(fpr, tpr, label=f"{model} (AUC={score:.3f})")
        plotted += 1
    if plotted:
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Pooled event-CV ROC curves for finalists")
        plt.legend(fontsize=8)
        savefig("12_event_roc_curves.png")
    else:
        plt.close()

    # 13: precision-recall curves
    plt.figure(figsize=(9, 7))
    plotted = 0
    for model in models:
        g = d[d["model"] == model]
        if g["y_true"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(g["y_true"].astype(int), g["y_prob"])
        ap = average_precision_score(g["y_true"].astype(int), g["y_prob"])
        plt.plot(recall, precision, label=f"{model} (AP={ap:.3f})")
        plotted += 1
    if plotted:
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Pooled event-CV precision-recall curves")
        plt.legend(fontsize=8)
        savefig("13_event_precision_recall_curves.png")
    else:
        plt.close()

    # 14: threshold sensitivity, using pooled event-held-out predictions.
    if models:
        thresholds = np.linspace(0.05, 0.95, 37)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
        for model in models:
            g = d[d["model"] == model]
            y = g["y_true"].astype(int).to_numpy()
            p = g["y_prob"].to_numpy()
            vals = {m: [] for m in ["F1", "Precision", "Recall"]}
            for t in thresholds:
                pred = (p >= t).astype(int)
                vals["F1"].append(f1_score(y, pred, zero_division=0))
                vals["Precision"].append(precision_score(y, pred, zero_division=0))
                vals["Recall"].append(recall_score(y, pred, zero_division=0))
            for ax, metric in zip(axes, ["F1", "Precision", "Recall"]):
                ax.plot(thresholds, vals[metric], label=model)
                ax.set_title(metric)
                ax.set_xlabel("Probability threshold")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("Score")
        axes[0].legend(fontsize=7)
        fig.suptitle("Threshold sensitivity on event-held-out predictions")
        savefig("14_threshold_sensitivity.png")

    # 15: calibration reliability diagram and Brier score
    if models:
        plt.figure(figsize=(8, 7))
        plotted = 0
        for model in models:
            g = d[d["model"] == model]
            y = g["y_true"].astype(int).to_numpy()
            p = g["y_prob"].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
            brier = brier_score_loss(y, p)
            plt.plot(mean_pred, frac_pos, marker="o", label=f"{model} (Brier={brier:.3f})")
            plotted += 1
        if plotted:
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
            plt.xlabel("Mean predicted probability")
            plt.ylabel("Observed positive fraction")
            plt.title("Event-CV probability calibration")
            plt.legend(fontsize=8)
            savefig("15_event_calibration.png")
        else:
            plt.close()

def dataset_figures(data):
    if "flooded" in data.columns:
        counts = data["flooded"].value_counts(dropna=False).sort_index()
        plt.figure(figsize=(7, 5))
        plt.bar([str(x) for x in counts.index], counts.values)
        plt.xlabel("Reference label (flooded)")
        plt.ylabel("Sample count")
        plt.title("Dataset reference-label balance")
        savefig("16_dataset_class_balance.png")

    # Study sample distribution; point map only, not a predicted susceptibility map.
    if {"lon", "lat", "flooded"}.issubset(data.columns):
        d = data.dropna(subset=["lon", "lat", "flooded"])
        plt.figure(figsize=(9, 7))
        for label, group in d.groupby("flooded"):
            plt.scatter(group["lon"], group["lat"], s=5, alpha=0.45, label=f"flooded={label}", rasterized=True)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Sample locations by Sentinel-1-derived reference label")
        plt.legend()
        plt.axis("equal")
        savefig("17_sample_location_reference_labels.png")

    # Event counts and positive rates
    if {"event", "flooded"}.issubset(data.columns):
        g = data.groupby("event")["flooded"].agg(["count", "mean"]).sort_index()
        fig, ax1 = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(g))
        ax1.bar(x, g["count"])
        ax1.set_ylabel("Sample count")
        ax1.set_xticks(x, g.index, rotation=20, ha="right")
        ax2 = ax1.twinx()
        ax2.plot(x, g["mean"], marker="o")
        ax2.set_ylabel("Flood-positive fraction")
        ax2.set_ylim(0, 1)
        plt.title("Dataset sample counts and reference-label rates by event")
        savefig("18_event_dataset_profile.png")

def protocol_figure():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    boxes = [
        (0.04, 0.55, 0.40, 0.28, "BLOCK / SPATIAL CV",
         "Split by spatial block\n3 folds\nTests transfer to unseen blocks"),
        (0.56, 0.55, 0.40, 0.28, "EVENT-HELD-OUT CV",
         "Leave one event out\n4 folds\nTests transfer to an unseen event"),
        (0.04, 0.12, 0.92, 0.22, "INTERPRETATION",
         "Both evaluations use Sentinel-1-derived reference labels.\nMetrics describe agreement with those labels, not independent flood-truth verification."),
    ]
    for x, y, w, h, title, body in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h*0.68, title, ha="center", va="center", fontsize=11, fontweight="bold", transform=ax.transAxes)
        ax.text(x+w/2, y+h*0.32, body, ha="center", va="center", fontsize=9, transform=ax.transAxes)
    ax.annotate("", xy=(0.52, 0.68), xytext=(0.46, 0.68), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", linewidth=1.3))
    plt.title("Evaluation design and interpretation", pad=12)
    savefig("19_evaluation_protocol.png")

def finalist_scorecard(leader):
    models = [m for m in FINALISTS if m in leader["model"].values]
    d = leader[leader["model"].isin(models)].copy()
    if d.empty:
        return
    d = d.sort_values("AUC_mean_event", ascending=False)
    cols = [
        ("AUC_mean_block", "Block AUC"),
        ("AUC_mean_event", "Event AUC"),
        ("F1_mean_event", "Event F1"),
        ("IoU_mean_event", "Event IoU"),
    ]
    cols = [(c, label) for c, label in cols if c in d.columns]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axis("off")
    cell_text = []
    for _, r in d.iterrows():
        row = [r["model"]]
        for c, _ in cols:
            v = r[c]
            row.append("—" if pd.isna(v) else f"{v:.3f}")
        cell_text.append(row)
    table = ax.table(
        cellText=cell_text,
        colLabels=["Model"] + [label for _, label in cols],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)
    plt.title("Finalist scorecard: spatial and event generalization")
    savefig("20_finalist_scorecard.png")

def main():
    folds, oof, leader, data = load_inputs()
    print(f"Loaded {len(folds)} fold-result rows, {len(oof)} OOF rows, {len(leader)} leaderboard models.")
    block_event_leaderboard(leader)
    event_level_figures(oof)
    probability_figures(oof)
    dataset_figures(data)
    protocol_figure()
    finalist_scorecard(leader)
    print(f"\nFinished. Figures are in: {FIG}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
