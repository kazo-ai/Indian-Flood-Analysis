from pathlib import Path
import warnings

import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# Flood susceptibility research dashboard
# Run from the project root:
#   streamlit run app.py
#
# This dashboard uses existing CV results, OOF predictions, dataset
# summaries, and generated figures. It does NOT retrain models.
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Flood Susceptibility Research Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "flood_dataset.csv"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

st.markdown(
    """
    
<style>
  :root {
    --jp-indigo: #6678bd;
    --jp-sakura: #e8a0b8;
    --jp-text: #f1f3fb;
    --jp-muted: #b5bdd0;
    --jp-border: rgba(155, 168, 215, 0.22);
    --jp-card: #171c29;
  }

  /* Page layout */
  .block-container {
    padding: 2rem 2rem 3rem;
    max-width: 1600px;
  }

  h1, h2, h3, h4 {
    color: var(--jp-text) !important;
    letter-spacing: -0.025em;
  }

  h1 {
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--jp-border);
    font-weight: 750;
  }

  p, li, label,
  [data-testid="stMarkdownContainer"] {
    color: #e0e4ef;
  }

  /* Japanese-inspired gradient rails */
  body::before,
  body::after {
    content: "";
    position: fixed;
    top: 0;
    bottom: 0;
    width: 5px;
    z-index: 999;
    pointer-events: none;
  }

  body::before {
    left: 0;
    background: linear-gradient(
      180deg, #6678bd 0%, #a18ebd 50%, #e8a0b8 100%
    );
  }

  body::after {
    right: 0;
    background: linear-gradient(
      180deg, #e8a0b8 0%, #a18ebd 50%, #6678bd 100%
    );
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    border-right: 1px solid var(--jp-border);
  }

  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {
    color: #aab8f0 !important;
  }

  section[data-testid="stSidebar"] label {
    color: #e4e7f2 !important;
  }

  /* FIX: dark metric cards with high-contrast labels */
  div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #20283b, #171c29);
    border: 1px solid var(--jp-border);
    border-radius: 14px;
    padding: 18px 18px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.15);
    transition: border-color 160ms ease, transform 160ms ease;
  }

  div[data-testid="stMetric"]:hover {
    border-color: rgba(232, 160, 184, 0.65);
    transform: translateY(-2px);
  }

  div[data-testid="stMetricLabel"],
  div[data-testid="stMetricLabel"] p {
    color: #c4cce0 !important;
    opacity: 1 !important;
    font-size: 0.9rem !important;
    font-weight: 550 !important;
  }

  div[data-testid="stMetricValue"],
  div[data-testid="stMetricValue"] > div {
    color: #f0b5cb !important;
    font-weight: 750 !important;
  }

  div[data-testid="stMetricDelta"] {
    color: #c4cce0 !important;
  }

  /* FIX: Japanese indigo selection chips instead of default red */
  section[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: #414e82 !important;
    border: 1px solid rgba(192, 199, 239, 0.28) !important;
    border-radius: 7px !important;
  }

  section[data-testid="stSidebar"] [data-baseweb="tag"] span {
    color: #ffffff !important;
  }

  section[data-testid="stSidebar"] [data-baseweb="tag"] svg {
    fill: #f0b5cb !important;
    color: #f0b5cb !important;
  }

  /* Selects, multiselects, and inputs */
  div[data-baseweb="select"] > div,
  div[data-baseweb="input"] > div {
    border-radius: 9px !important;
    border-color: var(--jp-border) !important;
  }

  div[data-baseweb="select"] > div:focus-within,
  div[data-baseweb="input"] > div:focus-within {
    border-color: #9ba9e4 !important;
    box-shadow: 0 0 0 1px #9ba9e4 !important;
  }

  /* Tabs */
  button[data-baseweb="tab"] {
    color: #c4cce0 !important;
  }

  button[data-baseweb="tab"][aria-selected="true"] {
    color: #f0b5cb !important;
    border-bottom-color: #f0b5cb !important;
  }

  /* Buttons */
  .stButton > button,
  .stDownloadButton > button {
    border: 1px solid rgba(155, 168, 215, 0.38);
    border-radius: 9px;
    transition: all 160ms ease;
  }

  .stButton > button:hover,
  .stDownloadButton > button:hover {
    border-color: #e8a0b8;
    color: #f0b5cb;
  }

  /* Data tables */
  div[data-testid="stDataFrame"] {
    border: 1px solid var(--jp-border);
    border-radius: 12px;
    overflow: hidden;
  }

  /* Expanders */
  details[data-testid="stExpander"] {
    border: 1px solid var(--jp-border);
    border-radius: 12px;
  }

  /* Keep warnings and research caveats readable */
  div[data-testid="stAlert"] {
    border-radius: 11px;
  }

  .small-note {
    color: var(--jp-muted);
    font-size: 0.88rem;
    line-height: 1.65;
  }

  @media (max-width: 768px) {
    .block-container {
      padding: 1.2rem 1rem 2rem;
    }

    body::before,
    body::after {
      width: 3px;
    }

    h1 {
      font-size: 1.8rem;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
      transition: none !important;
    }
  }
</style>

    """,
    unsafe_allow_html=True,
)

st.title("🌊 Flood Susceptibility Research Dashboard")
st.caption(
    "Model comparison, spatial/block cross-validation, event-held-out evaluation, "
    "and dataset diagnostics."
)

# --------------------------- Data loading ---------------------------

@st.cache_data(show_spinner=False)
def load_csv(path_string):
    path = Path(path_string)
    if not path.exists():
        return None
    return pd.read_csv(path)


dataset = load_csv(str(DATA_PATH))

# Load the current final evaluation, plus the separately evaluated
# H4 CNN-BiLSTM and M24 TabNet models.
base_folds = load_csv(str(OUTPUTS / "final_evaluation" / "results_per_fold.csv"))
h4_folds = load_csv(str(OUTPUTS / "difficult_models" / "H4_CNN-BiLSTM" / "results_per_fold.csv"))
m24_folds = load_csv(str(OUTPUTS / "difficult_models" / "M24_TabNet" / "results_per_fold.csv"))

fold_frames = [f for f in (base_folds, h4_folds, m24_folds)
               if f is not None and not f.empty]
folds = pd.concat(fold_frames, ignore_index=True) if fold_frames else None

# Preserve compatibility with the dashboard's existing summary and OOF views.
summary = load_csv(str(OUTPUTS / "summary_master.csv"))
oof = load_csv(str(OUTPUTS / "oof_predictions_master.csv"))
if oof is None:
    oof = pd.DataFrame()

# These leaderboards already contain all 26 models.
block_leaderboard = load_csv(str(OUTPUTS / "final_evaluation" / "combined_block_leaderboard.csv"))
event_leaderboard = load_csv(str(OUTPUTS / "final_evaluation" / "combined_event_leaderboard.csv"))
leaderboard = load_csv(str(OUTPUTS / "spatial_vs_event_leaderboard.csv"))

missing = []
for name, value, expected in [
    ("Processed dataset", dataset, DATA_PATH),
    ("Per-fold results", folds, OUTPUTS / "final_evaluation" / "results_per_fold.csv"),
    ("Summary results", summary, OUTPUTS / "summary_master.csv"),
    ("Spatial-vs-event leaderboard", leaderboard, OUTPUTS / "spatial_vs_event_leaderboard.csv"),
]:
    if value is None:
        missing.append(f"- **{name}**: `{expected.relative_to(ROOT)}`")

if missing:
    st.error(
        "Some project files are missing. This dashboard expects the following files "
        "inside the project directory:\n\n" + "\n".join(missing)
    )
    st.info(
        "Check that you are running `streamlit run app.py` from the "
        "`flood_susceptibility` project root and that the CSV files exist."
    )
    st.stop()

# Normalize string columns where present.
for frame in (folds, summary, oof, leaderboard, dataset):
    if frame is not None:
        for col in ("model", "cv", "kind", "event", "block", "lc_name"):
            if col in frame.columns:
                frame[col] = frame[col].astype("string")

# ----------------------------- Sidebar -----------------------------

st.sidebar.header("Dashboard controls")
cv_options = []
if "cv" in folds.columns:
    cv_options = [x for x in ["event", "block"] if x in set(folds["cv"].dropna().astype(str))]
if not cv_options:
    cv_options = sorted(folds["cv"].dropna().astype(str).unique().tolist()) if "cv" in folds else []

default_cv = "event" if "event" in cv_options else (cv_options[0] if cv_options else None)
selected_cv = st.sidebar.selectbox("Evaluation protocol", cv_options, index=cv_options.index(default_cv) if default_cv in cv_options else 0) if cv_options else None

all_models = sorted(folds["model"].dropna().astype(str).unique().tolist()) if "model" in folds.columns else []
preferred = [
    "M21 Extra Trees",
    "RF",
    "M22 Standalone CatBoost",
    "H2 Soft-Voting (RF+XGB+MLP)",
    "M16 DNN Ensemble",
    "M25 Calibrated Ensemble",
    "LR",
]
default_models = all_models
selected_models = st.sidebar.multiselect(
    "Models to compare",
    all_models,
    default=default_models,
)

events = sorted(dataset["event"].dropna().astype(str).unique().tolist()) if "event" in dataset.columns else []
selected_events = st.sidebar.multiselect("Events", events, default=events)

st.sidebar.divider()
st.sidebar.caption("This is an evaluation dashboard, not a live flood-warning system.")
st.sidebar.caption("The target labels are Sentinel-1-derived open-water detections; scores measure agreement with those derived labels.")

# ----------------------------- Helpers -----------------------------

def metric_value(frame, column, default=None):
    if column not in frame.columns:
        return default
    vals = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else default


def safe_metric_row(frame, metric, cv_value):
    if frame is None or metric not in frame.columns or "cv" not in frame.columns:
        return pd.DataFrame()
    subset = frame[frame["cv"].astype(str) == str(cv_value)].copy()
    if "model" in subset.columns:
        subset = subset[subset["model"].astype(str).isin(selected_models)]
    return subset


def display_existing_figures():
    if not FIGURES.exists():
        st.warning(
            "No `outputs/figures/` folder found yet. Run `python generate_research_figures.py` "
            "from the project root to create the report figures."
        )
        return
    images = sorted(FIGURES.glob("*.png"))
    if not images:
        st.warning("The `outputs/figures/` folder exists, but it contains no PNG files yet.")
        return

    labels = [p.name for p in images]
    selected_name = st.selectbox("Choose a generated figure", labels)
    chosen = FIGURES / selected_name
    st.image(str(chosen), caption=selected_name, use_container_width=True)

    with st.expander("Browse all generated figures"):
        cols = st.columns(2)
        for idx, image_path in enumerate(images):
            with cols[idx % 2]:
                st.image(str(image_path), caption=image_path.name, use_container_width=True)


# ------------------------------ Tabs ------------------------------

tab_overview, tab_models, tab_events, tab_predictions, tab_data, tab_figures = st.tabs(
    [
        "Overview",
        "Model comparison",
        "Event generalization",
        "OOF diagnostics",
        "Dataset explorer",
        "Research figures",
    ]
)

# ----------------------------- Overview ----------------------------

with tab_overview:
    st.subheader("Project overview")

    n_rows = len(dataset)
    n_events = dataset["event"].nunique() if "event" in dataset else 0
    n_blocks = dataset["block"].nunique() if "block" in dataset else 0
    flooded = int(pd.to_numeric(dataset["flooded"], errors="coerce").fillna(0).sum()) if "flooded" in dataset else None
    n_models = folds["model"].nunique() if "model" in folds else 0
    n_fold_records = len(folds)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dataset rows", f"{n_rows:,}")
    c2.metric("Flood events", f"{n_events}")
    c3.metric("Spatial blocks", f"{n_blocks}")
    c4.metric("Models evaluated", f"{n_models}")
    c5.metric("CV fold records", f"{n_fold_records:,}")

    if flooded is not None:
        c6, c7 = st.columns(2)
        c6.metric("Derived flooded labels", f"{flooded:,}")
        c7.metric("Derived non-flooded labels", f"{n_rows - flooded:,}")

    st.markdown("### Evaluation protocols")
    left, right = st.columns(2)
    with left:
        st.markdown("**Spatial/block cross-validation**")
        st.write(
            "Tests performance on held-out spatial blocks. This helps assess spatial transfer "
            "to locations not represented in the training blocks."
        )
    with right:
        st.markdown("**Event-held-out cross-validation**")
        st.write(
            "Trains on other flood events and tests on a held-out event. This is a stricter "
            "test of transfer to a different event, region, or acquisition context."
        )

    st.info(
        "Interpretation boundary: the labels come from Sentinel-1-derived open-water detection. "
        "High scores indicate agreement with these labels, not necessarily independently verified "
        "flood extent or real-time forecasting performance."
    )

    if "cv" in folds.columns:
        counts = folds.groupby("cv", dropna=False).agg(
            fold_records=("model", "size"),
            models=("model", "nunique"),
        ).reset_index()
        st.markdown("### Evaluation coverage")
        st.dataframe(counts, use_container_width=True, hide_index=True)

# -------------------------- Model comparison -----------------------

with tab_models:
    st.subheader("Compare models across evaluation protocols")

    if leaderboard.empty:
        st.warning("The spatial-vs-event leaderboard is empty.")
    else:
        view = leaderboard.copy()
        if "model" in view.columns and selected_models:
            view = view[view["model"].astype(str).isin(selected_models)]
        metric = st.selectbox("Metric", ["AUC", "F1", "IoU", "Precision", "Recall", "Accuracy"], index=0)
        block_col = f"{metric}_mean_block"
        event_col = f"{metric}_mean_event"
        if block_col in view.columns and event_col in view.columns:
            chart_df = view[["model", block_col, event_col]].melt(
                id_vars="model", var_name="protocol", value_name=metric
            )
            chart_df["protocol"] = chart_df["protocol"].map({
                block_col: "Spatial/block CV",
                event_col: "Event-held-out CV",
            })
            st.bar_chart(chart_df.pivot(index="model", columns="protocol", values=metric), horizontal=True)
            st.caption("Mean score across folds. For event-held-out CV, there are four held-out events.")
        else:
            st.warning(f"Could not find expected columns `{block_col}` and `{event_col}`.")

        with st.expander("Show comparison table"):
            cols = ["model"]
            for m in ["AUC", "F1", "IoU", "Precision", "Recall", "Accuracy"]:
                for suffix in ["mean_block", "mean_event", "std_event"]:
                    col = f"{m}_{suffix}"
                    if col in view.columns:
                        cols.append(col)
            st.dataframe(view[cols].sort_values(event_col, ascending=False) if event_col in view.columns else view[cols],
                         use_container_width=True, hide_index=True)

    st.markdown("### Fold-level results")
    fold_view = safe_metric_row(folds, "AUC", selected_cv) if selected_cv else folds.copy()
    if selected_events and "test_events" in fold_view.columns and selected_cv == "event":
        fold_view = fold_view[
            fold_view["test_events"].astype(str).apply(
                lambda value: any(
            event in str(value) for event in selected_events
        ) if pd.notna(value) else False
            )
        ]
    if selected_models and "model" in fold_view.columns:
        fold_view = fold_view[fold_view["model"].astype(str).isin(selected_models)]
    preferred_cols = [
        "model", "cv", "fold", "test_events", "test_blocks", "AUC", "F1", "IoU",
        "Precision", "Recall", "Accuracy", "elapsed_seconds"
    ]
    cols = [c for c in preferred_cols if c in fold_view.columns]
    st.dataframe(fold_view[cols].sort_values("AUC", ascending=False) if "AUC" in fold_view else fold_view[cols],
                 use_container_width=True, hide_index=True)

# ------------------------- Event generalization --------------------

with tab_events:
    st.subheader("Performance on held-out flood events")
    event_folds = folds[folds["cv"].astype(str) == "event"].copy() if "cv" in folds else pd.DataFrame()
    if selected_models and "model" in event_folds:
        event_folds = event_folds[event_folds["model"].astype(str).isin(selected_models)]

    if selected_events and "test_events" in event_folds:
        event_folds = event_folds[
            event_folds["test_events"].astype(str).apply(
                lambda value: any(
            event in str(value) for event in selected_events
        ) if pd.notna(value) else False
            )
        ]

    if event_folds.empty:
        st.warning("No event-held-out fold results match the current filters.")
    else:
        event_metric = st.selectbox("Event metric", ["AUC", "F1", "IoU", "Precision", "Recall", "Accuracy"], key="event_metric")
        if "test_events" in event_folds and event_metric in event_folds:
            st.bar_chart(
                event_folds.pivot_table(
                    index="test_events", columns="model", values=event_metric, aggfunc="mean"
                )
            )
            st.caption("Each row represents a held-out event. Differences between events can reveal domain shift.")
        st.markdown("### Event-held-out fold records")
        display_cols = [c for c in [
            "model", "fold", "test_events", "AUC", "F1", "IoU", "Precision", "Recall",
            "Accuracy", "TN", "FP", "FN", "TP"
        ] if c in event_folds.columns]
        st.dataframe(event_folds[display_cols].sort_values(event_metric, ascending=False),
                     use_container_width=True, hide_index=True)

        st.warning(
            "Treat event-level differences carefully: only four events are available, and "
            "the very different scores across events indicate that average performance alone "
            "can hide weak transfer on a difficult event."
        )

# ---------------------------- OOF diagnostics ----------------------

with tab_predictions:
    st.subheader("Out-of-fold prediction diagnostics")

    if oof.empty:
        st.info(
            "OOF predictions are not bundled with this deployment because the "
            "prediction file is very large. The model evaluation results and "
            "leaderboards remain available in the other dashboard sections."
        )
    else:
        required = {"y_true", "y_prob", "y_pred", "model"}
        if not required.issubset(set(oof.columns)):
            st.warning(
                f"OOF predictions are missing required columns: "
                f"{sorted(required - set(oof.columns))}"
            )
        else:
            oof_view = oof.copy()
            if selected_cv and "cv" in oof_view:
                oof_view = oof_view[oof_view["cv"].astype(str) == selected_cv]
            if selected_models:
                oof_view = oof_view[oof_view["model"].astype(str).isin(selected_models)]
            if selected_events and "event" in oof_view:
                oof_view = oof_view[oof_view["event"].astype(str).isin(selected_events)]

            if oof_view.empty:
                st.warning("No OOF predictions match the selected filters.")
            else:
                model_for_diag = st.selectbox(
                    "Model for diagnostic tables",
                    sorted(oof_view["model"].dropna().astype(str).unique().tolist()),
                )
                diag = oof_view[oof_view["model"].astype(str) == model_for_diag].copy()
                diag["y_true"] = pd.to_numeric(diag["y_true"], errors="coerce")
                diag["y_pred"] = pd.to_numeric(diag["y_pred"], errors="coerce")
                diag["y_prob"] = pd.to_numeric(diag["y_prob"], errors="coerce")
                diag = diag.dropna(subset=["y_true", "y_pred", "y_prob"])

                if len(diag):
                    tp = int(((diag.y_true == 1) & (diag.y_pred == 1)).sum())
                    tn = int(((diag.y_true == 0) & (diag.y_pred == 0)).sum())
                    fp = int(((diag.y_true == 0) & (diag.y_pred == 1)).sum())
                    fn = int(((diag.y_true == 1) & (diag.y_pred == 0)).sum())
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("True positives", f"{tp:,}")
                    p2.metric("False positives", f"{fp:,}")
                    p3.metric("True negatives", f"{tn:,}")
                    p4.metric("False negatives", f"{fn:,}")

                    cm = pd.DataFrame(
                        [[tn, fp], [fn, tp]],
                        index=["Actual non-flooded", "Actual flooded"],
                        columns=["Predicted non-flooded", "Predicted flooded"],
                    )
                    st.markdown("### Out-of-fold confusion matrix")
                    st.dataframe(cm, use_container_width=True)

                    st.markdown("### Probability distribution by derived label")
                    dist = diag[["y_true", "y_prob"]].copy()
                    dist["label"] = dist["y_true"].map(
                        {0: "Derived non-flooded", 1: "Derived flooded"}
                    )
                    hist = (
                        dist.groupby("label")["y_prob"]
                        .value_counts(bins=10, sort=False)
                        .unstack(fill_value=0).T
                    )
                    st.bar_chart(hist)

                    if "event" in diag.columns:
                        st.markdown("### OOF scores by event")
                        grouped = diag.groupby("event").apply(
                            lambda g: pd.Series({
                                "rows": len(g),
                                "accuracy": float((g.y_true == g.y_pred).mean()),
                                "precision": float(
                                    (((g.y_true == 1) & (g.y_pred == 1)).sum())
                                    / max(1, (g.y_pred == 1).sum())
                                ),
                                "recall": float(
                                    (((g.y_true == 1) & (g.y_pred == 1)).sum())
                                    / max(1, (g.y_true == 1).sum())
                                ),
                            }),
                            include_groups=False,
                        ).reset_index()
                        st.dataframe(grouped, use_container_width=True, hide_index=True)

                    with st.expander("Inspect filtered OOF rows"):
                        st.dataframe(diag.head(5000), use_container_width=True, hide_index=True)
                        st.caption("Showing at most 5,000 rows in the preview.")

# ---------------------------- Dataset explorer ---------------------

with tab_data:
    st.subheader("Dataset explorer")
    st.write(f"Loaded **{len(dataset):,} rows** and **{len(dataset.columns)} columns**.")

    search_text = st.text_input("Search column names", placeholder="e.g. rain, elevation, landcover")
    columns = dataset.columns.tolist()
    if search_text.strip():
        columns = [c for c in columns if search_text.lower() in c.lower()]
    selected_columns = st.multiselect(
        "Columns to display",
        columns,
        default=[c for c in ["event", "block", "flooded", "elevation", "slope", "rain_event_total", "ndvi_pre", "lc_name"] if c in columns],
    )

    if selected_events and "event" in dataset.columns:
        data_view = dataset[dataset["event"].astype(str).isin(selected_events)]
    else:
        data_view = dataset
    if selected_columns:
        st.dataframe(data_view[selected_columns].head(10000), use_container_width=True, hide_index=True)
    else:
        st.info("Select one or more columns to display.")
    st.caption("Preview limited to 10,000 rows for responsiveness.")

    st.markdown("### Dataset profile")
    numeric_cols = dataset.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        st.dataframe(dataset[numeric_cols].describe().T, use_container_width=True)

    if "event" in dataset.columns and "flooded" in dataset.columns:
        counts = dataset.groupby(["event", "flooded"]).size().unstack(fill_value=0)
        counts = counts.rename(columns={0: "Derived non-flooded", 1: "Derived flooded"})
        st.markdown("### Derived label counts by event")
        st.bar_chart(counts)

    if {"lon", "lat", "flooded"}.issubset(dataset.columns):
        st.markdown("### Sample locations")
        st.caption("Reference sample locations and derived labels only; this is not a predicted susceptibility map.")
        map_data = dataset[["lat", "lon", "flooded"]].dropna().copy()
        if len(map_data) > 5000:
            map_data = map_data.sample(5000, random_state=42)
        st.map(map_data.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]])

# --------------------------- Research figures ----------------------

with tab_figures:
    st.subheader("Generated report figures")
    display_existing_figures()

st.divider()
st.markdown(
    """
    <p class="small-note">
    Research prototype · Uses existing evaluation outputs · No models are retrained by this app.
    A deployable susceptibility predictor requires a saved, validated model plus compatible
    predictor inputs and an explicit prediction workflow.
    </p>
    """,
    unsafe_allow_html=True,
)
