"""
Central configuration for the flood susceptibility research framework.

Edit this file to change project settings, paths, model parameters,
cross-validation, spatial modelling, temporal modelling, and outputs.

Project:
    Flood Susceptibility Mapping and Forecasting

Hardware:
    Apple Silicon / CUDA / CPU

Important:
    Temporal models require genuine time-indexed observations.
    Spatial models require defensible spatial relationships.
"""

from pathlib import Path
import os


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATASET_PATH = PROCESSED_DATA_DIR / "flood_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

RESULTS_DIR = OUTPUT_DIR / "results"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"

DATA_DICTIONARY_PATH = (
    PROCESSED_DATA_DIR / "data_dictionary.md"
)

CLEANING_REPORT_PATH = (
    PROCESSED_DATA_DIR / "cleaning_report.txt"
)


def create_output_directories():
    """Create directories needed by the training framework."""

    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        RESULTS_DIR,
        PREDICTIONS_DIR,
        CHECKPOINTS_DIR,
        FIGURES_DIR,
        LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. GENERAL SETTINGS
# ============================================================

PROJECT_NAME = "Flood Susceptibility Research Framework"

SEED = 42

LOG_LEVEL = "INFO"

LOG_TO_FILE = True

LOG_TO_CONSOLE = True

REPRODUCIBILITY = {
    "seed": SEED,
    "deterministic_torch": True,
    "allow_tf32": False,
}


# ============================================================
# 3. GOOGLE EARTH ENGINE SETTINGS
# ============================================================

GEE_PROJECT = "flood-20-25-india"

DRIVE_FOLDER = "flood_susceptibility"


# ============================================================
# 4. SPATIAL SAMPLING SETTINGS
# ============================================================

BLOCK_DEG = 0.25

SAMPLING = {
    "n_flood": 4000,
    "n_nonflood": 4000,
    "scale": 30,
    "tile_scale": 8,
}


# ============================================================
# 5. FLOOD LABEL SETTINGS
# ============================================================

LABEL = {
    "diff_db": -3.0,
    "vv_max_db": -15.0,
    "perm_water_occ": 50,
    "slope_max_deg": 8.0,
    "hand_max_m": 30.0,
}


# ============================================================
# 6. FLOOD EVENTS
# ============================================================

# bbox format:
# [west, south, east, north]

EVENTS = {
    "kerala_2018": {
        "bbox": [76.2, 9.0, 77.0, 10.5],
        "pre": ("2018-01-01", "2018-02-28"),
        "flood": ("2018-08-08", "2018-08-22"),
    },

    "assam_2020": {
        "bbox": [90.0, 26.0, 92.5, 27.0],
        "pre": ("2020-05-15", "2020-06-15"),
        "flood": ("2020-06-30", "2020-07-18"),
    },

    "bihar_2020": {
        "bbox": [85.0, 25.6, 87.8, 26.6],
        "pre": ("2020-05-15", "2020-06-15"),
        "flood": ("2020-06-30", "2020-07-18"),
    },

    "assam_2022": {
        "bbox": [90.0, 26.0, 92.5, 27.0],
        "pre": ("2022-04-01", "2022-04-30"),
        "flood": ("2022-06-16", "2022-06-28"),
    },
}


# ============================================================
# 7. FEATURE CONFIGURATION
# ============================================================

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

CATEGORICAL_FEATURES = [
    "landcover",
]

LABEL_COL = "flooded"

META_COLS = [
    "lon",
    "lat",
    "event",
]

SPATIAL_COORDINATE_COLS = [
    "lon",
    "lat",
]

EVENT_COL = "event"

SPATIAL_BLOCK_COL = "block"

TIMESTAMP_COL = None

# Your prepared dataset contains one-hot encoded land-cover columns.
# Example: lc_bare, lc_built, lc_crop, lc_grass, lc_shrub, lc_tree.
#
# These columns are detected dynamically from the CSV.
# The original string column lc_name is not a predictor.

LANDCOVER_DUMMY_PREFIX = "lc_"

EXCLUDE_FROM_FEATURES = [
    LABEL_COL,
    "lon",
    "lat",
    "event",
    "block",
    "lc_name",
]

WORLDCOVER = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "crop",
    50: "built",
    60: "bare",
    70: "snow",
    80: "water",
    90: "wetland",
    95: "mangrove",
    100: "moss",
}


# ============================================================
# 8. DATA VALIDATION
# ============================================================

DATA_VALIDATION = {
    "require_binary_target": True,
    "expected_target_values": [0, 1],
    "allow_missing_features": True,
    "drop_missing_target": True,
    "reject_infinite_values": True,
    "warn_on_duplicate_coordinates": True,
    "check_feature_schema": True,
}


# ============================================================
# 9. MODEL REGISTRY SETTINGS
# ============================================================

# These IDs are stable identifiers for the proposed 25 models.
#
# Actual implementations will be registered in models.py and
# the model registry module.

MODEL_REGISTRY = {
    "M01": {
        "name": "Logistic Regression",
        "family": "traditional_ml",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M02": {
        "name": "Random Forest",
        "family": "traditional_ml",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M03": {
        "name": "XGBoost",
        "family": "traditional_ml",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M04": {
        "name": "LightGBM",
        "family": "traditional_ml",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M05": {
        "name": "MLP",
        "family": "traditional_ml",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M06": {
        "name": "Stacking Ensemble",
        "family": "hybrid_ml",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M07": {
        "name": "Soft-Voting Ensemble",
        "family": "hybrid_ml",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M08": {
        "name": "GA-XGBoost",
        "family": "hybrid_ml",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M09": {
        "name": "RF Feature Selection + LightGBM",
        "family": "hybrid_ml",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M10": {
        "name": "CatBoost Stacking Ensemble",
        "family": "hybrid_ml",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M11": {
        "name": "MLP-Embedding + XGBoost",
        "family": "tabular_dl",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M12": {
        "name": "Autoencoder-XGBoost",
        "family": "tabular_dl",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M13": {
        "name": "Transformer-XGBoost",
        "family": "tabular_dl",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M14": {
        "name": "FT-Transformer",
        "family": "tabular_dl",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M15": {
        "name": "CNN-Attention",
        "family": "tabular_dl",
        "task": "static",
        "requires_spatial_data": True,
        "requires_temporal_data": False,
    },

    "M16": {
        "name": "Deep Neural Network Ensemble",
        "family": "tabular_dl",
        "task": "static",
        "requires_spatial_data": False,
        "requires_temporal_data": False,
    },

    "M17": {
        "name": "GNN-XGBoost",
        "family": "spatial_dl",
        "task": "static",
        "requires_spatial_data": True,
        "requires_temporal_data": False,
    },

    "M18": {
        "name": "Graph Attention Network",
        "family": "spatial_dl",
        "task": "static",
        "requires_spatial_data": True,
        "requires_temporal_data": False,
    },

    "M19": {
        "name": "GraphSAGE-XGBoost",
        "family": "spatial_dl",
        "task": "static",
        "requires_spatial_data": True,
        "requires_temporal_data": False,
    },

    "M20": {
        "name": "CNN-GNN-Attention",
        "family": "spatial_dl",
        "task": "static",
        "requires_spatial_data": True,
        "requires_temporal_data": False,
    },

    "M21": {
        "name": "CNN-BiLSTM",
        "family": "temporal_dl",
        "task": "forecasting",
        "requires_spatial_data": False,
        "requires_temporal_data": True,
    },

    "M22": {
        "name": "CNN-BiLSTM-Attention",
        "family": "temporal_dl",
        "task": "forecasting",
        "requires_spatial_data": False,
        "requires_temporal_data": True,
    },

    "M23": {
        "name": "CNN-GRU-Attention",
        "family": "temporal_dl",
        "task": "forecasting",
        "requires_spatial_data": False,
        "requires_temporal_data": True,
    },

    "M24": {
        "name": "Transformer-LSTM",
        "family": "temporal_dl",
        "task": "forecasting",
        "requires_spatial_data": False,
        "requires_temporal_data": True,
    },

    "M25": {
        "name": "Temporal Fusion Transformer",
        "family": "temporal_dl",
        "task": "forecasting",
        "requires_spatial_data": False,
        "requires_temporal_data": True,
    },
}


# ============================================================
# 10. DEFAULT MODEL SETTINGS
# ============================================================

# Models are enabled individually after their implementations
# and input requirements have been verified.

MODEL_SETTINGS = {
    "M01": {
        "max_iter": 2000,
        "C": 1.0,
        "class_weight": "balanced",
    },

    "M02": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": 1,
    },

    "M03": {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 1,
    },

    "M04": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_jobs": 1,
    },

    "M05": {
        "hidden_layer_sizes": [64, 32],
        "max_iter": 300,
        "early_stopping": True,
    },

    "M06": {
        "cv": 5,
        "passthrough": False,
    },

    "M07": {
        "weights": [1.0, 1.0, 1.0],
    },

    "M08": {
        "population_size": 6,
        "generations": 3,
        "mutation_rate": 0.2,
        "crossover_rate": 0.5,
        "fitness_sample_size": 3000,
        "inner_cv": 3,
    },

    "M09": {
        "n_estimators": 300,
        "selection_threshold": "median",
    },

    "M10": {
        "cv": 5,
        "use_catboost": True,
    },

    "M11": {
        "hidden_dims": [64, 32],
        "max_epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.001,
        "patience": 10,
    },

    "M12": {
        "latent_dim": 8,
        "hidden_dims": [64, 32],
        "dropout": 0.1,
        "max_epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.001,
        "patience": 10,
    },

    "M13": {
        "embedding_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "max_epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.001,
        "patience": 10,
    },

    "M14": {
        "embedding_dim": 32,
        "num_heads": 4,
        "num_layers": 3,
        "dropout": 0.1,
        "max_epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.001,
        "patience": 10,
    },

    "M15": {
        "channels": 32,
        "dropout": 0.2,
        "max_epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.001,
    },

    "M16": {
        "ensemble_size": 5,
        "hidden_dims": [128, 64, 32],
        "dropout": 0.2,
        "max_epochs": 100,
        "batch_size": 256,
        "learning_rate": 0.001,
    },

    "M17": {
        "hidden_dim": 64,
        "embedding_dim": 32,
        "num_layers": 2,
        "neighbors": 8,
    },

    "M18": {
        "hidden_dim": 64,
        "embedding_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.2,
    },

    "M19": {
        "hidden_dim": 64,
        "embedding_dim": 32,
        "num_layers": 2,
        "neighbors": 8,
    },

    "M20": {
        "patch_size": 9,
        "channels": 32,
        "hidden_dim": 64,
        "embedding_dim": 32,
        "num_heads": 4,
    },

    "M21": {
        "sequence_length": 12,
        "prediction_horizon": 1,
        "hidden_dim": 32,
        "max_epochs": 100,
        "batch_size": 128,
    },

    "M22": {
        "sequence_length": 12,
        "prediction_horizon": 1,
        "hidden_dim": 32,
        "attention_heads": 4,
        "max_epochs": 100,
        "batch_size": 128,
    },

    "M23": {
        "sequence_length": 12,
        "prediction_horizon": 1,
        "hidden_dim": 32,
        "attention_heads": 4,
        "max_epochs": 100,
        "batch_size": 128,
    },

    "M24": {
        "sequence_length": 12,
        "prediction_horizon": 1,
        "transformer_heads": 4,
        "transformer_layers": 2,
        "lstm_hidden_dim": 32,
        "max_epochs": 100,
        "batch_size": 128,
    },

    "M25": {
        "sequence_length": 24,
        "prediction_horizon": 1,
        "hidden_size": 32,
        "attention_heads": 4,
        "max_epochs": 100,
        "batch_size": 128,
    },
}


# ============================================================
# 11. CROSS-VALIDATION SETTINGS
# ============================================================

CV_SETTINGS = {
    "default_method": "block",

    "n_splits": 5,

    "shuffle": True,

    "random_state": SEED,

    "block_column": SPATIAL_BLOCK_COL,

    "event_column": EVENT_COL,

    "temporal_column": TIMESTAMP_COL,

    "allow_random_cv": False,

    "require_both_classes_per_fold": True,

    "save_fold_metrics": True,

    "save_oof_predictions": True,
}


# ============================================================
# 12. SPATIAL MODEL SETTINGS
# ============================================================

SPATIAL_SETTINGS = {
    "enabled": True,

    "longitude_column": "lon",

    "latitude_column": "lat",

    "block_column": SPATIAL_BLOCK_COL,

    "graph_method": "knn",

    "n_neighbors": 8,

    "max_distance_meters": None,

    "coordinate_crs": "EPSG:4326",

    # A projected CRS is recommended when distance in metres
    # is used for graph construction.

    "projected_crs": None,

    "patch_size": 9,

    "graph_validation_mode": "inductive",

    "prevent_cross_fold_message_passing": True,
}


# ============================================================
# 13. TEMPORAL MODEL SETTINGS
# ============================================================

TEMPORAL_SETTINGS = {
    "enabled": False,

    "timestamp_column": None,

    "group_columns": [],

    "sequence_length": 12,

    "prediction_horizon": 1,

    "time_frequency": None,

    "target_column": LABEL_COL,

    "forecasting_mode": "binary_classification",

    "chronological_split": True,

    "allow_future_predictors": False,

    "require_repeated_observations": True,
}


# ============================================================
# 14. TRAINING SETTINGS
# ============================================================

TRAINING = {
    "fast_mode": False,

    "resume": True,

    "save_after_each_fold": True,

    "save_after_each_model": True,

    "save_checkpoints": True,

    "checkpoint_interval_epochs": 5,

    "early_stopping_patience": 10,

    "gradient_clip_norm": 1.0,

    "batch_size": 256,

    "max_epochs": 100,

    "learning_rate": 0.001,

    "weight_decay": 0.0001,

    "num_workers": 0,

    # Avoid uncontrolled parallelism on Apple Silicon.
    "max_cpu_threads": max(1, min(4, os.cpu_count() or 1)),

    "continue_on_model_error": True,

    "continue_on_fold_error": True,

    "atomic_csv_writes": True,
}


# ============================================================
# 15. DEVICE SETTINGS
# ============================================================

DEVICE_SETTINGS = {
    "preferred_device": "auto",

    "allow_cuda": True,

    "allow_mps": True,

    "allow_cpu": True,

    "fallback_to_cpu": True,
}


# ============================================================
# 16. CLASSIFICATION SETTINGS
# ============================================================

CLASSIFICATION = {
    "positive_label": 1,

    "negative_label": 0,

    "probability_threshold": 0.5,

    "zero_division": 0,

    "calculate_roc_auc": True,

    "calculate_pr_auc": True,

    "calculate_brier_score": True,

    "calculate_log_loss": True,

    "calculate_mcc": True,

    "calculate_balanced_accuracy": True,
}


# ============================================================
# 17. EVALUATION SETTINGS
# ============================================================

EVALUATION = {
    "save_confusion_matrix": True,

    "save_roc_curves": True,

    "save_precision_recall_curves": True,

    "save_calibration_curves": True,

    "save_feature_importance": True,

    "save_training_times": True,

    "save_inference_times": True,

    "save_prediction_uncertainty": True,

    "save_model_status": True,
}


# ============================================================
# 18. FAST DEVELOPMENT SETTINGS
# ============================================================

FAST_SETTINGS = {
    "max_epochs": 3,

    "batch_size": 256,

    "ga_population_size": 4,

    "ga_generations": 1,

    "ga_sample_size": 1000,

    "neural_ensemble_size": 2,

    "max_cv_folds": 2,
}


# ============================================================
# 19. ENVIRONMENT AND DEPENDENCY SETTINGS
# ============================================================

OPTIONAL_DEPENDENCIES = {
    "xgboost": True,
    "lightgbm": True,
    "catboost": True,
    "optuna": False,
    "torch_geometric": False,
    "shap": False,
    "pytorch_forecasting": False,
}


# ============================================================
# 20. HELPER FUNCTIONS
# ============================================================

def get_model_config(model_id):
    """Return configuration for a registered model."""

    if model_id not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model ID: {model_id}. "
            f"Valid IDs: {list(MODEL_REGISTRY)}"
        )

    return MODEL_REGISTRY[model_id]


def get_model_settings(model_id):
    """Return a copy of a model's hyperparameter configuration."""

    if model_id not in MODEL_SETTINGS:
        raise KeyError(
            f"No model settings found for {model_id}"
        )

    return dict(MODEL_SETTINGS[model_id])


def get_models_by_family(family):
    """Return model IDs belonging to a model family."""

    return [
        model_id
        for model_id, settings in MODEL_REGISTRY.items()
        if settings["family"] == family
    ]


def get_model_ids():
    """Return all registered model IDs."""

    return list(MODEL_REGISTRY.keys())


def get_static_model_ids():
    """Return model IDs for static susceptibility modelling."""

    return [
        model_id
        for model_id, settings in MODEL_REGISTRY.items()
        if settings["task"] == "static"
    ]


def get_temporal_model_ids():
    """Return model IDs requiring temporal observations."""

    return [
        model_id
        for model_id, settings in MODEL_REGISTRY.items()
        if settings["requires_temporal_data"]
    ]


if __name__ == "__main__":
    print(f"Project: {PROJECT_NAME}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Registered models: {len(MODEL_REGISTRY)}")
    print(f"Static models: {len(get_static_model_ids())}")
    print(f"Temporal models: {len(get_temporal_model_ids())}")








# """
# Central settings. Edit this file, not the other scripts.
# """
# GEE_PROJECT = "flood-20-25-india"      # <-- your Google Cloud project that has Earth Engine enabled
# DRIVE_FOLDER = "flood_susceptibility"    # folder created in your Google Drive for the CSV exports
# SEED = 42
# BLOCK_DEG = 0.25                         # spatial block size (degrees) used for spatial cross-validation

# # How many points to sample per event (balanced: flooded vs not flooded)
# SAMPLING = dict(n_flood=4000, n_nonflood=4000, scale=30, tile_scale=8)

# # Rules used to build the flood label from Sentinel-1 radar (see README, "Label")
# LABEL = dict(
#     diff_db=-3.0,         # flooded if radar backscatter dropped by more than 3 dB vs pre-flood
#     vv_max_db=-15.0,      # ...and is now below -15 dB (open water is very dark in radar)
#     perm_water_occ=50,    # JRC permanent water (>=50% of the time) is excluded
#     slope_max_deg=8.0,    # study domain: only gentle terrain can be flooded
#     hand_max_m=30.0,      # study domain: only land within 30 m height above nearest drainage
# )

# # Flood events. bbox = [west, south, east, north]. CHECK dates against flood reports for your paper.
# # If Earth Engine says "computation timed out", make the bbox smaller or lower n_flood/n_nonflood.
# EVENTS = {
#     "kerala_2018": dict(
#         bbox=[76.2, 9.0, 77.0, 10.5],
#         pre=("2018-01-01", "2018-02-28"),
#         flood=("2018-08-08", "2018-08-22"),
#     ),
#     "assam_2020": dict(
#         bbox=[90.0, 26.0, 92.5, 27.0],
#         pre=("2020-05-15", "2020-06-15"),
#         flood=("2020-06-30", "2020-07-18"),
#     ),
#     "bihar_2020": dict(
#         bbox=[85.0, 25.6, 87.8, 26.6],
#         pre=("2020-05-15", "2020-06-15"),
#         flood=("2020-06-30", "2020-07-18"),
#     ),
#     "assam_2022": dict(
#         bbox=[90.0, 26.0, 92.5, 27.0],
#         pre=("2022-04-01", "2022-04-30"),
#         flood=("2022-06-16", "2022-06-28"),
#     ),
#     # TEMPLATE for Nepal (Terai) or Bihar 2021: fill the dates from an official flood report, then uncomment.
#     # "nepal_terai_YYYY": dict(
#     #     bbox=[86.0, 26.4, 87.4, 27.2],
#     #     pre=("YYYY-MM-DD", "YYYY-MM-DD"),
#     #     flood=("YYYY-MM-DD", "YYYY-MM-DD"),
#     # ),
# }

# # Columns exported from Earth Engine (in this order)
# NUMERIC_FEATURES = [
#     "elevation", "slope", "curvature", "twi", "hand", "dist_river",
#     "rain_event_total", "rain_max_1day", "rain_max_3day", "rain_antecedent_30d",
#     "ndvi_pre", "clay", "sand",
# ]
# CATEGORICAL_FEATURES = ["landcover"]
# LABEL_COL = "flooded"
# META_COLS = ["lon", "lat", "event"]

# WORLDCOVER = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
#               60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}


