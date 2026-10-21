# Flood Susceptibility Mapping Using Machine Learning

## 1. Project Overview

This project develops a machine-learning pipeline for mapping flood
susceptibility across four historical flood events in India:

- Kerala, 2018
- Assam, 2020
- Assam, 2022
- Bihar, 2020

The pipeline integrates remotely sensed information, terrain characteristics,
rainfall, vegetation, hydrological variables, soil properties, and land cover.

The principal model configuration, M21, uses 21 predictor features and an
Extra Trees classifier to generate pixel-level flood susceptibility maps.

The project includes spatial block cross-validation, event-held-out
evaluation, model comparisons, sensitivity analysis, feature importance,
raster validation, and a Streamlit dashboard.

## 2. Project Status

Completed:
- Four-event dataset construction and preprocessing
- Construction of a 21-predictor feature set
- Extra Trees model training
- Spatial block cross-validation
- Event-held-out cross-validation
- Susceptibility raster generation
- Feature importance and permutation importance analysis
- Threshold, hyperparameter, and predictor sensitivity analysis
- Raster consistency checks
- Research figure generation
- Streamlit dashboard

Pending:
- Independent validation against external flood observations
- Fair re-evaluation of competing models using identical corrected data
  and validation splits
- Final model selection after the fair comparison

## 3. Dataset and Labels

The processed dataset is located at:

`data/processed/flood_dataset.csv`

It contains approximately 32,000 sampled observations and 21 predictor
features, alongside event identifiers, labels, and other metadata.

The binary target is named `flooded`.

### Important label limitation

The target is derived from Sentinel-1 radar backscatter changes and is a
weak label, not independently verified flood ground truth.

Consequently, model performance measures agreement with the constructed
labels. They do not, by themselves, establish equivalent accuracy against
observed flood extent.

The sampled dataset is approximately balanced between the two classes.
Metrics such as precision therefore should not be interpreted as estimates
of precision under the actual prevalence of flooding in the landscape.

ESA WorldCover 2021 is used in the processing workflow. Because the study
includes historical events, the temporal mismatch between this land-cover
product and the event dates is a limitation.

## 4. Predictor Variables

The M21 model uses 21 predictors spanning the following categories:

- Land cover
- Terrain
- Rainfall
- Vegetation
- Hydrology
- Soil properties

The predictor rasters are stored in:

`data/raw/predictor_rasters/`

The raster workflow uses a nominal 90 m grid. Coordinate reference systems
are event-specific projected coordinate systems.

See `config.py` and the processing scripts for the precise predictor
definitions and event configurations.

## 5. Model

The final M21 model is an Extra Trees classifier.

Model artifact:

`outputs/models/m21_extra_trees.joblib`

The model uses 400 trees, square-root feature subsampling, balanced class
weights, a fixed random seed, and median imputation within the model
pipeline.

The model artifact and associated metadata should be treated as the
reproducible outputs of the current training run.

## 6. Validation Methodology

### Spatial block cross-validation

Spatial block cross-validation evaluates the model using geographically
separated groups of observations. This reduces the optimistic estimates
that can arise when neighbouring pixels appear in both training and
validation sets.

Current M21 results:

| Metric | Mean |
|---|---:|
| ROC-AUC | 0.9072 |
| F1 score | 0.8350 |
| Intersection over Union | 0.7168 |
| Accuracy | 0.8244 |

### Event-held-out cross-validation

Event-held-out evaluation tests transfer to a flood event excluded from
training. It is a more demanding test of generalization across events,
although results can vary substantially with event characteristics.

Current M21 results:

| Metric | Mean |
|---|---:|
| ROC-AUC | 0.8656 |
| F1 score | 0.8182 |
| Intersection over Union | 0.7013 |
| Accuracy | 0.8053 |

The four event-held-out ROC-AUC values range from approximately 0.7033
to 0.9630. This variation indicates that performance is not uniform
across events.

These results are internal evaluations against the constructed labels.
Independent validation remains necessary.

## 7. Interpretability and Sensitivity Analysis

The project includes:

- Predictor and grouped-feature importance
- Permutation importance
- Importance-ranking stability across validation schemes
- Classification threshold sensitivity
- Hyperparameter sensitivity
- Sensitivity to removing land-cover predictors

Land-cover predictors are among the most influential features in the
current model. Removing land-cover information reduces internal
cross-validation performance.

Feature importance describes model reliance, not necessarily causal
relationships with flooding.

## 8. Susceptibility Maps

Susceptibility GeoTIFFs are stored in:

`outputs/susceptibility/`

The expected event outputs are:

- `m21_susceptibility_assam_2020.tif`
- `m21_susceptibility_assam_2022.tif`
- `m21_susceptibility_bihar_2020.tif`
- `m21_susceptibility_kerala_2018.tif`

The raster values represent model-estimated susceptibility scores
between 0 and 1 for valid prediction pixels. They should not be interpreted
as calibrated real-world flood probabilities unless calibration has
been independently established.

The plotting script is:

`06_plot_susceptibility_maps.py`

Raster validation is performed by:

`08_validate_rasters.py`

## 9. Dashboard

The Streamlit dashboard is defined in `app.py`.

It provides six sections:

1. Overview
2. Model comparison
3. Event generalization
4. Out-of-fold diagnostics
5. Dataset explorer
6. Research figures

To launch the dashboard:

```bash
source .venv/bin/activate
python -m pip install -r requirements_streamlit.txt
streamlit run app.py# Indian-Flood-Analysis
