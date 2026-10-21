"""
Apply saved M21 pipeline to predictor GeoTIFFs in chunks.
Run: python 05_predict_susceptibility.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

INPUT_DIR = Path("data/raw/predictor_rasters")
MODEL_PATH = Path("outputs/models/m21_extra_trees.joblib")
METADATA_PATH = Path("outputs/models/m21_extra_trees_metadata.json")
OUTPUT_DIR = Path("outputs/susceptibility")
NODATA_OUT = -9999.0
CHUNK_ROWS = 128

DOMAIN_CHECK_BANDS = [
    "elevation", "slope", "curvature", "twi", "hand", "dist_river",
    "rain_event_total", "rain_max_1day", "rain_max_3day", "rain_antecedent_30d",
]
LANDCOVER_BANDS = [
    "lc_bare", "lc_built", "lc_crop", "lc_grass",
    "lc_shrub", "lc_tree", "lc_water", "lc_wetland",
]


def event_from_filename(path):
    prefix, suffix = "predictors_", "_90m.tif"
    if path.name.startswith(prefix) and path.name.endswith(suffix):
        return path.name[len(prefix):-len(suffix)]
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open() as f:
        metadata = json.load(f)
    feature_names = metadata["feature_columns"]
    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise TypeError("Saved model does not support predict_proba.")
    if getattr(model, "n_features_in_", None) != len(feature_names):
        raise ValueError("Model feature count does not match metadata.")

    paths = sorted(INPUT_DIR.glob("predictors_*_90m.tif"))
    if not paths:
        raise FileNotFoundError(f"No predictor GeoTIFFs found in {INPUT_DIR}")

    summaries = []
    for input_path in paths:
        event = event_from_filename(input_path)
        if event is None:
            continue
        output_path = OUTPUT_DIR / f"m21_susceptibility_{event}.tif"
        print(f"\nProcessing {event}: {input_path.name}")

        with rasterio.open(input_path) as src:
            band_names = list(src.descriptions)
            if band_names != feature_names:
                raise ValueError(
                    f"{input_path.name}: raster band order differs from model metadata."
                )

            domain_indices = [
                band_names.index(name) for name in DOMAIN_CHECK_BANDS + LANDCOVER_BANDS
            ]
            profile = src.profile.copy()
            profile.update(
                driver="GTiff", count=1, dtype="float32", nodata=NODATA_OUT,
                compress="deflate", predictor=3, tiled=True, BIGTIFF="IF_SAFER"
            )

            n_valid = n_nodata = n_imputed_rows = 0
            p_sum, p_min, p_max = 0.0, 1.0, 0.0

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.set_band_description(1, "m21_probability_of_derived_flood_label")
                dst.update_tags(
                    model_id="M21",
                    event=event,
                    target_definition=metadata.get("label_note", "Sentinel-1-derived new-open-water label."),
                    nodata_policy="Outside domain=-9999; isolated predictor NaNs imputed by saved pipeline."
                )

                for row in range(0, src.height, CHUNK_ROWS):
                    height = min(CHUNK_ROWS, src.height - row)
                    window = Window(0, row, src.width, height)
                    cube = src.read(window=window, out_dtype="float32")
                    flat = cube.reshape(cube.shape[0], -1).T
                    finite = np.isfinite(flat)

                    # Outside-domain cells have NaNs in core terrain/rainfall or
                    # landcover bands. Soil/NDVI may have isolated missing values
                    # inside the domain and are handled by the fitted imputer.
                    valid = finite[:, domain_indices].all(axis=1)
                    valid_idx = np.flatnonzero(valid)
                    output = np.full(flat.shape[0], NODATA_OUT, dtype=np.float32)

                    if valid_idx.size:
                        X = flat[valid_idx].copy()
                        X[~np.isfinite(X)] = np.nan
                        probs = model.predict_proba(X)[:, 1].astype(np.float32)
                        if not np.isfinite(probs).all():
                            raise ValueError(f"{event}: non-finite predictions.")
                        output[valid_idx] = probs
                        n_valid += int(valid_idx.size)
                        n_imputed_rows += int(np.isnan(X).any(axis=1).sum())
                        p_sum += float(probs.sum())
                        p_min = min(p_min, float(probs.min()))
                        p_max = max(p_max, float(probs.max()))

                    n_nodata += int((~valid).sum())
                    dst.write(output.reshape(height, src.width), 1, window=window)

        total = n_valid + n_nodata
        summaries.append({
            "event": event,
            "input_file": input_path.name,
            "output_file": output_path.name,
            "width": profile["width"],
            "height": profile["height"],
            "crs": str(profile["crs"]),
            "nominal_pixel_size_m": 90,
            "valid_prediction_pixels": n_valid,
            "nodata_pixels": n_nodata,
            "valid_percent": round(100 * n_valid / total, 3) if total else 0,
            "pixels_with_imputed_predictors": n_imputed_rows,
            "probability_min": p_min if n_valid else np.nan,
            "probability_mean": p_sum / n_valid if n_valid else np.nan,
            "probability_max": p_max if n_valid else np.nan,
        })
        print(f"  Valid prediction pixels: {n_valid:,}")
        print(f"  NoData pixels: {n_nodata:,}")
        if n_valid:
            print(f"  Probability range: {p_min:.4f}–{p_max:.4f}; mean={p_sum/n_valid:.4f}")
        print(f"  Saved: {output_path}")

    summary_path = OUTPUT_DIR / "m21_susceptibility_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")
    print("These probabilities predict the Sentinel-1-derived label, not independently verified flood risk.")


if __name__ == "__main__":
    main()
