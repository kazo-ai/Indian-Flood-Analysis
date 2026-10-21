from pathlib import Path
import json
import joblib
import numpy as np
import rasterio

ROOT = Path(".")
PRED_DIR = ROOT / "data/raw/predictor_rasters"
OUT_DIR = ROOT / "outputs/susceptibility"
MODEL_PATH = ROOT / "outputs/models/m21_extra_trees.joblib"
META_PATH = ROOT / "outputs/models/m21_extra_trees_metadata.json"

model = joblib.load(MODEL_PATH)

with open(META_PATH) as f:
    metadata = json.load(f)

expected = metadata["feature_columns"]

print("EXPECTED FEATURES:", len(expected))
print("MODEL INPUT FEATURES:", model.named_steps["imputer"].n_features_in_)
print("FEATURE ORDER MATCH:", len(expected) == 21)

events = [
    "assam_2020",
    "assam_2022",
    "bihar_2020",
    "kerala_2018",
]

for event in events:
    pred_path = PRED_DIR / f"predictors_{event}_90m.tif"
    out_path = OUT_DIR / f"m21_susceptibility_{event}.tif"

    print("\n" + "=" * 65)
    print(event)

    with rasterio.open(pred_path) as src:
        names = list(src.descriptions)

        print("Predictor CRS:", src.crs)
        print("Predictor dimensions:", src.width, "x", src.height)
        print("Predictor resolution:", src.res)
        print("Band count:", src.count)
        print("Feature names/order match:", names == expected)
        print("All bands Float32:", all(
            np.dtype(dtype) == np.dtype("float32")
            for dtype in src.dtypes
        ))

        if names != expected:
            print("WARNING: feature order mismatch")
        if src.count != len(expected):
            print("WARNING: incorrect number of bands")

    with rasterio.open(out_path) as pred:
        arr = pred.read(1)
        valid = np.isfinite(arr) & (arr != pred.nodata)

        print("Output CRS:", pred.crs)
        print("Output dimensions:", pred.width, "x", pred.height)
        print("Output resolution:", pred.res)
        print("Output NoData value:", pred.nodata)
        print("Dimensions match:", (
            pred.width == src.width and pred.height == src.height
        ))
        print("CRS matches:", pred.crs == src.crs)
        print("Valid output pixels:", int(valid.sum()))

        if valid.any():
            values = arr[valid]
            print("Probability min:", float(values.min()))
            print("Probability max:", float(values.max()))
            print("All probabilities in [0, 1]:", bool(
                np.all((values >= 0) & (values <= 1))
            ))
        else:
            print("WARNING: no valid output pixels")

print("\nValidation pass finished.")
