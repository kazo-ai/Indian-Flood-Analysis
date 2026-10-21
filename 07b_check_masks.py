from pathlib import Path
import numpy as np
import rasterio

folder = Path("data/raw/predictor_rasters")

core = [
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
]

for path in sorted(folder.glob("predictors_*_90m.tif")):
    print("\n" + "=" * 65)
    print(path.name)

    with rasterio.open(path) as src:
        names = list(src.descriptions)
        arrays = {
            name: src.read(i).astype(np.float32)
            for i, name in enumerate(names, start=1)
        }

        masks = {
            name: np.isfinite(arrays[name])
            for name in names
        }

        reference = masks["elevation"]

        print("Pixels in raster:", reference.size)
        print("Elevation-valid pixels:", int(reference.sum()))

        for name in core:
            same = np.array_equal(reference, masks[name])
            print(f"{name:24s} same mask as elevation: {same}")

        # Count pixels where elevation is valid but other predictors are not.
        print("\nMissing predictors inside elevation-valid pixels:")

        for name in names:
            missing_inside = reference & ~masks[name]
            count = int(missing_inside.sum())

            if count:
                print(f"{name:24s} {count:>10,}")

        # Count pixels where every core predictor is finite.
        all_core_valid = np.logical_and.reduce(
            [masks[name] for name in core]
        )

        print(
            "\nPixels with every core predictor valid:",
            f"{int(all_core_valid.sum()):,}",
        )
