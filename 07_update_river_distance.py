from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

EVENTS = [
    "assam_2020",
    "assam_2022",
    "bihar_2020",
    "kerala_2018",
]

RAW_DIR = Path("data/raw")
RASTER_DIR = RAW_DIR / "predictor_rasters"

for event in EVENTS:
    csv_path = RAW_DIR / f"{event}.csv"
    raster_path = RASTER_DIR / f"predictors_{event}_90m.tif"

    df = pd.read_csv(csv_path)

    with rasterio.open(raster_path) as src:
        xs, ys = transform(
            "EPSG:4326",
            src.crs,
            df["lon"].tolist(),
            df["lat"].tolist(),
        )

        sampled = np.array([
            values[0]
            for values in src.sample(
                list(zip(xs, ys)),
                indexes=6,
            )
        ], dtype=float)

        if src.nodata is not None:
            sampled[sampled == src.nodata] = np.nan

    valid = np.isfinite(sampled)

    print(f"\n{event}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Valid river distances: {valid.sum():,}")
    print(f"  Missing river distances: {(~valid).sum():,}")

    if valid.any():
        print(f"  Minimum: {sampled[valid].min():.2f} m")
        print(f"  Median: {np.median(sampled[valid]):.2f} m")
        print(f"  Maximum: {sampled[valid].max():.2f} m")

    # Report missing locations without modifying the original CSV.
    if (~valid).any():
        report = df.loc[~valid, ["lon", "lat", "flooded"]].copy()
        report["sampled_dist_river"] = np.nan

        report_path = RAW_DIR / f"{event}_missing_river_points.csv"
        report.to_csv(report_path, index=False)

        print(f"  Missing-point report: {report_path}")

print("\nDiagnostic complete. Original CSVs were not modified.")



# from pathlib import Path
# import shutil

# import numpy as np
# import pandas as pd
# import rasterio
# from rasterio.warp import transform

# EVENTS = [
#     "assam_2020",
#     "assam_2022",
#     "bihar_2020",
#     "kerala_2018",
# ]

# RAW_DIR = Path("data/raw")
# RASTER_DIR = RAW_DIR / "predictor_rasters"
# BACKUP_DIR = RAW_DIR / "backup_before_river_fix"

# BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# for event in EVENTS:
#     csv_path = RAW_DIR / f"{event}.csv"
#     raster_path = RASTER_DIR / f"predictors_{event}_90m.tif"

#     if not csv_path.exists():
#         raise FileNotFoundError(f"Missing CSV: {csv_path}")

#     if not raster_path.exists():
#         raise FileNotFoundError(f"Missing raster: {raster_path}")

#     df = pd.read_csv(csv_path)

#     required = {"lon", "lat", "dist_river"}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

#     # Preserve the original CSV before modifying it.
#     backup_path = BACKUP_DIR / csv_path.name
#     if not backup_path.exists():
#         shutil.copy2(csv_path, backup_path)

#     with rasterio.open(raster_path) as src:
#         if src.count != 21:
#             raise ValueError(
#                 f"{raster_path}: expected 21 bands, found {src.count}"
#             )

#         # Band 6 is dist_river in the established 21-band layout.
#         xs, ys = transform(
#             "EPSG:4326",
#             src.crs,
#             df["lon"].to_numpy(dtype=float).tolist(),
#             df["lat"].to_numpy(dtype=float).tolist(),
#         )

#         coords = list(zip(xs, ys))
#         sampled = np.array(
#             [value[0] for value in src.sample(coords, indexes=6)],
#             dtype=float,
#         )

#         nodata = src.nodata
#         if nodata is not None:
#             sampled[sampled == nodata] = np.nan

#         if not np.isfinite(sampled).all():
#             bad = int((~np.isfinite(sampled)).sum())
#             raise ValueError(
#                 f"{event}: {bad} sample points have invalid river distances. "
#                 "Original CSV remains unchanged."
#             )

#         if (sampled < 0).any() or (sampled > 5000).any():
#             raise ValueError(
#                 f"{event}: river distances outside the expected 0–5000 m range. "
#                 "Original CSV remains unchanged."
#             )

#     old = df["dist_river"].to_numpy(dtype=float)
#     changed = int(np.count_nonzero(~np.isclose(old, sampled)))

#     df["dist_river"] = sampled
#     df.to_csv(csv_path, index=False)

#     print(f"\n{event}")
#     print(f"  Rows preserved: {len(df):,}")
#     print(f"  River-distance values changed: {changed:,}")
#     print(f"  Unique river distances: {df['dist_river'].nunique():,}")
#     print(f"  Minimum: {df['dist_river'].min():.2f} m")
#     print(f"  Median: {df['dist_river'].median():.2f} m")
#     print(f"  Maximum: {df['dist_river'].max():.2f} m")
#     print(f"  Backup: {backup_path}")

# print("\nCSV updates complete.")
# print("Next: verify the four CSVs before rebuilding the processed dataset.")
