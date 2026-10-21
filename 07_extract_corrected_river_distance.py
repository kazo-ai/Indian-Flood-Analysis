import ee
import pandas as pd
from pathlib import Path

import config as C

RAW_DIR = Path("data/raw")
OUTPUT_DIR = RAW_DIR / "corrected_river_distance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_CRS = {
    "assam_2020": "EPSG:32646",
    "assam_2022": "EPSG:32646",
    "bihar_2020": "EPSG:32645",
    "kerala_2018": "EPSG:32643",
}

SCALE_M = 90
MAX_DISTANCE_M = 5000


def init_ee():
    try:
        ee.Initialize(project=C.GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=C.GEE_PROJECT)


def build_distance_image(crs):
    merit = ee.Image("MERIT/Hydro/v1_0_1")

    # Stream cells have upstream catchment area >= 10 km².
    river = merit.select("upa").gte(10).unmask(0)

    # Calculate distance on the event's metric grid.
    return (
        river
        .reproject(crs=crs, scale=SCALE_M)
        .distance(ee.Kernel.euclidean(MAX_DISTANCE_M, "meters"))
        .unmask(MAX_DISTANCE_M)
        .clamp(0, MAX_DISTANCE_M)
        .rename("dist_river")
    )


def extract_event(event_name):
    csv_path = RAW_DIR / f"{event_name}.csv"
    df = pd.read_csv(csv_path)

    distance = build_distance_image(EVENT_CRS[event_name])

    features = [
        ee.Feature(
            ee.Geometry.Point([float(row.lon), float(row.lat)]),
            {"row_id": int(i)},
        )
        for i, row in df.iterrows()
    ]

    points = ee.FeatureCollection(features)

    sampled = distance.sampleRegions(
        collection=points,
        properties=["row_id"],
        scale=SCALE_M,
        projection=distance.projection(),
        geometries=False,
        tileScale=4,
    )

    total = sampled.size().getInfo()
    results = []

    for offset in range(0, total, 1000):
        cd ~/Downloads/flood_susceptibility

echo "=== SUSCEPTIBILITY FIGURES ==="
ls -lt outputs/figures/m21_susceptibility*

echo "=== MAP PLOTTING SCRIPT ==="
ls -lT 06_plot_susceptibility_maps.py

echo "=== FIGURE SCRIPTS ==="
ls -lT generate_research_figures.py generate_research_figures_pre_polish.pybatch = sampled.toList(1000, offset).getInfo()
        results.extend(batch)
        print(f"Retrieved {len(results)}/{total} points")

    print(f"Retrieved {len(results)} points in total")

    values = {
        int(f["properties"]["row_id"]):
        f["properties"].get("dist_river")
        for f in results
    }

    output = pd.DataFrame({
        "row_id": range(len(df)),
        "dist_river": [
            values.get(i) for i in range(len(df))
        ],
    })

    output_path = OUTPUT_DIR / f"{event_name}_river_distance.csv"
    output.to_csv(output_path, index=False)

    valid = output["dist_river"].notna()

    print(f"\n{event_name}")
    print(f"  Original rows: {len(df):,}")
    print(f"  Extracted values: {valid.sum():,}")
    print(f"  Missing values: {(~valid).sum():,}")

    if valid.any():
        print(f"  Minimum: {output.loc[valid, 'dist_river'].min():.2f} m")
        print(f"  Median: {output.loc[valid, 'dist_river'].median():.2f} m")
        print(f"  Maximum: {output.loc[valid, 'dist_river'].max():.2f} m")

    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    init_ee()

    for event_name in EVENT_CRS:
        extract_event(event_name)

    print("\nExtraction finished. Original CSVs were not modified.")