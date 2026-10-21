"""
Export the 21 M21 predictor bands from Google Earth Engine as one GeoTIFF per event.

Run from the project root:
    source .venv/bin/activate
    python 04_export_predictor_rasters.py --events assam_2020 assam_2022 bihar_2020 kerala_2018

The script starts Earth Engine Drive export tasks; it does not download the files automatically.
Monitor tasks at https://code.earthengine.google.com/tasks, then download the GeoTIFFs
from the configured Google Drive folder into data/raw/predictor_rasters/.
"""
import argparse
import datetime
import json
import math
from pathlib import Path

import ee
import config as C


# UTM zones appropriate for the configured event bounding boxes.
EVENT_CRS = {
    "assam_2020": "EPSG:32646",
    "assam_2022": "EPSG:32646",
    "bihar_2020": "EPSG:32645",
    "kerala_2018": "EPSG:32643",
}

LANDCOVER_CODES = {
    "lc_bare": 60,
    "lc_built": 50,
    "lc_crop": 40,
    "lc_grass": 30,
    "lc_shrub": 20,
    "lc_tree": 10,
    "lc_water": 80,
    "lc_wetland": 90,
}

EXPORT_SCALE_M = 90


def init_ee():
    try:
        ee.Initialize(project=C.GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=C.GEE_PROJECT)


def s1(region, start, end):
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )


def terrain_layers(crs):
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    slope = ee.Terrain.slope(dem).rename("slope")
    curvature = dem.convolve(
        ee.Kernel.laplacian8(1, False)
    ).rename("curvature")

    merit = ee.Image("MERIT/Hydro/v1_0_1")
    hand = merit.select("hnd").rename("hand")
    upa = merit.select("upa").max(0.01)
    tanb = slope.multiply(math.pi / 180).tan().max(0.001)
    twi = upa.multiply(1e6).divide(90).divide(tanb).log().rename("twi")

     # Binary stream image: 1 = stream, 0 = non-stream.
    river = merit.select("upa").gte(10).unmask(0)

    # Calculate distance on the event's projected grid.
    # Cap distances at 5,000 metres.
    dist = (
        river
        .reproject(crs, None, 90)
        .distance(ee.Kernel.euclidean(5000, "meters"))
        .unmask(5000)
        .clamp(0, 5000)
        .rename("dist_river")
    )

    return {
        "elevation": dem.rename("elevation"),
        "slope": slope,
        "curvature": curvature,
        "twi": twi,
        "hand": hand,
        "dist_river": dist,
    }


def rain_layers(ev):
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
    start, end = ev["flood"]
    window = chirps.filterDate(start, end)

    total = window.sum().rename("rain_event_total")
    max1 = window.max().rename("rain_max_1day")

    n_days = (
        datetime.date.fromisoformat(end) -
        datetime.date.fromisoformat(start)
    ).days

    rolling_3day = [
        chirps.filterDate(
            ee.Date(start).advance(i, "day"),
            ee.Date(start).advance(i + 3, "day"),
        ).sum()
        for i in range(max(n_days - 2, 1))
    ]
    max3 = ee.ImageCollection(rolling_3day).max().rename("rain_max_3day")
    antecedent = (
        chirps.filterDate(ee.Date(start).advance(-30, "day"), start)
        .sum()
        .rename("rain_antecedent_30d")
    )

    return {
        "rain_event_total": total,
        "rain_max_1day": max1,
        "rain_max_3day": max3,
        "rain_antecedent_30d": antecedent,
    }


def land_layers(ev):
    flood_start = ev["flood"][0]

    ndvi = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .select("NDVI")
        .filterDate(ee.Date(flood_start).advance(-60, "day"), flood_start)
        .median()
        .multiply(0.0001)
        .rename("ndvi_pre")
    )

    worldcover = (
        ee.ImageCollection("ESA/WorldCover/v200")
        .first()
        .select("Map")
        .rename("landcover")
    )

    clay = (
        ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02")
        .select("b0")
        .rename("clay")
    )
    sand = (
        ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02")
        .select("b0")
        .rename("sand")
    )

    layers = {
        "ndvi_pre": ndvi,
        "clay": clay,
        "sand": sand,
    }

    # Match the training dataset's eight retained one-hot predictor columns.
    for band_name, worldcover_code in LANDCOVER_CODES.items():
        layers[band_name] = (
            worldcover.eq(worldcover_code).toByte().rename(band_name)
        )

    return layers, worldcover


def build_predictors(event_name, ev, feature_names):
    region = ee.Geometry.Rectangle(ev["bbox"])

    pre = (
        s1(region, *ev["pre"])
        .median()
        .focal_median(30, "circle", "meters")
    )
    post = (
        s1(region, *ev["flood"])
        .median()
        .focal_median(30, "circle", "meters")
    )

    layers = terrain_layers(EVENT_CRS[event_name])
    layers.update(rain_layers(ev))
    land, worldcover = land_layers(ev)
    layers.update(land)

    # Match the training sampling domain, but do not include the flood label
    # or any label-derived quantity as a model predictor.
    label_cfg = C.LABEL
    gsw = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .unmask(0)
    )
    domain = (
        gsw.lt(label_cfg["perm_water_occ"])
        .And(pre.select("VV").gte(label_cfg["vv_max_db"]))
        .And(layers["slope"].lte(label_cfg["slope_max_deg"]))
        .And(layers["hand"].lte(label_cfg["hand_max_m"]))
        .And(post.select("VV").mask())
        .And(pre.select("VV").mask())
    )

    missing = [name for name in feature_names if name not in layers]
    extra = [name for name in layers if name not in feature_names]
    if missing:
        raise ValueError(f"{event_name}: predictor layers missing: {missing}")
    if extra:
        print(f"Note: layers not used by M21 are omitted: {extra}")

    # Earth Engine image exports require all bands to share a compatible data type.
    # Predictors include integer land-cover indicators and floating-point continuous
    # variables, so cast the stacked image to Float32 before exporting.
    image = ee.Image.cat([layers[name].rename(name) for name in feature_names])
    image = image.toFloat()
    return image.updateMask(domain).clip(region), region


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events",
        nargs="+",
        default=list(C.EVENTS),
        choices=list(C.EVENTS),
        help="Event(s) to export.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=EXPORT_SCALE_M,
        help="Nominal output grid scale in metres (default: 90).",
    )
    args = parser.parse_args()

    if args.scale <= 0:
        raise ValueError("--scale must be a positive integer.")

    metadata_path = Path("outputs/models/m21_extra_trees_metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Train/save the M21 model first."
        )
    with metadata_path.open() as f:
        metadata = json.load(f)

    feature_names = metadata.get("feature_columns")
    if not feature_names or len(feature_names) != 21:
        raise ValueError("Model metadata must contain the expected 21 feature_columns.")

    expected_lc = set(LANDCOVER_CODES)
    actual_lc = {name for name in feature_names if name.startswith("lc_")}
    if actual_lc != expected_lc:
        raise ValueError(
            "Land-cover columns differ from script mapping. "
            f"Expected {sorted(expected_lc)}, got {sorted(actual_lc)}"
        )

    init_ee()

    for event_name in args.events:
        if event_name not in EVENT_CRS:
            raise ValueError(f"No configured CRS for {event_name}")

        ev = C.EVENTS[event_name]
        image, region = build_predictors(event_name, ev, feature_names)
        description = f"m21_predictors_{event_name}_{args.scale}m"
        prefix = f"predictors_{event_name}_{args.scale}m"

        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=C.DRIVE_FOLDER,
            fileNamePrefix=prefix,
            region=region,
            scale=args.scale,
            crs=EVENT_CRS[event_name],
            maxPixels=1e13,
            fileFormat="GeoTIFF",
        )
        task.start()
        print(
            f"Started {event_name}: task_id={task.id}, "
            f"CRS={EVENT_CRS[event_name]}, scale={args.scale} m, "
            f"bands={len(feature_names)}, prefix={prefix}"
        )

    print("\nMonitor tasks at https://code.earthengine.google.com/tasks")
    print(f"Exports are being sent to Google Drive folder: {C.DRIVE_FOLDER}")
    print(
        "After export completes, download the GeoTIFFs into "
        "data/raw/predictor_rasters/."
    )
    print(
        "Caution: WorldCover is a 2021 product and therefore post-dates some "
        "historical events; document this temporal limitation."
    )


if __name__ == "__main__":
    main()
