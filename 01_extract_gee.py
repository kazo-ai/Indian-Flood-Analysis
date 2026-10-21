"""
Step 1. Build the flood dataset in Google Earth Engine and export one CSV per event to Google Drive.

Run:
    python 01_extract_gee.py --events kerala_2018 assam_2020

Then:
    Monitor progress at https://code.earthengine.google.com/tasks
    Download the CSVs from Google Drive into data/raw/
"""

import argparse
import datetime
import math

import ee
import config as C


def init():
    try:
        ee.Initialize(project=C.GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=C.GEE_PROJECT)


def s1(region, start, end):
    """Sentinel-1 radar, VV and VH, values are already in dB in Earth Engine."""
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VH"
            )
        )
        .select(["VV", "VH"])
    )


def terrain_layers():
    """Generate terrain and drainage predictors."""

    # Digital elevation model and terrain derivatives.
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")

    slope = ee.Terrain.slope(dem).rename("slope")

    curv = dem.convolve(
        ee.Kernel.laplacian8(1, False)
    ).rename("curvature")

    # MERIT Hydro drainage data.
    merit = ee.Image("MERIT/Hydro/v1_0_1")

    hand = merit.select("hnd").rename("hand")

    upa = merit.select("upa").max(0.01)

    # Topographic wetness index.
    tanb = slope.multiply(math.pi / 180).tan().max(0.001)

    twi = (
        upa.multiply(1e6)
        .divide(90)
        .divide(tanb)
        .log()
        .rename("twi")
    )

    # River-distance predictor.
    # Stream cells = 1; non-stream cells = 0.
    # Streams are defined by upstream catchment area >= 10 km².
    river = merit.select("upa").gte(10).unmask(0)

    # Calculate distance to the nearest stream within the 5 km
    # search neighborhood. Distances are expressed in metres.
    dist = (
        river
        .reproject("EPSG:4326", None, 90)
        .distance(ee.Kernel.euclidean(5000, "meters"))
        .rename("dist_river")
    )

    return (
        dem.rename("elevation"),
        slope,
        curv,
        twi,
        hand,
        dist,
    )


def rain_layers(ev):
    """Generate rainfall predictors for a flood event."""

    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .select("precipitation")
    )

    fs, fe = ev["flood"]

    window = chirps.filterDate(fs, fe)

    total = window.sum().rename("rain_event_total")

    max1 = window.max().rename("rain_max_1day")

    n = (
        datetime.date.fromisoformat(fe)
        - datetime.date.fromisoformat(fs)
    ).days

    roll = [
        chirps.filterDate(
            ee.Date(fs).advance(i, "day"),
            ee.Date(fs).advance(i + 3, "day"),
        ).sum()
        for i in range(max(n - 2, 1))
    ]

    max3 = ee.ImageCollection(roll).max().rename("rain_max_3day")

    ante = (
        chirps.filterDate(
            ee.Date(fs).advance(-30, "day"),
            fs,
        )
        .sum()
        .rename("rain_antecedent_30d")
    )

    return total, max1, max3, ante


def land_layers(ev):
    """Generate vegetation, land-cover, and soil predictors."""

    fs = ev["flood"][0]

    ndvi = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .select("NDVI")
        .filterDate(
            ee.Date(fs).advance(-60, "day"),
            fs,
        )
        .median()
        .multiply(0.0001)
        .rename("ndvi_pre")
    )

    lc = (
        ee.ImageCollection("ESA/WorldCover/v200")
        .first()
        .select("Map")
        .rename("landcover")
    )

    clay = (
        ee.Image(
            "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"
        )
        .select("b0")
        .rename("clay")
    )

    sand = (
        ee.Image(
            "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02"
        )
        .select("b0")
        .rename("sand")
    )

    return ndvi, lc, clay, sand


def build_event(name, ev):
    """Build a stratified sample for one flood event."""

    L = C.LABEL

    region = ee.Geometry.Rectangle(ev["bbox"])

    # Sentinel-1 before and during the flood event.
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

    # Weak label: new open water detected by radar.
    diff = post.select("VV").subtract(pre.select("VV"))

    flooded = (
        diff.lt(L["diff_db"])
        .And(post.select("VV").lt(L["vv_max_db"]))
        .rename("flooded")
        .toInt()
    )

    # Exclude permanent water, pre-existing water, steep land,
    # high-HAND locations, and pixels without radar observations.
    gsw = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("occurrence")
        .unmask(0)
    )

    elev, slope, curv, twi, hand, dist = terrain_layers()

    domain = (
        gsw.lt(L["perm_water_occ"])
        .And(pre.select("VV").gte(L["vv_max_db"]))
        .And(slope.lte(L["slope_max_deg"]))
        .And(hand.lte(L["hand_max_m"]))
        .And(post.select("VV").mask())
        .And(pre.select("VV").mask())
    )

    # Additional environmental predictors.
    rain = rain_layers(ev)

    ndvi, lc, clay, sand = land_layers(ev)

    stack = ee.Image.cat([
        elev,
        slope,
        curv,
        twi,
        hand,
        dist,
        *rain,
        ndvi,
        clay,
        sand,
        lc,
        flooded,
    ])

    stack = stack.updateMask(domain)

    # Stratified sampling.
    S = C.SAMPLING

    fc = stack.stratifiedSample(
        numPoints=0,
        classBand="flooded",
        region=region,
        scale=S["scale"],
        classValues=[0, 1],
        classPoints=[S["n_nonflood"], S["n_flood"]],
        seed=C.SEED,
        tileScale=S["tile_scale"],
        geometries=True,
    )

    def add_meta(f):
        xy = ee.List(f.geometry().coordinates())

        return f.set({
            "lon": xy.get(0),
            "lat": xy.get(1),
            "event": name,
        })

    return fc.map(add_meta)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--events",
        nargs="+",
        default=list(C.EVENTS),
    )

    args = ap.parse_args()

    # Reject unknown event names before contacting Earth Engine.
    unknown = sorted(set(args.events) - set(C.EVENTS))

    if unknown:
        ap.error(
            f"Unknown event(s): {unknown}. "
            f"Available events: {list(C.EVENTS)}"
        )

    init()

    cols = (
        C.NUMERIC_FEATURES
        + C.CATEGORICAL_FEATURES
        + [C.LABEL_COL]
        + C.META_COLS
    )

    for name in args.events:
        fc = build_event(name, C.EVENTS[name])

        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=f"flood_{name}",
            folder=C.DRIVE_FOLDER,
            fileNamePrefix=name,
            fileFormat="CSV",
            selectors=cols,
        )

        task.start()

        print(
            f"Started export for {name} "
            f"(task id: {task.id})"
        )

    print("\nMonitor: https://code.earthengine.google.com/tasks")
    print(
        f"When finished, download the CSVs from Google Drive "
        f"folder '{C.DRIVE_FOLDER}' into data/raw/"
    )


if __name__ == "__main__":
    main()




# """
# Step 1. Build the flood dataset in Google Earth Engine and export one CSV per event to Google Drive.

# Run:   python 01_extract_gee.py --events kerala_2018 assam_2020
# Then:  watch progress at https://code.earthengine.google.com/tasks
#        and download the CSVs from Drive into data/raw/
# """
# import argparse, datetime, math
# import ee
# import config as C


# def init():
#     try:
#         ee.Initialize(project=C.GEE_PROJECT)
#     except Exception:
#         ee.Authenticate()
#         ee.Initialize(project=C.GEE_PROJECT)


# def s1(region, start, end):
#     """Sentinel-1 radar, VV and VH, values are already in dB in Earth Engine."""
#     return (ee.ImageCollection("COPERNICUS/S1_GRD")
#             .filterBounds(region).filterDate(start, end)
#             .filter(ee.Filter.eq("instrumentMode", "IW"))
#             .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
#             .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
#             .select(["VV", "VH"]))


# def terrain_layers():
#     dem = ee.Image("USGS/SRTMGL1_003").select("elevation")             # 30 m
#     slope = ee.Terrain.slope(dem).rename("slope")
#     curv = dem.convolve(ee.Kernel.laplacian8(1, False)).rename("curvature")
#     merit = ee.Image("MERIT/Hydro/v1_0_1")                              # 90 m hydrography
#     hand = merit.select("hnd").rename("hand")                           # height above nearest drainage (m)
#     upa = merit.select("upa").max(0.01)                                 # upstream area (km2)
#     tanb = slope.multiply(math.pi / 180).tan().max(0.001)
#     twi = upa.multiply(1e6).divide(90).divide(tanb).log().rename("twi")  # topographic wetness index
#     river = merit.select("upa").gte(10).selfMask()                      # streams: > 10 km2 upstream area
#     dist = (river.reproject("EPSG:4326", None, 90)
#             .distance(ee.Kernel.euclidean(5000, "meters"))
#             .unmask(5000).rename("dist_river"))                        # metres (capped at 20 km)
#     return dem.rename("elevation"), slope, curv, twi, hand, dist


# def rain_layers(ev):
#     chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")  # ~5 km daily
#     fs, fe = ev["flood"]
#     window = chirps.filterDate(fs, fe)
#     total = window.sum().rename("rain_event_total")
#     max1 = window.max().rename("rain_max_1day")
#     n = (datetime.date.fromisoformat(fe) - datetime.date.fromisoformat(fs)).days
#     roll = [chirps.filterDate(ee.Date(fs).advance(i, "day"), ee.Date(fs).advance(i + 3, "day")).sum()
#             for i in range(max(n - 2, 1))]
#     max3 = ee.ImageCollection(roll).max().rename("rain_max_3day")
#     ante = (chirps.filterDate(ee.Date(fs).advance(-30, "day"), fs).sum().rename("rain_antecedent_30d"))
#     return total, max1, max3, ante


# def land_layers(ev):
#     fs = ev["flood"][0]
#     ndvi = (ee.ImageCollection("MODIS/061/MOD13Q1").select("NDVI")
#             .filterDate(ee.Date(fs).advance(-60, "day"), fs).median()
#             .multiply(0.0001).rename("ndvi_pre"))
#     lc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("landcover")
#     clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select("b0").rename("clay")
#     sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select("b0").rename("sand")
#     return ndvi, lc, clay, sand


# def build_event(name, ev):
#     L = C.LABEL
#     region = ee.Geometry.Rectangle(ev["bbox"])
#     pre = s1(region, *ev["pre"]).median().focal_median(30, "circle", "meters")
#     post = s1(region, *ev["flood"]).median().focal_median(30, "circle", "meters")

#     # --- Label: new open water = radar got much darker AND is now dark
#     diff = post.select("VV").subtract(pre.select("VV"))
#     flooded = diff.lt(L["diff_db"]).And(post.select("VV").lt(L["vv_max_db"])).rename("flooded").toInt()

#     # --- Study domain: remove permanent water, water already there before the flood, steep land, no-data
#     gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
#     elev, slope, curv, twi, hand, dist = terrain_layers()
#     domain = (gsw.lt(L["perm_water_occ"])
#               .And(pre.select("VV").gte(L["vv_max_db"]))
#               .And(slope.lte(L["slope_max_deg"]))
#               .And(hand.lte(L["hand_max_m"]))
#               .And(post.select("VV").mask()).And(pre.select("VV").mask()))

#     rain = rain_layers(ev)
#     ndvi, lc, clay, sand = land_layers(ev)
#     stack = ee.Image.cat([elev, slope, curv, twi, hand, dist, *rain, ndvi, clay, sand, lc, flooded])
#     stack = stack.updateMask(domain)

#     S = C.SAMPLING
#     fc = stack.stratifiedSample(numPoints=0, classBand="flooded", region=region, scale=S["scale"],
#                                 classValues=[0, 1], classPoints=[S["n_nonflood"], S["n_flood"]],
#                                 seed=C.SEED, tileScale=S["tile_scale"], geometries=True)

#     def add_meta(f):
#         xy = ee.List(f.geometry().coordinates())
#         return f.set({"lon": xy.get(0), "lat": xy.get(1), "event": name})
#     return fc.map(add_meta)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--events", nargs="+", default=list(C.EVENTS))
#     args = ap.parse_args()
#     init()
#     cols = C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES + [C.LABEL_COL] + C.META_COLS
#     for name in args.events:
#         fc = build_event(name, C.EVENTS[name])
#         task = ee.batch.Export.table.toDrive(collection=fc, description=f"flood_{name}",
#                                              folder=C.DRIVE_FOLDER, fileNamePrefix=name,
#                                              fileFormat="CSV", selectors=cols)
#         task.start()
#         print(f"started export for {name}  (task id: {task.id})")
#     print("\nMonitor: https://code.earthengine.google.com/tasks")
#     print(f"When finished, download the CSVs from Google Drive folder '{C.DRIVE_FOLDER}' into data/raw/")


# if __name__ == "__main__":
#     main()
