from pathlib import Path
import numpy as np
import rasterio

folder = Path("data/raw/predictor_rasters")

for path in sorted(folder.glob("predictors_*_90m.tif")):
    print("\n" + "=" * 75)
    print(path.name)

    with rasterio.open(path) as src:
        print(f"Dimensions: {src.width} x {src.height}")
        print(f"CRS: {src.crs}")
        print(f"Resolution: {src.res}")
        print(f"Raster NoData value: {src.nodata}")

        total = src.width * src.height

        for i, name in enumerate(src.descriptions, start=1):
            arr = src.read(i)
            finite = np.isfinite(arr).sum()
            missing = total - finite

            print(
                f"{str(name):24s} "
                f"Missing: {missing:>10,} "
                f"({missing / total * 100:6.2f}%)"
            )
