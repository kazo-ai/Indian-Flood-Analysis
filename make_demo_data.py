"""
DEMO ONLY. Creates FAKE CSV files with the same columns as the Earth Engine export,
so you can test steps 2 and 3 before your real data is ready.
Results from this fake data mean nothing. Delete data/raw/demo_* before using real data.
"""
import argparse, numpy as np, pandas as pd
import config as C

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=4000)
n = ap.parse_args().n
rng = np.random.default_rng(0)
centres = {"kerala_2018": (76.6, 9.8), "assam_2020": (91.2, 26.5), "bihar_2020": (86.4, 26.1), "assam_2022": (91.0, 26.4)}
for k, (lon0, lat0) in centres.items():
    lon = lon0 + rng.uniform(-0.4, 0.4, n); lat = lat0 + rng.uniform(-0.4, 0.4, n)
    hand = rng.gamma(1.5, 4, n).clip(0, 30); slope = rng.gamma(1.5, 1.5, n).clip(0, 8)
    twi = rng.normal(9, 2, n); dist = rng.gamma(2, 1500, n).clip(0, 20000)
    cell = (np.floor(lon / 0.05) * 1000 + np.floor(lat / 0.05)).astype(int)     # rainfall is constant per ~5 km cell
    cell_rain = {c: rng.gamma(6, 25) for c in np.unique(cell)}
    rain = np.array([cell_rain[c] for c in cell])
    z = -0.15 * hand - 0.25 * slope + 0.35 * (twi - 9) - 0.0002 * dist + 0.004 * (rain - 150) + rng.normal(0, 1.0, n)
    flooded = (z + rng.normal(0, 0.5, n) > np.percentile(z, 50)).astype(int)
    df = pd.DataFrame(dict(
        elevation=hand + rng.normal(20, 5, n), slope=slope, curvature=rng.normal(0, 1, n), twi=twi,
        hand=hand, dist_river=dist, rain_event_total=rain, rain_max_1day=rain * rng.uniform(0.2, 0.35, n),
        rain_max_3day=rain * rng.uniform(0.4, 0.6, n), rain_antecedent_30d=rain * rng.uniform(0.8, 1.5, n),
        ndvi_pre=rng.uniform(0.2, 0.8, n), clay=rng.uniform(10, 50, n), sand=rng.uniform(10, 60, n),
        landcover=rng.choice([10, 30, 40, 50, 90], n, p=[.15, .1, .55, .1, .1]),
        flooded=flooded, lon=lon, lat=lat, event=k))
    df.loc[rng.choice(n, 20, replace=False), "twi"] = np.nan           # a few missing values to test the cleaning
    df.to_csv(f"data/raw/demo_{k}.csv", index=False)
print("wrote fake demo files to data/raw/ (DO NOT use for results)")
