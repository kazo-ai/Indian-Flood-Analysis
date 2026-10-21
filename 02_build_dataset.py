"""
Step 2. Read the raw CSVs from Earth Engine, clean them, and write one tidy dataset
plus a data dictionary and a few quality-check figures.

Run:  python 02_build_dataset.py
"""
import glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

RANGES = {"slope": (0, 90), "hand": (0, None), "dist_river": (0, None), "rain_event_total": (0, None),
          "rain_max_1day": (0, None), "rain_max_3day": (0, None), "rain_antecedent_30d": (0, None),
          "ndvi_pre": (-1, 1), "clay": (0, 100), "sand": (0, 100)}

files = sorted(
    f"data/raw/{event}.csv"
    for event in [
        "assam_2020",
        "assam_2022",
        "bihar_2020",
        "kerala_2018",
    ]
)
if not files:
    raise SystemExit("No CSV files in data/raw/. Download them from Google Drive (or run make_demo_data.py).")
report = []
df = pd.concat([pd.read_csv(f).assign(_file=os.path.basename(f)) for f in files], ignore_index=True)
report.append(f"Raw rows: {len(df)} from {len(files)} files")
need = C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES + [C.LABEL_COL] + C.META_COLS
missing_cols = [c for c in need if c not in df.columns]
if missing_cols:
    raise SystemExit(f"Missing columns in the CSV files: {missing_cols}")

df = df.replace([np.inf, -np.inf], np.nan)
na = df[need].isna().sum()
report.append("Missing values per column (before cleaning):\n" + na[na > 0].to_string() if na.sum() else "No missing values")
n0 = len(df); df = df.dropna(subset=need); report.append(f"Dropped rows with missing values: {n0 - len(df)}")

n0 = len(df)
for col, (lo, hi) in RANGES.items():
    if lo is not None: df = df[df[col] >= lo]
    if hi is not None: df = df[df[col] <= hi]
report.append(f"Dropped rows with impossible values: {n0 - len(df)}")

n0 = len(df); df = df.drop_duplicates(subset=["event", "lon", "lat"]); report.append(f"Dropped duplicate points: {n0 - len(df)}")
df[C.LABEL_COL] = df[C.LABEL_COL].astype(int)
df["landcover"] = df["landcover"].astype(int)
df["lc_name"] = df["landcover"].map(C.WORLDCOVER).fillna("other")
df = pd.concat([df, pd.get_dummies(df["lc_name"], prefix="lc").astype(int)], axis=1)
df["block"] = (np.floor(df["lon"] / C.BLOCK_DEG).astype(int).astype(str) + "_" +
               np.floor(df["lat"] / C.BLOCK_DEG).astype(int).astype(str))

tab = df.groupby("event")[C.LABEL_COL].agg(n="count", flooded="sum")
tab["flooded_share"] = (tab["flooded"] / tab["n"]).round(3)
report.append("Rows per event:\n" + tab.to_string())
bad = tab[(tab["flooded"] < 200) | (tab["n"] - tab["flooded"] < 200)].index.tolist()
if bad:
    report.append(f"WARNING: events with fewer than 200 rows in a class were removed: {bad}")
    df = df[~df["event"].isin(bad)]

df = df.drop(columns=["_file"])
df.to_csv("data/processed/flood_dataset.csv", index=False)
report.append(f"Final rows: {len(df)}, events: {df['event'].nunique()}, spatial blocks: {df['block'].nunique()}")
report.append("NOTE: rainfall (CHIRPS ~5 km) is almost constant inside an event cell, so use spatial/event CV, never random CV.")

# ---- data dictionary
dd = """# Data dictionary
| column | meaning | source |
|---|---|---|
| elevation | height above sea level (m) | SRTM 30 m |
| slope | terrain slope (degrees) | SRTM 30 m |
| curvature | Laplacian curvature (+ valley, - ridge, rough) | SRTM 30 m |
| twi | topographic wetness index, higher = water collects | MERIT Hydro + SRTM |
| hand | height above nearest drainage (m) | MERIT Hydro |
| dist_river | distance to a stream with upstream catchment area >=10 km² (m, capped at 5 km) | MERIT Hydro |
| rain_event_total | rain over the flood window (mm) | CHIRPS |
| rain_max_1day | wettest single day in the flood window (mm) | CHIRPS |
| rain_max_3day | wettest 3 days in the flood window (mm) | CHIRPS |
| rain_antecedent_30d | rain in the 30 days before the flood window (mm) | CHIRPS |
| ndvi_pre | vegetation greenness in the 60 days before the flood | MODIS MOD13Q1 |
| clay, sand | soil clay and sand content at the surface (%) | OpenLandMap |
| landcover / lc_* | ESA WorldCover class / one-hot columns | ESA WorldCover 2021 |
| flooded | 1 = new open water in flood window (radar), 0 = not flooded | Sentinel-1 (weak label) |
| lon, lat, event, block | location, flood event, 0.25 degree spatial block (for CV) | - |
"""
open("data/processed/data_dictionary.md", "w").write(dd)
open("data/processed/cleaning_report.txt", "w").write("\n\n".join(report))
print("\n\n".join(report))

# ---- quality-check figures
os.makedirs("outputs", exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
tab[["n", "flooded"]].plot.bar(ax=ax[0]); ax[0].set_title("Rows and flooded rows per event")
cm = df[C.NUMERIC_FEATURES + [C.LABEL_COL]].corr()
im = ax[1].imshow(cm, cmap="coolwarm", vmin=-1, vmax=1); ax[1].set_title("Correlation")
ax[1].set_xticks(range(len(cm))); ax[1].set_xticklabels(cm.columns, rotation=90, fontsize=7)
ax[1].set_yticks(range(len(cm))); ax[1].set_yticklabels(cm.columns, fontsize=7); plt.colorbar(im, ax=ax[1], fraction=.046)
for v, c in [(0, "tab:blue"), (1, "tab:red")]:
    ax[2].hist(df.loc[df[C.LABEL_COL] == v, "hand"], bins=40, alpha=.55, color=c, label=["not flooded", "flooded"][v], density=True)
ax[2].set_xlabel("HAND (m)"); ax[2].legend(); ax[2].set_title("HAND: flooded vs not flooded")
plt.tight_layout(); plt.savefig("outputs/data_quality.png", dpi=150); plt.close()
print("\nSaved: data/processed/flood_dataset.csv, data_dictionary.md, cleaning_report.txt, outputs/data_quality.png")
