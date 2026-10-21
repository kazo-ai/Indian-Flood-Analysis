from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "flood_dataset.csv"
OUT = ROOT / "outputs" / "interpretability"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

features = [
    "elevation", "slope", "curvature", "twi", "hand", "dist_river",
    "rain_event_total", "rain_max_1day", "rain_max_3day",
    "rain_antecedent_30d", "ndvi_pre", "clay", "sand",
    "lc_bare", "lc_built", "lc_crop", "lc_grass",
    "lc_shrub", "lc_tree", "lc_water", "lc_wetland",
]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df[features] = df[features].replace([np.inf, -np.inf], np.nan)

# Event-level feature distributions and missingness.
records = []
for event, group in df.groupby("event"):
    for feature in features:
        values = group[feature]
        records.append({
            "event": event,
            "feature": feature,
            "count": int(values.notna().sum()),
            "missing_percent": float(values.isna().mean() * 100),
            "mean": float(values.mean()) if values.notna().any() else np.nan,
            "std": float(values.std()) if values.notna().any() else np.nan,
            "median": float(values.median()) if values.notna().any() else np.nan,
            "q05": float(values.quantile(0.05)) if values.notna().any() else np.nan,
            "q95": float(values.quantile(0.95)) if values.notna().any() else np.nan,
        })

summary = pd.DataFrame(records)
summary.to_csv(OUT / "event_feature_distributions.csv", index=False)

# Label balance and feature missingness by class within each event.
label_records = []
for (event, label), group in df.groupby(["event", "flooded"]):
    row = {
        "event": event,
        "flooded": int(label),
        "rows": len(group),
        "label_fraction_within_event": float(
            (df.loc[df["event"] == event, "flooded"] == label).mean()
        ),
    }
    for feature in features:
        row[f"{feature}_missing_percent"] = float(
            group[feature].isna().mean() * 100
        )
    label_records.append(row)

label_summary = pd.DataFrame(label_records)
label_summary.to_csv(OUT / "event_label_missingness.csv", index=False)

print("\nRows and label balance by event:")
print(
    df.groupby("event")["flooded"]
    .agg(rows="size", positives="sum", positive_rate="mean")
    .to_string()
)

print("\nMissingness by event (%):")
missing = df.groupby("event")[features].apply(
    lambda x: x.isna().mean() * 100
)
print(missing.round(2).to_string())

print("\nSaved:")
print(OUT / "event_feature_distributions.csv")
print(OUT / "event_label_missingness.csv")
