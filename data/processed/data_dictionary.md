# Data dictionary
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
