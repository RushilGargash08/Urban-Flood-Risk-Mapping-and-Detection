# urban-flood-risk-ai

AI-powered framework for site-specific urban flood risk prediction and flood mapping using geospatial and SAR-inspired data.

This project presents a site-specific, machine learning–based framework for predicting urban flood risk and generating flood vulnerability maps for undeveloped and emerging urban regions in India. The system integrates geospatial, environmental, and SAR-inspired features to estimate flood risk before construction, enabling data-driven and resilient urban planning.

# Urban Flood Risk Synthetic Dataset

This repository contains a **synthetic urban flood‑risk dataset** designed for experimenting with machine‑learning models (regression, classification, random forests, neural networks, etc.).

## What’s inside?
- `generate_flood_dataset.py` – Python script that creates `urban_flood_risk_dataset.csv`.
- `urban_flood_risk_dataset.csv` – 2 000 rows, each representing a synthetic observation (location) with:
  - Latitude & longitude (two geographic clusters):
    - **Region A** – outskirts of a city in **Punjab** (≈ Ludhiana area).
    - **Region B** – outskirts of **Delhi NCR** (≈ Gurgaon/Noida area).
  - Meteorological, topographic, soil, SAR‑inspired, and derived flood‑risk features.
  - `flood_risk_score` (0‑1) and categorical label (`Low`, `Medium`, `High`).

## How the data is generated
The script samples each feature uniformly within realistic ranges (e.g., rainfall 0‑300 mm / day, elevation 0‑500 m, etc.) and computes a flood‑risk score using a weighted linear model plus a non‑linear rain × built‑up interaction and random noise (±7 %).

Two distinct geographic regions are created by assigning latitude/longitude from separate bounding boxes:
- **Punjab outskirts** – lat ≈ 30.8‑31.2, lon ≈ 75.6‑76.0.
- **Delhi‑NCR outskirts** – lat ≈ 28.3‑28.7, lon ≈ 77.0‑77.5.

## Usage
```bash
# Clone the repo
git clone https://github.com/RushilGargash08/urban-flood-risk-ai.git
cd urban-flood-risk-ai

# (Re)generate the dataset
python generate_flood_dataset.py
```
The script will (re)create `urban_flood_risk_dataset.csv` in the repo root.

## Visualising the two regions
A quick example using `geopandas` and `matplotlib`:
```python
import pandas as pd, geopandas as gpd, matplotlib.pyplot as plt

df = pd.read_csv('urban_flood_risk_dataset.csv')
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'], df['lat']), crs='EPSG:4326')

region_a = gdf[gdf['lat'].between(30.8, 31.2)]
region_b = gdf[gdf['lat'].between(28.3, 28.7)]

fig, ax = plt.subplots(1, 2, figsize=(14,6))
region_a.plot(column='flood_risk_score', cmap='RdYlGn_r', legend=True, ax=ax[0])
ax[0].set_title('Punjab outskirts')
region_b.plot(column='flood_risk_score', cmap='RdYlGn_r', legend=True, ax=ax[1])
ax[1].set_title('Delhi NCR outskirts')
for a in ax: a.set_axis_off()
plt.show()
```

## License
This synthetic data is released under the MIT License – feel free to adapt, extend, or use it for research and demos.
