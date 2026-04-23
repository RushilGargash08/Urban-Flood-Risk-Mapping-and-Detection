# urban-flood-risk-ai

AI-powered framework for site-specific urban flood risk prediction and flood mapping using geospatial and SAR-inspired data.

This project presents a site-specific, machine learning–based framework for predicting urban flood risk and generating flood vulnerability maps for undeveloped and emerging urban regions in India. The system integrates geospatial, environmental, and SAR-inspired features to estimate flood risk before construction, enabling data-driven and resilient urban planning.

## Urban Flood Risk Predictor

This repository contains an advanced GridSearch optimized **Random Forest Model** explicitly trained to simulate and predict geographical flood damage. Features include:
- Historical REST API fetching from Open-Meteo for authentic disaster data.
- Live Pan-India Folium mapping for visualization of flood vulnerability.

## What’s inside?
- `Random Forest/train_rf_model.py` – Advanced Random Forest Classifier trained on GridSearch configurations.
- `Random Forest/generate_flood_dataset.py` – Python script that creates the base dataset.
- `Random Forest/api_unified_india_map.py` – Code to generate Live Pan-India Interactive Heatmaps.
- `.pkl` models and `.html` outputs representing 91% accuracy across severe flood benchmarks.

## Visualizing the Framework
You can interact geographically with the finished models by opening any of the mapped `.html` files (e.g. `api_unified_india_map.html`) to physically view how AI predictions unfold geographically across Chennai and Mumbai over historically authentic Disaster Dates!

## License
MIT License – feel free to adapt, extend, or use this architecture.
