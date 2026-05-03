# Urban Flood Mapping 

**AI-powered framework for site-specific urban flood risk prediction and interactive mapping using multi-algorithmic ensembles.**

This project presents a state-of-the-art, machine learning–based framework for predicting urban flood risk and generating high-resolution vulnerability maps for emerging urban regions in India. The system integrates real-time geospatial, topographical, and meteorological data to enable proactive and resilient urban planning.

## Key Features
- **Unified Ensemble Architecture:** Blends **MLP Neural Networks**, **Random Forest**, and **Ridge Regression** via a soft-voting mechanism to achieve a **94.25% predictive accuracy**.
- **Real-Time API Validation:** Integrated with the **Open-Meteo REST API** to validate predictions against historical "Temporal Peak" conditions (e.g., 2023 Monsoon and Cyclone seasons).
- **Interactive Spatial Mapping:** Generates high-fidelity **Heatmaps** and **Point-Risk Maps** using Folium, allowing for physical visualization of risk across diverse micro-climates.
- **SAR-Integrated Predictive Engine:** Incorporates **Calibrated SAR-Backscatter Response Proxies** (Sentinel-1 VV-band characteristics) to enhance flood vulnerability detection in urban vs. rural terrains.
- **Topographical Physics Engine:** The model prioritizes terrain physics (Elevation, Slope, Soil Type) over uniform meteorological inputs, ensuring accurate risk assessment in high-altitude vs. low-lying regions.

## Project Structure
- **`MLP/`**: Multi-Layer Perceptron (Neural Network) architecture optimized for capturing deep latent correlations between environmental proxies.
- **`Random Forest/`**: GridSearch-optimized ensemble of 400 deep decision trees, capturing localized feature interactions.
- **`ridge_regression/`**: Linear baseline models used to ensure predictive stability and mathematical anchoring.
- **`results/`**: Final interactive `.html` outputs, including regional stress-tests for Mumbai, Chennai (Velachery), and Greater Bengaluru (Nandi Hills).
- **`Findings_and_Results_Draft.md`**: Detailed analytical report on model performance and spatial validation findings.

## 📊 Performance Benchmarks
| Model | Accuracy | ROC-AUC |
| :--- | :--- | :--- |
| **Unified Ensemble** | **94.25%** | **0.992** |
| MLP Neural Network | 93.50% | 0.991 |
| Random Forest | 91.00% | 0.984 |
| Ridge Regression | 89.25% | 0.862 |

## Visualizing the Framework
You can interact with the predictive results by opening the files in the `results/` directory:
- `V3_FINAL_API_MAP.html`: Comprehensive API-validated map across Mumbai, Chennai, and Bengaluru.
- `GRAPHICAL_FLOOD_EXTENT_MAP.html`: Large-scale simulation (1,600 points) demonstrating model scalability.

## Tech Stack
- **Python:** Core processing and modeling.
- **Scikit-Learn & Joblib:** Algorithmic development and model persistence.
- **Folium & Leaflet.js:** Geospatial visualization.
- **Open-Meteo API:** Authentic historical and real-time environmental data fetching.

## License
MIT License – Created by Rushil Gargash. Feel free to adapt, extend, or use this architecture for resilient infrastructure research.
