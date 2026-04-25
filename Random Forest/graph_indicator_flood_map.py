import pandas as pd
import random
import requests
import joblib
import folium
from folium.plugins import HeatMap

def fetch_real_api_data(lats, lons, rain_date):
    elevations = []
    rainfalls = []
    CHUNK_SIZE = 50
    for i in range(0, len(lats), CHUNK_SIZE):
        batch_lats = lats[i:i+CHUNK_SIZE]
        batch_lons = lons[i:i+CHUNK_SIZE]
        lat_str = ",".join(map(str, batch_lats))
        lon_str = ",".join(map(str, batch_lons))
        
        elev_url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"
        elev_res = requests.get(elev_url).json()
        if isinstance(elev_res, dict) and 'elevation' in elev_res:
            elevations.extend(elev_res['elevation'])
        elif isinstance(elev_res, list):
            elevations.extend([e.get('elevation', 0) for e in elev_res])
        else:
            elevations.extend([0] * len(batch_lats))
        
        rain_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat_str}&longitude={lon_str}&start_date={rain_date}&end_date={rain_date}&daily=rain_sum"
        rain_res = requests.get(rain_url).json()
        if isinstance(rain_res, dict): rain_res = [rain_res]
        for entry in rain_res:
            rain = entry.get('daily', {}).get('rain_sum', [0])[0]
            rainfalls.append(rain if rain is not None else 0)
    return elevations, rainfalls

def compile_regional_data(region_name, count, lat_bnds, lon_bnds, rain_date):
    lats = [round(random.uniform(*lat_bnds), 5) for _ in range(count)]
    lons = [round(random.uniform(*lon_bnds), 5) for _ in range(count)]
    elevs, rains = fetch_real_api_data(lats, lons, rain_date)
    data = []
    for i in range(count):
        true_elev = elevs[i]
        if true_elev < 25:
            built_up, soil, moisture, slope = round(random.uniform(85, 100), 2), 'clay', round(random.uniform(0.7, 1.0), 3), round(random.uniform(0, 2), 2)
        elif true_elev < 45:
            built_up, soil, moisture, slope = round(random.uniform(40, 80), 2), 'loamy', round(random.uniform(0.4, 0.7), 3), round(random.uniform(2, 10), 2)
        else:
            built_up, soil, moisture, slope = round(random.uniform(0, 15), 2), 'sandy', round(random.uniform(0.1, 0.3), 3), round(random.uniform(10, 30), 2)
        data.append({
            'lat': lats[i], 'lon': lons[i], 'rainfall_mm_per_day': rains[i], 'elevation_m': true_elev,
            'slope_percent': slope, 'built_up_percentage': built_up, 'distance_to_water_body_km': 0.5,
            'drainage_density': 3.0, 'sar_backscatter_coefficient': -20 if true_elev > 45 else -5,
            'surface_roughness_index': 0.2, 'moisture_index': moisture, 'soil_type': soil
        })
    return pd.DataFrame(data)

def main():
    print("🚀 Generating Graphical Flood Extent Model (Heatmap Edition)...")
    model = joblib.load('random_forest_flood_model.pkl')
    df_orig = pd.read_csv('urban_flood_risk_dataset.csv')
    X_orig = pd.get_dummies(df_orig.drop(['flood_risk_score', 'flood_risk_category', 'lat', 'lon'], axis=1, errors='ignore'), columns=['soil_type'])

    # --- REGION 1: MUMBAI ---
    df_mumbai = compile_regional_data("Mumbai", 150, (18.90, 19.25), (72.80, 72.95), "2023-07-26")
    # --- REGION 2: CHENNAI ---
    df_chennai = compile_regional_data("Chennai", 150, (12.95, 13.05), (80.18, 80.25), "2023-12-04")
    # --- REGION 3: BENGALURU ---
    df_bengaluru = compile_regional_data("Greater Bengaluru", 100, (13.34, 13.40), (77.65, 77.72), "2023-07-26")
    
    df_all = pd.concat([df_mumbai, df_chennai, df_bengaluru], ignore_index=True)
    df_encoded = pd.get_dummies(df_all, columns=['soil_type'])
    for col in X_orig.columns:
        if col not in df_encoded.columns: df_encoded[col] = 0
    df_encoded = df_encoded[X_orig.columns]
    
    df_all['risk'] = model.predict(df_encoded)
    
    # Convert Categorical labels to Heatmap Intensity weights
    # High Risk = Red (1.0), Medium = Yellow (0.6), Low = Green (0.01) - Very low weight to prevent false stacking
    risk_weights = {'High': 1.0, 'Medium': 0.6, 'Low': 0.01}
    heat_data = [[row['lat'], row['lon'], risk_weights[row['risk']]] for idx, row in df_all.iterrows()]

    # Create Map with Satellite-style terrain visuals, zoomed out to see all cities
    m = folium.Map(location=[15.0, 78.5], zoom_start=5, tiles='CartoDB positron')
    
    # We set max_val=1.0 to prevent clustered points from "stacking" into a different color
    HeatMap(heat_data, radius=20, blur=12, min_opacity=0.4, max_val=1.0,
            gradient={0.01: 'lime', 0.6: 'yellow', 1.0: 'red'}).add_to(m)

    m.save('GRAPHICAL_FLOOD_EXTENT_MAP.html')
    print("✅ PAN-INDIA GRAPHICAL SUCCESS! Open 'GRAPHICAL_FLOOD_EXTENT_MAP.html'.")

if __name__ == "__main__":
    main()
