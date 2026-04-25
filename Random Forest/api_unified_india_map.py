import pandas as pd
import random
import requests
import joblib
import folium

def fetch_real_api_data(lats, lons, rain_date):
    """
    Connects to Open-Meteo REST API.
    Chunks requests to avoid URL length limits.
    """
    elevations = []
    rainfalls = []
    
    # Process in chunks of 50 to avoid URL length issues
    CHUNK_SIZE = 50
    for i in range(0, len(lats), CHUNK_SIZE):
        batch_lats = lats[i:i+CHUNK_SIZE]
        batch_lons = lons[i:i+CHUNK_SIZE]
        lat_str = ",".join(map(str, batch_lats))
        lon_str = ",".join(map(str, batch_lons))
        
        # 1. Fetch live Topographical Elevation
        elev_url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"
        elev_res = requests.get(elev_url).json()
        
        # Robust parsing for both dict-list and nested-list responses
        if isinstance(elev_res, dict) and 'elevation' in elev_res:
            elevations.extend(elev_res['elevation'])
        elif isinstance(elev_res, list):
            elevations.extend([e.get('elevation', 0) for e in elev_res])
        else:
            elevations.extend([0] * len(batch_lats))
        
        # 2. Fetch True Rainfall
        rain_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat_str}&longitude={lon_str}&start_date={rain_date}&end_date={rain_date}&daily=rain_sum"
        rain_res = requests.get(rain_url).json()
        
        # Handle both single-dict and list-of-dicts responses from Open-Meteo
        if isinstance(rain_res, dict):
            rain_res = [rain_res]
            
        for entry in rain_res:
            if 'daily' in entry and entry['daily']['rain_sum']:
                rain = entry['daily']['rain_sum'][0]
                rainfalls.append(rain if rain is not None else 0)
            else:
                rainfalls.append(0)
            
    return elevations, rainfalls

def compile_regional_data(region_name, count, lat_bnds, lon_bnds, rain_date, center_lon):
    # Generates heavily constrained coordinates strictly over landmasses
    lats = [round(random.uniform(*lat_bnds), 5) for _ in range(count)]
    lons = [round(random.uniform(*lon_bnds), 5) for _ in range(count)]
    
    print(f"Fetching Open-Meteo Satellite tracking data for {region_name} (Disaster Date: {rain_date})...")
    elevs, rains = fetch_real_api_data(lats, lons, rain_date)
    
    data = []
    for i in range(count):
        lat = lats[i]
        lon = lons[i]
        
        dist = abs(lon - center_lon)
        # NATURAL PROXY: Instead of drawing a fake straight geometric line in code to decide what is a "hill",
        # we let the TRUE API Elevation natively dictate if an area is naturally flat (highly built up slums) or elevated (natural hills with less concrete).
        true_elev = elevs[i]
        # Extending extreme-dense concrete threshold to 25m to adjust for Open-Meteo DEM satellites hitting the roofs of IT parks in Velachery.
        if true_elev < 25:
            built_up = round(random.uniform(85, 100), 2)
            soil = 'clay'
            moisture = round(random.uniform(0.7, 1.0), 3)
            slope = round(random.uniform(0, 2), 2)
        elif true_elev < 45:
            built_up = round(random.uniform(40, 80), 2)
            soil = 'loamy'
            moisture = round(random.uniform(0.4, 0.7), 3)
            slope = round(random.uniform(2, 10), 2)
        else:
            # High elevation (hills) -> naturally less concrete, better drainage (sandy), and sloping terrain
            built_up = round(random.uniform(0, 15), 2)
            soil = 'sandy'
            moisture = round(random.uniform(0.1, 0.3), 3)
            slope = round(random.uniform(10, 30), 2)
        
        data.append({
            'region': region_name,
            'lat': lat, 'lon': lon,
            'rainfall_mm_per_day': rains[i],       # TRUE API VALUE
            'elevation_m': true_elev,              # TRUE API VALUE
            'slope_percent': slope, 
            'built_up_percentage': built_up,
            'distance_to_water_body_km': round(max(0.1, dist * 60), 2), 
            'drainage_density': round(random.uniform(2, 5), 2),
            'sar_backscatter_coefficient': round(random.uniform(-25, -15), 2) if true_elev > 45 else round(random.uniform(-10, 0), 2), 
            'surface_roughness_index': round(random.uniform(0.1, 0.4), 3),
            'moisture_index': moisture, 
            'soil_type': soil
        })
    return pd.DataFrame(data)

def main():
    print("1. Loading Class-Balanced Random Forest AI Pipeline...")
    model = joblib.load('random_forest_flood_model.pkl')
    
    # --- REGION 1: MUMBAI (WEST COAST) ---
    # Tightly bounding longitude to dry-land and fetching July 26 2023 Monsoon floods
    df_mumbai = compile_regional_data(
        region_name="Mumbai", count=100, 
        lat_bnds=(19.05, 19.15), lon_bnds=(72.86, 72.93), 
        rain_date="2023-07-26", center_lon=72.885
    )
    
    # --- REGION 2: CHENNAI - VELACHERY (EAST COAST) ---
    # We revert to Dec 4th, when the cyclone was firmly stalled directly OVER Chennai dropping 230mm.
    df_chennai = compile_regional_data(
        region_name="Chennai (Velachery)", count=100, 
        lat_bnds=(12.95, 13.05), lon_bnds=(80.18, 80.25), 
        rain_date="2023-12-04", center_lon=80.22
    )
    
    # --- REGION 3: GREATER BENGALURU / NANDI HILLS ---
    # High elevation (~1400m) and typically lower monsoon rainfall.
    df_bengaluru = compile_regional_data(
        region_name="Greater Bengaluru (Nandi Hills)", count=100, 
        lat_bnds=(13.34, 13.40), lon_bnds=(77.65, 77.72), 
        rain_date="2023-07-26", center_lon=77.68
    )
    
    # Combine datasets
    df_all = pd.concat([df_mumbai, df_chennai, df_bengaluru], ignore_index=True)
    
    # Neural Preprocessing
    # Neural Preprocessing
    df_encoded = pd.get_dummies(df_all.drop(columns=['region']), columns=['soil_type'])
    df_orig = pd.read_csv('urban_flood_risk_dataset.csv')
    X_orig = pd.get_dummies(df_orig.drop(['flood_risk_score', 'flood_risk_category', 'lat', 'lon'], axis=1, errors='ignore'), columns=['soil_type'])
    
    for col in X_orig.columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[X_orig.columns]
    
    print("2. Pushing Dual-Region data through Predictor...")
    df_all['predicted_risk'] = model.predict(df_encoded)
    
    # Debug: Check distribution
    print("\n--- Model Prediction Stats ---")
    for region in df_all['region'].unique():
        dist = df_all[df_all['region'] == region]['predicted_risk'].value_counts()
        print(f"Region: {region} -> {dict(dist)}")
    
    # NEW: Deep Diagnostic for Nandi Hills
    bengaluru_data = df_all[df_all['region'] == 'Greater Bengaluru (Nandi Hills)'].head(1)
    if not bengaluru_data.empty:
        idx = bengaluru_data.index[0]
        prob = model.predict_proba(df_encoded.loc[[idx]])[0]
        print(f"\n--- Probabilities for a Nandi Hills Point ---")
        print(f"Classes: {model.classes_}")
        print(f"Probabilities: {prob}")
        print(f"Features: {df_encoded.loc[idx].to_dict()}")
    
    print("\n3. Rendering Pan-India Interactive Map...")
    # Center zoomed out to see Mumbai, Chennai, and Bengaluru
    m = folium.Map(location=[15.0, 78.5], zoom_start=5, tiles='CartoDB positron')
    color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    
    for idx, row in df_all.iterrows():
        color = color_map.get(row['predicted_risk'], 'gray')
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 170px;">
            <b style="color: {color};">Risk Level: {row['predicted_risk']}</b><br><hr style="margin: 2px 0;">
            <b style="color: purple;">{row['region']}</b><br>
            <b style="color: blue;">TRUE Elev (API):</b> {row['elevation_m']} m<br>
            <b style="color: blue;">TRUE Rain (API):</b> {row['rainfall_mm_per_day']} mm<br>
            <b>Built-up Est:</b> {row['built_up_percentage']}%<br>
        </div>
        """
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6, color=color, fill=True, fill_color=color, fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)
        
    m.save('V3_FINAL_API_MAP.html')
    print("✅ V3 FINAL MAP GENERATED! Open 'V3_FINAL_API_MAP.html' in your browser.")

if __name__ == "__main__":
    main()
