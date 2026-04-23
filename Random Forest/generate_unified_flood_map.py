import pandas as pd
import random
import joblib
import folium

def generate_data(num_points, lat_range, lon_range, logic_fn):
    data = []
    for _ in range(num_points):
        lat = round(random.uniform(*lat_range), 6)
        lon = round(random.uniform(*lon_range), 6)
        row_data = logic_fn(lat, lon)
        row_data['lat'] = lat
        row_data['lon'] = lon
        data.append(row_data)
    return pd.DataFrame(data)

# --- SPATIALLY COHERENT LOGIC FUNCTIONS FOR ALL 4 REGIONS ---

def logic_dhemaji(lat, lon):
    dist = abs(lat - 27.48)
    if dist < 0.02:
        return {'elevation_m': round(random.uniform(90, 100), 2), 'slope_percent': round(random.uniform(0, 2), 2), 'distance_to_water_body_km': round(dist * 50, 2), 'rainfall_mm_per_day': round(random.uniform(250, 300), 2), 'built_up_percentage': round(random.uniform(50, 75), 2), 'drainage_density': round(random.uniform(3, 5), 2), 'sar_backscatter_coefficient': round(random.uniform(-10, 0), 2), 'surface_roughness_index': round(random.uniform(0.1, 0.4), 3), 'moisture_index': round(random.uniform(0.8, 1.0), 3), 'soil_type': random.choice(['loamy', 'sandy'])}
    else:
        return {'elevation_m': round(100 + (dist * 2000), 2), 'slope_percent': round(random.uniform(5, 15), 2), 'distance_to_water_body_km': round(dist * 50, 2), 'rainfall_mm_per_day': round(random.uniform(100, 200), 2), 'built_up_percentage': round(random.uniform(0, 5), 2), 'drainage_density': round(random.uniform(1, 3), 2), 'sar_backscatter_coefficient': round(random.uniform(-20, -10), 2), 'surface_roughness_index': round(random.uniform(0.2, 0.6), 3), 'moisture_index': round(random.uniform(0.4, 0.6), 3), 'soil_type': random.choice(['loamy', 'sandy'])}

def logic_kolhapur(lat, lon):
    dist = abs(lat - 16.70)
    if dist < 0.025:
        return {'elevation_m': round(random.uniform(530, 540), 2), 'slope_percent': round(random.uniform(0, 3), 2), 'distance_to_water_body_km': round(dist * 40, 2), 'rainfall_mm_per_day': round(random.uniform(200, 300), 2), 'built_up_percentage': round(random.uniform(40, 75), 2), 'drainage_density': round(random.uniform(3, 5), 2), 'sar_backscatter_coefficient': round(random.uniform(-10, 0), 2), 'surface_roughness_index': round(random.uniform(0.1, 0.3), 3), 'moisture_index': round(random.uniform(0.7, 1.0), 3), 'soil_type': 'clay'}
    else:
        return {'elevation_m': round(540 + (dist * 1000), 2), 'slope_percent': round(random.uniform(5, 20), 2), 'distance_to_water_body_km': round(dist * 40, 2), 'rainfall_mm_per_day': round(random.uniform(100, 150), 2), 'built_up_percentage': round(random.uniform(0, 10), 2), 'drainage_density': round(random.uniform(1, 3), 2), 'sar_backscatter_coefficient': round(random.uniform(-20, -10), 2), 'surface_roughness_index': round(random.uniform(0.3, 0.6), 3), 'moisture_index': round(random.uniform(0.3, 0.6), 3), 'soil_type': 'clay'}

def logic_mumbai(lat, lon):
    dist = abs(lon - 72.885)
    is_hill = (lat > 19.13 and lon < 72.875)
    if not is_hill and dist < 0.015:
        return {'elevation_m': round(random.uniform(5, 15), 2), 'slope_percent': round(random.uniform(0, 2), 2), 'distance_to_water_body_km': round(dist * 60, 2), 'rainfall_mm_per_day': round(random.uniform(250, 300), 2), 'built_up_percentage': round(random.uniform(85, 100), 2), 'drainage_density': round(random.uniform(2, 5), 2), 'sar_backscatter_coefficient': round(random.uniform(-5, 5), 2), 'surface_roughness_index': round(random.uniform(0.0, 0.2), 3), 'moisture_index': round(random.uniform(0.8, 1.0), 3), 'soil_type': 'clay'}
    else:
        return {'elevation_m': round(random.uniform(100, 250), 2), 'slope_percent': round(random.uniform(10, 30), 2), 'distance_to_water_body_km': round(max(1.0, dist * 60), 2), 'rainfall_mm_per_day': round(random.uniform(50, 150), 2), 'built_up_percentage': round(random.uniform(0, 5), 2), 'drainage_density': round(random.uniform(1, 3), 2), 'sar_backscatter_coefficient': round(random.uniform(-20, -10), 2), 'surface_roughness_index': round(random.uniform(0.4, 0.8), 3), 'moisture_index': round(random.uniform(0.2, 0.4), 3), 'soil_type': 'clay'}

def logic_chiplun(lat, lon):
    dist = abs(lat - 17.53)
    if dist < 0.012:
        return {'elevation_m': round(random.uniform(5, 15), 2), 'slope_percent': round(random.uniform(0, 2), 2), 'distance_to_water_body_km': round(dist * 50, 2), 'rainfall_mm_per_day': round(random.uniform(250, 300), 2), 'built_up_percentage': round(random.uniform(50, 80), 2), 'drainage_density': round(random.uniform(2, 4), 2), 'sar_backscatter_coefficient': round(random.uniform(-5, 0), 2), 'surface_roughness_index': round(random.uniform(0.2, 0.6), 3), 'moisture_index': round(random.uniform(0.8, 1.0), 3), 'soil_type': 'clay'}
    else:
        return {'elevation_m': round(150 + (dist * 10000), 2), 'slope_percent': round(random.uniform(15, 30), 2), 'distance_to_water_body_km': round(dist * 50, 2), 'rainfall_mm_per_day': round(random.uniform(50, 100), 2), 'built_up_percentage': round(random.uniform(0, 5), 2), 'drainage_density': round(random.uniform(2, 4), 2), 'sar_backscatter_coefficient': round(random.uniform(-20, -15), 2), 'surface_roughness_index': round(random.uniform(0.2, 0.6), 3), 'moisture_index': round(random.uniform(0.3, 0.5), 3), 'soil_type': 'clay'}

def main():
    print("1. Loading the trained Random Forest model...")
    model = joblib.load('random_forest_flood_model.pkl')
    
    print("2. Generating and concatenating spatial data for all 4 distinct regions...")
    # Generate 400 data points per region = 1600 Total Points
    df_dhemaji = generate_data(400, (27.44, 27.52), (94.50, 94.60), logic_dhemaji)
    df_dhemaji['region'] = 'Dhemaji'
    
    df_kolhapur = generate_data(400, (16.65, 16.75), (74.30, 74.45), logic_kolhapur)
    df_kolhapur['region'] = 'Kolhapur'
    
    df_mumbai = generate_data(400, (19.10, 19.15), (72.85, 72.90), logic_mumbai)
    df_mumbai['region'] = 'Mumbai'
    
    df_chiplun = generate_data(400, (17.51, 17.55), (73.50, 73.54), logic_chiplun)
    df_chiplun['region'] = 'Chiplun'
    
    # Merge all rows into a single master Pandas array
    df_all = pd.concat([df_dhemaji, df_kolhapur, df_mumbai, df_chiplun], ignore_index=True)
    
    # 3. Mass Preprocessing matches
    df_encoded = pd.get_dummies(df_all.drop(columns=['region']), columns=['soil_type'])
    df_original = pd.read_csv('urban_flood_risk_dataset.csv')
    X_original = pd.get_dummies(df_original.drop(['flood_risk_score', 'flood_risk_category'], axis=1), columns=['soil_type'])
    
    for col in X_original.columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_encoded = df_encoded[X_original.columns]
    
    # 4. Mass Prediction
    print("3. Predicting unified risk profiles across all 1600 Pan-India points...")
    predictions = model.predict(df_encoded)
    df_all['predicted_risk'] = predictions
    
    # 5. Draw it all onto a single Pan-India view completely zoomed out
    print("4. Drawing the unified Interactive Map...")
    m = folium.Map(location=[21.0, 79.0], zoom_start=5, tiles='CartoDB positron')
    color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    
    for idx, row in df_all.iterrows():
        risk_level = row['predicted_risk']
        color = color_map.get(risk_level, 'gray')
        
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 170px;">
            <b style="color: {color};">Risk Level: {risk_level}</b><br><hr style="margin: 2px 0;">
            <b>Region:</b> {row['region']}<br>
            <b>Elevation:</b> {row['elevation_m']} m<br>
            <b>Built-up:</b> {row['built_up_percentage']}%<br>
            <b>Rainfall:</b> {row['rainfall_mm_per_day']} mm
        </div>
        """
        
        # We shrink the markers super tiny since we are plotting 1600 points from a distance
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(m)
        
    unified_map_name = 'unified_india_flood_map.html'
    m.save(unified_map_name)
    print(f"✅ Success! Your All-India unified map has been saved as '{unified_map_name}'")

if __name__ == "__main__":
    main()
