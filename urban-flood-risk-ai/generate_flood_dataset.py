import csv, random, math

random.seed(42)

NUM_ROWS = 2000

def soil_type_val():
    # probabilities: sandy 0.4, loamy 0.3, clay 0.3
    r = random.random()
    if r < 0.4:
        return 'sandy'
    elif r < 0.7:
        return 'loamy'
    else:
        return 'clay'

def compute_score(row):
    # base linear contributions (weights sum ~1)
    score = 0.0
    score += (row['rainfall_mm_per_day'] / 300) * 0.30
    score += (row['built_up_percentage'] / 100) * 0.20
    score += row['moisture_index'] * 0.20
    score += ((500 - row['elevation_m']) / 500) * 0.10
    score += ((30 - row['slope_percent']) / 30) * 0.05
    score += (1.0 if row['soil_type'] == 'clay' else 0.0) * 0.05
    score += ((row['sar_backscatter_coefficient'] + 25) / 25) * 0.05
    score -= (row['drainage_density'] / 5) * 0.05
    score -= (row['distance_to_water_body_km'] / 10) * 0.05
    # non‑linear interaction: high rain + high built‑up
    if row['rainfall_mm_per_day'] > 200 and row['built_up_percentage'] > 70:
        score += 0.10
    # add realistic noise (5‑10%)
    noise = random.uniform(-0.07, 0.07)
    score += noise
    # clip to [0,1]
    return max(0.0, min(1.0, score))

with open('urban_flood_risk_dataset.csv', 'w', newline='') as csvfile:
    fieldnames = [
        'lat', 'lon',
        'rainfall_mm_per_day', 'elevation_m', 'slope_percent', 'soil_type',
        'built_up_percentage', 'distance_to_water_body_km', 'drainage_density',
        'sar_backscatter_coefficient', 'surface_roughness_index', 'moisture_index',
        'flood_risk_score', 'flood_risk_category'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(NUM_ROWS):
        row = {}
        # Split into two geographic regions: first half Region A (Punjab outskirts), second half Region B (Delhi NCR outskirts)
        if i < NUM_ROWS // 2:
            # Region A bounds (Punjab outskirts, e.g., Ludhiana area)
            lat_min, lat_max = 30.8, 31.2
            lon_min, lon_max = 75.6, 76.0
        else:
            # Region B bounds (Delhi NCR outskirts, e.g., Gurgaon/Noida area)
            lat_min, lat_max = 28.3, 28.7
            lon_min, lon_max = 77.0, 77.5
        row['lat'] = round(random.uniform(lat_min, lat_max), 6)
        row['lon'] = round(random.uniform(lon_min, lon_max), 6)
        row['rainfall_mm_per_day'] = round(random.uniform(0, 300), 2)
        row['elevation_m'] = round(random.uniform(0, 500), 2)
        row['slope_percent'] = round(random.uniform(0, 30), 2)
        row['soil_type'] = soil_type_val()
        row['built_up_percentage'] = round(random.uniform(0, 100), 2)
        row['distance_to_water_body_km'] = round(random.uniform(0, 10), 2)
        row['drainage_density'] = round(random.uniform(0, 5), 2)
        row['sar_backscatter_coefficient'] = round(random.uniform(-25, 0), 2)
        row['surface_roughness_index'] = round(random.uniform(0, 1), 3)
        row['moisture_index'] = round(random.uniform(0, 1), 3)
        row['flood_risk_score'] = round(compute_score(row), 3)
        if row['flood_risk_score'] <= 0.33:
            row['flood_risk_category'] = 'Low'
        elif row['flood_risk_score'] <= 0.66:
            row['flood_risk_category'] = 'Medium'
        else:
            row['flood_risk_category'] = 'High'
        writer.writerow(row)
print('Dataset generated with lat/lon for two regions: urban_flood_risk_dataset.csv')
