import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

R = 6378137.0 
def add_meters_to_latlon(lat, lon, dx, dy):
    d_lat = dy / R
    d_lon = dx / (R * np.cos(np.pi * lat / 180.0))
    new_lat = lat + (d_lat * 180.0 / np.pi)
    new_lon = lon + (d_lon * 180.0 / np.pi)
    return new_lat, new_lon

def train_and_predict():
    print("Loading training data...")
    # Train on S1 and S2
    train_dfs = []
    for sid in ["S1", "S2"]:
        df = pd.read_csv(f"data/{sid}_features.csv")
        # Only train on non-blackout data where GPS speed is valid
        train_dfs.append(df[~df['blackout']])
        
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    feature_cols = ['accel_y_mean', 'accel_y_std', 'accel_z_std', 'gyro_z_std', 'accel_energy']
    
    X_train = train_df[feature_cols].values
    y_train = train_df['gps_speed'].values
    
    print("Training Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Optional: Save weights logic here later for TS. For now, generate the tracks!
    
    for sid in ["S1", "S2", "S3a"]:
        print(f"Generating AI-Fused track for {sid}...")
        df = pd.read_csv(f"data/{sid}_features.csv")
        df_raw = pd.read_csv(f"data/{sid}.csv") # need raw gyro_z
        
        X_test = df[feature_cols].values
        predicted_speeds = model.predict(X_test)
        
        track = []
        current_lat = df['lat'].iloc[0]
        current_lon = df['lon'].iloc[0]
        current_heading = df['gps_heading'].iloc[0] if not pd.isna(df['gps_heading'].iloc[0]) else 0
        
        prev_time = df['time_ms'].iloc[0] / 1000.0
        
        for i in range(len(df)):
            row = df.iloc[i]
            raw_row = df_raw.iloc[i]
            
            curr_time = row['time_ms'] / 1000.0
            dt = curr_time - prev_time
            if dt > 1.0: dt = 0.1
            if dt == 0: dt = 0.001
            prev_time = curr_time
            
            if not row['blackout']:
                current_lat = row['lat']
                current_lon = row['lon']
                current_heading = row['gps_heading'] if not pd.isna(row['gps_heading']) else current_heading
                speed = row['gps_speed']
            else:
                # AI FUSION!
                # Use ML predicted speed instead of noisy accelerometer integration
                speed = predicted_speeds[i]
                
                # Update heading using gyroscope (simple integration)
                gyro_z = raw_row['gyro_z'] if not pd.isna(raw_row['gyro_z']) else 0
                current_heading += np.degrees(gyro_z * dt)
                
                heading_rad = np.radians(current_heading)
                dx = speed * np.sin(heading_rad) * dt
                dy = speed * np.cos(heading_rad) * dt
                
                current_lat, current_lon = add_meters_to_latlon(current_lat, current_lon, dx, dy)
                
            track.append({
                "ts": curr_time,
                "lat": current_lat,
                "lon": current_lon,
                "speed_kmh": speed * 3.6,
                "heading": current_heading
            })
            
        with open(f"../public/data/segment_{sid}_ai_fused.json", "w") as f:
            json.dump(track, f)

if __name__ == "__main__":
    train_and_predict()
    print("AI-ML Fusion complete.")
