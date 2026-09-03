import os
import pandas as pd
import numpy as np

def generate_features(segment_id, window_size=10):
    print(f"Generating features for {segment_id}...")
    df = pd.read_csv(f"data/{segment_id}.csv")
    
    # We will predict vehicle speed using IMU statistics (vibration, acceleration).
    # This is a robust ML approach for land vehicles when GPS is lost.
    
    # Calculate rolling statistics
    features = pd.DataFrame()
    features['time_ms'] = df['time_ms']
    features['gps_speed'] = df['gps_speed'] * (1000.0 / 3600.0) # Target in m/s
    features['gps_heading'] = df['gps_heading']
    features['lat'] = df['lat']
    features['lon'] = df['lon']
    features['blackout'] = df['blackout']
    
    # Linear acceleration (approximate by subtracting gravity if available)
    if 'GRAVITY Y (m/s)' in df.columns:
        ay = df['accel_y'] - df['GRAVITY Y (m/s)']
        az = df['accel_z'] - df['GRAVITY Z (m/s)']
    else:
        ay = df['accel_y']
        az = df['accel_z']
        
    gz = df['gyro_z']
    
    # Rolling features
    features['accel_y_mean'] = ay.rolling(window=window_size, min_periods=1).mean()
    features['accel_y_std'] = ay.rolling(window=window_size, min_periods=1).std().fillna(0)
    features['accel_z_std'] = az.rolling(window=window_size, min_periods=1).std().fillna(0)
    features['gyro_z_std'] = gz.rolling(window=window_size, min_periods=1).std().fillna(0)
    
    # Additional features: energy (sum of squares)
    features['accel_energy'] = (ay**2 + az**2).rolling(window=window_size, min_periods=1).mean()
    
    features.to_csv(f"data/{segment_id}_features.csv", index=False)

if __name__ == "__main__":
    for sid in ["S1", "S2", "S3a"]:
        if os.path.exists(f"data/{sid}.csv"):
            generate_features(sid)
    print("Feature generation complete.")
