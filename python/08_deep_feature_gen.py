import os
import glob
import pandas as pd
import numpy as np
from multiprocessing import Pool

def process_file(path):
    try:
        df = pd.read_csv(path, encoding='latin1')
        
        # Rename columns mapped
        rename_map = {}
        for c in df.columns:
            if 'GPS SPEED' in c: rename_map[c] = 'gps_speed'
            elif 'ACCELEROMETER Y' in c: rename_map[c] = 'accel_y'
            elif 'ACCELEROMETER Z' in c: rename_map[c] = 'accel_z'
            elif 'GRAVITY Y' in c: rename_map[c] = 'grav_y'
            elif 'GRAVITY Z' in c: rename_map[c] = 'grav_z'
            elif 'GYROSCOPE Yaw' in c: rename_map[c] = 'gyro_z'
            
        df.rename(columns=rename_map, inplace=True)
        
        # Ensure we have the required columns
        required = ['gps_speed', 'accel_y', 'accel_z', 'gyro_z']
        if not all(c in df.columns for c in required):
            return None
            
        # Target speed in m/s
        speed = df['gps_speed'] * (1000.0 / 3600.0)
        
        # Linear acceleration
        ay = df['accel_y'] - df['grav_y'] if 'grav_y' in df.columns else df['accel_y']
        az = df['accel_z'] - df['grav_z'] if 'grav_z' in df.columns else df['accel_z']
        gz = df['gyro_z']
        
        window_size = 10
        ay_mean = ay.rolling(window=window_size, min_periods=1).mean()
        ay_std = ay.rolling(window=window_size, min_periods=1).std().fillna(0)
        az_std = az.rolling(window=window_size, min_periods=1).std().fillna(0)
        gz_std = gz.rolling(window=window_size, min_periods=1).std().fillna(0)
        energy = (ay**2 + az**2).rolling(window=window_size, min_periods=1).mean()
        
        features = pd.DataFrame({
            'gps_speed': speed,
            'accel_y_mean': ay_mean,
            'accel_y_std': ay_std,
            'accel_z_std': az_std,
            'gyro_z_std': gz_std,
            'accel_energy': energy
        })
        
        return features.dropna()
    except Exception as e:
        print(f"Failed to process {path}: {e}")
        return None

if __name__ == "__main__":
    print("Finding all smartphone CSV files...")
    files = glob.glob('../../IO-VNBD-temp/Synchronised V abd S datasets/Categorised IOVNB Dataset/**/S-*.csv', recursive=True)
    print(f"Found {len(files)} files. Extracting features in parallel...")
    
    with Pool(processes=8) as pool:
        results = pool.map(process_file, files)
        
    valid_results = [r for r in results if r is not None]
    final_df = pd.concat(valid_results, ignore_index=True)
    
    print(f"Extracted {len(final_df)} rows of features.")
    final_df.to_csv("data/deep_features.csv", index=False)
    print("Saved to data/deep_features.csv.")
