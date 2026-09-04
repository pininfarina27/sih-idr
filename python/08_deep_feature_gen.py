import os
import glob
import pandas as pd
import numpy as np
from multiprocessing import Pool

def process_file(path):
    try:
        df = pd.read_csv(path, encoding='latin1')
        
        rename_map = {}
        for c in df.columns:
            if 'GPS SPEED' in c: rename_map[c] = 'gps_speed'
            elif 'ACCELEROMETER Y' in c: rename_map[c] = 'accel_y'
            elif 'ACCELEROMETER Z' in c: rename_map[c] = 'accel_z'
            elif 'GRAVITY Y' in c: rename_map[c] = 'grav_y'
            elif 'GRAVITY Z' in c: rename_map[c] = 'grav_z'
            elif 'GYROSCOPE Yaw' in c: rename_map[c] = 'gyro_z'
            elif 'GPS ORIENTATION' in c: rename_map[c] = 'gps_heading'
            elif 'TIME SINCE START' in c: rename_map[c] = 'time_ms'
            
        df.rename(columns=rename_map, inplace=True)
        
        required = ['gps_speed', 'accel_y', 'accel_z', 'gyro_z', 'gps_heading', 'time_ms']
        if not all(c in df.columns for c in required):
            return None
            
        speed = df['gps_speed'] * (1000.0 / 3600.0)
        
        ay = df['accel_y'] - df['grav_y'] if 'grav_y' in df.columns else df['accel_y']
        az = df['accel_z'] - df['grav_z'] if 'grav_z' in df.columns else df['accel_z']
        gz = df['gyro_z']
        
        # Calculate true heading rate from GPS
        # heading is in degrees
        diff_heading = df['gps_heading'].diff()
        # Handle wrap-around (e.g. 359 to 1 is +2, not -358)
        diff_heading = (diff_heading + 180) % 360 - 180
        
        dt = df['time_ms'].diff() / 1000.0
        # Ignore extremely small dt to avoid division by zero
        dt = dt.replace(0, np.nan)
        heading_rate = diff_heading / dt
        
        window_size = 10
        ay_mean = ay.rolling(window=window_size, min_periods=1).mean()
        ay_std = ay.rolling(window=window_size, min_periods=1).std().fillna(0)
        az_std = az.rolling(window=window_size, min_periods=1).std().fillna(0)
        gz_mean = gz.rolling(window=window_size, min_periods=1).mean()
        gz_std = gz.rolling(window=window_size, min_periods=1).std().fillna(0)
        energy = (ay**2 + az**2).rolling(window=window_size, min_periods=1).mean()
        
        # Target heading rate is also smoothed over the window for stability
        hr_mean = heading_rate.rolling(window=window_size, min_periods=1).mean()
        
        features = pd.DataFrame({
            'gps_speed': speed,
            'heading_rate': hr_mean,
            'accel_y_mean': ay_mean,
            'accel_y_std': ay_std,
            'accel_z_std': az_std,
            'gyro_z_mean': gz_mean,
            'gyro_z_std': gz_std,
            'accel_energy': energy
        })
        
        # Filter out obvious GPS glitches (heading rate > 90 deg/s is unphysical for normal driving)
        features = features.dropna()
        features = features[abs(features['heading_rate']) < 90]
        return features
    except Exception as e:
        return None

if __name__ == "__main__":
    files = glob.glob('../../IO-VNBD-temp/Synchronised V abd S datasets/Categorised IOVNB Dataset/**/S-*.csv', recursive=True)
    with Pool(processes=8) as pool:
        results = pool.map(process_file, files)
        
    valid_results = [r for r in results if r is not None]
    final_df = pd.concat(valid_results, ignore_index=True)
    final_df.to_csv("data/deep_features.csv", index=False)
    print("Extracted", len(final_df), "rows with heading rate to data/deep_features.csv.")
