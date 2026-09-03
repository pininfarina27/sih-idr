import os
import json
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# WGS84 parameters
R = 6378137.0 

def add_meters_to_latlon(lat, lon, dx, dy):
    d_lat = dy / R
    d_lon = dx / (R * np.cos(np.pi * lat / 180.0))
    new_lat = lat + (d_lat * 180.0 / np.pi)
    new_lon = lon + (d_lon * 180.0 / np.pi)
    return new_lat, new_lon

def compute_raw_dr(segment_id):
    print(f"Computing raw DR for {segment_id}...")
    df = pd.read_csv(f"data/{segment_id}.csv")
    
    track = []
    
    current_lat = df['lat'].iloc[0]
    current_lon = df['lon'].iloc[0]
    # Speed in m/s
    current_speed = df['gps_speed'].iloc[0] * (1000.0 / 3600.0) 
    current_heading = df['gps_heading'].iloc[0] if 'gps_heading' in df.columns else 0
    
    prev_time = df['time_ms'].iloc[0] / 1000.0
    
    for i in range(len(df)):
        row = df.iloc[i]
        curr_time = row['time_ms'] / 1000.0
        dt = curr_time - prev_time
        prev_time = curr_time
        
        if dt > 1.0: dt = 0.1 # safety
        if dt == 0: dt = 0.001
        
        if not row['blackout']:
            # GPS is available
            current_lat = row['lat']
            current_lon = row['lon']
            current_speed = row['gps_speed'] * (1000.0 / 3600.0)
            current_heading = row['gps_heading'] if not pd.isna(row.get('gps_heading')) else current_heading
        else:
            # GPS lost -> Naive Dead Reckoning
            # We assume the phone is somewhat aligned with the car, and the forward acceleration is roughly accel_y or accel_z minus gravity.
            # For a NAIVE baseline that drifts badly, we just take the magnitude of linear acceleration or a specific axis, 
            # and integrate it. Let's use the Y axis of linear accel as forward for this naive baseline (typical phone upright).
            
            lin_acc_y = row['accel_y'] - row['GRAVITY Y (m/s)'] if 'GRAVITY Y (m/s)' in row else row['accel_y']
            
            # Simple integration (drifts catastrophically due to bias)
            current_speed += lin_acc_y * dt
            
            # Update position (naive heading assumes it doesn't change much, or integrates gyro Z)
            gyro_z = row['gyro_z'] if not pd.isna(row['gyro_z']) else 0
            current_heading += np.degrees(gyro_z * dt)
            
            # Convert speed to dx, dy (heading is usually clockwise from North)
            heading_rad = np.radians(current_heading)
            # dx is East, dy is North
            dx = current_speed * np.sin(heading_rad) * dt
            dy = current_speed * np.cos(heading_rad) * dt
            
            current_lat, current_lon = add_meters_to_latlon(current_lat, current_lon, dx, dy)
            
        track.append({
            "ts": curr_time,
            "lat": current_lat,
            "lon": current_lon,
            "speed_kmh": current_speed * 3.6,
            "heading": current_heading
        })
        
    with open(f"../public/data/segment_{segment_id}_raw_dr.json", "w") as f:
        json.dump(track, f)

if __name__ == "__main__":
    for sid in ["S1", "S2", "S3a"]:
        if os.path.exists(f"data/{sid}.csv"):
            compute_raw_dr(sid)
    print("Raw DR baseline complete.")
