import os
import pandas as pd
import json

segments_info = [
    {"id": "S1", "path": "../IO-VNBD-temp/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/S-S1.csv", "start_sec": 60, "duration": 180},
    {"id": "S2", "path": "../IO-VNBD-temp/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/S-S2.csv", "start_sec": 60, "duration": 180},
    {"id": "S3a", "path": "../IO-VNBD-temp/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/S-S3a.csv", "start_sec": 60, "duration": 180},
]

out_dir = "../public/data"
os.makedirs(out_dir, exist_ok=True)
os.makedirs("data", exist_ok=True)

meta = {"segments": []}

for seg in segments_info:
    print(f"Processing {seg['id']}...")
    df = pd.read_csv(seg['path'], encoding='latin1')
    
    # Rename columns to safe names
    rename_map = {}
    for c in df.columns:
        if 'LATITUDE' in c: rename_map[c] = 'lat'
        elif 'LONGITUDE' in c: rename_map[c] = 'lon'
        elif 'GPS SPEED' in c: rename_map[c] = 'gps_speed'
        elif 'GPS ORIENTATION' in c: rename_map[c] = 'gps_heading'
        elif 'TIME SINCE START' in c: rename_map[c] = 'time_ms'
        elif 'ACCELEROMETER X' in c: rename_map[c] = 'accel_x'
        elif 'ACCELEROMETER Y' in c: rename_map[c] = 'accel_y'
        elif 'ACCELEROMETER Z' in c: rename_map[c] = 'accel_z'
        elif 'GYROSCOPE Yaw' in c: rename_map[c] = 'gyro_z'
        elif 'GYROSCOPE Pitch' in c: rename_map[c] = 'gyro_x'
        elif 'GYROSCOPE Roll' in c: rename_map[c] = 'gyro_y'
        elif 'ORIENTATION (Yaw)' in c: rename_map[c] = 'yaw'
        elif 'ORIENTATION (Pitch)' in c: rename_map[c] = 'pitch'
        elif 'ORIENTATION (Roll' in c: rename_map[c] = 'roll'

    df.rename(columns=rename_map, inplace=True)
    
    # Filter to the requested time window
    t_start_ms = df['time_ms'].iloc[0] + seg['start_sec']*1000
    t_end_ms = t_start_ms + seg['duration']*1000
    df_slice = df[(df['time_ms'] >= t_start_ms) & (df['time_ms'] <= t_end_ms)].copy()
    
    # Let's define a blackout window starting at 60s into the slice for 40s
    blackout_start = df_slice['time_ms'].iloc[0] + 60000
    blackout_end = blackout_start + 40000
    df_slice['blackout'] = (df_slice['time_ms'] >= blackout_start) & (df_slice['time_ms'] <= blackout_end)
    
    df_slice.to_csv(f"data/{seg['id']}.csv", index=False)
    
    # Ground truth format for web
    gt_track = []
    for _, row in df_slice.iterrows():
        gt_track.append({
            "ts": row['time_ms'] / 1000.0,
            "lat": row['lat'],
            "lon": row['lon'],
            "speed_kmh": row['gps_speed'],
            "heading": row['gps_heading']
        })
        
    with open(f"{out_dir}/segment_{seg['id']}_gt.json", "w") as f:
        json.dump(gt_track, f)
        
    meta["segments"].append({
        "id": seg["id"],
        "name": f"Driving Route {seg['id']}",
        "duration": seg['duration'],
        "blackout_start_ts": blackout_start / 1000.0,
        "blackout_end_ts": blackout_end / 1000.0
    })

with open(f"{out_dir}/segments.json", "w") as f:
    json.dump(meta, f)
    
print("Pre-processing complete.")
