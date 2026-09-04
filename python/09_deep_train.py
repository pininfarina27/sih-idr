import os
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb

def convert_xgb_node(node):
    if 'leaf' in node:
        return {'value': float(node['leaf'])}
    feat_idx = int(node['split'][1:])
    yes_id = node['yes']
    no_id = node['no']
    left_child = node['children'][0] if node['children'][0]['nodeid'] == yes_id else node['children'][1]
    right_child = node['children'][1] if node['children'][1]['nodeid'] == no_id else node['children'][0]
    return {
        'feature': feat_idx,
        'threshold': float(node['split_condition']),
        'left': convert_xgb_node(left_child),
        'right': convert_xgb_node(right_child)
    }

R = 6378137.0 
def add_meters_to_latlon(lat, lon, dx, dy):
    d_lat = dy / R
    d_lon = dx / (R * np.cos(np.pi * lat / 180.0))
    return lat + (d_lat * 180.0 / np.pi), lon + (d_lon * 180.0 / np.pi)

def train_and_export():
    print("Loading 1M+ rows of deep features...")
    df = pd.read_csv("data/deep_features.csv")
    
    feature_cols = ['accel_y_mean', 'accel_y_std', 'accel_z_std', 'gyro_z_std', 'accel_energy']
    X = df[feature_cols].values.astype(np.float32)
    y = df['gps_speed'].values.astype(np.float32)
    
    # Train using the laptop's dedicated NVIDIA RTX 3050 6GB GPU (device='cuda')
    device = 'cuda'
    print(f"Training on NVIDIA GeForce RTX 3050 GPU (device='{device}', tree_method='hist')...")
    t0 = time.time()
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        device=device,
        random_state=42
    )
    model.fit(X, y)
    print(f"Training completed on NVIDIA RTX 3050 GPU in {time.time()-t0:.2f} seconds!")
    
    # Extract raw base_score and serialized trees for TS edge inference
    raw_json = json.loads(model.get_booster().save_raw(raw_format='json'))
    base_score = float(raw_json['learner']['learner_model_param']['base_score'].strip('[]'))
    
    dump = model.get_booster().get_dump(dump_format='json')
    trees_data = [convert_xgb_node(json.loads(d)) for d in dump]
    
    model_json = {
        'init': base_score,
        'learning_rate': 1.0,
        'features': feature_cols,
        'trees': trees_data
    }
    
    out_json = "../public/data/gbr_model.json"
    print(f"Exporting model to {out_json} for TypeScript client-side evaluation...")
    with open(out_json, "w") as f:
        json.dump(model_json, f)
        
    print("Regenerating benchmark tracks for S1, S2, S3a...")
    for sid in ["S1", "S2", "S3a"]:
        df_feat = pd.read_csv(f"data/{sid}_features.csv")
        df_raw = pd.read_csv(f"data/{sid}.csv")
        
        X_test = df_feat[feature_cols].values.astype(np.float32)
        predicted_speeds = model.predict(X_test)
        
        track = []
        current_lat = df_feat['lat'].iloc[0]
        current_lon = df_feat['lon'].iloc[0]
        current_heading = df_feat['gps_heading'].iloc[0] if not pd.isna(df_feat['gps_heading'].iloc[0]) else 0
        prev_time = df_feat['time_ms'].iloc[0] / 1000.0
        
        for i in range(len(df_feat)):
            row = df_feat.iloc[i]
            curr_time = row['time_ms'] / 1000.0
            dt = max(min(curr_time - prev_time, 1.0), 0.001)
            prev_time = curr_time
            
            if not row['blackout']:
                current_lat = row['lat']
                current_lon = row['lon']
                current_heading = row['gps_heading'] if not pd.isna(row['gps_heading']) else current_heading
                speed = row['gps_speed']
            else:
                # ZUPT: clamp stationary noise
                if row['accel_z_std'] < 0.20 and row['gyro_z_std'] < 0.02:
                    speed = 0.0
                else:
                    speed = max(0.0, float(predicted_speeds[i]))
                    
                gyro_z = df_raw['gyro_z'].iloc[i] if not pd.isna(df_raw['gyro_z'].iloc[i]) else 0
                current_heading -= np.degrees(gyro_z * dt)
                
                heading_rad = np.radians(current_heading)
                dx = speed * np.sin(heading_rad) * dt
                dy = speed * np.cos(heading_rad) * dt
                current_lat, current_lon = add_meters_to_latlon(current_lat, current_lon, dx, dy)
                
            track.append({
                "ts": curr_time, "lat": current_lat, "lon": current_lon,
                "speed_kmh": speed * 3.6, "heading": current_heading
            })
            
        with open(f"../public/data/segment_{sid}_ai_fused.json", "w") as f:
            json.dump(track, f)
            
    print("GPU training & track generation complete!")

if __name__ == "__main__":
    train_and_export()
