import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

def export_tree(model, feature_cols):
    init_val = float(model.init_.constant_[0][0])
    trees_data = []
    
    for estimator in model.estimators_:
        tree = estimator[0].tree_
        
        def build_node(node_id):
            if tree.children_left[node_id] == -1:
                return {"value": float(tree.value[node_id][0][0])}
            return {
                "feature": int(tree.feature[node_id]),
                "threshold": float(tree.threshold[node_id]),
                "left": build_node(tree.children_left[node_id]),
                "right": build_node(tree.children_right[node_id])
            }
        trees_data.append(build_node(0))
        
    return {
        "init": init_val,
        "learning_rate": model.learning_rate,
        "features": feature_cols,
        "trees": trees_data
    }

R = 6378137.0 
def add_meters_to_latlon(lat, lon, dx, dy):
    d_lat = dy / R
    d_lon = dx / (R * np.cos(np.pi * lat / 180.0))
    return lat + (d_lat * 180.0 / np.pi), lon + (d_lon * 180.0 / np.pi)

def train_and_export():
    print("Loading 1M+ rows of deep features...")
    df = pd.read_csv("data/deep_features.csv")
    
    # Optional: downsample for speed if needed, but 1M rows GBR(50 trees) takes maybe 10-20 secs.
    feature_cols = ['accel_y_mean', 'accel_y_std', 'accel_z_std', 'gyro_z_std', 'accel_energy']
    
    X = df[feature_cols].values
    # Target in m/s: correct the 3.6x underscaling from the Android logger
    y = df['gps_speed'].values * 3.6
    
    print("Training Deep Gradient Boosting Regressor (50 trees, depth 4)...")
    model = GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    
    print("Exporting model to JSON...")
    model_json = export_tree(model, feature_cols)
    with open("../public/data/gbr_model.json", "w") as f:
        json.dump(model_json, f)
        
    print("Regenerating benchmark tracks for S1, S2, S3a...")
    for sid in ["S1", "S2", "S3a"]:
        df_feat = pd.read_csv(f"data/{sid}_features.csv")
        df_raw = pd.read_csv(f"data/{sid}.csv")
        
        X_test = df_feat[feature_cols].values
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
                # Enhancement: ZUPT (Zero Velocity Update)
                if row['accel_z_std'] < 0.35 and row['gyro_z_std'] < 0.02:
                    speed = 0.0
                else:
                    speed = max(0.0, float(predicted_speeds[i]))
                    
                # Enhancement: Kinematic Turning Limit
                gyro_z = df_raw['gyro_z'].iloc[i] if not pd.isna(df_raw['gyro_z'].iloc[i]) else 0
                max_turn_rate = max(speed / 5.0, 0.1) if speed > 0 else 0.0
                clamped_gyro_z = np.clip(gyro_z, -max_turn_rate, max_turn_rate)
                
                current_heading -= np.degrees(clamped_gyro_z * dt)
                
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
            
    print("Deep training complete! Model exported and tracks updated.")

if __name__ == "__main__":
    train_and_export()

