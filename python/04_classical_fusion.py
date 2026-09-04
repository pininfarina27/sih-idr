import os
import json
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

# WGS84 parameters
R = 6378137.0 

def add_meters_to_latlon(lat, lon, dx, dy):
    d_lat = dy / R
    d_lon = dx / (R * np.cos(np.pi * lat / 180.0))
    new_lat = lat + (d_lat * 180.0 / np.pi)
    new_lon = lon + (d_lon * 180.0 / np.pi)
    return new_lat, new_lon

def apply_nhc(kf, heading_rad, sigma=0.2):
    """
    Non-Holonomic Constraint (NHC) pseudo-measurement.
    Enforces that vehicle lateral velocity (perpendicular to heading) is zero:
    v_lateral = v_x * cos(heading) - v_y * sin(heading) ≈ 0
    """
    H = np.array([[0, 0, np.cos(heading_rad), -np.sin(heading_rad)]])
    z = np.array([0.0])
    y = z - H @ kf.x
    R_mat = np.array([[sigma**2]])
    S = H @ kf.P @ H.T + R_mat
    K = kf.P @ H.T @ np.linalg.inv(S)
    kf.x = kf.x + (K @ y).flatten()
    I = np.eye(4)
    kf.P = (I - K @ H) @ kf.P

def compute_fused_dr(segment_id):
    print(f"Computing classical Kalman Filter fusion for {segment_id}...")
    df = pd.read_csv(f"data/{segment_id}.csv")
    
    # State: [x, y, vx, vy]  - 4D linear Kalman Filter with NHC
    kf = KalmanFilter(dim_x=4, dim_z=4)
    kf.x = np.zeros(4)
    kf.P = np.eye(4) * 10.0
    
    # State transition will be updated dynamically with dt
    # Measurement matrix for full 4D GPS fix
    kf.H = np.eye(4)
    # Measurement noise (GPS is noisy)
    kf.R = np.eye(4) * 5.0
    # Process noise (IMU is noisy)
    kf.Q = np.eye(4) * 0.1
    
    track = []
    
    origin_lat = df['lat'].iloc[0]
    origin_lon = df['lon'].iloc[0]
    
    def latlon_to_xy(lat, lon):
        dx = (lon - origin_lon) * (np.pi / 180.0) * R * np.cos(np.pi * origin_lat / 180.0)
        dy = (lat - origin_lat) * (np.pi / 180.0) * R
        return dx, dy
        
    def xy_to_latlon(dx, dy):
        return add_meters_to_latlon(origin_lat, origin_lon, dx, dy)
    
    current_heading = df['gps_heading'].iloc[0] if not pd.isna(df['gps_heading'].iloc[0]) else 0
    prev_time = df['time_ms'].iloc[0] / 1000.0
    
    for i in range(len(df)):
        row = df.iloc[i]
        curr_time = row['time_ms'] / 1000.0
        dt = curr_time - prev_time
        if dt > 1.0: dt = 0.1 
        if dt == 0: dt = 0.001
        prev_time = curr_time
        
        # State transition matrix F
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1,  0],
                         [0, 0, 0,  1]])
                         
        # Control input B and u (acceleration from IMU)
        lin_acc_y = row['accel_y'] - row['GRAVITY Y (m/s)'] if 'GRAVITY Y (m/s)' in row else row['accel_y']
        
        # Heading tracking: GPS when available, gyro integration during blackout
        if not row['blackout']:
            if not pd.isna(row.get('gps_heading')):
                current_heading = row['gps_heading']
        else:
            gyro_z = row['gyro_z'] if not pd.isna(row.get('gyro_z')) else 0
            current_heading -= np.degrees(gyro_z * dt)
            
        heading_rad = np.radians(current_heading)
        
        ax = lin_acc_y * np.sin(heading_rad)
        ay = lin_acc_y * np.cos(heading_rad)
        
        B = np.array([[0.5*dt**2, 0],
                      [0, 0.5*dt**2],
                      [dt, 0],
                      [0, dt]])
        u = np.array([ax, ay])
        
        # Prediction step
        kf.predict(B=B, u=u)
        
        if not row['blackout']:
            # GNSS Update
            gps_x, gps_y = latlon_to_xy(row['lat'], row['lon'])
            gps_speed = row['gps_speed'] * (1000.0 / 3600.0)
            gps_vx = gps_speed * np.sin(heading_rad)
            gps_vy = gps_speed * np.cos(heading_rad)
            
            z = np.array([gps_x, gps_y, gps_vx, gps_vy])
            kf.update(z)
        else:
            # Apply Non-Holonomic Constraint (NHC) pseudo-measurement during blackout
            # The vehicle cannot move sideways. We enforce velocity perpendicular to heading is ~0.
            apply_nhc(kf, heading_rad, sigma=0.2)
            
        current_lat, current_lon = xy_to_latlon(kf.x[0], kf.x[1])
        current_speed = np.sqrt(kf.x[2]**2 + kf.x[3]**2)
            
        track.append({
            "ts": curr_time,
            "lat": current_lat,
            "lon": current_lon,
            "speed_kmh": current_speed * 3.6,
            "heading": current_heading
        })
        
    with open(f"../public/data/segment_{segment_id}_fused.json", "w") as f:
        json.dump(track, f)

if __name__ == "__main__":
    for sid in ["S1", "S2", "S3a"]:
        if os.path.exists(f"data/{sid}.csv"):
            compute_fused_dr(sid)
    print("Classical Kalman Filter fusion complete.")
