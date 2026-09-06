import os
import pandas as pd
import numpy as np

WINDOW_SIZE = 10   # 1-second window at 10 Hz
WINDOW_2S   = 20   # 2-second window at 10 Hz

# Feature columns produced by this script — must match FEATURE_COLS in 09_deep_train.py
FEATURE_COLS = [
    'accel_y_mean', 'accel_y_std', 'accel_z_std', 'gyro_z_std', 'accel_energy',
    'accel_x_mean', 'accel_x_std', 'gyro_x_std', 'accel_energy_2s', 'accel_z_mean',
]


def generate_features(segment_id, window_size=WINDOW_SIZE):
    print(f'Generating features for {segment_id}...')
    df = pd.read_csv(f'data/{segment_id}.csv')

    features = pd.DataFrame()
    features['time_ms']    = df['time_ms']
    features['gps_speed']  = df['gps_speed']   # in m/s (Location.getSpeed from IO-VNBD)
    features['gps_heading'] = df['gps_heading']
    features['lat']        = df['lat']
    features['lon']        = df['lon']
    features['blackout']   = df['blackout']

    # Linear acceleration (subtract gravity if available)
    if ' GRAVITY Y (m/s)' in df.columns:
        ay = df['accel_y'] - df[' GRAVITY Y (m/s)']
        az = df['accel_z'] - df[' GRAVITY Z (m/s)']
    else:
        ay = df['accel_y']
        az = df['accel_z']

    gz = df['gyro_z']

    # Lateral acceleration (accel_x - grav_x)
    if 'accel_x' in df.columns:
        if ' GRAVITY X (m/s)' in df.columns:
            ax = df['accel_x'] - df[' GRAVITY X (m/s)']
        elif ' GRAVITY X (m/s^)' in df.columns:
            ax = df['accel_x'] - df[' GRAVITY X (m/s^)']
        else:
            ax = df['accel_x']
    else:
        ax = pd.Series(np.zeros(len(df)), index=df.index)

    # Gyroscope pitch (gyro_x)
    gx = df['gyro_x'] if 'gyro_x' in df.columns else pd.Series(np.zeros(len(df)), index=df.index)

    w  = window_size
    w2 = WINDOW_2S

    # ---- Original 5 features ----
    features['accel_y_mean']   = ay.rolling(w, min_periods=1).mean()
    features['accel_y_std']    = ay.rolling(w, min_periods=1).std().fillna(0)
    features['accel_z_std']    = az.rolling(w, min_periods=1).std().fillna(0)
    features['gyro_z_std']     = gz.rolling(w, min_periods=1).std().fillna(0)
    features['accel_energy']   = (ay**2 + az**2).rolling(w, min_periods=1).mean()

    # ---- Phase-1 new 5 features ----
    features['accel_x_mean']    = ax.rolling(w, min_periods=1).mean()
    features['accel_x_std']     = ax.rolling(w, min_periods=1).std().fillna(0)
    features['gyro_x_std']      = gx.rolling(w, min_periods=1).std().fillna(0)
    features['accel_energy_2s'] = (ay**2 + az**2).rolling(w2, min_periods=1).mean()
    features['accel_z_mean']    = az.rolling(w, min_periods=1).mean()

    features.to_csv(f'data/{segment_id}_features.csv', index=False)
    print(f'  -> Saved data/{segment_id}_features.csv ({len(features)} rows)')


if __name__ == '__main__':
    for sid in ['S1', 'S2', 'S3a']:
        if os.path.exists(f'data/{sid}.csv'):
            generate_features(sid)
        else:
            print(f'WARNING: data/{sid}.csv not found, skipping.')
    print('Feature generation complete.')
