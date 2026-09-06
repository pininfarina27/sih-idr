import os
import glob
import pandas as pd
import numpy as np
from multiprocessing import Pool

WINDOW_SIZE = 10   # 1-second window at 10 Hz
WINDOW_2S   = 20   # 2-second window at 10 Hz

def process_file(path):
    try:
        df = pd.read_csv(path, encoding='latin1')

        rename_map = {}
        for c in df.columns:
            if 'GPS SPEED'        in c: rename_map[c] = 'gps_speed'
            elif 'ACCELEROMETER Y' in c: rename_map[c] = 'accel_y'
            elif 'ACCELEROMETER Z' in c: rename_map[c] = 'accel_z'
            elif 'ACCELEROMETER X' in c: rename_map[c] = 'accel_x'
            elif 'GRAVITY Y'       in c: rename_map[c] = 'grav_y'
            elif 'GRAVITY Z'       in c: rename_map[c] = 'grav_z'
            elif 'GRAVITY X'       in c: rename_map[c] = 'grav_x'
            elif 'GYROSCOPE Yaw'   in c: rename_map[c] = 'gyro_z'
            elif 'GYROSCOPE Pitch' in c: rename_map[c] = 'gyro_x'
            elif 'GPS ORIENTATION' in c: rename_map[c] = 'gps_heading'
            elif 'TIME SINCE START' in c: rename_map[c] = 'time_ms'

        df.rename(columns=rename_map, inplace=True)

        required = ['gps_speed', 'accel_y', 'accel_z', 'gyro_z', 'gps_heading', 'time_ms']
        if not all(c in df.columns for c in required):
            return None

        # Speed: IO-VNBD GPS SPEED column is in m/s despite the label saying Kmh
        speed = df['gps_speed'] * (1000.0 / 3600.0)

        # Linear acceleration (subtract gravity component if available)
        ay = df['accel_y'] - df['grav_y'] if 'grav_y' in df.columns else df['accel_y']
        az = df['accel_z'] - df['grav_z'] if 'grav_z' in df.columns else df['accel_z']
        ax = df['accel_x'] - df['grav_x'] if ('accel_x' in df.columns and 'grav_x' in df.columns) \
             else (df['accel_x'] if 'accel_x' in df.columns else pd.Series(np.zeros(len(df)), index=df.index))
        gz = df['gyro_z']
        gx = df['gyro_x'] if 'gyro_x' in df.columns else pd.Series(np.zeros(len(df)), index=df.index)

        # Heading rate from GPS (deg/s) — used as a secondary target, filtered below
        diff_heading = df['gps_heading'].diff()
        diff_heading = (diff_heading + 180) % 360 - 180   # wrap-around safe
        dt = df['time_ms'].diff() / 1000.0
        dt = dt.replace(0, np.nan)
        heading_rate = diff_heading / dt

        w  = WINDOW_SIZE
        w2 = WINDOW_2S

        # ---- Original 5 features ----
        ay_mean   = ay.rolling(w, min_periods=1).mean()
        ay_std    = ay.rolling(w, min_periods=1).std().fillna(0)
        az_std    = az.rolling(w, min_periods=1).std().fillna(0)
        gz_mean   = gz.rolling(w, min_periods=1).mean()
        gz_std    = gz.rolling(w, min_periods=1).std().fillna(0)
        energy    = (ay**2 + az**2).rolling(w, min_periods=1).mean()

        # ---- Phase-1 new 5 features ----
        ax_mean   = ax.rolling(w, min_periods=1).mean()          # lateral accel mean (turn direction)
        ax_std    = ax.rolling(w, min_periods=1).std().fillna(0) # lateral accel std  (turn intensity)
        gx_std    = gx.rolling(w, min_periods=1).std().fillna(0) # pitch-rate std     (road quality)
        energy_2s = (ay**2 + az**2).rolling(w2, min_periods=1).mean()  # 2-s energy (highway speeds)
        az_mean   = az.rolling(w, min_periods=1).mean()          # vertical accel mean (tilt / gradient)

        hr_mean   = heading_rate.rolling(w, min_periods=1).mean()

        features = pd.DataFrame({
            'gps_speed':      speed,
            'heading_rate':   hr_mean,
            # original features
            'accel_y_mean':   ay_mean,
            'accel_y_std':    ay_std,
            'accel_z_std':    az_std,
            'gyro_z_mean':    gz_mean,
            'gyro_z_std':     gz_std,
            'accel_energy':   energy,
            # Phase-1 new features
            'accel_x_mean':   ax_mean,
            'accel_x_std':    ax_std,
            'gyro_x_std':     gx_std,
            'accel_energy_2s': energy_2s,
            'accel_z_mean':   az_mean,
        })

        # Drop rows with NaN and filter physically impossible heading rates
        features = features.dropna()
        features = features[abs(features['heading_rate']) < 90]
        return features

    except Exception:
        return None


if __name__ == '__main__':
    files = glob.glob(
        '../../IO-VNBD-temp/Synchronised V abd S datasets/Categorised IOVNB Dataset/**/S-*.csv',
        recursive=True
    )
    print(f'Found {len(files)} route CSV files.')
    with Pool(processes=8) as pool:
        results = pool.map(process_file, files)

    valid = [r for r in results if r is not None]
    print(f'Valid files: {len(valid)} / {len(files)}')
    final_df = pd.concat(valid, ignore_index=True)
    final_df.to_csv('data/deep_features.csv', index=False)
    print(f'Extracted {len(final_df):,} rows to data/deep_features.csv')
    print(f'Columns: {list(final_df.columns)}')
    print(f'Speed range: {final_df["gps_speed"].min():.3f} – {final_df["gps_speed"].max():.3f} m/s')
