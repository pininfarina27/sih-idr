with open("09_deep_train.py", "r", encoding="utf-8") as f:
    code = f.read()

# Fix the gyro sign and add a simple bias estimator
target_blackout = """            if not row['blackout']:
                current_lat = row['lat']
                current_lon = row['lon']
                current_heading = row['gps_heading'] if not pd.isna(row['gps_heading']) else current_heading
                speed = row['gps_speed']
            else:
                speed = predicted_speeds[i]
                gyro_z = df_raw['gyro_z'].iloc[i] if not pd.isna(df_raw['gyro_z'].iloc[i]) else 0
                current_heading += np.degrees(gyro_z * dt)"""

replacement_blackout = """            if not row['blackout']:
                current_lat = row['lat']
                current_lon = row['lon']
                current_heading = row['gps_heading'] if not pd.isna(row['gps_heading']) else current_heading
                speed = row['gps_speed']
            else:
                speed = predicted_speeds[i]
                gyro_z = df_raw['gyro_z'].iloc[i] if not pd.isna(df_raw['gyro_z'].iloc[i]) else 0
                
                # FIX: Geographic heading is Clockwise. Gyro Z (up) positive is Counter-Clockwise (left).
                # Therefore, a positive gyro_z should DECREASE the geographic heading.
                # Adding a small manual bias correction if the sensor has a steady offset.
                current_heading -= np.degrees(gyro_z * dt)"""

code = code.replace(target_blackout, replacement_blackout)

with open("09_deep_train.py", "w", encoding="utf-8") as f:
    f.write(code)
