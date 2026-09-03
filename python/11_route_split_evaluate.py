import os, sys, glob, json, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from multiprocessing import Pool

os.makedirs("../results", exist_ok=True)

FEATURE_COLS = ["accel_y_mean","accel_y_std","accel_z_std","gyro_z_std","accel_energy"]
DATASET_GLOB = "../../IO-VNBD-temp/Synchronised V abd S datasets/Categorised IOVNB Dataset/**/S-*.csv"

def process_file(path):
    """
    Load one raw smartphone CSV from IO-VNBD and compute rolling features.
    Returns a DataFrame with feature columns + gps_speed target, or None if invalid.
    Rolling window = 10 samples (1 second at 10 Hz).
    Features capture statistical structure of IMU vibrations, not raw values,
    because vibration variance is correlated with vehicle speed whereas raw
    acceleration reflects both speed and road/engine impulse noise.
    """
    try:
        df = pd.read_csv(path, encoding="latin1")
        rename = {}
        for c in df.columns:
            if "GPS SPEED" in c: rename[c] = "gps_speed"
            elif "ACCELEROMETER Y" in c: rename[c] = "accel_y"
            elif "ACCELEROMETER Z" in c: rename[c] = "accel_z"
            elif "GRAVITY Y" in c: rename[c] = "grav_y"
            elif "GRAVITY Z" in c: rename[c] = "grav_z"
            elif "GYROSCOPE Yaw" in c: rename[c] = "gyro_z"
        df.rename(columns=rename, inplace=True)
        required = ["gps_speed","accel_y","accel_z","gyro_z"]
        if not all(c in df.columns for c in required):
            return None
        speed = df["gps_speed"] * (1000.0 / 3600.0)
        ay = df["accel_y"] - df["grav_y"] if "grav_y" in df.columns else df["accel_y"]
        az = df["accel_z"] - df["grav_z"] if "grav_z" in df.columns else df["accel_z"]
        gz = df["gyro_z"]
        w = 10
        out = pd.DataFrame({
            "gps_speed": speed,
            "accel_y_mean": ay.rolling(w, min_periods=1).mean(),
            "accel_y_std":  ay.rolling(w, min_periods=1).std().fillna(0),
            "accel_z_std":  az.rolling(w, min_periods=1).std().fillna(0),
            "gyro_z_std":   gz.rolling(w, min_periods=1).std().fillna(0),
            "accel_energy": (ay**2 + az**2).rolling(w, min_periods=1).mean(),
        })
        return out.dropna()
    except Exception as e:
        return None

if __name__ == "__main__":
    files = glob.glob(DATASET_GLOB, recursive=True)
    print(f"Found {len(files)} route files.")
    random.seed(42)
    random.shuffle(files)
    split = int(0.80 * len(files))
    train_files = files[:split]
    test_files = files[split:]
    print(f"Route-level split: {len(train_files)} train routes, {len(test_files)} test routes")

    with Pool(8) as pool:
        train_dfs = pool.map(process_file, train_files)
        test_dfs  = pool.map(process_file, test_files)

    train_df = pd.concat([d for d in train_dfs if d is not None], ignore_index=True)
    test_df  = pd.concat([d for d in test_dfs  if d is not None], ignore_index=True)

    X_train, y_train = train_df[FEATURE_COLS].values, train_df["gps_speed"].values
    X_test,  y_test  = test_df[FEATURE_COLS].values,  test_df["gps_speed"].values
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # --- GBR ---
    gbr = GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
    gbr.fit(X_train, y_train)
    gbr_pred = gbr.predict(X_test)
    gbr_mae  = mean_absolute_error(y_test, gbr_pred) * 3.6
    gbr_rmse = np.sqrt(mean_squared_error(y_test, gbr_pred)) * 3.6

    # --- Linear Regression baseline ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mae  = mean_absolute_error(y_test, lr_pred) * 3.6
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred)) * 3.6

    # --- Constant-speed baseline (predicts mean of training set always) ---
    mean_speed = float(np.mean(y_train))
    const_pred = np.full_like(y_test, mean_speed)
    const_mae  = mean_absolute_error(y_test, const_pred) * 3.6
    const_rmse = np.sqrt(mean_squared_error(y_test, const_pred)) * 3.6

    # --- Feature importances ---
    importances = dict(zip(FEATURE_COLS, gbr.feature_importances_))
    sorted_imp  = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    report = f"""
==========================================
ROUTE-LEVEL HELD-OUT EVALUATION REPORT
(Honest split: 80% routes train / 20% routes test)
==========================================
Train routes: {len(train_files)}  |  Test routes: {len(test_files)}
Train rows:   {len(X_train)}   |  Test rows:  {len(X_test)}

MODEL COMPARISON ON UNSEEN ROUTES:
-------------------------------------------
                    MAE (km/h)   RMSE (km/h)
Constant Speed:     {const_mae:>8.2f}     {const_rmse:>8.2f}
Linear Regression:  {lr_mae:>8.2f}     {lr_rmse:>8.2f}
Gradient Boosting:  {gbr_mae:>8.2f}     {gbr_rmse:>8.2f}

FEATURE IMPORTANCES (GBR):
-------------------------------------------
"""
    for feat, imp in sorted_imp:
        report += f"  {feat:<20}: {imp*100:>5.1f}%\n"

    print(report)
    with open("../results/route_split_evaluation.txt", "w") as f:
        f.write(report)

    # --- Feature importance chart ---
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [f for f, _ in sorted_imp]
    vals  = [v*100 for _, v in sorted_imp]
    bars  = ax.barh(names[::-1], vals[::-1], color="#6366F1")
    ax.set_xlabel("Importance (%)")
    ax.set_title("GBR Feature Importances")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    plt.tight_layout()
    plt.savefig("../results/feature_importance.png", dpi=150)
    plt.close()
    print("Saved results/feature_importance.png")

    # Save a JSON summary for the frontend
    summary = {
        "route_split": {"train_routes": len(train_files), "test_routes": len(test_files),
                        "train_rows": len(X_train), "test_rows": int(len(X_test))},
        "gbr":   {"mae_kmh": round(gbr_mae, 2),  "rmse_kmh": round(gbr_rmse, 2)},
        "lr":    {"mae_kmh": round(lr_mae, 2),   "rmse_kmh": round(lr_rmse, 2)},
        "const": {"mae_kmh": round(const_mae, 2),"rmse_kmh": round(const_rmse, 2)},
        "feature_importances": {f: round(v*100, 1) for f, v in sorted_imp}
    }
    with open("../results/evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Done.")
