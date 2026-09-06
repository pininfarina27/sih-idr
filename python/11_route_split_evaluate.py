import os, sys, glob, json, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from multiprocessing import Pool

os.makedirs("../results", exist_ok=True)
os.makedirs("../public/data", exist_ok=True)

# =====================================================================
# Feature columns — MUST match 09_deep_train.py and
#                   05_generate_ml_features.py EXACTLY (Phase-1 updated)
# =====================================================================
FEATURE_COLS = [
    "accel_y_mean", "accel_y_std", "accel_z_std", "gyro_z_std", "accel_energy",
    "accel_x_mean", "accel_x_std", "gyro_x_std", "accel_energy_2s", "accel_z_mean",
]

WINDOW_SIZE = 10
WINDOW_2S   = 20
DATASET_GLOB = "../../IO-VNBD-temp/Synchronised V abd S datasets/Categorised IOVNB Dataset/**/S-*.csv"


def process_file(path):
    """Load one raw IO-VNBD CSV and compute all 10 rolling features."""
    try:
        df = pd.read_csv(path, encoding="latin1")
        rename = {}
        for c in df.columns:
            if "GPS SPEED"         in c: rename[c] = "gps_speed"
            elif "ACCELEROMETER Y" in c: rename[c] = "accel_y"
            elif "ACCELEROMETER Z" in c: rename[c] = "accel_z"
            elif "ACCELEROMETER X" in c: rename[c] = "accel_x"
            elif "GRAVITY Y"       in c: rename[c] = "grav_y"
            elif "GRAVITY Z"       in c: rename[c] = "grav_z"
            elif "GRAVITY X"       in c: rename[c] = "grav_x"
            elif "GYROSCOPE Yaw"   in c: rename[c] = "gyro_z"
            elif "GYROSCOPE Pitch" in c: rename[c] = "gyro_x"
        df.rename(columns=rename, inplace=True)

        required = ["gps_speed", "accel_y", "accel_z", "gyro_z"]
        if not all(c in df.columns for c in required):
            return None

        speed = df["gps_speed"] * (1000.0 / 3600.0)   # m/s
        ay = df["accel_y"] - df["grav_y"] if "grav_y" in df.columns else df["accel_y"]
        az = df["accel_z"] - df["grav_z"] if "grav_z" in df.columns else df["accel_z"]
        ax = (df["accel_x"] - df["grav_x"] if ("accel_x" in df.columns and "grav_x" in df.columns)
              else df.get("accel_x", pd.Series(np.zeros(len(df)))))
        gz = df["gyro_z"]
        gx = df.get("gyro_x", pd.Series(np.zeros(len(df)), index=df.index))

        w, w2 = WINDOW_SIZE, WINDOW_2S
        out = pd.DataFrame({
            "gps_speed":      speed,
            "accel_y_mean":   ay.rolling(w, min_periods=1).mean(),
            "accel_y_std":    ay.rolling(w, min_periods=1).std().fillna(0),
            "accel_z_std":    az.rolling(w, min_periods=1).std().fillna(0),
            "gyro_z_std":     gz.rolling(w, min_periods=1).std().fillna(0),
            "accel_energy":   (ay**2 + az**2).rolling(w, min_periods=1).mean(),
            "accel_x_mean":   ax.rolling(w, min_periods=1).mean(),
            "accel_x_std":    ax.rolling(w, min_periods=1).std().fillna(0),
            "gyro_x_std":     gx.rolling(w, min_periods=1).std().fillna(0),
            "accel_energy_2s":(ay**2 + az**2).rolling(w2, min_periods=1).mean(),
            "accel_z_mean":   az.rolling(w, min_periods=1).mean(),
        })
        return out.dropna()
    except Exception:
        return None


if __name__ == "__main__":
    files = glob.glob(DATASET_GLOB, recursive=True)
    if not files:
        files = glob.glob(
            "C:/Users/ranjo/OneDrive/Documents/Teckathon2/IO-VNBD-temp/"
            "Synchronised V abd S datasets/Categorised IOVNB Dataset/**/S-*.csv",
            recursive=True
        )
    print(f"Found {len(files)} route files.")
    random.seed(42)
    random.shuffle(files)
    split = int(0.80 * len(files))
    train_files = files[:split]
    test_files  = files[split:]
    print(f"Route-level split: {len(train_files)} train / {len(test_files)} test routes")

    with Pool(min(8, os.cpu_count() or 4)) as pool:
        train_dfs = pool.map(process_file, train_files)
        test_dfs  = pool.map(process_file, test_files)

    train_df = pd.concat([d for d in train_dfs if d is not None], ignore_index=True)
    test_df  = pd.concat([d for d in test_dfs  if d is not None], ignore_index=True)

    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df["gps_speed"].values.astype(np.float32)
    X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test  = test_df["gps_speed"].values.astype(np.float32)
    print(f"Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")

    # Detect CUDA
    device = "cuda"
    try:
        _t = xgb.XGBRegressor(n_estimators=1, tree_method="hist", device="cuda")
        _t.fit(np.array([[0.0]*len(FEATURE_COLS)], dtype=np.float32), np.array([0.0]))
        print("Using device=cuda  (NVIDIA RTX 3050)")
    except Exception as e:
        print(f"CUDA unavailable ({e}), using CPU")
        device = "cpu"

    # --- XGBoost (same hyperparams as 09_deep_train.py Phase-1) ---
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        tree_method="hist",
        device=device,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae  = mean_absolute_error(y_test, xgb_pred) * 3.6
    xgb_rmse = float(np.sqrt(mean_squared_error(y_test, xgb_pred))) * 3.6

    # --- Linear Regression baseline ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mae  = mean_absolute_error(y_test, lr_pred) * 3.6
    lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_pred))) * 3.6

    # --- Constant-speed baseline ---
    mean_speed = float(np.mean(y_train))
    const_pred = np.full_like(y_test, mean_speed)
    const_mae  = mean_absolute_error(y_test, const_pred) * 3.6
    const_rmse = float(np.sqrt(mean_squared_error(y_test, const_pred))) * 3.6

    # --- Feature importances ---
    raw_imp   = xgb_model.feature_importances_
    total_imp = np.sum(raw_imp) if np.sum(raw_imp) > 0 else 1.0
    importances = {f: float(i / total_imp) for f, i in zip(FEATURE_COLS, raw_imp)}
    sorted_imp  = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    report = f"""
==========================================
ROUTE-LEVEL HELD-OUT EVALUATION REPORT (XGBoost GPU — Phase-1 Model)
(Honest split: 80% routes train / 20% routes test)
==========================================
Train routes: {len(train_files)}  |  Test routes: {len(test_files)}
Train rows:   {len(X_train):,}   |  Test rows:  {len(X_test):,}
Features:     {len(FEATURE_COLS)} ({", ".join(FEATURE_COLS)})

MODEL COMPARISON ON UNSEEN ROUTES:
-------------------------------------------
                    MAE (km/h)   RMSE (km/h)
Constant Speed:     {const_mae:>8.2f}     {const_rmse:>8.2f}
Linear Regression:  {lr_mae:>8.2f}     {lr_rmse:>8.2f}
XGBoost Ensemble:   {xgb_mae:>8.2f}     {xgb_rmse:>8.2f}

FEATURE IMPORTANCES (XGBoost):
-------------------------------------------
"""
    for feat, imp in sorted_imp:
        report += f"  {feat:<22}: {imp*100:>5.1f}%\n"

    print(report)
    with open("../results/route_split_evaluation.txt", "w") as f:
        f.write(report)

    # Feature importance chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [f for f, _ in sorted_imp]
    vals  = [v*100 for _, v in sorted_imp]
    bars  = ax.barh(names[::-1], vals[::-1], color="#6366F1")
    ax.set_xlabel("Importance (%)")
    ax.set_title("XGBoost Feature Importances — Phase-1 Model (GPU Trained, 10 features)")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    plt.tight_layout()
    plt.savefig("../results/feature_importance.png", dpi=150)
    plt.close()
    print("Saved results/feature_importance.png")

    # Save summary for frontend
    summary = {
        "route_split": {"train_routes": len(train_files), "test_routes": len(test_files),
                        "train_rows": len(X_train), "test_rows": int(len(X_test))},
        "gbr":   {"mae_kmh": round(xgb_mae, 2),   "rmse_kmh": round(xgb_rmse, 2)},
        "lr":    {"mae_kmh": round(lr_mae, 2),    "rmse_kmh": round(lr_rmse, 2)},
        "const": {"mae_kmh": round(const_mae, 2), "rmse_kmh": round(const_rmse, 2)},
        "feature_importances": {f: round(v*100, 1) for f, v in sorted_imp},
    }
    for out_path in ["../results/evaluation_summary.json", "../public/data/evaluation_summary.json"]:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
    print("Done — evaluation_summary.json updated.")
