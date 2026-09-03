import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("Loading dataset...")
df = pd.read_csv("data/deep_features.csv")
feature_cols = ['accel_y_mean', 'accel_y_std', 'accel_z_std', 'gyro_z_std', 'accel_energy']

X = df[feature_cols].values
y = df['gps_speed'].values

print("Performing 80-20 Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} rows, Testing on {len(X_test)} rows...")
model = GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

print("Evaluating on unseen 20% test data...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Convert m/s to km/h for readable output
mae_kmh = mae * 3.6
rmse_kmh = rmse * 3.6

print("-" * 30)
print("MODEL ACCURACY METRICS (Unseen Data)")
print(f"Mean Absolute Error (MAE): {mae_kmh:.2f} km/h")
print(f"Root Mean Square Error (RMSE): {rmse_kmh:.2f} km/h")
print("-" * 30)
