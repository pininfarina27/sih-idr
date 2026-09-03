import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

def export_gbr():
    print("Loading data & retraining model to export...")
    train_dfs = []
    for sid in ["S1", "S2"]:
        df = pd.read_csv(f"data/{sid}_features.csv")
        train_dfs.append(df[~df['blackout']])
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    feature_cols = ['accel_y_mean', 'accel_y_std', 'accel_z_std', 'gyro_z_std', 'accel_energy']
    
    X_train = train_df[feature_cols].values
    y_train = train_df['gps_speed'].values
    
    # Tiny model for easy browser execution (20 trees)
    model = GradientBoostingRegressor(n_estimators=20, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Extract the mean initial value
    init_val = float(model.init_.constant_[0][0])
    
    # Extract trees
    trees_data = []
    for estimator in model.estimators_:
        tree = estimator[0].tree_
        
        def build_node(node_id):
            if tree.children_left[node_id] == -1: # Leaf node
                return {"value": float(tree.value[node_id][0][0])}
            return {
                "feature": int(tree.feature[node_id]),
                "threshold": float(tree.threshold[node_id]),
                "left": build_node(tree.children_left[node_id]),
                "right": build_node(tree.children_right[node_id])
            }
            
        trees_data.append(build_node(0))
        
    model_json = {
        "init": init_val,
        "learning_rate": model.learning_rate,
        "features": feature_cols,
        "trees": trees_data
    }
    
    with open("../public/data/gbr_model.json", "w") as f:
        json.dump(model_json, f)
        
    print("GBR Model exported to public/data/gbr_model.json")

if __name__ == "__main__":
    export_gbr()
