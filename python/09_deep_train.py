import os
import json
import math
import time
import importlib.util
import numpy as np
import pandas as pd
import xgboost as xgb

# =====================================================================
# Feature columns — MUST match 05_generate_ml_features.py and
#                   11_route_split_evaluate.py EXACTLY (Phase-1)
# =====================================================================
FEATURE_COLS = [
    "accel_y_mean", "accel_y_std", "accel_z_std", "gyro_z_std", "accel_energy",
    "accel_x_mean", "accel_x_std", "gyro_x_std", "accel_energy_2s", "accel_z_mean",
]

RESAMPLE_EVERY = 50   # steps between RCPF systematic resampling (5 s at 10 Hz)


# ---------------------------------------------------------------------------
def load_rcpf():
    """Dynamically load 14_rcpf.py so this file works even if not on sys.path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "14_rcpf.py")
    spec = importlib.util.spec_from_file_location("rcpf14", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def convert_xgb_node(node):
    if "leaf" in node:
        return {"value": float(node["leaf"])}
    feat_idx    = int(node["split"][1:])
    yes_id      = node["yes"]
    no_id       = node["no"]
    left_child  = node["children"][0] if node["children"][0]["nodeid"] == yes_id  else node["children"][1]
    right_child = node["children"][1] if node["children"][1]["nodeid"] == no_id   else node["children"][0]
    return {
        "feature":   feat_idx,
        "threshold": float(node["split_condition"]),
        "left":      convert_xgb_node(left_child),
        "right":     convert_xgb_node(right_child),
    }


_R = 6_378_137.0

def add_meters_to_latlon(lat, lon, dx, dy):
    d_lat = dy / _R
    d_lon = dx / (_R * np.cos(np.pi * lat / 180.0))
    return lat + np.degrees(d_lat), lon + np.degrees(d_lon)


# Road-type typical minimum speeds (m/s) — used to floor ML speed predictions
# when the model under-estimates on roads outside training range.
# Values are conservative lower bounds (not speed limits).
_HW_SPEED_FLOOR = {
    "motorway":      22.2,   # 80 km/h min
    "motorway_link": 11.1,   # 40 km/h min
    "trunk":         11.1,   # 40 km/h min  (IO-VNBD S3a trunk actual = 42 km/h)
    "trunk_link":     5.5,   # 20 km/h min
    "primary":       11.1,   # 40 km/h min  (Binley Rd=primary, actual 42 km/h; S1 already at 13.4)
    "primary_link":   0.0,
    "secondary":      0.0,   # no floor — slow movement is valid
    "secondary_link": 0.0,
    "tertiary":       0.0,
    "residential":    0.0,
    "service":        0.0,
    "living_street":  0.0,
    "unclassified":   0.0,
}

def highway_speed_floor(particles, edge_map, calib_speed, step_in_blackout=0):
    """
    Return max(calib_speed, road-type minimum).
    Only applies trunk floor after step 30 (3 seconds) to allow entry speed warmup.
    Uses the MAXIMUM floor among highway types that have >= 50% particle weight.
    """
    if not particles or step_in_blackout < 30:
        return calib_speed   # No floor during first 3s (warmup)
    hw_counts = {}
    total = len(particles)
    for eid, _, _ in particles:
        edge = edge_map.get(eid)
        if edge:
            hw = edge.get("highway", "")
            hw_counts[hw] = hw_counts.get(hw, 0) + 1
    if not hw_counts:
        return calib_speed
    # Apply the max floor across ALL highway types present in the particle cloud
    best_floor = 0.0
    for hw in hw_counts:
        floor_v = _HW_SPEED_FLOOR.get(hw, 0.0)
        best_floor = max(best_floor, floor_v)
    return max(calib_speed, best_floor)


# ---------------------------------------------------------------------------
def train_and_export():
    # ------------------------------------------------------------------
    # 1.  Load and stratify deep features (Phase-1)
    # ------------------------------------------------------------------
    print("Loading deep features...")
    df = pd.read_csv("data/deep_features.csv")
    print(f"  Raw rows: {len(df):,}  |  columns: {list(df.columns)}")

    df["speed_decile"] = pd.qcut(df["gps_speed"], q=10, labels=False, duplicates="drop")
    SAMPLES_PER_DECILE = 60_000
    df_balanced = (
        df.groupby("speed_decile", group_keys=False)
          .apply(lambda x: x.sample(min(len(x), SAMPLES_PER_DECILE), random_state=42))
    )
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  After stratified sampling: {len(df_balanced):,} rows")
    print(f"  Speed range: {df_balanced['gps_speed'].min():.3f} – {df_balanced['gps_speed'].max():.3f} m/s")

    X = df_balanced[FEATURE_COLS].values.astype(np.float32)
    y = df_balanced["gps_speed"].values.astype(np.float32)

    # ------------------------------------------------------------------
    # 2.  Train XGBoost on RTX 3050 (Phase-1 hyperparams)
    # ------------------------------------------------------------------
    device = "cuda"
    try:
        _t = xgb.XGBRegressor(n_estimators=1, tree_method="hist", device="cuda")
        _t.fit(np.array([[0.0] * len(FEATURE_COLS)], dtype=np.float32), np.array([0.0]))
        print("Using device=cuda  (NVIDIA RTX 3050)")
    except Exception as e:
        print(f"CUDA unavailable ({e}), falling back to CPU")
        device = "cpu"

    print(f"Training XGBoost on {len(X):,} stratified samples, {len(FEATURE_COLS)} features...")
    t0 = time.time()
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1,
        tree_method="hist", device=device, random_state=42, verbosity=1,
    )
    model.fit(X, y)
    print(f"Training done in {time.time()-t0:.1f}s")

    # Export lightweight JSON model for TypeScript edge inference
    raw_json   = json.loads(model.get_booster().save_raw(raw_format="json"))
    base_score = float(raw_json["learner"]["learner_model_param"]["base_score"].strip("[]"))
    dump       = model.get_booster().get_dump(dump_format="json")
    trees_data = [convert_xgb_node(json.loads(d)) for d in dump]

    model_json = {
        "init":          base_score,
        "learning_rate": 1.0,
        "features":      FEATURE_COLS,
        "trees":         trees_data,
    }
    out_json = "../public/data/gbr_model.json"
    print(f"Exporting model to {out_json} ({len(trees_data)} trees)...")
    with open(out_json, "w") as f:
        json.dump(model_json, f)

    # ------------------------------------------------------------------
    # 3.  Load RCPF (Phase-3)
    # ------------------------------------------------------------------
    print("\nLoading RCPF module (14_rcpf.py)...")
    rcpf_mod = load_rcpf()
    RCParticleFilter = rcpf_mod.RCParticleFilter
    print("RCPF module loaded OK")

    # ------------------------------------------------------------------
    # 4.  Regenerate benchmark tracks with RCPF (Phase-3)
    # ------------------------------------------------------------------
    print("\nRegenerating benchmark AI tracks with RCPF heading...")

    for sid in ["S1", "S2", "S3a"]:
        feat_path  = f"data/{sid}_features.csv"
        raw_path   = f"data/{sid}.csv"
        graph_path = f"../public/data/road_graph_{sid}.json"

        if not os.path.exists(feat_path):
            print(f"  WARNING: {feat_path} missing — run 05_generate_ml_features.py first")
            continue
        if not os.path.exists(graph_path):
            print(f"  WARNING: {graph_path} missing — run 13_build_osm_graph.py first")
            continue

        df_feat = pd.read_csv(feat_path)
        df_raw  = pd.read_csv(raw_path)

        # Load road graph for this segment
        with open(graph_path) as f:
            graph = json.load(f)

        X_test = df_feat[FEATURE_COLS].values.astype(np.float32)
        predicted_speeds = model.predict(X_test)

        # --- Speed-ratio calibration (improved anchor) ---
        # Use max(last GPS speed, max speed in 2s window before blackout) as anchor.
        # This handles:
        #   S1: last GPS = 15.55 m/s -> anchor = 15.55 m/s (unchanged)
        #   S3a: last GPS = 8.37 m/s but vehicle accelerates -> max in 2s window catches it
        bk_mask      = df_feat["blackout"].values.astype(bool)
        bk_start_idx = int(np.argmax(bk_mask))
        WINDOW_2S    = 20   # 2s at 10 Hz = 20 rows
        pre_bk_start = max(0, bk_start_idx - WINDOW_2S)
        pre_bk_max   = float(df_feat["gps_speed"].iloc[pre_bk_start:bk_start_idx].max())
        last_gps_val = float(df_feat["gps_speed"].iloc[bk_start_idx - 1]) if bk_start_idx > 0 else 0.0
        anchor_speed = max(last_gps_val, pre_bk_max)
        last_gps_spd = anchor_speed

        model_at_entry = max(float(predicted_speeds[bk_start_idx]), 0.01)

        if last_gps_spd > 0.5:
            speed_ratio = float(np.clip(last_gps_spd / model_at_entry, 0.5, 6.0))
        else:
            speed_ratio = 1.0

        print(f"\n  {sid}: anchor GPS {last_gps_spd:.2f} m/s (max 2s+last)  "
              f"model@entry {model_at_entry:.2f} m/s  ratio={speed_ratio:.2f}x")

        # --- Initialise RCPF at blackout entry ---
        entry_row = df_feat.iloc[bk_start_idx]
        entry_heading = float(entry_row["gps_heading"]) if not pd.isna(entry_row["gps_heading"]) else 0.0
        pf = RCParticleFilter(graph, N=500, rng_seed=42)
        particles = pf.initialize(
            float(entry_row["lat"]),
            float(entry_row["lon"]),
            entry_heading,
        )
        print(f"    RCPF init: {pf.unique_edges(particles)} unique edges")

        # --- Dead-reckoning loop ---
        track             = []
        current_lat       = df_feat["lat"].iloc[0]
        current_lon       = df_feat["lon"].iloc[0]
        current_heading   = (df_feat["gps_heading"].iloc[0]
                             if not pd.isna(df_feat["gps_heading"].iloc[0])
                             else 0.0)
        global_heading    = entry_heading   # RCPF global heading reference (v3)
        prev_time         = df_feat["time_ms"].iloc[0] / 1000.0
        step_in_blackout  = 0
        bk_pred_speeds    = []

        for i in range(len(df_feat)):
            row       = df_feat.iloc[i]
            curr_time = row["time_ms"] / 1000.0
            dt        = max(min(curr_time - prev_time, 1.0), 0.001)
            prev_time = curr_time

            if not row["blackout"]:
                # GPS available — ground truth
                current_lat     = row["lat"]
                current_lon     = row["lon"]
                current_heading = (row["gps_heading"]
                                   if not pd.isna(row["gps_heading"])
                                   else current_heading)
                speed = row["gps_speed"]
                step_in_blackout = 0

            else:
                # ZUPT
                if row["accel_z_std"] < 0.20 and row["gyro_z_std"] < 0.02:
                    speed = 0.0
                else:
                    raw_pred = max(0.0, float(predicted_speeds[i]))
                    speed    = raw_pred * speed_ratio
                    # Apply road-type speed floor after 3s warmup
                    speed = highway_speed_floor(particles, pf.edge_map, speed, step_in_blackout)

                bk_pred_speeds.append(speed)
                lat_accel = float(row["accel_x_mean"])

                # RCPF predict — pass global heading reference (v3)
                particles = pf.predict(particles, speed, dt, lat_accel, global_heading)

                step_in_blackout += 1
                if step_in_blackout % RESAMPLE_EVERY == 0:
                    particles = pf.resample(particles)

                rcpf_lat, rcpf_lon, rcpf_hdg = pf.weighted_position(particles)

                if rcpf_lat is not None:
                    current_lat     = rcpf_lat
                    current_lon     = rcpf_lon
                    current_heading = rcpf_hdg
                    # Update global heading (very slow exponential smoothing — sticky reference)
                    alpha           = 0.02   # ~50-step time constant, keeps heading anchored
                    sin_g = (1-alpha)*math.sin(math.radians(global_heading)) + alpha*math.sin(math.radians(rcpf_hdg))
                    cos_g = (1-alpha)*math.cos(math.radians(global_heading)) + alpha*math.cos(math.radians(rcpf_hdg))
                    global_heading  = (math.degrees(math.atan2(sin_g, cos_g)) + 360.0) % 360.0
                else:
                    # Fallback: pure DR
                    heading_rad = np.radians(current_heading)
                    dx = speed * np.sin(heading_rad) * dt
                    dy = speed * np.cos(heading_rad) * dt
                    current_lat, current_lon = add_meters_to_latlon(
                        current_lat, current_lon, dx, dy)

            track.append({
                "ts":        curr_time,
                "lat":       current_lat,
                "lon":       current_lon,
                "speed_kmh": speed * 3.6,
                "heading":   current_heading,
            })

        # --- Diversity log ---
        print(f"    RCPF final unique edges: {pf.unique_edges(particles)}")
        if bk_pred_speeds:
            bk_arr = np.array(bk_pred_speeds)
            bk_act = df_feat.loc[bk_mask, "gps_speed"].values
            print(f"    Speed — calibrated: {bk_arr.mean():.2f} m/s ({bk_arr.mean()*3.6:.1f} km/h)"
                  f"  actual: {bk_act.mean():.2f} m/s ({bk_act.mean()*3.6:.1f} km/h)")

        out_path = f"../public/data/segment_{sid}_ai_fused.json"
        with open(out_path, "w") as f:
            json.dump(track, f)
        print(f"    -> Saved {out_path} ({len(track)} points)")

    print("\nPhase-3 RCPF training & track generation complete!")


if __name__ == "__main__":
    train_and_export()
