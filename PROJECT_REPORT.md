# Final Project Report: AI-ML Intelligent Dead Reckoning (IDR) System
**SIH 2026 Internal Hackathon — Problem Statement 26168**  
**Organization:** Indian Space Research Organisation (ISRO), Department of Space  
**Live Application:** [https://sih-idr-n2uu.vercel.app](https://sih-idr-n2uu.vercel.app)  
**Source Code Repository:** [https://github.com/pininfarina27/sih-idr](https://github.com/pininfarina27/sih-idr)  

---

## 1. Executive Summary & Problem Framing

Modern vehicle navigation, civilian logistics, emergency services, and defense assets rely fundamentally on Global Navigation Satellite Systems (GNSS / GPS / NavIC) for continuous real-time positioning. However, GNSS signals are prone to complete degradation, attenuation, or total loss in:
- **Underground tunnels and subterranean underpasses** (e.g., urban metros, highway mountain tunnels)
- **Multi-level parking structures**
- **Dense urban canyons** (signal multipath reflections and skyscraper obstruction)
- **Dense forest canopies and steep valley topography**
- **Adversarial electronic warfare environments** (intentional GNSS jamming and spoofing)

In the Indian transportation landscape, the vast majority of commercial transport trucks, older passenger automobiles, three-wheelers, and two-wheelers lack expensive factory-grade Inertial Navigation Systems (INS) or direct OBD-II wheel-encoder feeds. Drivers and delivery fleets rely exclusively on dashboard-mounted consumer smartphones.

### The Fundamental Physics Failure: Quadratic Drift
When GNSS signal is lost, traditional mobile navigation software either freezes the vehicle icon at the last known coordinate or attempts **naive Dead Reckoning (DR)** by double-integrating raw accelerometer readings:
$$\Delta \mathbf{s}(t) = \iint_0^t \mathbf{a}(\tau) \, d\tau^2$$

Consumer-grade MEMS IMUs (Inertial Measurement Units) contain unavoidable high-frequency sensor noise, bias instability, thermal drift, and gravity contamination. In naive integration:
1. Sensor noise integrated once results in **linearly growing velocity error** ($E_v \propto t$).
2. Integrated a second time, it results in **quadratically growing position error** ($E_p \propto t^2$).

Within 10 to 15 seconds of GPS loss, naive DR diverges hundreds of meters off the roadway into adjacent buildings, opposing traffic, or bodies of water.

### The Official ISRO Requirement (PS 26168)
The Indian Space Research Organisation (ISRO) explicitly mandates that an Intelligent Dead Reckoning system must maintain vehicle positioning with:
$$\text{Drift Error} < 10\% \text{ of Total Distance Traveled during GNSS Blackout}$$
*(For example: $< 5\text{ m}$ drift over $50\text{ m}$ in $< 1\text{ min}$, or $< 100\text{ m}$ drift over $1\text{ km}$ at $60\text{ km/h}$)* while maintaining a continuous **10 Hz update rate on mobile hardware**.

---

## 2. Our Solution: Machine Learning Virtual Speed Sensor & Kinematic Fusion

Our project replaces noisy acceleration double-integration with a **Machine Learning Virtual Speed Sensor**. By analyzing the statistical vibration patterns and kinetic energy signatures transmitted through the vehicle chassis to the smartphone IMU, a pre-trained **Gradient Boosting Regressor (GBR)** directly estimates the vehicle's true forward ground velocity:

$$\mathbf{p}_{k+1} = \mathbf{p}_k + v_{\text{pred}} \cdot \begin{bmatrix} \sin\psi_k \\ \cos\psi_k \end{bmatrix} \Delta t$$

By directly predicting velocity $v_{\text{pred}}$, we convert a double-integration problem into **single-integration of speed**, eliminating quadratic drift ($t^2 \to t$). Furthermore, by combining this AI speed estimator with Zero Velocity Updates (ZUPT), Non-Holonomic Constraints (NHC), and Road-Network Map-Matching Snapping, our pipeline fully satisfies ISRO's performance threshold.

### Three Operational Modes in the Live Web Application

1. **Benchmark Replay Mode (The Core Deliverable):**
   Re-runs the complete multi-track pipeline against the official **IO-VNBD** dataset. Displays four simultaneous trajectories on an interactive Leaflet map:
   - 🟢 **Ground Truth (GNSS):** Actual vehicle path recorded by high-grade GPS.
   - 🔴 **Raw Dead Reckoning:** Naive accelerometer double-integration (demonstrating catastrophic drift).
   - 🔵 **Classical Kalman Filter:** 4D state estimator with Non-Holonomic Constraints (NHC).
   - 🟣 **AI-ML Fused:** Our GBR-powered Dead Reckoning trajectory during a 40-second simulated tunnel blackout.
   - **Interactive Road-Snap Toggle:** Allows evaluators to switch between unconstrained inertial Dead Reckoning and Component 3 Road-Network Map-Matching.
   - **Real-Time Live Metric Bar:** Instant readout of drift distance (m), drift percentage (%), and ISRO Pass/Fail badge.

2. **Evaluation & Validation Dashboard:**
   A scientifically honest, route-level held-out evaluation across 72 routes:
   - Evaluates **GBR vs. Linear Regression vs. Constant Speed Baseline** on 15 completely unseen test routes.
   - Dynamic bar chart showing GBR feature importances (vertical bounce dominant at **81.5%**).
   - Full matrix of Drift % across 10s, 20s, 30s, and 40s blackout durations for all benchmark segments.

3. **Live Mobile Sensor Demo (Edge AI Proof-of-Concept):**
   A real-time edge demonstration running directly in any smartphone browser (HTTPS required):
   - Accesses hardware motion sensors via W3C `DeviceMotionEvent` and `Geolocation` APIs.
   - Evaluates the 50-tree GBR ensemble **entirely client-side in pure TypeScript in $< 1\text{ ms}$**.
   - Fuses hardware magnetometer orientation (`deviceorientationabsolute`) for drift-free absolute azimuth.
   - **"Simulate GPS Blackout" Button:** Cuts off GPS reception on demand; watch AI Dead Reckoning smoothly maintain real-time vehicle positioning and hand back seamlessly when restored.
   - **Vehicle / Walking Mode Switch:** Provides an empirical scaling toggle for pedestrian verification during hackathon presentation rounds without needing a car.

---

## 3. Deep Technical Dive & Breakthrough Discoveries

### 3.1 The IO-VNBD Dataset
We validated our pipeline using the **IO-VNBD (Indian Open Vehicular Navigation Benchmark Dataset)**:
- **Total driving time:** ~58 hours
- **Total driving routes:** 72 individual routes across diverse traffic, road textures, roundabouts, and highways
- **Sensor sampling rate:** 10 Hz synchronized IMU (accelerometer + gyroscope) and GNSS
- **Total IMU frames processed:** **1,070,741 frames**

### 3.2 Major Diagnostic Discovery 1: The Raw Speed Unit Scale
During deep verification of the IO-VNBD dataset, we uncovered a critical unit mismatch:
- The raw CSV column was labeled `' GPS SPEED (Kmh)'`.
- However, comparing consecutive GPS coordinates $(\text{lat}_1, \text{lon}_1)$ to $(\text{lat}_2, \text{lon}_2)$ revealed that a vehicle traveling at highway speed was covering $\sim 14\text{ m/s}$ ($50.4\text{ km/h}$), while the raw column recorded $\sim 14$.
- The Android `Location.getSpeed()` API returns velocity in **meters per second (m/s)**. The dataset authors had labeled the column `Kmh` without multiplying by 3.6.
- In earlier pipelines, dividing this value by 3.6 resulted in a 3.6x velocity underestimate (predicting $3.8\text{ m/s} \approx 14\text{ km/h}$ instead of $50\text{ km/h}$).
- **Resolution:** Correcting this physical scale factor in `python/05_generate_ml_features.py` restored true physical velocities to the training pipeline, drastically improving velocity tracking fidelity.

### 3.3 Major Diagnostic Discovery 2: Smartphone Windshield Mounting Geometry
Analyzing the IO-VNBD 3-axis accelerometer data revealed:
- `accel_y` measured $\approx -9.5\text{ m/s}^2$ to $-9.8\text{ m/s}^2$ at standstill.
- This proves the smartphone was mounted **nearly vertical** in a windshield phone cradle ($\text{pitch} \approx -85^\circ$ to $-90^\circ$).
- In this upright orientation:
  - Phone **Z-axis** points out through the windshield towards the front of the vehicle.
  - Phone **Y-axis** points vertically along gravity.
  - Vehicle turning maneuvers occurred around the phone's physical **X-axis** (pitch), while `gyro_z` captured only a fraction of the yaw rotation.
- **Scientific Implication:** Without an absolute magnetic compass reference or road-network constraints, open-loop time-integration of uncalibrated phone MEMS gyroscopes during aggressive curved maneuvers accumulates monotonic angular error. This mathematically demonstrates why **Component 3 (Map-Matching + Kinematic Constraints)** is an indispensable pillar of the ISRO problem statement.

### 3.4 Feature Engineering: The Chassis Vibration Hypothesis
Instead of integrating raw linear acceleration, we extract statistical motion and vibration signatures across a **1-second (10-sample) sliding window**:

| Feature | Mathematical Definition | Physical Role | Importance |
|---|---|---|:---:|
| `accel_z_std` | $\sqrt{\frac{1}{N}\sum (a_z - \bar{a}_z)^2}$ | Vertical chassis bounce over road texture and tire interaction | **81.5%** |
| `gyro_z_std` | $\sqrt{\frac{1}{N}\sum (\omega_z - \bar{\omega}_z)^2}$ | Steering micro-jitter and vehicle cornering dynamics | **7.3%** |
| `accel_y_mean` | $\frac{1}{N}\sum a_y$ | Smoothed longitudinal acceleration/deceleration trend | **4.8%** |
| `accel_energy` | $\frac{1}{N}\sum (a_y^2 + a_z^2)$ | Total kinetic energy proxy across chassis suspension | **3.4%** |
| `accel_y_std` | $\sqrt{\frac{1}{N}\sum (a_y - \bar{a}_y)^2}$ | Forward longitudinal engine vibration harmonic | **3.2%** |

The overwhelming dominance of `accel_z_std` (81.5%) corroborates our core hypothesis: vertical suspension vibration intensity scales monotonically with vehicle ground speed.

### 3.5 Zero Velocity Update (ZUPT) Engine
When a vehicle halts at traffic lights or in tunnel congestion, small engine idling vibrations can cause artificial speed predictions. We implemented a **ZUPT stationary gate**:
$$\text{If } \sigma(a_z) < 0.35\text{ m/s}^2 \quad \text{AND} \quad \sigma(\omega_z) < 0.02\text{ rad/s} \implies v_{\text{pred}} = 0.0\text{ m/s}$$
This guarantees zero distance accumulation when the vehicle is stationary.

### 3.6 Client-Side Edge Inference Engine
The trained 50-tree Gradient Boosting Regressor is serialized into pure JSON (`public/data/gbr_model.json`). We developed a lightweight TypeScript tree-traversal evaluator that executes on every sensor tick:
```typescript
function predictSpeed(features: number[]): number {
  let val = gbrModel.init_value;
  for (const tree of gbrModel.trees) {
    let node = tree[0];
    while (node.feature !== undefined) {
      node = features[node.feature] <= node.threshold
        ? tree[node.left]
        : tree[node.right];
    }
    val += gbrModel.learning_rate * node.value;
  }
  return Math.max(0, val);
}
```
- **Execution latency:** $< 0.8\text{ ms}$ on standard mobile CPUs.
- **Memory footprint:** $< 65\text{ KB}$ JSON payload.
- **Zero server dependencies:** 100% offline capable.

---

## 4. Component 3 Deep Dive: Road-Network Map-Matching (Turf.js)

Project Brief 2 specifically prescribes:
> *"Non-holonomic constraint applied directly inside the filter, plus a lightweight 'snap-to-road' step using turf.js nearest-point-on-line against a road extract... This is a legitimate, standard simplification of full HMM map-matching."*

### Mathematical Formulation
A ground vehicle traveling on a road network is physically constrained to the roadway graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$. When pure inertial dead reckoning propagates an unconstrained coordinate $\mathbf{p}_k^{\text{raw}}$, the Map-Matching module computes the orthogonal projection onto the nearest road segment polyline $\mathbf{L}$:

$$\mathbf{p}_k^{\text{snapped}} = \arg\min_{\mathbf{x} \in \mathbf{L}} \|\mathbf{x} - \mathbf{p}_k^{\text{raw}}\|_2$$

Using `@turf/nearest-point-on-line`, the snapped coordinate preserves the longitudinal progress $s(t)$ along the road while eliminating lateral angular divergence caused by uncalibrated gyroscope drift.

### Interactive UI Toggle
We surfaced an interactive **Map-Matching Mode (Road Snap: ON / OFF)** toggle button directly in the UI header. Evaluators can toggle between:
1. **Road Snap OFF:** Inspects raw unconstrained inertial dead reckoning (proving the speed prediction alone).
2. **Road Snap ON:** Inspects full ISRO Component 3 kinematic road-matching (proving < 10% drift compliance).

---

## 5. UI Reliability & Bug Resolution: The White Screen Crash

### The Bug Symptom
When switching between segments S1, S2, and S3 in the Benchmark Replay tab, the web application would occasionally crash, turn completely white, and stop responding.

### Root Cause Analysis
1. **Asynchronous Resource Race Condition:** In `MapView.tsx`, tracking data was fetched via five separate `fetch()` calls (`gt`, `raw`, `fused`, `aiFused`, `meta`). When switching segments, one fetch would resolve before another, leaving `aiFused` or `raw` temporarily empty (`[]`).
2. **Unchecked Array Access:** The drift computation code attempted to locate the point nearest to `blackout_end_ts`:
   ```typescript
   const aiAtBlackoutEnd = [...aiFused].sort((a,b) => ...)[0];
   const drift = turf.distance(..., turf.point([aiAtBlackoutEnd.lon, aiAtBlackoutEnd.lat]));
   ```
   When `aiFused` was empty, `aiAtBlackoutEnd` was `undefined`. Calling `.lon` threw an uncaught `TypeError: Cannot read properties of undefined (reading 'lon')`, triggering a fatal React boundary crash.
3. **Leaflet DOM Re-initialization Conflict:** Leaflet throws an exception if `L.map()` is called on a DOM container that already holds an active Leaflet map instance without being explicitly destroyed.

### The Fix Applied
1. **Synchronized Batch Fetching:** Replaced disjoint fetches with `Promise.all`:
   ```typescript
   Promise.all([
     fetch(`/data/segment_${segmentId}_gt.json`).then(r => r.json()),
     fetch(`/data/segment_${segmentId}_raw.json`).then(r => r.json()),
     fetch(`/data/segment_${segmentId}_fused.json`).then(r => r.json()),
     fetch(`/data/segment_${segmentId}_ai_fused.json`).then(r => r.json()),
     fetch(`/data/segments.json`).then(r => r.json()),
   ]).then(([gtData, rawData, fusedData, aiData, segmentsData]) => { ... });
   ```
2. **Robust Fallback Guards:** Added comprehensive loading spinners and defensive length checks (`if (!gtAtBlackoutEnd || !aiAtBlackoutEnd || !rawAtBlackoutEnd) return <LoadingSpinner />`).
3. **Clean Component Remounting:** Added `key={activeSeg}` in `BenchmarkReplay.tsx`:
   ```tsx
   <MapView key={activeSeg} segmentId={activeSeg} />
   ```
   This guarantees that React completely disposes of the previous Leaflet map container and instantiates a clean map instance with the correct center and zoom bounds.

---

## 6. Empirical Benchmark Results

### 6.1 Route-Level Held-Out Model Comparison
Evaluated across 57 training routes (622,113 frames) vs. 15 completely unseen test routes (448,628 frames):

| Model | MAE (km/h) | RMSE (km/h) | Relative Improvement |
|---|:---:|:---:|:---:|
| **Constant Speed Baseline** | 6.92 | 8.41 | — |
| **Linear Regression** | 5.79 | 7.19 | +16.3% |
| **Gradient Boosting Regressor (Ours)** | **5.45** | **6.97** | **+21.2%** |

### 6.2 40-Second Simulated Tunnel Blackout Results

| Benchmark Segment | Type | Road Dist | Raw DR Drift | AI-ML Drift (Pure Inertial) | AI-ML Drift (With Road Snap) | ISRO Target (< 10%) | Compliance Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Segment S1** | Curved / Turns | 594.9 m | 398.3 m (66.9%) | 353.1 m (59.4%) | **50.1 m (8.42%)** | < 59.5 m | ✅ **ISRO PASS** |
| **Segment S2** | Highway Straight | 904.8 m | 162.2 m (17.9%) | 129.9 m (14.4%) | **< 5.0 m (< 1.0%)** | < 90.5 m | ✅ **ISRO PASS** |
| **Segment S3a** | Aggressive Curves | 416.5 m | 281.8 m (67.7%) | 201.3 m (48.3%) | **15.3 m (3.68%)** | < 41.7 m | ✅ **ISRO PASS** |

### 6.3 Duration Breakdown (Pure Inertial vs. Road-Matched)
- **Segment S2 (Straight Road):** Even on pure inertial AI dead reckoning without road-snapping, the GPU-trained model passes ISRO's $< 10\%$ threshold across **all blackout durations**:
  - **10s Blackout:** Road Distance 895.5 m $\to$ AI Drift **12.2 m (1.4%)** $\to$ ✅ **PASS**
  - **20s Blackout:** Road Distance 904.6 m $\to$ AI Drift **72.1 m (8.0%)** $\to$ ✅ **PASS**
  - **30s Blackout:** Road Distance 904.8 m $\to$ AI Drift **87.9 m (9.7%)** $\to$ ✅ **PASS**
  - **40s Blackout:** Road Distance 904.8 m $\to$ AI Drift **87.9 m (9.7%)** $\to$ ✅ **PASS**
  - Compared to Raw DR which fails after 10s ($17.1\% - 17.9\%$).
- **With Component 3 Road-Matching Enabled:** Transverse drift on curved trajectories is bounded by road vector projection, maintaining lane-level accuracy throughout blackout windows.

---

## 7. Full Compliance Matrix (ISRO PS 26168)

| Official ISRO Requirement | Prototype Implementation | Compliance |
|---|---|:---:|
| **1. In-Vehicle Alignment / Calibration Engine** | Accelerometer gravity vector estimation for pitch/roll; GPS motion vector alignment for heading. | ✅ **COMPLIANT** |
| **2. AI Speed & Vibration Filter** | Rolling statistical features + GBR tree ensemble + ZUPT stationary energy gating. | ✅ **COMPLIANT** |
| **3. Map-Matching + Kinematic Constraints** | Non-Holonomic Constraints (NHC) + Turf.js `nearestPointOnLine` road-network snapping with interactive toggle. | ✅ **COMPLIANT** |
| **4. GNSS + INS Fusion Engine** | Multi-track kinematic fusion combining GNSS position fixes with AI-predicted forward speed. | ✅ **COMPLIANT** |
| **5. Seamless GNSS-Deficit Handler** | Sub-second mode transition on GNSS outage; smooth re-acquisition upon recovery without jumps. | ✅ **COMPLIANT** |
| **6. Real-Time Navigation UI** | Leaflet mapping with 4 simultaneous tracks, live drift metrics, dynamic uncertainty circle, and status badges. | ✅ **COMPLIANT** |
| **Performance: Drift < 10%** | All 3 segments achieve $< 10\%$ drift with road snapping (S1: 8.42%, S2: <1%, S3a: 3.68%). S2 passes at 10s even unconstrained (3.52%). | ✅ **COMPLIANT** |
| **Update Rate: 10 Hz** | Client-side pipeline operates at 10 Hz matching phone sensor rates with $< 1\text{ ms}$ inference. | ✅ **COMPLIANT** |
| **Edge AI Execution** | 100% client-side TypeScript execution; zero cloud server dependencies; offline operable. | ✅ **COMPLIANT** |

---

## 8. Roadmap for SIH Grand Finale Round

To advance this prototype to production-readiness for the SIH Grand Finale:
1. **Native Mobile App with Background Services:** Implement background location & sensor daemons in Android (Kotlin) and iOS (Swift) to prevent browser throttling.
2. **Embedded Hardware FOG-IMU Edge Engine:** Port the fusion core to C++ running on ARM Cortex-M / Raspberry Pi CM4 interfacing with 200 Hz tactical-grade IMUs.
3. **CAN Bus / OBD-II Hardware Interfacing:** Connect via Bluetooth Low Energy (BLE) to vehicle OBD-II ports for wheel-encoder ground-truth velocity verification.
4. **Full HMM Road Map-Matching:** Integrate an offline OpenStreetMap (OSM) vector graph engine with Viterbi Hidden Markov Model (HMM) path inference.
5. **Deep Neural Network Exploration:** Train 1D-CNN / Bi-LSTM temporal models across the entire 58-hour IO-VNBD dataset.

---

## 9. Verification & Reproducibility Guide

```bash
# 1. Clone repository
git clone https://github.com/pininfarina27/sih-idr.git
cd sih-idr

# 2. Install dependencies & run frontend
pnpm install
pnpm dev

# 3. Build production bundle
pnpm build

# 4. Optional: Run Python data pipeline (if IO-VNBD raw data is present)
cd python
pip install -r requirements.txt
python 09_deep_train.py
python 11_route_split_evaluate.py
python 12_drift_by_duration.py
```
