# Engineering Rules, Standards & Anti-Patterns
# AI-ML Intelligent Dead Reckoning (IDR) System

**Problem Statement ID:** 26168 (ISRO, Department of Space)  
**Document Version:** 2.0 (Production Standard)  
**Classification:** Strict Engineering & Algorithmic Guidelines  
**Scope:** Frontend (React/TypeScript), Backend/Pipeline (Python/CUDA), Geospatial, & ML Modeling  

---

## 1. Core Engineering Principles

Every engineer and contributor to the `sih-idr` codebase must adhere strictly to these non-negotiable principles:

1. **Absolute Scientific Integrity:** Never falsify, round up, or fabricate compliance metrics. If a segment fails the ISRO $< 10\%$ criteria (such as curved segments S1 and S3a due to uncalibrated cradle pitch), report the exact failure percentage and explain the physical root cause. Scientific honesty is a primary judging criterion.
2. **Zero Ground-Truth Contamination:** Never snap predictions to the ground truth track. Map-matching must strictly project against an independent OpenStreetMap (OSM) road vector layer. Never feed future or privileged sensor readings (e.g. GPS heading during simulated blackout) into any evaluation track.
3. **Deterministic Single-Integration Paradigm:** Never attempt naive double-integration of linear acceleration for distance tracking. Always use the machine learning model to estimate velocity directly, transforming the formulation into single-integration of speed ($t^2 \to t$).
4. **Offline Edge Sovereignty:** The runtime navigation engine must never make network HTTP requests for inference, routing, or state propagation during active operation. It must execute 100% locally on the client within $< 1\text{ ms}$ per cycle.

---

## 2. Approved vs. Banned Tech Stack & Libraries

### 2.1 Approved Libraries & Tools
| Technology | Approved Version | Allowed Scope | Mandate Rationale |
|---|---|---|---|
| **React** | `19.0.0` | UI Presentation & Tabs | Modern concurrent rendering; requires strict hook hygiene |
| **TypeScript** | `5.7.2` | Client Application Code | Enforces strict static type safety (`strict: true`) |
| **Vite** | `6.2.0` | Client Bundler & Dev Server | Fast compilation, modern ES module bundling, tree-shaking |
| **Leaflet** | `1.9.4` | Interactive 2D Map Display | Ultra-lightweight DOM footprint (< 150 KB), zero WebGL crashes |
| **@turf/turf** | `7.2.0` | Offline Geodesic Utilities | Standard geodesic calculations (`distance`, `nearestPointOnLine`) |
| **Chart.js** | `4.4.8` | Metric Visualizations | Canvas-based performance, dynamic resizing, zero DOM clutter |
| **XGBoost** | `2.1.4` | Offline Model Training | High-efficiency gradient boosted trees with native CUDA GPU support |
| **Scikit-Learn** | `1.6.1` | Baselines & Evaluation | Standard LinearRegression and GradientBoosting baselines |
| **Filterpy** | `0.8.2` | Classical State Estimation | Standard Kalman filter state transition implementation |
| **Shapely** | `2.0.7` | Spatial Indexing (Python) | R-tree spatial indexing (`STRtree`) for sub-millisecond offline road snapping |

### 2.2 Strictly Banned Libraries & Patterns
| Prohibited Technology / Pattern | Why It Is Strictly Forbidden | Approved Alternative |
|---|---|---|
| ❌ **TensorFlow.js / ONNX Runtime Web** | Massive bundle overhead (> 25 MB), slow WASM compilation, high battery drain | Export 100 XGBoost decision trees to pure JSON (< 120 KB) with native TypeScript traversal |
| ❌ **Python Pickle (`.pkl`) in Production** | Arbitrary code execution vulnerability, binary incompatibility across platforms | Serialize model weights to pure JSON (`public/data/gbr_model.json`) |
| ❌ **Mapbox GL JS / MapLibre (heavy WebGL)**| Extreme GPU/battery consumption on mobile browsers, context loss crashes | Lightweight Leaflet 1.9.4 with SVG/Canvas overlays |
| ❌ **Cloud Inference APIs (FastAPI/Flask)** | Introduces network latency (100–500 ms), violates offline tunnel navigation mandate | 100% client-side TypeScript execution |
| ❌ **Naive Acceleration Double-Integration** | Explosive quadratic drift ($E_p \propto t^2$) diverges hundreds of meters in 15s | ML Virtual Speed Sensor (Single integration of velocity) |
| ❌ **Snapping to Ground Truth (`gt`)** | Circular logic: snaps prediction to the answer key; invalidates academic credibility | Snap strictly to genuine OpenStreetMap polylines (`osm_roads_*.json`) |
| ❌ **Reading GPS Heading During Blackout** | Privileged test data leakage; gives Kalman filter cheat information unavailable in tunnels | Integrate calibrated gyroscope yaw rate: $\psi_{k+1} = \psi_k + \omega_z \Delta t$ |

---

## 3. Frontend & React 19 Engineering Rules

### Rule 1: Zero Conditional Hooks (React Error #310)
- **Violation:** Calling `useMemo`, `useState`, or `useEffect` after an early `if (loading) return ...` or inside any conditional branch causes React to throw `Rendered more/fewer hooks than during the previous render` (Error #310), completely crashing the UI.
- **Rule:** **ALL hooks must be unconditionally declared at the very top of the functional component**, before ANY conditional return statement.

```typescript
// ❌ WRONG: Hook placed after conditional return
function MapView({ data }) {
  if (!data) return <LoadingSpinner />;
  const metrics = useMemo(() => computeMetrics(data), [data]); // CRASH!
  return <div>{metrics}</div>;
}

// ✅ CORRECT: Hooks unconditionally hoisted to top
function MapView({ data }) {
  const metrics = useMemo(() => (data ? computeMetrics(data) : null), [data]);
  if (!data || !metrics) return <LoadingSpinner />;
  return <div>{metrics}</div>;
}
```

### Rule 2: Zero Synchronous Heavy Geodesic Loops on Main Thread
- **Violation:** Running `@turf/nearest-point-on-line` for 400 trajectory points against 2,805 OSM road polylines synchronously on the main thread performs over $1.12 \times 10^6$ geodesic calculations, freezing the UI for $> 800\text{ ms}$ and causing mobile browser watchdog termination.
- **Rule:** Snapped coordinates must be **precomputed offline** via Python STRtree R-tree spatial indexes and embedded directly into the JSON data payload (`snapped_lat`, `snapped_lon`). If dynamic snapping is needed client-side, spatial bounding box pre-filtering must restrict candidates to $< 10$ road segments.

### Rule 3: Deterministic Leaflet Map Container Lifecycle
- **Violation:** Leaflet throws an unrecoverable exception if `L.map()` is initialized on a DOM container that already holds an active map instance.
- **Rule:** Always key the map component on the active segment:
```tsx
<MapView key={activeSeg} segmentId={activeSeg} />
```
This forces React to completely unmount and destroy the previous Leaflet DOM instance before mounting the new segment.

### Rule 4: Prevent Stale Closures in High-Frequency Sensor Handlers
- **Violation:** Reading React state (e.g. `isBlackout`) directly inside high-frequency event listeners (`window.addEventListener('devicemotion')`) captures the initial closure state, causing the handler to never recognize UI state toggles.
- **Rule:** Always store mutable flags in a React `useRef`:
```typescript
const isBlackoutRef = useRef<boolean>(false);
// Update in button click: isBlackoutRef.current = !isBlackoutRef.current;
// Read inside handleMotion: if (isBlackoutRef.current) { ... }
```

### Rule 5: Defensive Batch Fetching with Catch Handlers
- **Rule:** When loading multi-track segment data, always use `Promise.all` with individual catch fallbacks. Never leave empty array properties unhandled. Always guard array indexing:
```typescript
const point = points.length > 0 ? points[0] : null;
if (!point) return null;
```

---

## 4. Machine Learning & Data Pipeline Rules

### Rule 6: Absolute Speed Physical Unit Verification (The 3.6x Rule)
- **Violation:** In the IO-VNBD dataset, the column labeled `' GPS SPEED (Kmh)'` actually records speed in **meters per second (m/s)** (direct output of Android `Location.getSpeed()`). Dividing this column by 3.6 results in a severe 3.6x velocity underestimate ($14\text{ km/h}$ instead of $50\text{ km/h}$).
- **Rule:** Always verify physical units against coordinate displacement:
  $$\Delta s = \text{haversine}((\text{lat}_1, \text{lon}_1), (\text{lat}_2, \text{lon}_2))$$
  $$v_{\text{physical}} = \frac{\Delta s}{\Delta t} \approx \text{colValue} \implies \text{colValue is in m/s, NOT km/h!}$$
  In `05_generate_ml_features.py`, multiply by 3.6 to convert to true km/h before training.

### Rule 7: Strict Route-Level Partitioning (No Random Splits)
- **Violation:** Randomly splitting 10 Hz time-series data leaks temporally adjacent frames between train and test sets, artificially inflating $R^2$ to $> 0.95$ while failing catastrophically on new roads.
- **Rule:** Partition data **strictly by Route ID**:
  - Training: 57 routes (**617,548 frames**).
  - Held-Out Testing: 15 routes (**448,628 frames**).
  Zero route overlap is permitted between training and testing splits.

### Rule 8: Mandatory Zero Velocity Update (ZUPT) Thresholds
- **Rule:** Small engine idle vibrations at traffic lights or in tunnel congestion must never accumulate speed. Enforce strict stationary energy gating:
  $$\text{IF } \sigma(a_z) < 0.20\text{ m/s}^2 \quad \text{AND} \quad \sigma(\omega_z) < 0.02\text{ rad/s} \implies v_{\text{pred}} = 0.0\text{ m/s}$$

### Rule 9: Edge Tree Model Serialization Standard
- **Rule:** All trained tree models must be exported to a lightweight JSON structure specifying `features`, `learning_rate`, `init_value`, and an array of binary decision nodes with `feature`, `threshold`, `left`, `right`, and leaf `value`. Total payload size must not exceed $150\text{ KB}$.

---

## 5. Physical & Kinematic Modeling Rules

### Rule 10: In-Vehicle Smartphone Cradle Orientation Physics
- **Physical Reality:** In the IO-VNBD dataset, resting accelerometer values show $a_y \approx -9.8\text{ m/s}^2$. This proves the phone is mounted in an **upright windshield cradle** ($\text{pitch} \approx -85^\circ$):
  - Phone Y-axis = pointing down along gravity
  - Phone Z-axis = pointing forward through windshield
  - Phone X-axis = pointing right across cabin
- **Rule:** On curved roads, horizontal vehicle turning rotates around the phone's physical **X-axis** (pitch), while `gyro_z` registers near zero. In single-axis gyro DR, acknowledge that heading will project straight unless a full 3D direction cosine matrix (DCM) / quaternion filter resolves the tilt angle.

### Rule 11: Non-Holonomic Constraints (NHC)
- **Rule:** Ground vehicles cannot slide laterally or fly vertically. In the vehicle body frame:
  $$v_y \equiv 0 \quad (\text{zero lateral velocity}), \quad v_z \equiv 0 \quad (\text{zero vertical velocity})$$
  Only forward longitudinal velocity $v_x$ is non-zero.

### Rule 12: Gyroscope Yaw Integration Without Privileged Heading
- **Rule:** During simulated blackout intervals, classical Kalman filtering must integrate heading strictly from gyro yaw rate:
  $$\psi_{k+1} = \psi_k + \omega_{z, k} \Delta t$$
  Never read `row['gps_heading']` during blackout. That constitutes ground-truth leakage.

---

## 6. Performance & Quality Guardrails

| Parameter | Target Limit | Enforcement Mechanism |
|---|---|---|
| **Client Inference Latency** | $< 1.0\text{ ms}$ per sample | Pure TypeScript loop; zero memory allocations during inference |
| **Client Update Rate** | $\ge 10\text{ Hz}$ ($100\text{ ms}$ tick) | Synchronized with `DeviceMotionEvent` timestamp delta |
| **Max Memory Usage** | $< 50\text{ MB}$ Heap | Leaflet tile caching bound; precomputed JSON geometries |
| **Total Static Asset Bundle** | $< 5.0\text{ MB}$ | Gzipped Vite production build + optimized JSON files |
| **Blackout Mode Switch Time**| $< 100\text{ ms}$ | Immediate state switch on GPS accuracy drop or timeout |
| **Build Reproducibility** | Zero warnings / errors | `pnpm build` (`tsc -b && vite build`) must pass cleanly |
| **Consistency Verification** | 100% check pass | `cd python && python verify_consistency.py` must return `[OK]` |

---

## 7. Mandatory Pre-Commit & Verification Checklist

Before any commit or pull request is merged:
1. Run `cd python && python verify_consistency.py` — ensure all JSON data files, reports, and documentation numbers are in 100% mathematical parity.
2. Run `pnpm build` — ensure TypeScript compiles with zero errors and Vite bundles without warnings.
3. Test all 3 tabs in the browser:
   - **Benchmark Replay:** Switch between S1, S2, and S3a. Verify zero white screens, map auto-zooms, and Road-Snap toggle functions.
   - **Evaluation Dashboard:** Verify both Held-Out Models and Drift vs Duration charts render with correct data.
   - **Live Sensor Demo:** Verify accelerometer gauges respond, and the "Simulate GPS Blackout" button switches fusion logic cleanly.
