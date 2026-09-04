# Product Requirements Document (PRD)
# AI-ML Intelligent Dead Reckoning (IDR) System

**Project Name:** AI-ML Intelligent Dead Reckoning (IDR) for Vehicular Navigation in GNSS-Denied Environments  
**Problem Statement ID:** 26168  
**Sponsoring Organization:** Indian Space Research Organisation (ISRO), Department of Space  
**Event:** Smart India Hackathon (SIH) 2026  
**Document Version:** 2.0 (Production Release)  
**Live Production Application:** [https://sih-idr-n2uu.vercel.app](https://sih-idr-n2uu.vercel.app)  
**GitHub Repository:** [https://github.com/pininfarina27/sih-idr](https://github.com/pininfarina27/sih-idr)  

---

## 1. Executive Summary & Problem Framing

Global Navigation Satellite Systems (GNSS), including GPS and India's indigenous NavIC (IRNSS), provide high-precision positioning for civilian transport, commercial logistics, emergency services, and defense operations. However, satellite radio-frequency signals require an unobstructed Line-of-Sight (LoS) between orbital constellations and the terrestrial receiver. In practical vehicular scenarios, GNSS signals suffer from total blackout, severe attenuation, or multipath distortion in:
- **Underground tunnels and subterranean underpasses** (e.g., highway mountain tunnels, metro tunnels)
- **Multi-level enclosed parking garages**
- **Dense urban canyons** (high-rise skyscrapers reflecting signals, causing tens of meters of multipath error)
- **Dense forest canopies and deep valley gorges**
- **Electronic warfare environments** (tactical GNSS jamming and spoofing attacks)

### The Ground Reality of Indian Vehicles
The vast majority of commercial logistics trucks, older passenger automobiles, three-wheelers (auto-rickshaws), and two-wheelers operating in India lack factory-installed, high-grade Inertial Navigation Systems (INS) or direct Controller Area Network (CAN) / OBD-II wheel-speed encoder interfaces. Drivers, delivery fleets, and emergency dispatchers rely almost exclusively on dashboard-mounted or windshield-cradled consumer smartphones.

### The Physics Failure: Quadratic Drift ($t^2$)
When GNSS signal is lost, traditional mobile navigation applications either freeze the vehicle marker at the last known position or attempt **naive Dead Reckoning (DR)** by integrating linear accelerometer signals twice:
$$\mathbf{v}(t) = \mathbf{v}_0 + \int_0^t \mathbf{a}(\tau) \, d\tau$$
$$\Delta \mathbf{s}(t) = \int_0^t \mathbf{v}(\tau) \, d\tau = \mathbf{v}_0 t + \iint_0^t \mathbf{a}(\tau) \, d\tau^2$$

Consumer MEMS IMUs (Inertial Measurement Units) contain intrinsic thermal noise, non-zero bias instability, and gravity leakage. In naive double-integration:
1. Acceleration bias integrated once creates **linearly accumulating velocity error** ($E_v \propto t$).
2. Integrated a second time, it creates **quadratically exploding position error** ($E_p \propto t^2$).

Within 10 to 15 seconds of entering a tunnel, a naive double-integrating system accumulates hundreds of meters of drift, placing the vehicle icon in rivers, opposing highway lanes, or building interiors.

---

## 2. Official ISRO Mandate & Target Metrics

Under Problem Statement 26168, ISRO Department of Space specifies the following core engineering and performance criteria:

### 2.1 Core Metric Target
$$\text{Drift Error} < 10\% \text{ of Total Distance Traveled during GNSS Blackout}$$
- **Concrete Benchmarks:**
  - In short blackouts ($50\text{ m}$ traveled in under $1\text{ min}$): Drift must remain $< 5.0\text{ m}$.
  - In sustained blackouts ($1.0\text{ km}$ highway tunnel at $60\text{ km/h}$ over $60\text{ s}$): Drift must remain $< 100.0\text{ m}$.

### 2.2 Operational Hardware Constraints
- **Update Frequency:** Must maintain a continuous update rate of $\ge 10\text{ Hz}$ ($100\text{ ms}$ epoch cycle).
- **Execution Target:** Must run client-side on mobile / edge consumer hardware without cloud server round-trips.
- **Sensor Modality:** Must utilize standard consumer smartphone / vehicle MEMS sensors (3-axis accelerometer, 3-axis gyroscope, 3-axis magnetometer, and GNSS receiver).

---

## 3. Product Vision & Solution Paradigm

The **AI-ML Intelligent Dead Reckoning (IDR)** system replaces naive acceleration double-integration with a **Machine Learning Virtual Speed Sensor** coupled with kinematic constraints, zero-velocity energy gating, and independent road-network map-matching.

### 3.1 Single Integration Paradigm ($t^2 \to t$)
Instead of integrating acceleration twice, our pipeline uses a trained **XGBoost Decision Tree Regressor** that analyzes high-frequency statistical chassis vibrations and kinetic energy signatures transmitted through the vehicle chassis to predict forward ground velocity ($v_{\text{pred}}$) directly:
$$\mathbf{p}_{k+1} = \mathbf{p}_k + v_{\text{pred}, k} \cdot \begin{bmatrix} \sin \psi_k \\ \cos \psi_k \end{bmatrix} \Delta t$$
By predicting speed directly, the dead reckoning formulation becomes a **single integration of velocity**, reducing position error growth from quadratic ($t^2$) to linear ($t$).

### 3.2 The 6 Official Solution Components
As prescribed in the official technical brief, the system implements six tightly integrated architectural components:

1. **Component 1: In-Vehicle Alignment & Calibration Engine**
   - Estimates gravity vector decomposition from stationary accelerometer readings to determine phone tilt (pitch $\theta$ and roll $\phi$).
   - Aligns smartphone coordinate frame with the vehicle body frame using GNSS motion velocity vectors prior to signal loss.
2. **Component 2: AI Speed & Vibration Filter**
   - Computes rolling statistical features (standard deviation, mean, kinetic energy) over a 1-second sliding window.
   - Predicts instantaneous vehicle speed via an edge-optimized XGBoost regressor.
   - Enforces a Zero Velocity Update (ZUPT) stationary energy gate to clamp speed to $0.0\text{ m/s}$ during stops.
3. **Component 3: Map-Matching & Kinematic Constraints**
   - Applies Non-Holonomic Constraints (NHC) assuming zero lateral skid ($v_y \approx 0$) and zero vertical lift ($v_z \approx 0$).
   - Snaps unconstrained inertial coordinates to genuine OpenStreetMap (OSM) road vector centerlines using orthogonal polyline projection.
4. **Component 4: GNSS + INS Kinematic Fusion Engine**
   - Propagates multi-track state vectors ($[x, y, v, \psi]^T$) combining GNSS fixes when available with IMU/AI estimates during outages.
5. **Component 5: Seamless GNSS-Deficit Handler**
   - Detects GNSS outages within $< 100\text{ ms}$ and transitions smoothly from satellite positioning to AI Dead Reckoning.
   - Reacquires GNSS upon signal recovery with covariance convergence, eliminating visual or mathematical position teleportation.
6. **Component 6: Real-Time Navigation UI & Benchmarking Suite**
   - Interactive multi-track Leaflet mapping displaying Ground Truth, Raw DR, Classical Kalman Filter, and AI-ML Fused trajectories.
   - Real-time drift readout, compliance badges, dynamic error ellipse, and a live smartphone sensor testing sandbox.

---

## 4. User Personas & Primary Use Cases

### 4.1 Target Personas
| Persona | Role | Primary Need | Operating Environment |
|---|---|---|---|
| **Commercial Fleet Driver** | Long-haul logistics & freight | Continuous route tracking through tunnels, underpasses, and ghat roads | Heavy trucks, highway mountain passes, poor satellite visibility |
| **Urban Delivery Agent** | Last-mile courier (2W / 3W) | High-precision navigation inside dense skyscraper corridors and basements | Congested city streets, subterranean delivery hubs |
| **Emergency Responder** | Ambulance / Fire driver | Uninterrupted positioning under high-speed emergency response | Urban underpasses, flyovers, signal shadow zones |
| **Technical Evaluator / Judge** | SIH / ISRO Technical Committee | Scientifically honest verification of ISRO $< 10\%$ criteria with reproducible data | Web browser, mobile device, benchmark replay |

### 4.2 Key Use Cases
1. **UC-1: Tunnel Navigation Continuity:** Vehicle enters a 900-meter highway tunnel (40s blackout at 80 km/h). System detects GNSS loss, executes AI Dead Reckoning, and maintains vehicle position within 10% drift until exit.
2. **UC-2: Urban Flyover / Underpass Shadow:** Vehicle passes beneath a double-decker flyover for 15 seconds. Heading and forward velocity are preserved without marker freezing.
3. **UC-3: Congestion Standstill (ZUPT):** Vehicle stops in bumper-to-bumper tunnel traffic for 60 seconds. Engine idle vibration does not cause phantom forward movement.
4. **UC-4: Post-Tunnel GNSS Reacquisition:** Vehicle exits tunnel. GNSS signal locks on with low Dilution of Precision (DOP). System smoothly aligns the state vector without jumping across map tiles.

---

## 5. Functional Requirements (FR)

### Module 1: Sensor Ingestion & Preprocessing
- **FR-1.1:** System shall ingest 3-axis accelerometer ($a_x, a_y, a_z$), 3-axis gyroscope ($\omega_x, \omega_y, \omega_z$), 3-axis magnetometer, and GNSS position/speed at $10\text{ Hz}$.
- **FR-1.2:** System shall compute running statistical features over a 10-sample ($1.0\text{ s}$) rolling FIFO window.
- **FR-1.3:** Extracted features must strictly comprise: `accel_z_std`, `gyro_z_std`, `accel_energy`, `accel_y_mean`, and `accel_y_std`.

### Module 2: AI Virtual Speed Sensor
- **FR-2.1:** System shall infer instantaneous vehicle forward velocity using an ensemble of 100 gradient boosted decision trees.
- **FR-2.2:** In pure TypeScript on the client side, tree evaluation must complete in $< 1.0\text{ ms}$ per sample.
- **FR-2.3:** Output velocity must be clamped to non-negative values ($\max(0, v_{\text{pred}})$).

### Module 3: Zero Velocity Update (ZUPT)
- **FR-3.1:** System shall compute standard deviation of vertical acceleration $\sigma(a_z)$ and angular yaw rate $\sigma(\omega_z)$ over the current window.
- **FR-3.2:** If $\sigma(a_z) < 0.20\text{ m/s}^2$ AND $\sigma(\omega_z) < 0.02\text{ rad/s}$, system shall override model prediction and force $v_{\text{pred}} = 0.0\text{ m/s}$.

### Module 4: State Propagation & Kinematics
- **FR-4.1:** Heading shall be propagated via trapezoidal integration of calibrated gyroscope yaw rate:
  $$\psi_{k+1} = \psi_k + \omega_{z, k} \Delta t$$
- **FR-4.2:** Position coordinates shall be propagated in local East-North-Up (ENU) or geodetic coordinates via Great Circle / Haversine distance projection.
- **FR-4.3:** Classical filter baseline shall maintain a 4D state vector $[x, y, v, \psi]^T$ with Non-Holonomic Constraints.

### Module 5: OpenStreetMap Map-Matching
- **FR-5.1:** System shall load genuine OpenStreetMap GeoJSON road polylines for the evaluation corridor.
- **FR-5.2:** For each propagated coordinate, system shall compute orthogonal distance to nearest road polyline using `@turf/nearest-point-on-line`.
- **FR-5.3:** Snapping shall be toggleable via UI to allow independent evaluation of pure inertial DR vs road-matched DR.

### Module 6: Web Application & Visualization
- **FR-6.1:** Web UI shall render 4 distinct simultaneous trajectories on an interactive Leaflet map:
  - 🟢 Ground Truth (GNSS)
  - 🔴 Raw IMU Dead Reckoning (double-integration baseline)
  - 🔵 Classical Kalman Filter (NHC baseline)
  - 🟣 AI-ML Fused Dead Reckoning (XGBoost Virtual Speed)
- **FR-6.2:** System shall provide segment selection (S1 Urban Curved, S2 Highway Straight, S3a Aggressive Curves).
- **FR-6.3:** System shall compute real-time drift metrics: Total Distance Traveled ($m$), Drift Distance ($m$), Drift Percentage ($\%$), and an explicit **ISRO PASS / FAIL** badge.
- **FR-6.4:** System shall provide a Mobile Live Sensor Demo accessing W3C DeviceMotion and Geolocation APIs with on-demand GNSS blackout toggle.

---

## 6. Non-Functional Requirements (NFR)

| ID | Category | Requirement Description | Target Metric |
|---|---|---|---|
| **NFR-1** | **Inference Latency** | Execution time of full feature extraction and tree ensemble evaluation | $< 1.0\text{ ms}$ per sample (10 Hz = 100 ms budget) |
| **NFR-2** | **Memory Footprint** | Peak RAM consumption of the client web application | $< 50\text{ MB}$ total heap |
| **NFR-3** | **Network Bandwidth** | Static asset payload size for complete offline operation | $< 5.0\text{ MB}$ total bundle (including OSM vector geometries) |
| **NFR-4** | **Offline Capability** | Availability of dead reckoning core without internet or cellular connectivity | 100% offline runnable in browser service worker / local storage |
| **NFR-5** | **Update Frequency** | End-to-end sensor sampling and map update rate | $\ge 10\text{ Hz}$ ($100\text{ ms}$ refresh) |
| **NFR-6** | **Reliability** | Zero application white-screen crashes, uncaught exceptions, or memory leaks | 99.99% uptime, verified with React Error Boundary and JSDOM tests |
| **NFR-7** | **Platform Portability** | Cross-platform compatibility across modern desktop and mobile browsers | Chromium $\ge 100$, Safari $\ge 15$, Firefox $\ge 100$, Android Chrome, iOS Safari |
| **NFR-8** | **Scientific Integrity** | Absolute fidelity between reported documentation metrics and evaluated code | Zero fabricated data; 100% reproducible via `python/verify_consistency.py` |

---

## 7. Dataset Specifications: IO-VNBD

The project is trained and evaluated on the **Indian Open Vehicular Navigation Benchmark Dataset (IO-VNBD)**:
- **Total Duration:** 58.2 hours of driving data.
- **Routes:** 72 distinct driving sessions recorded across diverse road topographies, traffic densities, roundabouts, speed breakers, and highway stretches.
- **Sampling Rate:** Synchronized $10\text{ Hz}$ IMU and GPS fixes.
- **Total IMU Records:** **1,066,176 synchronized frames**.
- **Sensors Recorded:** 3-axis accelerometer ($m/s^2$), 3-axis gyroscope ($rad/s$), GPS Latitude, GPS Longitude, GPS Speed, GPS Heading, GPS Accuracy.

### Rigorous Held-Out Evaluation Protocol
To prevent data leakage, evaluation is conducted strictly across unseen routes:
- **Training Set:** 57 routes (**617,548 frames**, 57.9%).
- **Held-Out Test Set:** 15 routes (**448,628 frames**, 42.1%).
- **Split Strategy:** Grouped by Route ID to guarantee zero temporal overlap between train and test distributions.

---

## 8. Benchmark Targets vs. Empirical Reality

### 8.1 Speed Prediction Accuracy (Held-Out Test Set)
Evaluated across 448,628 unseen frames:

| Model | MAE (km/h) | RMSE (km/h) | Improvement vs Baseline |
|---|:---:|:---:|:---:|
| Constant Speed Baseline | 6.92 | 8.41 | Baseline |
| Linear Regression | 5.79 | 7.19 | +16.3% |
| **XGBoost Regressor (Ours, GPU Trained)** | **5.41** | **6.96** | **+21.8%** |

### 8.2 Drift Performance Across Blackout Durations (10s to 40s)
Evaluated via `python/12_drift_by_duration.py`:

| Segment | Blackout Duration | Road Distance | Raw IMU Drift % | AI Inertial Drift % | OSM Snapped Drift % | ISRO Standard (< 10%) | Compliance Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **S2 (Highway)** | **10s** | 895.5 m | 2.19% | **1.36%** | **1.32%** | $< 10\%$ | ✅ **PASS** |
| **S2 (Highway)** | **20s** | 904.6 m | 17.30% | **7.97%** | **7.87%** | $< 10\%$ | ✅ **PASS** |
| **S2 (Highway)** | **30s** | 904.8 m | 17.15% | **9.71%** | 10.65% | $< 10\%$ | ✅ **PASS** |
| **S2 (Highway)** | **40s** | 904.8 m | 17.93% | **9.71%** | 10.65% | $< 10\%$ | ✅ **PASS** |
| **S1 (Curved)** | 10s–40s | 160m–595m | 66.9%–80.6% | 72.6%–74.9% | 73.5%–74.9% | $< 10\%$ | ❌ FAIL (Cradle Tilt) |
| **S3a (Curved)** | 10s–40s | 86m–417m | 67.7%–112.1% | 65.9%–73.6% | 65.7%–73.7% | $< 10\%$ | ❌ FAIL (Cradle Tilt) |

### 8.3 The Physical Diagnostic of Curved Route Failures
On straight highway corridors (Segment S2), vehicle heading remains constant ($\Delta \psi \approx 0$). In this regime, drift is governed almost entirely by forward velocity estimation. Our AI virtual speed sensor achieves **1.36% to 9.71% drift**, passing the ISRO requirement without relying on map-matching.

On curved trajectories (Segments S1 and S3a), smartphone accelerometer readings indicate $a_y \approx -9.8\text{ m/s}^2$ at rest, proving the device was mounted in an upright windshield cradle ($\text{pitch} \approx -85^\circ$). In this orientation, vehicle cornering occurs around the phone's physical **X-axis** (pitch) while the gyroscope Z-axis (`gyro_z`) registers near-zero angular velocity. Open-loop yaw integration thus fails to turn, projecting the vehicle straight forward. This empirical finding establishes the necessity of 3D attitude alignment and topological HMM map-matching for production deployment.

---

## 9. Grand Finale Technical Roadmap

1. **Full 3D DCM / Quaternion Attitude Filter:** Implement full 9-axis Madgwick / EKF attitude estimation to resolve the upright cradle pitch gimbal lock before integrating heading.
2. **Offline Hidden Markov Model (HMM) Viterbi Map-Matching:** Transition from orthogonal distance snapping to a topological graph-based HMM matching road azimuth priors and emission probabilities.
3. **Embedded Edge Engine (C++ / ARM Cortex):** Port the inference and kinematic fusion pipeline to C++17 for deployment on ARM Cortex-A53 / Raspberry Pi CM4 with hardware CAN-bus integration.
4. **Deep Temporal Architecture:** Train a 1D-CNN + Bi-LSTM neural network on raw high-frequency IMU micro-vibrations across the full 58-hour IO-VNBD dataset.
5. **Background Android/iOS Service Daemons:** Wrap the solution in a native Android (Kotlin) foreground service with wake-locks to bypass mobile browser background execution throttling.
