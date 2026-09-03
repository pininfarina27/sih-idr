import { useState, useEffect, useRef } from "react";
import { MapContainer, TileLayer, Polyline, Tooltip, CircleMarker } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const EARTH_R = 6378137.0;
function addMetersToLatLon(lat: number, lon: number, dx: number, dy: number): [number, number] {
  return [
    lat  + (dy / EARTH_R) * (180.0 / Math.PI),
    lon  + (dx / (EARTH_R * Math.cos(Math.PI * lat / 180.0))) * (180.0 / Math.PI)
  ];
}

function predictSpeed(features: number[], model: any): number {
  if (!model) return 0;
  let speed = model.init;
  for (const tree of model.trees) {
    let node = tree;
    while (node.value === undefined) {
      node = features[node.feature] <= node.threshold ? node.left : node.right;
    }
    speed += model.learning_rate * node.value;
  }
  return Math.max(0, speed);
}

export default function LiveSensorDemo() {
  const [isActive, setIsActive] = useState(false);
  const [model, setModel] = useState<any>(null);
  const [status, setStatus] = useState("Waiting to start...");
  const [compassStatus, setCompassStatus] = useState("Initializing compass...");
  const [gpsAccuracy, setGpsAccuracy] = useState<number | null>(null);
  const [track, setTrack] = useState<[number, number][]>([]);
  const [mode, setMode] = useState<"driving" | "walking">("walking");
  const [isBlackout, setIsBlackout] = useState(false);
  const [fusionMode, setFusionMode] = useState<"GPS" | "AI-DR">("GPS");

  const state = useRef({
    lat: 0, lon: 0, heading: 0,
    accelY: [] as number[], accelZ: [] as number[], gyroZ: [] as number[],
    lastMotionTime: 0,
    watchId: null as number | null,
    compassFromHardware: false,
  });

  useEffect(() => {
    fetch("/data/gbr_model.json").then(r => r.json()).then(setModel);
  }, []);

  // GPS watch — continuous stream, recalibrates heading when not in blackout
  const startGpsWatch = () => {
    const watchId = navigator.geolocation.watchPosition(
      (pos) => {
        const s = state.current;
        setGpsAccuracy(Math.round(pos.coords.accuracy));
        if (!isBlackout) {
          // Not in blackout: use GPS directly and recalibrate DR baseline
          s.lat = pos.coords.latitude;
          s.lon = pos.coords.longitude;
          if (pos.coords.heading !== null && !isNaN(pos.coords.heading)) {
            s.heading = pos.coords.heading;
          }
          setFusionMode("GPS");
          setTrack(prev => [...prev, [pos.coords.latitude, pos.coords.longitude]]);
          setStatus(`GPS ACTIVE — Accuracy: ${Math.round(pos.coords.accuracy)}m`);
        }
        // If blackout simulated, ignore GPS (AI-DR continues from last known state)
      },
      (err) => setStatus(`GPS Error: ${err.message}`),
      { enableHighAccuracy: true, maximumAge: 1000 }
    );
    state.current.watchId = watchId;
  };

  const handleStart = async () => {
    if (typeof (DeviceMotionEvent as any).requestPermission === "function") {
      const perm = await (DeviceMotionEvent as any).requestPermission();
      if (perm !== "granted") { setStatus("IMU permission denied."); return; }
    }
    setStatus("Acquiring GPS...");
    startGpsWatch();
    setIsActive(true);
    window.addEventListener("devicemotion", handleMotion);
    window.addEventListener("deviceorientationabsolute", handleOrientation, true);
    window.addEventListener("deviceorientation", handleOrientation, true);
  };

  const handleStop = () => {
    setIsActive(false);
    if (state.current.watchId !== null) {
      navigator.geolocation.clearWatch(state.current.watchId);
      state.current.watchId = null;
    }
    window.removeEventListener("devicemotion", handleMotion);
    window.removeEventListener("deviceorientationabsolute", handleOrientation);
    window.removeEventListener("deviceorientation", handleOrientation);
    setStatus("Stopped. Tap Start to begin a new session.");
    setFusionMode("GPS");
  };

  const toggleBlackout = () => {
    const next = !isBlackout;
    setIsBlackout(next);
    setFusionMode(next ? "AI-DR" : "GPS");
    setStatus(next ? "BLACKOUT SIMULATED — Running AI Dead Reckoning..." : "GPS RESTORED — Recalibrating...");
  };

  const handleOrientation = (e: any) => {
    let h: number | null = null;
    if (e.webkitCompassHeading != null) {
      h = e.webkitCompassHeading; // iOS
    } else if (e.absolute && e.alpha != null) {
      h = 360 - e.alpha;           // Android
    }
    if (h !== null) {
      state.current.heading = h;
      state.current.compassFromHardware = true;
      setCompassStatus("Absolute Compass Active");
    }
  };

  const handleMotion = (e: DeviceMotionEvent) => {
    const s = state.current;
    if (!e.accelerationIncludingGravity || !e.rotationRate) return;

    s.accelY.push(e.accelerationIncludingGravity.y ?? 0);
    s.accelZ.push(e.accelerationIncludingGravity.z ?? 0);
    s.gyroZ.push((e.rotationRate.gamma ?? 0) * (Math.PI / 180.0));
    if (s.accelY.length > 10) { s.accelY.shift(); s.accelZ.shift(); s.gyroZ.shift(); }

    const now = Date.now();
    const dt = Math.min((now - (s.lastMotionTime || now)) / 1000.0, 0.5);
    s.lastMotionTime = now;

    if (s.accelY.length < 5) return;  // need a few samples

    // Only do DR when in blackout mode
    if (!isBlackout) return;

    const mean = (a: number[]) => a.reduce((x,y) => x+y, 0) / a.length;
    const std  = (a: number[], m: number) =>
      Math.sqrt(a.reduce((x,y) => x + (y-m)**2, 0) / a.length);

    const ay_mean = mean(s.accelY);
    const ay_std  = std(s.accelY, ay_mean);
    const az_std  = std(s.accelZ, mean(s.accelZ));
    const gz_std  = std(s.gyroZ,  mean(s.gyroZ));
    const energy  = mean(s.accelY.map((v, i) => v*v + s.accelZ[i]*s.accelZ[i]));

    const rawSpeed = predictSpeed([ay_mean, ay_std, az_std, gz_std, energy], model);
    const speed = mode === "walking" ? rawSpeed * 0.22 : rawSpeed;

    // Heading update only if compass not available
    if (!s.compassFromHardware) {
      s.heading -= (mean(s.gyroZ) * 180 / Math.PI) * dt;
    }

    const heading_rad = s.heading * Math.PI / 180.0;
    const [newLat, newLon] = addMetersToLatLon(
      s.lat, s.lon,
      speed * Math.sin(heading_rad) * dt,
      speed * Math.cos(heading_rad) * dt
    );
    s.lat = newLat;
    s.lon = newLon;

    setTrack(prev => [...prev, [newLat, newLon]]);
    setStatus(`AI-DR: ${(speed * 3.6).toFixed(1)} km/h | Heading: ${s.heading.toFixed(0)}°`);
  };

  return (
    <div className="w-full flex flex-col gap-4">
      {/* Control Panel */}
      <div className="bg-white border rounded-xl p-4 shadow-sm">
        <div className="flex justify-between items-start flex-wrap gap-4">
          <div>
            <h2 className="text-xl font-bold text-gray-800">Live Sensor Demo</h2>
            <p className="text-gray-500 text-sm mt-0.5">{status}</p>
            <p className="text-indigo-500 text-xs mt-0.5">{compassStatus}</p>
            {gpsAccuracy !== null && (
              <p className="text-green-600 text-xs mt-0.5">GPS Accuracy: ±{gpsAccuracy}m</p>
            )}

            {/* Mode toggle */}
            <div className="flex gap-2 mt-3">
              <button
                onClick={() => setMode("walking")}
                className={`px-3 py-1 text-xs rounded-full border transition-colors ${mode === "walking" ? "bg-indigo-100 border-indigo-500 text-indigo-700 font-medium" : "bg-gray-50 text-gray-600"}`}
              >
                🚶 Walking (illustrative scale)
              </button>
              <button
                onClick={() => setMode("driving")}
                className={`px-3 py-1 text-xs rounded-full border transition-colors ${mode === "driving" ? "bg-indigo-100 border-indigo-500 text-indigo-700 font-medium" : "bg-gray-50 text-gray-600"}`}
              >
                🚗 Driving Mode
              </button>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex flex-col gap-2 items-end">
            {!isActive ? (
              <button onClick={handleStart} className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 font-medium text-sm">
                Start Session
              </button>
            ) : (
              <button onClick={handleStop} className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 font-medium text-sm">
                Stop
              </button>
            )}
            {isActive && (
              <button
                onClick={toggleBlackout}
                className={`px-6 py-2 rounded-lg font-medium text-sm transition-colors ${
                  isBlackout
                    ? "bg-red-500 hover:bg-red-600 text-white animate-pulse"
                    : "bg-green-500 hover:bg-green-600 text-white"
                }`}
              >
                {isBlackout ? "🔴 GPS BLACKOUT (click to restore)" : "🟢 Simulate GPS Blackout"}
              </button>
            )}
          </div>
        </div>

        {/* Fusion Mode Badge */}
        {isActive && (
          <div className="mt-3">
            <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold ${
              fusionMode === "GPS" ? "bg-green-100 text-green-700 border border-green-300" : "bg-red-100 text-red-700 border border-red-300"
            }`}>
              <span className={`w-2 h-2 rounded-full ${fusionMode === "GPS" ? "bg-green-500" : "bg-red-500 animate-pulse"}`}></span>
              {fusionMode === "GPS" ? "GPS ACTIVE" : "AI DEAD RECKONING"}
            </span>
          </div>
        )}
      </div>

      {/* Map */}
      <div className="h-[500px] w-full rounded-xl overflow-hidden border border-gray-200 shadow-sm relative z-0">
        {track.length > 0 ? (
          <MapContainer center={track[0]} zoom={18} scrollWheelZoom={true} style={{ height: "100%", width: "100%" }}>
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            <Polyline positions={track} color="#8B5CF6" weight={5}>
              <Tooltip sticky>Live AI-Fused Path</Tooltip>
            </Polyline>
            {track.length > 0 && (
              <CircleMarker center={track[track.length - 1]} radius={7} fillColor="#8B5CF6" color="#fff" weight={2} fillOpacity={1} />
            )}
          </MapContainer>
        ) : (
          <div className="h-full flex flex-col items-center justify-center bg-gray-100 text-gray-400 gap-2">
            <div className="text-4xl">📱</div>
            <p className="font-medium">Open on your mobile device</p>
            <p className="text-sm">Tap Start Session, walk around, then press Simulate GPS Blackout</p>
          </div>
        )}
      </div>
    </div>
  );
}
