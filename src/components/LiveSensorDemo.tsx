import { useState, useEffect, useRef } from "react";
import { MapContainer, TileLayer, Polyline, Tooltip, CircleMarker } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const R = 6378137.0;
function addMetersToLatLon(lat: number, lon: number, dx: number, dy: number) {
  const d_lat = dy / R;
  const d_lon = dx / (R * Math.cos(Math.PI * lat / 180.0));
  return [
    lat + (d_lat * 180.0 / Math.PI),
    lon + (d_lon * 180.0 / Math.PI)
  ];
}

function predictSpeed(features: number[], model: any) {
  if (!model) return 0;
  let speed = model.init;
  for (const tree of model.trees) {
    let node = tree;
    while (node.value === undefined) {
      if (features[node.feature] <= node.threshold) node = node.left;
      else node = node.right;
    }
    speed += model.learning_rate * node.value;
  }
  return speed;
}

export default function LiveSensorDemo() {
  const [isActive, setIsActive] = useState(false);
  const [model, setModel] = useState<any>(null);
  const [status, setStatus] = useState("Waiting to start...");
  const [track, setTrack] = useState<[number, number][]>([]);
  
  const [mode, setMode] = useState<"driving"|"walking">("walking");
  const [headingMode, setHeadingMode] = useState("Initializing compass...");
  
  const state = useRef({
    lat: 0, lon: 0, heading: 0, 
    accelY: [] as number[], accelZ: [] as number[], gyroZ: [] as number[],
    lastTime: 0
  });

  useEffect(() => {
    fetch("/data/gbr_model.json").then(r => r.json()).then(setModel);
  }, []);

  const handleStart = async () => {
    if (typeof (DeviceMotionEvent as any).requestPermission === "function") {
      const permission = await (DeviceMotionEvent as any).requestPermission();
      if (permission !== "granted") {
        setStatus("Permission to access IMU denied.");
        return;
      }
    }
    
    setStatus("Acquiring GPS fix...");
    navigator.geolocation.getCurrentPosition((pos) => {
      state.current.lat = pos.coords.latitude;
      state.current.lon = pos.coords.longitude;
      state.current.heading = pos.coords.heading || 0;
      setTrack([[pos.coords.latitude, pos.coords.longitude]]);
      
      setStatus("GPS Acquired. Starting AI Fusion...");
      setIsActive(true);
      window.addEventListener("devicemotion", handleMotion);
      window.addEventListener("deviceorientationabsolute", handleOrientation);
      window.addEventListener("deviceorientation", handleOrientation);
    }, (err) => {
      setStatus(`GPS Error: ${err.message}. Ensure location is enabled.`);
    }, { enableHighAccuracy: true });
  };

  const handleStop = () => {
    setIsActive(false);
    window.removeEventListener("devicemotion", handleMotion);
    window.removeEventListener("deviceorientationabsolute", handleOrientation);
    window.removeEventListener("deviceorientation", handleOrientation);
    setStatus("Stopped.");
  };

  const handleOrientation = (e: any) => {
    let h = null;
    if (e.webkitCompassHeading) {
      h = e.webkitCompassHeading;
    } else if (e.absolute && e.alpha !== null) {
      h = 360 - e.alpha;
    }
    if (h !== null) {
      state.current.heading = h;
      setHeadingMode("Compass Active (Absolute)");
    }
  };

  const handleMotion = (e: DeviceMotionEvent) => {
    const s = state.current;
    if (!e.accelerationIncludingGravity || !e.rotationRate) return;
    
    s.accelY.push(e.accelerationIncludingGravity.y || 0);
    s.accelZ.push(e.accelerationIncludingGravity.z || 0);
    s.gyroZ.push((e.rotationRate.gamma || 0) * (Math.PI / 180.0));
    
    if (s.accelY.length > 10) {
      s.accelY.shift(); s.accelZ.shift(); s.gyroZ.shift();
    }
    
    const now = Date.now();
    if (!s.lastTime) s.lastTime = now;
    const dt = (now - s.lastTime) / 1000.0;
    
    if (dt >= 0.1 && s.accelY.length === 10) {
      s.lastTime = now;
      
      const mean = (arr: number[]) => arr.reduce((a,b)=>a+b,0)/arr.length;
      const std = (arr: number[], m: number) => Math.sqrt(arr.reduce((a,b)=>a+Math.pow(b-m,2),0)/arr.length);
      
      const ay_mean = mean(s.accelY);
      const ay_std = std(s.accelY, ay_mean);
      const az_std = std(s.accelZ, mean(s.accelZ));
      const gz_std = std(s.gyroZ, mean(s.gyroZ));
      const energy = mean(s.accelY.map((v,i) => v*v + s.accelZ[i]*s.accelZ[i]));
      
      const features = [ay_mean, ay_std, az_std, gz_std, energy];
      const speed = predictSpeed(features, model);
      
      if (headingMode.includes("Initializing")) {
        const avgGyroZ = mean(s.gyroZ);
        s.heading += (avgGyroZ * 180 / Math.PI) * dt;
      }
      
      let adjustedSpeed = speed;
      if (mode === "walking") {
         adjustedSpeed = Math.min(speed * 0.15, 2.0);
      }
      
      const dx = adjustedSpeed * Math.sin(s.heading * Math.PI / 180) * dt;
      const dy = adjustedSpeed * Math.cos(s.heading * Math.PI / 180) * dt;
      
      const [newLat, newLon] = addMetersToLatLon(s.lat, s.lon, dx, dy);
      s.lat = newLat;
      s.lon = newLon;
      
      setTrack(prev => [...prev, [newLat, newLon]]);
      setStatus(`AI Fusion Running... Speed: ${(adjustedSpeed*3.6).toFixed(1)} km/h`);
    }
  };

  return (
    <div className="w-full flex flex-col gap-4">
      <div className="flex justify-between items-center bg-white p-4 rounded-xl border shadow-sm">
        <div>
          <h2 className="text-xl font-bold">Live Sensor Demo (Mobile Only)</h2>
          <p className="text-gray-500 text-sm">{status}</p>
          <p className="text-indigo-500 text-xs mt-1">{headingMode}</p>
          
          <div className="mt-3 flex gap-2">
            <button onClick={() => setMode("walking")} className={`px-3 py-1 text-xs rounded-full border ${mode === "walking" ? "bg-indigo-100 border-indigo-500 text-indigo-700" : "bg-gray-50"}`}>🚶 Walking Mode</button>
            <button onClick={() => setMode("driving")} className={`px-3 py-1 text-xs rounded-full border ${mode === "driving" ? "bg-indigo-100 border-indigo-500 text-indigo-700" : "bg-gray-50"}`}>🚗 Driving Mode</button>
          </div>
        </div>
        <div>
          {!isActive ? (
            <button onClick={handleStart} className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 font-medium">Start Fusion</button>
          ) : (
            <button onClick={handleStop} className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 font-medium">Stop</button>
          )}
        </div>
      </div>
      
      <div className="h-[500px] w-full rounded-xl overflow-hidden border border-gray-200 shadow-sm relative z-0">
        {track.length > 0 ? (
          <MapContainer center={track[0]} zoom={18} scrollWheelZoom={true} style={{ height: "100%", width: "100%" }}>
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            <Polyline positions={track} color="#8B5CF6" weight={6}>
              <Tooltip sticky>Live AI-Fused Path</Tooltip>
            </Polyline>
            <CircleMarker center={track[track.length-1]} radius={6} fillColor="#8B5CF6" color="#fff" weight={2} fillOpacity={1} />
          </MapContainer>
        ) : (
          <div className="h-full flex items-center justify-center bg-gray-100 text-gray-400">
            Click Start on your mobile device to begin mapping...
          </div>
        )}
      </div>
    </div>
  );
}
