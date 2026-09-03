import { useState, useEffect, useRef } from "react";
import { MapContainer, TileLayer, Polyline, Tooltip, CircleMarker } from "react-leaflet";
import * as turf from "@turf/turf";
import "leaflet/dist/leaflet.css";

// Hack to fix Leaflet marker icons in React
import L from "leaflet";
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

interface Point { ts: number; lat: number; lon: number; speed_kmh: number; heading: number; }
interface SegmentMeta { id: string; name: string; duration: number; blackout_start_ts: number; blackout_end_ts: number; }

function MapView({ segmentId }: { segmentId: string }) {
  const [gt, setGt] = useState<Point[]>([]);
  const [raw, setRaw] = useState<Point[]>([]);
  const [fused, setFused] = useState<Point[]>([]);
  const [aiFused, setAiFused] = useState<Point[]>([]);

  useEffect(() => {
    fetch(`/data/segment_${segmentId}_gt.json`).then(r => r.json()).then(setGt);
    fetch(`/data/segment_${segmentId}_raw_dr.json`).then(r => r.json()).then(setRaw);
    fetch(`/data/segment_${segmentId}_fused.json`).then(r => r.json()).then(setFused);
    fetch(`/data/segment_${segmentId}_ai_fused.json`).then(r => r.json()).then(setAiFused);
  }, [segmentId]);

  if (!gt.length) return <div className="p-8 text-center text-gray-500">Loading tracking data...</div>;

  const center: [number, number] = [gt[0].lat, gt[0].lon];

  // Convert to leaflet format
  const gtPath: [number, number][] = gt.map(p => [p.lat, p.lon]);
  const rawPath: [number, number][] = raw.map(p => [p.lat, p.lon]);
  const fusedPath: [number, number][] = fused.map(p => [p.lat, p.lon]);
  const aiPath: [number, number][] = aiFused.map(p => [p.lat, p.lon]);

  // Calculate Drift Stats
  let distanceTravelled = 0;
  let finalDrift = 0;
  let driftPercentage = 0;
  
  if (gt.length > 0 && aiFused.length > 0) {
    const endPt = turf.point([gt[gt.length - 1].lon, gt[gt.length - 1].lat]);
    
    const line = turf.lineString(gt.map(p => [p.lon, p.lat]));
    distanceTravelled = turf.length(line, {units: 'meters'});
    
    const aiEndPt = turf.point([aiFused[aiFused.length - 1].lon, aiFused[aiFused.length - 1].lat]);
    finalDrift = turf.distance(endPt, aiEndPt, {units: 'meters'});
    
    driftPercentage = distanceTravelled > 0 ? (finalDrift / distanceTravelled) * 100 : 0;
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white border rounded-lg p-4 shadow-sm">
          <div className="text-sm text-gray-500 font-medium">Distance Travelled</div>
          <div className="text-2xl font-bold text-gray-800">{distanceTravelled.toFixed(1)} m</div>
        </div>
        <div className="bg-white border rounded-lg p-4 shadow-sm">
          <div className="text-sm text-gray-500 font-medium">Final AI Drift Error</div>
          <div className="text-2xl font-bold text-gray-800">{finalDrift.toFixed(1)} m</div>
        </div>
        <div className={`border rounded-lg p-4 shadow-sm ${driftPercentage < 10 ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
          <div className="text-sm text-gray-700 font-medium">Drift % (ISRO Target &lt; 10%)</div>
          <div className={`text-2xl font-bold ${driftPercentage < 10 ? 'text-green-700' : 'text-red-700'}`}>
            {driftPercentage.toFixed(2)}%
          </div>
        </div>
      </div>

      <div className="h-[600px] w-full rounded-xl overflow-hidden border border-gray-200 shadow-sm relative">
        <MapContainer center={center} zoom={16} scrollWheelZoom={true} style={{ height: "100%", width: "100%" }}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          
          <Polyline positions={gtPath} color="#10B981" weight={6} opacity={0.7}>
            <Tooltip sticky>Ground Truth (GPS)</Tooltip>
          </Polyline>

          <Polyline positions={rawPath} color="#EF4444" weight={4} dashArray="5, 10">
            <Tooltip sticky>Raw Dead Reckoning (Naive Integration)</Tooltip>
          </Polyline>

          <Polyline positions={aiPath} color="#8B5CF6" weight={6}>
            <Tooltip sticky>AI-ML Fused (GBR)</Tooltip>
          </Polyline>

          <Polyline positions={fusedPath} color="#3B82F6" weight={4} opacity={0.5}>
            <Tooltip sticky>Classical Fused (EKF)</Tooltip>
          </Polyline>
          
          <CircleMarker center={gtPath[0]} radius={8} fillColor="#10B981" color="#fff" weight={2} fillOpacity={1}>
             <Tooltip>Start Point</Tooltip>
          </CircleMarker>
        </MapContainer>
        
        <div className="absolute top-4 right-4 bg-white/90 backdrop-blur p-4 rounded-lg shadow-lg border border-gray-200 z-[400] text-sm">
          <h3 className="font-bold mb-2">Legend</h3>
          <div className="flex items-center gap-2 mb-1"><div className="w-4 h-1 bg-[#10B981]"></div> Ground Truth</div>
          <div className="flex items-center gap-2 mb-1"><div className="w-4 h-1 bg-[#EF4444] border-t border-dashed border-[#EF4444] bg-transparent"></div> Raw DR (Drift)</div>
          <div className="flex items-center gap-2 mb-1"><div className="w-4 h-1 bg-[#3B82F6]"></div> Classical Fused (EKF)</div>
          <div className="flex items-center gap-2"><div className="w-4 h-1 bg-[#8B5CF6]"></div> AI-ML Fused (GBR)</div>
        </div>
      </div>
    </div>
  );
}

function BenchmarkReplay() {
  const [meta, setMeta] = useState<{segments: SegmentMeta[]}>({segments: []});
  const [activeSeg, setActiveSeg] = useState<string>("S1");

  useEffect(() => {
    fetch("/data/segments.json").then(r => r.json()).then(setMeta);
  }, []);

  return (
    <div className="w-full flex flex-col gap-4">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Benchmark Replay</h2>
          <p className="text-gray-500">Visualizing Phase 1 Baseline against Ground Truth.</p>
        </div>
        <select 
          className="bg-white border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 p-2.5"
          value={activeSeg}
          onChange={(e) => setActiveSeg(e.target.value)}
        >
          {meta.segments.map(s => (
            <option key={s.id} value={s.id}>{s.name} ({s.duration}s)</option>
          ))}
        </select>
      </div>
      
      <MapView segmentId={activeSeg} />
    </div>
  );
}

// Live Sensor Demo Components
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

function LiveSensorDemo() {
  const [isActive, setIsActive] = useState(false);
  const [model, setModel] = useState<any>(null);
  const [status, setStatus] = useState("Waiting to start...");
  const [track, setTrack] = useState<[number, number][]>([]);
  const [mode, setMode] = useState<'driving'|'walking'>('walking');
  const [headingMode, setHeadingMode] = useState('Initializing compass...');
  
  const state = useRef({
    lat: 0, lon: 0, heading: 0, 
    accelY: [] as number[], accelZ: [] as number[], gyroZ: [] as number[],
    lastTime: 0
  });

  useEffect(() => {
    fetch('/data/gbr_model.json').then(r => r.json()).then(setModel);
  }, []);

  const handleStart = async () => {
    if (typeof (DeviceMotionEvent as any).requestPermission === 'function') {
      const permission = await (DeviceMotionEvent as any).requestPermission();
      if (permission !== 'granted') {
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
      window.addEventListener('devicemotion', handleMotion);
      window.addEventListener('deviceorientationabsolute', handleOrientation);
      window.addEventListener('deviceorientation', handleOrientation);
    }, (err) => {
      setStatus(`GPS Error: ${err.message}. Ensure location is enabled.`);
    }, { enableHighAccuracy: true });
  };

  const handleStop = () => {
    setIsActive(false);
    window.removeEventListener('devicemotion', handleMotion);
    window.removeEventListener('deviceorientationabsolute', handleOrientation);
    window.removeEventListener('deviceorientation', handleOrientation);
    setStatus("Stopped.");
  };

  const handleOrientation = (e: any) => {
    let h = null;
    if (e.webkitCompassHeading) {
      h = e.webkitCompassHeading; // iOS
    } else if (e.absolute && e.alpha !== null) {
      h = 360 - e.alpha; // Android
    }
    if (h !== null) {
      state.current.heading = h;
      setHeadingMode("Compass Active (Absolute)");
    }
  };

  const handleMotion = (e: DeviceMotionEvent) => {
    const s = state.current;
    if (!e.accelerationIncludingGravity || !e.rotationRate) return;
    
    // Very naive approximation of linear accel (just relying on ML to filter out gravity bias)
    s.accelY.push(e.accelerationIncludingGravity.y || 0);
    s.accelZ.push(e.accelerationIncludingGravity.z || 0);
    s.gyroZ.push((e.rotationRate.gamma || 0) * (Math.PI / 180.0)); // degrees/s to rad/s
    
    if (s.accelY.length > 10) {
      s.accelY.shift(); s.accelZ.shift(); s.gyroZ.shift();
    }
    
    const now = Date.now();
    if (!s.lastTime) s.lastTime = now;
    const dt = (now - s.lastTime) / 1000.0;
    
    if (dt >= 0.1 && s.accelY.length === 10) {
      s.lastTime = now;
      
      // Calculate features
      const mean = (arr: number[]) => arr.reduce((a,b)=>a+b,0)/arr.length;
      const std = (arr: number[], m: number) => Math.sqrt(arr.reduce((a,b)=>a+Math.pow(b-m,2),0)/arr.length);
      
      const ay_mean = mean(s.accelY);
      const ay_std = std(s.accelY, ay_mean);
      const az_std = std(s.accelZ, mean(s.accelZ));
      const gz_std = std(s.gyroZ, mean(s.gyroZ));
      const energy = mean(s.accelY.map((v,i) => v*v + s.accelZ[i]*s.accelZ[i]));
      
      const features = [ay_mean, ay_std, az_std, gz_std, energy];
      const speed = predictSpeed(features, model);
      
      // Note: heading is now updated by the compass! If compass failed, we fallback to gyro
      if (headingMode.includes('Initializing')) {
        const avgGyroZ = mean(s.gyroZ);
        s.heading += (avgGyroZ * 180 / Math.PI) * dt;
      }
      
      // If walking mode, scale down the car-trained ML speed to pedestrian speeds (~1-2 m/s)
      let adjustedSpeed = speed;
      if (mode === 'walking') {
         adjustedSpeed = Math.min(speed * 0.15, 2.0); // dampen the car model for foot steps
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
            <button onClick={() => setMode('walking')} className={`px-3 py-1 text-xs rounded-full border ${mode === 'walking' ? 'bg-indigo-100 border-indigo-500 text-indigo-700' : 'bg-gray-50'}`}>🚶 Walking Mode</button>
            <button onClick={() => setMode('driving')} className={`px-3 py-1 text-xs rounded-full border ${mode === 'driving' ? 'bg-indigo-100 border-indigo-500 text-indigo-700' : 'bg-gray-50'}`}>🚗 Driving Mode</button>
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
      
      <div className="h-[500px] w-full rounded-xl overflow-hidden border border-gray-200 shadow-sm relative">
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

export default function App() {
  const [activeTab, setActiveTab] = useState<"replay" | "live">("replay");

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-indigo-600 text-white p-4 shadow-md">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-bold tracking-tight">AI-ML IDR Prototype</h1>
          <div className="text-sm bg-indigo-800 px-3 py-1 rounded-full opacity-80">
            ISRO PS26168
          </div>
        </div>
      </header>
      
      <div className="bg-white border-b shadow-sm">
        <div className="max-w-7xl mx-auto flex gap-4 px-4">
          <button 
            className={`py-3 px-4 font-medium text-sm transition-colors ${
              activeTab === "replay" ? "border-b-2 border-indigo-600 text-indigo-600" : "text-gray-500 hover:text-gray-800"
            }`}
            onClick={() => setActiveTab("replay")}
          >
            Benchmark Replay
          </button>
          <button 
            className={`py-3 px-4 font-medium text-sm transition-colors ${
              activeTab === "live" ? "border-b-2 border-indigo-600 text-indigo-600" : "text-gray-500 hover:text-gray-800"
            }`}
            onClick={() => setActiveTab("live")}
          >
            Live Sensor Demo
          </button>
        </div>
      </div>

      <main className="flex-1 max-w-7xl mx-auto w-full p-4 flex flex-col">
        {activeTab === "replay" && <BenchmarkReplay />}
        {activeTab === "live" && <LiveSensorDemo />}
      </main>
    </div>
  );
}
