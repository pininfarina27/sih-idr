import { useState, useEffect } from "react";
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
    
    driftPercentage = (finalDrift / distanceTravelled) * 100;
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
          
          {/* Ground Truth - Green */}
          <Polyline positions={gtPath} color="#10B981" weight={6} opacity={0.7}>
            <Tooltip sticky>Ground Truth (GPS)</Tooltip>
          </Polyline>

          {/* Raw DR - Red */}
          <Polyline positions={rawPath} color="#EF4444" weight={4} dashArray="5, 10">
            <Tooltip sticky>Raw Dead Reckoning (Naive Integration)</Tooltip>
          </Polyline>

          {/* AI Fused - Purple */}
          <Polyline positions={aiPath} color="#8B5CF6" weight={6}>
            <Tooltip sticky>AI-ML Fused (GBR)</Tooltip>
          </Polyline>

          {/* Fused - Blue */}
          <Polyline positions={fusedPath} color="#3B82F6" weight={4} opacity={0.5}>
            <Tooltip sticky>Classical Fused (EKF)</Tooltip>
          </Polyline>
          
          {/* Start/End Markers */}
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

export default function App() {
  const [activeTab, setActiveTab] = useState<"replay" | "live">("replay");

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-indigo-600 text-white p-4 shadow-md">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-bold tracking-tight">AI-ML IDR Prototype</h1>
          <div className="text-sm bg-indigo-800 px-3 py-1 rounded-full opacity-80">
            ISRO PS26168 - IO-VNBD Dataset
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
        {activeTab === "live" && (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-8 border-2 border-dashed border-gray-300 rounded-xl bg-gray-50">
            <h2 className="text-2xl font-bold text-gray-700 mb-2">Live Sensor Demo</h2>
            <p className="text-gray-500 max-w-md">
              Coming in Phase 3. Will use your device's IMU to perform inference locally.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
