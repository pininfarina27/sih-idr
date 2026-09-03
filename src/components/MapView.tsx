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

export default function MapView({ segmentId }: { segmentId: string }) {
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

  const gtPath: [number, number][] = gt.map(p => [p.lat, p.lon]);
  const rawPath: [number, number][] = raw.map(p => [p.lat, p.lon]);
  const fusedPath: [number, number][] = fused.map(p => [p.lat, p.lon]);
  const aiPath: [number, number][] = aiFused.map(p => [p.lat, p.lon]);

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

      <div className="h-[600px] w-full rounded-xl overflow-hidden border border-gray-200 shadow-sm relative z-0">
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
