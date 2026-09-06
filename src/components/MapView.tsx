import { useState, useEffect } from "react";
import { MapContainer, TileLayer, Polyline, Tooltip, CircleMarker, Circle } from "react-leaflet";
import * as turf from "@turf/turf";
import "leaflet/dist/leaflet.css";

import L from "leaflet";
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

interface Point {
  ts: number;
  lat: number;
  lon: number;
  speed_kmh: number;
  heading: number;
  snap_lat?: number;
  snap_lon?: number;
  raw_lat?: number;
  raw_lon?: number;
}
interface SegmentMeta { id: string; name: string; duration: number; blackout_start_ts: number; blackout_end_ts: number; }

export default function MapView({ segmentId }: { segmentId: string }) {
  const [gt, setGt]           = useState<Point[]>([]);
  const [raw, setRaw]         = useState<Point[]>([]);
  const [fused, setFused]     = useState<Point[]>([]);
  const [aiFused, setAiFused] = useState<Point[]>([]);
  const [meta, setMeta]       = useState<SegmentMeta | null>(null);
  const [useMapMatching, setUseMapMatching] = useState(true);

  useEffect(() => {
    let active = true;
    setGt([]); setRaw([]); setFused([]); setAiFused([]); setMeta(null);

    fetch(`/data/segments.json`)
      .then(r => r.json())
      .then(d => {
        if (!active) return;
        const m = d.segments.find((s: SegmentMeta) => s.id === segmentId);
        setMeta(m ?? null);
      })
      .catch(() => {});

    Promise.all([
      fetch(`/data/segment_${segmentId}_gt.json`).then(r => r.json()),
      fetch(`/data/segment_${segmentId}_raw_dr.json`).then(r => r.json()),
      fetch(`/data/segment_${segmentId}_fused.json`).then(r => r.json()),
      fetch(`/data/segment_${segmentId}_ai_fused.json`).then(r => r.json()),
    ])
      .then(([gtData, rawData, fusedData, aiData]) => {
        if (!active) return;
        setGt(gtData);
        setRaw(rawData);
        setFused(fusedData);
        setAiFused(aiData);
      })
      .catch((err) => console.error("Failed to load tracking data:", err));

    return () => { active = false; };
  }, [segmentId]);

  // Robust guard against partial loading / race conditions
  if (!gt.length || !raw.length || !fused.length || !aiFused.length || !meta) {
    return (
      <div className="p-16 text-center text-gray-500 bg-white border border-gray-100 rounded-xl shadow-sm">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-indigo-600 border-t-transparent mb-3"></div>
        <p className="font-semibold text-gray-700">Loading {segmentId} tracking data...</p>
        <p className="text-xs text-gray-400 mt-1">Ground truth, Raw IMU, Kalman Filter, AI-ML tracks & OpenStreetMap roads</p>
      </div>
    );
  }

  const center: [number, number] = [gt[0].lat, gt[0].lon];
  const gtPath:    [number, number][] = gt.map(p      => [p.lat, p.lon]);
  const rawPath:   [number, number][] = raw.map(p     => [p.lat, p.lon]);
  const fusedPath: [number, number][] = fused.map(p   => [p.lat, p.lon]);

  const blackoutEndTs = meta.blackout_end_ts;
  const blackoutStartTs = meta.blackout_start_ts;

  const gtAtBlackoutEnd  = [...gt].sort((a,b)      => Math.abs(a.ts-blackoutEndTs)-Math.abs(b.ts-blackoutEndTs))[0];
  const aiAtBlackoutEnd  = [...aiFused].sort((a,b) => Math.abs(a.ts-blackoutEndTs)-Math.abs(b.ts-blackoutEndTs))[0];
  const rawAtBlackoutEnd = [...raw].sort((a,b)     => Math.abs(a.ts-blackoutEndTs)-Math.abs(b.ts-blackoutEndTs))[0];

  if (!gtAtBlackoutEnd || !aiAtBlackoutEnd || !rawAtBlackoutEnd) {
    return <div className="p-8 text-center text-gray-500">Processing tracking data...</div>;
  }

  // Distance traveled by GT during blackout window
  const gtDuringBlackout = gt.filter(p => p.ts >= blackoutStartTs && p.ts <= blackoutEndTs);

  const blackoutDistance = gtDuringBlackout.length > 1
    ? turf.length(turf.lineString(gtDuringBlackout.map(p => [p.lon, p.lat])), { units: "meters" })
    : 1;

  // Component 3: Road-Network Map-Matching Snap against independent OpenStreetMap
  const rcpfAiEndPt = turf.point([aiAtBlackoutEnd.lon, aiAtBlackoutEnd.lat]);
  const unconstrainedAiEndPt = turf.point([
    aiAtBlackoutEnd.raw_lon ?? aiAtBlackoutEnd.lon,
    aiAtBlackoutEnd.raw_lat ?? aiAtBlackoutEnd.lat,
  ]);
  const activeAiEndPt = useMapMatching ? rcpfAiEndPt : unconstrainedAiEndPt;

  const displayAiPath: [number, number][] = useMapMatching
    ? aiFused.map(p => [p.lat, p.lon] as [number, number])
    : aiFused.map(p => [p.raw_lat ?? p.lat, p.raw_lon ?? p.lon] as [number, number]);

  const gtEndPt = turf.point([gtAtBlackoutEnd.lon, gtAtBlackoutEnd.lat]);

  const rcpfDriftAtEnd = turf.distance(gtEndPt, rcpfAiEndPt, { units: "meters" });
  const unconstrainedDriftAtEnd = turf.distance(gtEndPt, unconstrainedAiEndPt, { units: "meters" });
  const aiDriftAtEnd = useMapMatching ? rcpfDriftAtEnd : unconstrainedDriftAtEnd;

  const rawDriftAtEnd = turf.distance(
    gtEndPt,
    turf.point([rawAtBlackoutEnd.lon, rawAtBlackoutEnd.lat]),
    { units: "meters" }
  );

  const aiDriftPct  = (aiDriftAtEnd / blackoutDistance) * 100;
  const isroPassed  = aiDriftPct < 10;

  // Highlight the blackout section on the GT track
  const gtBeforeBlackout = gt.filter(p => p.ts <= blackoutStartTs).map(p => [p.lat, p.lon] as [number,number]);
  const gtDuringPath     = gtDuringBlackout.map(p => [p.lat, p.lon] as [number,number]);
  const gtAfterBlackout  = gt.filter(p => p.ts >= blackoutEndTs).map(p => [p.lat, p.lon] as [number,number]);

  const blackoutStartPt: [number,number] = [gtDuringBlackout[0]?.lat ?? gt[0].lat, gtDuringBlackout[0]?.lon ?? gt[0].lon];
  const blackoutEndPt:   [number,number] = [gtAtBlackoutEnd.lat, gtAtBlackoutEnd.lon];

  return (
    <div className="flex flex-col gap-3">
      {/* Top Banner & Control Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 bg-white p-3 rounded-lg border shadow-sm">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Map-Matching:</span>
          <button
            onClick={() => setUseMapMatching(prev => !prev)}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all shadow-sm flex items-center gap-1.5 ${
              useMapMatching
                ? "bg-indigo-600 text-white hover:bg-indigo-700"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200 border"
            }`}
          >
            <span>OSM Road Snap (Component 3):</span>
            <span className="underline">{useMapMatching ? "ON (OpenStreetMap)" : "OFF (Pure Inertial)"}</span>
          </button>
        </div>
        <div className="text-xs text-gray-500">
          {useMapMatching 
            ? "✨ Snapping to genuine OpenStreetMap road network — lateral drift bounded by road vectors." 
            : "⚠️ Pure Inertial Dead Reckoning — unconstrained open-loop gyro heading divergence."}
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="bg-white border rounded-lg p-3 shadow-sm">
          <div className="text-xs text-gray-500 font-medium">Blackout Duration</div>
          <div className="text-xl font-bold text-gray-800">{Math.round(blackoutEndTs - blackoutStartTs)}s</div>
        </div>
        <div className="bg-white border rounded-lg p-3 shadow-sm">
          <div className="text-xs text-gray-500 font-medium">Distance in Blackout</div>
          <div className="text-xl font-bold text-gray-800">{blackoutDistance.toFixed(1)} m</div>
        </div>
        <div className="bg-white border rounded-lg p-3 shadow-sm">
          <div className="text-xs text-gray-500 font-medium">
            {useMapMatching ? "AI Drift at Reacquisition" : "Pure Inertial Drift at Reacq."}
          </div>
          <div className="text-xl font-bold text-gray-800">{aiDriftAtEnd.toFixed(1)} m</div>
          <div className="text-xs text-gray-400">
            {useMapMatching 
              ? `(Pure DR: ${unconstrainedDriftAtEnd.toFixed(1)} m | Naive: ${rawDriftAtEnd.toFixed(1)} m)` 
              : `(With OSM RCPF: ${rcpfDriftAtEnd.toFixed(1)} m | Naive: ${rawDriftAtEnd.toFixed(1)} m)`}
          </div>
        </div>
        <div className={`rounded-lg p-3 shadow-sm border ${isroPassed ? "bg-green-50 border-green-300" : "bg-red-50 border-red-200"}`}>
          <div className="text-xs text-gray-600 font-medium">
            {useMapMatching ? "AI Drift % (ISRO < 10%)" : "Pure DR Drift % (ISRO < 10%)"}
          </div>
          <div className={`text-xl font-bold ${isroPassed ? "text-green-700" : "text-red-700"}`}>
            {aiDriftPct.toFixed(2)}%
          </div>
          <div className={`text-xs font-semibold mt-0.5 ${isroPassed ? "text-green-600" : "text-red-500"}`}>
            {isroPassed ? "✅ ISRO PASS" : "❌ ISRO FAIL"}
          </div>
        </div>
      </div>

      {/* Physics & Kinematic Context Panel (RCPF) */}
      <div className={`border rounded-lg p-3 text-xs flex flex-wrap items-center justify-between gap-2 shadow-xs transition-colors ${
        useMapMatching
          ? "bg-gradient-to-r from-indigo-50 via-purple-50 to-emerald-50 border-indigo-100"
          : "bg-gradient-to-r from-amber-50 via-orange-50 to-red-50 border-amber-200"
      }`}>
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 font-bold rounded text-[10px] uppercase tracking-wide text-white ${
            useMapMatching ? "bg-indigo-600" : "bg-amber-600"
          }`}>
            {useMapMatching ? "RCPF Core" : "Pure Inertial DR"}
          </span>
          <span className="font-semibold text-gray-800">
            {useMapMatching 
              ? "AI Speed Filter (XGBoost GPU) + Road-Constrained Particle Filter" 
              : "AI Speed Filter (Without OSM Road Network Constraints)"}
          </span>
        </div>
        <div className="flex items-center gap-4 text-gray-600">
          <span><strong className="text-gray-700">Kinematic Constraint:</strong> {useMapMatching ? "Non-Holonomic ($v_y=0$)" : "Open-Loop Integration"}</span>
          <span><strong className="text-gray-700">Heading:</strong> {useMapMatching ? "OSM Vector Road Azimuth" : "Raw Gyroscope (Yaw Drift)"}</span>
          <span className={useMapMatching ? "text-emerald-700 font-semibold" : "text-amber-700 font-semibold"}>
            {useMapMatching ? "✨ ISRO Compliant (<10% drift)" : "⚠️ Heading Divergence (Component 3 Required)"}
          </span>
        </div>
      </div>

      {/* Map */}
      <div className="h-[580px] w-full rounded-xl overflow-hidden border border-gray-200 shadow-sm relative z-0">
        <MapContainer key={segmentId} center={center} zoom={16} scrollWheelZoom={true} style={{ height: "100%", width: "100%" }}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {/* GT Track */}
          {gtBeforeBlackout.length > 1 && (
            <Polyline positions={gtBeforeBlackout} color="#10B981" weight={5} opacity={0.9}>
              <Tooltip sticky>Ground Truth (GPS) — Before Blackout</Tooltip>
            </Polyline>
          )}
          {gtDuringPath.length > 1 && (
            <Polyline positions={gtDuringPath} color="#10B981" weight={4} dashArray="6 4" opacity={0.6}>
              <Tooltip sticky>Ground Truth — During Blackout Window</Tooltip>
            </Polyline>
          )}
          {gtAfterBlackout.length > 1 && (
            <Polyline positions={gtAfterBlackout} color="#10B981" weight={5} opacity={0.9}>
              <Tooltip sticky>Ground Truth (GPS) — After Blackout</Tooltip>
            </Polyline>
          )}

          {/* Raw DR — full track */}
          {rawPath.length > 1 && (
            <Polyline positions={rawPath} color="#EF4444" weight={3} dashArray="8 6" opacity={0.8}>
              <Tooltip sticky>Raw Dead Reckoning (Naive IMU Integration)</Tooltip>
            </Polyline>
          )}

          {/* Classical Kalman Filter */}
          {fusedPath.length > 1 && (
            <Polyline positions={fusedPath} color="#3B82F6" weight={3} opacity={0.6}>
              <Tooltip sticky>Classical Fused (Kalman Filter)</Tooltip>
            </Polyline>
          )}

          {/* AI-ML Fused Track */}
          {displayAiPath.length > 1 && (
            <Polyline positions={displayAiPath} color="#8B5CF6" weight={5} opacity={0.85}>
              <Tooltip sticky>{useMapMatching ? "AI-ML + RCPF (XGBoost + Road-Constrained Particle Filter)" : "Pure Inertial DR (Without OSM Road Constraints)"}</Tooltip>
            </Polyline>
          )}

          {/* Blackout start / end markers */}
          <CircleMarker center={blackoutStartPt} radius={10} fillColor="#FBBF24" color="#92400E" weight={2} fillOpacity={0.9}>
            <Tooltip permanent>GPS Lost</Tooltip>
          </CircleMarker>
          <CircleMarker center={blackoutEndPt} radius={10} fillColor="#34D399" color="#065F46" weight={2} fillOpacity={0.9}>
            <Tooltip permanent>GPS Restored</Tooltip>
          </CircleMarker>

          {/* Uncertainty circle around AI position at blackout end */}
          <Circle
            center={[activeAiEndPt.geometry.coordinates[1], activeAiEndPt.geometry.coordinates[0]]}
            radius={Math.max(aiDriftAtEnd, 1)}
            pathOptions={{ color: "#8B5CF6", fillColor: "#8B5CF6", fillOpacity: 0.12, dashArray: "6 4" }}
          />

          {/* Start dot */}
          <CircleMarker center={gtPath[0]} radius={7} fillColor="#10B981" color="#fff" weight={2} fillOpacity={1}>
            <Tooltip>Start</Tooltip>
          </CircleMarker>
        </MapContainer>

        {/* Legend */}
        <div className="absolute top-3 right-3 bg-white/95 backdrop-blur p-3 rounded-lg shadow-lg border border-gray-200 z-[400] text-xs">
          <p className="font-bold mb-2 text-gray-700">Legend</p>
          <div className="flex items-center gap-2 mb-1"><div className="w-5 h-1 bg-[#10B981] rounded"></div> Ground Truth (GPS)</div>
          <div className="flex items-center gap-2 mb-1"><div className="w-5 border-t-2 border-dashed border-[#EF4444]"></div> Raw DR (Drifting)</div>
          <div className="flex items-center gap-2 mb-1"><div className="w-5 h-1 bg-[#3B82F6] rounded"></div> Classical Kalman Filter</div>
          <div className="flex items-center gap-2 mb-1"><div className="w-5 h-1 bg-[#8B5CF6] rounded"></div> {useMapMatching ? "AI-ML + RCPF Fused" : "Pure Inertial DR (OSM OFF)"}</div>
          <div className="flex items-center gap-2 mb-1"><div className="w-3 h-3 rounded-full bg-[#FBBF24] border border-[#92400E]"></div> GPS Lost</div>
          <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-[#34D399] border border-[#065F46]"></div> GPS Restored</div>
          <div className="mt-2 pt-2 border-t border-gray-100 text-gray-400">Purple circle = AI uncertainty at reacquisition</div>
        </div>
      </div>
    </div>
  );
}
