import { useState, useEffect } from "react";
import MapView from "./MapView";

interface SegmentMeta { id: string; name: string; duration: number; }

export default function BenchmarkReplay() {
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
          <p className="text-gray-500">Visualizing AI-ML Pipeline against Ground Truth.</p>
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
      
      <MapView key={activeSeg} segmentId={activeSeg} />
    </div>
  );
}
