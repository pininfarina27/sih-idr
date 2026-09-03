import { useState, useEffect } from "react";

interface DriftResult {
  road_dist_m: number;
  raw_drift_m: number;
  ai_drift_m: number;
  raw_pct: number;
  ai_pct: number;
  isro_pass: boolean;
}

interface EvalSummary {
  route_split: { train_routes: number; test_routes: number; train_rows: number; test_rows: number };
  gbr: { mae_kmh: number; rmse_kmh: number };
  lr: { mae_kmh: number; rmse_kmh: number };
  const: { mae_kmh: number; rmse_kmh: number };
  feature_importances: Record<string, number>;
}

export default function DriftChart() {
  const [driftData, setDriftData] = useState<{segments: Record<string, Record<string, DriftResult>>, durations: number[]} | null>(null);
  const [evalData, setEvalData] = useState<EvalSummary | null>(null);

  useEffect(() => {
    fetch("/data/drift_results.json").then(r => r.json()).then(setDriftData).catch(() => {});
    fetch("/data/evaluation_summary.json").then(r => r.json()).then(setEvalData).catch(() => {});
  }, []);

  return (
    <div className="w-full flex flex-col gap-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-800">Evaluation & Validation</h2>
        <p className="text-gray-500 mt-1">Honest, route-level held-out evaluation. Train: 57 routes. Test: 15 completely unseen routes.</p>
      </div>

      {/* Model Comparison Table */}
      {evalData && (
        <div className="bg-white border rounded-xl p-5 shadow-sm">
          <h3 className="font-bold text-lg text-gray-800 mb-1">Model Comparison — Route-Held-Out Test</h3>
          <p className="text-sm text-gray-500 mb-4">
            Split: <strong>{evalData.route_split.train_routes} training routes</strong> ({evalData.route_split.train_rows.toLocaleString()} rows) vs <strong>{evalData.route_split.test_routes} unseen test routes</strong> ({evalData.route_split.test_rows.toLocaleString()} rows).
            This avoids temporal data leakage from a row-level random split.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="bg-gray-50">
                  <th className="text-left p-3 border border-gray-200 font-semibold">Model</th>
                  <th className="text-center p-3 border border-gray-200 font-semibold">MAE (km/h)</th>
                  <th className="text-center p-3 border border-gray-200 font-semibold">RMSE (km/h)</th>
                  <th className="text-center p-3 border border-gray-200 font-semibold">Improvement vs Constant</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { name: "Constant Speed (Baseline)", mae: evalData.const.mae_kmh, rmse: evalData.const.rmse_kmh, color: "text-gray-600" },
                  { name: "Linear Regression", mae: evalData.lr.mae_kmh, rmse: evalData.lr.rmse_kmh, color: "text-blue-600" },
                  { name: "Gradient Boosting (Ours)", mae: evalData.gbr.mae_kmh, rmse: evalData.gbr.rmse_kmh, color: "text-indigo-600" },
                ].map((m, i) => {
                  const impPct = i === 0 ? 0 : ((evalData.const.mae_kmh - m.mae) / evalData.const.mae_kmh * 100);
                  return (
                    <tr key={m.name} className={i === 2 ? "bg-indigo-50 font-medium" : ""}>
                      <td className={`p-3 border border-gray-200 ${m.color}`}>{m.name}</td>
                      <td className="p-3 border border-gray-200 text-center">{m.mae}</td>
                      <td className="p-3 border border-gray-200 text-center">{m.rmse}</td>
                      <td className="p-3 border border-gray-200 text-center">
                        {i === 0 ? "—" : <span className="text-green-700 font-semibold">+{impPct.toFixed(1)}% better</span>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Feature Importance */}
          <h4 className="font-semibold text-gray-700 mt-5 mb-2">Feature Importances (GBR)</h4>
          <div className="flex flex-col gap-1.5">
            {Object.entries(evalData.feature_importances).sort((a,b) => b[1]-a[1]).map(([feat, imp]) => (
              <div key={feat} className="flex items-center gap-3 text-sm">
                <span className="w-40 text-gray-600 font-mono text-xs">{feat}</span>
                <div className="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
                  <div className="h-full bg-indigo-500 rounded-full" style={{ width: `${imp}%` }} />
                </div>
                <span className="w-12 text-right font-medium">{imp}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Drift vs Duration Table */}
      {driftData && (
        <div className="bg-white border rounded-xl p-5 shadow-sm">
          <h3 className="font-bold text-lg text-gray-800 mb-1">Drift % vs GPS Blackout Duration</h3>
          <p className="text-sm text-gray-500 mb-4">
            Drift measured at GPS reacquisition after N seconds of complete simulated GNSS loss.
            ISRO target: AI drift &lt; 10% of distance traveled during blackout.
          </p>
          {Object.entries(driftData.segments).map(([seg, results]) => (
            <div key={seg} className="mb-5">
              <h4 className="font-semibold text-gray-700 mb-2">Segment {seg}</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-gray-50">
                      <th className="text-left p-2 border border-gray-200">Blackout Duration</th>
                      <th className="text-center p-2 border border-gray-200">Road Distance</th>
                      <th className="text-center p-2 border border-gray-200">Raw DR Drift%</th>
                      <th className="text-center p-2 border border-gray-200">AI-ML Drift%</th>
                      <th className="text-center p-2 border border-gray-200">ISRO Pass?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(results).map(([dur, r]) => (
                      <tr key={dur}>
                        <td className="p-2 border border-gray-200 font-medium">{dur}s</td>
                        <td className="p-2 border border-gray-200 text-center text-gray-600">{r.road_dist_m}m</td>
                        <td className={`p-2 border border-gray-200 text-center font-medium ${r.raw_pct >= 10 ? "bg-red-50 text-red-700" : "bg-green-50 text-green-700"}`}>
                          {r.raw_pct}%
                        </td>
                        <td className={`p-2 border border-gray-200 text-center font-bold ${r.ai_pct < 10 ? "bg-green-100 text-green-800" : "bg-red-50 text-red-700"}`}>
                          {r.ai_pct}%
                        </td>
                        <td className={`p-2 border border-gray-200 text-center font-bold ${r.isro_pass ? "text-green-700" : "text-red-600"}`}>
                          {r.isro_pass ? "✅ YES" : "❌ NO"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}

      {!driftData && !evalData && (
        <div className="p-8 text-center text-gray-400">Loading evaluation data...</div>
      )}
    </div>
  );
}
