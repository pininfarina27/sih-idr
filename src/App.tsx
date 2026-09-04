import { useState } from "react";
import BenchmarkReplay from "./components/BenchmarkReplay";
import LiveSensorDemo from "./components/LiveSensorDemo";
import DriftChart from "./components/DriftChart";
import ErrorBoundary from "./components/ErrorBoundary";

type Tab = "replay" | "live" | "evaluation";

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("replay");

  const tabs: { id: Tab; label: string }[] = [
    { id: "replay",     label: "Benchmark Replay" },
    { id: "evaluation", label: "Evaluation" },
    { id: "live",       label: "Live Sensor Demo" },
  ];

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-indigo-600 text-white p-4 shadow-md">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div>
            <h1 className="text-xl font-bold tracking-tight">AI-ML Intelligent Dead Reckoning</h1>
            <p className="text-indigo-200 text-xs mt-0.5">SIH 2026 • PS 26168 • ISRO</p>
          </div>
          <div className="text-sm bg-indigo-800 px-3 py-1 rounded-full opacity-80 hidden sm:block">
            IO-VNBD Dataset • 58h • 72 Routes
          </div>
        </div>
      </header>

      <div className="bg-white border-b shadow-sm">
        <div className="max-w-7xl mx-auto flex gap-1 px-4">
          {tabs.map(tab => (
            <button
              key={tab.id}
              className={`py-3 px-5 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? "border-b-2 border-indigo-600 text-indigo-600"
                  : "text-gray-500 hover:text-gray-800"
              }`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <main className="flex-1 max-w-7xl mx-auto w-full p-4 flex flex-col">
        <ErrorBoundary>
          {activeTab === "replay"     && <BenchmarkReplay />}
          {activeTab === "evaluation" && <DriftChart />}
          {activeTab === "live"       && <LiveSensorDemo />}
        </ErrorBoundary>
      </main>
    </div>
  );
}
