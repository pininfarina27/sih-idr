import { useState } from "react";
import BenchmarkReplay from "./components/BenchmarkReplay";
import LiveSensorDemo from "./components/LiveSensorDemo";

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
