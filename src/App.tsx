import { useState } from "react";
import "leaflet/dist/leaflet.css";

function App() {
  const [activeTab, setActiveTab] = useState<"replay" | "live">("replay");

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-indigo-600 text-white p-4 shadow-md">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-xl font-bold tracking-tight">AI-ML IDR Prototype</h1>
          <div className="text-sm bg-indigo-800 px-3 py-1 rounded-full opacity-80">
            ISRO Problem Statement 26168
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
        {activeTab === "replay" && (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-8 border-2 border-dashed border-gray-300 rounded-xl bg-gray-50">
            <h2 className="text-2xl font-bold text-gray-700 mb-2">Benchmark Replay Mode</h2>
            <p className="text-gray-500 max-w-md">
              Loads IO-VNBD dataset subset, simulates GNSS blackout, and plots Ground Truth vs Raw Dead Reckoning vs AI-Fused tracks.
            </p>
            <div className="mt-6 px-4 py-2 bg-indigo-100 text-indigo-700 rounded-lg text-sm font-medium">
              Coming soon
            </div>
          </div>
        )}

        {activeTab === "live" && (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-8 border-2 border-dashed border-gray-300 rounded-xl bg-gray-50">
            <h2 className="text-2xl font-bold text-gray-700 mb-2">Live Sensor Demo</h2>
            <p className="text-gray-500 max-w-md">
              Uses your phone's DeviceMotion and Geolocation APIs to run the fusion pipeline live.
            </p>
            <div className="mt-6 px-4 py-2 bg-indigo-100 text-indigo-700 rounded-lg text-sm font-medium">
              Coming soon
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
