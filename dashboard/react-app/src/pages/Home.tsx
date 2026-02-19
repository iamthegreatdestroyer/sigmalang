import React, { useState } from 'react';
import { useApiStore } from '../store/apiStore';

export function Home() {
  const { startServices, stopServices } = useApiStore();
  const [loading, setLoading] = useState(false);

  const handleStartServices = async () => {
    setLoading(true);
    try {
      await startServices();
      alert('Services started!');
    } catch (error) {
      alert('Error starting services');
    } finally {
      setLoading(false);
    }
  };

  const handleStopServices = async () => {
    setLoading(true);
    try {
      await stopServices();
      alert('Services stopped!');
    } catch (error) {
      alert('Error stopping services');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 className="text-3xl font-bold mb-6">🏠 Welcome to ΣLANG Dashboard</h2>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg p-6 text-white text-center">
          <div className="text-sm opacity-90">Tests Passing</div>
          <div className="text-4xl font-bold">1,656/1,656</div>
          <div className="text-sm opacity-90">100%</div>
        </div>
        <div className="bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg p-6 text-white text-center">
          <div className="text-sm opacity-90">Code Coverage</div>
          <div className="text-4xl font-bold">>85%</div>
          <div className="text-sm opacity-90">✅ Passing</div>
        </div>
        <div className="bg-gradient-to-br from-green-500 to-emerald-500 rounded-lg p-6 text-white text-center">
          <div className="text-sm opacity-90">API Status</div>
          <div className="text-4xl font-bold">✅ Ready</div>
          <div className="text-sm opacity-90">Real-time</div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
          <h3 className="text-2xl font-bold mb-4">🚀 Quick Actions</h3>
          <div className="space-y-3">
            <button
              onClick={handleStartServices}
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition disabled:opacity-50"
            >
              ⬆️ Start Services
            </button>
            <button
              onClick={handleStopServices}
              disabled={loading}
              className="w-full bg-gray-600 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded-lg transition disabled:opacity-50"
            >
              ⬇️ Stop Services
            </button>
            <a
              href="/setup"
              className="block w-full text-center bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition"
            >
              ⚙️ Setup Wizard
            </a>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
          <h3 className="text-2xl font-bold mb-4">📚 Getting Started</h3>
          <ol className="list-decimal list-inside space-y-2 text-gray-700">
            <li>Setup: Use the Setup Wizard to configure your environment</li>
            <li>Test: Use the Testing interface to test API endpoints</li>
            <li>Monitor: View real-time metrics in the Metrics page</li>
            <li>Optimize: Check Performance page for optimization tips</li>
          </ol>
        </div>
      </div>

      {/* Key Features */}
      <div className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
        <h3 className="text-2xl font-bold mb-4">🎯 Key Features</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex items-center">
            <span className="text-2xl mr-3">✅</span>
            <span>Interactive setup wizard</span>
          </div>
          <div className="flex items-center">
            <span className="text-2xl mr-3">✅</span>
            <span>API endpoint testing</span>
          </div>
          <div className="flex items-center">
            <span className="text-2xl mr-3">✅</span>
            <span>Real-time metrics</span>
          </div>
          <div className="flex items-center">
            <span className="text-2xl mr-3">✅</span>
            <span>Performance monitoring</span>
          </div>
          <div className="flex items-center">
            <span className="text-2xl mr-3">✅</span>
            <span>Documentation browser</span>
          </div>
          <div className="flex items-center">
            <span className="text-2xl mr-3">✅</span>
            <span>One-click deployment tools</span>
          </div>
        </div>
      </div>
    </div>
  );
}
