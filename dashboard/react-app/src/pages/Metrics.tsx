import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const data = [
  { date: 'Feb 10', requests: 1000, latency: 10, errors: 0.01 },
  { date: 'Feb 11', requests: 1050, latency: 10.5, errors: 0.011 },
  { date: 'Feb 12', requests: 1100, latency: 11, errors: 0.012 },
  { date: 'Feb 13', requests: 1150, latency: 11.5, errors: 0.013 },
  { date: 'Feb 14', requests: 1200, latency: 12, errors: 0.014 },
];

export function Metrics() {
  return (
    <div>
      <h2 className="text-3xl font-bold mb-6">📊 Real-time Metrics</h2>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg p-4 text-white text-center">
          <div className="text-sm opacity-90">Requests/min</div>
          <div className="text-3xl font-bold">1,250</div>
        </div>
        <div className="bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg p-4 text-white text-center">
          <div className="text-sm opacity-90">Avg Latency</div>
          <div className="text-3xl font-bold">12.5ms</div>
        </div>
        <div className="bg-gradient-to-br from-green-500 to-emerald-500 rounded-lg p-4 text-white text-center">
          <div className="text-sm opacity-90">Error Rate</div>
          <div className="text-3xl font-bold">0.02%</div>
        </div>
        <div className="bg-gradient-to-br from-orange-500 to-red-500 rounded-lg p-4 text-white text-center">
          <div className="text-sm opacity-90">Cache Hit Rate</div>
          <div className="text-3xl font-bold">92.3%</div>
        </div>
      </div>

      {/* Compression Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white rounded-lg shadow p-4 text-center border border-gray-200">
          <div className="text-sm text-gray-600">Avg Ratio</div>
          <div className="text-2xl font-bold text-purple-600">15.2x</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4 text-center border border-gray-200">
          <div className="text-sm text-gray-600">Min Ratio</div>
          <div className="text-2xl font-bold text-blue-600">5.1x</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4 text-center border border-gray-200">
          <div className="text-sm text-gray-600">Max Ratio</div>
          <div className="text-2xl font-bold text-green-600">48.3x</div>
        </div>
        <div className="bg-white rounded-lg shadow p-4 text-center border border-gray-200">
          <div className="text-sm text-gray-600">Bytes Saved</div>
          <div className="text-2xl font-bold text-orange-600">125.4 GB</div>
        </div>
      </div>

      {/* Performance Trends */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h3 className="text-xl font-bold mb-4">Performance Trends</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Legend />
            <Line yAxisId="left" type="monotone" dataKey="requests" stroke="#667eea" name="Requests" />
            <Line yAxisId="right" type="monotone" dataKey="latency" stroke="#764ba2" name="Latency (ms)" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* External Links */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <a href="http://localhost:26900" target="_blank" rel="noopener noreferrer"
           className="block bg-blue-50 border border-blue-200 rounded-lg p-4 hover:bg-blue-100 transition">
          <h4 className="font-bold text-blue-600 mb-2">📊 Prometheus</h4>
          <p className="text-sm text-gray-600">View detailed metrics and performance data</p>
        </a>
        <a href="http://localhost:26910" target="_blank" rel="noopener noreferrer"
           className="block bg-green-50 border border-green-200 rounded-lg p-4 hover:bg-green-100 transition">
          <h4 className="font-bold text-green-600 mb-2">📈 Grafana</h4>
          <p className="text-sm text-gray-600">View dashboards and visualizations</p>
        </a>
      </div>
    </div>
  );
}
