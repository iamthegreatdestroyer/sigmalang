import React from 'react';

export function Performance() {
  const benchmarks = [
    { operation: 'Encoding (Low)', speed: '<5ms', compression: '5-8x' },
    { operation: 'Encoding (Medium)', speed: '10-20ms', compression: '10-20x' },
    { operation: 'Encoding (High)', speed: '50-100ms', compression: '20-50x' },
    { operation: 'Entity Extract', speed: '15-25ms', compression: 'N/A' },
    { operation: 'Analogy Solve', speed: '20-30ms', compression: 'N/A' },
  ];

  return (
    <div>
      <h2 className="text-3xl font-bold mb-6">⚡ Performance Optimization</h2>

      {/* Tips */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h3 className="text-xl font-bold mb-4">📈 Performance Tips</h3>
        <div className="space-y-4">
          <div className="border-l-4 border-blue-600 pl-4">
            <h4 className="font-bold text-blue-600">Use Low Optimization for Speed</h4>
            <p className="text-gray-600">Trade-off: Lower compression but faster. Best for: Real-time applications</p>
          </div>
          <div className="border-l-4 border-green-600 pl-4">
            <h4 className="font-bold text-green-600">Use High Optimization for Compression</h4>
            <p className="text-gray-600">Trade-off: Slower but better compression. Best for: Batch processing, offline</p>
          </div>
          <div className="border-l-4 border-purple-600 pl-4">
            <h4 className="font-bold text-purple-600">Enable Caching</h4>
            <p className="text-gray-600">Significantly improves repeated operations. Configure via environment variables</p>
          </div>
          <div className="border-l-4 border-orange-600 pl-4">
            <h4 className="font-bold text-orange-600">Monitor Metrics</h4>
            <p className="text-gray-600">Check Prometheus and view dashboards in Grafana</p>
          </div>
        </div>
      </div>

      {/* Benchmarks Table */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-6">
        <h3 className="text-xl font-bold mb-4">🎯 Benchmarks</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b-2 border-gray-300">
                <th className="text-left p-3">Operation</th>
                <th className="text-left p-3">Speed</th>
                <th className="text-left p-3">Compression</th>
              </tr>
            </thead>
            <tbody>
              {benchmarks.map((b, i) => (
                <tr key={i} className="border-b border-gray-200 hover:bg-gray-50">
                  <td className="p-3">{b.operation}</td>
                  <td className="p-3">{b.speed}</td>
                  <td className="p-3">{b.compression}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Tools */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <button className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 px-6 rounded-lg transition text-left">
          <div className="text-2xl mb-2">🚀</div>
          <div>Run Load Test</div>
          <p className="text-sm opacity-90">Test with realistic load scenarios</p>
        </button>
        <button className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-4 px-6 rounded-lg transition text-left">
          <div className="text-2xl mb-2">📊</div>
          <div>Check Coverage</div>
          <p className="text-sm opacity-90">Generate coverage report (fast/full)</p>
        </button>
      </div>
    </div>
  );
}
