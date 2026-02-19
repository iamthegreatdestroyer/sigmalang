import React, { useState } from 'react';

export function Tools() {
  const [tool, setTool] = useState<'tests' | 'coverage' | 'load' | 'sdks' | 'logs'>('tests');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);

  const runCommand = async (command: string) => {
    setLoading(true);
    try {
      const response = await fetch('/api/run-command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command })
      });
      const data = await response.json();
      setOutput(data.output);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 className="text-3xl font-bold mb-6">🔧 Tools & Utilities</h2>

      {/* Tool Selector */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {[
          { id: 'tests' as const, label: 'Run Tests', icon: '🧪' },
          { id: 'coverage' as const, label: 'Coverage', icon: '📊' },
          { id: 'load' as const, label: 'Load Test', icon: '🚀' },
          { id: 'sdks' as const, label: 'Generate SDKs', icon: '📦' },
          { id: 'logs' as const, label: 'View Logs', icon: '📝' },
        ].map((t) => (
          <button
            key={t.id}
            onClick={() => setTool(t.id)}
            className={`whitespace-nowrap px-4 py-2 rounded-lg transition ${
              tool === t.id
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        {tool === 'tests' && (
          <div>
            <h3 className="text-xl font-bold mb-4">🧪 Run Test Suite</h3>
            <p className="text-gray-600 mb-4">Run ΣLANG's comprehensive test suite (1,656 tests):</p>
            <button
              onClick={() => runCommand('pytest tests/ -q')}
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition disabled:opacity-50 mb-4"
            >
              {loading ? '⏳ Running...' : '🧪 Run All Tests'}
            </button>
          </div>
        )}

        {tool === 'coverage' && (
          <div>
            <h3 className="text-xl font-bold mb-4">📊 Generate Coverage Report</h3>
            <p className="text-gray-600 mb-4">Generate code coverage analysis:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
              <button
                onClick={() => runCommand('bash run_coverage.sh --fast')}
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition disabled:opacity-50"
              >
                {loading ? '⏳ Running...' : '⚡ Fast (5 min)'}
              </button>
              <button
                onClick={() => runCommand('bash run_coverage.sh --full')}
                disabled={loading}
                className="bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded-lg transition disabled:opacity-50"
              >
                {loading ? '⏳ Running...' : '📊 Full (10 min)'}
              </button>
            </div>
          </div>
        )}

        {tool === 'load' && (
          <div>
            <h3 className="text-xl font-bold mb-4">🚀 Run Load Tests</h3>
            <p className="text-gray-600 mb-4">Test system performance under load:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
              {['baseline', 'normal', 'peak', 'spike', 'endurance'].map((test) => (
                <button
                  key={test}
                  onClick={() => runCommand(`bash run_load_tests.sh ${test}`)}
                  disabled={loading}
                  className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg transition disabled:opacity-50 text-sm"
                >
                  {loading ? '⏳ Running...' : `${test.charAt(0).toUpperCase() + test.slice(1)}`}
                </button>
              ))}
            </div>
          </div>
        )}

        {tool === 'sdks' && (
          <div>
            <h3 className="text-xl font-bold mb-4">📦 Generate SDKs</h3>
            <p className="text-gray-600 mb-4">Generate SDKs for multiple languages:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
              {['all', 'typescript', 'java', 'python', 'go'].map((lang) => (
                <button
                  key={lang}
                  onClick={() => runCommand(`bash generate_sdks.sh ${lang}`)}
                  disabled={loading}
                  className="bg-orange-600 hover:bg-orange-700 text-white font-bold py-2 px-4 rounded-lg transition disabled:opacity-50 text-sm"
                >
                  {loading ? '⏳ Generating...' : `${lang.toUpperCase()}`}
                </button>
              ))}
            </div>
          </div>
        )}

        {tool === 'logs' && (
          <div>
            <h3 className="text-xl font-bold mb-4">📝 View Service Logs</h3>
            <p className="text-gray-600 mb-4">View logs from Docker services:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
              {['sigmalang', 'redis', 'prometheus', 'grafana'].map((service) => (
                <button
                  key={service}
                  onClick={() => runCommand(`docker compose logs ${service} | tail -50`)}
                  disabled={loading}
                  className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg transition disabled:opacity-50"
                >
                  {loading ? '⏳ Loading...' : `${service}`}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Output */}
        {output && (
          <div className="mt-6">
            <h4 className="font-bold mb-2">Output:</h4>
            <pre className="bg-gray-900 text-green-400 p-4 rounded overflow-auto max-h-64 text-sm">
              {output}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
