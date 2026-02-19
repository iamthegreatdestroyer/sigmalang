import React, { useState } from 'react';

export function Documentation() {
  const [section, setSection] = useState('getting-started');

  return (
    <div>
      <h2 className="text-3xl font-bold mb-6">📝 Documentation</h2>

      {/* Section Selector */}
      <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
        {['getting-started', 'api', 'config', 'deployment', 'troubleshooting'].map((s) => (
          <button
            key={s}
            onClick={() => setSection(s)}
            className={`whitespace-nowrap px-4 py-2 rounded-lg transition ${
              section === s
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            {s === 'getting-started' && 'Getting Started'}
            {s === 'api' && 'API Reference'}
            {s === 'config' && 'Configuration'}
            {s === 'deployment' && 'Deployment'}
            {s === 'troubleshooting' && 'Troubleshooting'}
          </button>
        ))}
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        {section === 'getting-started' && (
          <div>
            <h3 className="text-2xl font-bold mb-4">Quick Start</h3>
            <div className="space-y-4 text-gray-700">
              <div>
                <h4 className="font-bold mb-2">1. Install Prerequisites</h4>
                <ul className="list-disc list-inside ml-4 space-y-1">
                  <li>Docker and Docker Compose</li>
                  <li>Python 3.9+</li>
                </ul>
              </div>
              <div>
                <h4 className="font-bold mb-2">2. Clone Repository</h4>
                <pre className="bg-gray-100 p-3 rounded overflow-x-auto">
                  git clone https://github.com/iamthegreatdestroyer/sigmalang.git
                  cd sigmalang
                </pre>
              </div>
              <div>
                <h4 className="font-bold mb-2">3. Start Services</h4>
                <pre className="bg-gray-100 p-3 rounded">docker compose up -d</pre>
              </div>
              <div>
                <h4 className="font-bold mb-2">4. Verify Installation</h4>
                <pre className="bg-gray-100 p-3 rounded">curl http://localhost:26080/health</pre>
              </div>
            </div>
          </div>
        )}

        {section === 'api' && (
          <div>
            <h3 className="text-2xl font-bold mb-4">Available Endpoints</h3>
            <div className="space-y-3">
              <div className="border-l-4 border-blue-600 pl-4">
                <div className="font-bold text-blue-600">POST /api/encode</div>
                <p className="text-gray-600">Encode text with specified optimization level</p>
              </div>
              <div className="border-l-4 border-green-600 pl-4">
                <div className="font-bold text-green-600">POST /api/decode</div>
                <p className="text-gray-600">Decode previously encoded data</p>
              </div>
              <div className="border-l-4 border-purple-600 pl-4">
                <div className="font-bold text-purple-600">POST /api/entities</div>
                <p className="text-gray-600">Extract entities from text</p>
              </div>
              <div className="border-l-4 border-orange-600 pl-4">
                <div className="font-bold text-orange-600">POST /api/analogy</div>
                <p className="text-gray-600">Solve word analogies</p>
              </div>
              <div className="border-l-4 border-red-600 pl-4">
                <div className="font-bold text-red-600">GET /health</div>
                <p className="text-gray-600">Health check endpoint</p>
              </div>
              <div className="border-l-4 border-gray-600 pl-4">
                <div className="font-bold text-gray-600">GET /metrics</div>
                <p className="text-gray-600">Prometheus metrics</p>
              </div>
            </div>
          </div>
        )}

        {section === 'config' && (
          <div>
            <h3 className="text-2xl font-bold mb-4">Environment Variables</h3>
            <div className="space-y-3 text-gray-700">
              <div className="bg-gray-50 p-3 rounded border border-gray-200">
                <div className="font-bold">SIGMALANG_API_PORT</div>
                <p className="text-sm">API port (default: 8000)</p>
              </div>
              <div className="bg-gray-50 p-3 rounded border border-gray-200">
                <div className="font-bold">SIGMALANG_DEBUG</div>
                <p className="text-sm">Debug mode (default: false)</p>
              </div>
              <div className="bg-gray-50 p-3 rounded border border-gray-200">
                <div className="font-bold">SIGMALANG_LOG_LEVEL</div>
                <p className="text-sm">Log level (default: INFO)</p>
              </div>
              <div className="bg-gray-50 p-3 rounded border border-gray-200">
                <div className="font-bold">SIGMALANG_CACHE_ENABLED</div>
                <p className="text-sm">Enable caching (default: true)</p>
              </div>
            </div>
          </div>
        )}

        {section === 'deployment' && (
          <div>
            <h3 className="text-2xl font-bold mb-4">Deployment Options</h3>
            <div className="space-y-6">
              <div className="border-l-4 border-blue-600 pl-4">
                <h4 className="font-bold text-blue-600 mb-2">Docker Compose (Local/Dev)</h4>
                <pre className="bg-gray-100 p-3 rounded overflow-x-auto">docker compose up -d</pre>
              </div>
              <div className="border-l-4 border-green-600 pl-4">
                <h4 className="font-bold text-green-600 mb-2">Kubernetes (Production)</h4>
                <pre className="bg-gray-100 p-3 rounded overflow-x-auto">helm install sigmalang sigmalang/sigmalang</pre>
              </div>
            </div>
          </div>
        )}

        {section === 'troubleshooting' && (
          <div>
            <h3 className="text-2xl font-bold mb-4">Common Issues</h3>
            <div className="space-y-4">
              <div className="border-l-4 border-red-600 pl-4">
                <h4 className="font-bold text-red-600">API won't start</h4>
                <ul className="list-disc list-inside text-sm text-gray-600 mt-2">
                  <li>Check Docker is running: `docker ps`</li>
                  <li>View logs: `docker compose logs sigmalang`</li>
                </ul>
              </div>
              <div className="border-l-4 border-orange-600 pl-4">
                <h4 className="font-bold text-orange-600">Port already in use</h4>
                <ul className="list-disc list-inside text-sm text-gray-600 mt-2">
                  <li>Change port in docker-compose.yml</li>
                  <li>Or stop other services</li>
                </ul>
              </div>
              <div className="border-l-4 border-yellow-600 pl-4">
                <h4 className="font-bold text-yellow-600">Connection refused</h4>
                <ul className="list-disc list-inside text-sm text-gray-600 mt-2">
                  <li>Ensure services are running: `docker compose ps`</li>
                  <li>Wait 10 seconds for services to start</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
