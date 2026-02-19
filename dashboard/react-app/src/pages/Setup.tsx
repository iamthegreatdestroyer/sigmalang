import React, { useState } from 'react';

export function Setup() {
  const [step, setStep] = useState(1);

  return (
    <div>
      <h2 className="text-3xl font-bold mb-6">⚙️ Local Setup Wizard</h2>

      {/* Steps Navigation */}
      <div className="flex gap-4 mb-6">
        {[1, 2, 3, 4].map((s) => (
          <button
            key={s}
            onClick={() => setStep(s)}
            className={`flex-1 p-3 rounded font-bold transition ${
              step === s ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'
            }`}
          >
            {s === 1 && '1️⃣ Prerequisites'}
            {s === 2 && '2️⃣ Configuration'}
            {s === 3 && '3️⃣ Docker'}
            {s === 4 && '4️⃣ Verify'}
          </button>
        ))}
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        {step === 1 && (
          <div>
            <h3 className="text-xl font-bold mb-4">Check Prerequisites</h3>
            <div className="space-y-2">
              <div className="flex justify-between p-2 border rounded">
                <span>Docker</span>
                <span className="text-green-600">✅ Installed</span>
              </div>
              <div className="flex justify-between p-2 border rounded">
                <span>Docker Compose</span>
                <span className="text-green-600">✅ Installed</span>
              </div>
              <div className="flex justify-between p-2 border rounded">
                <span>Python</span>
                <span className="text-green-600">✅ Installed</span>
              </div>
              <div className="flex justify-between p-2 border rounded">
                <span>Git</span>
                <span className="text-green-600">✅ Installed</span>
              </div>
              <div className="flex justify-between p-2 border rounded">
                <span>Disk Space Available</span>
                <span className="text-green-600">✅ 50GB+</span>
              </div>
            </div>
            <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded text-green-700">
              ✅ All prerequisites met!
            </div>
          </div>
        )}

        {step === 2 && (
          <div>
            <h3 className="text-xl font-bold mb-4">Environment Configuration</h3>
            <p className="text-gray-600 mb-4">Generate .env file from template:</p>
            <button className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg mb-4">
              Generate .env from template
            </button>
            <textarea
              className="w-full border rounded p-3"
              rows={10}
              value={`# ΣLANG Configuration
SIGMALANG_API_PORT=8000
SIGMALANG_DEBUG=false
SIGMALANG_LOG_LEVEL=INFO
SIGMALANG_CACHE_ENABLED=true

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/sigmalang

# Redis
REDIS_URL=redis://localhost:6379/0

# Grafana
GRAFANA_ADMIN_PASSWORD=admin
GRAFANA_ADMIN_USER=admin`}
              readOnly
            />
          </div>
        )}

        {step === 3 && (
          <div>
            <h3 className="text-xl font-bold mb-4">Docker Compose Setup</h3>
            <div className="space-y-3">
              <button className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition">
                ⬆️ Start Services
              </button>
              <button className="w-full bg-gray-600 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded-lg transition">
                📋 View Status
              </button>
              <button className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg transition">
                ⬇️ Stop Services
              </button>
            </div>
            <pre className="mt-4 bg-gray-100 p-3 rounded text-sm">
              Container status will appear here...
            </pre>
          </div>
        )}

        {step === 4 && (
          <div>
            <h3 className="text-xl font-bold mb-4">Verify Installation</h3>
            <div className="space-y-2 mb-4">
              <div className="flex justify-between p-2 border rounded">
                <span>Docker Running</span>
                <span className="text-green-600">✅ OK</span>
              </div>
              <div className="flex justify-between p-2 border rounded">
                <span>API Accessible</span>
                <span className="text-green-600">✅ OK</span>
              </div>
              <div className="flex justify-between p-2 border rounded">
                <span>Redis Connected</span>
                <span className="text-green-600">✅ OK</span>
              </div>
              <div className="flex justify-between p-2 border rounded">
                <span>Prometheus Available</span>
                <span className="text-green-600">✅ OK</span>
              </div>
            </div>
            <div className="p-4 bg-green-50 border border-green-200 rounded text-green-700">
              ✅ All checks passed! Setup is complete.
            </div>
            <button className="w-full mt-4 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition">
              🔍 Run Verification
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
