import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { Home } from './pages/Home';
import { Setup } from './pages/Setup';
import { Testing } from './pages/Testing';
import { Metrics } from './pages/Metrics';
import { Performance } from './pages/Performance';
import { Documentation } from './pages/Documentation';
import { Tools } from './pages/Tools';
import { useApiStore } from './store/apiStore';

export function App() {
  const [apiHealth, setApiHealth] = useState(false);
  const { checkApiHealth } = useApiStore();

  useEffect(() => {
    const updateHealth = async () => {
      const healthy = await checkApiHealth();
      setApiHealth(healthy);
    };

    updateHealth();
    const interval = setInterval(updateHealth, 5000);
    return () => clearInterval(interval);
  }, [checkApiHealth]);

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg">
          <div className="container mx-auto px-4 py-4 flex justify-between items-center">
            <h1 className="text-3xl font-bold">🔤 ΣLANG Dashboard</h1>
            <span className="text-lg font-bold">
              {apiHealth ? '🟢 Healthy' : '🔴 Offline'}
            </span>
          </div>
        </header>

        {/* Sidebar Navigation */}
        <div className="flex">
          <aside className="w-64 bg-white shadow-lg">
            <nav className="p-4">
              <div className="space-y-2">
                <NavLink to="/" icon="🏠" label="Home" />
                <NavLink to="/setup" icon="⚙️" label="Setup Wizard" />
                <NavLink to="/testing" icon="🧪" label="Testing" />
                <NavLink to="/metrics" icon="📊" label="Metrics" />
                <NavLink to="/performance" icon="⚡" label="Performance" />
                <NavLink to="/documentation" icon="📝" label="Documentation" />
                <NavLink to="/tools" icon="🔧" label="Tools" />
              </div>

              <div className="mt-8 pt-8 border-t">
                <div className="text-sm font-bold text-gray-600 mb-3">Quick Links</div>
                <a href="http://localhost:26080/docs" target="_blank" rel="noopener noreferrer"
                   className="block text-sm text-blue-600 hover:underline mb-2">
                  🌐 API Docs
                </a>
                <a href="http://localhost:26910" target="_blank" rel="noopener noreferrer"
                   className="block text-sm text-blue-600 hover:underline">
                  📊 Grafana
                </a>
              </div>
            </nav>
          </aside>

          {/* Main Content */}
          <main className="flex-1 p-8">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/setup" element={<Setup />} />
              <Route path="/testing" element={<Testing />} />
              <Route path="/metrics" element={<Metrics />} />
              <Route path="/performance" element={<Performance />} />
              <Route path="/documentation" element={<Documentation />} />
              <Route path="/tools" element={<Tools />} />
            </Routes>
          </main>
        </div>

        {/* Footer */}
        <footer className="bg-gray-800 text-white text-center py-4 mt-8">
          <p>ΣLANG Local Dashboard • <a href="https://github.com/iamthegreatdestroyer/sigmalang" className="underline">GitHub</a></p>
        </footer>
      </div>
    </BrowserRouter>
  );
}

function NavLink({ to, icon, label }: { to: string; icon: string; label: string }) {
  return (
    <Link
      to={to}
      className="flex items-center gap-3 px-4 py-2 rounded-lg hover:bg-gray-100 text-gray-700 hover:text-gray-900 transition"
    >
      <span className="text-xl">{icon}</span>
      <span>{label}</span>
    </Link>
  );
}

export default App;
