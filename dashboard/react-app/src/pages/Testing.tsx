import React, { useState } from 'react';
import { useApiStore } from '../store/apiStore';

export function Testing() {
  const { testEncode, testEntities, testAnalogy } = useApiStore();
  const [tab, setTab] = useState<'encode' | 'entities' | 'analogy' | 'search'>('encode');

  // Encoding
  const [encodeText, setEncodeText] = useState('The quick brown fox jumps over the lazy dog');
  const [encodeOpt, setEncodeOpt] = useState('medium');
  const [encodeResult, setEncodeResult] = useState<any>(null);
  const [encodeLoading, setEncodeLoading] = useState(false);

  // Entities
  const [entityText, setEntityText] = useState('Apple Inc is located in Cupertino, California');
  const [entityResult, setEntityResult] = useState<any>(null);
  const [entityLoading, setEntityLoading] = useState(false);

  // Analogy
  const [word1, setWord1] = useState('king');
  const [word2, setWord2] = useState('queen');
  const [word3, setWord3] = useState('man');
  const [analogyResult, setAnalogyResult] = useState<any>(null);
  const [analogyLoading, setAnalogyLoading] = useState(false);

  const handleEncode = async () => {
    setEncodeLoading(true);
    try {
      const result = await testEncode(encodeText, encodeOpt);
      setEncodeResult(result);
    } finally {
      setEncodeLoading(false);
    }
  };

  const handleEntities = async () => {
    setEntityLoading(true);
    try {
      const result = await testEntities(entityText);
      setEntityResult(result);
    } finally {
      setEntityLoading(false);
    }
  };

  const handleAnalogy = async () => {
    setAnalogyLoading(true);
    try {
      const result = await testAnalogy(word1, word2, word3);
      setAnalogyResult(result);
    } finally {
      setAnalogyLoading(false);
    }
  };

  return (
    <div>
      <h2 className="text-3xl font-bold mb-6">🧪 API Testing Interface</h2>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 border-b">
        <button
          onClick={() => setTab('encode')}
          className={`px-4 py-2 font-bold ${tab === 'encode' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-600'}`}
        >
          📝 Encoding
        </button>
        <button
          onClick={() => setTab('entities')}
          className={`px-4 py-2 font-bold ${tab === 'entities' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-600'}`}
        >
          🏷️ Entities
        </button>
        <button
          onClick={() => setTab('analogy')}
          className={`px-4 py-2 font-bold ${tab === 'analogy' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-600'}`}
        >
          🔤 Analogy
        </button>
        <button
          onClick={() => setTab('search')}
          className={`px-4 py-2 font-bold ${tab === 'search' ? 'border-b-2 border-blue-600 text-blue-600' : 'text-gray-600'}`}
        >
          📚 Search
        </button>
      </div>

      {/* Encoding Tab */}
      {tab === 'encode' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-xl font-bold mb-4">📝 Test Text Encoding</h3>
            <textarea
              value={encodeText}
              onChange={(e) => setEncodeText(e.target.value)}
              className="w-full border rounded p-2 mb-3 h-24"
            />
            <select
              value={encodeOpt}
              onChange={(e) => setEncodeOpt(e.target.value)}
              className="w-full border rounded p-2 mb-3"
            >
              <option value="low">Low Optimization</option>
              <option value="medium">Medium Optimization</option>
              <option value="high">High Optimization</option>
            </select>
            <button
              onClick={handleEncode}
              disabled={encodeLoading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition disabled:opacity-50"
            >
              🚀 Encode
            </button>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-xl font-bold mb-4">Results</h3>
            {encodeResult && (
              <>
                {encodeResult.error ? (
                  <div className="text-red-600">❌ {encodeResult.error}</div>
                ) : (
                  <div className="space-y-2">
                    <div className="flex justify-between border-b pb-2">
                      <span>Original Bytes:</span>
                      <span className="font-bold">{encodeText.length}</span>
                    </div>
                    <div className="flex justify-between border-b pb-2">
                      <span>Encoded Bytes:</span>
                      <span className="font-bold">{encodeResult.encoded_bytes || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between border-b pb-2">
                      <span>Compression:</span>
                      <span className="font-bold">{encodeResult.compression_ratio?.toFixed(1) || 'N/A'}x</span>
                    </div>
                  </div>
                )}
                <pre className="mt-4 bg-gray-100 p-3 rounded text-xs overflow-auto max-h-40">
                  {JSON.stringify(encodeResult, null, 2)}
                </pre>
              </>
            )}
          </div>
        </div>
      )}

      {/* Entities Tab */}
      {tab === 'entities' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-xl font-bold mb-4">🏷️ Test Entity Extraction</h3>
            <textarea
              value={entityText}
              onChange={(e) => setEntityText(e.target.value)}
              className="w-full border rounded p-2 mb-3 h-24"
            />
            <button
              onClick={handleEntities}
              disabled={entityLoading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition disabled:opacity-50"
            >
              🏷️ Extract
            </button>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-xl font-bold mb-4">Results</h3>
            {entityResult && (
              <pre className="bg-gray-100 p-3 rounded text-xs overflow-auto max-h-40">
                {JSON.stringify(entityResult, null, 2)}
              </pre>
            )}
          </div>
        </div>
      )}

      {/* Analogy Tab */}
      {tab === 'analogy' && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-bold mb-4">🔤 Test Analogy Solving</h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-3">
            <input
              type="text"
              value={word1}
              onChange={(e) => setWord1(e.target.value)}
              placeholder="Word 1"
              className="border rounded p-2"
            />
            <input
              type="text"
              value={word2}
              onChange={(e) => setWord2(e.target.value)}
              placeholder="Word 2"
              className="border rounded p-2"
            />
            <input
              type="text"
              value={word3}
              onChange={(e) => setWord3(e.target.value)}
              placeholder="Word 3"
              className="border rounded p-2"
            />
            <button
              onClick={handleAnalogy}
              disabled={analogyLoading}
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition disabled:opacity-50"
            >
              🔤 Solve
            </button>
          </div>
          {analogyResult && (
            <pre className="bg-gray-100 p-3 rounded text-xs overflow-auto max-h-40">
              {JSON.stringify(analogyResult, null, 2)}
            </pre>
          )}
        </div>
      )}

      {/* Search Tab */}
      {tab === 'search' && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-bold mb-4">📚 Semantic Search</h3>
          <p className="text-gray-600">Semantic search functionality coming soon...</p>
        </div>
      )}
    </div>
  );
}
