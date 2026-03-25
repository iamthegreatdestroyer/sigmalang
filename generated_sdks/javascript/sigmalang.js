/**
 * ΣLANG JavaScript SDK
 * Sub-Linear Algorithmic Neural Glyph Language
 *
 * Full-featured client for the SigmaLang API.
 * Compatible with Node.js >= 14 and modern browsers (via fetch).
 *
 * @version 1.0.0
 * @license MIT
 */

"use strict";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

class SigmaLangError extends Error {
  constructor(message, statusCode = null, detail = null) {
    super(message);
    this.name = "SigmaLangError";
    this.statusCode = statusCode;
    this.detail = detail;
  }
}

class SigmaLangAPIError extends SigmaLangError {
  constructor(statusCode, body) {
    const msg = body?.detail || body?.message || `HTTP ${statusCode}`;
    super(msg, statusCode, body);
    this.name = "SigmaLangAPIError";
  }
}

class SigmaLangConnectionError extends SigmaLangError {
  constructor(cause) {
    super(`Cannot connect to SigmaLang API: ${cause}`);
    this.name = "SigmaLangConnectionError";
    this.cause = cause;
  }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

class SigmaLang {
  /**
   * Create a SigmaLang API client.
   *
   * @param {Object} opts
   * @param {string} [opts.baseUrl='http://localhost:8000']  API base URL
   * @param {string} [opts.apiKey]                           Optional API key
   * @param {number} [opts.timeout=30000]                    Request timeout ms
   * @param {number} [opts.retries=3]                        Auto-retry count
   */
  constructor({
    baseUrl = "http://localhost:8000",
    apiKey = null,
    timeout = 30000,
    retries = 3,
  } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.timeout = timeout;
    this.retries = retries;
  }

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------

  _headers(extra = {}) {
    const h = { "Content-Type": "application/json", ...extra };
    if (this.apiKey) h["X-API-Key"] = this.apiKey;
    return h;
  }

  async _request(method, path, body = null, attempt = 0) {
    const url = `${this.baseUrl}${path}`;
    const opts = {
      method,
      headers: this._headers(),
      signal: AbortSignal.timeout
        ? AbortSignal.timeout(this.timeout)
        : undefined,
    };
    if (body !== null) opts.body = JSON.stringify(body);

    let res;
    try {
      res = await fetch(url, opts);
    } catch (err) {
      if (attempt < this.retries) {
        await this._sleep(Math.pow(2, attempt) * 200);
        return this._request(method, path, body, attempt + 1);
      }
      throw new SigmaLangConnectionError(err.message);
    }

    const text = await res.text();
    let json;
    try {
      json = JSON.parse(text);
    } catch {
      json = { detail: text };
    }

    if (!res.ok) {
      if (res.status >= 500 && attempt < this.retries) {
        await this._sleep(Math.pow(2, attempt) * 200);
        return this._request(method, path, body, attempt + 1);
      }
      throw new SigmaLangAPIError(res.status, json);
    }
    return json;
  }

  _sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  async _get(path) {
    return this._request("GET", path);
  }
  async _post(path, body) {
    return this._request("POST", path, body);
  }

  // -------------------------------------------------------------------------
  // Core endpoints
  // -------------------------------------------------------------------------

  /**
   * Encode text into Σ-primitives.
   *
   * @param {string} text               Input text to compress
   * @param {Object} [opts]
   * @param {string} [opts.strategy]    'auto'|'pattern'|'reference'|'delta'|'full'
   * @param {boolean} [opts.returnTree] Include semantic tree in response
   * @returns {Promise<{encoded: string, ratio: number, strategy: string}>}
   */
  async encode(text, { strategy = "auto", returnTree = false } = {}) {
    return this._post("/encode", { text, strategy, return_tree: returnTree });
  }

  /**
   * Decode Σ-primitives back to text.
   *
   * @param {string} encoded  Encoded string from encode()
   * @returns {Promise<{decoded: string, success: boolean}>}
   */
  async decode(encoded) {
    return this._post("/decode", { encoded });
  }

  /**
   * Encode a batch of texts.
   *
   * @param {string[]} texts     Array of texts to encode
   * @param {Object}   [opts]    Same options as encode()
   * @returns {Promise<Array>}
   */
  async encodeBatch(texts, opts = {}) {
    return this._post("/encode/batch", { texts, ...opts });
  }

  /**
   * Round-trip: encode then decode and return both.
   *
   * @param {string} text
   * @returns {Promise<{original: string, encoded: string, decoded: string, ratio: number}>}
   */
  async roundTrip(text) {
    const enc = await this.encode(text);
    const dec = await this.decode(enc.encoded);
    return { original: text, ...enc, decoded: dec.decoded };
  }

  // -------------------------------------------------------------------------
  // Analogy
  // -------------------------------------------------------------------------

  /**
   * Solve analogy A:B::C:?
   *
   * @param {string} a
   * @param {string} b
   * @param {string} c
   * @param {Object} [opts]
   * @param {number} [opts.topK=1]    Number of candidates to return
   * @returns {Promise<{answer: string, confidence: number, candidates: Array}>}
   */
  async analogy(a, b, c, { topK = 1 } = {}) {
    return this._post("/analogy", { a, b, c, top_k: topK });
  }

  /**
   * Explain an analogy relationship.
   *
   * @param {string} a
   * @param {string} b
   * @returns {Promise<{relationship: string, similarity: number, explanation: string}>}
   */
  async analogyExplain(a, b) {
    return this._post("/analogy/explain", { a, b });
  }

  // -------------------------------------------------------------------------
  // Semantic Search
  // -------------------------------------------------------------------------

  /**
   * Search for semantically similar content.
   *
   * @param {string} query
   * @param {Object} [opts]
   * @param {number} [opts.k=5]         Top K results
   * @param {number} [opts.threshold]   Minimum similarity threshold (0–1)
   * @returns {Promise<{results: Array<{id, text, score}>}>}
   */
  async search(query, { k = 5, threshold = null } = {}) {
    const body = { query, k };
    if (threshold !== null) body.threshold = threshold;
    return this._post("/search", body);
  }

  /**
   * Add a document to the search index.
   *
   * @param {string} documentId
   * @param {string} text
   * @returns {Promise}
   */
  async indexDocument(documentId, text) {
    return this._post("/search/index", { id: documentId, text });
  }

  // -------------------------------------------------------------------------
  // Entities & Relations
  // -------------------------------------------------------------------------

  /**
   * Extract named entities from text.
   *
   * @param {string} text
   * @returns {Promise<{entities: Array<{text, type, start, end, confidence}>}>}
   */
  async extractEntities(text) {
    return this._post("/entities", { text });
  }

  /**
   * Extract semantic relations from text.
   *
   * @param {string} text
   * @returns {Promise<{relations: Array<{subject, predicate, object, confidence}>}>}
   */
  async extractRelations(text) {
    return this._post("/relations", { text });
  }

  // -------------------------------------------------------------------------
  // Embeddings
  // -------------------------------------------------------------------------

  /**
   * Get HD semantic embedding vector for text.
   *
   * @param {string} text
   * @returns {Promise<{embedding: number[], dimension: number}>}
   */
  async embed(text) {
    return this._post("/embed", { text });
  }

  /**
   * Compute semantic similarity between two texts.
   *
   * @param {string} textA
   * @param {string} textB
   * @returns {Promise<{similarity: number, distance: number}>}
   */
  async similarity(textA, textB) {
    return this._post("/similarity", { text_a: textA, text_b: textB });
  }

  /**
   * Embed a batch of texts.
   *
   * @param {string[]} texts
   * @returns {Promise<{embeddings: number[][]}>}
   */
  async embedBatch(texts) {
    return this._post("/embed/batch", { texts });
  }

  // -------------------------------------------------------------------------
  // Compression Analytics
  // -------------------------------------------------------------------------

  /**
   * Get compression statistics for text.
   *
   * @param {string} text
   * @returns {Promise<{ratio, entropy, strategy, timing_ms}>}
   */
  async analyze(text) {
    return this._post("/analyze", { text });
  }

  // -------------------------------------------------------------------------
  // Health & Metadata
  // -------------------------------------------------------------------------

  /**
   * Health check.
   * @returns {Promise<{status: string, version: string, uptime_s: number}>}
   */
  async health() {
    return this._get("/health");
  }

  /**
   * Detailed system status (liveness/readiness).
   * @returns {Promise}
   */
  async status() {
    return this._get("/status");
  }

  /**
   * Get server version info.
   * @returns {Promise<{version, python_version, build_date}>}
   */
  async version() {
    return this._get("/version");
  }

  /**
   * Prometheus metrics (text format).
   * @returns {Promise<string>}
   */
  async metrics() {
    const res = await fetch(`${this.baseUrl}/metrics`);
    return res.text();
  }
}

// ---------------------------------------------------------------------------
// Streaming client (WebSocket)
// ---------------------------------------------------------------------------

class SigmaLangStream {
  /**
   * Real-time streaming encoder via WebSocket.
   *
   * @param {string} [baseUrl='ws://localhost:8000']
   */
  constructor(baseUrl = "ws://localhost:8000") {
    this.baseUrl = baseUrl.replace(/^http/, "ws").replace(/\/$/, "");
  }

  /**
   * Stream-encode text, receiving results chunk by chunk.
   *
   * @param {string}   text
   * @param {Function} onChunk   Called for each encoded chunk: ({chunk, encoded, done})
   * @returns {Promise<void>}
   */
  encode(text, onChunk) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(`${this.baseUrl}/ws/encode`);
      ws.onopen = () => ws.send(JSON.stringify({ text }));
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        onChunk(data);
        if (data.done) {
          ws.close();
          resolve();
        }
      };
      ws.onerror = (err) =>
        reject(new SigmaLangConnectionError(err.message || "WebSocket error"));
      ws.onclose = () => resolve();
    });
  }
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

if (typeof module !== "undefined" && module.exports) {
  // Node.js
  module.exports = {
    SigmaLang,
    SigmaLangStream,
    SigmaLangError,
    SigmaLangAPIError,
    SigmaLangConnectionError,
  };
} else if (typeof window !== "undefined") {
  // Browser global
  window.SigmaLang = SigmaLang;
  window.SigmaLangStream = SigmaLangStream;
}
