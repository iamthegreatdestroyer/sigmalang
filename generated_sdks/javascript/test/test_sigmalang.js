"use strict";
/**
 * Tests for the ΣLANG JavaScript SDK
 * Run with: node test/test_sigmalang.js
 * (API server must be running on localhost:8000)
 *
 * These tests are integration tests - they hit a live API.
 * For unit tests, mock the fetch() function.
 */

const {
  SigmaLang,
  SigmaLangError,
  SigmaLangAPIError,
  SigmaLangConnectionError,
} = require("../sigmalang");

// ---------------------------------------------------------------------------
// Minimal test runner (no external deps required)
// ---------------------------------------------------------------------------
let passed = 0,
  failed = 0;

async function test(name, fn) {
  try {
    await fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (err) {
    console.error(`  ✗ ${name}: ${err.message}`);
    failed++;
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg || "Assertion failed");
}

function assertDeepEqual(a, b, msg) {
  const sa = JSON.stringify(a),
    sb = JSON.stringify(b);
  if (sa !== sb) throw new Error(msg || `${sa} !== ${sb}`);
}

// ---------------------------------------------------------------------------
// Unit tests (no live server needed)
// ---------------------------------------------------------------------------

async function testClientInit() {
  console.log("\n── Client Initialization ──");

  await test("default options", () => {
    const c = new SigmaLang();
    assert(c.baseUrl === "http://localhost:8000", "default baseUrl");
    assert(c.timeout === 30000, "default timeout");
    assert(c.retries === 3, "default retries");
    assert(c.apiKey === null, "default apiKey");
  });

  await test("custom options", () => {
    const c = new SigmaLang({
      baseUrl: "http://example.com:9000/",
      apiKey: "tk_123",
      timeout: 5000,
      retries: 1,
    });
    assert(c.baseUrl === "http://example.com:9000", "trailing slash stripped");
    assert(c.apiKey === "tk_123", "apiKey set");
    assert(c.timeout === 5000, "timeout set");
    assert(c.retries === 1, "retries set");
  });

  await test("headers include API key when set", () => {
    const c = new SigmaLang({ apiKey: "my-key" });
    const h = c._headers();
    assert(h["X-API-Key"] === "my-key", "API key in header");
    assert(h["Content-Type"] === "application/json", "Content-Type set");
  });

  await test("headers no API key when not set", () => {
    const c = new SigmaLang();
    const h = c._headers();
    assert(!("X-API-Key" in h), "no spurious API key header");
  });
}

async function testErrorClasses() {
  console.log("\n── Error Classes ──");

  await test("SigmaLangError is Error", () => {
    const e = new SigmaLangError("test", 400, { detail: "bad" });
    assert(e instanceof Error, "is Error");
    assert(e.name === "SigmaLangError", "name");
    assert(e.statusCode === 400, "statusCode");
    assert(e.detail?.detail === "bad", "detail");
  });

  await test("SigmaLangAPIError", () => {
    const e = new SigmaLangAPIError(422, { detail: "Validation error" });
    assert(e instanceof SigmaLangError, "is SigmaLangError");
    assert(e.name === "SigmaLangAPIError", "name");
    assert(e.statusCode === 422, "statusCode");
    assert(e.message === "Validation error", "message from detail");
  });

  await test("SigmaLangConnectionError", () => {
    const e = new SigmaLangConnectionError("ECONNREFUSED");
    assert(e instanceof SigmaLangError, "is SigmaLangError");
    assert(e.name === "SigmaLangConnectionError", "name");
    assert(e.cause === "ECONNREFUSED", "cause");
  });
}

// ---------------------------------------------------------------------------
// Integration tests (require live server)
// ---------------------------------------------------------------------------

async function testLiveAPI(client) {
  console.log("\n── Integration Tests (live server) ──");

  await test("health check returns status ok", async () => {
    const result = await client.health();
    assert(
      result.status === "ok" || result.status === "healthy",
      `status=${result.status}`,
    );
  });

  await test("encode returns string with ratio", async () => {
    const result = await client.encode(
      "Create a Python function to sort a list by length.",
    );
    assert(typeof result.encoded === "string", "encoded is string");
    assert(result.encoded.length > 0, "encoded not empty");
    assert(typeof result.ratio === "number", "ratio is number");
    assert(result.ratio > 1, `compression achieved ratio=${result.ratio}`);
  });

  await test("decode reverses encode", async () => {
    const text = "The quick brown fox jumps over the lazy dog.";
    const enc = await client.encode(text);
    const dec = await client.decode(enc.encoded);
    assert(typeof dec.decoded === "string", "decoded is string");
    // Allow minor whitespace differences
    assert(
      dec.decoded.trim().toLowerCase() === text.trim().toLowerCase(),
      `round-trip mismatch: "${dec.decoded}" vs "${text}"`,
    );
  });

  await test("encode batch processes multiple texts", async () => {
    const texts = ["Hello world", "Python sort list", "Neural networks"];
    const results = await client.encodeBatch(texts);
    assert(
      Array.isArray(results) || typeof results === "object",
      "returns array or object",
    );
  });

  await test("analogy returns answer", async () => {
    const result = await client.analogy("hot", "cold", "light");
    assert(typeof result.answer === "string", "answer is string");
    assert(result.answer.length > 0, "answer not empty");
  });

  await test("embed returns vector", async () => {
    const result = await client.embed("machine learning");
    assert(Array.isArray(result.embedding), "embedding is array");
    assert(result.embedding.length > 0, "embedding has dimensions");
    assert(result.dimension === result.embedding.length, "dimension matches");
  });

  await test("similarity returns float in [-1, 1]", async () => {
    const result = await client.similarity("cat", "dog");
    assert(typeof result.similarity === "number", "similarity is number");
    assert(
      result.similarity >= -1 && result.similarity <= 1,
      `out of range: ${result.similarity}`,
    );
  });

  await test("analyze returns compression stats", async () => {
    const result = await client.analyze(
      "Analyzing semantic compression statistics.",
    );
    assert(typeof result.ratio === "number", "ratio is number");
  });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log("ΣLANG JavaScript SDK Test Suite");
  console.log("=".repeat(50));

  // Unit tests always run
  await testClientInit();
  await testErrorClasses();

  // Integration tests: check if server is reachable
  const client = new SigmaLang({ timeout: 5000, retries: 1 });
  let serverAvailable = false;
  try {
    await client.health();
    serverAvailable = true;
  } catch {
    console.log("\n⚠ API server not reachable — skipping integration tests");
    console.log(
      "  Start with: uvicorn sigmalang.core.api_server:app --port 8000",
    );
  }

  if (serverAvailable) {
    await testLiveAPI(client);
  }

  console.log("\n" + "=".repeat(50));
  console.log(`Results: ${passed} passed, ${failed} failed`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
