# ΣLANG JavaScript SDK

Full-featured JavaScript/TypeScript client for the [ΣLANG](https://github.com/iamthegreatdestroyer/sigmalang) semantic compression API.

## Installation

```bash
npm install sigmalang
# or
yarn add sigmalang
```

Or for a local dev setup, copy `sigmalang.js` into your project.

## Quick Start

```javascript
const { SigmaLang } = require("sigmalang");

const client = new SigmaLang({ baseUrl: "http://localhost:8000" });

// Encode text
const result = await client.encode(
  "Create a Python function that sorts a list.",
);
console.log(result);
// { encoded: 'Σ001◈Σ042◈Σ017...', ratio: 14.3, strategy: 'pattern' }

// Decode back
const decoded = await client.decode(result.encoded);
console.log(decoded.decoded);
// 'Create a Python function that sorts a list.'

// Analogy: king - man + woman = ?
const analogy = await client.analogy("king", "man", "woman");
console.log(analogy.answer); // 'queen'

// Semantic search
const search = await client.search("sorting algorithms", { k: 5 });
search.results.forEach((r) => console.log(r.text, r.score));
```

## TypeScript

```typescript
import { SigmaLang, SigmaLangError } from "sigmalang";

const client = new SigmaLang({ baseUrl: "http://localhost:8000" });

const result = await client.encode("Hello world");
console.log(result.ratio);
```

## API Reference

### `new SigmaLang(opts?)`

| Option    | Type     | Default                   | Description              |
| --------- | -------- | ------------------------- | ------------------------ |
| `baseUrl` | `string` | `'http://localhost:8000'` | API base URL             |
| `apiKey`  | `string` | `null`                    | Optional API key header  |
| `timeout` | `number` | `30000`                   | Request timeout (ms)     |
| `retries` | `number` | `3`                       | Auto-retry on 5xx errors |

### Core Methods

| Method                      | Description                     |
| --------------------------- | ------------------------------- |
| `encode(text, opts?)`       | Compress text to Σ-primitives   |
| `decode(encoded)`           | Decompress Σ-primitives         |
| `encodeBatch(texts, opts?)` | Batch encode texts              |
| `roundTrip(text)`           | Encode + decode with comparison |

### Analogy Methods

| Method                    | Description                          |
| ------------------------- | ------------------------------------ |
| `analogy(a, b, c, opts?)` | Solve A:B::C:?                       |
| `analogyExplain(a, b)`    | Explain relationship between A and B |

### Search Methods

| Method                    | Description                  |
| ------------------------- | ---------------------------- |
| `search(query, opts?)`    | Semantic search              |
| `indexDocument(id, text)` | Add document to search index |

### NLP Methods

| Method                     | Description                  |
| -------------------------- | ---------------------------- |
| `extractEntities(text)`    | Named entity recognition     |
| `extractRelations(text)`   | Semantic relation extraction |
| `embed(text)`              | Get HD embedding vector      |
| `similarity(textA, textB)` | Compute cosine similarity    |
| `embedBatch(texts)`        | Batch embed texts            |

### Utility Methods

| Method          | Description               |
| --------------- | ------------------------- |
| `analyze(text)` | Compression statistics    |
| `health()`      | Health check              |
| `status()`      | System status             |
| `version()`     | Server version info       |
| `metrics()`     | Prometheus metrics (text) |

### Streaming (WebSocket)

```javascript
const { SigmaLangStream } = require("sigmalang");

const stream = new SigmaLangStream("ws://localhost:8000");

await stream.encode("Long document text...", (chunk) => {
  if (!chunk.done) {
    console.log("Chunk:", chunk.encoded);
  } else {
    console.log("Stream complete");
  }
});
```

## Error Handling

```javascript
const {
  SigmaLang,
  SigmaLangAPIError,
  SigmaLangConnectionError,
} = require("sigmalang");

const client = new SigmaLang();
try {
  const result = await client.encode("test");
} catch (err) {
  if (err instanceof SigmaLangAPIError) {
    console.error(`API error ${err.statusCode}:`, err.detail);
  } else if (err instanceof SigmaLangConnectionError) {
    console.error("Cannot connect:", err.message);
  }
}
```

## License

MIT
