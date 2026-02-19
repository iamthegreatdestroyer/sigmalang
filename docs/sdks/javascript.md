# JavaScript/TypeScript SDK

The JavaScript SDK provides browser and Node.js support for ΣLANG.

## Installation

```bash
npm install sigmalang
# or
yarn add sigmalang
# or
pnpm add sigmalang
```

## Quick Start

```javascript
import { SigmaEncoder } from 'sigmalang';

const encoder = new SigmaEncoder();
const text = "Machine learning transforms data into insights";
const encoded = await encoder.encode(text);

console.log(`Compression: ${text.length / encoded.length}x`);
```

## TypeScript

```typescript
import { SigmaEncoder, EncodingResult } from 'sigmalang';

const encoder = new SigmaEncoder();
const result: EncodingResult = await encoder.encode("text");

console.log(result.compressionRatio);
```

## Browser Usage

```html
<script src="https://cdn.sigmalang.io/sigmalang.umd.js"></script>
<script>
  const encoder = new SIGMALANG.SigmaEncoder();
  encoder.encode("text").then(result => {
    console.log('Compression:', result.compressionRatio);
  });
</script>
```

## Examples

### Node.js

```javascript
const { SigmaEncoder } = require('sigmalang');

async function main() {
  const encoder = new SigmaEncoder();
  const encoded = await encoder.encode("Hello, World!");
  console.log('Encoded:', encoded);
}

main();
```

### React

```jsx
import { useState } from 'react';
import { SigmaEncoder } from 'sigmalang';

export function CompressionDemo() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);

  const handleEncode = async () => {
    const encoder = new SigmaEncoder();
    const encoded = await encoder.encode(text);
    setResult(encoded);
  };

  return (
    <div>
      <textarea value={text} onChange={e => setText(e.target.value)} />
      <button onClick={handleEncode}>Encode</button>
      {result && <p>Compression: {result.compressionRatio}x</p>}
    </div>
  );
}
```

### Vue

```vue
<template>
  <div>
    <textarea v-model="text" />
    <button @click="encode">Encode</button>
    <p v-if="result">Compression: {{ result.compressionRatio }}x</p>
  </div>
</template>

<script>
import { SigmaEncoder } from 'sigmalang';

export default {
  data() {
    return {
      text: '',
      result: null,
      encoder: new SigmaEncoder()
    };
  },
  methods: {
    async encode() {
      this.result = await this.encoder.encode(this.text);
    }
  }
};
</script>
```

## API Reference

### SigmaEncoder

```typescript
class SigmaEncoder {
  encode(text: string, options?: EncodingOptions): Promise<EncodedResult>;
  decode(data: Uint8Array): Promise<string>;
  encodeFile(file: File): Promise<EncodedResult>;
}
```

### Options

```typescript
interface EncodingOptions {
  optimization?: 'low' | 'medium' | 'high';
  cache?: boolean;
}
```

## Performance

| Optimization | Speed | Compression |
|---|---|---|
| Low | Fast | 5-8x |
| Medium | Balanced | 10-20x |
| High | Slow | 20-50x |

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Documentation

- [Getting Started](../getting-started/installation.md)
- [Basic Usage](../getting-started/basic-usage.md)
- [REST API](../api/rest-api.md)

## Support

- GitHub Issues: [iamthegreatdestroyer/sigmalang](https://github.com/iamthegreatdestroyer/sigmalang)
- NPM Package: [sigmalang](https://www.npmjs.com/package/sigmalang)
