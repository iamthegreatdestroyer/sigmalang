# Command-Line Interface (CLI) Reference

## Installation

The CLI is available after installing the package:

```bash
pip install sigmalang
sigmalang --version
```

## Global Options

All commands support these options:

```bash
--help              Show command help
--version           Show version
--verbose           Enable verbose output
--log-level LEVEL   Set log level (DEBUG, INFO, WARNING, ERROR)
```

## Commands

### encode

Encode text to ΣLANG format.

#### Usage

```bash
sigmalang encode [OPTIONS] [TEXT]
```

#### Options

- `-i, --input FILE`: Read from file instead of argument
- `-o, --output FILE`: Write to file instead of stdout
- `--optimization LEVEL`: Compression level (low, medium, high) [default: medium]
- `--show-stats`: Display compression statistics
- `--benchmark`: Measure encoding time

#### Examples

**Encode text from command line:**
```bash
sigmalang encode "The quick brown fox jumps over the lazy dog"
```

Output:
```
Original: 44 bytes
Encoded: 16 bytes
Compression: 2.75x
```

**Encode from file:**
```bash
sigmalang encode -i input.txt -o encoded.bin --show-stats
```

Output:
```
Reading from: input.txt
Writing to: encoded.bin
Original: 1024 bytes
Encoded: 128 bytes
Compression: 8.0x
Processing time: 5.2ms
```

**High compression:**
```bash
sigmalang encode -i large.txt -o compressed.bin --optimization high
```

### decode

Decode ΣLANG format back to text.

#### Usage

```bash
sigmalang decode [OPTIONS] FILE
```

#### Options

- `-i, --input FILE`: Input encoded file
- `-o, --output FILE`: Output decoded file [default: stdout]
- `--show-metadata`: Display encoding metadata

#### Examples

**Decode file:**
```bash
sigmalang decode -i encoded.bin -o decoded.txt
```

**Decode with metadata:**
```bash
sigmalang decode -i encoded.bin --show-metadata
```

Output:
```
Decoded: decoded.txt
Metadata:
  Optimization: high
  Original size: 1024
  Encoded size: 128
  Timestamp: 2026-02-19T12:00:00Z
```

### entities

Extract entities from text.

#### Usage

```bash
sigmalang entities [OPTIONS] [TEXT]
```

#### Options

- `-i, --input FILE`: Read from file
- `-o, --output FILE`: Write to file
- `--types TYPES`: Filter by entity type (comma-separated)
- `--relationships`: Include relationship information
- `--format FORMAT`: Output format (json, csv, table) [default: table]

#### Examples

**Extract entities:**
```bash
sigmalang entities "Apple Inc is located in Cupertino, California"
```

Output:
```
Entity                Type             Confidence
──────────────────────────────────────────────────
Apple Inc             ORGANIZATION     0.98
Cupertino             LOCATION         0.95
California            LOCATION         0.97
```

**With relationships:**
```bash
sigmalang entities "Steve Jobs founded Apple Inc" --relationships
```

Output:
```
Entities:
  - Steve Jobs (PERSON, 0.96)
  - Apple Inc (ORGANIZATION, 0.98)

Relationships:
  - Steve Jobs [FOUNDED] Apple Inc
```

**JSON output:**
```bash
sigmalang entities "Apple is in California" --format json
```

Output:
```json
{
  "entities": [
    {"text": "Apple", "type": "ORGANIZATION", "confidence": 0.98},
    {"text": "California", "type": "LOCATION", "confidence": 0.97}
  ]
}
```

### analogy

Solve word analogies.

#### Usage

```bash
sigmalang analogy [OPTIONS]
```

#### Options

- `--word1 WORD`: First word (required)
- `--word2 WORD`: Second word (required)
- `--word3 WORD`: Third word (required)
- `--top K`: Show top K candidates [default: 1]
- `--threshold SCORE`: Confidence threshold (0-1) [default: 0.0]

#### Examples

**Simple analogy:**
```bash
sigmalang analogy --word1 king --word2 queen --word3 man
```

Output:
```
Answer: woman
Confidence: 0.94
```

**Top 5 candidates:**
```bash
sigmalang analogy --word1 king --word2 queen --word3 man --top 5
```

Output:
```
1. woman      (0.94)
2. lady       (0.89)
3. female     (0.82)
4. matriarch  (0.75)
5. princess   (0.68)
```

**With confidence filter:**
```bash
sigmalang analogy --word1 good --word2 bad --word3 hot --top 5 --threshold 0.8
```

### search

Search through indexed data semantically.

#### Usage

```bash
sigmalang search [OPTIONS] QUERY
```

#### Options

- `-d, --database DB`: Database file or path
- `-o, --output FILE`: Output results to file
- `--limit N`: Maximum results [default: 10]
- `--threshold SCORE`: Similarity threshold [default: 0.7]
- `--format FORMAT`: Output format (json, table)

#### Examples

**Search in database:**
```bash
sigmalang search --database documents.db "technology companies"
```

Output:
```
Result                                              Similarity
────────────────────────────────────────────────────────────────
Apple Inc manufactures consumer electronics        0.92
Microsoft develops software solutions              0.88
Google provides search services                    0.85
```

**JSON output:**
```bash
sigmalang search "machine learning" --format json
```

### server

Start the API server.

#### Usage

```bash
sigmalang-server [OPTIONS]
```

#### Options

- `--host HOST`: Bind address [default: 0.0.0.0]
- `--port PORT`: Listen port [default: 8000]
- `--workers N`: Number of worker processes [default: 4]
- `--reload`: Enable auto-reload on code changes (dev only)
- `--debug`: Enable debug mode

#### Examples

**Start server:**
```bash
sigmalang-server
```

**Custom port:**
```bash
sigmalang-server --port 8001
```

**Development mode:**
```bash
sigmalang-server --reload --debug
```

### version

Display version information.

#### Usage

```bash
sigmalang --version
sigmalang version
```

Output:
```
ΣLANG 1.0.0
Python 3.10.8
```

### help

Display help information.

#### Usage

```bash
sigmalang --help
sigmalang COMMAND --help
```

## Input/Output Formats

### Text Input

Read from file or stdin:

```bash
# From file
sigmalang encode -i document.txt -o output.bin

# From stdin (pipe)
cat document.txt | sigmalang encode -o output.bin

# From argument
sigmalang encode "Text here"
```

### Output Formats

**Binary (default):**
```bash
sigmalang encode -i input.txt -o output.bin
```

**Base64:**
```bash
sigmalang encode -i input.txt | base64
```

**JSON:**
```bash
sigmalang encode "text" --format json
```

## Batch Processing

### Process Multiple Files

```bash
for file in *.txt; do
    sigmalang encode -i "$file" -o "${file%.txt}.bin"
done
```

### Parallel Processing

```bash
find . -name "*.txt" | parallel sigmalang encode -i {} -o {.}.bin
```

## Environment Variables

- `SIGMALANG_LOG_LEVEL`: Default log level (DEBUG, INFO, WARNING, ERROR)
- `SIGMALANG_CACHE_ENABLED`: Enable caching (true/false)
- `SIGMALANG_OPTIMIZATION`: Default optimization level (low, medium, high)

## Examples

### Log Compression

```bash
# Compress application logs
cat application.log | \
  sigmalang encode --optimization high -o logs.bin

# Size before/after
ls -lh application.log logs.bin
```

### Batch Entity Extraction

```bash
# Extract entities from multiple documents
for doc in reports/*.txt; do
    echo "Processing $doc"
    sigmalang entities -i "$doc" --format json > "${doc%.txt}.json"
done
```

### Analogy Batch Testing

```bash
# Read analogies from file and solve
while IFS=',' read -r w1 w2 w3; do
    sigmalang analogy --word1 "$w1" --word2 "$w2" --word3 "$w3"
done < analogies.csv
```

### Integration with Other Tools

```bash
# Pipe to other commands
sigmalang encode "Input text" | od -A x -t x1z -v

# Combine with compression
sigmalang encode -i large.txt | gzip > compressed.bin.gz

# Chain multiple operations
cat data.txt | \
  sigmalang encode --optimization high | \
  base64 | \
  curl -X POST --data-binary @- http://api.example.com/upload
```

## Troubleshooting

### Command not found

```bash
# Ensure installation
pip install --upgrade sigmalang

# Check installation
which sigmalang
pip show sigmalang
```

### File not found

```bash
# Use absolute paths
sigmalang encode -i /absolute/path/to/file.txt

# Check current directory
pwd
ls -la file.txt
```

### Permission denied

```bash
# Check file permissions
chmod 644 input.txt
chmod 755 output_dir/

# Use sudo if needed (not recommended)
sudo sigmalang encode -i /restricted/file.txt
```

## Performance Tips

1. **Use high optimization** for large files:
   ```bash
   sigmalang encode -i large.txt -o output.bin --optimization high
   ```

2. **Batch process** instead of individual commands:
   ```bash
   # Better: one tool invocation
   find . -name "*.txt" | parallel sigmalang encode

   # Avoid: multiple tool invocations
   for f in *.txt; do sigmalang encode "$f"; done
   ```

3. **Use pipes** to avoid intermediate files:
   ```bash
   # Better
   cat data.txt | sigmalang encode | gzip

   # Avoid
   sigmalang encode -i data.txt -o temp.bin
   gzip temp.bin
   ```

## Next Steps

- Try [Basic Usage](../getting-started/basic-usage.md) examples
- Explore [REST API](rest-api.md)
- Read [Python API](python-api.md) reference
