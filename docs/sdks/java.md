# Java SDK

The Java SDK provides enterprise-grade access to ΣLANG for Java and JVM applications.

## Installation

### Maven

```xml
<dependency>
  <groupId>io.sigmalang</groupId>
  <artifactId>sigmalang-java</artifactId>
  <version>1.0.0</version>
</dependency>
```

### Gradle

```gradle
dependencies {
  implementation 'io.sigmalang:sigmalang-java:1.0.0'
}
```

### SBT

```scala
libraryDependencies += "io.sigmalang" % "sigmalang-java" % "1.0.0"
```

## Quick Start

```java
import io.sigmalang.SigmaEncoder;

public class Example {
  public static void main(String[] args) throws Exception {
    SigmaEncoder encoder = new SigmaEncoder();
    String text = "Machine learning transforms data into insights";

    byte[] encoded = encoder.encode(text);
    double ratio = (double) text.length() / encoded.length;

    System.out.println("Compression: " + ratio + "x");
  }
}
```

## Configuration

```java
import io.sigmalang.SigmaConfig;
import io.sigmalang.OptimizationLevel;

SigmaConfig config = SigmaConfig.builder()
  .optimizationLevel(OptimizationLevel.HIGH)
  .cacheEnabled(true)
  .cacheTtl(3600)
  .build();

SigmaEncoder encoder = new SigmaEncoder(config);
```

## Examples

### Spring Boot Integration

```java
import org.springframework.stereotype.Service;
import io.sigmalang.SigmaEncoder;

@Service
public class CompressionService {
  private final SigmaEncoder encoder = new SigmaEncoder();

  public EncodingResult encode(String text) throws Exception {
    byte[] encoded = encoder.encode(text);
    return new EncodingResult(text.length(), encoded.length);
  }
}

@RestController
@RequestMapping("/api")
public class CompressionController {
  @Autowired
  private CompressionService service;

  @PostMapping("/encode")
  public EncodingResult encode(@RequestBody String text) throws Exception {
    return service.encode(text);
  }
}
```

### Batch Processing

```java
import io.sigmalang.BatchProcessor;

BatchProcessor processor = new BatchProcessor(100);
List<String> texts = Arrays.asList(
  "text1", "text2", "text3"
);

List<byte[]> results = processor.encodeAll(texts);
for (byte[] encoded : results) {
  System.out.println("Encoded: " + encoded.length + " bytes");
}
```

### File Compression

```java
import io.sigmalang.FileCompressor;
import java.nio.file.Path;
import java.nio.file.Paths;

FileCompressor compressor = new FileCompressor();
Path input = Paths.get("input.txt");
Path output = Paths.get("output.bin");

compressor.compressFile(input, output);
System.out.println("File compressed successfully");
```

### Analogy Engine

```java
import io.sigmalang.AnalogyEngine;

AnalogyEngine engine = new AnalogyEngine();
String answer = engine.solve("king", "queen", "man");

System.out.println("Answer: " + answer); // "woman"
```

## API Reference

### SigmaEncoder

```java
public class SigmaEncoder {
  public byte[] encode(String text) throws EncodingException;
  public String decode(byte[] data) throws DecodingException;
  public EncodingResult encodeWithStats(String text);
}
```

### EncodingResult

```java
public class EncodingResult {
  public String getText();
  public byte[] getEncoded();
  public int getOriginalBytes();
  public int getEncodedBytes();
  public double getCompressionRatio();
  public long getProcessingTimeMs();
}
```

### Configuration

```java
public class SigmaConfig {
  public OptimizationLevel getOptimizationLevel();
  public boolean isCacheEnabled();
  public long getCacheTtl();
  public int getNumWorkers();

  public static SigmaConfig.Builder builder();
}
```

## Performance Tuning

### Thread Pool Configuration

```java
SigmaConfig config = SigmaConfig.builder()
  .numWorkers(8)  // Match CPU cores
  .build();
```

### Caching Strategy

```java
SigmaConfig config = SigmaConfig.builder()
  .cacheEnabled(true)
  .cacheTtl(7200)  // 2 hours
  .build();
```

## Error Handling

```java
try {
  byte[] encoded = encoder.encode(text);
} catch (EncodingException e) {
  System.err.println("Encoding failed: " + e.getMessage());
  // Handle error
} catch (Exception e) {
  System.err.println("Unexpected error: " + e.getMessage());
  e.printStackTrace();
}
```

## Supported Java Versions

- Java 8+
- Java 11 (LTS)
- Java 17 (LTS)
- Java 21 (LTS)

## Performance Benchmarks

| Optimization | Throughput | Compression |
|---|---|---|
| Low | ~50 MB/s | 5-8x |
| Medium | ~20 MB/s | 10-20x |
| High | ~5 MB/s | 20-50x |

## Testing

```java
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class EncoderTest {
  private SigmaEncoder encoder = new SigmaEncoder();

  @Test
  public void testBasicEncoding() throws Exception {
    String text = "Hello, World!";
    byte[] encoded = encoder.encode(text);

    assertNotNull(encoded);
    assertTrue(encoded.length > 0);
  }

  @Test
  public void testCompressionRatio() throws Exception {
    String text = "x".repeat(1000);
    byte[] encoded = encoder.encode(text);
    double ratio = (double) text.length() / encoded.length;

    assertTrue(ratio > 5.0, "Should achieve at least 5x compression");
  }
}
```

## Documentation

- [Getting Started](../getting-started/installation.md)
- [API Documentation](../api/overview.md)
- [Configuration Guide](../deployment/docker.md)

## Maven Central

- [io.sigmalang:sigmalang-java](https://mvnrepository.com/artifact/io.sigmalang/sigmalang-java)

## Support

- GitHub: [iamthegreatdestroyer/sigmalang](https://github.com/iamthegreatdestroyer/sigmalang)
- Issues: [GitHub Issues](https://github.com/iamthegreatdestroyer/sigmalang/issues)
