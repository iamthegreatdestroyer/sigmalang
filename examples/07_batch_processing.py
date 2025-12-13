#!/usr/bin/env python3
"""
Example 07: Batch Processing
============================

This example demonstrates efficient batch processing of multiple texts
using sigmalang. Batch processing is essential for:

- Processing large document collections
- High-throughput encoding pipelines
- Building search indexes
- Data preprocessing for ML workflows
"""

import time
import json
from pathlib import Path
from typing import Iterator, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import sigmalang components
try:
    from sigmalang.core.encoder import SigmaEncoder, SigmaDecoder
    ENCODER_AVAILABLE = True
except ImportError:
    ENCODER_AVAILABLE = False
    print("Note: SigmaEncoder not available. Using demonstration mode.")


# Sample documents for batch processing
SAMPLE_DOCUMENTS = [
    "Machine learning is transforming how we process data.",
    "Natural language processing enables computers to understand text.",
    "Deep neural networks have revolutionized artificial intelligence.",
    "Computer vision allows machines to interpret visual information.",
    "Reinforcement learning teaches agents through trial and error.",
    "Transfer learning enables knowledge sharing between tasks.",
    "Attention mechanisms improve sequence modeling performance.",
    "Transformer architectures have become the foundation of modern NLP.",
    "Large language models demonstrate emergent reasoning capabilities.",
    "Semantic embeddings capture meaning in vector representations.",
]


class BatchProcessor:
    """
    Efficient batch processor for sigmalang encoding.
    
    Features:
    - Configurable batch sizes
    - Progress tracking
    - Error handling and recovery
    - Memory-efficient streaming
    """
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Number of texts to process in each batch
            max_workers: Maximum parallel workers for processing
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        if ENCODER_AVAILABLE:
            self.encoder = SigmaEncoder()
            self.decoder = SigmaDecoder()
        else:
            self.encoder = None
            self.decoder = None
        
        # Statistics
        self.total_processed = 0
        self.total_errors = 0
        self.processing_time = 0.0
    
    def encode_single(self, text: str) -> dict:
        """Encode a single text."""
        if self.encoder:
            result = self.encoder.encode(text)
            return {
                "text": text,
                "glyphs": result.get("glyphs", []),
                "tokens": result.get("tokens", []),
                "success": True
            }
        else:
            # Demonstration mode
            words = text.split()
            return {
                "text": text,
                "glyphs": [f"Σ{hash(w) % 256}" for w in words],
                "tokens": words,
                "success": True,
                "demo_mode": True
            }
    
    def process_batch(self, texts: list[str]) -> list[dict]:
        """
        Process a batch of texts.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of encoding results
        """
        results = []
        
        for text in texts:
            try:
                result = self.encode_single(text)
                results.append(result)
                self.total_processed += 1
            except Exception as e:
                results.append({
                    "text": text,
                    "error": str(e),
                    "success": False
                })
                self.total_errors += 1
        
        return results
    
    def process_stream(self, texts: Iterator[str]) -> Iterator[dict]:
        """
        Process texts as a stream (memory efficient).
        
        Args:
            texts: Iterator of texts to process
            
        Yields:
            Encoding results one at a time
        """
        batch = []
        
        for text in texts:
            batch.append(text)
            
            if len(batch) >= self.batch_size:
                for result in self.process_batch(batch):
                    yield result
                batch = []
        
        # Process remaining texts
        if batch:
            for result in self.process_batch(batch):
                yield result
    
    def process_parallel(self, texts: list[str]) -> list[dict]:
        """
        Process texts in parallel using thread pool.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of encoding results (order preserved)
        """
        results = [None] * len(texts)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks with index
            future_to_idx = {
                executor.submit(self.encode_single, text): idx
                for idx, text in enumerate(texts)
            }
            
            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                    self.total_processed += 1
                except Exception as e:
                    results[idx] = {
                        "text": texts[idx],
                        "error": str(e),
                        "success": False
                    }
                    self.total_errors += 1
        
        return results
    
    def process_file(self, input_path: Path, output_path: Optional[Path] = None) -> dict:
        """
        Process a text file line by line.
        
        Args:
            input_path: Path to input text file
            output_path: Optional path for JSON output
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        results = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for result in self.process_stream(line.strip() for line in f if line.strip()):
                results.append(result)
        
        self.processing_time = time.time() - start_time
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        
        return {
            "input_file": str(input_path),
            "output_file": str(output_path) if output_path else None,
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "processing_time": self.processing_time,
            "throughput": self.total_processed / self.processing_time if self.processing_time > 0 else 0
        }
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(1, self.total_processed + self.total_errors),
            "processing_time": self.processing_time,
            "throughput": self.total_processed / max(0.001, self.processing_time)
        }


def example_basic_batch():
    """Demonstrate basic batch processing."""
    print("=" * 60)
    print("Basic Batch Processing")
    print("=" * 60)
    
    processor = BatchProcessor(batch_size=4)
    
    start = time.time()
    results = processor.process_batch(SAMPLE_DOCUMENTS)
    elapsed = time.time() - start
    
    print(f"\nProcessed {len(results)} documents in {elapsed:.3f}s")
    print(f"Throughput: {len(results)/elapsed:.1f} docs/second")
    
    # Show sample result
    if results:
        sample = results[0]
        print(f"\nSample result:")
        print(f"  Text: {sample['text'][:50]}...")
        print(f"  Glyphs: {len(sample.get('glyphs', []))} symbols")


def example_streaming():
    """Demonstrate memory-efficient streaming."""
    print("\n" + "=" * 60)
    print("Streaming Processing")
    print("=" * 60)
    
    processor = BatchProcessor(batch_size=2)
    
    # Simulate a stream of documents
    def document_stream():
        for doc in SAMPLE_DOCUMENTS:
            yield doc
            # Simulate reading from a large file
            time.sleep(0.01)
    
    print("\nProcessing documents as stream...")
    count = 0
    for result in processor.process_stream(document_stream()):
        count += 1
        if count <= 3:
            print(f"  [{count}] Processed: {result['text'][:40]}...")
        elif count == 4:
            print(f"  ... (processing remaining {len(SAMPLE_DOCUMENTS) - 3} documents)")
    
    print(f"\nTotal streamed: {count} documents")
    print("Memory usage: Constant (only batch_size in memory)")


def example_parallel():
    """Demonstrate parallel processing."""
    print("\n" + "=" * 60)
    print("Parallel Processing")
    print("=" * 60)
    
    # Compare sequential vs parallel
    documents = SAMPLE_DOCUMENTS * 5  # 50 documents
    
    # Sequential
    processor_seq = BatchProcessor()
    start = time.time()
    results_seq = processor_seq.process_batch(documents)
    time_seq = time.time() - start
    
    # Parallel
    processor_par = BatchProcessor(max_workers=4)
    start = time.time()
    results_par = processor_par.process_parallel(documents)
    time_par = time.time() - start
    
    print(f"\nDocuments: {len(documents)}")
    print(f"Sequential: {time_seq:.3f}s ({len(documents)/time_seq:.1f} docs/s)")
    print(f"Parallel (4 workers): {time_par:.3f}s ({len(documents)/time_par:.1f} docs/s)")
    print(f"Speedup: {time_seq/time_par:.2f}x")


def example_error_handling():
    """Demonstrate error handling in batch processing."""
    print("\n" + "=" * 60)
    print("Error Handling")
    print("=" * 60)
    
    # Mix of valid and problematic inputs
    mixed_inputs = [
        "Valid text for encoding",
        "",  # Empty string
        "Another valid text",
        None,  # This would cause an error
        "Final valid text",
    ]
    
    # Filter out None values for safe processing
    safe_inputs = [t for t in mixed_inputs if t is not None]
    
    processor = BatchProcessor()
    results = processor.process_batch(safe_inputs)
    
    print(f"\nProcessed {len(results)} texts")
    
    for i, result in enumerate(results):
        status = "✓" if result.get('success') else "✗"
        text = result.get('text', '')[:30] or "(empty)"
        print(f"  {status} [{i+1}] {text}")
    
    stats = processor.get_stats()
    print(f"\nError rate: {stats['error_rate']:.1%}")


def example_file_processing():
    """Demonstrate file-based batch processing."""
    print("\n" + "=" * 60)
    print("File Processing")
    print("=" * 60)
    
    # Create a temporary file for demonstration
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for doc in SAMPLE_DOCUMENTS:
            f.write(doc + "\n")
        input_path = Path(f.name)
    
    output_path = input_path.with_suffix('.json')
    
    try:
        processor = BatchProcessor(batch_size=4)
        stats = processor.process_file(input_path, output_path)
        
        print(f"\nInput: {stats['input_file']}")
        print(f"Output: {stats['output_file']}")
        print(f"Processed: {stats['total_processed']} lines")
        print(f"Errors: {stats['total_errors']}")
        print(f"Time: {stats['processing_time']:.3f}s")
        print(f"Throughput: {stats['throughput']:.1f} lines/s")
        
        # Show output sample
        with open(output_path) as f:
            data = json.load(f)
            print(f"\nOutput contains {len(data)} encoded documents")
    
    finally:
        # Cleanup
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def example_custom_pipeline():
    """Demonstrate custom processing pipeline."""
    print("\n" + "=" * 60)
    print("Custom Processing Pipeline")
    print("=" * 60)
    
    class CustomPipeline:
        """Custom pipeline with preprocessing and postprocessing."""
        
        def __init__(self):
            self.processor = BatchProcessor()
        
        def preprocess(self, text: str) -> str:
            """Clean and normalize text."""
            # Remove extra whitespace
            text = ' '.join(text.split())
            # Convert to lowercase
            text = text.lower()
            return text
        
        def postprocess(self, result: dict) -> dict:
            """Add metadata to results."""
            result['word_count'] = len(result.get('text', '').split())
            result['glyph_count'] = len(result.get('glyphs', []))
            result['compression_ratio'] = (
                result['glyph_count'] / max(1, result['word_count'])
            )
            return result
        
        def process(self, texts: list[str]) -> list[dict]:
            """Full pipeline: preprocess -> encode -> postprocess."""
            # Preprocess
            cleaned = [self.preprocess(t) for t in texts]
            
            # Encode
            results = self.processor.process_batch(cleaned)
            
            # Postprocess
            results = [self.postprocess(r) for r in results]
            
            return results
    
    pipeline = CustomPipeline()
    results = pipeline.process(SAMPLE_DOCUMENTS[:3])
    
    print("\nPipeline results:")
    for result in results:
        print(f"  • Words: {result['word_count']}, "
              f"Glyphs: {result['glyph_count']}, "
              f"Ratio: {result['compression_ratio']:.2f}")


def main():
    """Run all batch processing examples."""
    print("Sigmalang Batch Processing Examples")
    print("=" * 60)
    
    example_basic_batch()
    example_streaming()
    example_parallel()
    example_error_handling()
    example_file_processing()
    example_custom_pipeline()
    
    print("\n" + "=" * 60)
    print("Batch Processing Best Practices")
    print("=" * 60)
    print("""
1. Choose appropriate batch size:
   - Smaller batches (8-32): Lower latency, more overhead
   - Larger batches (64-256): Higher throughput, more memory

2. Use streaming for large datasets:
   - Constant memory usage
   - Process files larger than RAM

3. Parallel processing:
   - Use for CPU-bound encoding
   - workers = CPU cores for best performance

4. Error handling:
   - Always check result['success']
   - Log errors for debugging
   - Use continue-on-error for batch jobs

5. Monitor throughput:
   - Track processing statistics
   - Adjust batch size based on performance
""")


if __name__ == "__main__":
    main()
