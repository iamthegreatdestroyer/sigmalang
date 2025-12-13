#!/usr/bin/env python3
"""
Example 05: CLI Usage
=====================

This example demonstrates how to use the sigmalang CLI from Python
by showing the equivalent subprocess calls.

The CLI provides a convenient interface for:
- Text encoding and decoding
- Analogy solving
- Semantic search
- Running the REST API server

Note: This file shows both CLI command examples and their Python equivalents.
"""

import subprocess
import sys
import json
from typing import Optional


def run_cli_command(args: list[str], capture_output: bool = True) -> tuple[int, str, str]:
    """
    Run a sigmalang CLI command.
    
    Args:
        args: Command arguments (without 'sigmalang' prefix)
        capture_output: Whether to capture stdout/stderr
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    cmd = [sys.executable, "-m", "sigmalang"] + args
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", "Python not found"


def example_encode():
    """Demonstrate the encode command."""
    print("=" * 60)
    print("CLI Encode Command")
    print("=" * 60)
    
    # CLI equivalent: sigmalang encode "Hello, World!"
    print("\n$ sigmalang encode \"Hello, World!\"")
    
    code, stdout, stderr = run_cli_command(["encode", "Hello, World!"])
    
    if code == 0:
        print(f"Output: {stdout.strip()}")
    else:
        print(f"Error: {stderr}")
        # Show what it would output
        print("\nExpected output format:")
        print('{"text": "Hello, World!", "glyphs": ["Σ42", "Σ15", ...], "tokens": [...]}')
    
    # With output format option
    print("\n$ sigmalang encode \"AI is transforming the world\" --format json")
    code, stdout, stderr = run_cli_command(["encode", "AI is transforming the world", "--format", "json"])
    
    if code == 0:
        try:
            data = json.loads(stdout)
            print(f"Encoded {len(data.get('glyphs', []))} glyphs")
        except json.JSONDecodeError:
            print(f"Output: {stdout.strip()}")


def example_decode():
    """Demonstrate the decode command."""
    print("\n" + "=" * 60)
    print("CLI Decode Command")
    print("=" * 60)
    
    # CLI equivalent: sigmalang decode "Σ42 Σ15 Σ78"
    print("\n$ sigmalang decode \"Σ42 Σ15 Σ78\"")
    
    code, stdout, stderr = run_cli_command(["decode", "Σ42 Σ15 Σ78"])
    
    if code == 0:
        print(f"Output: {stdout.strip()}")
    else:
        print(f"Note: {stderr if stderr else 'Decode command demonstrated'}")
        print("\nExpected: Reconstructed text from glyphs")


def example_analogy():
    """Demonstrate the analogy command."""
    print("\n" + "=" * 60)
    print("CLI Analogy Command")
    print("=" * 60)
    
    # CLI equivalent: sigmalang analogy "king" "queen" "man"
    print("\n$ sigmalang analogy king queen man")
    print("# Solves: king:queen :: man:?")
    
    code, stdout, stderr = run_cli_command(["analogy", "king", "queen", "man"])
    
    if code == 0:
        print(f"Answer: {stdout.strip()}")
    else:
        print("Expected answer: woman")
        print("\nAnalogy types supported:")
        print("  - Gender: king:queen :: man:woman")
        print("  - Nationality: Paris:France :: Tokyo:Japan")
        print("  - Comparative: big:bigger :: small:smaller")
        print("  - Profession: doctor:hospital :: teacher:school")


def example_search():
    """Demonstrate the search command."""
    print("\n" + "=" * 60)
    print("CLI Search Command")
    print("=" * 60)
    
    # Note: Search requires an indexed corpus
    print("\n$ sigmalang search \"machine learning\" --corpus ./documents/")
    print("# Searches for semantically similar content")
    
    code, stdout, stderr = run_cli_command(["search", "machine learning", "--limit", "5"])
    
    if code == 0:
        print(f"Results:\n{stdout.strip()}")
    else:
        print("\nSearch command options:")
        print("  --corpus PATH    Directory or file to search")
        print("  --limit N        Maximum results (default: 10)")
        print("  --mode MODE      Search mode: exact|semantic|hybrid|fuzzy")
        print("  --threshold F    Similarity threshold (0.0-1.0)")


def example_serve():
    """Demonstrate the serve command."""
    print("\n" + "=" * 60)
    print("CLI Serve Command")
    print("=" * 60)
    
    print("\n$ sigmalang serve --host 0.0.0.0 --port 8000")
    print("# Starts the REST API server")
    
    print("\nServer options:")
    print("  --host HOST      Bind address (default: 127.0.0.1)")
    print("  --port PORT      Port number (default: 8000)")
    print("  --workers N      Number of worker processes")
    print("  --reload         Enable auto-reload for development")
    print("  --log-level LVL  Logging level: debug|info|warning|error")
    
    print("\nOnce running, access the API at:")
    print("  http://localhost:8000/")
    print("  http://localhost:8000/docs (Swagger UI)")
    print("  http://localhost:8000/health")


def example_batch_file():
    """Demonstrate processing a file."""
    print("\n" + "=" * 60)
    print("CLI Batch Processing")
    print("=" * 60)
    
    print("\n# Process a text file")
    print("$ sigmalang encode --input document.txt --output encoded.json")
    
    print("\n# Process multiple files")
    print("$ sigmalang encode --input ./texts/ --output ./encoded/ --batch")
    
    print("\n# Pipe from stdin")
    print("$ cat document.txt | sigmalang encode --stdin")
    
    print("\n# Output formats")
    print("$ sigmalang encode \"text\" --format json    # JSON output")
    print("$ sigmalang encode \"text\" --format csv     # CSV output")
    print("$ sigmalang encode \"text\" --format binary  # Binary format")


def example_config():
    """Demonstrate configuration options."""
    print("\n" + "=" * 60)
    print("CLI Configuration")
    print("=" * 60)
    
    print("\n# Show current configuration")
    print("$ sigmalang config show")
    
    print("\n# Set configuration value")
    print("$ sigmalang config set encoder.model transformer")
    
    print("\n# Use environment variables")
    print("$ export SIGMALANG_LOG_LEVEL=debug")
    print("$ export SIGMALANG_ENCODER_MODEL=transformer")
    print("$ sigmalang encode \"text\"")
    
    print("\n# Use configuration file")
    print("$ sigmalang --config ./sigmalang.yaml encode \"text\"")
    
    print("\nConfiguration file example (sigmalang.yaml):")
    print("""
sigmalang:
  encoder:
    model: transformer
    batch_size: 32
  search:
    mode: hybrid
    threshold: 0.7
  api:
    host: 0.0.0.0
    port: 8000
    workers: 4
""")


def main():
    """Run all CLI examples."""
    print("Sigmalang CLI Usage Examples")
    print("=" * 60)
    print("\nThis script demonstrates CLI commands and their usage.")
    print("Install sigmalang to use these commands:\n")
    print("  pip install sigmalang")
    print("  # or")
    print("  pip install -e .")
    
    example_encode()
    example_decode()
    example_analogy()
    example_search()
    example_serve()
    example_batch_file()
    example_config()
    
    print("\n" + "=" * 60)
    print("CLI Help")
    print("=" * 60)
    print("\nFor more information, run:")
    print("  sigmalang --help")
    print("  sigmalang encode --help")
    print("  sigmalang analogy --help")
    print("  sigmalang search --help")
    print("  sigmalang serve --help")


if __name__ == "__main__":
    main()
