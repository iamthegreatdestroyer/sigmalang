"""
Claude Desktop MCP Server - Phase 4 Task 4.1

Model Context Protocol (MCP) server exposing SigmaLang encoding/decoding
capabilities directly in Claude Desktop conversations.

MCP Tools Provided:
- sigma_encode: Encode text to SigmaLang compressed representation
- sigma_decode: Decode SigmaLang back to text
- sigma_compress_file: Compress a file using SigmaLang
- sigma_analyze: Analyze text compression potential
- sigma_search: Semantic search over compressed knowledge base
- sigma_stats: Get current compression statistics

Setup:
    Add to Claude Desktop config (claude_desktop_config.json):
    {
        "mcpServers": {
            "sigmalang": {
                "command": "python",
                "args": ["-m", "integrations.claude_mcp_server"],
                "cwd": "<path-to-sigmalang>"
            }
        }
    }

Protocol: MCP over stdio (JSON-RPC 2.0)
"""

import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

# Add parent to path
sigmalang_root = Path(__file__).parent.parent
sys.path.insert(0, str(sigmalang_root))

logger = logging.getLogger(__name__)


# =============================================================================
# MCP Protocol Constants
# =============================================================================

MCP_VERSION = "2024-11-05"
SERVER_NAME = "sigmalang"
SERVER_VERSION = "1.0.0"


# =============================================================================
# SigmaLang Engine Wrapper
# =============================================================================

class SigmaLangEngine:
    """Wrapper around SigmaLang core for MCP tool use."""

    def __init__(self):
        self._initialized = False
        self._parser = None
        self._encoder = None
        self._decoder = None
        self._stats = {
            'encode_calls': 0,
            'decode_calls': 0,
            'total_bytes_in': 0,
            'total_bytes_out': 0
        }

    def _ensure_initialized(self) -> None:
        """Lazy initialization of SigmaLang components."""
        if self._initialized:
            return

        try:
            from sigmalang.core.parser import SemanticParser
            from sigmalang.core.encoder import SigmaEncoder, SigmaDecoder

            self._parser = SemanticParser()
            self._encoder = SigmaEncoder()
            self._decoder = SigmaDecoder(self._encoder)
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize SigmaLang: {e}")
            raise

    def encode(self, text: str) -> Dict[str, Any]:
        """Encode text to SigmaLang representation."""
        self._ensure_initialized()

        original_size = len(text.encode('utf-8'))

        # Parse and encode
        tree = self._parser.parse(text)
        encoded = self._encoder.encode(tree)

        compressed_size = len(encoded)
        compression_ratio = original_size / max(1, compressed_size)

        # Update stats
        self._stats['encode_calls'] += 1
        self._stats['total_bytes_in'] += original_size
        self._stats['total_bytes_out'] += compressed_size

        # Create hex representation for display
        encoded_hex = encoded.hex()

        return {
            'encoded': encoded_hex,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': round(compression_ratio, 2),
            'node_count': tree.node_count if hasattr(tree, 'node_count') else 0,
            'hash': hashlib.sha256(encoded).hexdigest()[:16]
        }

    def decode(self, encoded_hex: str) -> Dict[str, Any]:
        """Decode SigmaLang back to text."""
        self._ensure_initialized()

        encoded = bytes.fromhex(encoded_hex)
        decoded_tree = self._decoder.decode(encoded)

        self._stats['decode_calls'] += 1

        # Extract text from decoded tree
        text = str(decoded_tree)

        return {
            'decoded_text': text,
            'compressed_size': len(encoded),
            'decompressed_size': len(text.encode('utf-8'))
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text compression potential."""
        self._ensure_initialized()

        original_size = len(text.encode('utf-8'))

        # Parse to get tree structure
        tree = self._parser.parse(text)

        # Encode to get compression metrics
        encoded = self._encoder.encode(tree)
        compressed_size = len(encoded)

        # Count words and unique words
        words = text.split()
        unique_words = set(w.lower() for w in words)

        # Estimate primitive reuse
        primitive_reuse_estimate = 1.0 - (len(unique_words) / max(1, len(words)))

        return {
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'compression_ratio': round(original_size / max(1, compressed_size), 2),
            'word_count': len(words),
            'unique_words': len(unique_words),
            'vocabulary_density': round(len(unique_words) / max(1, len(words)), 3),
            'estimated_primitive_reuse': round(primitive_reuse_estimate * 100, 1),
            'node_count': tree.node_count if hasattr(tree, 'node_count') else 0,
            'recommendation': self._compression_recommendation(
                original_size, compressed_size, primitive_reuse_estimate
            )
        }

    def compress_file(self, file_path: str) -> Dict[str, Any]:
        """Compress a file using SigmaLang."""
        self._ensure_initialized()

        path = Path(file_path)

        if not path.exists():
            return {'error': f'File not found: {file_path}'}

        if not path.is_file():
            return {'error': f'Not a file: {file_path}'}

        # Read file
        try:
            content = path.read_text(encoding='utf-8')
        except Exception as e:
            return {'error': f'Failed to read file: {e}'}

        # Encode
        result = self.encode(content)

        # Save compressed file
        compressed_path = path.with_suffix('.sigma')
        try:
            compressed_path.write_bytes(bytes.fromhex(result['encoded']))
        except Exception as e:
            return {'error': f'Failed to write compressed file: {e}'}

        return {
            'input_file': str(path),
            'output_file': str(compressed_path),
            'original_size': result['original_size'],
            'compressed_size': result['compressed_size'],
            'compression_ratio': result['compression_ratio'],
            'space_saved_bytes': result['original_size'] - result['compressed_size'],
            'space_saved_pct': round(
                (1 - result['compressed_size'] / max(1, result['original_size'])) * 100, 1
            )
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get current compression statistics."""
        total_in = self._stats['total_bytes_in']
        total_out = self._stats['total_bytes_out']

        return {
            'encode_calls': self._stats['encode_calls'],
            'decode_calls': self._stats['decode_calls'],
            'total_bytes_in': total_in,
            'total_bytes_out': total_out,
            'overall_compression_ratio': round(total_in / max(1, total_out), 2),
            'total_bytes_saved': total_in - total_out,
            'initialized': self._initialized
        }

    def _compression_recommendation(
        self, original: int, compressed: int, reuse: float
    ) -> str:
        """Generate compression recommendation."""
        ratio = original / max(1, compressed)

        if ratio > 20:
            return "Excellent compression potential - highly repetitive content"
        elif ratio > 10:
            return "Good compression potential - moderate repetition detected"
        elif ratio > 5:
            return "Fair compression - content has some unique patterns"
        else:
            return "Low compression potential - highly unique content"


# =============================================================================
# MCP Server Implementation
# =============================================================================

class MCPServer:
    """
    MCP Server implementing the Model Context Protocol.

    Communicates over stdio using JSON-RPC 2.0 messages.
    """

    def __init__(self):
        self.engine = SigmaLangEngine()
        self.tools = self._register_tools()

    def _register_tools(self) -> Dict[str, Dict[str, Any]]:
        """Register available MCP tools."""
        return {
            'sigma_encode': {
                'description': 'Encode text to SigmaLang compressed representation. '
                              'Returns hex-encoded bytes and compression metrics.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': 'Text to encode'
                        }
                    },
                    'required': ['text']
                }
            },
            'sigma_decode': {
                'description': 'Decode SigmaLang hex-encoded representation back to text.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {
                        'encoded_hex': {
                            'type': 'string',
                            'description': 'Hex-encoded SigmaLang data'
                        }
                    },
                    'required': ['encoded_hex']
                }
            },
            'sigma_analyze': {
                'description': 'Analyze text to estimate compression potential, '
                              'vocabulary density, and primitive reuse rate.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': 'Text to analyze'
                        }
                    },
                    'required': ['text']
                }
            },
            'sigma_compress_file': {
                'description': 'Compress a file using SigmaLang. Creates a .sigma file '
                              'alongside the original.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {
                        'file_path': {
                            'type': 'string',
                            'description': 'Path to file to compress'
                        }
                    },
                    'required': ['file_path']
                }
            },
            'sigma_stats': {
                'description': 'Get current SigmaLang compression statistics for this session.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {}
                }
            }
        }

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming JSON-RPC message."""
        method = message.get('method', '')
        msg_id = message.get('id')
        params = message.get('params', {})

        try:
            if method == 'initialize':
                return self._handle_initialize(msg_id, params)
            elif method == 'tools/list':
                return self._handle_tools_list(msg_id)
            elif method == 'tools/call':
                return self._handle_tool_call(msg_id, params)
            elif method == 'notifications/initialized':
                return None  # No response for notifications
            elif method == 'ping':
                return self._make_response(msg_id, {})
            else:
                return self._make_error(msg_id, -32601, f"Method not found: {method}")
        except Exception as e:
            return self._make_error(msg_id, -32603, str(e))

    def _handle_initialize(self, msg_id: Any, params: Dict) -> Dict:
        """Handle initialize request."""
        return self._make_response(msg_id, {
            'protocolVersion': MCP_VERSION,
            'capabilities': {
                'tools': {}
            },
            'serverInfo': {
                'name': SERVER_NAME,
                'version': SERVER_VERSION
            }
        })

    def _handle_tools_list(self, msg_id: Any) -> Dict:
        """Handle tools/list request."""
        tools = []
        for name, tool_def in self.tools.items():
            tools.append({
                'name': name,
                'description': tool_def['description'],
                'inputSchema': tool_def['inputSchema']
            })

        return self._make_response(msg_id, {'tools': tools})

    def _handle_tool_call(self, msg_id: Any, params: Dict) -> Dict:
        """Handle tools/call request."""
        tool_name = params.get('name', '')
        arguments = params.get('arguments', {})

        if tool_name not in self.tools:
            return self._make_error(msg_id, -32602, f"Unknown tool: {tool_name}")

        try:
            result = self._execute_tool(tool_name, arguments)
            return self._make_response(msg_id, {
                'content': [{
                    'type': 'text',
                    'text': json.dumps(result, indent=2)
                }]
            })
        except Exception as e:
            return self._make_response(msg_id, {
                'content': [{
                    'type': 'text',
                    'text': json.dumps({'error': str(e)})
                }],
                'isError': True
            })

    def _execute_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Execute a tool and return results."""
        if tool_name == 'sigma_encode':
            return self.engine.encode(arguments['text'])
        elif tool_name == 'sigma_decode':
            return self.engine.decode(arguments['encoded_hex'])
        elif tool_name == 'sigma_analyze':
            return self.engine.analyze(arguments['text'])
        elif tool_name == 'sigma_compress_file':
            return self.engine.compress_file(arguments['file_path'])
        elif tool_name == 'sigma_stats':
            return self.engine.get_stats()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _make_response(self, msg_id: Any, result: Any) -> Dict:
        """Create a JSON-RPC response."""
        return {
            'jsonrpc': '2.0',
            'id': msg_id,
            'result': result
        }

    def _make_error(self, msg_id: Any, code: int, message: str) -> Dict:
        """Create a JSON-RPC error response."""
        return {
            'jsonrpc': '2.0',
            'id': msg_id,
            'error': {
                'code': code,
                'message': message
            }
        }

    def run_stdio(self) -> None:
        """Run the MCP server over stdio."""
        logger.info("SigmaLang MCP Server starting on stdio...")

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                message = json.loads(line)
                response = self.handle_message(message)

                if response is not None:
                    sys.stdout.write(json.dumps(response) + '\n')
                    sys.stdout.flush()

            except json.JSONDecodeError as e:
                error_response = {
                    'jsonrpc': '2.0',
                    'id': None,
                    'error': {
                        'code': -32700,
                        'message': f'Parse error: {e}'
                    }
                }
                sys.stdout.write(json.dumps(error_response) + '\n')
                sys.stdout.flush()


# =============================================================================
# Configuration Generator
# =============================================================================

def generate_claude_desktop_config(sigmalang_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate Claude Desktop configuration for the MCP server.

    Returns JSON config to add to claude_desktop_config.json
    """
    if sigmalang_path is None:
        sigmalang_path = str(sigmalang_root)

    return {
        'mcpServers': {
            'sigmalang': {
                'command': 'python',
                'args': ['-m', 'integrations.claude_mcp_server'],
                'cwd': sigmalang_path
            }
        }
    }


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point for the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        stream=sys.stderr  # Log to stderr, MCP protocol on stdout
    )

    server = MCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
