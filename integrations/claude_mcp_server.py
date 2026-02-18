"""
Claude Desktop MCP Server - Phase 4 Task 4.1 + Phase 7 Track 11

Model Context Protocol (MCP) server exposing SigmaLang encoding/decoding
capabilities directly in Claude Desktop conversations.

MCP Tools Provided:
- sigma_encode: Encode text to SigmaLang compressed representation
- sigma_decode: Decode SigmaLang back to text
- sigma_compress_file: Compress a file using SigmaLang
- sigma_analyze: Analyze text compression potential
- sigma_search: Semantic search over compressed knowledge base
- sigma_stats: Get current compression statistics
- sigma_batch: Batch encode/decode multiple items in one call
- sigma_compose: Chain multiple tools in a pipeline
- sigma_stream_encode: Streaming encode for large texts (chunked)
- sigma_health: System health and diagnostics

MCP Resources Provided (Phase 7):
- sigmalang://codebook/stats: Live codebook statistics
- sigmalang://compression/history: Recent compression history

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
SERVER_VERSION = "2.0.0"

# Batch and streaming limits
MAX_BATCH_SIZE = 50
STREAM_CHUNK_SIZE = 4096  # characters per streaming chunk


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

    # Phase 7 Track 11: Batch operations
    def batch_encode(self, items: List[str]) -> List[Dict[str, Any]]:
        """Encode multiple texts in one call."""
        return [self.encode(text) for text in items[:MAX_BATCH_SIZE]]

    def batch_decode(self, items: List[str]) -> List[Dict[str, Any]]:
        """Decode multiple hex-encoded items in one call."""
        return [self.decode(hex_str) for hex_str in items[:MAX_BATCH_SIZE]]

    def stream_encode(self, text: str, chunk_size: int = STREAM_CHUNK_SIZE) -> List[Dict[str, Any]]:
        """
        Streaming encode: split large text into chunks and encode each.

        Returns list of per-chunk results with an overall summary.
        """
        if len(text) <= chunk_size:
            return [self.encode(text)]

        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            result = self.encode(chunk)
            result['chunk_index'] = len(chunks)
            result['chunk_offset'] = i
            chunks.append(result)

        return chunks

    def compose_pipeline(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a pipeline of tool steps sequentially.

        Each step: {"tool": "sigma_encode", "args": {"text": "..."}}
        Later steps can reference $prev to use previous output.
        """
        results = []
        prev_result = None

        for i, step in enumerate(steps):
            tool = step.get('tool', '')
            args = step.get('args', {})

            # Substitute $prev references
            if prev_result and isinstance(args, dict):
                for k, v in args.items():
                    if v == '$prev' and prev_result:
                        if tool == 'sigma_decode' and k == 'encoded_hex':
                            args[k] = prev_result.get('encoded', '')
                        elif tool == 'sigma_encode' and k == 'text':
                            args[k] = prev_result.get('decoded_text', '')

            try:
                if tool == 'sigma_encode':
                    result = self.encode(args.get('text', ''))
                elif tool == 'sigma_decode':
                    result = self.decode(args.get('encoded_hex', ''))
                elif tool == 'sigma_analyze':
                    result = self.analyze(args.get('text', ''))
                elif tool == 'sigma_stats':
                    result = self.get_stats()
                else:
                    result = {'error': f'Unknown tool in pipeline: {tool}'}
            except Exception as e:
                result = {'error': str(e), 'step': i}

            result['_step'] = i
            results.append(result)
            prev_result = result

        return {
            'pipeline_steps': len(steps),
            'results': results,
            'final_result': results[-1] if results else None,
        }

    def get_health(self) -> Dict[str, Any]:
        """Get system health and diagnostics."""
        import platform

        health = {
            'status': 'healthy',
            'server_version': SERVER_VERSION,
            'protocol_version': MCP_VERSION,
            'python_version': platform.python_version(),
            'initialized': self._initialized,
            'stats': self.get_stats(),
        }

        # Check component availability
        components = {}
        try:
            from sigmalang.core.parser import SemanticParser
            components['parser'] = True
        except ImportError:
            components['parser'] = False

        try:
            from sigmalang.core.encoder import SigmaEncoder
            components['encoder'] = True
        except ImportError:
            components['encoder'] = False

        try:
            from sigmalang.core.entropy_estimator import EntropyAnalyzer
            components['entropy_analyzer'] = True
        except ImportError:
            components['entropy_analyzer'] = False

        try:
            from sigmalang.core.vector_compressor import VectorCompressor
            components['vector_compressor'] = True
        except ImportError:
            components['vector_compressor'] = False

        try:
            from sigmalang.core.multimodal_vq import MultiModalVQ
            components['multimodal_vq'] = True
        except ImportError:
            components['multimodal_vq'] = False

        health['components'] = components
        health['components_available'] = sum(1 for v in components.values() if v)
        health['components_total'] = len(components)

        return health

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
            },
            # Phase 7 Track 11: New tools
            'sigma_batch': {
                'description': 'Batch encode or decode multiple items in one call. '
                              'Accepts up to 50 items.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {
                        'operation': {
                            'type': 'string',
                            'enum': ['encode', 'decode'],
                            'description': 'Operation: encode or decode'
                        },
                        'items': {
                            'type': 'array',
                            'items': {'type': 'string'},
                            'description': 'List of texts (encode) or hex strings (decode)'
                        }
                    },
                    'required': ['operation', 'items']
                }
            },
            'sigma_compose': {
                'description': 'Chain multiple SigmaLang tools in a pipeline. '
                              'Use $prev in args to reference the previous step output.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {
                        'steps': {
                            'type': 'array',
                            'items': {
                                'type': 'object',
                                'properties': {
                                    'tool': {'type': 'string'},
                                    'args': {'type': 'object'}
                                }
                            },
                            'description': 'Pipeline steps: [{tool, args}, ...]'
                        }
                    },
                    'required': ['steps']
                }
            },
            'sigma_stream_encode': {
                'description': 'Streaming encode for large texts. Splits into chunks '
                              'and encodes each separately.',
                'inputSchema': {
                    'type': 'object',
                    'properties': {
                        'text': {
                            'type': 'string',
                            'description': 'Large text to encode in chunks'
                        },
                        'chunk_size': {
                            'type': 'integer',
                            'description': 'Characters per chunk (default 4096)'
                        }
                    },
                    'required': ['text']
                }
            },
            'sigma_health': {
                'description': 'Get system health, component availability, and diagnostics.',
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
            elif method == 'resources/list':
                return self._handle_resources_list(msg_id)
            elif method == 'resources/read':
                return self._handle_resource_read(msg_id, params)
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
                'tools': {},
                'resources': {},
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

    def _execute_tool(self, tool_name: str, arguments: Dict) -> Any:
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
        # Phase 7 Track 11: New tools
        elif tool_name == 'sigma_batch':
            op = arguments.get('operation', 'encode')
            items = arguments.get('items', [])
            if op == 'encode':
                return {'results': self.engine.batch_encode(items), 'count': len(items)}
            else:
                return {'results': self.engine.batch_decode(items), 'count': len(items)}
        elif tool_name == 'sigma_compose':
            return self.engine.compose_pipeline(arguments.get('steps', []))
        elif tool_name == 'sigma_stream_encode':
            chunk_size = arguments.get('chunk_size', STREAM_CHUNK_SIZE)
            chunks = self.engine.stream_encode(arguments['text'], chunk_size)
            return {'chunks': chunks, 'total_chunks': len(chunks)}
        elif tool_name == 'sigma_health':
            return self.engine.get_health()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    # Phase 7 Track 11: Resource provider
    def _handle_resources_list(self, msg_id: Any) -> Dict:
        """Handle resources/list — expose live data as MCP resources."""
        resources = [
            {
                'uri': 'sigmalang://codebook/stats',
                'name': 'Codebook Statistics',
                'description': 'Live statistics about the SigmaLang codebook',
                'mimeType': 'application/json',
            },
            {
                'uri': 'sigmalang://compression/history',
                'name': 'Compression History',
                'description': 'Recent compression operation results',
                'mimeType': 'application/json',
            },
            {
                'uri': 'sigmalang://system/health',
                'name': 'System Health',
                'description': 'Server health and component diagnostics',
                'mimeType': 'application/json',
            },
        ]
        return self._make_response(msg_id, {'resources': resources})

    def _handle_resource_read(self, msg_id: Any, params: Dict) -> Dict:
        """Handle resources/read — return resource content."""
        uri = params.get('uri', '')

        try:
            if uri == 'sigmalang://codebook/stats':
                content = json.dumps(self.engine.get_stats(), indent=2)
            elif uri == 'sigmalang://compression/history':
                content = json.dumps({
                    'stats': self.engine.get_stats(),
                    'note': 'Per-operation history available via sigma_stats tool',
                }, indent=2)
            elif uri == 'sigmalang://system/health':
                content = json.dumps(self.engine.get_health(), indent=2)
            else:
                return self._make_error(msg_id, -32602, f"Unknown resource: {uri}")

            return self._make_response(msg_id, {
                'contents': [{
                    'uri': uri,
                    'mimeType': 'application/json',
                    'text': content,
                }]
            })
        except Exception as e:
            return self._make_error(msg_id, -32603, str(e))

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
