"""
ΣLANG WebSocket Real-Time Streaming Server

Provides WebSocket endpoints for real-time streaming encode/decode
and bidirectional compression over persistent connections.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI is optional; we guard the import so the module still imports even
# when fastapi / uvicorn are not installed (e.g. in minimal test envs).
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False
    FastAPI = None  # type: ignore
    WebSocket = None  # type: ignore
    WebSocketDisconnect = None  # type: ignore

try:
    from .encoder import SigmaEncoder
    _HAS_ENCODER = True
except ImportError:
    _HAS_ENCODER = False

try:
    from .parser import SemanticParser
    _HAS_PARSER = True
except ImportError:
    _HAS_PARSER = False

try:
    from .bidirectional_codec import BidirectionalSemanticCodec
    _HAS_CODEC = True
except ImportError:
    _HAS_CODEC = False


# =============================================================================
# Connection Manager
# =============================================================================

class ConnectionManager:
    """Track active WebSocket connections."""

    def __init__(self) -> None:
        self.active: list = []

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: Any) -> None:
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        for connection in list(self.active):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()


# =============================================================================
# Streaming helpers
# =============================================================================

def _make_parser():
    """Return a parser instance (or a lightweight stub)."""
    if _HAS_PARSER:
        return SemanticParser()
    class _StubParser:
        def parse(self, text: str):
            return text  # stub returns the string itself
    return _StubParser()


def _make_encoder():
    """Return an encoder instance (or a lightweight stub if unavailable)."""
    if _HAS_ENCODER:
        return SigmaEncoder()
    # Fallback stub used in testing / minimal environments
    class _StubEncoder:
        def encode(self, tree, original_text: str = "") -> str:
            text = original_text or str(tree)
            return f"[encoded:{len(text)}chars]"
    return _StubEncoder()


def _make_codec():
    """Return a codec instance (or a lightweight stub if unavailable)."""
    if _HAS_CODEC:
        return BidirectionalSemanticCodec()
    class _StubCodec:
        def compress(self, text: str, **kwargs) -> bytes:
            return text.encode('utf-8')  # passthrough stub
    return _StubCodec()


# =============================================================================
# FastAPI App
# =============================================================================

def create_websocket_app() -> Optional[Any]:
    """
    Build and return the FastAPI application exposing WebSocket endpoints.

    Returns ``None`` when FastAPI is not installed.
    """
    if not _HAS_FASTAPI:
        logger.warning("FastAPI not installed; WebSocket server unavailable.")
        return None

    app = FastAPI(
        title="ΣLANG WebSocket Server",
        description="Real-time streaming encode/decode and bidirectional compression",
        version="1.0.0",
    )

    # ------------------------------------------------------------------
    # REST health endpoint (handy for liveness probes)
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "transport": "websocket"})

    # ------------------------------------------------------------------
    # /ws/encode  — stream a text sentence-by-sentence
    # ------------------------------------------------------------------

    @app.websocket("/ws/encode")
    async def websocket_encode(websocket: WebSocket):
        """
        Real-time streaming encoder via WebSocket.

        Protocol (JSON):
          Client → Server: {"text": "<input text>"}
          Server → Client: {"chunk": "<chunk>", "encoded": "<result>", "done": false}
          Server → Client: {"done": true}
        """
        await manager.connect(websocket)
        encoder = _make_encoder()
        parser = _make_parser()

        try:
            async for message in websocket.iter_text():
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "error": "Invalid JSON", "done": True
                    })
                    continue

                text: str = data.get("text", "")
                if not text:
                    await websocket.send_json({"error": "Empty text", "done": True})
                    continue

                # Stream encoding results chunk by chunk (split on ". ")
                chunks = [c.strip() for c in text.split(". ") if c.strip()]
                if not chunks:
                    chunks = [text]

                for chunk in chunks:
                    try:
                        tree = parser.parse(chunk)
                        result = encoder.encode(tree, original_text=chunk)
                    except Exception:
                        result = f"[encoded:{len(chunk)}chars]"
                    await websocket.send_json({
                        "chunk": chunk,
                        "encoded": str(result),
                        "done": False,
                    })
                    await asyncio.sleep(0)  # yield control

                await websocket.send_json({"done": True})

        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as exc:
            logger.error("websocket_encode error: %s", exc, exc_info=True)
            try:
                await websocket.send_json({"error": str(exc), "done": True})
            except Exception:
                pass
            manager.disconnect(websocket)

    # ------------------------------------------------------------------
    # /ws/stream — bidirectional binary compression
    # ------------------------------------------------------------------

    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        """
        Bidirectional streaming compression.

        Protocol (binary):
          Client → Server: raw bytes
          Server → Client: compressed bytes
        """
        await manager.connect(websocket)
        codec = _make_codec()

        try:
            async for message in websocket.iter_bytes():
                try:
                    text = message.decode('utf-8', errors='replace')
                    compressed = codec.compress(text)
                    if isinstance(compressed, str):
                        compressed = compressed.encode('utf-8')
                except Exception:
                    compressed = message  # passthrough on error
                await websocket.send_bytes(compressed)
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as exc:
            logger.error("websocket_stream error: %s", exc, exc_info=True)
            manager.disconnect(websocket)

    # ------------------------------------------------------------------
    # /ws/analyze — on-the-fly semantic similarity streaming
    # ------------------------------------------------------------------

    @app.websocket("/ws/analyze")
    async def websocket_analyze(websocket: WebSocket):
        """
        Streaming semantic analysis.

        Protocol (JSON):
          Client → Server: {"texts": ["text1", "text2", ...]}
          Server → Client: {"index": 0, "text": "...", "tokens": N, "done": false}
          Server → Client: {"done": true, "total": N, "elapsed_ms": X}
        """
        await manager.connect(websocket)
        encoder = _make_encoder()
        parser = _make_parser()

        try:
            async for message in websocket.iter_text():
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON", "done": True})
                    continue

                texts = data.get("texts", [])
                if not texts:
                    await websocket.send_json({"error": "No texts provided", "done": True})
                    continue

                t0 = time.perf_counter()
                for idx, text in enumerate(texts):
                    text_str = str(text)
                    try:
                        tree = parser.parse(text_str)
                        result = encoder.encode(tree, original_text=text_str)
                    except Exception:
                        result = f"[encoded:{len(text_str)}chars]"
                    token_count = len(text_str.split())
                    await websocket.send_json({
                        "index": idx,
                        "text": str(text)[:100],
                        "encoded": str(result),
                        "tokens": token_count,
                        "done": False,
                    })
                    await asyncio.sleep(0)

                elapsed_ms = (time.perf_counter() - t0) * 1000
                await websocket.send_json({
                    "done": True,
                    "total": len(texts),
                    "elapsed_ms": round(elapsed_ms, 2),
                })

        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as exc:
            logger.error("websocket_analyze error: %s", exc, exc_info=True)
            try:
                await websocket.send_json({"error": str(exc), "done": True})
            except Exception:
                pass
            manager.disconnect(websocket)

    return app


# Module-level app (created lazily so import doesn't fail in minimal envs)
_ws_app = None


def get_websocket_app() -> Optional[Any]:
    """Get (or lazily create) the singleton WebSocket FastAPI app."""
    global _ws_app
    if _ws_app is None:
        _ws_app = create_websocket_app()
    return _ws_app


# Convenience alias used in tests / launch scripts
app = get_websocket_app()


# =============================================================================
# Entry-point
# =============================================================================

def run(host: str = "0.0.0.0", port: int = 26090) -> None:  # nosec B104  # pragma: no cover
    """Launch the WebSocket server with uvicorn."""
    try:
        import uvicorn
    except ImportError:
        raise RuntimeError("uvicorn is required: pip install uvicorn")

    ws_app = get_websocket_app()
    if ws_app is None:
        raise RuntimeError("FastAPI is required: pip install fastapi")

    logger.info("Starting ΣLANG WebSocket server on %s:%d", host, port)
    uvicorn.run(ws_app, host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    run()
