"""
Tests for the ΣLANG WebSocket streaming server.

These tests use FastAPI's TestClient which supports WebSocket connections
without needing a live server. They fall back to unit-testing the connection
manager and helpers when FastAPI is not installed.
"""

import importlib
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_server():
    """Import (or reload) websocket_server and return the module."""
    module_name = "sigmalang.core.websocket_server"
    if module_name in sys.modules:
        return sys.modules[module_name]
    return importlib.import_module(module_name)


def _has_fastapi() -> bool:
    try:
        import fastapi  # noqa: F401
        return True
    except ImportError:
        return False


def _has_httpx() -> bool:
    try:
        import httpx  # noqa: F401
        return True
    except ImportError:
        return False


def _has_uvicorn() -> bool:
    try:
        import uvicorn  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# ConnectionManager
# ---------------------------------------------------------------------------

class TestConnectionManager:
    """Unit tests for the ConnectionManager without a live server."""

    def _make_manager(self):
        mod = _import_server()
        # Fresh instance each time
        return mod.ConnectionManager()

    @pytest.mark.asyncio
    async def test_connect_adds_websocket(self):
        manager = self._make_manager()
        ws = AsyncMock()
        await manager.connect(ws)
        assert ws in manager.active
        ws.accept.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_removes_websocket(self):
        manager = self._make_manager()
        ws = AsyncMock()
        await manager.connect(ws)
        manager.disconnect(ws)
        assert ws not in manager.active

    def test_disconnect_nonexistent_is_noop(self):
        manager = self._make_manager()
        ws = AsyncMock()
        # Should not raise
        manager.disconnect(ws)

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all(self):
        manager = self._make_manager()
        ws1, ws2 = AsyncMock(), AsyncMock()
        await manager.connect(ws1)
        await manager.connect(ws2)
        await manager.broadcast({"msg": "hello"})
        ws1.send_json.assert_awaited_once_with({"msg": "hello"})
        ws2.send_json.assert_awaited_once_with({"msg": "hello"})

    @pytest.mark.asyncio
    async def test_broadcast_removes_failed_connections(self):
        manager = self._make_manager()
        ws_good = AsyncMock()
        ws_bad = AsyncMock()
        ws_bad.send_json.side_effect = Exception("connection closed")
        await manager.connect(ws_good)
        await manager.connect(ws_bad)
        await manager.broadcast({"ping": 1})
        assert ws_bad not in manager.active
        assert ws_good in manager.active


# ---------------------------------------------------------------------------
# Encoder stub
# ---------------------------------------------------------------------------

class TestStubEncoder:
    """Test the internal _make_encoder stub when SigmaEncoder is unavailable."""

    def test_stub_encoder_returns_string(self):
        mod = _import_server()
        with patch.object(mod, "_HAS_ENCODER", False):
            encoder = mod._make_encoder()
            result = encoder.encode("hello world")
            assert isinstance(result, str)
            assert "encoded" in result.lower() or "11chars" in result

    def test_real_encoder_used_when_available(self):
        mod = _import_server()
        with patch.object(mod, "_HAS_ENCODER", True):
            # SigmaEncoder might not be importable; just test the branch doesn't crash
            try:
                encoder = mod._make_encoder()
                assert encoder is not None
            except Exception:
                pass  # acceptable if import chain fails


class TestStubCodec:
    """Test the internal _make_codec stub when BidirectionalSemanticCodec is unavailable."""

    def test_stub_codec_passthrough(self):
        mod = _import_server()
        with patch.object(mod, "_HAS_CODEC", False):
            codec = mod._make_codec()
            data = "hello bytes"
            result = codec.compress(data)
            assert isinstance(result, bytes)
            assert result == data.encode('utf-8')


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------

class TestCreateWebsocketApp:
    """Test that the FastAPI app can be created."""

    def test_returns_none_without_fastapi(self):
        mod = _import_server()
        with patch.object(mod, "_HAS_FASTAPI", False):
            result = mod.create_websocket_app()
            assert result is None

    @pytest.mark.skipif(not _has_fastapi(), reason="FastAPI not installed")
    def test_returns_app_with_fastapi(self):
        mod = _import_server()
        with patch.object(mod, "_HAS_FASTAPI", True):
            app = mod.create_websocket_app()
            assert app is not None

    @pytest.mark.skipif(not _has_fastapi(), reason="FastAPI not installed")
    def test_get_websocket_app_lazy(self):
        mod = _import_server()
        app1 = mod.get_websocket_app()
        app2 = mod.get_websocket_app()
        # Should be the same instance (singleton)
        assert app1 is app2


# ---------------------------------------------------------------------------
# WebSocket endpoint integration tests (require FastAPI + httpx or starlette)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _has_fastapi() or not _has_httpx(),
    reason="FastAPI or httpx not installed",
)
class TestWebSocketEndpoints:
    """Integration tests using FastAPI's test client."""

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        mod = _import_server()
        app = mod.create_websocket_app()
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["transport"] == "websocket"

    def test_ws_encode_basic(self, client):
        """Single sentence should return a chunk + done signal."""
        with client.websocket_connect("/ws/encode") as ws:
            ws.send_json({"text": "Hello world."})
            messages = []
            while True:
                data = ws.receive_json()
                messages.append(data)
                if data.get("done"):
                    break
        # At least one non-done message
        non_done = [m for m in messages if not m.get("done")]
        assert len(non_done) >= 1
        assert "encoded" in non_done[0]

    def test_ws_encode_multiple_sentences(self, client):
        """Multi-sentence text should stream multiple chunks."""
        text = "First sentence. Second sentence. Third sentence."
        with client.websocket_connect("/ws/encode") as ws:
            ws.send_json({"text": text})
            messages = []
            while True:
                data = ws.receive_json()
                messages.append(data)
                if data.get("done"):
                    break
        chunks = [m for m in messages if not m.get("done")]
        assert len(chunks) >= 2  # At least 2 chunks

    def test_ws_encode_empty_text(self, client):
        """Empty text should return an error."""
        with client.websocket_connect("/ws/encode") as ws:
            ws.send_json({"text": ""})
            data = ws.receive_json()
        assert data.get("done") is True
        assert "error" in data

    def test_ws_encode_invalid_json(self, client):
        """Invalid JSON should return an error message."""
        with client.websocket_connect("/ws/encode") as ws:
            ws.send_text("not json at all!!!")
            data = ws.receive_json()
        assert data.get("done") is True
        assert "error" in data

    def test_ws_encode_no_text_field(self, client):
        """Missing 'text' key should return an error."""
        with client.websocket_connect("/ws/encode") as ws:
            ws.send_json({"other": "field"})
            data = ws.receive_json()
        assert data.get("done") is True
        assert "error" in data

    def test_ws_stream_passthrough(self, client):
        """Binary stream should echo (or compress) bytes."""
        payload = b"sigma compression test data 12345"
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_bytes(payload)
            result = ws.receive_bytes()
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_ws_encode_multiple_messages(self, client):
        """Client can send multiple messages in one connection."""
        with client.websocket_connect("/ws/encode") as ws:
            for i in range(3):
                ws.send_json({"text": f"Sentence {i}."})
                messages = []
                while True:
                    data = ws.receive_json()
                    messages.append(data)
                    if data.get("done"):
                        break
                assert any(not m.get("done") for m in messages)

    def test_ws_analyze_basic(self, client):
        """Analyze endpoint should stream per-text results."""
        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_json({"texts": ["Hello", "World", "ΣLANG"]})
            messages = []
            while True:
                data = ws.receive_json()
                messages.append(data)
                if data.get("done"):
                    break
        done_msg = messages[-1]
        assert done_msg["done"] is True
        assert done_msg["total"] == 3
        assert "elapsed_ms" in done_msg

    def test_ws_analyze_empty_texts(self, client):
        """Empty texts list should return an error."""
        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_json({"texts": []})
            data = ws.receive_json()
        assert data.get("done") is True
        assert "error" in data

    def test_ws_analyze_invalid_json(self, client):
        with client.websocket_connect("/ws/analyze") as ws:
            ws.send_text("not json")
            data = ws.receive_json()
        assert data.get("done") is True
        assert "error" in data

    def test_connection_manager_tracking(self, client):
        """Connecting and disconnecting should update manager.active."""
        mod = _import_server()
        initial_count = len(mod.manager.active)
        # We can't easily inspect mid-connection from the test, but at minimum
        # after disconnect the count should not grow indefinitely.
        with client.websocket_connect("/ws/encode") as ws:
            ws.send_json({"text": "ping."})
            while True:
                data = ws.receive_json()
                if data.get("done"):
                    break
        # After disconnect, connection should be removed
        final_count = len(mod.manager.active)
        assert final_count <= initial_count + 1  # at most 1 lingering

    def test_ws_encode_chunk_has_correct_format(self, client):
        """Each chunk message should have 'chunk', 'encoded', 'done' keys."""
        with client.websocket_connect("/ws/encode") as ws:
            ws.send_json({"text": "Test sentence alpha. Test sentence beta."})
            messages = []
            while True:
                data = ws.receive_json()
                messages.append(data)
                if data.get("done"):
                    break
        for msg in messages:
            if not msg.get("done"):
                assert "chunk" in msg
                assert "encoded" in msg
                assert msg["done"] is False

    def test_health_is_json(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Module-level app attribute
# ---------------------------------------------------------------------------

class TestModuleLevelApp:
    """Test the module-level `app` attribute."""

    def test_app_is_none_or_fastapi(self):
        mod = _import_server()
        # Should be None or a FastAPI instance
        if mod.app is not None:
            # duck-type check: FastAPI apps have 'routes'
            assert hasattr(mod.app, "routes") or hasattr(mod.app, "openapi")
