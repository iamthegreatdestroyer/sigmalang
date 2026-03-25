"""
Tests for KV-Cache Mixed-Precision Quantization (kv_quantization.py)
"""

import pytest
import numpy as np
from sigmalang.core.kv_quantization import (
    KVQuantizer,
    MixedPrecisionKVCache,
    MixedPrecisionResult,
    Precision,
    QuantizationResult,
)


# Shared fixture helpers
def _random_tensor(shape, seed=0, dtype=np.float32, scale=1.0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * scale).astype(dtype)


# ===========================================================================
# Precision Enum
# ===========================================================================

class TestPrecision:
    def test_bytes_per_element(self):
        assert Precision.FP32.bytes_per_element == 4.0
        assert Precision.FP16.bytes_per_element == 2.0
        assert Precision.INT8.bytes_per_element == 1.0
        assert Precision.INT4.bytes_per_element == 0.5
        assert Precision.INT2.bytes_per_element == 0.25

    def test_compression_vs_fp32(self):
        assert Precision.FP16.compression_vs_fp32 == 2.0
        assert Precision.INT8.compression_vs_fp32 == 4.0
        assert Precision.INT4.compression_vs_fp32 == 8.0


# ===========================================================================
# KVQuantizer — FP16
# ===========================================================================

class TestKVQuantizerFP16:
    def test_fp16_output_dtype(self):
        q = KVQuantizer()
        x = _random_tensor((16, 64))
        r = q.quantize_fp16(x)
        assert r.data.dtype == np.float16

    def test_fp16_compression_ratio(self):
        q = KVQuantizer()
        r = q.quantize_fp16(_random_tensor((4, 32)))
        assert r.compression_ratio == 2.0

    def test_fp16_precision_label(self):
        q = KVQuantizer()
        r = q.quantize_fp16(_random_tensor((4, 32)))
        assert r.precision == Precision.FP16

    def test_fp16_dequantize_shape(self):
        q = KVQuantizer()
        x = _random_tensor((8, 64))
        r = q.quantize_fp16(x)
        out = q.dequantize_fp16(r)
        assert out.shape == x.shape
        assert out.dtype == np.float32

    def test_fp16_low_error(self):
        q = KVQuantizer()
        x = _random_tensor((16, 128))
        r = q.quantize_fp16(x)
        out = q.dequantize_fp16(r)
        rmse = float(np.sqrt(np.mean((x - out) ** 2)))
        # FP16 is very accurate; RMSE should be tiny
        assert rmse < 0.01, f"FP16 RMSE too high: {rmse}"

    def test_fp16_roundtrip_close(self):
        q = KVQuantizer()
        x = np.array([[0.1, -0.2, 3.14, -1.0, 0.0]], dtype=np.float32)
        r = q.quantize_fp16(x)
        out = q.dequantize_fp16(r)
        np.testing.assert_allclose(x, out, rtol=1e-3, atol=1e-3)

    def test_fp16_zeros(self):
        q = KVQuantizer()
        x = np.zeros((8, 64), dtype=np.float32)
        r = q.quantize_fp16(x)
        out = q.dequantize_fp16(r)
        np.testing.assert_array_equal(out, np.zeros_like(x))


# ===========================================================================
# KVQuantizer — INT8
# ===========================================================================

class TestKVQuantizerINT8:
    def test_int8_output_dtype(self):
        q = KVQuantizer()
        r = q.quantize_int8(_random_tensor((10, 64)))
        assert r.data.dtype == np.int8

    def test_int8_compression_ratio(self):
        q = KVQuantizer()
        r = q.quantize_int8(_random_tensor((10, 64)))
        assert r.compression_ratio == 4.0

    def test_int8_scale_positive(self):
        q = KVQuantizer()
        r = q.quantize_int8(_random_tensor((10, 64), scale=5.0))
        assert r.scale > 0

    def test_int8_values_in_range(self):
        q = KVQuantizer()
        r = q.quantize_int8(_random_tensor((50, 128)))
        assert r.data.min() >= -128
        assert r.data.max() <= 127

    def test_int8_zero_tensor(self):
        q = KVQuantizer()
        x = np.zeros((8, 32), dtype=np.float32)
        r = q.quantize_int8(x)
        out = q.dequantize_int8(r)
        np.testing.assert_array_equal(out, np.zeros_like(x))

    def test_int8_dequantize_shape(self):
        q = KVQuantizer()
        x = _random_tensor((8, 64))
        r = q.quantize_int8(x)
        out = q.dequantize_int8(r)
        assert out.shape == x.shape
        assert out.dtype == np.float32

    def test_int8_error_acceptable(self):
        q = KVQuantizer()
        x = _random_tensor((100, 64), scale=1.0)
        r = q.quantize_int8(x)
        out = q.dequantize_int8(r)
        # INT8 has ~0.78% quantization error relative to max
        rmse = float(np.sqrt(np.mean((x - out) ** 2)))
        max_val = float(np.abs(x).max())
        relative_err = rmse / max(max_val, 1e-8)
        assert relative_err < 0.02, f"INT8 relative RMSE too high: {relative_err:.4f}"

    def test_int8_snr_adequate(self):
        q = KVQuantizer()
        x = _random_tensor((64, 64), scale=1.0)
        r = q.quantize_int8(x)
        metrics = q.reconstruction_error(x, r)
        assert metrics["snr_db"] > 40.0, f"INT8 SNR too low: {metrics['snr_db']:.1f}dB"

    @pytest.mark.parametrize("shape", [(1, 8), (32, 64), (128, 256), (4, 4)])
    def test_int8_various_shapes(self, shape):
        q = KVQuantizer()
        x = _random_tensor(shape)
        r = q.quantize_int8(x)
        out = q.dequantize_int8(r)
        assert out.shape == x.shape


# ===========================================================================
# KVQuantizer — INT4
# ===========================================================================

class TestKVQuantizerINT4:
    def test_int4_output_dtype(self):
        q = KVQuantizer()
        r = q.quantize_int4(_random_tensor((16, 64)))
        assert r.data.dtype == np.uint8

    def test_int4_dequantize_shape(self):
        q = KVQuantizer()
        x = _random_tensor((8, 64))
        r = q.quantize_int4(x)
        out = q.dequantize_int4(r)
        assert out.shape == x.shape
        assert out.dtype == np.float32

    def test_int4_compression_ratio(self):
        q = KVQuantizer()
        r = q.quantize_int4(_random_tensor((8, 64)))
        assert r.compression_ratio == 8.0

    def test_int4_zero_tensor(self):
        q = KVQuantizer()
        x = np.zeros((8, 32), dtype=np.float32)
        r = q.quantize_int4(x)
        out = q.dequantize_int4(r)
        np.testing.assert_array_almost_equal(out, np.zeros_like(x), decimal=5)

    def test_int4_dequantize_via_universal(self):
        q = KVQuantizer()
        x = _random_tensor((8, 32))
        r = q.quantize_int4(x)
        out = q.dequantize(r)
        assert out.shape == x.shape

    def test_int4_packing_symmetry(self):
        """INT4 pack → unpack should recover original values (within range)."""
        q = KVQuantizer()
        # Create values exactly in INT4 range
        x_vals = np.array([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7],
                          dtype=np.float32).reshape(1, -1)
        r = q.quantize_int4(x_vals)
        out = q.dequantize_int4(r)
        # Within INT4 range, reconstruction should be near-exact
        np.testing.assert_allclose(x_vals, out, atol=r.scale + 0.01)

    def test_int4_reduces_memory(self):
        q = KVQuantizer()
        x = _random_tensor((64, 128))
        r = q.quantize_int4(x)
        assert r.data.nbytes < x.nbytes


# ===========================================================================
# KVQuantizer — Adaptive
# ===========================================================================

class TestKVQuantizerAdaptive:
    def test_high_importance_fp16(self):
        q = KVQuantizer()
        x = _random_tensor((64,))
        r = q.quantize_adaptive(x, importance=0.9)
        assert r.precision == Precision.FP16

    def test_medium_importance_int8(self):
        q = KVQuantizer()
        x = _random_tensor((64,))
        r = q.quantize_adaptive(x, importance=0.5)
        assert r.precision == Precision.INT8

    def test_low_importance_int4(self):
        q = KVQuantizer()
        x = _random_tensor((64,))
        r = q.quantize_adaptive(x, importance=0.1)
        assert r.precision == Precision.INT4

    def test_boundary_fp16(self):
        """Exactly at fp16_threshold → FP16."""
        q = KVQuantizer()
        x = _random_tensor((32,))
        r = q.quantize_adaptive(x, importance=0.8, fp16_threshold=0.8)
        assert r.precision == Precision.FP16

    def test_universal_dequantize_all_precisions(self):
        q = KVQuantizer()
        x = _random_tensor((64,))
        for imp in [0.9, 0.5, 0.1]:
            r = q.quantize_adaptive(x, importance=imp)
            out = q.dequantize(r)
            assert out.shape == x.shape
            assert out.dtype == np.float32


# ===========================================================================
# Reconstruction Error Metrics
# ===========================================================================

class TestReconstructionError:
    def test_fp16_metrics(self):
        q = KVQuantizer()
        x = _random_tensor((32, 64))
        r = q.quantize_fp16(x)
        metrics = q.reconstruction_error(x, r)
        assert "mse" in metrics and "rmse" in metrics
        assert "max_abs_error" in metrics and "snr_db" in metrics
        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["compression_ratio"] == 2.0

    def test_int8_snr_better_than_int4(self):
        q = KVQuantizer()
        x = _random_tensor((32, 64), scale=2.0)
        r8 = q.quantize_int8(x)
        r4 = q.quantize_int4(x)
        m8 = q.reconstruction_error(x, r8)
        m4 = q.reconstruction_error(x, r4)
        assert m8["snr_db"] > m4["snr_db"], "INT8 should have better SNR than INT4"

    def test_perfect_reconstruction_fp16(self):
        q = KVQuantizer()
        # Small tensor where FP16 should be near-lossless
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        r = q.quantize_fp16(x)
        metrics = q.reconstruction_error(x, r)
        assert metrics["mse"] < 1e-4


# ===========================================================================
# MixedPrecisionKVCache
# ===========================================================================

class TestMixedPrecisionKVCache:
    def _make_kv(self, seq_len, d_k=64, d_v=64, seed=0):
        keys = _random_tensor((seq_len, d_k), seed=seed)
        values = _random_tensor((seq_len, d_v), seed=seed+1)
        attn = np.abs(_random_tensor((seq_len,), seed=seed+2, scale=1.0))
        return keys, values, attn

    def test_compress_returns_result(self):
        cache = MixedPrecisionKVCache()
        k, v, a = self._make_kv(32)
        result = cache.compress(k, v, a)
        assert isinstance(result, MixedPrecisionResult)

    def test_compress_token_count(self):
        cache = MixedPrecisionKVCache()
        k, v, a = self._make_kv(64)
        result = cache.compress(k, v, a)
        assert result.original_tokens == 64
        assert len(result.keys) == 64
        assert len(result.values) == 64

    def test_compress_reduces_memory(self):
        cache = MixedPrecisionKVCache()
        k, v, a = self._make_kv(128, d_k=512, d_v=512)
        result = cache.compress(k, v, a)
        assert result.memory_after_bytes < result.memory_before_bytes
        assert result.overall_ratio > 1.0

    def test_compress_without_attention(self):
        """compress() works without attention scores (uniform)."""
        cache = MixedPrecisionKVCache()
        k, v, _ = self._make_kv(32)
        result = cache.compress(k, v, attention_scores=None)
        assert result.original_tokens == 32

    def test_sink_tokens_are_fp16(self):
        """First N tokens should be assigned FP16 (attention sinks)."""
        sink_n = 4
        cache = MixedPrecisionKVCache(sink_tokens=sink_n)
        k, v, a = self._make_kv(32)
        result = cache.compress(k, v, a)
        for i in range(sink_n):
            assert result.token_precisions[i] == Precision.FP16, (
                f"Token {i} (sink) should be FP16, got {result.token_precisions[i]}"
            )

    def test_recent_tokens_boosted(self):
        """Recent window tokens should generally be FP16."""
        recent_w = 8
        cache = MixedPrecisionKVCache(recent_window=recent_w, fp16_threshold=0.7)
        k, v, a = self._make_kv(64)
        result = cache.compress(k, v, a)
        # Recent tokens should be FP16 or INT8 (never INT4 due to boost)
        for i in range(-recent_w, 0):
            assert result.token_precisions[i] in (Precision.FP16, Precision.INT8)

    def test_precision_counts(self):
        """Token precisions add up to total tokens."""
        cache = MixedPrecisionKVCache()
        k, v, a = self._make_kv(100)
        result = cache.compress(k, v, a)
        assert len(result.token_precisions) == 100

    def test_decompress_shape(self):
        """decompress() restores correct shapes."""
        cache = MixedPrecisionKVCache()
        k, v, a = self._make_kv(32, d_k=64, d_v=128)
        result = cache.compress(k, v, a)
        k_out, v_out = cache.decompress(result)
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape

    def test_decompress_dtype(self):
        """decompress() outputs float32."""
        cache = MixedPrecisionKVCache()
        k, v, a = self._make_kv(16)
        result = cache.compress(k, v, a)
        k_out, v_out = cache.decompress(result)
        assert k_out.dtype == np.float32
        assert v_out.dtype == np.float32

    def test_fp16_tokens_have_low_error(self):
        """FP16 tokens should reconstruct with very low error."""
        cache = MixedPrecisionKVCache(
            sink_tokens=4,
            fp16_threshold=0.99,   # Only sinks at FP16
        )
        k, v, a = self._make_kv(8, d_k=32, d_v=32)
        result = cache.compress(k, v, a)
        k_out, v_out = cache.decompress(result)
        # First 4 tokens (sinks) should be near-lossless
        for i in range(4):
            err = float(np.abs(k[i] - k_out[i]).max())
            assert err < 0.01, f"Sink token {i} reconstruction error: {err}"

    def test_elapsed_ms_positive(self):
        cache = MixedPrecisionKVCache()
        k, v, a = self._make_kv(32)
        result = cache.compress(k, v, a)
        assert result.elapsed_ms > 0

    def test_overall_ratio_positive(self):
        cache = MixedPrecisionKVCache()
        k, v, a = self._make_kv(64)
        result = cache.compress(k, v, a)
        assert result.overall_ratio > 0

    @pytest.mark.parametrize("seq_len,d_k,d_v", [
        (4, 16, 16),
        (32, 64, 64),
        (128, 256, 256),
        (512, 128, 128),
    ])
    def test_various_shapes(self, seq_len, d_k, d_v):
        cache = MixedPrecisionKVCache()
        k, v, a = self._make_kv(seq_len, d_k=d_k, d_v=d_v)
        result = cache.compress(k, v, a)
        k_out, v_out = cache.decompress(result)
        assert k_out.shape == (seq_len, d_k)
        assert v_out.shape == (seq_len, d_v)
