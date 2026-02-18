"""
Mobile Export - Phase 7 Track 9

Export SigmaLang codebooks and PQ-compressed models to lightweight
formats suitable for edge/mobile deployment.

Export Formats:
    1. SigmaPack: Custom binary format (minimal overhead)
    2. JSON: Human-readable, larger but debuggable
    3. NumPy: .npz format for Python-based mobile/edge
    4. C Header: Embed codebook directly in C/C++ code

Usage:
    from sigmalang.export.mobile_export import MobileExporter

    exporter = MobileExporter()
    exporter.export_sigmapack(codebook, pq_codes, "model.sigma")
    exporter.export_c_header(codebook, "sigmalang_codebook.h")
"""

import json
import struct
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# SigmaPack format version
SIGMAPACK_MAGIC = b'\xCE\xA3\x4D\x4F'  # Sigma-MO (mobile)
SIGMAPACK_VERSION = 1


class MobileExporter:
    """
    Export SigmaLang models to mobile-friendly formats.
    """

    def export_sigmapack(
        self,
        codebook: np.ndarray,
        output_path: str,
        pq_codes: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Export to SigmaPack binary format.

        Format:
            [4B] Magic: \\xCE\\xA3\\x4D\\x4F
            [1B] Version
            [4B] Codebook size (N)
            [4B] Embedding dim (D)
            [1B] Has PQ codes (0/1)
            [1B] Quantization bits (8=uint8, 16=float16, 32=float32)
            [4B] Metadata JSON length
            [?B] Metadata JSON
            [?B] Codebook data
            [?B] PQ codes (if present)
        """
        N, D = codebook.shape
        quant_bits = 16  # Default to FP16 for mobile

        # Quantize to FP16
        cb_fp16 = codebook.astype(np.float16)

        parts = []
        parts.append(SIGMAPACK_MAGIC)
        parts.append(struct.pack('<B', SIGMAPACK_VERSION))
        parts.append(struct.pack('<II', N, D))
        parts.append(struct.pack('<B', 1 if pq_codes is not None else 0))
        parts.append(struct.pack('<B', quant_bits))

        # Metadata
        meta_json = json.dumps(metadata or {}).encode('utf-8')
        parts.append(struct.pack('<I', len(meta_json)))
        parts.append(meta_json)

        # Codebook data
        parts.append(cb_fp16.tobytes())

        # PQ codes
        if pq_codes is not None:
            pq_bytes = pq_codes.astype(np.uint8).tobytes()
            parts.append(struct.pack('<I', len(pq_bytes)))
            parts.append(pq_bytes)

        data = b''.join(parts)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        stats = {
            'format': 'sigmapack',
            'output_path': str(path),
            'size_bytes': len(data),
            'codebook_entries': N,
            'embedding_dim': D,
            'quantization': f'float{quant_bits}',
            'has_pq': pq_codes is not None,
        }
        logger.info(f"Exported SigmaPack: {stats['size_bytes']} bytes to {path}")
        return stats

    def load_sigmapack(self, input_path: str) -> Dict[str, Any]:
        """Load a SigmaPack file."""
        data = Path(input_path).read_bytes()
        offset = 0

        magic = data[offset:offset + 4]
        offset += 4
        if magic != SIGMAPACK_MAGIC:
            raise ValueError("Invalid SigmaPack magic bytes")

        version = struct.unpack_from('<B', data, offset)[0]
        offset += 1

        N, D = struct.unpack_from('<II', data, offset)
        offset += 8

        has_pq = struct.unpack_from('<B', data, offset)[0]
        offset += 1

        quant_bits = struct.unpack_from('<B', data, offset)[0]
        offset += 1

        meta_len = struct.unpack_from('<I', data, offset)[0]
        offset += 4

        metadata = json.loads(data[offset:offset + meta_len].decode('utf-8'))
        offset += meta_len

        # Load codebook
        dtype = {8: np.uint8, 16: np.float16, 32: np.float32}[quant_bits]
        cb_bytes = N * D * np.dtype(dtype).itemsize
        codebook = np.frombuffer(data[offset:offset + cb_bytes], dtype=dtype).reshape(N, D)
        offset += cb_bytes

        result = {
            'version': version,
            'codebook': codebook.astype(np.float32),
            'metadata': metadata,
            'pq_codes': None,
        }

        if has_pq:
            pq_len = struct.unpack_from('<I', data, offset)[0]
            offset += 4
            result['pq_codes'] = np.frombuffer(
                data[offset:offset + pq_len], dtype=np.uint8
            )

        return result

    def export_json(
        self,
        codebook: np.ndarray,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Export codebook as JSON (human-readable, larger)."""
        N, D = codebook.shape

        export = {
            'format': 'sigmalang_codebook_json',
            'version': 1,
            'codebook_size': N,
            'embedding_dim': D,
            'metadata': metadata or {},
            'embeddings': codebook.tolist(),
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export, f)

        size = path.stat().st_size
        logger.info(f"Exported JSON: {size} bytes to {path}")
        return {'format': 'json', 'output_path': str(path), 'size_bytes': size}

    def export_npz(
        self,
        codebook: np.ndarray,
        output_path: str,
        pq_codes: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Export as NumPy .npz archive."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {'codebook': codebook.astype(np.float16)}
        if pq_codes is not None:
            arrays['pq_codes'] = pq_codes

        np.savez_compressed(str(path), **arrays)

        size = path.stat().st_size
        logger.info(f"Exported NPZ: {size} bytes to {path}")
        return {'format': 'npz', 'output_path': str(path), 'size_bytes': size}

    def export_c_header(
        self,
        codebook: np.ndarray,
        output_path: str,
        array_name: str = "sigmalang_codebook",
    ) -> Dict[str, Any]:
        """
        Export codebook as a C header file for embedding in C/C++ applications.

        Generates a static const float array.
        """
        N, D = codebook.shape
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "/* Auto-generated by SigmaLang Mobile Exporter */",
            f"/* Codebook: {N} entries x {D} dimensions */",
            f"#ifndef {array_name.upper()}_H",
            f"#define {array_name.upper()}_H",
            "",
            f"#define {array_name.upper()}_SIZE {N}",
            f"#define {array_name.upper()}_DIM {D}",
            "",
            f"static const float {array_name}[{N}][{D}] = {{",
        ]

        for i in range(N):
            values = ", ".join(f"{v:.6f}f" for v in codebook[i])
            comma = "," if i < N - 1 else ""
            lines.append(f"  {{{values}}}{comma}")

        lines.extend([
            "};",
            "",
            f"#endif /* {array_name.upper()}_H */",
            "",
        ])

        content = "\n".join(lines)
        path.write_text(content, encoding='utf-8')

        size = path.stat().st_size
        logger.info(f"Exported C header: {size} bytes to {path}")
        return {'format': 'c_header', 'output_path': str(path), 'size_bytes': size}

    def get_size_comparison(
        self,
        codebook: np.ndarray,
        pq_codes: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compare sizes across formats without writing files."""
        N, D = codebook.shape
        return {
            'original_fp32_bytes': N * D * 4,
            'fp16_bytes': N * D * 2,
            'pq_codes_bytes': pq_codes.nbytes if pq_codes is not None else 0,
            'sigmapack_estimate': N * D * 2 + 64,  # FP16 + header
            'json_estimate': N * D * 10,  # ~10 chars per float
            'c_header_estimate': N * D * 12,  # ~12 chars per float
            'compression_ratio_fp16': round(N * D * 4 / max(1, N * D * 2), 1),
        }
