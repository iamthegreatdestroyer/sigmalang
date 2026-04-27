"""
Image Semantic Encoder - Phase 7 Track 1

Encodes images into SigmaLang primitives by extracting semantic content:
scene descriptions, objects, spatial relations, colors, and textures.
Maps visual semantics to the same 256 Sigma-Primitive system used for text.

Architecture:
    Image Input (PIL/path/bytes)
        |
        v
    Feature Extractor
        |-- Color histogram (dominant colors)
        |-- Edge density (texture complexity)
        |-- Region segmentation (coarse grid)
        |-- Brightness/contrast analysis
        |
        v
    Semantic Descriptor
        |-- Scene type classification
        |-- Object-like region descriptors
        |-- Spatial layout encoding
        |-- Color palette primitives
        |
        v
    Primitive Mapper
        |-- Map descriptors to Sigma-Primitives
        |-- Compose semantic tree
        |
        v
    Sigma-Encoded Output (same format as text)

Dependencies:
    - Pillow (optional, graceful fallback to metadata-only mode)

Research Basis:
    - SemantiCodec (2024): dual-encoder semantic codec
    - Modality-Agnostic VQ-INR (2023): unified compression across data types
    - "When Tokens Talk Too Much" (Jul 2025): multimodal token compression

Usage:
    from sigmalang.core.image_encoder import ImageEncoder

    encoder = ImageEncoder()
    result = encoder.encode("photo.jpg")
    print(f"Encoded to {result['num_primitives']} primitives")
    print(f"Scene: {result['scene_type']}")
    print(f"Compression: {result['ratio']:.0f}x")
"""

import hashlib
import logging
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import Pillow
_PIL_AVAILABLE = False
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    logger.debug("Pillow not available; image encoder will use metadata-only mode")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ImageEncoderConfig:
    """Configuration for image semantic encoding."""

    grid_size: int = 8              # NxN grid for spatial analysis
    color_bins: int = 16            # Bins per channel for color histogram
    max_primitives: int = 64        # Max primitives per image
    target_size: int = 256          # Resize longest edge before analysis
    include_spatial: bool = True    # Include spatial layout encoding


# =============================================================================
# Scene Types (mapped to primitives)
# =============================================================================

SCENE_TYPES = {
    'outdoor_bright': {'primitives': [0x10, 0x20], 'label': 'bright outdoor scene'},
    'outdoor_dark': {'primitives': [0x10, 0x21], 'label': 'dark outdoor scene'},
    'indoor_bright': {'primitives': [0x11, 0x20], 'label': 'bright indoor scene'},
    'indoor_dark': {'primitives': [0x11, 0x21], 'label': 'dark indoor scene'},
    'natural': {'primitives': [0x12, 0x30], 'label': 'natural/landscape'},
    'urban': {'primitives': [0x13, 0x31], 'label': 'urban/architectural'},
    'abstract': {'primitives': [0x14, 0x40], 'label': 'abstract/pattern'},
    'text_heavy': {'primitives': [0x15, 0x50], 'label': 'text/document'},
}

# Color-to-primitive mappings
COLOR_PRIMITIVES = {
    'red': 0x60, 'orange': 0x61, 'yellow': 0x62, 'green': 0x63,
    'cyan': 0x64, 'blue': 0x65, 'purple': 0x66, 'pink': 0x67,
    'white': 0x68, 'gray': 0x69, 'black': 0x6A, 'brown': 0x6B,
}

# Texture-to-primitive mappings
TEXTURE_PRIMITIVES = {
    'smooth': 0x70, 'rough': 0x71, 'textured': 0x72,
    'patterned': 0x73, 'gradient': 0x74, 'noisy': 0x75,
}


# =============================================================================
# Feature Extractors
# =============================================================================

class ColorAnalyzer:
    """Extract color features from an image."""

    @staticmethod
    def dominant_colors(pixels: np.ndarray, n_colors: int = 5) -> List[Dict[str, Any]]:
        """
        Find dominant colors using histogram binning.

        Args:
            pixels: (H*W, 3) RGB array
            n_colors: Number of dominant colors to return

        Returns:
            List of {color_name, rgb, proportion}
        """
        if len(pixels) == 0:
            return []

        # Quantize to reduce color space
        quantized = (pixels // 32) * 32 + 16  # 8 levels per channel
        unique, counts = np.unique(quantized, axis=0, return_counts=True)

        # Sort by frequency
        sorted_idx = np.argsort(counts)[::-1][:n_colors]

        results = []
        total = len(pixels)
        for idx in sorted_idx:
            rgb = unique[idx].tolist()
            proportion = int(counts[idx]) / total
            color_name = ColorAnalyzer._rgb_to_name(rgb)
            results.append({
                'color_name': color_name,
                'rgb': rgb,
                'proportion': round(proportion, 3),
                'primitive': COLOR_PRIMITIVES.get(color_name, 0x69),
            })

        return results

    @staticmethod
    def _rgb_to_name(rgb: List[int]) -> str:
        """Map RGB to a color name."""
        r, g, b = rgb
        brightness = (r + g + b) / 3

        if brightness > 220:
            return 'white'
        if brightness < 35:
            return 'black'
        if brightness < 80 and max(r, g, b) - min(r, g, b) < 30:
            return 'gray'

        # Check saturation
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        if max_c - min_c < 30:
            return 'gray'

        # Hue-based classification
        if r > g and r > b:
            if g > 150:
                return 'yellow' if g > b else 'orange'
            return 'red' if b < 100 else 'pink'
        elif g > r and g > b:
            return 'green' if r < 150 else 'yellow'
        else:  # b dominant
            return 'blue' if r < 100 else 'purple'


class TextureAnalyzer:
    """Analyze image texture characteristics."""

    @staticmethod
    def analyze(gray_pixels: np.ndarray, grid_size: int = 8) -> Dict[str, Any]:
        """
        Analyze texture using edge density and variance.

        Args:
            gray_pixels: (H, W) grayscale array
            grid_size: Grid divisions for spatial analysis

        Returns:
            Texture descriptor dict
        """
        h, w = gray_pixels.shape

        # Simple edge detection via gradient magnitude
        if h > 2 and w > 2:
            gx = np.diff(gray_pixels.astype(np.float32), axis=1)[:, :-1] if w > 1 else np.zeros((h, 1))
            gy = np.diff(gray_pixels.astype(np.float32), axis=0)[:-1, :] if h > 1 else np.zeros((1, w))
            min_h = min(gx.shape[0], gy.shape[0])
            min_w = min(gx.shape[1], gy.shape[1])
            edges = np.sqrt(gx[:min_h, :min_w]**2 + gy[:min_h, :min_w]**2)
            edge_density = float(edges.mean()) / 128.0
        else:
            edge_density = 0.0

        # Overall variance (texture complexity)
        variance = float(gray_pixels.astype(np.float32).var()) / (255**2)

        # Classify texture
        if edge_density < 0.05 and variance < 0.01:
            texture_type = 'smooth'
        elif edge_density > 0.3:
            texture_type = 'rough'
        elif variance > 0.1:
            texture_type = 'noisy'
        elif edge_density > 0.1:
            texture_type = 'textured'
        else:
            texture_type = 'gradient'

        return {
            'edge_density': round(edge_density, 4),
            'variance': round(variance, 4),
            'texture_type': texture_type,
            'primitive': TEXTURE_PRIMITIVES.get(texture_type, 0x72),
        }


class SceneClassifier:
    """Classify scene type from image features."""

    @staticmethod
    def classify(
        brightness: float,
        edge_density: float,
        color_variety: int,
        dominant_color: str
    ) -> str:
        """Classify scene type based on features."""
        if edge_density > 0.25 and color_variety < 3:
            return 'text_heavy'
        if edge_density > 0.2:
            if brightness > 0.5:
                return 'urban' if color_variety < 5 else 'outdoor_bright'
            return 'outdoor_dark'
        if color_variety > 6:
            return 'natural' if dominant_color in ('green', 'blue', 'brown') else 'abstract'
        if brightness > 0.6:
            return 'indoor_bright'
        return 'indoor_dark'


# =============================================================================
# Spatial Layout Encoder
# =============================================================================

class SpatialLayoutEncoder:
    """Encode spatial layout of an image as a grid of feature descriptors."""

    def __init__(self, grid_size: int = 8):
        self.grid_size = grid_size

    def encode(self, pixels: np.ndarray) -> List[Dict[str, Any]]:
        """
        Encode image as a grid of region descriptors.

        Args:
            pixels: (H, W, 3) RGB array

        Returns:
            List of region descriptors with position and features
        """
        h, w = pixels.shape[:2]
        cell_h = max(1, h // self.grid_size)
        cell_w = max(1, w // self.grid_size)
        regions = []

        for gy in range(self.grid_size):
            for gx in range(self.grid_size):
                y1 = gy * cell_h
                y2 = min((gy + 1) * cell_h, h)
                x1 = gx * cell_w
                x2 = min((gx + 1) * cell_w, w)

                cell = pixels[y1:y2, x1:x2]
                if cell.size == 0:
                    continue

                mean_color = cell.reshape(-1, 3).mean(axis=0).astype(int).tolist()
                brightness = sum(mean_color) / (3 * 255)

                regions.append({
                    'grid_x': gx,
                    'grid_y': gy,
                    'mean_rgb': mean_color,
                    'brightness': round(brightness, 3),
                    'color_name': ColorAnalyzer._rgb_to_name(mean_color),
                })

        return regions


# =============================================================================
# Main Image Encoder
# =============================================================================

class ImageEncoder:
    """
    Encode images into SigmaLang primitives via semantic feature extraction.

    Extracts scene type, dominant colors, texture, and spatial layout,
    then maps these to Sigma-Primitives for compressed representation.

    Usage:
        encoder = ImageEncoder()

        # Encode from file path
        result = encoder.encode("photo.jpg")

        # Encode from PIL Image
        result = encoder.encode_pil(pil_image)

        # Encode from numpy array
        result = encoder.encode_array(rgb_array)
    """

    def __init__(self, config: Optional[ImageEncoderConfig] = None):
        self.config = config or ImageEncoderConfig()
        self._spatial = SpatialLayoutEncoder(self.config.grid_size)
        self._stats = {'images_encoded': 0, 'total_primitives': 0}

    def encode(self, source: Any) -> Dict[str, Any]:
        """
        Encode an image from various sources.

        Args:
            source: File path (str/Path), PIL Image, or numpy array

        Returns:
            Encoding result with primitives, scene type, and stats
        """
        if isinstance(source, (str, Path)):
            return self._encode_from_path(Path(source))
        elif isinstance(source, np.ndarray):
            return self.encode_array(source)
        elif _PIL_AVAILABLE and isinstance(source, Image.Image):
            return self.encode_pil(source)
        else:
            return {'error': f'Unsupported source type: {type(source).__name__}'}

    def _encode_from_path(self, path: Path) -> Dict[str, Any]:
        """Encode from file path."""
        if not path.exists():
            return {'error': f'File not found: {path}'}

        if not _PIL_AVAILABLE:
            # Metadata-only mode
            file_size = path.stat().st_size
            return self._metadata_only_encode(path, file_size)

        try:
            img = Image.open(path).convert('RGB')
            result = self.encode_pil(img)
            result['source_file'] = str(path)
            result['file_size'] = path.stat().st_size
            return result
        except Exception as e:
            return {'error': f'Failed to load image: {e}'}

    def encode_pil(self, img) -> Dict[str, Any]:
        """Encode from PIL Image."""
        # Resize for efficiency
        w, h = img.size
        max_dim = self.config.target_size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        pixels = np.array(img)
        result = self.encode_array(pixels)
        result['original_size'] = (w, h)
        return result

    def encode_array(self, pixels: np.ndarray) -> Dict[str, Any]:
        """
        Encode from numpy RGB array.

        Args:
            pixels: (H, W, 3) uint8 RGB array

        Returns:
            Encoding result dict
        """
        if len(pixels.shape) != 3 or pixels.shape[2] != 3:
            return {'error': f'Expected (H, W, 3) RGB array, got shape {pixels.shape}'}

        h, w = pixels.shape[:2]
        flat_pixels = pixels.reshape(-1, 3)

        # Grayscale for texture analysis
        gray = (0.299 * pixels[:, :, 0] + 0.587 * pixels[:, :, 1] + 0.114 * pixels[:, :, 2]).astype(np.uint8)

        # 1. Color analysis
        colors = ColorAnalyzer.dominant_colors(flat_pixels, n_colors=5)

        # 2. Texture analysis
        texture = TextureAnalyzer.analyze(gray, self.config.grid_size)

        # 3. Brightness
        brightness = float(gray.mean()) / 255.0

        # 4. Scene classification
        dominant_color = colors[0]['color_name'] if colors else 'gray'
        scene_type = SceneClassifier.classify(
            brightness, texture['edge_density'],
            len(colors), dominant_color
        )

        # 5. Spatial layout
        spatial = []
        if self.config.include_spatial:
            spatial = self._spatial.encode(pixels)

        # 6. Build primitive sequence
        primitives = self._build_primitives(scene_type, colors, texture, spatial)

        # 7. Serialize
        encoded = self._serialize(primitives, scene_type, w, h)

        # Stats
        self._stats['images_encoded'] += 1
        self._stats['total_primitives'] += len(primitives)

        original_bytes = h * w * 3
        compressed_bytes = len(encoded)

        return {
            'primitives': primitives,
            'num_primitives': len(primitives),
            'scene_type': scene_type,
            'scene_label': SCENE_TYPES.get(scene_type, {}).get('label', scene_type),
            'dominant_colors': [c['color_name'] for c in colors],
            'texture': texture['texture_type'],
            'brightness': round(brightness, 3),
            'dimensions': (w, h),
            'original_bytes': original_bytes,
            'encoded_bytes': compressed_bytes,
            'ratio': round(original_bytes / max(1, compressed_bytes), 1),
            'encoded_data': encoded,
        }

    def _build_primitives(
        self,
        scene_type: str,
        colors: List[Dict],
        texture: Dict,
        spatial: List[Dict]
    ) -> List[int]:
        """Build primitive sequence from extracted features."""
        primitives = []

        # Scene primitives
        scene_prims = SCENE_TYPES.get(scene_type, {}).get('primitives', [0x14])
        primitives.extend(scene_prims)

        # Color primitives (top 5)
        for color in colors[:5]:
            primitives.append(color['primitive'])

        # Texture primitive
        primitives.append(texture['primitive'])

        # Spatial primitives (compressed grid summary)
        if spatial:
            # Encode as unique region colors
            seen = set()
            for region in spatial:
                p = COLOR_PRIMITIVES.get(region['color_name'], 0x69)
                if p not in seen and len(primitives) < self.config.max_primitives:
                    primitives.append(p)
                    seen.add(p)

        return primitives[:self.config.max_primitives]

    def _serialize(self, primitives: List[int], scene_type: str, w: int, h: int) -> bytes:
        """Serialize encoded image to bytes."""
        parts = []
        # Magic: "SI" (Sigma Image)
        parts.append(b'\xCE\xA3\x49')
        # Version
        parts.append(struct.pack('B', 1))
        # Dimensions
        parts.append(struct.pack('>HH', w, h))
        # Scene type hash (1 byte)
        parts.append(struct.pack('B', hash(scene_type) % 256))
        # Primitive count
        parts.append(struct.pack('>H', len(primitives)))
        # Primitives
        parts.append(bytes(primitives))
        return b''.join(parts)

    def _metadata_only_encode(self, path: Path, file_size: int) -> Dict[str, Any]:
        """Encode using only file metadata (no Pillow)."""
        suffix = path.suffix.lower()
        scene_type = 'abstract'

        return {
            'primitives': [0x14, 0x40],
            'num_primitives': 2,
            'scene_type': scene_type,
            'scene_label': 'metadata-only (Pillow not installed)',
            'file_size': file_size,
            'file_type': suffix,
            'encoded_bytes': 16,
            'ratio': file_size / 16,
            'note': 'Install Pillow for full image analysis: pip install Pillow',
        }

    def get_stats(self) -> Dict[str, Any]:
        return self._stats.copy()
