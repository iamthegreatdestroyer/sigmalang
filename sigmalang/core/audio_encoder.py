"""
Audio Semantic Encoder - Phase 7 Track 1

Encodes audio into SigmaLang primitives by extracting semantic features:
spectral characteristics, rhythm patterns, energy distribution, and
tonal properties. Maps audio semantics to Sigma-Primitives.

Architecture:
    Audio Input (WAV/path/numpy array)
        |
        v
    Feature Extractor
        |-- Spectral centroid (brightness)
        |-- Spectral bandwidth (tonal spread)
        |-- Zero crossing rate (noisiness)
        |-- RMS energy envelope
        |-- Tempo estimation
        |
        v
    Semantic Descriptor
        |-- Audio type (speech/music/ambient/noise)
        |-- Tonal qualities (bright/dark/warm/harsh)
        |-- Rhythm profile (fast/slow/steady/variable)
        |-- Energy contour (loud/quiet/dynamic)
        |
        v
    Primitive Mapper -> Sigma-Encoded Output

Dependencies:
    - numpy (required)
    - librosa (optional, graceful fallback to basic WAV analysis)

Research Basis:
    - SemantiCodec (2024): dual-encoder audio codec
    - X-Codec (2024): semantic features in audio generation

Usage:
    from sigmalang.core.audio_encoder import AudioEncoder

    encoder = AudioEncoder()
    result = encoder.encode("speech.wav")
    print(f"Type: {result['audio_type']}")
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

# Try to import librosa
_LIBROSA_AVAILABLE = False
try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    logger.debug("librosa not available; audio encoder will use basic WAV mode")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AudioEncoderConfig:
    """Configuration for audio semantic encoding."""

    sample_rate: int = 22050        # Target sample rate
    max_duration: float = 300.0     # Max seconds to analyze
    frame_size: int = 2048          # FFT frame size
    hop_size: int = 512             # Hop between frames
    max_primitives: int = 48        # Max primitives per audio


# =============================================================================
# Audio Primitives Mapping
# =============================================================================

AUDIO_TYPE_PRIMITIVES = {
    'speech': [0x80, 0x90],
    'music': [0x81, 0x91],
    'ambient': [0x82, 0x92],
    'noise': [0x83, 0x93],
    'silence': [0x84, 0x94],
}

TONAL_PRIMITIVES = {
    'bright': 0xA0, 'dark': 0xA1, 'warm': 0xA2, 'harsh': 0xA3,
    'thin': 0xA4, 'full': 0xA5, 'muffled': 0xA6, 'clear': 0xA7,
}

RHYTHM_PRIMITIVES = {
    'fast': 0xB0, 'slow': 0xB1, 'steady': 0xB2, 'variable': 0xB3,
    'percussive': 0xB4, 'flowing': 0xB5,
}

ENERGY_PRIMITIVES = {
    'loud': 0xC0, 'quiet': 0xC1, 'dynamic': 0xC2, 'flat': 0xC3,
    'crescendo': 0xC4, 'decrescendo': 0xC5,
}


# =============================================================================
# Basic WAV Reader (no dependencies)
# =============================================================================

def read_wav_basic(path: Path) -> Tuple[np.ndarray, int]:
    """
    Read a WAV file using only numpy + struct.
    Supports PCM 16-bit mono/stereo WAV files.

    Returns: (samples_float32, sample_rate)
    """
    with open(path, 'rb') as f:
        # RIFF header
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError("Not a WAV file")
        f.read(4)  # file size
        wave = f.read(4)
        if wave != b'WAVE':
            raise ValueError("Not a WAV file")

        # Find fmt and data chunks
        sample_rate = 22050
        channels = 1
        bits_per_sample = 16
        audio_data = b''

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack('<I', f.read(4))[0]

            if chunk_id == b'fmt ':
                fmt_data = f.read(chunk_size)
                struct.unpack('<H', fmt_data[0:2])[0]
                channels = struct.unpack('<H', fmt_data[2:4])[0]
                sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
                bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]
            elif chunk_id == b'data':
                audio_data = f.read(chunk_size)
                break
            else:
                f.read(chunk_size)

    if not audio_data:
        return np.zeros(0, dtype=np.float32), sample_rate

    # Convert to float32
    if bits_per_sample == 16:
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif bits_per_sample == 8:
        samples = (np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    # Convert stereo to mono
    if channels == 2 and len(samples) > 1:
        samples = (samples[0::2] + samples[1::2]) / 2.0

    return samples, sample_rate


# =============================================================================
# Feature Extractors
# =============================================================================

class SpectralAnalyzer:
    """Extract spectral features from audio."""

    @staticmethod
    def analyze(samples: np.ndarray, sr: int, frame_size: int = 2048) -> Dict[str, float]:
        """
        Compute spectral features.

        Returns:
            Dict with centroid, bandwidth, rolloff, zero_crossing_rate
        """
        if len(samples) < frame_size:
            return {
                'centroid': 0.0, 'bandwidth': 0.0,
                'rolloff': 0.0, 'zcr': 0.0,
            }

        # FFT of the entire signal (simplified)
        windowed = samples[:frame_size] * np.hanning(frame_size)
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(frame_size, 1.0 / sr)

        # Spectral centroid (weighted mean frequency)
        total = spectrum.sum()
        if total > 0:
            centroid = float(np.sum(freqs * spectrum) / total)
        else:
            centroid = 0.0

        # Spectral bandwidth
        if total > 0:
            bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / total))
        else:
            bandwidth = 0.0

        # Spectral rolloff (85% of energy)
        cumsum = np.cumsum(spectrum)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * total) if total > 0 else 0
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        # Zero crossing rate
        zcr = float(np.sum(np.abs(np.diff(np.sign(samples)))) / (2 * len(samples)))

        return {
            'centroid': round(centroid, 1),
            'bandwidth': round(bandwidth, 1),
            'rolloff': round(rolloff, 1),
            'zcr': round(zcr, 4),
        }


class EnergyAnalyzer:
    """Analyze energy characteristics."""

    @staticmethod
    def analyze(samples: np.ndarray, hop_size: int = 512) -> Dict[str, Any]:
        """Compute energy features."""
        if len(samples) == 0:
            return {'rms_mean': 0.0, 'rms_std': 0.0, 'dynamic_range_db': 0.0, 'profile': 'flat'}

        # RMS energy per frame
        n_frames = max(1, len(samples) // hop_size)
        rms_values = []
        for i in range(n_frames):
            start = i * hop_size
            end = min(start + hop_size, len(samples))
            frame = samples[start:end]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            rms_values.append(rms)

        rms_arr = np.array(rms_values)
        rms_mean = float(rms_arr.mean())
        rms_std = float(rms_arr.std())

        # Dynamic range
        rms_max = rms_arr.max()
        rms_min = max(rms_arr.min(), 1e-10)
        dynamic_range_db = 20 * math.log10(rms_max / rms_min) if rms_max > 0 else 0.0

        # Energy profile
        if rms_std / (rms_mean + 1e-10) > 0.5:
            profile = 'dynamic'
        elif rms_mean > 0.1:
            profile = 'loud'
        elif rms_mean < 0.01:
            profile = 'quiet'
        else:
            profile = 'flat'

        # Trend detection
        if n_frames > 4:
            first_half = rms_arr[:n_frames // 2].mean()
            second_half = rms_arr[n_frames // 2:].mean()
            if second_half > first_half * 1.5:
                profile = 'crescendo'
            elif first_half > second_half * 1.5:
                profile = 'decrescendo'

        return {
            'rms_mean': round(rms_mean, 4),
            'rms_std': round(rms_std, 4),
            'dynamic_range_db': round(dynamic_range_db, 1),
            'profile': profile,
            'primitive': ENERGY_PRIMITIVES.get(profile, 0xC3),
        }


class TempoEstimator:
    """Estimate tempo/rhythm from audio."""

    @staticmethod
    def estimate(samples: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Estimate tempo using autocorrelation of energy envelope.

        Returns:
            Dict with estimated_bpm, rhythm_type, primitive
        """
        if len(samples) < sr:  # Need at least 1 second
            return {'estimated_bpm': 0, 'rhythm_type': 'unknown', 'primitive': 0xB2}

        # Compute energy envelope
        hop = 512
        n_frames = len(samples) // hop
        envelope = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop
            end = min(start + hop, len(samples))
            envelope[i] = np.sqrt(np.mean(samples[start:end] ** 2))

        # Autocorrelation
        envelope = envelope - envelope.mean()
        if envelope.std() < 1e-10:
            return {'estimated_bpm': 0, 'rhythm_type': 'steady', 'primitive': 0xB2}

        autocorr = np.correlate(envelope, envelope, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]

        # Find first significant peak after lag 0
        # BPM range: 40-200 -> lag range in frames
        min_lag = int(60 / 200 * sr / hop)
        max_lag = min(int(60 / 40 * sr / hop), len(autocorr) - 1)

        if max_lag <= min_lag:
            return {'estimated_bpm': 0, 'rhythm_type': 'steady', 'primitive': 0xB2}

        search_range = autocorr[min_lag:max_lag + 1]
        if len(search_range) == 0:
            return {'estimated_bpm': 0, 'rhythm_type': 'steady', 'primitive': 0xB2}

        peak_lag = min_lag + int(np.argmax(search_range))
        bpm = int(60 / (peak_lag * hop / sr)) if peak_lag > 0 else 0

        # Classify rhythm
        if bpm == 0:
            rhythm_type = 'steady'
        elif bpm > 140:
            rhythm_type = 'fast'
        elif bpm < 70:
            rhythm_type = 'slow'
        else:
            # Check regularity
            peak_val = autocorr[peak_lag]
            if peak_val > 0.3 * autocorr[0]:
                rhythm_type = 'steady'
            else:
                rhythm_type = 'variable'

        return {
            'estimated_bpm': bpm,
            'rhythm_type': rhythm_type,
            'primitive': RHYTHM_PRIMITIVES.get(rhythm_type, 0xB2),
        }


class AudioTypeClassifier:
    """Classify audio type (speech/music/ambient/noise)."""

    @staticmethod
    def classify(spectral: Dict, energy: Dict, zcr: float) -> str:
        """Classify based on spectral and energy features."""
        centroid = spectral.get('centroid', 0)
        bandwidth = spectral.get('bandwidth', 0)
        rms = energy.get('rms_mean', 0)

        if rms < 0.005:
            return 'silence'

        # Speech: moderate centroid, moderate ZCR, moderate bandwidth
        if 200 < centroid < 3000 and 0.02 < zcr < 0.15 and bandwidth < 3000:
            return 'speech'

        # Music: wide bandwidth, varied energy
        if bandwidth > 3000 or (centroid > 500 and energy.get('dynamic_range_db', 0) > 20):
            return 'music'

        # Noise: high ZCR, high centroid
        if zcr > 0.2 or centroid > 5000:
            return 'noise'

        return 'ambient'


# =============================================================================
# Main Audio Encoder
# =============================================================================

class AudioEncoder:
    """
    Encode audio into SigmaLang primitives via semantic feature extraction.

    Usage:
        encoder = AudioEncoder()
        result = encoder.encode("speech.wav")
        print(f"Type: {result['audio_type']}, Primitives: {result['num_primitives']}")
    """

    def __init__(self, config: Optional[AudioEncoderConfig] = None):
        self.config = config or AudioEncoderConfig()
        self._stats = {'audio_encoded': 0, 'total_seconds': 0.0}

    def encode(self, source: Any) -> Dict[str, Any]:
        """
        Encode audio from file path or numpy array.

        Args:
            source: File path (str/Path) or numpy float32 array

        Returns:
            Encoding result with primitives and analysis
        """
        if isinstance(source, (str, Path)):
            return self._encode_from_path(Path(source))
        elif isinstance(source, np.ndarray):
            return self.encode_array(source, self.config.sample_rate)
        else:
            return {'error': f'Unsupported source type: {type(source).__name__}'}

    def _encode_from_path(self, path: Path) -> Dict[str, Any]:
        """Encode from file path."""
        if not path.exists():
            return {'error': f'File not found: {path}'}

        try:
            if _LIBROSA_AVAILABLE:
                samples, sr = librosa.load(
                    str(path), sr=self.config.sample_rate,
                    duration=self.config.max_duration
                )
            elif path.suffix.lower() == '.wav':
                samples, sr = read_wav_basic(path)
            else:
                return {
                    'error': f'Cannot read {path.suffix} without librosa. '
                             f'Install: pip install librosa'
                }
        except Exception as e:
            return {'error': f'Failed to load audio: {e}'}

        result = self.encode_array(samples, sr)
        result['source_file'] = str(path)
        result['file_size'] = path.stat().st_size
        return result

    def encode_array(self, samples: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Encode from numpy float32 audio array.

        Args:
            samples: (N,) float32 audio samples
            sr: Sample rate

        Returns:
            Encoding result dict
        """
        duration = len(samples) / max(1, sr)

        # Trim to max duration
        max_samples = int(self.config.max_duration * sr)
        if len(samples) > max_samples:
            samples = samples[:max_samples]

        # Extract features
        spectral = SpectralAnalyzer.analyze(samples, sr, self.config.frame_size)
        energy = EnergyAnalyzer.analyze(samples, self.config.hop_size)
        tempo = TempoEstimator.estimate(samples, sr)

        # Classify audio type
        audio_type = AudioTypeClassifier.classify(spectral, energy, spectral['zcr'])

        # Tonal classification
        tonal = self._classify_tonal(spectral)

        # Build primitive sequence
        primitives = self._build_primitives(audio_type, tonal, tempo, energy)

        # Serialize
        encoded = self._serialize(primitives, audio_type, duration, sr)

        # Stats
        self._stats['audio_encoded'] += 1
        self._stats['total_seconds'] += duration

        original_bytes = len(samples) * 4  # float32
        compressed_bytes = len(encoded)

        return {
            'primitives': primitives,
            'num_primitives': len(primitives),
            'audio_type': audio_type,
            'tonal_quality': tonal,
            'tempo': tempo,
            'energy_profile': energy['profile'],
            'spectral': spectral,
            'duration_seconds': round(duration, 2),
            'sample_rate': sr,
            'original_bytes': original_bytes,
            'encoded_bytes': compressed_bytes,
            'ratio': round(original_bytes / max(1, compressed_bytes), 1),
            'encoded_data': encoded,
        }

    def _classify_tonal(self, spectral: Dict) -> str:
        """Classify tonal quality from spectral features."""
        centroid = spectral.get('centroid', 0)
        bandwidth = spectral.get('bandwidth', 0)

        if centroid > 4000:
            return 'bright'
        if centroid < 500:
            return 'dark'
        if bandwidth > 4000:
            return 'full'
        if bandwidth < 500:
            return 'thin'
        if centroid > 2000:
            return 'clear'
        return 'warm'

    def _build_primitives(
        self, audio_type: str, tonal: str,
        tempo: Dict, energy: Dict
    ) -> List[int]:
        """Build primitive sequence from audio features."""
        primitives = []

        # Audio type
        type_prims = AUDIO_TYPE_PRIMITIVES.get(audio_type, [0x83])
        primitives.extend(type_prims)

        # Tonal quality
        primitives.append(TONAL_PRIMITIVES.get(tonal, 0xA2))

        # Rhythm
        primitives.append(tempo.get('primitive', 0xB2))

        # Energy
        primitives.append(energy.get('primitive', 0xC3))

        return primitives[:self.config.max_primitives]

    def _serialize(
        self, primitives: List[int], audio_type: str,
        duration: float, sr: int
    ) -> bytes:
        """Serialize encoded audio to bytes."""
        parts = []
        # Magic: "SA" (Sigma Audio)
        parts.append(b'\xCE\xA3\x41')
        # Version
        parts.append(struct.pack('B', 1))
        # Duration (seconds, 2 bytes)
        parts.append(struct.pack('>H', min(int(duration), 65535)))
        # Sample rate (2 bytes)
        parts.append(struct.pack('>H', min(sr, 65535)))
        # Audio type hash
        parts.append(struct.pack('B', hash(audio_type) % 256))
        # Primitive count
        parts.append(struct.pack('>H', len(primitives)))
        # Primitives
        parts.append(bytes(primitives))
        return b''.join(parts)

    def get_stats(self) -> Dict[str, Any]:
        return self._stats.copy()
