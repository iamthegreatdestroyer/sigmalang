"""
ΣLANG Primitives System
=======================

Defines the 256 root glyphs (Σ₀ - Σ₂₅₅) that form the atomic units
of semantic encoding. These are not words - they are meaning atoms.

Tier 0 (Σ₀₀₀ - Σ₀₁₅): Existential Primitives - Universal concepts
Tier 1 (Σ₀₁₆ - Σ₁₂₇): Domain Primitives - Specialized encodings
Tier 2 (Σ₁₂₈ - Σ₂₅₅): Learned Primitives - Dynamically allocated

Copyright 2025 - Ryot LLM Project
"""

from enum import IntEnum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np


class GlyphType(IntEnum):
    """Classification of glyph encoding types."""
    PRIMITIVE = 0b00      # Direct Σ-primitive
    REFERENCE = 0b01      # Pointer to Σ-Bank
    DELTA = 0b10          # Context-relative encoding
    COMPOSITE = 0b11      # Aggregated structure


class PrimitiveTier(IntEnum):
    """Tier classification for primitives."""
    EXISTENTIAL = 0       # Σ₀₀₀ - Σ₀₁₅: Universal concepts
    DOMAIN = 1            # Σ₀₁₆ - Σ₁₂₇: Specialized encodings  
    LEARNED = 2           # Σ₁₂₈ - Σ₂₅₅: Dynamic allocation


# ============================================================================
# TIER 0: EXISTENTIAL PRIMITIVES (Σ₀₀₀ - Σ₀₁₅)
# These encode the fundamental categories of meaning
# ============================================================================

class ExistentialPrimitive(IntEnum):
    """Universal semantic categories - the atoms of meaning."""
    ENTITY = 0x00         # Something that exists
    ACTION = 0x01         # State change or process
    RELATION = 0x02       # Connection between entities
    ATTRIBUTE = 0x03      # Property of an entity
    QUANTITY = 0x04       # Numeric or comparative value
    TEMPORAL = 0x05       # Time relationship
    SPATIAL = 0x06        # Space relationship
    CAUSAL = 0x07         # Cause-effect link
    MODAL = 0x08          # Possibility, necessity, desire
    NEGATION = 0x09       # Absence or opposition
    REFERENCE = 0x0A      # Pointer to existing structure
    COMPOSITE = 0x0B      # Aggregation marker
    TRANSFORM = 0x0C      # Modification operation
    CONDITION = 0x0D      # If-then structure
    ITERATION = 0x0E      # Repetition pattern
    ABSTRACT = 0x0F       # Concept without physical form


# ============================================================================
# TIER 1: DOMAIN PRIMITIVES (Σ₀₁₆ - Σ₁₂₇)
# Specialized encodings for common domains
# ============================================================================

class CodePrimitive(IntEnum):
    """Programming and code structure primitives."""
    FUNCTION = 0x10       # Callable unit
    VARIABLE = 0x11       # Named storage
    CLASS = 0x12          # Object template
    LOOP = 0x13           # Iteration construct
    BRANCH = 0x14         # Conditional flow
    RETURN = 0x15         # Value emission
    IMPORT = 0x16         # External reference
    PARAMETER = 0x17      # Input binding
    TYPE = 0x18           # Data classification
    OPERATOR = 0x19       # Computation symbol
    LITERAL = 0x1A        # Constant value
    EXPRESSION = 0x1B     # Computed value
    STATEMENT = 0x1C      # Execution unit
    BLOCK = 0x1D          # Scope container
    EXCEPTION = 0x1E      # Error handling
    ASYNC = 0x1F          # Concurrent execution


class MathPrimitive(IntEnum):
    """Mathematical operation primitives."""
    ADD = 0x20            # Addition/sum
    SUBTRACT = 0x21       # Subtraction/difference
    MULTIPLY = 0x22       # Multiplication/product
    DIVIDE = 0x23         # Division/quotient
    POWER = 0x24          # Exponentiation
    ROOT = 0x25           # Nth root
    LOG = 0x26            # Logarithm
    DERIVATIVE = 0x27     # Rate of change
    INTEGRAL = 0x28       # Accumulation
    LIMIT = 0x29          # Approach value
    SUMMATION = 0x2A      # Series sum
    PRODUCT_OP = 0x2B     # Series product
    MATRIX = 0x2C         # 2D array structure
    VECTOR = 0x2D         # 1D array structure
    TENSOR = 0x2E         # ND array structure
    SET_OP = 0x2F         # Set operation


class LogicPrimitive(IntEnum):
    """Logical relation primitives."""
    AND = 0x30            # Conjunction
    OR = 0x31             # Disjunction
    NOT = 0x32            # Negation
    IMPLIES = 0x33        # Implication
    EQUIVALENT = 0x34     # Bi-conditional
    FORALL = 0x35         # Universal quantifier
    EXISTS = 0x36         # Existential quantifier
    EQUALS = 0x37         # Identity
    GREATER = 0x38        # Comparison >
    LESS = 0x39           # Comparison <
    MEMBER = 0x3A         # Set membership
    SUBSET = 0x3B         # Set inclusion
    UNION = 0x3C          # Set union
    INTERSECT = 0x3D      # Set intersection
    COMPLEMENT = 0x3E     # Set complement
    XOR = 0x3F            # Exclusive or


class EntityPrimitive(IntEnum):
    """Physical and abstract entity primitives."""
    OBJECT = 0x40         # Physical thing
    AGENT = 0x41          # Acting entity
    LOCATION = 0x42       # Place/position
    INSTRUMENT = 0x43     # Tool/means
    MATERIAL = 0x44       # Substance
    CONTAINER = 0x45      # Holding structure
    PATH = 0x46           # Route/trajectory
    BOUNDARY = 0x47       # Edge/limit
    GROUP = 0x48          # Collection
    PART = 0x49           # Component
    WHOLE = 0x4A          # Complete entity
    INSTANCE = 0x4B       # Specific occurrence
    CATEGORY = 0x4C       # Classification
    PROPERTY = 0x4D       # Characteristic
    STATE = 0x4E          # Condition
    EVENT = 0x4F          # Occurrence


class ActionPrimitive(IntEnum):
    """Action and process primitives."""
    CREATE = 0x50         # Bring into existence
    DESTROY = 0x51        # Remove from existence
    MODIFY = 0x52         # Change properties
    MOVE = 0x53           # Change location
    TRANSFER = 0x54       # Change ownership
    COMBINE = 0x55        # Join together
    SEPARATE = 0x56       # Split apart
    COMPARE = 0x57        # Evaluate difference
    SEARCH = 0x58         # Look for
    SELECT = 0x59         # Choose from options
    SORT = 0x5A           # Arrange in order
    FILTER = 0x5B         # Remove by criteria
    MAP = 0x5C            # Transform each
    REDUCE = 0x5D         # Combine to one
    VALIDATE = 0x5E       # Check correctness
    EXECUTE = 0x5F        # Perform action


class CommunicationPrimitive(IntEnum):
    """Communication and information primitives."""
    QUERY = 0x60          # Ask/request info
    ASSERT = 0x61         # State as true
    COMMAND = 0x62        # Direct to act
    REQUEST = 0x63        # Ask for action
    EXPLAIN = 0x64        # Provide understanding
    DESCRIBE = 0x65       # Detail properties
    DEFINE = 0x66         # Establish meaning
    EXAMPLE = 0x67        # Illustrative instance
    REASON = 0x68         # Logical basis
    CONCLUSION = 0x69     # Derived result
    ASSUMPTION = 0x6A     # Taken as true
    HYPOTHESIS = 0x6B     # Proposed explanation
    EVIDENCE = 0x6C       # Supporting data
    ARGUMENT = 0x6D       # Logical structure
    CONTEXT = 0x6E        # Surrounding info
    REFERENCE_COM = 0x6F  # Point to source


class StructurePrimitive(IntEnum):
    """Data structure primitives."""
    LIST = 0x70           # Ordered sequence
    DICT = 0x71           # Key-value mapping
    TREE = 0x72           # Hierarchical structure
    GRAPH = 0x73          # Node-edge network
    STACK = 0x74          # LIFO structure
    QUEUE = 0x75          # FIFO structure
    HEAP = 0x76           # Priority structure
    ARRAY = 0x77          # Indexed collection
    STRING = 0x78         # Character sequence
    NUMBER = 0x79         # Numeric value
    BOOLEAN = 0x7A        # True/false value
    NULL = 0x7B           # Absence of value
    POINTER = 0x7C        # Memory reference
    BUFFER = 0x7D         # Temporary storage
    STREAM = 0x7E         # Data flow
    RECORD = 0x7F         # Structured data


# Reserved: 0x80 - 0x7F for future domain primitives


# ============================================================================
# TIER 2: LEARNED PRIMITIVES (Σ₁₂₈ - Σ₂₅₅)
# Dynamically allocated based on user patterns
# ============================================================================

LEARNED_PRIMITIVE_START = 0x80
LEARNED_PRIMITIVE_END = 0xFF
LEARNED_PRIMITIVE_COUNT = LEARNED_PRIMITIVE_END - LEARNED_PRIMITIVE_START + 1


@dataclass
class LearnedPrimitive:
    """A dynamically learned semantic pattern."""
    id: int                           # Σ₁₂₈ - Σ₂₅₅
    pattern_signature: bytes          # Unique pattern identifier
    expansion_template: str           # Human-readable template
    semantic_embedding: np.ndarray    # Vector representation
    frequency: int = 0                # Usage count
    compression_ratio: float = 1.0    # Achieved compression
    created_at: float = 0.0           # Timestamp
    last_used: float = 0.0            # Last access time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'pattern_signature': self.pattern_signature.hex(),
            'expansion_template': self.expansion_template,
            'semantic_embedding': self.semantic_embedding.tolist(),
            'frequency': self.frequency,
            'compression_ratio': self.compression_ratio,
            'created_at': self.created_at,
            'last_used': self.last_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearnedPrimitive':
        return cls(
            id=data['id'],
            pattern_signature=bytes.fromhex(data['pattern_signature']),
            expansion_template=data['expansion_template'],
            semantic_embedding=np.array(data['semantic_embedding']),
            frequency=data['frequency'],
            compression_ratio=data['compression_ratio'],
            created_at=data['created_at'],
            last_used=data['last_used']
        )


# ============================================================================
# SEMANTIC TREE STRUCTURES
# ============================================================================

@dataclass
class SemanticNode:
    """A node in the semantic tree representation."""
    primitive: int                    # Σ-primitive ID
    value: Optional[Any] = None       # Associated value (if any)
    children: List['SemanticNode'] = field(default_factory=list)
    modifiers: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.primitive, str(self.value), len(self.children)))
    
    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def node_count(self) -> int:
        return 1 + sum(child.node_count() for child in self.children)
    
    def primitives_used(self) -> Set[int]:
        result = {self.primitive}
        for child in self.children:
            result.update(child.primitives_used())
        return result


@dataclass  
class SemanticTree:
    """Complete semantic representation of an input."""
    root: SemanticNode
    source_text: str                  # Original human input
    embedding: Optional[np.ndarray] = None  # Vector representation
    sigma_hash: Optional[int] = None  # Computed Σ-hash
    
    def serialize(self) -> bytes:
        """Serialize tree to bytes for storage."""
        import pickle
        return pickle.dumps(self)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'SemanticTree':
        """Deserialize tree from bytes."""
        import pickle
        return pickle.loads(data)
    
    @property
    def depth(self) -> int:
        return self.root.depth()
    
    @property
    def node_count(self) -> int:
        return self.root.node_count()
    
    @property
    def primitives_used(self) -> Set[int]:
        return self.root.primitives_used()


# ============================================================================
# PRIMITIVE REGISTRY
# ============================================================================

class PrimitiveRegistry:
    """
    Central registry for all Σ-primitives.
    Provides lookup, validation, and metadata access.
    """
    
    def __init__(self):
        self._primitives: Dict[int, Dict[str, Any]] = {}
        self._name_to_id: Dict[str, int] = {}
        self._initialize_primitives()
    
    def _initialize_primitives(self):
        """Register all built-in primitives."""
        # Tier 0: Existential
        for p in ExistentialPrimitive:
            self._register(p.value, p.name, PrimitiveTier.EXISTENTIAL)
        
        # Tier 1: Domain primitives
        for enum_class in [CodePrimitive, MathPrimitive, LogicPrimitive,
                          EntityPrimitive, ActionPrimitive, 
                          CommunicationPrimitive, StructurePrimitive]:
            for p in enum_class:
                self._register(p.value, p.name, PrimitiveTier.DOMAIN)
    
    def _register(self, id: int, name: str, tier: PrimitiveTier):
        """Register a primitive."""
        self._primitives[id] = {
            'name': name,
            'tier': tier,
            'id': id
        }
        self._name_to_id[name.upper()] = id
    
    def get_name(self, id: int) -> str:
        """Get primitive name by ID."""
        if id in self._primitives:
            return self._primitives[id]['name']
        if LEARNED_PRIMITIVE_START <= id <= LEARNED_PRIMITIVE_END:
            return f"LEARNED_{id - LEARNED_PRIMITIVE_START}"
        return f"UNKNOWN_{id}"
    
    def get_id(self, name: str) -> Optional[int]:
        """Get primitive ID by name."""
        return self._name_to_id.get(name.upper())
    
    def get_tier(self, id: int) -> PrimitiveTier:
        """Get primitive tier."""
        if id in self._primitives:
            return self._primitives[id]['tier']
        if LEARNED_PRIMITIVE_START <= id <= LEARNED_PRIMITIVE_END:
            return PrimitiveTier.LEARNED
        raise ValueError(f"Unknown primitive ID: {id}")
    
    def is_valid(self, id: int) -> bool:
        """Check if primitive ID is valid."""
        return id in self._primitives or \
               LEARNED_PRIMITIVE_START <= id <= LEARNED_PRIMITIVE_END
    
    def all_primitives(self) -> List[Dict[str, Any]]:
        """Get all registered primitives."""
        return list(self._primitives.values())


# Global registry instance
PRIMITIVE_REGISTRY = PrimitiveRegistry()


# ============================================================================
# GLYPH STRUCTURES
# ============================================================================

@dataclass
class Glyph:
    """
    A single ΣLANG glyph - the atomic unit of encoding.
    
    Binary format (v2 with payload flag):
    [2 bits] TYPE - GlyphType enum
    [1 bit]  HAS_PAYLOAD - 1 if payload follows, 0 otherwise
    [5 bits] PRIMITIVE_ID - Σ-primitive (0-30 inline, 31 = escape for extended)
    [optional 8 bits] EXTENDED_ID - present if PRIMITIVE_ID == 31
    [optional variable] PAYLOAD - length-prefixed content if HAS_PAYLOAD == 1
    """
    glyph_type: GlyphType
    primitive_id: int
    payload: Optional[bytes] = None
    
    def to_bytes(self) -> bytes:
        """Encode glyph to binary format with explicit payload flag."""
        has_payload = 1 if self.payload is not None else 0
        
        if self.primitive_id > 30:
            # Extended format: use 0x1F (31) as escape code, then full primitive_id
            first_byte = (self.glyph_type << 6) | (has_payload << 5) | 0x1F
            result = bytes([first_byte, self.primitive_id])
        else:
            # Inline format: type (2 bits) + has_payload (1 bit) + primitive_id (5 bits)
            first_byte = (self.glyph_type << 6) | (has_payload << 5) | self.primitive_id
            result = bytes([first_byte])
        
        # Add payload if present (including empty payload b'')
        if self.payload is not None:
            # Length prefix (1-2 bytes depending on size)
            if len(self.payload) < 128:
                result += bytes([len(self.payload)])
            else:
                result += bytes([0x80 | (len(self.payload) >> 8), 
                                len(self.payload) & 0xFF])
            result += self.payload
        
        return result
    
    @classmethod
    def from_bytes(cls, data: bytes) -> Tuple['Glyph', int]:
        """
        Decode glyph from binary format v2.
        
        Format v2 uses explicit has_payload flag:
        - Bits 7-6: glyph_type (0-3)
        - Bit 5: has_payload flag (1 = payload present, 0 = no payload)
        - Bits 4-0: primitive_id (0-30 inline, 31 = extended format)
        
        Extended format (when primitive_id bits = 0x1F):
        - Next byte contains actual primitive_id (for IDs 31-255)
        
        Returns (glyph, bytes_consumed).
        """
        first_byte = data[0]
        glyph_type = GlyphType((first_byte >> 6) & 0x03)
        has_payload = (first_byte >> 5) & 0x01
        primitive_id = first_byte & 0x1F  # Only 5 bits now
        offset = 1
        
        # Check for extended primitive ID (0x1F is escape code)
        if primitive_id == 0x1F and len(data) > offset:
            primitive_id = data[offset]
            offset += 1
        
        # Only read payload if has_payload flag is set
        payload = None
        if has_payload and offset < len(data):
            length_byte = data[offset]
            offset += 1
            if length_byte & 0x80:
                # Two-byte length
                length = ((length_byte & 0x7F) << 8) | data[offset]
                offset += 1
            else:
                length = length_byte
            
            if offset + length <= len(data):
                payload = data[offset:offset + length]  # Works for length=0 too (empty payload)
                offset += length
        
        return cls(glyph_type, primitive_id, payload), offset


@dataclass
class GlyphStream:
    """A sequence of glyphs forming a complete encoding."""
    glyphs: List[Glyph]
    context_id: int = 0
    version: int = 1
    flags: int = 0
    
    def to_bytes(self) -> bytes:
        """Encode complete glyph stream with header."""
        # Header: 4 bytes
        # [2 bits] version, [6 bits] flags, [12 bits] context_id, [12 bits] length
        header = (
            ((self.version & 0x03) << 30) |
            ((self.flags & 0x3F) << 24) |
            ((self.context_id & 0xFFF) << 12) |
            (len(self.glyphs) & 0xFFF)
        )
        result = header.to_bytes(4, 'big')
        
        # Glyph data
        for glyph in self.glyphs:
            result += glyph.to_bytes()
        
        # CRC-16 checksum
        crc = self._compute_crc16(result)
        result += crc.to_bytes(2, 'big')
        
        return result
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'GlyphStream':
        """Decode glyph stream from binary."""
        # Parse header
        header = int.from_bytes(data[:4], 'big')
        version = (header >> 30) & 0x03
        flags = (header >> 24) & 0x3F
        context_id = (header >> 12) & 0xFFF
        num_glyphs = header & 0xFFF
        
        # Parse glyphs
        offset = 4
        glyphs = []
        for _ in range(num_glyphs):
            glyph, consumed = Glyph.from_bytes(data[offset:])
            glyphs.append(glyph)
            offset += consumed
        
        # Verify checksum
        expected_crc = int.from_bytes(data[-2:], 'big')
        actual_crc = cls._compute_crc16_static(data[:-2])
        if expected_crc != actual_crc:
            raise ValueError("CRC checksum mismatch")
        
        return cls(glyphs, context_id, version, flags)
    
    def _compute_crc16(self, data: bytes) -> int:
        return self._compute_crc16_static(data)
    
    @staticmethod
    def _compute_crc16_static(data: bytes) -> int:
        """Compute CRC-16-CCITT checksum."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte << 8
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ 0x1021
                else:
                    crc <<= 1
                crc &= 0xFFFF
        return crc
