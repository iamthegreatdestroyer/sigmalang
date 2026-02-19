# Semantic Primitives

## Primitive Types

### Existential Primitives (Tier 0)

These fundamental semantic concepts form the foundation of all ΣLANG encoding:

#### ENTITY (Σ₀₀₀)
Represents concrete or abstract things.
- Examples: "apple", "intelligence", "company"
- Attributes: Name, type, category
- Relations: Is-a, part-of, instance-of

#### ACTION (Σ₀₀₁)
Represents processes, events, or verbs.
- Examples: "create", "analyze", "transform"
- Attributes: Tense, aspect, voice
- Relations: Agent, patient, instrument

#### RELATION (Σ₀₀₂)
Represents connections and relationships.
- Examples: "is-a", "has", "located-in"
- Types: Semantic, syntactic, structural

#### ATTRIBUTE (Σ₀₀₃)
Represents properties and qualities.
- Examples: "red", "large", "efficient"
- Values: Can be scalar or symbolic

#### QUANTITY (Σ₀₀₄)
Represents numbers and measures.
- Examples: "10", "3.14", "infinite"
- Units: Can include measurement units

#### TEMPORAL (Σ₀₀₅)
Represents time-related concepts.
- Examples: "yesterday", "2026", "next week"
- Resolution: Second to century

#### SPATIAL (Σ₀₀₆)
Represents location and geometry.
- Examples: "north", "above", "inside"
- Dimensions: 1D (line), 2D (plane), 3D (space)

#### CAUSAL (Σ₀₀₇)
Represents causality and dependencies.
- Examples: "because", "causes", "requires"
- Strength: Weak, moderate, strong

### Domain Primitives (Tier 1)

Specialized encodings for specific domains:

#### Code Domain (Σ₀₁₆ - Σ₀₂₄)
- FUNCTION, VARIABLE, CLASS, LOOP, CONDITION
- IMPORT, EXCEPTION, DECORATOR, COMMENT

#### Math Domain (Σ₀₂₅ - Σ₀₃₂)
- EQUATION, MATRIX, VECTOR, OPERATION
- FUNCTION_MATH, LIMIT, DERIVATIVE, INTEGRAL

#### Logic Domain (Σ₀₃₃ - Σ₀₃₉)
- BOOLEAN, AND, OR, NOT
- IMPLIES, EQUIVALENT, EXISTS, FORALL

#### Communication Domain (Σ₀₄₀ - Σ₀₄₇)
- STATEMENT, QUESTION, COMMAND, REQUEST
- EMOTION, SENTIMENT, TONE, FORMALITY

#### Data Structure Domain (Σ₀₄₈ - Σ₀₅₅)
- ARRAY, MAP, GRAPH, TREE
- STACK, QUEUE, LINKED_LIST, SET

### Learned Primitives (Tier 2)

Dynamically allocated based on patterns observed in specific domains:

- Allocated as needed up to Σ₂₅₅
- Trained from domain-specific corpora
- Optimized for common use patterns

## Primitive Composition

Primitives combine to form complex semantic structures:

```python
from sigmalang.core.primitives import SemanticNode, ExistentialPrimitive

# Create semantic nodes
subject = SemanticNode(primitive=ExistentialPrimitive.ENTITY, value="Alice")
action = SemanticNode(primitive=ExistentialPrimitive.ACTION, value="creates")
obj = SemanticNode(primitive=ExistentialPrimitive.ENTITY, value="algorithm")

# Link relationships
action.add_child(subject)
action.add_child(obj)

# Encode the structure
encoded = encoder.encode(action)
```

## Working with Primitives in Python

### Create a Primitive

```python
from sigmalang.core.primitives import ExistentialPrimitive, SemanticNode

node = SemanticNode(
    primitive=ExistentialPrimitive.ENTITY,
    value="machine learning",
    confidence=0.95
)
```

### Query Primitive Information

```python
# Get primitive type
print(node.primitive)  # ExistentialPrimitive.ENTITY

# Get primitive name
print(node.primitive.name)  # "ENTITY"

# Get primitive code
print(node.primitive.value)  # 0
```

### Build Semantic Structures

```python
# Entity with attributes
company = SemanticNode(primitive=ExistentialPrimitive.ENTITY, value="Google")
company.add_child(
    SemanticNode(primitive=ExistentialPrimitive.ATTRIBUTE, value="technology")
)
company.add_child(
    SemanticNode(primitive=ExistentialPrimitive.LOCATION, value="Mountain View")
)
```

## Primitive Encoding

Each primitive is encoded as 1-2 bytes depending on tier:

| Tier | Range | Encoding | Size |
|------|-------|----------|------|
| Tier 0 (Existential) | Σ₀₀₀-Σ₀₁₅ | Direct | 1 byte |
| Tier 1 (Domain) | Σ₀₁₆-Σ₁₂₇ | Prefix + ID | 1-2 bytes |
| Tier 2 (Learned) | Σ₁₂₈-Σ₂₅₅ | Prefix + ID | 1-2 bytes |

## Best Practices

1. **Use appropriate primitives**: Choose the most specific primitive available
2. **Minimize redundancy**: Avoid repeating primitive information
3. **Structure relationships**: Use hierarchies to represent connections
4. **Leverage domain primitives**: Use Tier 1 primitives when available for your domain
5. **Monitor learned primitives**: Track Tier 2 usage patterns

## Next Steps

- Learn about [Compression Techniques](compression.md)
- Explore [Analogy Engine](analogy.md)
- See [API Reference](../api/python-api.md) for implementation details
