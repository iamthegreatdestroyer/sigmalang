# Phase 4 Feature Expansion - Integration Guide

## Quick Start Guide

### Import the Components

```python
from sigmalang.core.feature_expansion import (
    PatternObservation,
    PatternObserver,
    SemanticVectorSpace,
    ApproximateNearestNeighbor,
    Entity,
    Relation,
    KnowledgeGraph,
    EntityRelationExtractor,
)
```

## Feature 1: Learned Codebook Pattern Learning

### Basic Usage

```python
from sigmalang.core.feature_expansion import PatternObserver
from pathlib import Path

# Initialize the pattern observer
observer = PatternObserver(
    promotion_threshold=0.3,      # 30% compression benefit minimum
    min_occurrence_threshold=3,   # At least 3 occurrences
    max_patterns=128             # Keep at most 128 patterns
)

# In your encoding loop, observe patterns:
def encode_with_pattern_learning(data, encoder):
    pattern_data = extract_pattern(data)
    original_size = len(data)
    encoded = encoder.encode(data)
    encoded_size = len(encoded)

    # Observe this pattern
    observation = observer.observe_pattern(
        pattern_data=pattern_data,
        original_size=original_size,
        encoded_size=encoded_size
    )

    return encoded, observation

# Periodically check promotion candidates
def periodic_promotion_check():
    candidates = observer.get_promotion_candidates()

    if candidates:
        print(f"Ready to promote {len(candidates)} patterns")

        for pattern_obs in candidates[:10]:  # Promote top 10
            print(f"  Pattern: {pattern_obs.pattern_hash[:8]}...")
            print(f"    Occurrences: {pattern_obs.occurrence_count}")
            print(f"    Benefit: {pattern_obs.compression_benefit:.1%}")

            # In real implementation, add to learned codebook
            learned_primitive = create_primitive(pattern_obs.pattern_data)
            pattern_obs.is_learned = True
            pattern_obs.primitive_id = learned_primitive.id

# Save patterns to disk
observer.save_learned_patterns(Path("learned_patterns.json"))

# Get statistics
stats = observer.get_statistics()
print(f"Total patterns observed: {stats['total_observed']}")
print(f"Total patterns promoted: {stats['total_learned']}")
print(f"Average compression benefit: {stats['avg_compression_benefit']:.1%}")
print(f"Total compression benefit: {stats['total_compression_benefit']:.1%}")
```

### Configuration Examples

```python
# Conservative: High-quality patterns only
conservative = PatternObserver(
    promotion_threshold=0.5,    # 50% benefit
    min_occurrence_threshold=10, # 10 occurrences
    max_patterns=64
)

# Aggressive: Capture more patterns
aggressive = PatternObserver(
    promotion_threshold=0.1,    # 10% benefit
    min_occurrence_threshold=1,  # Just 1 occurrence
    max_patterns=256
)

# Balanced: Good tradeoff
balanced = PatternObserver(
    promotion_threshold=0.3,    # 30% benefit
    min_occurrence_threshold=3,  # 3 occurrences
    max_patterns=128
)
```

## Feature 2: Advanced Analogy Engine

### Basic Usage

```python
from sigmalang.core.feature_expansion import SemanticVectorSpace
import numpy as np

# Create vector space
space = SemanticVectorSpace(
    base_dim=512,      # Base semantic space
    learnable_dim=256  # Learnable dimensions
)

# Register semantic anchors for known concepts
king = np.random.randn(512)
queen = np.random.randn(512)
man = np.random.randn(512)
woman = np.random.randn(512)

space.register_semantic_anchor("king", king)
space.register_semantic_anchor("queen", queen)
space.register_semantic_anchor("man", man)
space.register_semantic_anchor("woman", woman)

# Use in analogy solving
# A:B::C:?
# king:queen::man:?

# Get anchors for analogy
A_vec = space.anchors["king"]
B_vec = space.anchors["queen"]
C_vec = space.anchors["man"]

# Compute analogy: B_vec - A_vec gives the "transformation"
# Apply to C_vec: D_vec = C_vec + (B_vec - A_vec)
transformation = B_vec - A_vec
D_vec = C_vec + transformation

# D_vec should be similar to "woman"
similarity = space.compute_learned_similarity(D_vec, space.anchors["woman"])
print(f"Analogy accuracy: {similarity:.3f}")  # Should be close to 1.0
```

### Integration with Semantic Analogy Engine

```python
from sigmalang.core.semantic_analogy_engine import SemanticAnalogyEngine
from sigmalang.core.feature_expansion import SemanticVectorSpace

# Create both
analogy_engine = SemanticAnalogyEngine()
vector_space = SemanticVectorSpace()

# Register knowledge in vector space
for concept, vector in concept_vectors.items():
    vector_space.register_semantic_anchor(concept, vector)

# Use vector space for enhanced similarity computation
# In your analogy solving:

def solve_analogy_enhanced(A, B, C, candidates):
    # Compute traditional analogy vector
    transformation = B - A
    D_candidate = C + transformation

    # Score candidates using learned similarity
    results = []
    for candidate_vec in candidates:
        score = vector_space.compute_learned_similarity(
            D_candidate,
            candidate_vec
        )
        results.append(score)

    return np.argmax(results)
```

## Feature 3: Semantic Search Capabilities

### Basic Usage

```python
from sigmalang.core.feature_expansion import ApproximateNearestNeighbor
import numpy as np

# Create ANN index
ann = ApproximateNearestNeighbor(
    num_tables=10,     # 10 LSH tables
    hash_width=32      # 32-bit hashes
)

# Add documents with their vectors
documents = {
    "doc1": np.random.randn(256),
    "doc2": np.random.randn(256),
    "doc3": np.random.randn(256),
    # ... more documents
}

for doc_id, vector in documents.items():
    ann.add(doc_id, vector)

# Search for similar documents
query_vector = np.random.randn(256)
top_k_results = ann.search(query_vector, k=5)

print("Top 5 most similar documents:")
for doc_id, similarity in top_k_results:
    print(f"  {doc_id}: similarity={similarity:.3f}")
```

### Integration with Document Processing

```python
from sigmalang.core.feature_expansion import ApproximateNearestNeighbor
from sigmalang.core.encoder import SigmaEncoder

# Create components
encoder = SigmaEncoder()
ann = ApproximateNearestNeighbor(num_tables=10)

# Build semantic search index
documents = load_documents()

for doc_id, text in documents.items():
    # Encode document to vector
    vector = encoder.encode_to_vector(text)

    # Add to search index
    ann.add(doc_id, vector)

# User queries
def search_documents(query_text, k=10):
    # Encode query
    query_vector = encoder.encode_to_vector(query_text)

    # Find similar documents
    results = ann.search(query_vector, k=k)

    # Return documents with similarity scores
    return [
        (documents[doc_id], similarity)
        for doc_id, similarity in results
    ]

# Example
query = "machine learning algorithms"
for doc, similarity in search_documents(query):
    print(f"Similarity: {similarity:.2%}")
    print(f"Document: {doc[:100]}...")
    print()
```

### Performance Tuning

```python
# For speed (lower accuracy):
fast_ann = ApproximateNearestNeighbor(
    num_tables=3,      # Fewer tables = faster but less accurate
    hash_width=16      # Fewer bits = faster but less accurate
)

# For accuracy (slower):
accurate_ann = ApproximateNearestNeighbor(
    num_tables=20,     # More tables = slower but more accurate
    hash_width=64      # More bits = slower but more accurate
)

# Balanced:
balanced_ann = ApproximateNearestNeighbor(
    num_tables=10,
    hash_width=32
)
```

## Feature 4: Entity/Relation Extraction

### Basic Usage

```python
from sigmalang.core.feature_expansion import EntityRelationExtractor

# Create extractor
extractor = EntityRelationExtractor()

# Extract entities and build knowledge graph
text = """
Steve Jobs founded Apple in Cupertino, California.
Tim Cook is the CEO of Apple.
Apple makes iPhones and MacBooks.
"""

# Build knowledge graph
kg = extractor.build_knowledge_graph(text)

# Query the knowledge graph
print("Entities found:")
for entity_id, entity in kg.entities.items():
    print(f"  {entity.text} ({entity.entity_type}) - confidence: {entity.confidence:.2f}")

print("\nRelations found:")
for relation in kg.relations:
    print(f"  {relation.source_entity.text} --[{relation.relation_type}]--> {relation.target_entity.text}")

# Get statistics
stats = kg.get_statistics()
print(f"\nGraph Statistics:")
print(f"  Total entities: {stats['total_entities']}")
print(f"  Total relations: {stats['total_relations']}")
print(f"  Entity types: {stats['entity_types']}")
print(f"  Relation types: {stats['relation_types']}")

# Export to JSON (Neo4j compatible)
graph_json = kg.export_json()
print(f"\nGraph JSON:")
print(json.dumps(graph_json, indent=2))
```

### Integration with Document Processing

```python
from sigmalang.core.feature_expansion import EntityRelationExtractor
import json

# Create extractor
extractor = EntityRelationExtractor()

# Process multiple documents
documents = load_documents()
master_kg = None

for doc_id, text in documents.items():
    # Extract knowledge from document
    kg = extractor.build_knowledge_graph(text)

    # Merge into master graph (simple version)
    if master_kg is None:
        master_kg = kg
    else:
        # Add entities and relations to master graph
        for entity in kg.entities.values():
            master_kg.add_entity(entity)
        for relation in kg.relations:
            master_kg.add_relation(relation)

# Export combined knowledge graph
graph_data = master_kg.export_json()

# Save to file
with open("knowledge_graph.json", "w") as f:
    json.dump(graph_data, f, indent=2)

# Import into Neo4j
# LOAD JSON INTO NEO4J WITH CYPHER:
# CALL apoc.load.json("knowledge_graph.json") YIELD value
# ...
```

### Pattern Customization

```python
# You can extend EntityRelationExtractor with custom patterns
from sigmalang.core.feature_expansion import EntityRelationExtractor

class CustomExtractor(EntityRelationExtractor):
    def extract_entities(self, text):
        # Call parent implementation
        entities = super().extract_entities(text)

        # Add custom pattern matching
        # e.g., EMAIL, PHONE, PRODUCT_NAME
        import re

        # Extract emails
        for match in re.finditer(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            email = match.group()
            entity = Entity(
                text=email,
                entity_type="EMAIL",
                start_position=match.start(),
                end_position=match.end(),
                confidence=0.95
            )
            entities.append(entity)

        return entities

# Use custom extractor
custom_extractor = CustomExtractor()
kg = custom_extractor.build_knowledge_graph(text)
```

## Integration Examples

### Example 1: End-to-End Pipeline with Pattern Learning

```python
from sigmalang.core.encoder import SigmaEncoder
from sigmalang.core.feature_expansion import PatternObserver
import numpy as np

# Setup
encoder = SigmaEncoder()
observer = PatternObserver(
    promotion_threshold=0.3,
    min_occurrence_threshold=3
)

# Process data with pattern learning
def process_with_learning(data_stream):
    for batch in data_stream:
        for item in batch:
            # Encode
            original_size = len(item)
            encoded = encoder.encode(item)
            encoded_size = len(encoded)

            # Extract pattern
            pattern = extract_semantic_pattern(item)

            # Observe pattern
            observation = observer.observe_pattern(
                pattern_data=pattern,
                original_size=original_size,
                encoded_size=encoded_size
            )

# Periodic maintenance
def maintenance_cycle():
    # Check for promotions
    candidates = observer.get_promotion_candidates()
    for pattern_obs in candidates[:5]:
        promote_to_codebook(pattern_obs)
        pattern_obs.is_learned = True

    # Save state
    observer.save_learned_patterns(Path("patterns.json"))

    # Report stats
    stats = observer.get_statistics()
    print(f"Patterns learned so far: {stats['total_learned']}")
```

### Example 2: Semantic Search with Learned Patterns

```python
from sigmalang.core.feature_expansion import (
    ApproximateNearestNeighbor,
    PatternObserver
)

# Create components
ann = ApproximateNearestNeighbor(num_tables=10)
observer = PatternObserver()

# Build index with pattern learning
def build_semantic_index(documents):
    for doc_id, doc_text in documents.items():
        # Generate vector (could use encoder or embeddings)
        vector = generate_semantic_vector(doc_text)

        # Add to search index
        ann.add(doc_id, vector)

        # Observe patterns in document
        pattern = extract_pattern(doc_text)
        observer.observe_pattern(pattern, len(doc_text), len(vector))

# Search with learned patterns
def intelligent_search(query, learned_patterns=None):
    # Enhance query with learned patterns
    if learned_patterns:
        enhanced_query = apply_learned_patterns(query, learned_patterns)
    else:
        enhanced_query = query

    # Generate vector
    query_vector = generate_semantic_vector(enhanced_query)

    # Search
    results = ann.search(query_vector, k=10)

    return results
```

### Example 3: Knowledge Graph Construction & Analysis

```python
from sigmalang.core.feature_expansion import EntityRelationExtractor
import json

# Extract knowledge from corpus
extractor = EntityRelationExtractor()
corpus = load_corpus()

# Build integrated knowledge graph
kg = None
for doc_text in corpus:
    doc_kg = extractor.build_knowledge_graph(doc_text)

    if kg is None:
        kg = doc_kg
    else:
        # Merge graphs (simple union)
        for entity in doc_kg.entities.values():
            kg.add_entity(entity)
        for relation in doc_kg.relations:
            kg.add_relation(relation)

# Analyze graph
def analyze_knowledge_graph(kg):
    stats = kg.get_statistics()

    # Most common entity types
    print("Entity type distribution:")
    for etype, count in stats['entity_types'].items():
        print(f"  {etype}: {count}")

    # Most common relation types
    print("\nRelation type distribution:")
    for rtype, count in stats['relation_types'].items():
        print(f"  {rtype}: {count}")

    # Most connected entities
    entity_degrees = {}
    for entity_id in kg.entities:
        relations = kg.get_entity_relations(entity_id)
        entity_degrees[entity_id] = len(relations)

    top_entities = sorted(
        entity_degrees.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print("\nMost connected entities:")
    for entity_id, degree in top_entities:
        entity = kg.entities[entity_id]
        print(f"  {entity.text}: {degree} relations")

analyze_knowledge_graph(kg)

# Export for external tools
graph_json = kg.export_json()
with open("knowledge_graph.json", "w") as f:
    json.dump(graph_json, f, indent=2)
```

## Testing Your Integrations

```python
import pytest
from sigmalang.core.feature_expansion import (
    PatternObserver,
    ApproximateNearestNeighbor,
    EntityRelationExtractor
)

def test_pattern_learning_integration():
    """Test pattern learning in encoding pipeline"""
    observer = PatternObserver()

    # Simulate encoding
    for i in range(5):
        observer.observe_pattern(
            {"type": "test", "id": i},
            original_size=100,
            encoded_size=60
        )

    stats = observer.get_statistics()
    assert stats['total_observed'] == 5

def test_semantic_search_integration():
    """Test semantic search with multiple documents"""
    import numpy as np

    ann = ApproximateNearestNeighbor(num_tables=10)

    # Add vectors
    vectors = {
        f"doc{i}": np.random.randn(256)
        for i in range(100)
    }

    for doc_id, vector in vectors.items():
        ann.add(doc_id, vector)

    # Search
    query = np.random.randn(256)
    results = ann.search(query, k=10)

    assert len(results) <= 10

def test_entity_extraction_integration():
    """Test entity extraction and graph building"""
    extractor = EntityRelationExtractor()

    text = "Steve Jobs founded Apple."
    kg = extractor.build_knowledge_graph(text)

    assert len(kg.entities) > 0
    assert len(kg.relations) >= 0
```

## Best Practices

1. **Pattern Learning**: Check promotion candidates regularly (e.g., every N documents)
2. **Vector Search**: Use appropriate hash table count based on your collection size
3. **Entity Extraction**: Extend patterns for domain-specific entity types
4. **Knowledge Graphs**: Merge from multiple sources carefully (handle deduplication)
5. **Performance**: Monitor memory usage, especially for large pattern/vector sets

## Troubleshooting

### PatternObserver not promoting patterns?

- Check `promotion_threshold` is reasonable (0.3 is good default)
- Verify `min_occurrence_threshold` is being met
- Ensure `max_patterns` isn't limiting growth

### ANN search not finding similar documents?

- Increase `num_tables` for better recall
- Check vector quality and dimensionality
- Verify `hash_width` (32 bits is good default)

### Entity extraction missing entities?

- Extend EntityRelationExtractor with custom patterns
- Check confidence thresholds
- Consider domain-specific entity types

### Knowledge graph too large?

- Implement graph pruning (low-confidence edges)
- Filter by confidence threshold
- Use graph sampling for analysis
