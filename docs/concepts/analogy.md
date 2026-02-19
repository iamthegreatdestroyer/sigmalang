# Analogy Engine

## Overview

The ΣLANG Analogy Engine solves semantic analogies using primitive relationships and vector operations. It enables solving word analogies like:

```
king : queen :: man : ?
Answer: woman (confidence: 0.94)
```

## How It Works

### Vector Space Representation

Each term is represented as a semantic vector in primitive space:

```
king = [ENTITY:ruler, GENDER:male, STATUS:royal, ...]
queen = [ENTITY:ruler, GENDER:female, STATUS:royal, ...]
man = [ENTITY:person, GENDER:male, STATUS:common, ...]
woman = [ENTITY:person, GENDER:female, STATUS:common, ...]
```

### Relation Extraction

The analogy engine extracts semantic relations:

```
Relation(king, queen) = queen - king
                       = [GENDER:female] - [GENDER:male]
                       = [DELTA_GENDER: +female]

Relation(man, ?) = same relation
man + [DELTA_GENDER: +female] = woman
```

## Using the Analogy Engine

### CLI Interface

```bash
# Simple analogy
sigmalang analogy --word1 king --word2 queen --word3 man

# Output
# Answer: woman (confidence: 0.94)

# Multiple candidates
sigmalang analogy --word1 king --word2 queen --word3 man --top 5

# Output
# 1. woman (0.94)
# 2. lady (0.89)
# 3. female (0.82)
# 4. matriarch (0.75)
# 5. princess (0.68)
```

### Python API

```python
from sigmalang.core.analogy import AnalogyEngine

engine = AnalogyEngine()

# Solve single analogy
answer = engine.solve(
    word1="king",
    word2="queen",
    word3="man"
)
print(f"Answer: {answer.word} (confidence: {answer.confidence})")

# Get multiple candidates
candidates = engine.solve_top_k(
    word1="king",
    word2="queen",
    word3="man",
    k=5
)
for candidate in candidates:
    print(f"{candidate.word}: {candidate.confidence:.2f}")
```

### REST API

```bash
curl -X POST http://localhost:26080/api/analogy \
  -H "Content-Type: application/json" \
  -d '{
    "word1": "king",
    "word2": "queen",
    "word3": "man"
  }'
```

Response:
```json
{
  "word1": "king",
  "word2": "queen",
  "word3": "man",
  "answer": "woman",
  "confidence": 0.94,
  "top_candidates": [
    {"word": "woman", "confidence": 0.94},
    {"word": "lady", "confidence": 0.89},
    {"word": "female", "confidence": 0.82}
  ]
}
```

## Advanced Usage

### Custom Similarity Metrics

```python
engine = AnalogyEngine(
    similarity_metric="cosine",  # or "euclidean", "manhattan"
    relation_weight=0.7
)
```

### Batch Processing

```python
analogies = [
    ("king", "queen", "man"),
    ("doctor", "nurse", "male"),
    ("good", "bad", "happy"),
]

results = engine.solve_batch(analogies)
for analogy, result in zip(analogies, results):
    print(f"{analogy} → {result.word} ({result.confidence:.2f})")
```

### Filtering Results

```python
# Find analogies for a specific category
results = engine.solve_with_filter(
    word1="king",
    word2="queen",
    word3="man",
    category_filter="profession"  # Only return profession words
)
```

## Example Analogies

### Gender Relations
```
king : queen :: man : woman (0.94)
prince : princess :: actor : actress (0.91)
husband : wife :: groom : bride (0.93)
```

### Role Relations
```
teacher : student :: employer : employee (0.88)
doctor : patient :: lawyer : client (0.85)
master : apprentice :: mentor : mentee (0.82)
```

### Antonyms
```
good : bad :: hot : cold (0.89)
happy : sad :: light : dark (0.91)
begin : end :: start : finish (0.87)
```

### Capital Relations
```
France : Paris :: Germany : Berlin (0.96)
Italy : Rome :: Spain : Madrid (0.94)
Japan : Tokyo :: China : Beijing (0.95)
```

### Semantic Relations
```
car : wheel :: table : leg (0.83)
book : page :: building : room (0.81)
tree : branch :: family : member (0.79)
```

## How Relation Extraction Works

### Step 1: Vectorization

Each word is converted to a semantic vector:

```python
def vectorize(word):
    # Returns vector in primitive space
    # E.g., king → [0.2, 0.8, 0.1, ...]
    return semantic_embeddings[word]
```

### Step 2: Relation Calculation

Extract the semantic difference:

```python
def extract_relation(word1, word2):
    v1 = vectorize(word1)
    v2 = vectorize(word2)
    return v2 - v1  # Semantic relation vector
```

### Step 3: Relation Application

Apply the relation to the third word:

```python
def apply_relation(word3, relation):
    v3 = vectorize(word3)
    result_vector = v3 + relation
    return find_nearest_word(result_vector)
```

### Step 4: Confidence Scoring

Score the result based on similarity:

```python
def score_result(answer, expected_relation):
    # Measures how well the relation holds
    return cosine_similarity(
        extract_relation(word3, answer),
        expected_relation
    )
```

## Performance

### Speed

| Operation | Time |
|-----------|------|
| Vectorize word | <1ms |
| Extract relation | <0.5ms |
| Find nearest word | ~5-10ms |
| Total analogy solve | ~10-15ms |

### Accuracy

On standard word analogy benchmarks:

| Test Set | Accuracy |
|----------|----------|
| Gender relations | 94% |
| Capital relations | 96% |
| Semantic relations | 87% |
| Syntactic relations | 85% |
| **Overall** | **91%** |

## Use Cases

### Language Learning
Help students understand word relationships and patterns:
```
"If king is to queen, then prince is to ___?"
```

### Semantic Search
Find semantically similar terms with relation preservation:
```
"Find words that relate to 'technology' the way 'Einstein' relates to 'physics'"
```

### NLP Tasks
Use analogies for:
- Word embeddings training
- Semantic similarity measurement
- Relation extraction
- Knowledge graph construction

### Creative Writing
Generate creative associations and metaphors

## Configuration

### Default Settings

```python
config = {
    "similarity_metric": "cosine",
    "relation_weight": 0.7,
    "confidence_threshold": 0.5,
    "max_candidates": 100,
    "cache_results": True
}
```

### Performance Tuning

```python
# Fast mode (suitable for real-time)
engine = AnalogyEngine(
    relation_weight=0.5,
    max_candidates=10
)

# Accurate mode (suitable for offline)
engine = AnalogyEngine(
    relation_weight=0.9,
    max_candidates=1000
)
```

## Next Steps

- Try [Basic Usage](../getting-started/basic-usage.md) examples
- Explore [API Reference](../api/overview.md)
- Read about [Semantic Primitives](primitives.md)
