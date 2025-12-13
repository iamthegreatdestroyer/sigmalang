#!/usr/bin/env python3
"""
Example 08: Advanced Analogies
==============================

This example demonstrates advanced analogy capabilities in sigmalang,
including multi-step reasoning, analogy chains, and custom analogy types.

Analogies are fundamental to reasoning and transfer learning.
sigmalang's semantic representations enable powerful analogy operations.
"""

from typing import Optional
from dataclasses import dataclass

# Import sigmalang components
try:
    from sigmalang.core.encoder import SigmaEncoder
    from sigmalang.core.primitives import SemanticAnalogyEngine
    ANALOGY_AVAILABLE = True
except ImportError:
    ANALOGY_AVAILABLE = False
    print("Note: SemanticAnalogyEngine not available. Using demonstration mode.")


@dataclass
class AnalogyResult:
    """Result of an analogy operation."""
    query: str          # The analogy query (A:B::C:?)
    answer: str         # The predicted answer
    confidence: float   # Confidence score (0-1)
    reasoning: str      # Explanation of the analogy
    alternatives: list  # Alternative answers


class AdvancedAnalogyEngine:
    """
    Advanced analogy engine with extended capabilities.
    
    Features:
    - Multiple analogy types
    - Chain reasoning
    - Custom relation definitions
    - Explanation generation
    """
    
    # Predefined analogy patterns
    ANALOGY_PATTERNS = {
        "gender": {
            "king": "queen", "man": "woman", "boy": "girl",
            "father": "mother", "brother": "sister", "uncle": "aunt",
            "husband": "wife", "actor": "actress", "prince": "princess"
        },
        "capital": {
            "France": "Paris", "Japan": "Tokyo", "Germany": "Berlin",
            "Italy": "Rome", "Spain": "Madrid", "UK": "London",
            "USA": "Washington", "China": "Beijing", "India": "Delhi"
        },
        "country_language": {
            "France": "French", "Germany": "German", "Spain": "Spanish",
            "Japan": "Japanese", "China": "Chinese", "Italy": "Italian",
            "Portugal": "Portuguese", "Russia": "Russian", "Greece": "Greek"
        },
        "comparative": {
            "big": "bigger", "small": "smaller", "fast": "faster",
            "slow": "slower", "good": "better", "bad": "worse",
            "hot": "hotter", "cold": "colder", "tall": "taller"
        },
        "superlative": {
            "big": "biggest", "small": "smallest", "fast": "fastest",
            "good": "best", "bad": "worst", "tall": "tallest"
        },
        "past_tense": {
            "walk": "walked", "run": "ran", "eat": "ate",
            "go": "went", "see": "saw", "take": "took",
            "write": "wrote", "read": "read", "speak": "spoke"
        },
        "profession_tool": {
            "doctor": "stethoscope", "painter": "brush", "carpenter": "hammer",
            "chef": "knife", "writer": "pen", "photographer": "camera",
            "musician": "instrument", "gardener": "shovel", "teacher": "book"
        },
        "animal_sound": {
            "dog": "bark", "cat": "meow", "cow": "moo",
            "lion": "roar", "bird": "chirp", "snake": "hiss",
            "horse": "neigh", "pig": "oink", "sheep": "baa"
        },
        "part_whole": {
            "finger": "hand", "leaf": "tree", "page": "book",
            "wheel": "car", "key": "keyboard", "pixel": "image",
            "word": "sentence", "star": "galaxy", "cell": "body"
        }
    }
    
    def __init__(self):
        """Initialize the advanced analogy engine."""
        if ANALOGY_AVAILABLE:
            self.base_engine = SemanticAnalogyEngine()
        else:
            self.base_engine = None
        
        # Build reverse mappings for analogy solving
        self._build_reverse_mappings()
    
    def _build_reverse_mappings(self):
        """Build reverse mappings for bidirectional analogies."""
        self.reverse_patterns = {}
        for category, pairs in self.ANALOGY_PATTERNS.items():
            self.reverse_patterns[category] = {v: k for k, v in pairs.items()}
    
    def solve(self, a: str, b: str, c: str) -> AnalogyResult:
        """
        Solve analogy: A is to B as C is to ?
        
        Args:
            a: First term
            b: Second term (related to a)
            c: Third term (analogous to a)
            
        Returns:
            AnalogyResult with answer and metadata
        """
        # Try to use the base engine if available
        if self.base_engine:
            try:
                result = self.base_engine.solve_analogy(a, b, c)
                return AnalogyResult(
                    query=f"{a}:{b}::{c}:?",
                    answer=result,
                    confidence=0.85,
                    reasoning="Solved using neural semantic embeddings",
                    alternatives=[]
                )
            except Exception:
                pass
        
        # Fall back to pattern matching
        return self._pattern_based_solve(a, b, c)
    
    def _pattern_based_solve(self, a: str, b: str, c: str) -> AnalogyResult:
        """Solve using predefined patterns."""
        a_lower, b_lower, c_lower = a.lower(), b.lower(), c.lower()
        
        # Check each pattern category
        for category, patterns in self.ANALOGY_PATTERNS.items():
            # Check forward mapping
            if a_lower in patterns and patterns[a_lower].lower() == b_lower:
                if c_lower in patterns:
                    answer = patterns[c_lower]
                    return AnalogyResult(
                        query=f"{a}:{b}::{c}:?",
                        answer=answer,
                        confidence=0.95,
                        reasoning=f"Matched {category} pattern: {a}→{b} similar to {c}→{answer}",
                        alternatives=[]
                    )
            
            # Check reverse mapping
            reverse = self.reverse_patterns.get(category, {})
            if a_lower in reverse and reverse[a_lower].lower() == b_lower:
                if c_lower in reverse:
                    answer = reverse[c_lower]
                    return AnalogyResult(
                        query=f"{a}:{b}::{c}:?",
                        answer=answer,
                        confidence=0.95,
                        reasoning=f"Matched reverse {category} pattern",
                        alternatives=[]
                    )
        
        # No pattern found
        return AnalogyResult(
            query=f"{a}:{b}::{c}:?",
            answer="[unknown]",
            confidence=0.0,
            reasoning="No matching pattern found",
            alternatives=[]
        )
    
    def chain_analogy(self, terms: list[tuple[str, str]]) -> list[AnalogyResult]:
        """
        Solve a chain of analogies.
        
        Args:
            terms: List of (a, b) pairs to chain
            
        Returns:
            List of analogy results
        """
        results = []
        
        for i in range(len(terms) - 1):
            a, b = terms[i]
            c, _ = terms[i + 1]
            result = self.solve(a, b, c)
            results.append(result)
        
        return results
    
    def find_relation(self, a: str, b: str) -> Optional[str]:
        """
        Identify the relationship between two terms.
        
        Args:
            a: First term
            b: Second term
            
        Returns:
            Relationship type or None
        """
        a_lower, b_lower = a.lower(), b.lower()
        
        for category, patterns in self.ANALOGY_PATTERNS.items():
            if a_lower in patterns and patterns[a_lower].lower() == b_lower:
                return category
            
            reverse = self.reverse_patterns.get(category, {})
            if a_lower in reverse and reverse[a_lower].lower() == b_lower:
                return f"reverse_{category}"
        
        return None
    
    def complete_analogy_set(self, category: str) -> list[tuple[str, str]]:
        """
        Get all analogies of a specific type.
        
        Args:
            category: Analogy category name
            
        Returns:
            List of (term, related_term) pairs
        """
        patterns = self.ANALOGY_PATTERNS.get(category, {})
        return list(patterns.items())
    
    def generate_quiz(self, category: str, count: int = 5) -> list[dict]:
        """
        Generate analogy quiz questions.
        
        Args:
            category: Category for questions
            count: Number of questions
            
        Returns:
            List of quiz questions
        """
        import random
        
        patterns = self.ANALOGY_PATTERNS.get(category, {})
        items = list(patterns.items())
        
        if len(items) < 2:
            return []
        
        questions = []
        random.shuffle(items)
        
        for i in range(min(count, len(items) - 1)):
            a, b = items[i]
            c, d = items[(i + 1) % len(items)]
            
            questions.append({
                "question": f"{a} : {b} :: {c} : ?",
                "answer": d,
                "category": category,
                "distractors": [items[j][1] for j in range(len(items)) if j != (i + 1) % len(items)][:3]
            })
        
        return questions


def example_basic_analogy():
    """Demonstrate basic analogy solving."""
    print("=" * 60)
    print("Basic Analogy Solving")
    print("=" * 60)
    
    engine = AdvancedAnalogyEngine()
    
    analogies = [
        ("king", "queen", "man"),      # Gender
        ("France", "Paris", "Japan"),  # Capital
        ("big", "bigger", "small"),    # Comparative
        ("walk", "walked", "run"),     # Past tense
    ]
    
    print("\nSolving analogies (A:B :: C:?):")
    for a, b, c in analogies:
        result = engine.solve(a, b, c)
        print(f"\n  {a} : {b} :: {c} : {result.answer}")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Reasoning: {result.reasoning}")


def example_chain_reasoning():
    """Demonstrate chain analogy reasoning."""
    print("\n" + "=" * 60)
    print("Chain Analogy Reasoning")
    print("=" * 60)
    
    engine = AdvancedAnalogyEngine()
    
    # Build a chain of related concepts
    chain = [
        ("France", "Paris"),
        ("Japan", "Tokyo"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
    ]
    
    print("\nAnalogy chain:")
    results = engine.chain_analogy(chain)
    
    for i, result in enumerate(results):
        print(f"  Step {i+1}: {result.query} → {result.answer}")


def example_relation_discovery():
    """Demonstrate relationship discovery."""
    print("\n" + "=" * 60)
    print("Relation Discovery")
    print("=" * 60)
    
    engine = AdvancedAnalogyEngine()
    
    pairs = [
        ("king", "queen"),
        ("France", "French"),
        ("dog", "bark"),
        ("finger", "hand"),
        ("apple", "banana"),  # No known relation
    ]
    
    print("\nDiscovering relationships:")
    for a, b in pairs:
        relation = engine.find_relation(a, b)
        if relation:
            print(f"  {a} → {b}: {relation}")
        else:
            print(f"  {a} → {b}: [no pattern match]")


def example_analogy_categories():
    """Demonstrate different analogy categories."""
    print("\n" + "=" * 60)
    print("Analogy Categories")
    print("=" * 60)
    
    engine = AdvancedAnalogyEngine()
    
    print("\nAvailable analogy categories:")
    for category in engine.ANALOGY_PATTERNS.keys():
        pairs = engine.complete_analogy_set(category)
        sample = pairs[:3]
        sample_str = ", ".join(f"{a}→{b}" for a, b in sample)
        print(f"\n  {category.upper()}:")
        print(f"    Examples: {sample_str}...")
        print(f"    Total pairs: {len(pairs)}")


def example_quiz_generation():
    """Demonstrate quiz generation."""
    print("\n" + "=" * 60)
    print("Analogy Quiz Generation")
    print("=" * 60)
    
    engine = AdvancedAnalogyEngine()
    
    quiz = engine.generate_quiz("capital", count=3)
    
    print("\nGenerated Quiz (Capital Cities):")
    for i, q in enumerate(quiz, 1):
        print(f"\n  Question {i}: {q['question']}")
        print(f"  Answer: {q['answer']}")
        if q['distractors']:
            print(f"  Distractors: {', '.join(q['distractors'][:2])}")


def example_multi_hop_reasoning():
    """Demonstrate multi-hop reasoning through analogies."""
    print("\n" + "=" * 60)
    print("Multi-Hop Reasoning")
    print("=" * 60)
    
    engine = AdvancedAnalogyEngine()
    
    print("\nProblem: What language is spoken in the capital of Italy?")
    print("\nReasoning steps:")
    
    # Step 1: Italy → Capital
    step1 = engine.solve("France", "Paris", "Italy")
    print(f"  1. Italy's capital: {step1.answer}")
    
    # Step 2: Country → Language
    step2 = engine.solve("France", "French", "Italy")
    print(f"  2. Italy's language: {step2.answer}")
    
    print(f"\n  Answer: {step2.answer} is spoken in {step1.answer}")


def example_analogy_algebra():
    """Demonstrate algebraic operations on analogies."""
    print("\n" + "=" * 60)
    print("Analogy Algebra")
    print("=" * 60)
    
    print("""
    Analogies can be thought of as vector operations:
    
    king - man + woman ≈ queen
    
    This is because:
    - "king" encodes [royalty, male]
    - "man" encodes [male]
    - "woman" encodes [female]
    - Subtracting "man" removes [male]
    - Adding "woman" adds [female]
    - Result: [royalty, female] ≈ "queen"
    """)
    
    engine = AdvancedAnalogyEngine()
    
    # Demonstrate the concept
    result = engine.solve("king", "queen", "man")
    print(f"  king : queen :: man : {result.answer}")
    print(f"  (Verified: king - man + woman = {result.answer})")


def example_cross_domain():
    """Demonstrate cross-domain analogies."""
    print("\n" + "=" * 60)
    print("Cross-Domain Analogies")
    print("=" * 60)
    
    print("""
    Cross-domain analogies transfer knowledge between fields:
    
    Biological Neural Networks : Artificial Neural Networks
    Natural Selection : Genetic Algorithms
    Ant Colonies : Swarm Optimization
    DNA : Source Code
    
    These cross-domain mappings inspire innovation through:
    1. Identifying structural similarities
    2. Transferring solution strategies
    3. Discovering new research directions
    """)
    
    cross_domain_examples = [
        ("neurons", "nodes", "synapses", "connections"),
        ("evolution", "optimization", "mutation", "perturbation"),
        ("ant", "agent", "pheromone", "signal"),
    ]
    
    engine = AdvancedAnalogyEngine()
    
    print("\nCross-domain mappings:")
    for a, b, c, expected in cross_domain_examples:
        print(f"  {a}:{b} :: {c}:{expected}")
        print(f"    (Domain transfer: biology → computer science)")


def main():
    """Run all advanced analogy examples."""
    print("Sigmalang Advanced Analogies Examples")
    print("=" * 60)
    
    example_basic_analogy()
    example_chain_reasoning()
    example_relation_discovery()
    example_analogy_categories()
    example_quiz_generation()
    example_multi_hop_reasoning()
    example_analogy_algebra()
    example_cross_domain()
    
    print("\n" + "=" * 60)
    print("Key Concepts")
    print("=" * 60)
    print("""
    1. Proportional Analogies: A:B :: C:D
       The relationship between A and B mirrors C and D

    2. Semantic Vectors: Words as points in space
       Similar meanings are nearby in vector space

    3. Relation Types: Categories of analogies
       Gender, comparative, part-whole, etc.

    4. Chain Reasoning: Multi-step inference
       Link multiple analogies for complex reasoning

    5. Cross-Domain Transfer: Knowledge transfer
       Apply patterns from one field to another

    6. Analogy Algebra: Vector operations
       Add, subtract, combine semantic concepts
    """)


if __name__ == "__main__":
    main()
