#!/usr/bin/env python3
"""
ΣLANG Entity Extraction Example
================================

Demonstrates named entity recognition and relation extraction.
"""

import sys
sys.path.insert(0, '..')


def main():
    """Run entity extraction examples."""
    print("=" * 60)
    print("ΣLANG Entity Extraction Example")
    print("=" * 60)
    
    # Sample texts for entity extraction
    texts = [
        """
        Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne 
        in April 1976. The company is headquartered in Cupertino, California.
        Tim Cook became CEO in August 2011.
        """,
        """
        Albert Einstein developed the theory of relativity while working at 
        the Swiss Patent Office in Bern. He was awarded the Nobel Prize in 
        Physics in 1921.
        """,
        """
        The Python programming language was created by Guido van Rossum and 
        first released in 1991. It is widely used for machine learning, web 
        development, and data science applications.
        """
    ]
    
    # Initialize entity extractor
    print("\n1. Initializing entity extractor...")
    try:
        from sigmalang.core.entity_relation_extraction import EntityRelationExtractor
        extractor = EntityRelationExtractor()
        print("   ✓ Entity extractor ready")
        extractor_available = True
    except ImportError as e:
        print(f"   ⚠ EntityRelationExtractor not available: {e}")
        print("   Using pattern-based fallback...")
        extractor_available = False
    
    # Entity types
    print("\n2. Supported entity types:")
    print("-" * 50)
    
    entity_types = [
        ("PERSON", "People's names"),
        ("ORGANIZATION", "Companies, institutions"),
        ("LOCATION", "Places, cities, countries"),
        ("DATE", "Dates and time expressions"),
        ("PRODUCT", "Products, software, devices"),
        ("EVENT", "Named events, conferences"),
        ("CONCEPT", "Abstract concepts, technologies"),
    ]
    
    for etype, desc in entity_types:
        print(f"   {etype:15} | {desc}")
    
    # Extract entities from each text
    print("\n3. Entity extraction results:")
    print("=" * 60)
    
    for i, text in enumerate(texts):
        text = text.strip()
        print(f"\nDocument {i+1}:")
        print(f"   \"{text[:100]}...\"")
        print("-" * 50)
        
        if extractor_available:
            try:
                result = extractor.extract(text)
                
                # Print entities
                print("   Entities:")
                if hasattr(result, 'entities'):
                    for ent in result.entities[:8]:
                        etype = ent.type if hasattr(ent, 'type') else 'ENTITY'
                        conf = ent.confidence if hasattr(ent, 'confidence') else 0.9
                        print(f"      • {ent.text} ({etype}) [{conf:.0%}]")
                
                # Print relations
                if hasattr(result, 'relations') and result.relations:
                    print("\n   Relations:")
                    for rel in result.relations[:5]:
                        print(f"      • {rel.source} --[{rel.type}]--> {rel.target}")
            except Exception as e:
                print(f"   Error extracting entities: {e}")
        else:
            # Pattern-based fallback
            import re
            
            # Simple patterns
            patterns = {
                'PERSON': r'\b[A-Z][a-z]+ (?:van )?[A-Z][a-z]+\b',
                'ORGANIZATION': r'\b[A-Z][a-z]+ Inc\.?\b|\bSwiss Patent Office\b',
                'DATE': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b|\b\d{4}\b',
                'LOCATION': r'\bCupertino\b|\bCalifornia\b|\bBern\b',
            }
            
            print("   Entities (pattern-based):")
            for etype, pattern in patterns.items():
                matches = re.findall(pattern, text)
                for match in matches[:3]:
                    print(f"      • {match} ({etype})")
    
    # Relation types
    print("\n4. Supported relation types:")
    print("-" * 50)
    
    relation_types = [
        ("FOUNDED_BY", "Organization founded by person"),
        ("WORKS_FOR", "Person employed by organization"),
        ("LOCATED_IN", "Entity located in place"),
        ("CEO_OF", "Person leads organization"),
        ("CREATED_BY", "Product created by person"),
        ("AWARDED_TO", "Award given to person"),
    ]
    
    for rtype, desc in relation_types:
        print(f"   {rtype:15} | {desc}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
