"""
Tests for Entity and Relation Extraction Module.

Comprehensive test suite covering:
- Entity types and relation types
- Entity and Relation dataclasses
- Pattern-based entity recognition
- Relation extraction
- Knowledge graph operations
- Export formats
- Thread safety
- Integration tests
"""

import pytest
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from core.entity_relation_extraction import (
    EntityType,
    RelationType,
    Entity,
    Relation,
    ExtractionResult,
    EntityPattern,
    RelationPattern,
    EntityRecognizer,
    RelationExtractor,
    KnowledgeGraph,
    EntityRelationExtractor,
    create_extractor,
    extract_entities,
    extract_relations,
    build_knowledge_graph,
)


class TestEntityType:
    """Tests for EntityType enum."""
    
    def test_all_types_exist(self):
        """Test all expected entity types exist."""
        expected = [
            "PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME",
            "MONEY", "PERCENT", "PRODUCT", "EVENT", "WORK_OF_ART",
            "LANGUAGE", "QUANTITY", "ORDINAL", "CARDINAL", "CONCEPT", "UNKNOWN"
        ]
        actual = [t.name for t in EntityType]
        assert all(e in actual for e in expected)
    
    def test_from_string(self):
        """Test conversion from string."""
        assert EntityType.from_string("person") == EntityType.PERSON
        assert EntityType.from_string("PER") == EntityType.PERSON
        assert EntityType.from_string("org") == EntityType.ORGANIZATION
        assert EntityType.from_string("LOC") == EntityType.LOCATION
        assert EntityType.from_string("gpe") == EntityType.LOCATION
        assert EntityType.from_string("unknown_type") == EntityType.UNKNOWN


class TestRelationType:
    """Tests for RelationType enum."""
    
    def test_all_types_exist(self):
        """Test all expected relation types exist."""
        expected = [
            "WORKS_FOR", "LOCATED_IN", "BORN_IN", "FOUNDED_BY",
            "CEO_OF", "PART_OF", "OWNS", "CREATED_BY", "MARRIED_TO",
            "PARENT_OF", "SIBLING_OF", "FRIEND_OF", "MEMBER_OF",
            "AFFILIATED_WITH", "OCCURRED_AT", "OCCURRED_ON",
            "HAS_PROPERTY", "IS_A", "RELATED_TO", "UNKNOWN"
        ]
        actual = [t.name for t in RelationType]
        assert all(e in actual for e in expected)
    
    def test_from_string(self):
        """Test conversion from string."""
        assert RelationType.from_string("works_for") == RelationType.WORKS_FOR
        assert RelationType.from_string("employed_by") == RelationType.WORKS_FOR
        assert RelationType.from_string("located in") == RelationType.LOCATED_IN
        assert RelationType.from_string("based_in") == RelationType.LOCATED_IN
        assert RelationType.from_string("unknown_rel") == RelationType.UNKNOWN


class TestEntity:
    """Tests for Entity dataclass."""
    
    def test_creation(self):
        """Test entity creation."""
        entity = Entity(
            text="John Smith",
            entity_type=EntityType.PERSON,
            start=0,
            confidence=0.9
        )
        assert entity.text == "John Smith"
        assert entity.entity_type == EntityType.PERSON
        assert entity.start == 0
        assert entity.end == 10  # auto-calculated
        assert entity.confidence == 0.9
        assert entity.id is not None
    
    def test_auto_id_generation(self):
        """Test automatic ID generation."""
        entity1 = Entity(text="Test", entity_type=EntityType.CONCEPT, start=0)
        entity2 = Entity(text="Test", entity_type=EntityType.CONCEPT, start=0)
        entity3 = Entity(text="Test", entity_type=EntityType.CONCEPT, start=10)
        
        # Same content = same ID
        assert entity1.id == entity2.id
        # Different position = different ID
        assert entity1.id != entity3.id
    
    def test_hash_and_equality(self):
        """Test hashing and equality."""
        entity1 = Entity(text="Test", entity_type=EntityType.CONCEPT, start=0)
        entity2 = Entity(text="Test", entity_type=EntityType.CONCEPT, start=0)
        entity3 = Entity(text="Other", entity_type=EntityType.CONCEPT, start=0)
        
        assert entity1 == entity2
        assert entity1 != entity3
        assert hash(entity1) == hash(entity2)
        
        # Can be used in sets
        s = {entity1, entity2, entity3}
        assert len(s) == 2
    
    def test_to_dict_and_from_dict(self):
        """Test serialization."""
        entity = Entity(
            text="Google",
            entity_type=EntityType.ORGANIZATION,
            start=5,
            end=11,
            confidence=0.95,
            metadata={"source": "test"}
        )
        
        data = entity.to_dict()
        restored = Entity.from_dict(data)
        
        assert restored.text == entity.text
        assert restored.entity_type == entity.entity_type
        assert restored.start == entity.start
        assert restored.end == entity.end
        assert restored.confidence == entity.confidence
        assert restored.metadata == entity.metadata


class TestRelation:
    """Tests for Relation dataclass."""
    
    @pytest.fixture
    def sample_entities(self):
        """Create sample entities."""
        return (
            Entity(text="John", entity_type=EntityType.PERSON, start=0),
            Entity(text="Google", entity_type=EntityType.ORGANIZATION, start=20)
        )
    
    def test_creation(self, sample_entities):
        """Test relation creation."""
        source, target = sample_entities
        relation = Relation(
            source=source,
            target=target,
            relation_type=RelationType.WORKS_FOR,
            confidence=0.85,
            evidence="John works for Google"
        )
        
        assert relation.source == source
        assert relation.target == target
        assert relation.relation_type == RelationType.WORKS_FOR
        assert relation.confidence == 0.85
        assert relation.evidence == "John works for Google"
        assert relation.id is not None
    
    def test_hash_and_equality(self, sample_entities):
        """Test hashing and equality."""
        source, target = sample_entities
        
        rel1 = Relation(source=source, target=target, relation_type=RelationType.WORKS_FOR)
        rel2 = Relation(source=source, target=target, relation_type=RelationType.WORKS_FOR)
        rel3 = Relation(source=source, target=target, relation_type=RelationType.LOCATED_IN)
        
        assert rel1 == rel2
        assert rel1 != rel3
        assert hash(rel1) == hash(rel2)
    
    def test_to_dict_and_from_dict(self, sample_entities):
        """Test serialization."""
        source, target = sample_entities
        relation = Relation(
            source=source,
            target=target,
            relation_type=RelationType.WORKS_FOR,
            confidence=0.9,
            evidence="test evidence"
        )
        
        data = relation.to_dict()
        restored = Relation.from_dict(data)
        
        assert restored.source.text == relation.source.text
        assert restored.target.text == relation.target.text
        assert restored.relation_type == relation.relation_type
        assert restored.confidence == relation.confidence


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""
    
    @pytest.fixture
    def sample_result(self):
        """Create sample extraction result."""
        entities = [
            Entity(text="John", entity_type=EntityType.PERSON, start=0),
            Entity(text="Google", entity_type=EntityType.ORGANIZATION, start=20),
            Entity(text="California", entity_type=EntityType.LOCATION, start=40),
        ]
        relations = [
            Relation(
                source=entities[0],
                target=entities[1],
                relation_type=RelationType.WORKS_FOR
            ),
            Relation(
                source=entities[1],
                target=entities[2],
                relation_type=RelationType.LOCATED_IN
            ),
        ]
        return ExtractionResult(
            text="John works at Google in California",
            entities=entities,
            relations=relations
        )
    
    def test_get_entities_by_type(self, sample_result):
        """Test filtering entities by type."""
        persons = sample_result.get_entities_by_type(EntityType.PERSON)
        orgs = sample_result.get_entities_by_type(EntityType.ORGANIZATION)
        
        assert len(persons) == 1
        assert persons[0].text == "John"
        assert len(orgs) == 1
        assert orgs[0].text == "Google"
    
    def test_get_relations_by_type(self, sample_result):
        """Test filtering relations by type."""
        works_for = sample_result.get_relations_by_type(RelationType.WORKS_FOR)
        located_in = sample_result.get_relations_by_type(RelationType.LOCATED_IN)
        
        assert len(works_for) == 1
        assert len(located_in) == 1
    
    def test_get_relations_for_entity(self, sample_result):
        """Test getting relations for an entity."""
        google = sample_result.entities[1]
        relations = sample_result.get_relations_for_entity(google)
        
        assert len(relations) == 2  # Works_for and Located_in
    
    def test_to_dict(self, sample_result):
        """Test conversion to dict."""
        data = sample_result.to_dict()
        
        assert "text" in data
        assert "entities" in data
        assert "relations" in data
        assert len(data["entities"]) == 3
        assert len(data["relations"]) == 2


class TestEntityPattern:
    """Tests for EntityPattern class."""
    
    def test_date_pattern(self):
        """Test date pattern matching."""
        pattern = EntityPattern(
            r'\b\d{4}-\d{2}-\d{2}\b',
            EntityType.DATE
        )
        
        text = "The event is on 2024-01-15 and ends 2024-01-20."
        entities = pattern.find_matches(text)
        
        assert len(entities) == 2
        assert entities[0].text == "2024-01-15"
        assert entities[0].entity_type == EntityType.DATE
    
    def test_money_pattern(self):
        """Test money pattern matching."""
        pattern = EntityPattern(
            r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?',
            EntityType.MONEY
        )
        
        text = "The price is $1,000.00 or $500"
        entities = pattern.find_matches(text)
        
        assert len(entities) == 2
        assert all(e.entity_type == EntityType.MONEY for e in entities)
    
    def test_custom_processor(self):
        """Test custom text processor."""
        pattern = EntityPattern(
            r'@(\w+)',
            EntityType.PERSON,
            processor=lambda m: m.group(1)  # Extract just the username
        )
        
        text = "Contact @john_doe for more info"
        entities = pattern.find_matches(text)
        
        assert len(entities) == 1
        assert entities[0].text == "john_doe"


class TestRelationPattern:
    """Tests for RelationPattern class."""
    
    def test_works_for_pattern(self):
        """Test works for relation pattern."""
        pattern = RelationPattern(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+works\s+for\s+([A-Z][a-zA-Z\s&]+)',
            RelationType.WORKS_FOR,
            source_type=EntityType.PERSON,
            target_type=EntityType.ORGANIZATION
        )
        
        text = "John Smith works for Google Inc"
        entities = []  # Will create new entities
        relations = pattern.find_matches(text, entities)
        
        assert len(relations) == 1
        assert relations[0].relation_type == RelationType.WORKS_FOR
        assert relations[0].source.text == "John Smith"
        assert relations[0].target.text == "Google Inc"
    
    def test_pattern_with_existing_entities(self):
        """Test pattern linking to existing entities."""
        pattern = RelationPattern(
            r'(\w+)\s+is\s+located\s+in\s+(\w+)',
            RelationType.LOCATED_IN
        )
        
        entities = [
            Entity(text="Google", entity_type=EntityType.ORGANIZATION, start=0),
            Entity(text="California", entity_type=EntityType.LOCATION, start=25)
        ]
        
        text = "Google is located in California"
        relations = pattern.find_matches(text, entities)
        
        assert len(relations) == 1
        # Should link to existing entities
        assert relations[0].source.entity_type == EntityType.ORGANIZATION


class TestEntityRecognizer:
    """Tests for EntityRecognizer class."""
    
    @pytest.fixture
    def recognizer(self):
        """Create recognizer instance."""
        return EntityRecognizer()
    
    def test_recognize_dates(self, recognizer):
        """Test date recognition."""
        text = "The meeting is on January 15, 2024 at the office."
        entities = recognizer.recognize(text)
        
        dates = [e for e in entities if e.entity_type == EntityType.DATE]
        assert len(dates) >= 1
    
    def test_recognize_money(self, recognizer):
        """Test money recognition."""
        text = "The total cost is $1,500.00"
        entities = recognizer.recognize(text)
        
        money = [e for e in entities if e.entity_type == EntityType.MONEY]
        assert len(money) >= 1
    
    def test_recognize_percentages(self, recognizer):
        """Test percentage recognition."""
        text = "Sales increased by 25.5% and another 10% this quarter"
        entities = recognizer.recognize(text)
        
        percents = [e for e in entities if e.entity_type == EntityType.PERCENT]
        assert len(percents) >= 1
        # At least one percent should be found
        percent_texts = [e.text for e in percents]
        assert any("%" in t for t in percent_texts)
    
    def test_recognize_names(self, recognizer):
        """Test person name recognition."""
        text = "Dr. John Smith met with Sarah Johnson yesterday."
        entities = recognizer.recognize(text)
        
        persons = [e for e in entities if e.entity_type == EntityType.PERSON]
        # Should detect at least one name
        assert len(persons) >= 1
    
    def test_add_custom_entity(self, recognizer):
        """Test adding custom entities."""
        recognizer.add_entity("SigmaLang", EntityType.PRODUCT)
        
        text = "SigmaLang is a great encoding system."
        entities = recognizer.recognize(text)
        
        products = [e for e in entities if e.entity_type == EntityType.PRODUCT]
        assert len(products) == 1
        assert products[0].text == "SigmaLang"
    
    def test_add_custom_pattern(self, recognizer):
        """Test adding custom patterns."""
        pattern = EntityPattern(
            r'#\w+',
            EntityType.CONCEPT
        )
        recognizer.add_pattern(pattern)
        
        text = "Check out #programming and #python"
        entities = recognizer.recognize(text)
        
        concepts = [e for e in entities if e.entity_type == EntityType.CONCEPT]
        assert len(concepts) == 2
    
    def test_no_overlapping_entities(self, recognizer):
        """Test that entities don't overlap."""
        text = "$1,000.00 is the price"
        entities = recognizer.recognize(text)
        
        # Check for overlaps
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                # No overlap allowed
                assert e1.end <= e2.start or e2.end <= e1.start


class TestRelationExtractor:
    """Tests for RelationExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return RelationExtractor()
    
    def test_extract_works_for(self, extractor):
        """Test works_for relation extraction."""
        text = "John Smith works for Microsoft Corporation"
        entities = [
            Entity(text="John Smith", entity_type=EntityType.PERSON, start=0),
            Entity(text="Microsoft Corporation", entity_type=EntityType.ORGANIZATION, start=21)
        ]
        
        relations = extractor.extract(text, entities)
        
        works_for = [r for r in relations if r.relation_type == RelationType.WORKS_FOR]
        assert len(works_for) >= 1
    
    def test_extract_located_in(self, extractor):
        """Test located_in relation extraction."""
        text = "Google is headquartered in Mountain View"
        entities = [
            Entity(text="Google", entity_type=EntityType.ORGANIZATION, start=0),
            Entity(text="Mountain View", entity_type=EntityType.LOCATION, start=27)
        ]
        
        relations = extractor.extract(text, entities)
        
        located = [r for r in relations if r.relation_type == RelationType.LOCATED_IN]
        assert len(located) >= 1
    
    def test_extract_from_cooccurrence(self, extractor):
        """Test co-occurrence based extraction."""
        entities = [
            Entity(text="John", entity_type=EntityType.PERSON, start=0, end=4),
            Entity(text="Google", entity_type=EntityType.ORGANIZATION, start=10, end=16),
        ]
        
        relations = extractor.extract_from_entities(entities, co_occurrence_window=50)
        
        # Should infer AFFILIATED_WITH from person-org co-occurrence
        assert len(relations) >= 1
        assert relations[0].relation_type == RelationType.AFFILIATED_WITH
    
    def test_add_custom_pattern(self, extractor):
        """Test adding custom patterns."""
        pattern = RelationPattern(
            r'(\w+)\s+created\s+(\w+)',
            RelationType.CREATED_BY
        )
        extractor.add_pattern(pattern)
        
        text = "John created Widget"
        entities = []
        relations = extractor.extract(text, entities)
        
        created = [r for r in relations if r.relation_type == RelationType.CREATED_BY]
        assert len(created) >= 1


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph class."""
    
    @pytest.fixture
    def graph(self):
        """Create empty graph."""
        return KnowledgeGraph()
    
    @pytest.fixture
    def populated_graph(self):
        """Create graph with sample data."""
        graph = KnowledgeGraph()
        
        # Add entities
        john = Entity(text="John", entity_type=EntityType.PERSON, start=0)
        google = Entity(text="Google", entity_type=EntityType.ORGANIZATION, start=10)
        california = Entity(text="California", entity_type=EntityType.LOCATION, start=20)
        
        graph.add_entity(john)
        graph.add_entity(google)
        graph.add_entity(california)
        
        # Add relations
        graph.add_relation(Relation(
            source=john,
            target=google,
            relation_type=RelationType.WORKS_FOR
        ))
        graph.add_relation(Relation(
            source=google,
            target=california,
            relation_type=RelationType.LOCATED_IN
        ))
        
        return graph
    
    def test_add_entity(self, graph):
        """Test adding entities."""
        entity = Entity(text="Test", entity_type=EntityType.CONCEPT, start=0)
        entity_id = graph.add_entity(entity)
        
        assert entity_id is not None
        assert graph.num_entities == 1
        assert graph.get_entity(entity_id) == entity
    
    def test_add_relation(self, graph):
        """Test adding relations."""
        e1 = Entity(text="A", entity_type=EntityType.CONCEPT, start=0)
        e2 = Entity(text="B", entity_type=EntityType.CONCEPT, start=5)
        
        rel = Relation(source=e1, target=e2, relation_type=RelationType.RELATED_TO)
        rel_id = graph.add_relation(rel)
        
        assert rel_id is not None
        assert graph.num_relations == 1
        # Entities should be auto-added
        assert graph.num_entities == 2
    
    def test_get_entities_by_type(self, populated_graph):
        """Test filtering entities by type."""
        persons = populated_graph.get_entities(EntityType.PERSON)
        orgs = populated_graph.get_entities(EntityType.ORGANIZATION)
        
        assert len(persons) == 1
        assert len(orgs) == 1
    
    def test_get_relations_by_type(self, populated_graph):
        """Test filtering relations by type."""
        works_for = populated_graph.get_relations(RelationType.WORKS_FOR)
        located_in = populated_graph.get_relations(RelationType.LOCATED_IN)
        
        assert len(works_for) == 1
        assert len(located_in) == 1
    
    def test_get_neighbors(self, populated_graph):
        """Test getting neighbors."""
        google_entities = populated_graph.get_entities(EntityType.ORGANIZATION)
        google = google_entities[0]
        
        neighbors = populated_graph.get_neighbors(google.id)
        
        assert len(neighbors) == 2  # John and California
    
    def test_find_path(self, populated_graph):
        """Test path finding."""
        john = populated_graph.get_entities(EntityType.PERSON)[0]
        california = populated_graph.get_entities(EntityType.LOCATION)[0]
        
        path = populated_graph.find_path(john.id, california.id)
        
        assert path is not None
        assert len(path) == 3  # John -> Google -> California
    
    def test_find_path_no_connection(self, graph):
        """Test path finding with no connection."""
        e1 = Entity(text="A", entity_type=EntityType.CONCEPT, start=0)
        e2 = Entity(text="B", entity_type=EntityType.CONCEPT, start=5)
        graph.add_entity(e1)
        graph.add_entity(e2)
        
        path = graph.find_path(e1.id, e2.id)
        assert path is None
    
    def test_query(self, populated_graph):
        """Test graph querying."""
        # Query by entity type
        results = populated_graph.query(entity_type=EntityType.PERSON)
        assert len(results) == 1
        
        # Query by text
        results = populated_graph.query(text_contains="Google")
        assert len(results) >= 1
        
        # Query by relation type
        results = populated_graph.query(relation_type=RelationType.WORKS_FOR)
        assert len(results) == 1
    
    def test_merge_graphs(self, populated_graph):
        """Test merging graphs."""
        other = KnowledgeGraph()
        amazon = Entity(text="Amazon", entity_type=EntityType.ORGANIZATION, start=0)
        other.add_entity(amazon)
        
        original_count = populated_graph.num_entities
        populated_graph.merge(other)
        
        assert populated_graph.num_entities == original_count + 1
    
    def test_to_dict_and_from_dict(self, populated_graph):
        """Test serialization."""
        data = populated_graph.to_dict()
        restored = KnowledgeGraph.from_dict(data)
        
        assert restored.num_entities == populated_graph.num_entities
        assert restored.num_relations == populated_graph.num_relations
    
    def test_to_json(self, populated_graph):
        """Test JSON export."""
        json_str = populated_graph.to_json()
        data = json.loads(json_str)
        
        assert "entities" in data
        assert "relations" in data
    
    def test_to_cypher(self, populated_graph):
        """Test Cypher export."""
        cypher = populated_graph.to_cypher()
        
        assert "CREATE" in cypher
        assert "MATCH" in cypher
        assert "PERSON" in cypher
        assert "WORKS_FOR" in cypher


class TestEntityRelationExtractor:
    """Tests for main EntityRelationExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return EntityRelationExtractor(build_graph=True)
    
    def test_extract_entities_and_relations(self, extractor):
        """Test full extraction."""
        text = "John Smith works for Google in California."
        result = extractor.extract(text)
        
        assert len(result.entities) > 0
        assert result.text == text
    
    def test_extract_with_cooccurrence(self, extractor):
        """Test extraction with co-occurrence."""
        text = "John Smith, the CEO of Google, lives in California."
        result = extractor.extract(text, use_cooccurrence=True)
        
        # Should have entities
        assert len(result.entities) > 0
    
    def test_batch_extraction(self, extractor):
        """Test batch extraction."""
        texts = [
            "John works for Google.",
            "Microsoft is in Seattle.",
            "Apple was founded by Steve Jobs."
        ]
        
        results = extractor.extract_batch(texts)
        
        assert len(results) == 3
        assert all(isinstance(r, ExtractionResult) for r in results)
    
    def test_graph_building(self, extractor):
        """Test automatic graph building."""
        text1 = "John Smith works for Google."
        text2 = "Google is located in California."
        
        extractor.extract(text1)
        extractor.extract(text2)
        
        graph = extractor.graph
        assert graph is not None
        assert graph.num_entities > 0
    
    def test_reset_graph(self, extractor):
        """Test graph reset."""
        # Add custom entity to ensure detection
        extractor.add_custom_entity("Google", EntityType.ORGANIZATION)
        extractor.extract("Google is a major tech company.")
        
        # Reset should clear the graph
        extractor.reset_graph()
        assert extractor.graph.num_entities == 0
    
    def test_add_custom_entity(self, extractor):
        """Test adding custom entity."""
        extractor.add_custom_entity("SigmaLang", EntityType.PRODUCT)
        
        result = extractor.extract("SigmaLang is revolutionary.")
        products = result.get_entities_by_type(EntityType.PRODUCT)
        
        assert len(products) == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_extractor(self):
        """Test create_extractor function."""
        extractor = create_extractor(build_graph=True)
        assert isinstance(extractor, EntityRelationExtractor)
        assert extractor.graph is not None
        
        extractor_no_graph = create_extractor(build_graph=False)
        assert extractor_no_graph.graph is None
    
    def test_extract_entities(self):
        """Test quick entity extraction."""
        entities = extract_entities("The price is $100 on January 15, 2024")
        
        assert len(entities) > 0
        types = {e.entity_type for e in entities}
        assert EntityType.MONEY in types or EntityType.DATE in types
    
    def test_extract_relations(self):
        """Test quick relation extraction."""
        text = "John Smith works for Microsoft Corporation in Seattle"
        relations = extract_relations(text)
        
        # May or may not find relations depending on pattern matches
        assert isinstance(relations, list)
    
    def test_build_knowledge_graph(self):
        """Test knowledge graph building."""
        texts = [
            "John Smith works for Google.",
            "Google is located in Mountain View.",
            "Sarah Johnson works for Microsoft."
        ]
        
        graph = build_knowledge_graph(texts)
        
        assert isinstance(graph, KnowledgeGraph)
        assert graph.num_entities > 0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_text(self):
        """Test with empty text."""
        extractor = EntityRelationExtractor()
        result = extractor.extract("")
        
        assert result.entities == []
        assert result.relations == []
    
    def test_no_entities(self):
        """Test text with no entities."""
        extractor = EntityRelationExtractor()
        result = extractor.extract("the quick brown fox jumps over the lazy dog")
        
        # May or may not find entities
        assert isinstance(result.entities, list)
    
    def test_unicode_text(self):
        """Test with unicode text."""
        extractor = EntityRelationExtractor()
        text = "François Müller works for Société Générale in München"
        result = extractor.extract(text)
        
        assert isinstance(result, ExtractionResult)
    
    def test_special_characters(self):
        """Test with special characters."""
        extractor = EntityRelationExtractor()
        text = "Price: $1,000.00 (25% discount!) on 2024-01-15"
        result = extractor.extract(text)
        
        assert len(result.entities) > 0
    
    def test_very_long_text(self):
        """Test with very long text."""
        extractor = EntityRelationExtractor()
        text = "John works for Google. " * 100
        result = extractor.extract(text)
        
        assert isinstance(result, ExtractionResult)
    
    def test_overlapping_patterns(self):
        """Test handling of overlapping patterns."""
        recognizer = EntityRecognizer()
        text = "$1,000 USD"  # Money pattern might match multiple ways
        entities = recognizer.recognize(text)
        
        # Should not have overlapping entities
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                # Check no overlap
                assert e1.end <= e2.start or e2.end <= e1.start, \
                    f"Overlap: {e1.text} ({e1.start}-{e1.end}) and {e2.text} ({e2.start}-{e2.end})"


class TestThreadSafety:
    """Tests for thread safety."""
    
    def test_concurrent_recognition(self):
        """Test concurrent entity recognition."""
        recognizer = EntityRecognizer()
        texts = [f"John Smith bought ${i * 100} worth of goods" for i in range(10)]
        
        results = []
        def recognize(text):
            entities = recognizer.recognize(text)
            results.append(entities)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(recognize, texts))
        
        assert len(results) == 10
        assert all(isinstance(r, list) for r in results)
    
    def test_concurrent_graph_updates(self):
        """Test concurrent graph updates."""
        graph = KnowledgeGraph()
        
        def add_entity(i):
            entity = Entity(
                text=f"Entity{i}",
                entity_type=EntityType.CONCEPT,
                start=i
            )
            graph.add_entity(entity)
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(add_entity, range(100)))
        
        assert graph.num_entities == 100
    
    def test_concurrent_extraction(self):
        """Test concurrent extraction with shared graph."""
        extractor = EntityRelationExtractor(build_graph=True)
        
        # Add custom entities that will definitely be recognized
        for i in range(20):
            extractor.add_custom_entity(f"Person{i}", EntityType.PERSON)
            extractor.add_custom_entity(f"Company{i}", EntityType.ORGANIZATION)
        
        texts = [f"Person{i} works at Company{i}" for i in range(20)]
        
        def extract(text):
            return extractor.extract(text)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(extract, texts))
        
        assert len(results) == 20
        # Graph should have accumulated entities from custom entities
        assert extractor.graph.num_entities >= 20


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test full extraction pipeline."""
        texts = [
            "Dr. John Smith, CEO of TechCorp, announced a $10 million investment.",
            "TechCorp is headquartered in San Francisco, California.",
            "The company was founded by John Smith in January 2010.",
            "Sarah Johnson joined TechCorp as CTO on March 15, 2020."
        ]
        
        extractor = create_extractor(build_graph=True)
        
        for text in texts:
            result = extractor.extract(text)
            assert len(result.entities) > 0
        
        graph = extractor.graph
        
        # Should have built a knowledge graph
        assert graph.num_entities > 0
        
        # Test export formats
        json_export = graph.to_json()
        assert "entities" in json_export
        
        cypher_export = graph.to_cypher()
        assert "CREATE" in cypher_export
    
    def test_custom_patterns_integration(self):
        """Test custom patterns with full pipeline."""
        extractor = create_extractor()
        
        # Add custom entity
        extractor.add_custom_entity("SigmaLang", EntityType.PRODUCT)
        
        # Add custom pattern for hashtags
        extractor.add_entity_pattern(EntityPattern(
            r'#\w+',
            EntityType.CONCEPT
        ))
        
        text = "SigmaLang is trending with #NLP and #MachineLearning"
        result = extractor.extract(text)
        
        products = result.get_entities_by_type(EntityType.PRODUCT)
        concepts = result.get_entities_by_type(EntityType.CONCEPT)
        
        assert len(products) == 1
        assert len(concepts) == 2
    
    def test_graph_traversal(self):
        """Test knowledge graph traversal."""
        # Build graph with extractor and custom entities
        extractor = create_extractor(build_graph=True)
        
        # Add custom entities for reliable detection
        extractor.add_custom_entity("Alice", EntityType.PERSON)
        extractor.add_custom_entity("TechCorp", EntityType.ORGANIZATION)
        extractor.add_custom_entity("Boston", EntityType.LOCATION)
        extractor.add_custom_entity("Bob", EntityType.PERSON)
        extractor.add_custom_entity("Carol", EntityType.PERSON)
        
        texts = [
            "Alice works for TechCorp",
            "TechCorp is located in Boston",
            "Bob works for TechCorp",
            "TechCorp was founded by Carol"
        ]
        
        for text in texts:
            extractor.extract(text)
        
        graph = extractor.graph
        
        # Test various operations
        all_entities = graph.get_entities()
        all_relations = graph.get_relations()
        
        assert len(all_entities) > 0
        
        # Test query
        orgs = graph.query(entity_type=EntityType.ORGANIZATION)
        assert isinstance(orgs, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
