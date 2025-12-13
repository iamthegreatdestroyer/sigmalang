"""
ΣLANG Semantic Parser
=====================

Extracts semantic primitives from human natural language input.
Transforms unstructured text into structured SemanticTree representations.

Uses a hybrid approach:
1. Rule-based pattern matching for common structures
2. Dependency parsing for grammatical relationships
3. Entity/action extraction via lightweight NER
4. Template matching for learned patterns

Copyright 2025 - Ryot LLM Project
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum, auto

from .primitives import (
    SemanticNode, SemanticTree, 
    ExistentialPrimitive, CodePrimitive, MathPrimitive,
    LogicPrimitive, EntityPrimitive, ActionPrimitive,
    CommunicationPrimitive, StructurePrimitive,
    PRIMITIVE_REGISTRY
)


class IntentType(Enum):
    """High-level classification of user intent."""
    QUERY = auto()          # Asking for information
    COMMAND = auto()        # Requesting action
    STATEMENT = auto()      # Providing information
    CODE_REQUEST = auto()   # Requesting code generation
    EXPLANATION = auto()    # Requesting explanation
    MODIFICATION = auto()   # Requesting changes
    COMPARISON = auto()     # Requesting comparison
    DEFINITION = auto()     # Requesting definition


@dataclass
class ParsedEntity:
    """An extracted entity from the input."""
    text: str
    entity_type: int        # Primitive ID
    start: int              # Start position in source
    end: int                # End position in source
    confidence: float       # Extraction confidence
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class ParsedRelation:
    """An extracted relation between entities."""
    source: ParsedEntity
    target: ParsedEntity
    relation_type: int      # Primitive ID
    confidence: float


class SemanticParser:
    """
    Parses human input into semantic tree structures.
    
    The parser operates in multiple passes:
    1. Tokenization and normalization
    2. Intent classification
    3. Entity extraction
    4. Relation extraction
    5. Tree construction
    """
    
    def __init__(self):
        self._init_patterns()
        self._init_keywords()
    
    def _init_patterns(self):
        """Initialize regex patterns for extraction."""
        # Code-related patterns
        self.code_patterns = {
            'function_request': re.compile(
                r'(?:create|write|make|build|implement)\s+(?:a\s+)?'
                r'(?:(?P<lang>\w+)\s+)?(?:function|method|def)\s+'
                r'(?:that\s+|to\s+|which\s+)?(?P<action>.+)',
                re.IGNORECASE
            ),
            'class_request': re.compile(
                r'(?:create|write|make|build)\s+(?:a\s+)?'
                r'(?:(?P<lang>\w+)\s+)?class\s+(?:for\s+|that\s+)?(?P<purpose>.+)',
                re.IGNORECASE
            ),
            'code_action': re.compile(
                r'(?P<action>sort|filter|map|reduce|search|find|validate|parse|convert|transform)\s+'
                r'(?:the\s+)?(?:a\s+)?(?P<target>.+)',
                re.IGNORECASE
            ),
            'fix_request': re.compile(
                r'(?:fix|debug|repair|correct|resolve)\s+(?:the\s+)?(?:this\s+)?'
                r'(?P<issue>.+)',
                re.IGNORECASE
            ),
        }
        
        # Query patterns
        self.query_patterns = {
            'what_is': re.compile(
                r'what\s+(?:is|are)\s+(?:a\s+|an\s+|the\s+)?(?P<subject>.+)\??',
                re.IGNORECASE
            ),
            'how_to': re.compile(
                r'how\s+(?:do\s+(?:i|you|we)\s+|to\s+|can\s+(?:i|you|we)\s+)?(?P<action>.+)\??',
                re.IGNORECASE
            ),
            'why': re.compile(
                r'why\s+(?:does|do|is|are|did|would|should)\s+(?P<subject>.+)\??',
                re.IGNORECASE
            ),
            'explain': re.compile(
                r'(?:explain|describe|tell\s+me\s+about)\s+(?P<subject>.+)',
                re.IGNORECASE
            ),
        }
        
        # Action patterns
        self.action_patterns = {
            'create': re.compile(
                r'(?:create|make|build|generate|produce)\s+(?:a\s+|an\s+)?(?P<target>.+)',
                re.IGNORECASE
            ),
            'modify': re.compile(
                r'(?:modify|change|update|edit|alter)\s+(?:the\s+)?(?P<target>.+)',
                re.IGNORECASE
            ),
            'delete': re.compile(
                r'(?:delete|remove|destroy|eliminate)\s+(?:the\s+)?(?P<target>.+)',
                re.IGNORECASE
            ),
            'compare': re.compile(
                r'compare\s+(?P<first>.+?)\s+(?:with|to|and|vs\.?)\s+(?P<second>.+)',
                re.IGNORECASE
            ),
        }
        
        # Data type patterns
        self.datatype_patterns = {
            'list': re.compile(r'\b(?:list|array|sequence|collection)\b', re.IGNORECASE),
            'dict': re.compile(r'\b(?:dict|dictionary|map|mapping|hash)\b', re.IGNORECASE),
            'string': re.compile(r'\b(?:string|text|str)\b', re.IGNORECASE),
            'number': re.compile(r'\b(?:number|int|integer|float|numeric)\b', re.IGNORECASE),
            'boolean': re.compile(r'\b(?:boolean|bool|true|false)\b', re.IGNORECASE),
        }
    
    def _init_keywords(self):
        """Initialize keyword mappings to primitives."""
        self.action_keywords = {
            # Creation
            'create': ActionPrimitive.CREATE,
            'make': ActionPrimitive.CREATE,
            'build': ActionPrimitive.CREATE,
            'generate': ActionPrimitive.CREATE,
            'write': ActionPrimitive.CREATE,
            'implement': ActionPrimitive.CREATE,
            
            # Destruction
            'delete': ActionPrimitive.DESTROY,
            'remove': ActionPrimitive.DESTROY,
            'destroy': ActionPrimitive.DESTROY,
            
            # Modification
            'modify': ActionPrimitive.MODIFY,
            'change': ActionPrimitive.MODIFY,
            'update': ActionPrimitive.MODIFY,
            'edit': ActionPrimitive.MODIFY,
            'fix': ActionPrimitive.MODIFY,
            
            # Movement/Transfer
            'move': ActionPrimitive.MOVE,
            'transfer': ActionPrimitive.TRANSFER,
            'copy': ActionPrimitive.TRANSFER,
            
            # Combination
            'combine': ActionPrimitive.COMBINE,
            'merge': ActionPrimitive.COMBINE,
            'join': ActionPrimitive.COMBINE,
            'concatenate': ActionPrimitive.COMBINE,
            
            # Separation
            'split': ActionPrimitive.SEPARATE,
            'separate': ActionPrimitive.SEPARATE,
            'divide': ActionPrimitive.SEPARATE,
            
            # Analysis
            'compare': ActionPrimitive.COMPARE,
            'search': ActionPrimitive.SEARCH,
            'find': ActionPrimitive.SEARCH,
            'select': ActionPrimitive.SELECT,
            'choose': ActionPrimitive.SELECT,
            'sort': ActionPrimitive.SORT,
            'filter': ActionPrimitive.FILTER,
            'validate': ActionPrimitive.VALIDATE,
            
            # Transformation
            'map': ActionPrimitive.MAP,
            'transform': ActionPrimitive.MAP,
            'convert': ActionPrimitive.MAP,
            'reduce': ActionPrimitive.REDUCE,
            'aggregate': ActionPrimitive.REDUCE,
            
            # Execution
            'run': ActionPrimitive.EXECUTE,
            'execute': ActionPrimitive.EXECUTE,
            'perform': ActionPrimitive.EXECUTE,
        }
        
        self.communication_keywords = {
            'what': CommunicationPrimitive.QUERY,
            'how': CommunicationPrimitive.QUERY,
            'why': CommunicationPrimitive.QUERY,
            'when': CommunicationPrimitive.QUERY,
            'where': CommunicationPrimitive.QUERY,
            'which': CommunicationPrimitive.QUERY,
            'explain': CommunicationPrimitive.EXPLAIN,
            'describe': CommunicationPrimitive.DESCRIBE,
            'define': CommunicationPrimitive.DEFINE,
            'example': CommunicationPrimitive.EXAMPLE,
            'show': CommunicationPrimitive.EXAMPLE,
        }
        
        self.code_keywords = {
            'function': CodePrimitive.FUNCTION,
            'method': CodePrimitive.FUNCTION,
            'def': CodePrimitive.FUNCTION,
            'class': CodePrimitive.CLASS,
            'variable': CodePrimitive.VARIABLE,
            'var': CodePrimitive.VARIABLE,
            'loop': CodePrimitive.LOOP,
            'for': CodePrimitive.LOOP,
            'while': CodePrimitive.LOOP,
            'if': CodePrimitive.BRANCH,
            'condition': CodePrimitive.BRANCH,
            'return': CodePrimitive.RETURN,
            'import': CodePrimitive.IMPORT,
            'parameter': CodePrimitive.PARAMETER,
            'argument': CodePrimitive.PARAMETER,
            'type': CodePrimitive.TYPE,
        }
        
        self.structure_keywords = {
            'list': StructurePrimitive.LIST,
            'array': StructurePrimitive.ARRAY,
            'dict': StructurePrimitive.DICT,
            'dictionary': StructurePrimitive.DICT,
            'map': StructurePrimitive.DICT,
            'tree': StructurePrimitive.TREE,
            'graph': StructurePrimitive.GRAPH,
            'stack': StructurePrimitive.STACK,
            'queue': StructurePrimitive.QUEUE,
            'string': StructurePrimitive.STRING,
            'number': StructurePrimitive.NUMBER,
            'boolean': StructurePrimitive.BOOLEAN,
        }
        
        # Programming language identifiers
        self.languages = {
            'python', 'javascript', 'typescript', 'java', 'c', 'cpp', 'c++',
            'rust', 'go', 'ruby', 'php', 'swift', 'kotlin', 'scala',
            'haskell', 'lisp', 'clojure', 'elixir', 'r', 'julia', 'sql'
        }
    
    def parse(self, text: str) -> SemanticTree:
        """
        Parse human input into a semantic tree.
        
        Args:
            text: Human natural language input
            
        Returns:
            SemanticTree representation of the input
        """
        # Normalize input
        normalized = self._normalize(text)
        
        # Classify intent
        intent = self._classify_intent(normalized)
        
        # Extract entities
        entities = self._extract_entities(normalized)
        
        # Extract relations
        relations = self._extract_relations(normalized, entities)
        
        # Build semantic tree
        root = self._build_tree(intent, entities, relations, normalized)
        
        return SemanticTree(root=root, source_text=text)
    
    def _normalize(self, text: str) -> str:
        """Normalize input text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove trailing punctuation for matching (keep original)
        return text.strip()
    
    def _classify_intent(self, text: str) -> IntentType:
        """Classify the high-level intent of the input."""
        text_lower = text.lower()
        
        # Check for code requests first (high priority)
        for pattern in self.code_patterns.values():
            if pattern.search(text_lower):
                return IntentType.CODE_REQUEST
        
        # Check for queries
        if any(text_lower.startswith(q) for q in ['what', 'how', 'why', 'when', 'where', 'which']):
            return IntentType.QUERY
        
        # Check for explanation requests
        if any(word in text_lower for word in ['explain', 'describe', 'tell me about']):
            return IntentType.EXPLANATION
        
        # Check for definition requests
        if 'define' in text_lower or text_lower.startswith('what is'):
            return IntentType.DEFINITION
        
        # Check for comparison requests
        if 'compare' in text_lower or ' vs ' in text_lower or ' versus ' in text_lower:
            return IntentType.COMPARISON
        
        # Check for modification requests
        if any(word in text_lower for word in ['fix', 'modify', 'change', 'update', 'edit']):
            return IntentType.MODIFICATION
        
        # Check for commands
        if any(word in text_lower for word in ['create', 'make', 'build', 'generate', 'write']):
            return IntentType.COMMAND
        
        # Default to statement
        return IntentType.STATEMENT
    
    def _extract_entities(self, text: str) -> List[ParsedEntity]:
        """Extract entities from the input text."""
        entities = []
        text_lower = text.lower()
        words = text_lower.split()
        
        # Extract programming language mentions
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w+#]', '', word)
            if clean_word in self.languages:
                start = text_lower.find(clean_word)
                entities.append(ParsedEntity(
                    text=clean_word,
                    entity_type=CodePrimitive.TYPE,
                    start=start,
                    end=start + len(clean_word),
                    confidence=0.95,
                    attributes={'language': clean_word}
                ))
        
        # Extract code structure mentions
        for keyword, primitive in self.code_keywords.items():
            if keyword in text_lower:
                start = text_lower.find(keyword)
                entities.append(ParsedEntity(
                    text=keyword,
                    entity_type=primitive,
                    start=start,
                    end=start + len(keyword),
                    confidence=0.9
                ))
        
        # Extract data structure mentions
        for keyword, primitive in self.structure_keywords.items():
            if keyword in text_lower:
                start = text_lower.find(keyword)
                entities.append(ParsedEntity(
                    text=keyword,
                    entity_type=primitive,
                    start=start,
                    end=start + len(keyword),
                    confidence=0.9
                ))
        
        # Extract action mentions
        for keyword, primitive in self.action_keywords.items():
            if keyword in text_lower:
                start = text_lower.find(keyword)
                entities.append(ParsedEntity(
                    text=keyword,
                    entity_type=primitive,
                    start=start,
                    end=start + len(keyword),
                    confidence=0.85
                ))
        
        # Sort by position
        entities.sort(key=lambda e: e.start)
        
        # Remove duplicates at same position
        seen_positions = set()
        unique_entities = []
        for entity in entities:
            if entity.start not in seen_positions:
                seen_positions.add(entity.start)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_relations(self, text: str, entities: List[ParsedEntity]) -> List[ParsedRelation]:
        """Extract relations between entities."""
        relations = []
        
        # Simple sequential relation extraction
        # Entities that appear near each other are likely related
        for i, entity in enumerate(entities):
            if i + 1 < len(entities):
                next_entity = entities[i + 1]
                # Check if entities are close (within 5 words)
                gap = next_entity.start - entity.end
                text_between = text[entity.end:next_entity.start].lower()
                
                # Determine relation type based on text between
                relation_type = ExistentialPrimitive.RELATION
                
                if any(word in text_between for word in ['to', 'into', 'for']):
                    relation_type = ExistentialPrimitive.ACTION
                elif any(word in text_between for word in ['with', 'using', 'by']):
                    relation_type = EntityPrimitive.INSTRUMENT
                elif any(word in text_between for word in ['in', 'on', 'at']):
                    relation_type = ExistentialPrimitive.SPATIAL
                elif any(word in text_between for word in ['that', 'which', 'who']):
                    relation_type = ExistentialPrimitive.ATTRIBUTE
                
                if gap < 50:  # Characters, not words
                    relations.append(ParsedRelation(
                        source=entity,
                        target=next_entity,
                        relation_type=relation_type,
                        confidence=max(0.5, 1.0 - gap / 100)
                    ))
        
        return relations
    
    def _build_tree(self, intent: IntentType, entities: List[ParsedEntity],
                    relations: List[ParsedRelation], text: str) -> SemanticNode:
        """Build semantic tree from extracted components."""
        
        # Determine root primitive based on intent
        intent_to_primitive = {
            IntentType.QUERY: CommunicationPrimitive.QUERY,
            IntentType.COMMAND: ExistentialPrimitive.ACTION,
            IntentType.STATEMENT: CommunicationPrimitive.ASSERT,
            IntentType.CODE_REQUEST: ActionPrimitive.CREATE,
            IntentType.EXPLANATION: CommunicationPrimitive.EXPLAIN,
            IntentType.MODIFICATION: ActionPrimitive.MODIFY,
            IntentType.COMPARISON: ActionPrimitive.COMPARE,
            IntentType.DEFINITION: CommunicationPrimitive.DEFINE,
        }
        
        root_primitive = intent_to_primitive.get(intent, ExistentialPrimitive.ABSTRACT)
        
        # Create root node
        root = SemanticNode(
            primitive=root_primitive,
            modifiers={'intent': intent.name}
        )
        
        # Add entities as children
        entity_nodes: Dict[int, SemanticNode] = {}
        
        for entity in entities:
            node = SemanticNode(
                primitive=entity.entity_type,
                value=entity.text,
                modifiers=entity.attributes
            )
            entity_nodes[entity.start] = node
        
        # Connect via relations
        connected = set()
        for relation in relations:
            source_node = entity_nodes.get(relation.source.start)
            target_node = entity_nodes.get(relation.target.start)
            
            if source_node and target_node:
                # Create relation node
                relation_node = SemanticNode(
                    primitive=relation.relation_type,
                    children=[target_node]
                )
                source_node.children.append(relation_node)
                connected.add(relation.source.start)
                connected.add(relation.target.start)
        
        # Add unconnected entities directly to root
        for start, node in entity_nodes.items():
            if start not in connected or node not in [c for n in root.children for c in n.children]:
                # Only add if not already a child
                if node not in root.children:
                    root.children.append(node)
        
        # Ensure at least some content
        if not root.children:
            # Create a content node with the text
            content_node = SemanticNode(
                primitive=ExistentialPrimitive.ABSTRACT,
                value=text[:100]  # Truncate for safety
            )
            root.children.append(content_node)
        
        return root
    
    def parse_code_request(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Specialized parser for code generation requests.
        Returns structured code request details.
        """
        result = {
            'language': None,
            'structure': None,  # function, class, etc.
            'action': None,
            'target': None,
            'modifiers': []
        }
        
        text_lower = text.lower()
        
        # Extract language
        for lang in self.languages:
            if lang in text_lower:
                result['language'] = lang
                break
        
        # Try function request pattern
        match = self.code_patterns['function_request'].search(text)
        if match:
            result['structure'] = 'function'
            if match.group('lang'):
                result['language'] = match.group('lang')
            result['action'] = match.group('action')
            return result
        
        # Try class request pattern
        match = self.code_patterns['class_request'].search(text)
        if match:
            result['structure'] = 'class'
            if match.group('lang'):
                result['language'] = match.group('lang')
            result['action'] = match.group('purpose')
            return result
        
        # Try code action pattern
        match = self.code_patterns['code_action'].search(text)
        if match:
            result['action'] = match.group('action')
            result['target'] = match.group('target')
            return result
        
        return None
    
    def extract_modifiers(self, text: str) -> List[Tuple[str, Any]]:
        """Extract modifier phrases from text."""
        modifiers = []
        text_lower = text.lower()
        
        # Order modifiers
        order_patterns = [
            (r'in\s+(ascending|descending|reverse)\s+order', 'order'),
            (r'(ascending|descending|alphabetical|numerical)\s*(?:order)?', 'order'),
            (r'sorted\s+by\s+(\w+)', 'sort_key'),
        ]
        
        for pattern, modifier_type in order_patterns:
            match = re.search(pattern, text_lower)
            if match:
                modifiers.append((modifier_type, match.group(1)))
        
        # Size/limit modifiers
        size_patterns = [
            (r'(?:top|first|last)\s+(\d+)', 'limit'),
            (r'(\d+)\s+(?:items?|elements?|results?)', 'count'),
        ]
        
        for pattern, modifier_type in size_patterns:
            match = re.search(pattern, text_lower)
            if match:
                modifiers.append((modifier_type, int(match.group(1))))
        
        # Condition modifiers
        if ' where ' in text_lower or ' when ' in text_lower or ' if ' in text_lower:
            modifiers.append(('has_condition', True))
        
        return modifiers


class SemanticTreePrinter:
    """Utility for visualizing semantic trees."""
    
    @staticmethod
    def print_tree(tree: SemanticTree, indent: int = 0) -> str:
        """Return string representation of semantic tree."""
        lines = []
        SemanticTreePrinter._print_node(tree.root, lines, indent)
        return '\n'.join(lines)
    
    @staticmethod
    def _print_node(node: SemanticNode, lines: List[str], indent: int):
        """Recursively print node and children."""
        prefix = '  ' * indent
        name = PRIMITIVE_REGISTRY.get_name(node.primitive)
        
        value_str = f": {node.value}" if node.value else ""
        mod_str = f" {node.modifiers}" if node.modifiers else ""
        
        lines.append(f"{prefix}├── Σ[{name}]{value_str}{mod_str}")
        
        for child in node.children:
            SemanticTreePrinter._print_node(child, lines, indent + 1)
