from typing import (
    List, 
    Literal, 
    LiteralString,
    Sequence, 
    Mapping,
    MutableSequence,
    Union,
    Set, 
    Final,
    Tuple, 
    Optional, 
    Any,
    Dict
)
from dataclasses import (
    dataclass,
    field,
    asdict
)
from enum import (
    Enum, 
    Flag, 
    auto
)
import hashlib, re, json
from datetime import datetime

from ..explict import (
    TreeNode,
    TreeTraversal
)

__all__ = [
    "Language",
    "TokenType",
    "Token",
    "ASTNode",
    "ComparisonNode",
    "LogicalNode",
    "UnaryNode",
    "ProximityNode",
    "WithinNode",
    "PartOfSpeech",
    "GrammaticalCase",
    "SyntacticFunction",
    "Person",
    "Number",
    "Gender",
    "Mood",
    "Tense",
    "Voice",
    "TraversalMode",
    "ClauseType",
    "ClauseFunction",
    "RelationType",
    "HypothesisType",
    "Morphology",
    "WordOccurrence",
    "SyntaxNode",
    "Word",
    "Phrase",
    "Clause",
    "Sentence",
    "ConceptNode",
    "ConceptEdge",
    "Morpheme",
    "Hypothesis",
    "PixelAnchor",
    "HypothesisResult",
    "SemanticProfile",
    "ClauseGraph",
    "SyntaxTree",
    "TextualVariationInstance",
    "LanguageEra",
    "HypothesisTestResult",
    "MLSuggestion",
    "MorphologyFilter"
]


class PartOfSpeech(Enum):
    """Part of speech classifications."""
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PRONOUN = "pronoun"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"
    ARTICLE = "article"
    PARTICLE = "particle"
    INTERJECTION = "interjection"
    NUMERAL = "numeral"

@dataclass
class LanguageEra:
    """Historical language period."""
    name: str
    start_date: str  # e.g., "500 BC"
    end_date: str
    language: str  # e.g., "biblical_hebrew", "koine_greek"
    corpus_size: int

@dataclass
class TextualVariationInstance:
    """Record of a specific textual variation."""
    reference: str
    main_reading: str
    variant_reading: str
    manuscripts: List[str]
    significance: float
    semantic_impact: bool

class TokenType(Enum):
    """Lexical token types."""
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    EQ = "EQ"
    LT = "LT"
    GT = "GT"
    LTE = "LTE"
    GTE = "GTE"
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    INTEGER = "INTEGER"
    EOF = "EOF"
    COMMA = "COMMA"
    WITHIN = "WITHIN"
    PROXIMITY = "PROXIMITY"

@dataclass
class Token:
    """Lexical token."""
    type: TokenType
    value: Any
    position: int = 0

class Language(Enum):
    HEBREW = "he"
    ARAMAIC = "arc"
    GREEK = "grc"
    GREEK_NT = "grc-nt"

class Person(Enum):
    FIRST = "1"
    SECOND = "2"
    THIRD = "3"

class Number(Enum):
    SINGULAR = "sg"
    PLURAL = "pl"
    DUAL = "du"

class Gender(Enum):
    MASCULINE = "m"
    FEMININE = "f"
    NEUTER = "n"

class Mood(Enum):
    INDICATIVE = "ind"
    SUBJUNCTIVE = "subj"
    OPTATIVE = "opt"
    IMPERATIVE = "imp"
    CONDITIONAL = "cond"
    INFINITIVE = "inf"

class GrammaticalCase(Enum):
    """Grammatical case."""
    NOMINATIVE = "nominative"
    GENITIVE = "genitive"
    DATIVE = "dative"
    ACCUSATIVE = "accusative"
    VOCATIVE = "vocative"
    LOCATIVE = "locative"
    INSTRUMENTAL = "instrumental"

class Tense(Enum):
    PRESENT = "pres"
    IMPERFECT = "impf"
    AORIST = "aor"
    PERFECT = "perf"
    PLUPERFECT = "plup"
    FUTURE = "fut"

class Voice(Enum):
    ACTIVE = "act"
    MIDDLE = "mid"
    PASSIVE = "pass"
    MIDDLE_PASSIVE = "mid-pass"

class TraversalMode(Enum):
    FRONT_FOOTPRINT = "front"      # Root -> Leaves (causal flow)
    REVERSE_FOOTPRINT = "reverse"  # Leaves -> Root (dependency flow) 
    TRANSVERSE = "transverse"      # Cross-hierarchical connections

class ClauseType(Enum):
    MAIN = "main"
    SUBORDINATE = "subordinate"
    RELATIVE = "relative"
    CONDITIONAL = "conditional"
    CONCESSIVE = "concessive"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    FINAL = "final"
    CONSECUTIVE = "consecutive"

class SyntacticFunction(Enum):
    """Role of a phrase/clause in sentence structure."""
    SUBJECT = "subject"
    PREDICATE = "predicate"
    DIRECT_OBJECT = "direct_object"
    INDIRECT_OBJECT = "indirect_object"
    SUBJECT_COMPLEMENT = "subject_complement"
    OBJECT_COMPLEMENT = "object_complement"
    ADVERBIAL_MODIFIER = "adverbial_modifier"
    ADJECTIVAL_MODIFIER = "adjectival_modifier"
    PREPOSITIONAL_PHRASE = "prepositional_phrase"
    APPOSITIVE = "appositive"

@dataclass
class MorphologyFilter:
    """Morphological filter for word search."""
    lemma: Optional[str] = None
    pos: Optional[PartOfSpeech] = None
    tense: Optional[Tense] = None
    voice: Optional[Voice] = None
    mood: Optional[Mood] = None
    person: Optional[Person] = None
    gender: Optional[Gender] = None
    number: Optional[Number] = None
    case: Optional[GrammaticalCase] = None
    frequency_min: Optional[int] = None
    frequency_max: Optional[int] = None
    speaker: Optional[str] = None
    discourse_context: Optional[str] = None

class ClauseFunction(Enum):
    SUBJECT = "subj"
    PREDICATE = "pred"
    OBJECT = "obj"
    COMPLEMENT = "comp"
    ADVERBIAL = "adv"
    NOMINAL = "nom"
    RELATIVE = "rel"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    CONDITIONAL = "cond"
    PURPOSE = "purpose"
    RESULT = "result"

class RelationType(Enum):
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    HYPONYM = "hyponym"
    HYPERNYM = "hypernym"
    MERONYM = "meronym"
    HOLONYM = "holonym"
    ASSOCIATION = "association"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    INFLUENCES = "influences"
    DEPENDS_ON = "depends_on"
    REFERENCES = "references"
    EXTENDS = "extends"
    CONTRADICTS = "contradicts"

class HypothesisType(Enum):
    FREQUENCY = "frequency"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    SIGNIFICANCE = "significance"


@dataclass
class ASTNode:
    """Base AST node."""
    pass

@dataclass
class ComparisonNode(ASTNode):
    """Comparison: field=value or field>value, etc."""
    field: str
    operator: str
    value: Any

@dataclass
class LogicalNode(ASTNode):
    """Logical operation: AND, OR."""
    operator: str
    left: ASTNode
    right: ASTNode

@dataclass
class UnaryNode(ASTNode):
    """Unary NOT."""
    operand: ASTNode

@dataclass
class ProximityNode(ASTNode):
    """Proximity query: lemmas PROXIMITY distance."""
    lemmas: List[str]
    distance: int

@dataclass
class WithinNode(ASTNode):
    """Clause-aware: query WITHIN clause_type."""
    query: ASTNode
    clause_type: str

@dataclass
class Hypothesis:
    """A testable hypothesis about biblical text."""
    hypothesis_id: str
    title: str
    description: str
    
    # The query that defines the "treatment" group
    treatment_query: str  # DSL query
    # Optional control group query
    control_query: Optional[str] = None
    
    # Statistical test to apply
    test_type: HypothesisType = HypothesisType.FREQUENCY
    
    # Expected outcome
    expected_direction: str = "unknown"  # "increase", "decrease", "different"

@dataclass
class HypothesisResult:
    """Output of hypothesis test."""
    hypothesis_id: str
    treatment_count: int
    treatment_freq: float
    control_count: Optional[int] = None
    control_freq: Optional[float] = None
    
    # Statistical results
    p_value: Optional[float] = None
    effect_size: Optional[float] = None  # Cohen's h for proportions
    confidence_interval: Optional[tuple] = None  # (lower, upper)
    
    significant: bool = False
    conclusion: str = ""

@dataclass
class HypothesisTestResult:
    """Result of hypothesis test."""
    hypothesis_id: str
    test_name: str
    p_value: float
    test_statistic: float
    result: str  # "accepted", "rejected", "inconclusive"
    confidence: float
    evidence: Dict


@dataclass
class SemanticProfile:
    """Semantic fingerprint of a lemma in a context."""
    lemma_id: str
    context: str  # "gospels", "paul", "revelation", etc.
    
    # Co-occurrence partners (most frequent companions)
    cooccurrence_lemmas: List[Tuple[str, int]]  # (lemma_id, count)
    
    # Morphological distribution
    tense_dist: Dict[str, int]  # tense -> count
    mood_dist: Dict[str, int]
    voice_dist: Dict[str, int]
    
    # Syntactic role distribution
    clause_function_dist: Dict[str, int]  # function -> count
    clause_type_dist: Dict[str, int]

@dataclass
class ConceptNode:
    """A theological concept."""
    concept_id: str
    lemma_ids: List[str]  # Which lemmas express this concept
    name: str
    description: str
    semantic_field: str  # e.g., "SOTERIOLOGY", "CHRISTOLOGY", "PNEUMATOLOGY"
    
    frequency: int = 0  # How many times concepts appears

@dataclass
class ConceptEdge:
    """Relationship between concepts."""
    edge_id: str
    source_concept_id: str
    target_concept_id: str
    relation_type: RelationType
    
    # Strength
    weight: float = 1.0  # Weighted by co-occurrence frequency
    
    # Evidence
    co_occurrence_count: int = 0  # How often they co-occur
    manuscripts: List[str] = None  # In which manuscripts
    
    # Confidence
    confidence: float = 1.0  # Manual / ML derived

@dataclass
class Morphology:
    """Immutable morphological annotation for a word."""
    lemma_id: str  # Stable ID for lemma (e.g., "grc-1234")
    lemma_text: str  # Surface form of lemma (e.g., "λόγος")
    language: Language
    
    # Normalized tags (empty = don't care in queries)
    pos: str  # Part of speech: noun, verb, adj, etc.
    tense: Optional[Tense] = None
    mood: Optional[Mood] = None
    voice: Optional[Voice] = None
    person: Optional[Person] = None
    number: Optional[Number] = None
    gender: Optional[Gender] = None
    case: Optional[str] = None  # nom, gen, dat, acc, voc, loc, abl
    degree: Optional[str] = None  # pos, comp, super (adjectives)
    
    # Metadata
    frequency: int = 0  # How many times this lemma appears in corpus
    semantic_class: Optional[str] = None  # e.g., "AGENT", "LOCATION", "PROPERTY"
    
    def __hash__(self):
        """Stable hash for indexing."""
        canonical = f"{self.lemma_id}|{self.pos}|{self.tense}|{self.mood}|{self.voice}|{self.person}|{self.number}|{self.gender}|{self.case}"
        return int(hashlib.sha256(canonical.encode()).hexdigest(), 16)

@dataclass
class MLSuggestion:
    """ML-generated semantic suggestion."""
    source_lemma: str
    suggested_translation: str
    confidence: float  # 0.0-1.0
    model_name: str
    rationale: str
    is_override: bool = False
    human_correction: Optional[str] = None

@dataclass
class Morpheme:
    """Atomic morphological unit."""
    form: str  # Surface form
    lemma: str  # Base form
    pos: PartOfSpeech

    # Optional morphological features
    tense: Optional[Tense] = None
    voice: Optional[Voice] = None
    mood: Optional[Mood] = None
    person: Optional[Person] = None
    gender: Optional[Gender] = None
    number: Optional[Number] = None
    case: Optional[GrammaticalCase] = None

    # Metadata
    lexeme_id: Optional[str] = None  # Reference to lexicon entry
    frequency: int = 0
    semantic_domain: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling Enums."""
        data = asdict(self)
        # Convert Enums to strings
        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value
        return data

    def matches_morphology(self, **filters) -> bool:
        """Check if this morpheme matches given morphological filters."""
        for attr, value in filters.items():
            if value is None:
                continue
            if isinstance(value, Enum):
                value = value.value
            actual = getattr(self, attr, None)
            if isinstance(actual, Enum):
                actual = actual.value
            if actual != value:
                return False
        return True

@dataclass
class Word:
    """A single word in the text with full linguistic annotation."""
    word_id: str  # Unique ID (e.g., "B01.001.001.001")
    morpheme: Morpheme
    position_in_clause: int  # 0-indexed position

    # Syntactic relationships
    head_word_id: Optional[str] = None  # ID of syntactic head
    dependents: List[str] = field(default_factory=list)  # IDs of dependent words

    # Semantic
    semantic_sense: Optional[str] = None  # WordNet or similar
    gloss: Optional[str] = None  # English gloss

    # Speaker/discourse attribution
    speaker: Optional[str] = None
    discourse_context: Optional[str] = None

    # Textual variant tracking
    manuscript_variants: Dict[str, str] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "word_id": self.word_id,
            "morpheme": self.morpheme.to_dict(),
            "position_in_clause": self.position_in_clause,
            "head_word_id": self.head_word_id,
            "dependents": self.dependents,
            "semantic_sense": self.semantic_sense,
            "gloss": self.gloss,
            "speaker": self.speaker,
            "discourse_context": self.discourse_context,
            "manuscript_variants": self.manuscript_variants,
        }

@dataclass
class Phrase:
    """Hierarchical phrase unit."""
    phrase_id: str
    phrase_type: str  # e.g., "noun_phrase", "verb_phrase", "prepositional_phrase"
    words: List[Word]
    function: SyntacticFunction
    head_word_id: Optional[str] = None
    parent_clause_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phrase_id": self.phrase_id,
            "phrase_type": self.phrase_type,
            "words": [w.to_dict() for w in self.words],
            "function": self.function.value,
            "head_word_id": self.head_word_id,
            "parent_clause_id": self.parent_clause_id,
        }

@dataclass
class Clause:
    """Linguistic clause with full syntactic analysis."""
    clause_id: str
    clause_type: ClauseType
    phrases: List[Phrase]
    parent_sentence_id: Optional[str] = None

    # Syntactic relationships
    parent_clause_id: Optional[str] = None  # If subordinate
    subordinate_clause_ids: List[str] = field(default_factory=list)

    # Content
    main_verb_id: Optional[str] = None

    # Discourse
    discourse_function: Optional[str] = None
    theme_and_rheme: Optional[Tuple[str, str]] = None  # (theme, rheme) word IDs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "clause_type": self.clause_type.value,
            "phrases": [p.to_dict() for p in self.phrases],
            "parent_sentence_id": self.parent_sentence_id,
            "parent_clause_id": self.parent_clause_id,
            "subordinate_clause_ids": self.subordinate_clause_ids,
            "main_verb_id": self.main_verb_id,
            "discourse_function": self.discourse_function,
        }

@dataclass
class PixelAnchor:
    """Maps text to manuscript image coordinates."""
    word_id: str
    manuscript_id: str
    pixel_coords: Tuple[int, int, int, int]
    confidence: float
    folio: str

    def to_dict(self) -> Dict:
        return {
            "word_id": self.word_id,
            "manuscript_id": self.manuscript_id,
            "pixel_coords": self.pixel_coords,
            "confidence": self.confidence,
            "folio": self.folio,
        }

@dataclass
class Sentence:
    """Complete sentence with all linguistic annotations."""
    sentence_id: str
    clauses: List[Clause]
    text: str  # Raw text
    language: str  # e.g., "koine_greek", "biblical_hebrew", "aramaic"

    # Metadata
    reference: str = None
    translation: Optional[Language] = None  # English translation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "clauses": [c.to_dict() for c in self.clauses],
            "text": self.text,
            "language": self.language,
            "translation": self.translation,
        }

@dataclass
class WordOccurrence:
    """An instance of a word in a manuscript."""
    occurrence_id: str  # Stable UUID (deterministic from manuscript position)
    manuscript_id: str  # Which manuscript
    book: str  # Standard abbreviation (e.g., "Matt", "Mark")
    chapter: int
    verse: int
    word_index: int  # 0-based position in verse
    surface_form: str  # Actual text as it appears
    morphology: Morphology  # Normalized annotations
    
    # Anchoring
    image_region: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) on folio image
    folio_id: Optional[str] = None  # Which folio/page
    
    # Discourse
    speaker_id: Optional[str] = None  # Who is speaking (e.g., "Jesus")
    discourse_unit: Optional[str] = None  # ID of containing discourse unit
    
    def stable_id(self):
        """Deterministic ID independent of insertion order."""
        canonical = f"{self.manuscript_id}|{self.book}|{self.chapter}|{self.verse}|{self.word_index}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

@dataclass
class SyntaxNode:
    """A node in the syntax tree."""
    node_id: str  # Stable UUID
    label: str  # Grammatical label (e.g., "VP", "NP", "S")
    type_label: str  # Detailed type (e.g., "Verb_Phrase_Aorist_Active")
    
    # Tree structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    word_occurrences: List[str] = field(default_factory=list)  # word_occurrence_ids
    
    # Attributes
    clause_type: Optional[ClauseType] = None
    clause_function: Optional[ClauseFunction] = None
    discourse_function: Optional[str] = None  # e.g., "SUBJECT OF DISCOURSE"
    
    # Metadata
    confidence: float = 1.0
    source: str = "manual"  # "manual", "rule-based", "ml"
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict."""
        return {
            "node_id": self.node_id,
            "label": self.label,
            "type_label": self.type_label,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "word_occurrences": self.word_occurrences,
            "clause_type": self.clause_type.value if self.clause_type else None,
            "clause_function": self.clause_function.value if self.clause_function else None,
            "confidence": self.confidence,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SyntaxNode":
        """Reconstruct from dict."""
        return SyntaxNode(
            node_id=d["node_id"],
            label=d["label"],
            type_label=d["type_label"],
            parent_id=d.get("parent_id"),
            children_ids=d.get("children_ids", []),
            word_occurrences=d.get("word_occurrences", []),
            clause_type=ClauseType(d["clause_type"]) if d.get("clause_type") else None,
            clause_function=ClauseFunction(d["clause_function"]) if d.get("clause_function") else None,
            confidence=d.get("confidence", 1.0),
        )

class ClauseGraph:
    """Graph representation of clause relationships."""

    def __init__(self):
        # Adjacency list: clause_id → [(dependent_clause_id, relation_type)]
        self.adjacency: Dict[str, List[Tuple[str, str]]] = {}
        self.clauses: Dict[str, Clause] = {}

    def add_clause(self, clause: Clause) -> None:
        """Add clause to graph."""
        self.clauses[clause.clause_id] = clause
        if clause.clause_id not in self.adjacency:
            self.adjacency[clause.clause_id] = []

    def add_dependency(self, parent_id: str, child_id: str, relation: str = "subordinate") -> None:
        """Add dependency edge."""
        if parent_id not in self.adjacency:
            self.adjacency[parent_id] = []
        self.adjacency[parent_id].append((child_id, relation))

    def get_dependents(self, clause_id: str) -> List[str]:
        """Get all dependent clause IDs."""
        return [dep_id for dep_id, _ in self.adjacency.get(clause_id, [])]

    def get_dependency_depth(self, clause_id: str) -> int:
        """Get maximum depth of dependencies."""
        if not self.get_dependents(clause_id):
            return 0
        return 1 + max(self.get_dependency_depth(dep) for dep in self.get_dependents(clause_id))

    def topological_sort(self) -> List[str]:
        """Topological sort of clauses (dependency order)."""
        visited = set()
        result = []

        def visit(clause_id: str):
            if clause_id in visited:
                return
            visited.add(clause_id)
            for dep_id, _ in self.adjacency.get(clause_id, []):
                visit(dep_id)
            result.append(clause_id)

        for clause_id in self.clauses.keys():
            visit(clause_id)

        return result

    def find_paths(self, start: str, end: str) -> List[List[str]]:
        """Find all paths from start to end clause."""
        paths = []

        def dfs(current: str, target: str, path: List[str]) -> None:
            if current == target:
                paths.append(path + [current])
                return
            for next_id, _ in self.adjacency.get(current, []):
                if next_id not in path:
                    dfs(next_id, target, path + [current])

        dfs(start, end, [])
        return paths
    
class SyntaxTree:
    """Complete syntax tree for a sentence."""

    def __init__(self, root_node: TreeNode):
        self.root = root_node
        self._node_map: Dict[str, TreeNode] = {}
        self._build_node_map()

    def _build_node_map(self) -> None:
        """Build map of node_id → node."""
        for node in TreeTraversal.preorder(self.root):
            self._node_map[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        """Get node by ID."""
        return self._node_map.get(node_id)

    def get_clauses(self) -> List[Clause]:
        """Get all clauses in tree."""
        clauses = []
        for node in TreeTraversal.preorder(self.root):
            if node.node_type == "clause" and isinstance(node.value, Clause):
                clauses.append(node.value)
        return clauses

    def get_words(self) -> List[Word]:
        """Get all words in pre-order (reading order)."""
        words = []
        for node in TreeTraversal.preorder(self.root):
            if node.node_type == "word" and isinstance(node.value, Word):
                words.append(node.value)
        return words

    def get_main_clauses(self) -> List[Clause]:
        """Get main clauses (not subordinate)."""
        clauses = self.get_clauses()
        return [c for c in clauses if c.parent_clause_id is None]

    def get_subordinate_clauses(self, parent: Clause) -> List[Clause]:
        """Get subordinate clauses of a parent."""
        clauses = self.get_clauses()
        return [c for c in clauses if c.parent_clause_id == parent.clause_id]

    def find_clauses_by_type(self, clause_type: ClauseType) -> List[Clause]:
        """Find clauses of specific type."""
        return [c for c in self.get_clauses() if c.clause_type == clause_type]

    def find_clause_chain(self, start_clause: Clause) -> List[Clause]:
        """Get chain: main clause → direct subordinates."""
        chain = [start_clause]
        for sub_clause_id in start_clause.subordinate_clause_ids:
            sub_clause = next((c for c in self.get_clauses() if c.clause_id == sub_clause_id), None)
            if sub_clause:
                chain.append(sub_clause)
        return chain

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tree to dictionary."""
        def node_to_dict(node: TreeNode) -> Dict[str, Any]:
            result = {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "metadata": node.metadata,
            }
            if node.value and hasattr(node.value, 'to_dict'):
                result["value"] = node.value.to_dict()
            result["children"] = [node_to_dict(child) for child in node.children]
            return result

        return node_to_dict(self.root)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)
    