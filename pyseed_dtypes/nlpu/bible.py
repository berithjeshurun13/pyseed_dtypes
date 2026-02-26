from dataclasses import dataclass, field
from typing import (
    Optional, List, Tuple, Union, Dict, Any
)
from ._nlpu import (
    SyntaxNode, ClauseType
)
import hashlib, re, json
from enum import Enum


__all__ = [
    "Period",
    "DiachronicPath",
    "WordAnchor",
    "ClauseAnchor",
    "VerseAnchor",
    "Citation",
    "SyntaxTree",
    "BibleReference",
    "TextualVariant",
    "ManuscriptReference",
    "Abbreviation",
    "AutoReferencer"
]


# ---------------------------
# Canonical mappings
# ---------------------------
_LOWER_ABBR = {
        "gen": "Genesis",
        "exod": "Exodus",
        "lev": "Leviticus",
        "num": "Numbers",
        "deut": "Deuteronomy",
        "josh": "Joshua",
        "judg": "Judges",
        "ruth": "Ruth",
        "1sam": "1 Samuel",
        "2sam": "2 Samuel",
        "1kgs": "1 Kings",
        "2kgs": "2 Kings",
        "1chr": "1 Chronicles",
        "2chr": "2 Chronicles",
        "ezra": "Ezra",
        "neh": "Nehemiah",
        "esth": "Esther",
        "job": "Job",
        "ps": "Psalms",
        "prov": "Proverbs",
        "eccl": "Ecclesiastes",
        "song": "Song of Songs",
        "isa": "Isaiah",
        "jer": "Jeremiah",
        "lam": "Lamentations",
        "ezek": "Ezekiel",
        "dan": "Daniel",
        "hos": "Hosea",
        "joel": "Joel",
        "amos": "Amos",
        "obad": "Obadiah",
        "jonah": "Jonah",
        "mic": "Micah",
        "nah": "Nahum",
        "hab": "Habakkuk",
        "zeph": "Zephaniah",
        "hag": "Haggai",
        "zech": "Zechariah",
        "mal": "Malachi",
        "matt": "Matthew",
        "mark": "Mark",
        "luke": "Luke",
        "john": "John",
        "acts": "Acts",
        "rom": "Romans",
        "1cor": "1 Corinthians",
        "2cor": "2 Corinthians",
        "gal": "Galatians",
        "eph": "Ephesians",
        "phil": "Philippians",
        "col": "Colossians",
        "1thess": "1 Thessalonians",
        "2thess": "2 Thessalonians",
        "1tim": "1 Timothy",
        "2tim": "2 Timothy",
        "titus": "Titus",
        "phlm": "Philemon",
        "heb": "Hebrews",
        "jas": "James",
        "1pet": "1 Peter",
        "2pet": "2 Peter",
        "1john": "1 John",
        "2john": "2 John",
        "3john": "3 John",
        "jude": "Jude",
        "rev": "Revelation",
}

_UPPER_ABBR = {
        "GEN": "Genesis",
        "EXO": "Exodus",
        "LEV": "Leviticus",
        "NUM": "Numbers",
        "DEU": "Deuteronomy",
        "JOS": "Joshua",
        "JDG": "Judges",
        "RUT": "Ruth",
        "1SA": "1 Samuel",
        "2SA": "2 Samuel",
        "1KI": "1 Kings",
        "2KI": "2 Kings",
        "1CH": "1 Chronicles",
        "2CH": "2 Chronicles",
        "EZR": "Ezra",
        "NEH": "Nehemiah",
        "EST": "Esther",
        "JOB": "Job",
        "PSA": "Psalms",
        "PRO": "Proverbs",
        "ECC": "Ecclesiastes",
        "SOS": "Song of Songs",
        "ISA": "Isaiah",
        "JER": "Jeremiah",
        "LAM": "Lamentations",
        "EZE": "Ezekiel",
        "DAN": "Daniel",
        "HOS": "Hosea",
        "JOE": "Joel",
        "AMO": "Amos",
        "OBA": "Obadiah",
        "JON": "Jonah",
        "MIC": "Micah",
        "NAH": "Nahum",
        "HAB": "Habakkuk",
        "ZEP": "Zephaniah",
        "HAG": "Haggai",
        "ZEC": "Zechariah",
        "MAL": "Malachi",
        "MAT": "Matthew",
        "MAR": "Mark",
        "LUK": "Luke",
        "JOH": "John",
        "ACT": "Acts",
        "ROM": "Romans",
        "1CO": "1 Corinthians",
        "2CO": "2 Corinthians",
        "GAL": "Galatians",
        "EPH": "Ephesians",
        "PHP": "Philippians",
        "COL": "Colossians",
        "1TH": "1 Thessalonians",
        "2TH": "2 Thessalonians",
        "1TI": "1 Timothy",
        "2TI": "2 Timothy",
        "TIT": "Titus",
        "PHM": "Philemon",
        "HEB": "Hebrews",
        "JAM": "James",
        "1PE": "1 Peter",
        "2PE": "2 Peter",
        "1JO": "1 John",
        "2JO": "2 John",
        "3JO": "3 John",
        "JDE": "Jude",
        "REV": "Revelation",
}

# reverse maps (full -> abbr)
_FULL_TO_LOWER = {v.lower(): k for k, v in _LOWER_ABBR.items()}
_FULL_TO_UPPER = {v.lower(): k for k, v in _UPPER_ABBR.items()}


class Period(Enum):
    HEBREW_BIBLE = "hebrew_bible"
    ARAMAIC_SECTIONS = "aramaic"
    SEPTUAGINT = "lxx"
    NT_PAUL = "nt_paul"
    NT_GOSPELS = "nt_gospels"
    NT_JOHN = "nt_john"


@dataclass
class BibleReference:
    """Normalized Bible verse reference."""
    book: str  # e.g., "John"
    chapter: int
    verse: int
    subverse: Optional[str] = None  # For multi-part verses like 3:16a

    def __hash__(self):
        return hash((self.book, self.chapter, self.verse, self.subverse))

    def __eq__(self, other):
        if not isinstance(other, BibleReference):
            return False
        return (self.book == other.book and 
                self.chapter == other.chapter and 
                self.verse == other.verse and 
                self.subverse == other.subverse)

    def to_string(self) -> str:
        """Format as 'John 3:16' or 'John 3:16a'."""
        result = f"{self.book} {self.chapter}:{self.verse}"
        if self.subverse:
            result += self.subverse
        return result

    @staticmethod
    def from_string(ref: str) -> "BibleReference":
        """Parse from string like 'John 3:16' or 'John 3:16a'."""
        # Simple parser; production version would be more robust
        parts = ref.strip().split()
        book = parts[0]
        verse_part = parts[1]

        chapter, verse_and_sub = verse_part.split(":")
        verse = int(verse_and_sub.rstrip("abc"))
        subverse = verse_and_sub[len(str(verse)):] if verse_and_sub != str(verse) else None

        return BibleReference(book, int(chapter), verse, subverse)
    
@dataclass
class ManuscriptReference:
    """Reference to a physical manuscript location."""
    manuscript_id: str  # e.g., "P45", "Codex Sinaiticus"
    folio: Optional[str] = None  # Page/column number
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    pixel_coords: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) for image anchoring
    image_uri: Optional[str] = None  # URL to image

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manuscript_id": self.manuscript_id,
            "folio": self.folio,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "pixel_coords": self.pixel_coords,
            "image_uri": self.image_uri,
        }

@dataclass
class TextualVariant:
    """Record of textual variation across manuscripts."""
    variant_id: str
    reference: BibleReference
    word_id: str

    # Variants
    main_text_reading: str  # Typically NA/UBS text
    variant_readings: Dict[str, List[str]]  # {reading: [manuscript_ids]}

    # Metadata
    significance: float  # 0-1, importance for interpretation
    commentary: Optional[str] = None

@dataclass
class DiachronicPath:
    """Evolution of a semantic concept over time."""
    path_id: str
    concept_id: str  # The theological concept
    
    # Timeline
    periods: List[Period]
    lemmas_by_period: Dict[Period, List[str]]
    
    # Metadata
    description: str = ""
    examples: List[str] = None  # Sample verses showing the evolution

@dataclass
class WordAnchor:
    """Canonical, deterministic identifier for a word occurrence."""
    anchor_id: str  # SHA256 of canonical string
    manuscript_id: str
    book: str
    chapter: int
    verse: int
    word_index: int  # 0-based within verse
    
    # Image region (optional)
    folio_id: Optional[str] = None
    image_x: Optional[int] = None
    image_y: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    
    @staticmethod
    def compute_id(
        manuscript_id: str, book: str, chapter: int, verse: int, word_index: int
    ) -> str:
        """Deterministic ID computation."""
        canonical = f"{manuscript_id}|{book}|{chapter}|{verse}|{word_index}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    @classmethod
    def create(
        cls,
        manuscript_id: str,
        book: str,
        chapter: int,
        verse: int,
        word_index: int,
        folio_id: Optional[str] = None,
        image_coords: Optional[Tuple[int, int, int, int]] = None,
    ) -> "WordAnchor":
        """Factory method."""
        anchor_id = cls.compute_id(manuscript_id, book, chapter, verse, word_index)
        x, y, w, h = image_coords if image_coords else (None, None, None, None)
        return cls(
            anchor_id=anchor_id,
            manuscript_id=manuscript_id,
            book=book,
            chapter=chapter,
            verse=verse,
            word_index=word_index,
            folio_id=folio_id,
            image_x=x,
            image_y=y,
            image_width=w,
            image_height=h,
        )

@dataclass
class ClauseAnchor:
    """Stable ID for a clause (derived from word anchors)."""
    anchor_id: str  # Derived from word_anchor_ids
    word_anchor_ids: List[str]  # In order
    tree_id: str
    clause_id: str
    
    @staticmethod
    def compute_id(word_anchor_ids: List[str]) -> str:
        """Stable ID from contained words."""
        canonical = "|".join(sorted(word_anchor_ids))  # Order-independent for some use cases
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

@dataclass
class VerseAnchor:
    """Reference to a verse (higher-level abstraction)."""
    anchor_id: str
    manuscript_id: str
    book: str
    chapter: int
    verse: int
    word_anchor_ids: List[str]  # Ordered list of words in this verse
    
    @staticmethod
    def compute_id(manuscript_id: str, book: str, chapter: int, verse: int) -> str:
        canonical = f"{manuscript_id}|{book}|{chapter}|{verse}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

@dataclass
class Citation:
    """Stable, human-readable citation."""
    reference: str  # e.g., "Matt 5:3-5", "John 1:1 (word 5)"
    anchors: List[Union[WordAnchor, ClauseAnchor, VerseAnchor]]
    
    @staticmethod
    def parse(citation_str: str) -> "Citation":
        """Parse human-readable citations."""
        # Matt 5:3-5 (word 5)
        # Mark 1:1
        # Luke 10:25-37 (clauses)
        pattern = r"^(\w+)\s+(\d+):(\d+)(?:-(\d+))?\s*(?:\(([^)]+)\))?$"
        match = re.match(pattern, citation_str)
        
        if not match:
            raise ValueError(f"Invalid citation format: {citation_str}")
        
        book, chapter, verse_start, verse_end, specifier = match.groups()
        verse_end = verse_end or verse_start
        
        return Citation(
            reference=citation_str,
            anchors=[]  # Would be populated from DB lookup
        )
    
@dataclass
class SyntaxTree:
    """A complete syntax tree for a sentence/pericope."""
    tree_id: str  # Stable UUID
    manuscript_id: str
    book: str
    chapter: int
    verse_start: int
    verse_end: int
    
    root_node_id: str
    nodes: Dict[str, SyntaxNode]  # node_id -> SyntaxNode
    
    # Metadata
    language: str  # "grc", "he", etc.
    created_at: str
    modified_at: str
    source: str  # "manual", "auto-parsed"
    
    def to_json(self) -> str:
        """Serialize to JSON for storage."""
        return json.dumps({
            "tree_id": self.tree_id,
            "manuscript_id": self.manuscript_id,
            "book": self.book,
            "chapter": self.chapter,
            "verse_start": self.verse_start,
            "verse_end": self.verse_end,
            "root_node_id": self.root_node_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "language": self.language,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "source": self.source,
        })
    
    @staticmethod
    def from_json(s: str) -> "SyntaxTree":
        """Deserialize from JSON."""
        d = json.loads(s)
        return SyntaxTree(
            tree_id=d["tree_id"],
            manuscript_id=d["manuscript_id"],
            book=d["book"],
            chapter=d["chapter"],
            verse_start=d["verse_start"],
            verse_end=d["verse_end"],
            root_node_id=d["root_node_id"],
            nodes={nid: SyntaxNode.from_dict(node_d) for nid, node_d in d["nodes"].items()},
            language=d["language"],
            created_at=d["created_at"],
            modified_at=d["modified_at"],
            source=d["source"],
        )
    
    def get_clause_by_id(self, node_id: str) -> Optional[SyntaxNode]:
        """Retrieve a clause by ID."""
        return self.nodes.get(node_id)
    
    def get_clauses_of_type(self, clause_type: ClauseType) -> List[SyntaxNode]:
        """Find all clauses of a specific type."""
        return [n for n in self.nodes.values() if n.clause_type == clause_type]
    
    def get_path_to_root(self, node_id: str) -> List[str]:
        """Get ancestor chain from node to root."""
        path = [node_id]
        current = node_id
        while current in self.nodes and self.nodes[current].parent_id:
            current = self.nodes[current].parent_id
            path.append(current)
        return path
    
    def get_descendants(self, node_id: str) -> List[SyntaxNode]:
        """Get all descendants of a node."""
        if node_id not in self.nodes:
            return []
        node = self.nodes[node_id]
        descendants = []
        for child_id in node.children_ids:
            descendants.append(self.nodes[child_id])
            descendants.extend(self.get_descendants(child_id))
        return descendants
    

class Abbreviation:
    def __init__(self, value: str) -> None:
        self._value = value.strip()

    def flush(self, *, capital: bool = False) -> str:
        """
        Converts:
        - abbreviation -> full name
        - full name -> abbreviation

        capital=True => return OSIS-style uppercase abbreviation
        capital=False => return lowercase abbreviation or full name
        """

        key = self._value.strip()
        if key.lower() in _LOWER_ABBR:
            return _LOWER_ABBR[key.lower()]
        if key.upper() in _UPPER_ABBR:
            return _UPPER_ABBR[key.upper()]
        full_key = key.lower()
        if capital:
            if full_key in _FULL_TO_UPPER:
                return _FULL_TO_UPPER[full_key]
        else:
            if full_key in _FULL_TO_LOWER:
                return _FULL_TO_LOWER[full_key]
        return self._value

    def __str__(self) -> str:
        return self._value

class AutoReferencer:
    __slots__ = (
        "__category",
        "__book",
        "__chapter",
        "__verse",
        "__result",
    )

    def __init__(self, ref: str):
        self.__category: Optional[str] = None
        self.__book: Optional[str] = None
        self.__chapter: Optional[int] = None
        self.__verse: Optional[List[int]] = None
        self.__result: bool = False

        self.__process(ref.strip())

    def __process(self, ref: str) -> None:
        """
        Parses Biblical references:
        - Simple:     Jude 1:2
        - Range:      Jude 1:2-6
        - Selective:  Jude 1:2,4,7
        """

        patterns = {
            "range": re.compile(
                r"^(?P<book>[A-Za-z0-9\s]+?)\s+"
                r"(?P<chapter>\d+):(?P<start>\d+)-(?P<end>\d+)$",
                re.IGNORECASE,
            ),
            "selective": re.compile(
                r"^(?P<book>[A-Za-z0-9\s]+?)\s+"
                r"(?P<chapter>\d+):(?P<verses>\d+(?:,\d+)*)$",
                re.IGNORECASE,
            ),
            "simple": re.compile(
                r"^(?P<book>[A-Za-z0-9\s]+?)\s+"
                r"(?P<chapter>\d+):(?P<verse>\d+)$",
                re.IGNORECASE,
            ),
        }

        for category, pattern in patterns.items():
            match = pattern.match(ref)
            if not match:
                continue

            self.__category = category
            self.__book = " ".join(match.group("book").split())
            self.__chapter = int(match.group("chapter"))

            if category == "simple":
                self.__verse = [int(match.group("verse"))]

            elif category == "range":
                start = int(match.group("start"))
                end = int(match.group("end"))
                if end < start:
                    return
                self.__verse = list(range(start, end + 1))

            elif category == "selective":
                verses = [int(v) for v in match.group("verses").split(",")]
                if not verses:
                    return
                self.__verse = verses

            self.__result = True
            return

        self.__result = False

    # --------- Public API ---------

    @property
    def result(self) -> bool:
        return self.__result

    @property
    def category(self) -> Optional[str]:
        return self.__category

    @property
    def book(self) -> Optional[str]:
        return self.__book

    @property
    def chapter(self) -> Optional[int]:
        return self.__chapter

    @property
    def verse(self) -> Optional[List[int]]:
        return self.__verse

    def __bool__(self) -> bool:
        return self.__result

    def __repr__(self) -> str:
        if not self:
            return "<Reference invalid>"
        return f"<Reference {self.book} {self.chapter}:{self.verse}>"


