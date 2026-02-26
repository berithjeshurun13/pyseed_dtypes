# pyseed_dtypes/explicit.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Final, Mapping, TypeAlias, Union,
    List, Callable, Optional, Any, Dict,
    Generator, Set
)
from collections import Counter
from pyseed_logger import log
from enum import auto, Enum, Flag
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import uuid, hashlib, json
sha1 = lambda x : hashlib.sha1(str(x).encode()).hexdigest()
__all__ = [
    "DType",
    "DTypeLike",
    "Float16", "Float32", "Float64",
    "Int8", "Int16", "Int32", "Int64",
    "Bool",
    "DTYPES",
    "get_dtype",
    "register_dtype",
    "PathLike",
    "TreeNode",
    "TreeTraversal",
    "Leaf",
    "Branch",
    "Tree",
    "Node",
    "Net",
    "GraphObject",
    "ChunkSize",
    "LBookFileFormat",

    "sha1",
    "table_dnodehashlist",
    "LoggingFlags",
    "ErrorLevel",
    "FileType",
    "ProjectNodeType",
    "GlobalException",
    "ProjectNode",
    "DNodeHashList",
    "_node_to_dict",
    "_validate_node",
    "ThemeData"

]

# Use NumPy dtype.kind codes for consistency:
# 'b' bool, 'i' signed int, 'u' unsigned int, 'f' float, 'c' complex, ...
# (documented by NumPy)
Kind: TypeAlias = str

def gen_id() -> str:
    return str(uuid.uuid4())[:8]

def _node_to_dict(n: ProjectNode) -> Dict[str, Any]:
    return {
        "id": n.id,
        "text": n.text,
        "icon": n.icon,
        "children": [_node_to_dict(c) for c in n.children],
    }

def _validate_node(n : ProjectNode) -> bool :
    if table_dnodehashlist.contains(n.hash) :
        return False
    
    table_dnodehashlist.add(n.hash)
    return True



class LoggingFlags(Flag) :
    LOW  = auto()
    MEDIUM = auto()
    HIGH = auto()

    VERBOSE = LOW

class ErrorLevel(Flag) :
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    EXCEPTION = auto()
    DEBUG = auto()

class FileType(Flag) :
    
    HtmlFile  = auto()
    PdfFile   = auto()
    CsvFile   = auto()
    LhbkFile  = auto()
    MarkDownFile = auto()


class ProjectNodeType(Flag) :
    Scriptures = auto()
    Workspaces = auto()
    Seromons   = auto()

@dataclass
class ProjectNode:
    id: str
    text: str
    icon: str = "file"   # key into ICONS map in JS
    children: List["ProjectNode"] = field(default_factory=list)

    _cached_hash: str = field(default="", repr=False)

    @property
    def hash(self) -> str:
        if self._cached_hash:
            return self._cached_hash

        self._cached_hash = sha1(
            f"{self.icon}/{self.text.upper()}"
        )
        return self._cached_hash

@dataclass
class ThemeData:
    COL_BG_PRIMARY : str = "#0B0B10"
    COL_BG_SECONDARY : str = "#141422"
    COL_SURFACE : str = "#1E1B2E"
    COL_PRIMARY : str = "#7A2CFF"
    COL_ACCENT : str = "#FF2BD6"
    COL_ACTIVE : str = "#C9A7FF"
    COL_TEXT_MAIN : str = "#F2F3FF"
    COL_TEXT_MUTED : str = "#B8B6CC"
    COL_HIGHLIGHT : str = COL_ACTIVE

    COL_GRAD_BEAM : str = json.dumps({"from": "#FF2BD6","to": "#7A2CFF"})
    COL_GRAD_NIGHT : str =  json.dumps({"from": "#0B0B10","to": "#141422"})

    VAR_DEG : int = 125



@dataclass
class GlobalException :
    text : str
    level : ErrorLevel
    message : str
    errorcode : str
    debug : str = ""

@dataclass
class DNodeHashList:
    """Registry of node hashes to prevent duplicates."""
    data: Set[str] = field(default_factory=set)

    def contains(self, node_hash: str) -> bool:
        return node_hash in self.data

    def add(self, node_hash: str) -> bool:
        """Add hash if not present."""
        if node_hash in self.data:
            return False
        self.data.add(node_hash)
        return True

    def remove(self, node_hash: str) -> bool:
        """Remove hash if present."""
        if node_hash not in self.data:
            return False
        self.data.remove(node_hash)
        return True

    def replace(self, old_hash: str, new_hash: str) -> bool:
        """
        Replace old hash with new hash.

        Returns:
            True  -> replaced successfully
            False -> old hash missing OR new hash already exists
        """
        if old_hash not in self.data:
            return False

        if new_hash in self.data and new_hash != old_hash:
            return False

        self.data.remove(old_hash)
        self.data.add(new_hash)
        return True

@dataclass(frozen=True, slots=True)
class Author :
    name : str
    birth_year : int
    death_year : int

class GuetendexFiles(Enum) :
    HTML = "text/html"
    JPEG = "image/jpeg"
    XML = "application/rdf+xml"
    EPUB = "application/epub+zip"
    EBOOK = "application/x-mobipocket-ebook"
    TEXT = "text/plain"
    UNKNOWN = "unknown"

class ChunkSize(Flag) :
    Bit4 = 4
    Bit8 = 8
    Bit12 = 12
    Bit16 = 16
    Bit24 = 24
    Bit32 = 32
    Bit52 = 52
    Bit64 = 64
    Bit72 = 72
    Bit80 = 80
    Bit94 = 94
    Bit128 = 128

class LBookFileFormat(Flag) :
    Sqlite3 = auto()
    Json    = auto()
    Csv     = auto()
    Xml     = auto()

@dataclass(frozen=True, slots=True)
class GutendexBook :
    book : str
    authors : List[Author]
    sources : Dict[str, GuetendexFiles]
    

@dataclass(frozen=True, slots=True)
class DType:
    name: str
    np_dtype: np.dtype
    kind: Kind
    bits: int

    @property
    def itemsize(self) -> int:
        return self.np_dtype.itemsize

    def is_bool(self) -> bool:
        return self.kind == "b"

    def is_int(self) -> bool:
        return self.kind == "i"

    def is_float(self) -> bool:
        return self.kind == "f"

    def as_numpy(self) -> np.dtype:
        return self.np_dtype

    def __repr__(self) -> str:
        return f"DType(name={self.name!r})"


# Canonical dtypes
Float16: Final = DType("float16", np.dtype(np.float16), "f", 16)
Float32: Final = DType("float32", np.dtype(np.float32), "f", 32)
Float64: Final = DType("float64", np.dtype(np.float64), "f", 64)

Int8:   Final = DType("int8",  np.dtype(np.int8),  "i", 8)
Int16:  Final = DType("int16", np.dtype(np.int16), "i", 16)
Int32:  Final = DType("int32", np.dtype(np.int32), "i", 32)
Int64:  Final = DType("int64", np.dtype(np.int64), "i", 64)

Bool:   Final = DType("bool",  np.dtype(np.bool_), "b", 1)



DTYPES: Final[dict[str, DType]] = {
    d.name: d
    for d in (Float16, Float32, Float64, Int8, Int16, Int32, Int64, Bool)
}

_ALIASES: Final[dict[str, str]] = {
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "bool_": "bool",
    "boolean": "bool",
}

_NUMPY_TO_PYSEED: Final[dict[np.dtype, DType]] = {
    d.np_dtype: d for d in DTYPES.values()
}

DTypeLike: TypeAlias = DType | str | np.dtype | type
PathLike = Union[str, Path]


def register_dtype(dtype: DType, *, aliases: Mapping[str, str] | None = None) -> None:
    DTYPES[dtype.name] = dtype
    _NUMPY_TO_PYSEED[dtype.np_dtype] = dtype
    if aliases:
        for k, v in aliases.items():
            _ALIASES[k] = v


def get_dtype(x: DTypeLike) -> DType:
    """
    Parse dtype from:
    - DType: returned as-is
    - str: name or alias (e.g., "float32", "f32")
    - np.dtype or numpy scalar type (e.g., np.float32)
    """
    if isinstance(x, DType):
        return x

    if isinstance(x, str):
        key = x.strip().lower()
        key = _ALIASES.get(key, key)
        try:
            return DTYPES[key]
        except KeyError:
            raise KeyError(f"Unknown dtype name: {x!r}")

    # np.dtype(...) accepts dtype objects and scalar types
    dt = np.dtype(x)
    try:
        return _NUMPY_TO_PYSEED[dt]
    except KeyError:
        raise TypeError(f"Unsupported numpy dtype: {dt!r}")

@dataclass
class TreeNode:
    """Generic tree node for syntax structures."""
    node_id: str
    node_type: str  # "sentence", "clause", "phrase", "word"
    value: Any  # The actual object (Clause, Phrase, Word)
    children: List["TreeNode"] = field(default_factory=list)
    parent: Optional["TreeNode"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: "TreeNode") -> None:
        """Add child node."""
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "TreeNode") -> None:
        """Remove child node."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return len(self.children) == 0

    def is_root(self) -> bool:
        """Check if node is root."""
        return self.parent is None

    def depth(self) -> int:
        """Get depth in tree."""
        if self.is_root():
            return 0
        return 1 + self.parent.depth()

    def height(self) -> int:
        """Get height of subtree."""
        if self.is_leaf():
            return 0
        return 1 + max(child.height() for child in self.children)

    def path_to_root(self) -> List["TreeNode"]:
        """Get path from this node to root."""
        path = [self]
        current = self
        while current.parent:
            path.insert(0, current.parent)
            current = current.parent
        return path

    def sibling(self, offset: int = 1) -> Optional["TreeNode"]:
        """Get sibling at offset."""
        if not self.parent:
            return None
        try:
            idx = self.parent.children.index(self)
            return self.parent.children[idx + offset]
        except (ValueError, IndexError):
            return None

    def clone(self) -> "TreeNode":
        """Deep clone subtree."""
        new_node = TreeNode(
            node_id=self.node_id,
            node_type=self.node_type,
            value=self.value,
            metadata=self.metadata.copy()
        )
        for child in self.children:
            new_node.add_child(child.clone())
        return new_node


class TreeTraversal:
    """Tree traversal algorithms."""

    @staticmethod
    def preorder(node: TreeNode) -> List[TreeNode]:
        """Pre-order traversal: parent before children."""
        result = [node]
        for child in node.children:
            result.extend(TreeTraversal.preorder(child))
        return result

    @staticmethod
    def postorder(node: TreeNode) -> List[TreeNode]:
        """Post-order traversal: children before parent."""
        result = []
        for child in node.children:
            result.extend(TreeTraversal.postorder(child))
        result.append(node)
        return result

    @staticmethod
    def levelorder(node: TreeNode) -> List[TreeNode]:
        """Level-order traversal: breadth-first."""
        result = []
        queue = [node]
        while queue:
            current = queue.pop(0)
            result.append(current)
            queue.extend(current.children)
        return result

    @staticmethod
    def find_all(node: TreeNode, predicate: Callable[[TreeNode], bool]) -> List[TreeNode]:
        """Find all nodes matching predicate."""
        result = []
        for n in TreeTraversal.preorder(node):
            if predicate(n):
                result.append(n)
        return result

    @staticmethod
    def lowest_common_ancestor(node1: TreeNode, node2: TreeNode) -> Optional[TreeNode]:
        """Find lowest common ancestor of two nodes."""
        path1 = set(n.node_id for n in node1.path_to_root())
        current = node2
        while current:
            if current.node_id in path1:
                return current
            current = current.parent
        return None

@dataclass
class Leaf:
    data: Any
    parent: Optional["Branch"] = field(default=None, repr=False)
    id: str = field(default_factory=gen_id)

    def __repr__(self):
        return f"Leaf(id={self.id[:6]}, data={self.data!r})"

@dataclass
class Branch:
    data: Optional[Any] = None
    leaves: List[Leaf] = field(default_factory=list)
    parent: Optional["Tree"] = field(default=None, repr=False)
    id: str = field(default_factory=gen_id)

    def add_leaf(self, leaf_data: Any) -> Leaf:
        leaf = Leaf(data=leaf_data, parent=self)
        self.leaves.append(leaf)
        return leaf

    def walk_leaves(self) -> Generator[Leaf, None, None]:
        for leaf in self.leaves:
            yield leaf

    def __repr__(self):
        return f"Branch(id={self.id[:6]}, data={self.data!r}, leaves={self.leaves})"

@dataclass
class Tree:
    data: Optional[Any] = None
    branches: List[Branch] = field(default_factory=list)
    parent: Optional["Node"] = field(default=None, repr=False)
    id: str = field(default_factory=gen_id)

    def add_branch(self, branch_data: Any = None) -> Branch:
        branch = Branch(data=branch_data, parent=self)
        self.branches.append(branch)
        return branch

    def walk_branches(self) -> Generator[Branch, None, None]:
        for branch in self.branches:
            yield branch
            yield from branch.walk_leaves()

    def __repr__(self):
        return f"Tree(id={self.id[:6]}, data={self.data!r}, branches={self.branches})"

@dataclass
class Node:
    data: Optional[Any] = None
    trees: List[Tree] = field(default_factory=list)
    parent: Optional["Net"] = field(default=None, repr=False)
    id: str = field(default_factory=gen_id)

    def add_tree(self, tree_data: Any = None) -> Tree:
        tree = Tree(data=tree_data, parent=self)
        self.trees.append(tree)
        return tree

    def walk_trees(self) -> Generator[Tree, None, None]:
        for tree in self.trees:
            yield tree
            yield from tree.walk_branches()

    def __repr__(self):
        return f"Node(id={self.id[:6]}, data={self.data!r}, trees={self.trees})"

@dataclass
class Net:
    nodes: List[Node] = field(default_factory=list)

    def add_node(self, node_data: Any = None) -> Node:
        node = Node(data=node_data, parent=self)
        self.nodes.append(node)
        return node

    def walk_nodes(self) -> Generator[Node, None, None]:
        for node in self.nodes:
            yield node
            yield from node.walk_trees()

    def find(self, predicate: Callable[[Any], bool]) -> Optional[Any]:
        for node in self.walk_nodes():
            if predicate(node): return node
            for tree in node.trees:
                if predicate(tree): return tree
                for branch in tree.branches:
                    if predicate(branch): return branch
                    for leaf in branch.leaves:
                        if predicate(leaf): return leaf
        return None

    def __repr__(self):
        return f"Net(nodes={self.nodes})"

class GraphObject(object) :
    def __init__(self, value, **kwargs):
        super().__init__()

        self.__VAL__ = value
        self.kwargs = kwargs


    @property
    def value(self) : return self.__VAL__

    def run(self) -> None : 
        plt.clf()
        self.setsize()
        self.__build__()

    
    def __build__(self) -> None :
        graph_work = self.kwargs.get('_graph_object_type', None)
        if graph_work == 'WordPopulationAnalyzer' :
            mode = self.kwargs.get('use', None)
            if mode == 'chapter' or mode == 'chapters' :
                b_type = self.kwargs.get('b_type', None)
                if b_type == 'focused' :
                    focused_chapters = sorted(int(entry['chapter']) for entry in self.value if entry['book'] == self.kwargs.get('focus', 'God'))
                    plt.plot(focused_chapters, range(1, len(focused_chapters) + 1), marker='o')
                    plt.title(f"Progression of Chapter Mentions in {self.kwargs.get('focus', None)}")
                    plt.xlabel("Chapter Number")
                    plt.ylabel("Order Appeared")
                    plt.grid(True)
                    log.success(f'Graph model Generation successfull for {graph_work}')
                else :
                    counts = Counter(entry['book'] for entry in self.value)
                    plt.bar(counts.keys(), counts.values(), color='skyblue')
                    plt.title("Chapter Frequency per Book")
                    plt.xlabel("Book")
                    plt.ylabel("Number of Chapters")
                    log.success(f'Graph model Generation successfull for {graph_work}')



    def to_image(self, path : str, dpi : int = 300) -> str :
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        return path

    def setsize(self, size : tuple[int, int] = (8,5)) :
        plt.figure(figsize=size)
        return list(size)
    
# Runtime Variables
table_dnodehashlist : DNodeHashList = DNodeHashList()