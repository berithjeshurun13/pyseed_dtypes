# pyseed_dtypes/graph.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..explict import DTypeLike, get_dtype
from ._tensors import Tensor, NDSparseTensor

__all__ = [
    "GraphKind",
    "EdgeListTensor",
    "AdjacencyTensor",
    "GraphTensor",
]


class GraphKind(Enum):
    GENERIC = auto()
    KNOWLEDGE_GRAPH = auto()  # triples (h, r, t)


@dataclass(frozen=True, slots=True)
class EdgeListTensor:
    """
    Edge list representation.

    Shapes:
      - generic graph: edges shape (E, 2) = (src, dst)
      - weighted: weights shape (E,)
      - knowledge graph: triples shape (E, 3) = (head, relation, tail)

    All indices are int64 ids (you can map strings externally).
    """
    edges: Tensor                     # int64, (E,2) or (E,3)
    weights: Optional[Tensor] = None  # numeric, (E,)
    kind: GraphKind = GraphKind.GENERIC
    num_nodes: Optional[int] = None
    num_relations: Optional[int] = None
    directed: bool = True
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        e = self.edges.numpy()
        if e.ndim != 2 or e.shape[1] not in (2, 3):
            raise ValueError("edges must have shape (E,2) or (E,3)")
        if e.dtype.kind not in ("i", "u"):
            raise TypeError("edges must be integer ids")
        if self.kind == GraphKind.KNOWLEDGE_GRAPH and e.shape[1] != 3:
            raise ValueError("KNOWLEDGE_GRAPH requires edges shape (E,3) = (h,r,t)")

        if self.weights is not None:
            w = self.weights.numpy()
            if w.shape != (e.shape[0],):
                raise ValueError("weights must have shape (E,)")

        if self.num_nodes is not None and self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive if provided")
        if self.num_relations is not None and self.num_relations <= 0:
            raise ValueError("num_relations must be positive if provided")

    @property
    def E(self) -> int:
        return int(self.edges.shape[0])

    @staticmethod
    def from_numpy(
        edges: Any,
        *,
        weights: Any | None = None,
        kind: GraphKind = GraphKind.GENERIC,
        num_nodes: int | None = None,
        num_relations: int | None = None,
        directed: bool = True,
        meta: Dict[str, Any] | None = None,
    ) -> "EdgeListTensor":
        e = np.asarray(edges, dtype=np.int64)
        et = Tensor.from_data(e, dtype=get_dtype(np.dtype(np.int64)))
        wt = None
        if weights is not None:
            w = np.asarray(weights)
            wt = Tensor.from_data(w, dtype=get_dtype(w.dtype))
        return EdgeListTensor(
            edges=et,
            weights=wt,
            kind=kind,
            num_nodes=num_nodes,
            num_relations=num_relations,
            directed=directed,
            meta=meta or {},
        )

    def to_adjacency(self, *, coalesce: bool = True) -> "AdjacencyTensor":
        e = self.edges.numpy()
        if self.kind == GraphKind.KNOWLEDGE_GRAPH:
            # shape = (N, N, R) where coords = (h, t, r)
            if self.num_nodes is None or self.num_relations is None:
                raise ValueError("Knowledge graph requires num_nodes and num_relations for adjacency")
            h = e[:, 0]; r = e[:, 1]; t = e[:, 2]
            coords = np.vstack([h, t, r]).astype(np.int64)
            shape = (self.num_nodes, self.num_nodes, self.num_relations)
        else:
            if self.num_nodes is None:
                raise ValueError("Generic graph requires num_nodes for adjacency")
            src = e[:, 0]; dst = e[:, 1]
            coords = np.vstack([src, dst]).astype(np.int64)
            shape = (self.num_nodes, self.num_nodes)

        if self.weights is None:
            data = np.ones((e.shape[0],), dtype=np.float32)
            dt = get_dtype(np.dtype(np.float32))
        else:
            data = np.asarray(self.weights.numpy())
            dt = get_dtype(data.dtype)

        sp = NDSparseTensor.from_coo(coords, data, shape=shape, dtype=dt, coalesce=coalesce, check_bounds=True)
        return AdjacencyTensor(sparse=sp, kind=self.kind, num_nodes=self.num_nodes, num_relations=self.num_relations,
                               directed=self.directed, meta=dict(self.meta))

@dataclass(frozen=True, slots=True)
class AdjacencyTensor:
    """
    Sparse adjacency tensor using NDSparseTensor.

    - Generic graph: shape (N, N), coords=(src,dst)
    - Knowledge graph: shape (N, N, R), coords=(head, tail, relation)
      where each triple (h,r,t) encodes an edge labeled by relation.
    """
    sparse: NDSparseTensor
    kind: GraphKind = GraphKind.GENERIC
    num_nodes: Optional[int] = None
    num_relations: Optional[int] = None
    directed: bool = True
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        shp = self.sparse.shape
        if self.kind == GraphKind.GENERIC:
            if len(shp) != 2 or shp[0] != shp[1]:
                raise ValueError("GENERIC adjacency must have shape (N,N)")
        else:
            if len(shp) != 3 or shp[0] != shp[1]:
                raise ValueError("KNOWLEDGE_GRAPH adjacency must have shape (N,N,R)")

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.sparse.shape)

    @property
    def nnz(self) -> int:
        return self.sparse.nnz

    def to_dense(self) -> np.ndarray:
        # beware: can be huge; but datatype allows it
        return self.sparse.numpy()

    def to_edge_list(self) -> EdgeListTensor:
        coords, data = self.sparse.coords, self.sparse.data
        if self.kind == GraphKind.GENERIC:
            edges = np.stack([coords[0], coords[1]], axis=1)
            return EdgeListTensor.from_numpy(edges, weights=data, kind=self.kind, num_nodes=self.shape[0],
                                             directed=self.directed, meta=dict(self.meta))
        else:
            edges = np.stack([coords[0], coords[2], coords[1]], axis=1)  # (h,r,t)
            return EdgeListTensor.from_numpy(edges, weights=data, kind=self.kind, num_nodes=self.shape[0],
                                             num_relations=self.shape[2], directed=self.directed, meta=dict(self.meta))

    def symmetrize(self) -> "AdjacencyTensor":
        """
        Mirror edges for undirected graphs by swapping src/dst (or h/t).
        If duplicates occur, COO coalesce will sum them (common sparse behavior).
        """
        if not self.directed:
            return self

        c = self.sparse.coords
        d = self.sparse.data
        c2 = c.copy()
        c2[0, :], c2[1, :] = c[1, :], c[0, :]
        new_coords = np.concatenate([c, c2], axis=1)
        new_data = np.concatenate([d, d], axis=0)

        sp2 = NDSparseTensor.from_coo(new_coords, new_data, shape=self.sparse.shape,
                                      dtype=self.sparse.dtype, coalesce=True, check_bounds=False)
        return AdjacencyTensor(sparse=sp2, kind=self.kind, num_nodes=self.num_nodes, num_relations=self.num_relations,
                               directed=False, meta=dict(self.meta))

    def __repr__(self) -> str:
        return f"AdjacencyTensor(shape={self.shape}, nnz={self.nnz}, kind={self.kind.name}, directed={self.directed})"

@dataclass(frozen=True, slots=True)
class GraphTensor:
    """
    High-level graph container datatype.

    Holds either edge-list or adjacency plus optional node/edge feature tensors.
    """
    adjacency: Optional[AdjacencyTensor] = None
    edges: Optional[EdgeListTensor] = None

    node_features: Optional[Tensor] = None   # e.g., (N, F)
    edge_features: Optional[Tensor] = None   # e.g., (E, F)

    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if (self.adjacency is None) == (self.edges is None):
            raise ValueError("GraphTensor requires exactly one of adjacency or edges")

    @property
    def kind(self) -> GraphKind:
        return self.adjacency.kind if self.adjacency is not None else self.edges.kind

    @property
    def num_nodes(self) -> Optional[int]:
        if self.adjacency is not None:
            return int(self.adjacency.shape[0])
        return self.edges.num_nodes

    def to_adjacency(self) -> AdjacencyTensor:
        return self.adjacency if self.adjacency is not None else self.edges.to_adjacency()

    def to_edges(self) -> EdgeListTensor:
        return self.edges if self.edges is not None else self.adjacency.to_edge_list()

    def __repr__(self) -> str:
        rep = "adjacency" if self.adjacency is not None else "edges"
        return f"GraphTensor({rep}, kind={self.kind.name}, num_nodes={self.num_nodes})"
