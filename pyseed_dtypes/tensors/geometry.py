# pyseed_dtypes/geometry.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..explict import DType, DTypeLike, get_dtype
from ._tensors import Tensor

__all__ = [
    "AxisOrder",
    "PointCloud",
    "MeshTensor",
    "TransformTensor",
    "BBoxGTensor",
]


class AxisOrder(Enum):
    XYZ = auto()


def _as_float32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def _as_int64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class TransformTensor:
    """
    Homogeneous transform(s) in 3D.

    - single: shape (4,4)
    - batch:  shape (N,4,4)

    Standard structure:
      [ R | t ]
      [ 0 | 1 ]
    """
    matrix: Tensor
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        m = self.matrix.numpy()
        if m.shape == (4, 4):
            return
        if m.ndim == 3 and m.shape[1:] == (4, 4):
            return
        raise ValueError("TransformTensor expects shape (4,4) or (N,4,4)")

    @staticmethod
    def identity(*, batch: int | None = None, meta: Dict[str, Any] | None = None) -> "TransformTensor":
        if batch is None:
            m = np.eye(4, dtype=np.float32)
        else:
            m = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], batch, axis=0)
        return TransformTensor(matrix=Tensor.from_data(m, dtype=get_dtype(np.dtype(np.float32))), meta=meta or {})

    @staticmethod
    def from_rt(
        R: Any,
        t: Any,
        *,
        meta: Dict[str, Any] | None = None,
    ) -> "TransformTensor":
        R = _as_float32(R)
        t = _as_float32(t).reshape(3)

        if R.shape != (3, 3):
            raise ValueError("R must be (3,3)")
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = R
        m[:3, 3] = t
        return TransformTensor(matrix=Tensor.from_data(m, dtype=get_dtype(np.dtype(np.float32))), meta=meta or {})

    def numpy(self) -> np.ndarray:
        return self.matrix.numpy()

    def inverse(self) -> "TransformTensor":
        m = self.numpy()
        inv = np.linalg.inv(m)
        return TransformTensor(matrix=Tensor.from_data(inv.astype(np.float32)), meta=dict(self.meta))

    def __repr__(self) -> str:
        return f"TransformTensor(shape={self.matrix.shape})"

@dataclass(frozen=True, slots=True)
class PointCloud:
    """
    Point cloud datatype.

    points: (N,3) float32 (XYZ)
    colors: optional (N,3) float32 in [0,1]
    normals: optional (N,3) float32
    """
    points: Tensor
    colors: Optional[Tensor] = None
    normals: Optional[Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        p = self.points.numpy()
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError("PointCloud.points must have shape (N,3)")
        if p.dtype not in (np.float32, np.float64):
            raise TypeError("PointCloud.points should be float32/float64")

        if self.colors is not None:
            c = self.colors.numpy()
            if c.shape != p.shape:
                raise ValueError("colors must have same shape as points (N,3)")

        if self.normals is not None:
            n = self.normals.numpy()
            if n.shape != p.shape:
                raise ValueError("normals must have same shape as points (N,3)")

    @property
    def n(self) -> int:
        return int(self.points.shape[0])

    def numpy(self) -> np.ndarray:
        return self.points.numpy()

    @staticmethod
    def from_numpy(
        points: Any,
        *,
        colors: Any | None = None,
        normals: Any | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "PointCloud":
        p = _as_float32(points)
        pt = Tensor.from_data(p, dtype=get_dtype(np.dtype(np.float32)))
        ct = Tensor.from_data(_as_float32(colors), dtype=get_dtype(np.dtype(np.float32))) if colors is not None else None
        nt = Tensor.from_data(_as_float32(normals), dtype=get_dtype(np.dtype(np.float32))) if normals is not None else None
        return PointCloud(points=pt, colors=ct, normals=nt, meta=meta or {})

    def transform(self, T: TransformTensor) -> "PointCloud":
        pts = self.points.numpy()
        m = T.numpy()
        if m.shape != (4, 4):
            raise ValueError("PointCloud.transform currently supports single (4,4) transform only")

        homog = np.ones((pts.shape[0], 4), dtype=pts.dtype)
        homog[:, :3] = pts
        out = (homog @ m.T)[:, :3]  # apply transform
        return PointCloud.from_numpy(out, colors=self.colors.numpy() if self.colors else None,
                                     normals=self.normals.numpy() if self.normals else None,
                                     meta=dict(self.meta))

    def aabb(self) -> "BBoxGTensor":
        pts = self.points.numpy()
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        return BBoxGTensor.from_min_max(mn, mx, meta=dict(self.meta))

    def __repr__(self) -> str:
        return f"PointCloud(n={self.n})"

@dataclass(frozen=True, slots=True)
class MeshTensor:
    """
    Triangle mesh datatype (indexed face list):
      - vertices: (V,3)
      - faces: (F,3) indices into vertices
    Indexed face lists are a common mesh representation.
    """
    vertices: Tensor
    faces: Tensor
    normals: Optional[Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        v = self.vertices.numpy()
        f = self.faces.numpy()

        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("vertices must have shape (V,3)")
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError("faces must have shape (F,3)")

        if f.dtype.kind not in ("i", "u"):
            raise TypeError("faces must be integer indices")

        if f.size > 0:
            if f.min() < 0 or f.max() >= v.shape[0]:
                raise IndexError("faces contain out-of-range vertex indices")

        if self.normals is not None:
            n = self.normals.numpy()
            if n.shape != v.shape:
                raise ValueError("normals must have shape (V,3)")

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    @staticmethod
    def from_numpy(
        vertices: Any,
        faces: Any,
        *,
        normals: Any | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "MeshTensor":
        v = Tensor.from_data(_as_float32(vertices), dtype=get_dtype(np.dtype(np.float32)))
        f = Tensor.from_data(_as_int64(faces), dtype=get_dtype(np.dtype(np.int64)))
        n = Tensor.from_data(_as_float32(normals), dtype=get_dtype(np.dtype(np.float32))) if normals is not None else None
        return MeshTensor(vertices=v, faces=f, normals=n, meta=meta or {})

    def aabb(self) -> "BBoxGTensor":
        v = self.vertices.numpy()
        return BBoxGTensor.from_min_max(v.min(axis=0), v.max(axis=0), meta=dict(self.meta))

    def transform(self, T: TransformTensor) -> "MeshTensor":
        v = self.vertices.numpy()
        m = T.numpy()
        if m.shape != (4, 4):
            raise ValueError("MeshTensor.transform currently supports single (4,4) transform only")

        homog = np.ones((v.shape[0], 4), dtype=v.dtype)
        homog[:, :3] = v
        out_v = (homog @ m.T)[:, :3]

        return MeshTensor.from_numpy(out_v, self.faces.numpy(),
                                     normals=self.normals.numpy() if self.normals else None,
                                     meta=dict(self.meta))

    def __repr__(self) -> str:
        return f"MeshTensor(V={self.num_vertices}, F={self.num_faces})"

@dataclass(frozen=True, slots=True)
class BBoxGTensor:
    """
    Axis-aligned bounding box in 3D.

    Stored as:
      - min_xyz: (3,)
      - max_xyz: (3,)
    """
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_min_max(min_xyz: Any, max_xyz: Any, *, meta: Dict[str, Any] | None = None) -> "BBoxGTensor":
        mn = _as_float32(min_xyz).reshape(3)
        mx = _as_float32(max_xyz).reshape(3)
        lo = np.minimum(mn, mx)
        hi = np.maximum(mn, mx)
        return BBoxGTensor(min_xyz=lo, max_xyz=hi, meta=meta or {})

    @property
    def center(self) -> np.ndarray:
        return (self.min_xyz + self.max_xyz) / 2.0

    @property
    def extent(self) -> np.ndarray:
        return (self.max_xyz - self.min_xyz)

    def contains(self, points: Any) -> np.ndarray:
        p = _as_float32(points)
        if p.ndim == 1:
            p = p.reshape(1, 3)
        return np.all((p >= self.min_xyz) & (p <= self.max_xyz), axis=1)

    def __repr__(self) -> str:
        return f"BBoxGTensor(min={self.min_xyz.tolist()}, max={self.max_xyz.tolist()})"
