# pyseed_dtypes/tensors.py
"""
pyseed_dtypes.tensors
=====================

Typed, immutable, and backend-agnostic tensor system for dense, sparse, and symbolic computation.

This module provides a unified framework for representing and manipulating multi-dimensional
arrays (tensors) with strict data types, metadata tracking, and support for symbolic expressions.

Classes
-------

1. Tensor
   --------
   - Dense n-dimensional numeric tensor.
   - Immutable and dtype-safe.
   - Supports element-wise operations (+, -, *, /), indexing, and conversion to NumPy arrays.
   - Metadata can be stored alongside the tensor.
   - Example use cases: physics simulations, ML datasets, images, numeric computation.

2. SparseTensor
   --------------
   - 2D sparse tensor (matrix) in COO format.
   - Stores non-zero values with (row, col) coordinates.
   - Efficient memory usage for large sparse matrices.
   - Supports coalescing duplicates, bounds checking, and conversion to dense arrays.
   - Example use cases: adjacency matrices, finite element simulations, recommendation systems.

3. NDSparseTensor
   ----------------
   - N-dimensional sparse tensor in COO format.
   - Stores non-zero values with multi-dimensional coordinates.
   - Supports coalescing duplicates and conversion to dense arrays.
   - Example use cases: volumetric data, multi-dimensional grids, sparse embeddings in ML.

4. SymbolicTensor
   ----------------
   - N-dimensional tensor of symbolic expressions (SymPy objects).
   - Allows symbolic manipulation, simplification, substitution, and numeric evaluation.
   - Supports element-wise operations, free symbol collection, lambdify compilation.
   - Example use cases: symbolic physics, robotics, symbolic optimization, algebraic computation.

Key Features
------------
- Immutable tensor objects for functional and reproducible pipelines.
- Unified dtype system via DType for all tensor types.
- Metadata tracking for each tensor (`meta` dictionary).
- Interoperable with NumPy for numeric operations and SymPy for symbolic operations.
- COO-based sparse representation with coalescing of duplicates.
- N-Dimensional sparse tensors and symbolic tensors for high-dimensional and algebraic use cases.

Dependencies
------------
- numpy (required)
- sympy (optional, required for SymbolicTensor)

Notes
-----
- SymbolicTensor operations require SymPy (`pip install sympy`).
- Sparse tensors enforce bounds and type safety for indices.
- NDSparseTensor and SparseTensor provide efficient memory usage for mostly-zero arrays.
- All tensor classes are immutable and use slots for memory efficiency.

"""

from __future__ import annotations
from enum import Enum, auto, Flag
from ..explict import (
    DType,
    Float16, Float32, Float64,
    Int8, Int16, Int32, Int64,
    Bool,
    get_dtype, DTypeLike
)
from typing import (
    Tuple, Dict, Any, Set, List,
    Iterable, Sequence, Mapping,
    Callable, Optional
)
from dataclasses import dataclass, field
import numpy as np
from functools import reduce
from operator import mul

try:
    import sympy as sp
except Exception:  # keep import soft
    sp = None


__all__ = [
    "Tensor",
    "SparseTensor",
    "NDSparseTensor",
    "SymbolicTensor",
    "GradTensor",
    "DiagonalTensor",
    "CSRSparseTensor", 
    "CSCSparseTensor",
    "MaskedTensor",
    "IntervalTensor",
    "CPTensor",
    "TuckerTensor",
    "ChronoTensor",
    "AdjacencyTensor",
    "ComplexTensor", 
    "QuantumTensor"
]

def _require_sympy():
    if sp is None:
        raise ImportError("SymbolicTensor requires sympy (pip install sympy).")

def _infer_dtype(data) -> DType:
    arr = np.asarray(data)
    if arr.dtype == np.dtype("object"):
        raise TypeError("Object or mixed dtype is not allowed")
    return get_dtype(arr.dtype)


def _as_int64_1d(x, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={arr.shape}")
    return arr

def _as_coords(coords, *, ndim: int) -> np.ndarray:
    """
    Accept coords as either:
      - np.ndarray shape (ndim, nnz)
      - tuple/list of length ndim, each 1D of length nnz
    Store internally as int64 ndarray of shape (ndim, nnz).
    """
    if isinstance(coords, np.ndarray):
        c = np.asarray(coords, dtype=np.int64)
        if c.ndim != 2:
            raise ValueError(f"coords must be 2D (ndim, nnz), got {c.ndim}D")
        if c.shape[0] != ndim:
            raise ValueError(f"coords first axis must be ndim={ndim}, got {c.shape[0]}")
        return c

    if not isinstance(coords, (tuple, list)) or len(coords) != ndim:
        raise TypeError("coords must be an array (ndim, nnz) or a tuple/list of length ndim")

    cols = [_as_int64_1d(coords[i], name=f"coords[{i}]") for i in range(ndim)]
    nnz = len(cols[0]) if ndim else 0
    if any(len(v) != nnz for v in cols):
        raise ValueError("All coordinate arrays must have the same length (nnz)")
    return np.stack(cols, axis=0)

def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Sum grad along broadcasted axes to match target_shape.
    """
    if grad.shape == target_shape:
        return grad

    # If target has fewer dims, sum extra leading dims
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # Sum axes where target dim is 1 but grad dim is >1
    for axis, (g, t) in enumerate(zip(grad.shape, target_shape)):
        if t == 1 and g != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    if grad.shape != target_shape:
        raise ValueError(f"Cannot unbroadcast grad shape {grad.shape} -> {target_shape}")
    return grad

def _validate_compressed(
    *,
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    shape: Tuple[int, int],
    axis: int,  # 0 for CSR rows, 1 for CSC cols
    check_bounds: bool,
) -> None:
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D arrays")

    if data.shape[0] != indices.shape[0]:
        raise ValueError("data and indices must have same length (nnz)")

    major = shape[axis]
    minor = shape[1 - axis]

    if indptr.shape[0] != major + 1:
        raise ValueError(f"indptr must have length {major + 1} (major_dim + 1)")

    if indptr[0] != 0 or indptr[-1] != data.shape[0]:
        raise ValueError("indptr must start at 0 and end at nnz")

    if np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("indptr must be non-decreasing")

    if check_bounds and data.shape[0] > 0:
        if np.any(indices < 0) or np.any(indices >= minor):
            raise IndexError("indices out of bounds for minor dimension")

def _prod(xs: Sequence[int]) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p

def _check_factors(factors: Sequence[np.ndarray]) -> Tuple[int, Tuple[int, ...]]:
    if len(factors) == 0:
        raise ValueError("Need at least one factor matrix")
    ranks = [f.shape[1] for f in factors]
    if len(set(ranks)) != 1:
        raise ValueError(f"All factor matrices must have same rank R, got {ranks}")
    shape = tuple(int(f.shape[0]) for f in factors)
    return int(ranks[0]), shape

def _is_complex_dtype(dt: np.dtype) -> bool:
    return dt.kind == "c"


@dataclass(frozen=True, slots=True)
class Tensor:
    """
    Dense n-dimensional numeric tensor with immutable storage and metadata.

    Attributes:
        _data (np.ndarray): Internal dense data array.
        shape (Tuple[int, ...]): Shape of the tensor.
        dtype (DType): Data type of the tensor.
        meta (Dict[str, Any]): Optional metadata dictionary.
    """

    _data: np.ndarray
    shape: Tuple[int, ...]
    dtype: DType
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_data(data, *, dtype: DTypeLike | None = None, meta=None) -> "Tensor":
        """
        Construct a Tensor from array-like data.

        Args:
            data: Array-like input.
            dtype: Optional DType for casting.
            meta: Optional metadata dictionary.

        Returns:
            Tensor: New immutable Tensor instance.
        """
        dt = get_dtype(dtype) if dtype is not None else None
        arr = np.asarray(data, dtype=dt.np_dtype if dt else None)
        return Tensor(
            _data=arr,
            shape=tuple(arr.shape),
            dtype=dt or _infer_dtype(arr),
            meta=meta or {},
        )

    @staticmethod
    def from_dense(array, *, dtype: DTypeLike | None = None, meta=None, shape=None) -> "Tensor":
        """
        Construct a Tensor from a dense array, optionally validating shape.

        Args:
            array: Dense array input.
            dtype: Optional DType.
            meta: Optional metadata.
            shape: Expected shape of the tensor.

        Returns:
            Tensor: New Tensor instance.

        Raises:
            ValueError: If provided shape does not match array.
        """
        dt = get_dtype(dtype) if dtype is not None else None
        arr = np.asarray(array, dtype=dt.np_dtype if dt else None)
        if shape is not None and tuple(arr.shape) != tuple(shape):
            raise ValueError(f"dense array shape {arr.shape} != expected {shape}")
        return Tensor.from_data(arr, dtype=dt, meta=meta)

    @property
    def ndim(self) -> int:
        """int: Number of dimensions of the tensor."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """int: Total number of elements in the tensor."""
        return reduce(mul, self.shape, 1)

    def numpy(self) -> np.ndarray:
        """Return a copy of the tensor as a NumPy ndarray."""
        return np.asarray(self._data)

    def tolist(self):
        """Convert tensor to a nested Python list."""
        return self.numpy().tolist()

    def __getitem__(self, idx):
        """Index tensor and return a new Tensor instance."""
        result = self.numpy()[idx]
        if np.isscalar(result):
            # strict tensor world
            return Tensor.from_data(result, dtype=self.dtype, meta=self.meta)
        return Tensor.from_data(result, dtype=self.dtype, meta=self.meta)

    def _binary_op(self, other, op):
        """
        Elementwise binary operation with another Tensor.

        Args:
            other (Tensor): Other tensor.
            op (Callable): Numpy ufunc to apply.

        Returns:
            Tensor: Result tensor.

        Raises:
            TypeError: If other is not a Tensor.
        """
        if not isinstance(other, Tensor):
            raise TypeError("Tensor operations require Tensor")
        result = op(self.numpy(), other.numpy())
        return Tensor.from_data(result)

    def __add__(self, other): return self._binary_op(other, np.add)
    def __sub__(self, other): return self._binary_op(other, np.subtract)
    def __mul__(self, other): return self._binary_op(other, np.multiply)
    def __truediv__(self, other): return self._binary_op(other, np.divide)
    def __eq__(self, value):
        pass

    def copy(self) -> "Tensor":
        """Return a deep copy of the tensor."""
        return Tensor.from_data(
            self.numpy().copy(),
            dtype=self.dtype,
            meta=dict(self.meta),
        )

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype.name})"
    
@dataclass(frozen=True, slots=True)
class SparseTensor:
    """
    2D COO (Coordinate) sparse tensor.
    - COO sparse tensor for 2D only (matrix): - data[k] stored at (row[k], col[k])
    Attributes:
        data (np.ndarray): Non-zero values, shape (nnz,).
        row (np.ndarray): Row indices, shape (nnz,).
        col (np.ndarray): Column indices, shape (nnz,).
        shape (Tuple[int, int]): Matrix shape (m, n).
        dtype (DType): Data type.
        meta (Dict[str, Any]): Optional metadata.

    """
    data: np.ndarray                 # shape (nnz,)
    row: np.ndarray                  # shape (nnz,)
    col: np.ndarray                  # shape (nnz,)
    shape: Tuple[int, int]           # (m, n)
    dtype: DType
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_coo(
        row: Sequence[int] | np.ndarray,
        col: Sequence[int] | np.ndarray,
        data: Sequence[Any] | np.ndarray,
        *,
        shape: Tuple[int, int],
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
        coalesce: bool = True,
        check_bounds: bool = True,
    ) -> "SparseTensor":
        """
        Create a SparseTensor from COO format.

        Args:
            row: Row indices of non-zero elements.
            col: Column indices of non-zero elements.
            data: Values of non-zero elements.
            shape: Shape of the matrix (m, n).
            dtype: Optional data type.
            meta: Optional metadata.
            coalesce: Sum duplicates if True.
            check_bounds: Validate indices are within shape if True.

        Returns:
            SparseTensor: New SparseTensor instance.

        Raises:
            ValueError: If input lengths mismatch or shape invalid.
            IndexError: If indices out of bounds (check_bounds=True).
        """
        dt = get_dtype(dtype) if dtype is not None else None

        r = _as_int64_1d(row, name="row")
        c = _as_int64_1d(col, name="col")
        x = np.asarray(data, dtype=dt.np_dtype if dt else None)

        if x.ndim != 1:
            raise ValueError(f"data must be 1D, got shape={x.shape}")
        if not (len(r) == len(c) == len(x)):
            raise ValueError("row, col, data must have the same length")

        m, n = shape
        if m < 0 or n < 0:
            raise ValueError(f"Invalid shape={shape}")

        if check_bounds and len(x) > 0:
            if (r.min() < 0) or (c.min() < 0) or (r.max() >= m) or (c.max() >= n):
                raise IndexError("row/col indices out of bounds for shape")

        inferred = dt or get_dtype(x.dtype)
        out = SparseTensor(
            data=x,
            row=r,
            col=c,
            shape=(m, n),
            dtype=inferred,
            meta=meta or {},
        )
        return out.coalesce_sum() if coalesce else out

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return int(self.data.shape[0])

    @property
    def ndim(self) -> int:
        """Number of dimensions (always 2)."""
        return 2

    def numpy(self) -> np.ndarray:
        """Convert sparse tensor to dense NumPy array."""
        m, n = self.shape
        out = np.zeros((m, n), dtype=self.dtype.np_dtype)
        if self.nnz:
            out[self.row, self.col] += self.data
        return out

    def to_coo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (row, col, data) arrays as-is."""
        return self.row, self.col, self.data

    def copy(self) -> "SparseTensor":
        """Return a deep copy of the sparse tensor."""
        return SparseTensor(
            data=self.data.copy(),
            row=self.row.copy(),
            col=self.col.copy(),
            shape=self.shape,
            dtype=self.dtype,
            meta=dict(self.meta),
        )

    def coalesce_sum(self) -> "SparseTensor":
        """
        Combine duplicate (row, col) entries by summing values.

        Returns:
            SparseTensor: New coalesced tensor.
        """
        if self.nnz == 0:
            return self

        m, n = self.shape
        # unique id per coordinate for 2:]
        key = self.row * n + self.col
        order = np.argsort(key, kind="mergesort")

        key_s = key[order]
        row_s = self.row[order]
        col_s = self.col[order]
        data_s = self.data[order]

        change = np.empty_like(key_s, dtype=bool)
        change[0] = True
        change[1:] = key_s[1:] != key_s[:-1]
        idx = np.nonzero(change)[0]

        out_row = row_s[idx]
        out_col = col_s[idx]
        out_data = np.add.reduceat(data_s, idx)

        return SparseTensor(
            data=out_data,
            row=out_row,
            col=out_col,
            shape=self.shape,
            dtype=self.dtype,
            meta=dict(self.meta),
        )

    def __repr__(self) -> str:
        return f"SparseTensor(shape={self.shape}, nnz={self.nnz}, dtype={self.dtype.name})"

@dataclass(frozen=True, slots=True)
class NDSparseTensor:
    """
    N-dimensional COO (Coordinate) sparse tensor.

    Attributes:
        data (np.ndarray): Non-zero values of shape (nnz,).
        coords (np.ndarray): Coordinates of non-zero elements, shape (ndim, nnz).
        shape (Tuple[int, ...]): Shape of the tensor.
        dtype (DType): Data type of the tensor.
        meta (Dict[str, Any]): Optional metadata dictionary.

    Internal representation:
      - coords: int64 array of shape (ndim, nnz)
      - data:   array of shape (nnz,)
      - shape:  tuple[int, ...] with length ndim

    Semantics: entry k stored at tuple(coords[:, k]).
    This mirrors standard COO descriptions used by sparse array libraries.
    """
    data: np.ndarray
    coords: np.ndarray                  # (ndim, nnz)
    shape: Tuple[int, ...]
    dtype: DType
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_coo(
        coords,
        data,
        *,
        shape: Tuple[int, ...],          # required: defines ndim at start
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
        coalesce: bool = True,
        check_bounds: bool = True,
    ) -> "NDSparseTensor":
        """
        Create an NDSparseTensor from coordinates and data (COO format).

        Args:
            coords: Array of shape (ndim, nnz) or tuple/list of 1D arrays for each dimension.
            data: Non-zero values of shape (nnz,).
            shape: Shape of the tensor (defines ndim).
            dtype: Optional DType to cast data.
            meta: Optional metadata dictionary.
            coalesce: If True, combine duplicates by summing values.
            check_bounds: If True, verify that all coordinates are within tensor shape.

        Returns:
            NDSparseTensor: New sparse tensor instance.

        Raises:
            ValueError: If shapes are invalid or lengths mismatch.
            IndexError: If coordinates are out of bounds and check_bounds=True.
        """
        if not isinstance(shape, tuple) or len(shape) == 0:
            raise ValueError("shape must be a non-empty tuple[int, ...] for NDSparseTensor")
        if any((not isinstance(s, int)) or s < 0 for s in shape):
            raise ValueError(f"Invalid shape={shape!r}")

        ndim = len(shape)
        dt = get_dtype(dtype) if dtype is not None else None

        x = np.asarray(data, dtype=dt.np_dtype if dt else None)
        if x.ndim != 1:
            raise ValueError(f"data must be 1D, got shape={x.shape}")

        c = _as_coords(coords, ndim=ndim)
        if c.shape[1] != x.shape[0]:
            raise ValueError(f"coords nnz={c.shape[1]} must match data nnz={x.shape[0]}")

        if check_bounds and x.shape[0] > 0:
            # coords must be within [0, shape[axis])
            for ax, dim in enumerate(shape):
                v = c[ax]
                if (v.min() < 0) or (v.max() >= dim):
                    raise IndexError(f"coords out of bounds on axis {ax} for dim {dim}")

        inferred = dt or get_dtype(x.dtype)
        out = NDSparseTensor(
            data=x,
            coords=c,
            shape=shape,
            dtype=inferred,
            meta=meta or {},
        )
        return out.coalesce_sum() if coalesce else out
    
    @staticmethod
    def from_dense(
        array,
        *,
        zero=0,
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
        coalesce: bool = False,
    ) -> "NDSparseTensor":
        """
        Build an NDSparseTensor from a dense array by storing only non-zero entries.

        Args:
            array: Dense array input.
            zero: Value to consider as zero (ignored in sparse representation).
            dtype: Optional DType to cast values.
            meta: Optional metadata dictionary.
            coalesce: If True, combine duplicates by summing (rarely needed for dense input).

        Returns:
            NDSparseTensor: Sparse representation of the array.

        Raises:
            ValueError: If array is zero-dimensional.

        `Expensive for huge arrays because it scans the whole dense buffer.`
        """
        arr = np.asarray(array)
        shape = tuple(arr.shape)
        if len(shape) == 0:
            raise ValueError("from_dense expects at least 1-D array")

        dt = get_dtype(dtype) if dtype is not None else None
        # Cast numeric dtype if requested (keeps object arrays as-is if dt not given)
        if dt is not None:
            arr = np.asarray(arr, dtype=dt.np_dtype)

        mask = (arr != zero)
        nz = np.nonzero(mask)  # tuple of arrays, one per dimension
        nnz = len(nz[0]) if nz else 0

        coords = np.vstack([np.asarray(ax, dtype=np.int64) for ax in nz])  # (ndim, nnz)
        data = arr[nz].reshape(nnz)  # non-zero values in C-order

        # shape fixes ndim "at the start"
        return NDSparseTensor.from_coo(
            coords,
            data,
            shape=shape,
            dtype=dt,
            meta=meta,
            coalesce=coalesce,
            check_bounds=False,
        )

    @property
    def ndim(self) -> int:
        """Number of dimensions of the tensor."""
        return len(self.shape)

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return int(self.data.shape[0])

    def to_coo(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (coords, data) arrays."""
        return self.coords, self.data

    def numpy(self) -> np.ndarray:
        """Convert sparse tensor to a dense NumPy array."""
        out = np.zeros(self.shape, dtype=self.dtype.np_dtype)
        if self.nnz == 0:
            return out
        # N-D advanced indexing: build tuple of index arrays
        idx = tuple(self.coords[ax] for ax in range(self.ndim))
        out[idx] += self.data
        return out

    def copy(self) -> "NDSparseTensor":
        """Return a deep copy of the sparse tensor."""
        return NDSparseTensor(
            data=self.data.copy(),
            coords=self.coords.copy(),
            shape=self.shape,
            dtype=self.dtype,
            meta=dict(self.meta),
        )

    def coalesce_sum(self) -> "NDSparseTensor":
        """
        Combine duplicate coordinates by summing their values.
        COO formats typically allow duplicates; when normalizing/converting,
        duplicates are summed. 

        Returns:
            NDSparseTensor: New coalesced tensor.
        """
        if self.nnz == 0:
            return self

        # Use np.ravel_multi_index to map N-D coords -> flat keys for sorting/grouping.
        keys = np.ravel_multi_index(tuple(self.coords[ax] for ax in range(self.ndim)), self.shape)
        order = np.argsort(keys, kind="mergesort")

        keys_s = keys[order]
        data_s = self.data[order]
        coords_s = self.coords[:, order]

        change = np.empty_like(keys_s, dtype=bool)
        change[0] = True
        change[1:] = keys_s[1:] != keys_s[:-1]
        idx = np.nonzero(change)[0]

        out_data = np.add.reduceat(data_s, idx)
        out_coords = coords_s[:, idx]

        return NDSparseTensor(
            data=out_data,
            coords=out_coords,
            shape=self.shape,
            dtype=self.dtype,
            meta=dict(self.meta),
        )

    def __repr__(self) -> str:
        return f"NDSparseTensor(shape={self.shape}, nnz={self.nnz}, dtype={self.dtype.name})"
    
@dataclass(frozen=True, slots=True)
class SymbolicTensor:
    """
    N-dimensional tensor of symbolic expressions (SymPy objects).
    - shape is fixed at construction time.
    - stored as numpy object array (each element is a SymPy expr or similar).

    Attributes:
        _data (np.ndarray): Object array containing SymPy expressions.
        shape (Tuple[int, ...]): Shape of the tensor.
        dtype (DType): Numeric dtype for evaluation purposes.
        meta (Dict[str, Any]): Optional metadata.
    
    `This mirrors SymPy's idea of N-dim arrays containing expressions.`
    """
    _data: np.ndarray                 # dtype=object, shape=shape
    shape: Tuple[int, ...]
    dtype: DType                      # your dtype system (often float32/float64)
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_data(
        data: Any,
        *,
        shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
        strict_shape: bool = True,
    ) -> "SymbolicTensor":
        """
        Construct a SymbolicTensor from array-like symbolic data.

        Args:
            data: Array-like or scalar symbolic input.
            shape: Shape of the tensor. Required if strict_shape=True.
            dtype: Optional numeric DType for evaluation.
            meta: Optional metadata dictionary.
            strict_shape: If True, enforce shape to match data.

        Returns:
            SymbolicTensor: New symbolic tensor instance.

        Raises:
            ImportError: If SymPy is not installed.
            ValueError: If shape does not match data and strict_shape=True.
        """
        _require_sympy()

        arr = np.asarray(data, dtype=object)

        if shape is None:
            if strict_shape:
                raise ValueError("shape must be provided (SymbolicTensor is fixed-shape).")
            shape = tuple(arr.shape)

        if tuple(arr.shape) != tuple(shape):
            # allow scalar broadcast if user passed a scalar expression
            if arr.shape == ():
                arr = np.full(shape, arr.item(), dtype=object)
            else:
                raise ValueError(f"data shape {arr.shape} does not match shape {shape}")

        dt = get_dtype(dtype) if dtype is not None else get_dtype("float64")
        return SymbolicTensor(_data=arr, shape=tuple(shape), dtype=dt, meta=meta or {})
    
    @staticmethod
    def from_dense(array, *, shape: Tuple[int, ...], dtype: DTypeLike | None = None, meta=None) -> "SymbolicTensor":
        """
        Construct a SymbolicTensor from a dense array.

        Args:
            array: Dense array of symbolic expressions.
            shape: Expected shape of the tensor.
            dtype: Optional numeric DType for evaluation.
            meta: Optional metadata.

        Returns:
            SymbolicTensor: New symbolic tensor instance.

        Raises:
            ImportError: If SymPy is not installed.
            ValueError: If shape does not match array.
        """
        _require_sympy()
        arr = np.asarray(array, dtype=object)
        if tuple(arr.shape) != tuple(shape):
            raise ValueError(f"dense array shape {arr.shape} != expected {shape}")
        dt = get_dtype(dtype) if dtype is not None else get_dtype("float64")
        return SymbolicTensor.from_data(arr, shape=shape, dtype=dt, meta=meta, strict_shape=True)

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the tensor."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Return the total number of elements in the tensor (product of shape)."""
        s = 1
        for d in self.shape:
            s *= int(d)
        return s

    def numpy_object(self) -> np.ndarray:
        """
        Return a NumPy object array containing the symbolic expressions.

        Returns:
            np.ndarray: Array of dtype=object containing all symbolic elements.
        """
        return np.asarray(self._data, dtype=object)

    def tolist(self):
        """
        Convert the symbolic tensor to a nested Python list.

        Returns:
            list: Nested list representing the tensor elements.
        """
        return self.numpy_object().tolist()

    def __getitem__(self, idx) -> "SymbolicTensor":
        """
        Index into the symbolic tensor.

        Args:
            idx: Index, slice, or tuple of indices.

        Returns:
            SymbolicTensor: A new SymbolicTensor corresponding to the selected subset.

        Notes:
            0-D results are returned as SymbolicTensor of shape ().
        """
        out = self.numpy_object()[idx]
        out_arr = np.asarray(out, dtype=object)
        # Ensure returned result is still a SymbolicTensor (0-D allowed)
        return SymbolicTensor.from_data(out_arr, shape=tuple(out_arr.shape), dtype=self.dtype, strict_shape=False)

    def free_symbols(self) -> set:
        """
        Collect all free symbols present in the tensor.

        Returns:
            set: A set of SymPy symbols found in the tensor elements.

        Raises:
            ImportError: If SymPy is not installed.
        """
        _require_sympy()
        syms: set = set()
        it = np.nditer(self._data, flags=["refs_ok", "multi_index"])
        for x in it:
            expr = x.item()
            if hasattr(expr, "free_symbols"):
                syms |= set(expr.free_symbols)
        return syms

    def subs(self, mapping: Mapping[Any, Any]) -> "SymbolicTensor":
        """
        Substitute symbolic variables with values or other expressions.

        Args:
            mapping (Mapping[Any, Any]): Dictionary mapping symbols to values or expressions.

        Returns:
            SymbolicTensor: New symbolic tensor with substitutions applied.

        Raises:
            ImportError: If SymPy is not installed.
        """
        _require_sympy()
        out = np.empty(self.shape, dtype=object)
        it = np.nditer(self._data, flags=["refs_ok", "multi_index"])
        for x in it:
            expr = x.item()
            out[it.multi_index] = expr.subs(mapping) if hasattr(expr, "subs") else expr
        return SymbolicTensor.from_data(out, shape=self.shape, dtype=self.dtype)

    def simplify(self) -> "SymbolicTensor":
        """
        Simplify all elements of the symbolic tensor using SymPy.

        Returns:
            SymbolicTensor: New symbolic tensor with simplified expressions.

        Raises:
            ImportError: If SymPy is not installed.
        """
        _require_sympy()
        out = np.empty(self.shape, dtype=object)
        it = np.nditer(self._data, flags=["refs_ok", "multi_index"])
        for x in it:
            out[it.multi_index] = sp.simplify(x.item())
        return SymbolicTensor.from_data(out, shape=self.shape, dtype=self.dtype)

    def evalf(self, *, subs: Mapping[Any, Any] | None = None) -> np.ndarray:
        """
        Numerically evaluate the tensor elements.

        Args:
            subs (Mapping[Any, Any], optional): Optional substitution mapping to apply before evaluation.

        Returns:
            np.ndarray: Dense NumPy array with numeric values.

        Raises:
            ImportError: If SymPy is not installed.
        """
        _require_sympy()
        base = self.subs(subs) if subs else self
        out = np.empty(base.shape, dtype=base.dtype.np_dtype)
        it = np.nditer(base._data, flags=["refs_ok", "multi_index"])
        for x in it:
            out[it.multi_index] = float(sp.N(x.item()))
        return out

    def lambdify(self, symbols: Sequence[Any], *, modules: str = "numpy"):
        """
        Generate a callable function from the symbolic tensor using SymPy's lambdify.

        Args:
            symbols (Sequence[Any]): List of symbols to be input arguments.
            modules (str): Backend module to use for numeric evaluation (default 'numpy').

        Returns:
            Callable: Function that accepts numeric inputs for symbols and returns evaluated tensor.

        Raises:
            ImportError: If SymPy is not installed.

        Example:
            >>> f = tensor.lambdify([x, y])
            >>> f(1, 2)  # returns dense numeric array
        """
        _require_sympy()
        # Create a function per element; vectorizing is possible later, keep it simple/robust.
        flat_exprs = [self._data.reshape(-1)[i] for i in range(self.size)]
        f = sp.lambdify(list(symbols), flat_exprs, modules=modules)

        def fn(*vals):
            flat = np.asarray(f(*vals), dtype=self.dtype.np_dtype)
            return flat.reshape(self.shape)

        return fn

    def __repr__(self) -> str:
        """Return a concise string representation of the symbolic tensor."""
        return f"SymbolicTensor(shape={self.shape}, dtype={self.dtype.name})"

    def _binary_op(self, other, op):
        """
        Perform elementwise binary operation with another SymbolicTensor or scalar.

        Args:
            other (SymbolicTensor or scalar): Tensor or scalar operand.
            op (Callable): Binary operator function (e.g., lambda a,b: a+b).

        Returns:
            SymbolicTensor: Result of elementwise operation.

        Raises:
            TypeError: If `other` is not a scalar or SymbolicTensor.
            ValueError: If shapes do not match for elementwise operation.
            ImportError: If SymPy is not installed.
        """
        _require_sympy()
        if isinstance(other, SymbolicTensor):
            if self.shape != other.shape:
                raise ValueError("Shapes must match for elementwise operation")
            out = np.empty(self.shape, dtype=object)
            it = np.nditer([self._data, other._data, out], flags=["refs_ok"], op_flags=[["readonly"], ["readonly"], ["writeonly"]])
            for a, b, o in it:
                o[...] = op(a.item(), b.item())
            return SymbolicTensor.from_data(out, shape=self.shape, dtype=self.dtype)
        elif np.isscalar(other):
            out = np.empty(self.shape, dtype=object)
            it = np.nditer([self._data, out], flags=["refs_ok"], op_flags=[["readonly"], ["writeonly"]])
            for a, o in it:
                o[...] = op(a.item(), other)
            return SymbolicTensor.from_data(out, shape=self.shape, dtype=self.dtype)
        else:
            raise TypeError(f"Unsupported type {type(other)} for SymbolicTensor operation")

    def __add__(self, other) : self._binary_op(other, lambda a, b: a + b)
    def __sub__(self, other): self._binary_op(other, lambda a, b: a - b)
    def __mul__(self, other): self._binary_op(other, lambda a, b: a * b)
    def __truediv__(self, other): self._binary_op(other, lambda a, b: a / b)

@dataclass
class GradTensor:
    """
    Dense autodiff tensor (reverse-mode) storing:
      - value: ndarray
      - grad: ndarray or None
      - graph links + backward function

    Similar spirit to PyTorch: leaf tensors can accumulate .grad when requires_grad=True.
    """
    value: np.ndarray
    dtype: DType
    requires_grad: bool = False
    grad: Optional[np.ndarray] = None

    # autograd internals
    _prev: Tuple["GradTensor", ...] = field(default_factory=tuple, repr=False)
    _backward: Callable[[np.ndarray], None] = field(default=lambda g: None, repr=False)

    @staticmethod
    def from_data(data: Any, *, dtype: DTypeLike | None = None, requires_grad: bool = False) -> "GradTensor":
        dt = get_dtype(dtype) if dtype is not None else None
        arr = np.asarray(data, dtype=dt.np_dtype if dt else None)
        return GradTensor(value=arr, dtype=dt or get_dtype(arr.dtype), requires_grad=requires_grad)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.value.shape)

    @property
    def ndim(self) -> int:
        return self.value.ndim

    def numpy(self) -> np.ndarray:
        return np.asarray(self.value)

    def zero_grad(self) -> None:
        self.grad = None

    def backward(self, grad: Any | None = None) -> None:
        """
        Backprop from this node.
        If grad is None, assumes scalar output and uses 1.0.
        """
        if not self.requires_grad:
            return

        if grad is None:
            if self.value.shape != ():
                raise ValueError("grad must be provided for non-scalar outputs")
            grad_arr = np.ones((), dtype=self.dtype.np_dtype)
        else:
            grad_arr = np.asarray(grad, dtype=self.dtype.np_dtype)

        # Topological order of graph (reverse-mode backprop)
        topo: List[GradTensor] = []
        visited: Set[int] = set()

        def build(v: GradTensor):
            vid = id(v)
            if vid in visited:
                return
            visited.add(vid)
            for p in v._prev:
                build(p)
            topo.append(v)

        build(self)

        # Seed output gradient
        self.grad = grad_arr if self.grad is None else (self.grad + grad_arr)

        # Traverse in reverse topological order
        for v in reversed(topo):
            if v.grad is None:
                continue
            v._backward(v.grad)

    def __add__(self, other: Any) -> "GradTensor":
        other = other if isinstance(other, GradTensor) else GradTensor.from_data(other, dtype=self.dtype, requires_grad=False)
        out_val = self.value + other.value
        out = GradTensor(out_val, dtype=self.dtype, requires_grad=self.requires_grad or other.requires_grad, _prev=(self, other))

        def _backward(gout: np.ndarray):
            if self.requires_grad:
                g = _unbroadcast(gout, self.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = _unbroadcast(gout, other.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        return out

    def __mul__(self, other: Any) -> "GradTensor":
        other = other if isinstance(other, GradTensor) else GradTensor.from_data(other, dtype=self.dtype, requires_grad=False)
        out_val = self.value * other.value
        out = GradTensor(out_val, dtype=self.dtype, requires_grad=self.requires_grad or other.requires_grad, _prev=(self, other))

        def _backward(gout: np.ndarray):
            if self.requires_grad:
                g = _unbroadcast(gout * other.value, self.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = _unbroadcast(gout * self.value, other.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        return out

    def __neg__(self) -> "GradTensor":
        out_val = -self.value
        out = GradTensor(out_val, dtype=self.dtype, requires_grad=self.requires_grad, _prev=(self,))

        def _backward(gout: np.ndarray):
            if self.requires_grad:
                g = -gout
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def __sub__(self, other: Any) -> "GradTensor":
        return self + (-other if isinstance(other, GradTensor) else -GradTensor.from_data(other, dtype=self.dtype))

    def __truediv__(self, other: Any) -> "GradTensor":
        other = other if isinstance(other, GradTensor) else GradTensor.from_data(other, dtype=self.dtype, requires_grad=False)
        out_val = self.value / other.value
        out = GradTensor(out_val, dtype=self.dtype, requires_grad=self.requires_grad or other.requires_grad, _prev=(self, other))

        def _backward(gout: np.ndarray):
            if self.requires_grad:
                g = _unbroadcast(gout / other.value, self.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = _unbroadcast(-gout * self.value / (other.value ** 2), other.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        return out

    # common reductions
    def sum(self) -> "GradTensor":
        out_val = self.value.sum()
        out = GradTensor(np.asarray(out_val), dtype=self.dtype, requires_grad=self.requires_grad, _prev=(self,))

        def _backward(gout: np.ndarray):
            if self.requires_grad:
                g = np.ones_like(self.value, dtype=self.dtype.np_dtype) * gout
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"GradTensor(shape={self.shape}, dtype={self.dtype.name}, requires_grad={self.requires_grad})"
    
@dataclass(frozen=True, slots=True)
class DiagonalTensor:
    """
    Stores only the main diagonal of a square tensor.

    Supported shapes:
      - 2D: (n, n)
      - ND: (n, n, ..., n)  (all dims equal)

    Storage:
      - diag: ndarray shape (n,)
      - shape: tuple[int, ...] fixed at construction
    """
    diag: np.ndarray               # (n,)
    shape: Tuple[int, ...]         # (n,n,...) fixed
    dtype: DType
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_diag(
        diag,
        *,
        shape: Tuple[int, ...] | None = None,
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "DiagonalTensor":
        dt = get_dtype(dtype) if dtype is not None else None
        d = np.asarray(diag, dtype=dt.np_dtype if dt else None)
        if d.ndim != 1:
            raise ValueError(f"diag must be 1D, got shape={d.shape}")

        n = int(d.shape[0])
        if shape is None:
            shape = (n, n)

        if not isinstance(shape, tuple) or len(shape) < 2:
            raise ValueError("shape must be a tuple with ndim >= 2")

        if any(int(s) != n for s in shape):
            raise ValueError(f"All dims in shape must equal len(diag)={n}, got shape={shape}")

        inferred = dt or get_dtype(d.dtype)
        return DiagonalTensor(diag=d, shape=shape, dtype=inferred, meta=meta or {})

    @staticmethod
    def from_dense(
        array,
        *,
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "DiagonalTensor":
        """
        Extract main diagonal from a dense square tensor.
        For 2D, equivalent to numpy.diag(array).
        For ND, takes elements where all indices are equal: a[i,i,...,i].
        """
        dt = get_dtype(dtype) if dtype is not None else None
        a = np.asarray(array, dtype=dt.np_dtype if dt else None)

        shape = tuple(a.shape)
        if len(shape) < 2:
            raise ValueError("from_dense expects ndim >= 2")

        if len(set(shape)) != 1:
            raise ValueError(f"from_dense requires all dims equal (square), got shape={shape}")

        n = shape[0]
        idx = (np.arange(n),) * len(shape)   # (i,i,...,i)
        d = a[idx]
        return DiagonalTensor.from_diag(d, shape=shape, dtype=dt, meta=meta)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def n(self) -> int:
        return int(self.diag.shape[0])

    def numpy(self) -> np.ndarray:
        """
        Convert to dense ndarray filled with zeros off-diagonal.
        """
        out = np.zeros(self.shape, dtype=self.dtype.np_dtype)
        if self.n == 0:
            return out
        idx = (np.arange(self.n),) * self.ndim
        out[idx] = self.diag
        return out

    def tolist(self):
        return self.numpy().tolist()

    def copy(self) -> "DiagonalTensor":
        return DiagonalTensor(
            diag=self.diag.copy(),
            shape=self.shape,
            dtype=self.dtype,
            meta=dict(self.meta),
        )

    def __repr__(self) -> str:
        return f"DiagonalTensor(shape={self.shape}, dtype={self.dtype.name})"

@dataclass(frozen=True, slots=True)
class CSRSparseTensor:
    """
    CSR (Compressed Sparse Row) matrix tensor (2D).

    For row i, columns are indices[indptr[i]:indptr[i+1]]
    and values are data[indptr[i]:indptr[i+1]].
    """
    data: np.ndarray
    indices: np.ndarray     # column indices
    indptr: np.ndarray      # row pointer, length = n_rows + 1
    shape: Tuple[int, int]
    dtype: DType
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_csr(
        data: Sequence[Any] | np.ndarray,
        indices: Sequence[int] | np.ndarray,
        indptr: Sequence[int] | np.ndarray,
        *,
        shape: Tuple[int, int],
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
        check_bounds: bool = True,
    ) -> "CSRSparseTensor":
        dt = get_dtype(dtype) if dtype is not None else None
        x = np.asarray(data, dtype=dt.np_dtype if dt else None)
        idx = _as_int64_1d(indices, name="indices")
        ip = _as_int64_1d(indptr, name="indptr")

        inferred = dt or get_dtype(x.dtype)
        _validate_compressed(
            data=x, indices=idx, indptr=ip, shape=shape, axis=0, check_bounds=check_bounds
        )
        return CSRSparseTensor(x, idx, ip, shape, inferred, meta=meta or {})

    @property
    def nnz(self) -> int:
        return int(self.data.shape[0])

    @property
    def ndim(self) -> int:
        return 2

    def numpy(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=self.dtype.np_dtype)
        m, _ = self.shape
        for i in range(m):
            start, end = int(self.indptr[i]), int(self.indptr[i + 1])
            cols = self.indices[start:end]
            out[i, cols] += self.data[start:end]
        return out

    def to_coo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, _ = self.shape
        rows = np.empty(self.nnz, dtype=np.int64)
        for i in range(m):
            start, end = int(self.indptr[i]), int(self.indptr[i + 1])
            rows[start:end] = i
        cols = self.indices.copy()
        data = self.data.copy()
        return rows, cols, data

    def __repr__(self) -> str:
        return f"CSRSparseTensor(shape={self.shape}, nnz={self.nnz}, dtype={self.dtype.name})"

@dataclass(frozen=True, slots=True)
class CSCSparseTensor:
    """
    CSC (Compressed Sparse Column) matrix tensor (2D).

    For column j, rows are indices[indptr[j]:indptr[j+1]]
    and values are data[indptr[j]:indptr[j+1]].
    """
    data: np.ndarray
    indices: np.ndarray     # row indices
    indptr: np.ndarray      # col pointer, length = n_cols + 1
    shape: Tuple[int, int]
    dtype: DType
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_csc(
        data: Sequence[Any] | np.ndarray,
        indices: Sequence[int] | np.ndarray,
        indptr: Sequence[int] | np.ndarray,
        *,
        shape: Tuple[int, int],
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
        check_bounds: bool = True,
    ) -> "CSCSparseTensor":
        dt = get_dtype(dtype) if dtype is not None else None
        x = np.asarray(data, dtype=dt.np_dtype if dt else None)
        idx = _as_int64_1d(indices, name="indices")
        ip = _as_int64_1d(indptr, name="indptr")

        inferred = dt or get_dtype(x.dtype)
        _validate_compressed(
            data=x, indices=idx, indptr=ip, shape=shape, axis=1, check_bounds=check_bounds
        )
        return CSCSparseTensor(x, idx, ip, shape, inferred, meta=meta or {})

    @property
    def nnz(self) -> int:
        return int(self.data.shape[0])

    @property
    def ndim(self) -> int:
        return 2

    def numpy(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=self.dtype.np_dtype)
        _, n = self.shape
        for j in range(n):
            start, end = int(self.indptr[j]), int(self.indptr[j + 1])
            rows = self.indices[start:end]
            out[rows, j] += self.data[start:end]
        return out

    def to_coo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, n = self.shape
        cols = np.empty(self.nnz, dtype=np.int64)
        for j in range(n):
            start, end = int(self.indptr[j]), int(self.indptr[j + 1])
            cols[start:end] = j
        rows = self.indices.copy()
        data = self.data.copy()
        return rows, cols, data

    def __repr__(self) -> str:
        return f"CSCSparseTensor(shape={self.shape}, nnz={self.nnz}, dtype={self.dtype.name})"

@dataclass(frozen=True, slots=True)
class MaskedTensor:
    """
    Dense tensor + boolean mask.
      - mask=True means the corresponding element is masked/ignored (invalid).
    """
    data: np.ndarray
    mask: np.ndarray                 # bool array, same shape as data
    shape: Tuple[int, ...]
    dtype: DType
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_data(
        data: Any,
        *,
        mask: Any = False,
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "MaskedTensor":
        dt = get_dtype(dtype) if dtype is not None else None
        arr = np.asarray(data, dtype=dt.np_dtype if dt else None)
        m = np.asarray(mask, dtype=bool)

        # allow scalar mask broadcast
        if m.shape == ():
            m = np.full(arr.shape, bool(m), dtype=bool)

        if m.shape != arr.shape:
            raise ValueError(f"mask shape {m.shape} must match data shape {arr.shape}")

        inferred = dt or get_dtype(arr.dtype)
        return MaskedTensor(data=arr, mask=m, shape=tuple(arr.shape), dtype=inferred, meta=meta or {})

    @staticmethod
    def from_dense(
        array: Any,
        *,
        mask: Any = False,
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "MaskedTensor":
        # alias for clarity
        return MaskedTensor.from_data(array, mask=mask, dtype=dtype, meta=meta)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numpy(self, *, fill_value: Any = 0) -> np.ndarray:
        """
        Return a dense ndarray with masked entries replaced by fill_value.
        """
        out = np.asarray(self.data).copy()
        out[self.mask] = fill_value
        return out

    def to_masked_array(self, *, fill_value: Any = None) -> np.ma.MaskedArray:
        """
        Convert to np.ma.MaskedArray (NumPy's built-in masked array type).
        """
        ma = np.ma.array(self.data, mask=self.mask, copy=True)
        if fill_value is not None:
            ma.set_fill_value(fill_value)
        return ma

    def valid(self) -> np.ndarray:
        """Boolean array where True means unmasked/valid."""
        return ~self.mask

    def copy(self) -> "MaskedTensor":
        return MaskedTensor(
            data=self.data.copy(),
            mask=self.mask.copy(),
            shape=self.shape,
            dtype=self.dtype,
            meta=dict(self.meta),
        )

    # --- elementwise ops (mask-propagating) ---

    def _binary_op(self, other: Any, op) -> "MaskedTensor":
        if isinstance(other, MaskedTensor):
            if self.shape != other.shape:
                raise ValueError("Shapes must match for MaskedTensor ops")
            out_data = op(self.data, other.data)
            out_mask = self.mask | other.mask
            return MaskedTensor.from_data(out_data, mask=out_mask, dtype=self.dtype, meta=self.meta)

        if np.isscalar(other) or isinstance(other, np.ndarray):
            out_data = op(self.data, other)
            return MaskedTensor.from_data(out_data, mask=self.mask, dtype=self.dtype, meta=self.meta)

        raise TypeError(f"Unsupported type {type(other)} for MaskedTensor operation")

    def __add__(self, other): return self._binary_op(other, np.add)
    def __sub__(self, other): return self._binary_op(other, np.subtract)
    def __mul__(self, other): return self._binary_op(other, np.multiply)
    def __truediv__(self, other): return self._binary_op(other, np.divide)

    def __repr__(self) -> str:
        masked = int(self.mask.sum())
        return f"MaskedTensor(shape={self.shape}, masked={masked}, dtype={self.dtype.name})"

@dataclass(frozen=True, slots=True)
class IntervalTensor:
    """
    Interval tensor: each element is [lo, hi].

    Storage:
      - lo: ndarray
      - hi: ndarray
    with invariants lo <= hi elementwise (after normalization).

    Interval arithmetic aims to propagate bounds that contain the exact result.
    """
    lo: np.ndarray
    hi: np.ndarray
    shape: Tuple[int, ...]
    dtype: DType
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_bounds(
        lo,
        hi,
        *,
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
        normalize: bool = True,
    ) -> "IntervalTensor":
        dt = get_dtype(dtype) if dtype is not None else None
        a = np.asarray(lo, dtype=dt.np_dtype if dt else None)
        b = np.asarray(hi, dtype=dt.np_dtype if dt else None)

        if a.shape != b.shape:
            raise ValueError(f"lo shape {a.shape} != hi shape {b.shape}")

        inferred = dt or get_dtype(a.dtype)

        if normalize:
            lo2 = np.minimum(a, b)
            hi2 = np.maximum(a, b)
        else:
            lo2, hi2 = a, b
            if np.any(lo2 > hi2):
                raise ValueError("Invalid interval: lo > hi found and normalize=False")

        return IntervalTensor(lo=lo2, hi=hi2, shape=tuple(lo2.shape), dtype=inferred, meta=meta or {})

    @staticmethod
    def from_value_error(
        value,
        error,
        *,
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
        abs_error: bool = True,
    ) -> "IntervalTensor":
        """
        Build intervals from value ± error (default: absolute error).
        This matches the common 'x ± u' interval representation in uncertainty contexts.
        """
        dt = get_dtype(dtype) if dtype is not None else None
        v = np.asarray(value, dtype=dt.np_dtype if dt else None)
        e = np.asarray(error, dtype=dt.np_dtype if dt else None)

        if abs_error:
            e = np.abs(e)

        return IntervalTensor.from_bounds(v - e, v + e, dtype=dt, meta=meta, normalize=True)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def mid(self) -> np.ndarray:
        """Midpoint estimate."""
        return (self.lo + self.hi) / 2

    def radius(self) -> np.ndarray:
        """Half-width."""
        return (self.hi - self.lo) / 2

    def numpy(self) -> np.ndarray:
        """
        Convenience: return midpoint as dense numeric array.
        (You could also choose to raise to force explicit mid/lo/hi usage.)
        """
        return self.mid()

    # --- Interval arithmetic (basic) ---

    def _as_interval(self, other: Any) -> "IntervalTensor":
        if isinstance(other, IntervalTensor):
            return other
        # treat scalar/ndarray as point interval [x, x]
        x = np.asarray(other, dtype=self.dtype.np_dtype)
        return IntervalTensor.from_bounds(x, x, dtype=self.dtype)

    def __add__(self, other: Any) -> "IntervalTensor":
        o = self._as_interval(other)
        return IntervalTensor.from_bounds(self.lo + o.lo, self.hi + o.hi, dtype=self.dtype, normalize=False)

    def __sub__(self, other: Any) -> "IntervalTensor":
        o = self._as_interval(other)
        # [a,b] - [c,d] = [a-d, b-c]
        return IntervalTensor.from_bounds(self.lo - o.hi, self.hi - o.lo, dtype=self.dtype, normalize=False)

    def __mul__(self, other: Any) -> "IntervalTensor":
        o = self._as_interval(other)
        # [a,b]*[c,d] = [min(ac, ad, bc, bd), max(ac, ad, bc, bd)]
        ac = self.lo * o.lo
        ad = self.lo * o.hi
        bc = self.hi * o.lo
        bd = self.hi * o.hi
        lo = np.minimum(np.minimum(ac, ad), np.minimum(bc, bd))
        hi = np.maximum(np.maximum(ac, ad), np.maximum(bc, bd))
        return IntervalTensor.from_bounds(lo, hi, dtype=self.dtype, normalize=False)

    def __truediv__(self, other: Any) -> "IntervalTensor":
        o = self._as_interval(other)
        # Division is tricky if 0 is inside denominator interval; reject for safety.
        if np.any((o.lo <= 0) & (o.hi >= 0)):
            raise ZeroDivisionError("Interval division undefined when denominator spans 0")
        inv_lo = 1 / o.hi
        inv_hi = 1 / o.lo
        return self * IntervalTensor.from_bounds(inv_lo, inv_hi, dtype=self.dtype, normalize=True)

    def __repr__(self) -> str:
        return f"IntervalTensor(shape={self.shape}, dtype={self.dtype.name})"

@dataclass(frozen=True, slots=True)
class CPTensor:
    """
    CP / Kruskal tensor:
      X[i1,...,iN] = sum_{r=1..R} lambda[r] * Π_n U_n[i_n, r]
    Stores:
      - factors: list of N matrices U_n of shape (I_n, R)
      - lambdas: (R,) weights (optional; defaults to ones)
    """
    factors: Tuple[np.ndarray, ...]             # each (I_n, R)
    lambdas: np.ndarray                         # (R,)
    shape: Tuple[int, ...]
    dtype: DType
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_factors(
        factors: Sequence[Any],
        *,
        lambdas: Any | None = None,
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "CPTensor":
        dt = get_dtype(dtype) if dtype is not None else None
        mats = tuple(np.asarray(A, dtype=dt.np_dtype if dt else None) for A in factors)
        R, shape = _check_factors(mats)

        if lambdas is None:
            lam = np.ones((R,), dtype=mats[0].dtype)
        else:
            lam = np.asarray(lambdas, dtype=mats[0].dtype)
            if lam.shape != (R,):
                raise ValueError(f"lambdas must have shape ({R},), got {lam.shape}")

        inferred = dt or get_dtype(mats[0].dtype)
        return CPTensor(factors=mats, lambdas=lam, shape=shape, dtype=inferred, meta=meta or {})

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def rank(self) -> int:
        return int(self.lambdas.shape[0])

    def materialize(self) -> np.ndarray:
        """
        Dense reconstruction (expensive).
        """
        # Build by summing rank-1 outer products
        out = np.zeros(self.shape, dtype=self.dtype.np_dtype)
        R = self.rank
        for r in range(R):
            v = self.lambdas[r]
            # start with first mode vector
            core = self.factors[0][:, r]
            # successive outer products
            tmp = core
            for n in range(1, self.ndim):
                tmp = np.multiply.outer(tmp, self.factors[n][:, r])
            out += v * tmp
        return out

    def contract_last_with_vector(self, v: np.ndarray) -> "CPTensor":
        """
        Contract along last mode with a dense vector v of shape (I_last,).
        Result is another CP tensor with one fewer mode:
          new_lambda[r] = lambda[r] * <U_last[:,r], v>
        No full materialization needed.
        """
        v = np.asarray(v, dtype=self.dtype.np_dtype)
        Ilast = self.shape[-1]
        if v.shape != (Ilast,):
            raise ValueError(f"v must have shape ({Ilast},), got {v.shape}")

        Ulast = self.factors[-1]  # (Ilast, R)
        dots = Ulast.T @ v        # (R,)
        new_lam = self.lambdas * dots
        return CPTensor(
            factors=self.factors[:-1],
            lambdas=new_lam,
            shape=self.shape[:-1],
            dtype=self.dtype,
            meta=dict(self.meta),
        )

    def __repr__(self) -> str:
        return f"CPTensor(shape={self.shape}, rank={self.rank}, dtype={self.dtype.name})"

@dataclass(frozen=True, slots=True)
class TuckerTensor:
    """
    Tucker tensor:
      X = G x1 U1 x2 U2 ... xN UN
    where G is a core tensor and U_n are factor matrices.

    Stores:
      - core: ndarray shape (R1, R2, ..., RN)
      - factors: list of N matrices U_n shape (I_n, R_n)
    """
    core: np.ndarray
    factors: Tuple[np.ndarray, ...]            # each (I_n, R_n)
    shape: Tuple[int, ...]
    dtype: DType
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_core_factors(
        core: Any,
        factors: Sequence[Any],
        *,
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "TuckerTensor":
        dt = get_dtype(dtype) if dtype is not None else None
        G = np.asarray(core, dtype=dt.np_dtype if dt else None)
        mats = tuple(np.asarray(A, dtype=G.dtype) for A in factors)

        if len(mats) != G.ndim:
            raise ValueError("Number of factor matrices must match core.ndim")

        # check factor shapes (I_n, R_n) where R_n matches core dim
        shape: List[int] = []
        for n, A in enumerate(mats):
            if A.ndim != 2:
                raise ValueError(f"factor[{n}] must be 2D, got {A.ndim}D")
            In, Rn = A.shape
            if Rn != G.shape[n]:
                raise ValueError(f"factor[{n}] second dim {Rn} must match core.shape[{n}]={G.shape[n]}")
            shape.append(int(In))

        inferred = dt or get_dtype(G.dtype)
        return TuckerTensor(core=G, factors=mats, shape=tuple(shape), dtype=inferred, meta=meta or {})

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def ranks(self) -> Tuple[int, ...]:
        return tuple(int(r) for r in self.core.shape)

    def materialize(self) -> np.ndarray:
        """
        Dense reconstruction (expensive).
        Uses successive mode-n products implemented via tensordot+transpose.
        """
        X = self.core
        for mode, U in enumerate(self.factors):
            # X: (..., R_mode, ...) ; U: (I_mode, R_mode)
            X = np.tensordot(U, X, axes=(1, mode))  # new axis 0 is I_mode
            # Move new axis into correct position
            axes = list(range(1, mode + 1)) + [0] + list(range(mode + 1, X.ndim))
            X = np.transpose(X, axes)
        return np.asarray(X, dtype=self.dtype.np_dtype)

    def __repr__(self) -> str:
        return f"TuckerTensor(shape={self.shape}, ranks={self.ranks}, dtype={self.dtype.name})"

@dataclass(frozen=True, slots=True)
class ChronoTensor:
    """
    N-D tensor with a designated time axis.

    - time_axis: which axis is time (0..ndim-1)
    - data stored densely as np.ndarray
    """
    data: np.ndarray
    shape: Tuple[int, ...]
    dtype: DType
    time_axis: int
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_data(
        data: Any,
        *,
        time_axis: int = 0,
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "ChronoTensor":
        dt = get_dtype(dtype) if dtype is not None else None
        arr = np.asarray(data, dtype=dt.np_dtype if dt else None)
        if arr.ndim == 0:
            raise ValueError("ChronoTensor requires ndim >= 1")
        ta = int(time_axis)
        if not (0 <= ta < arr.ndim):
            raise ValueError(f"time_axis must be in [0, {arr.ndim-1}], got {time_axis}")
        inferred = dt or get_dtype(arr.dtype)
        return ChronoTensor(data=arr, shape=tuple(arr.shape), dtype=inferred, time_axis=ta, meta=meta or {})

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def T(self) -> int:
        """Length of time axis."""
        return int(self.shape[self.time_axis])

    def numpy(self) -> np.ndarray:
        return np.asarray(self.data)

    # ---- time operations ----

    def time_slice(self, slc: slice | int) -> "ChronoTensor":
        """
        Slice along time axis (keeps same time_axis index).
        """
        idx = [slice(None)] * self.ndim
        idx[self.time_axis] = slc
        out = self.data[tuple(idx)]
        # time_axis remains valid because slicing keeps axis (unless you pass an int)
        if isinstance(slc, int):
            # axis removed -> choose to forbid or re-wrap as plain Tensor-like
            raise ValueError("time_slice(int) would drop the time axis; use slice(...) instead.")
        return ChronoTensor(data=out, shape=tuple(out.shape), dtype=self.dtype, time_axis=self.time_axis, meta=dict(self.meta))

    def window_view(self, window: int, *, step: int = 1) -> np.ndarray:
        """
        Rolling window view along time axis.
        Returns a view with an extra axis for window contents.

        Uses numpy.lib.stride_tricks.sliding_window_view (rolling/moving window)
        """
        if window <= 0:
            raise ValueError("window must be > 0")
        if step <= 0:
            raise ValueError("step must be > 0")
        if window > self.T:
            raise ValueError("window cannot exceed time length")

        w = np.lib.stride_tricks.sliding_window_view(self.data, window_shape=window, axis=self.time_axis)
        idx = [slice(None)] * w.ndim
        idx[self.time_axis] = slice(None, None, step)
        return w[tuple(idx)]

    def mean_time(self, *, keepdims: bool = False) -> np.ndarray:
        """Mean over the time axis."""
        return np.mean(self.data, axis=self.time_axis, keepdims=keepdims)

    def sum_time(self, *, keepdims: bool = False) -> np.ndarray:
        """Sum over the time axis."""
        return np.sum(self.data, axis=self.time_axis, keepdims=keepdims)

    def __repr__(self) -> str:
        return f"ChronoTensor(shape={self.shape}, time_axis={self.time_axis}, dtype={self.dtype.name})"

@dataclass(frozen=True, slots=True)
class AdjacencyTensor:
    """
    Graph/network tensor built on top of NDSparseTensor (COO).

    shape convention:
      shape = (num_nodes, num_nodes, *edge_feature_shape)

    coords convention (COO):
      coords[0, k] = src node index
      coords[1, k] = dst node index
      coords[2:,k] = optional feature indices (e.g., relation id, time id, channel)

    weights:
      stored in sparse.data (float/int/bool), so unweighted graphs can use 1s.
    """
    sparse: NDSparseTensor
    num_nodes: int
    directed: bool = True
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_edges(
        edges: Sequence[Tuple[int, int]] | np.ndarray,
        *,
        num_nodes: int,
        weights: Any | None = None,
        edge_feature_coords: Optional[np.ndarray] = None,
        edge_feature_shape: Tuple[int, ...] = (),
        dtype: DTypeLike | None = None,
        directed: bool = True,
        coalesce: bool = True,
        meta: Dict[str, Any] | None = None,
    ) -> "AdjacencyTensor":
        """
        Build adjacency tensor from edge list.

        - edges: (E,2) pairs (src,dst)
        - edge_feature_coords: optional int64 array shape (F, E) for feature indices per edge
        """
        dt = get_dtype(dtype) if dtype is not None else None

        e = np.asarray(edges, dtype=np.int64)
        if e.ndim != 2 or e.shape[1] != 2:
            raise ValueError("edges must have shape (E, 2)")
        E = int(e.shape[0])

        if weights is None:
            w = np.ones((E,), dtype=(dt.np_dtype if dt else np.float32))
        else:
            w = np.asarray(weights, dtype=(dt.np_dtype if dt else None))
            if w.shape != (E,):
                raise ValueError(f"weights must have shape ({E},), got {w.shape}")

        src = e[:, 0]
        dst = e[:, 1]

        if edge_feature_coords is None:
            coords = np.vstack([src, dst])  # (2, E)
            shape = (int(num_nodes), int(num_nodes))
        else:
            fc = np.asarray(edge_feature_coords, dtype=np.int64)
            if fc.ndim != 2 or fc.shape[1] != E:
                raise ValueError("edge_feature_coords must have shape (F, E)")
            coords = np.vstack([src, dst, fc])  # (2+F, E)
            shape = (int(num_nodes), int(num_nodes), *tuple(edge_feature_shape))

        sp = NDSparseTensor.from_coo(
            coords,
            w,
            shape=tuple(shape),
            dtype=dt,
            coalesce=coalesce,
            check_bounds=True,
        )
        return AdjacencyTensor(sparse=sp, num_nodes=int(num_nodes), directed=directed, meta=meta or {})

    def to_sparse(self) -> NDSparseTensor:
        return self.sparse

    def to_dense(self) -> np.ndarray:
        return self.sparse.numpy()

    def symmetrize(self, *, reduce: str = "sum") -> "AdjacencyTensor":
        """
        For undirected graphs: mirror edges.
        If reduce='sum', duplicates will be summed by coalesce_sum (standard COO behavior)
        """
        if not self.directed:
            return self

        coords = self.sparse.coords
        data = self.sparse.data

        coords_T = coords.copy()
        coords_T[0, :], coords_T[1, :] = coords[1, :], coords[0, :]

        new_coords = np.concatenate([coords, coords_T], axis=1)
        new_data = np.concatenate([data, data], axis=0)

        sp2 = NDSparseTensor.from_coo(
            new_coords,
            new_data,
            shape=self.sparse.shape,
            dtype=self.sparse.dtype,
            coalesce=True,
            check_bounds=False,
        )
        return AdjacencyTensor(sparse=sp2, num_nodes=self.num_nodes, directed=False, meta=dict(self.meta))

    def __repr__(self) -> str:
        return f"AdjacencyTensor(num_nodes={self.num_nodes}, shape={self.sparse.shape}, nnz={self.sparse.nnz}, directed={self.directed})"

@dataclass(frozen=True, slots=True)
class ComplexTensor:
    """
    Dense tensor optimized/guarded for complex dtypes (complex64/complex128).
    """
    data: np.ndarray
    shape: Tuple[int, ...]
    dtype: DType
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_data(data: Any, *, dtype: DTypeLike | None = None, meta: Dict[str, Any] | None = None) -> "ComplexTensor":
        dt = get_dtype(dtype) if dtype is not None else None

        # Default to complex128 if not specified
        np_dt = dt.np_dtype if dt is not None else np.dtype(np.complex128)
        arr = np.asarray(data, dtype=np_dt)

        if not _is_complex_dtype(arr.dtype):
            raise TypeError(f"ComplexTensor requires complex dtype, got {arr.dtype!r}")

        inferred = dt or get_dtype(arr.dtype)
        return ComplexTensor(data=arr, shape=tuple(arr.shape), dtype=inferred, meta=meta or {})

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numpy(self) -> np.ndarray:
        return np.asarray(self.data)

    def conj(self) -> "ComplexTensor":
        return ComplexTensor.from_data(np.conjugate(self.data), dtype=self.dtype, meta=dict(self.meta))

    def T(self) -> "ComplexTensor":
        """Transpose (all axes reversed, NumPy .T behavior for ndarrays)."""
        return ComplexTensor.from_data(self.data.T, dtype=self.dtype, meta=dict(self.meta))

    def H(self) -> "ComplexTensor":
        """Hermitian conjugate (conjugate transpose)."""
        return ComplexTensor.from_data(np.conjugate(self.data).T, dtype=self.dtype, meta=dict(self.meta))

    def _bin(self, other: Any, op) -> "ComplexTensor":
        b = other.data if isinstance(other, ComplexTensor) else other
        out = op(self.data, b)
        return ComplexTensor.from_data(out, dtype=self.dtype, meta=dict(self.meta))

    def __add__(self, other): return self._bin(other, np.add)
    def __sub__(self, other): return self._bin(other, np.subtract)
    def __mul__(self, other): return self._bin(other, np.multiply)
    def __truediv__(self, other): return self._bin(other, np.divide)

    def vdot(self, other: "ComplexTensor") -> complex:
        """
        Vector dot product using np.vdot which conjugates the first argument.
        Note: np.vdot flattens inputs; use only when that's intended.
        """
        return np.vdot(self.data, other.data)

    def matmul(self, other: "ComplexTensor") -> "ComplexTensor":
        out = self.data @ other.data
        return ComplexTensor.from_data(out, dtype=self.dtype, meta=dict(self.meta))

    def __repr__(self) -> str:
        return f"ComplexTensor(shape={self.shape}, dtype={self.dtype.name})"

@dataclass(frozen=True, slots=True)
class QuantumTensor:
    """
    Quantum-focused wrapper around ComplexTensor.

    Conventions:
      - ket: shape (N,) or (N,1)
      - bra: Hermitian conjugate of ket
      - operator: shape (N,N)
    """
    x: ComplexTensor
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_data(data: Any, *, dtype: DTypeLike | None = None, meta: Dict[str, Any] | None = None) -> "QuantumTensor":
        return QuantumTensor(x=ComplexTensor.from_data(data, dtype=dtype, meta=meta), meta=meta or {})

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.x.shape

    def numpy(self) -> np.ndarray:
        return self.x.numpy()

    def dagger(self) -> "QuantumTensor":
        """Hermitian conjugate / adjoint."""
        return QuantumTensor(x=self.x.H(), meta=dict(self.meta))

    def kron(self, other: "QuantumTensor") -> "QuantumTensor":
        """
        Tensor/Kronecker product using numpy.kron.
        """
        out = np.kron(self.x.data, other.x.data)
        return QuantumTensor(x=ComplexTensor.from_data(out, dtype=self.x.dtype), meta=dict(self.meta))

    def expectation(self, op: "QuantumTensor") -> complex:
        """
        Expectation value <psi|A|psi>.

        Uses:
          - vdot for <psi|v> which conjugates first argument.
        """
        psi = self.x.data
        A = op.x.data

        # normalize common shapes: allow (N,1) kets
        if psi.ndim == 2 and psi.shape[1] == 1:
            psi = psi[:, 0]
        if psi.ndim != 1:
            raise ValueError("State must be a vector (N,) or (N,1)")
        N = psi.shape[0]
        if A.shape != (N, N):
            raise ValueError(f"Operator must have shape {(N, N)}, got {A.shape}")

        v = A @ psi
        return np.vdot(psi, v)

    def __repr__(self) -> str:
        return f"QuantumTensor(shape={self.shape}, dtype={self.x.dtype.name})"
