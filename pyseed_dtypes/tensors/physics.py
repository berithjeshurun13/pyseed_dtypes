# pyseed_dtypes/physics.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ._tensors import Tensor
from ..explict import DTypeLike, get_dtype

__all__ = ["QuantityTensor", "FieldKind", "FieldTensor"]


def _try_import_pint():
    try:
        import pint  # type: ignore
        return pint
    except Exception:
        return None

class FieldKind(Enum):
    SCALAR = auto()
    VECTOR = auto()
    TENSOR = auto()


@dataclass(frozen=True, slots=True)
class QuantityTensor:
    """
    Physical quantity = numeric tensor + unit + (optional) dimension label.

    Example unit string: "m/s^2" (acceleration is commonly written m/s^2).
    For real unit conversions, Pint is the recommended backend.
    """
    tensor: Tensor
    unit: str
    dimension: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_data(
        data: Any,
        *,
        unit: str,
        dimension: str | None = None,
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "QuantityTensor":
        dt = get_dtype(dtype) if dtype is not None else None
        t = Tensor.from_data(data, dtype=dt, meta=meta)
        return QuantityTensor(tensor=t, unit=str(unit), dimension=dimension, meta=meta or {})

    def numpy(self) -> np.ndarray:
        return self.tensor.numpy()

    def to_pint(self):
        """
        Return a Pint Quantity wrapping the underlying ndarray.
        Pint represents physical quantities as value * unit.
        """
        pint = _try_import_pint()
        if pint is None:
            raise ImportError("Pint is required: pip install pint")
        ureg = pint.UnitRegistry()
        return self.tensor.numpy() * ureg(self.unit)

    def to(self, unit: str) -> "QuantityTensor":
        """
        Convert to another unit using Pint.
        """
        q = self.to_pint()
        q2 = q.to(unit)  # Pint quantity conversion
        arr = np.asarray(q2.magnitude)
        return QuantityTensor.from_data(arr, unit=str(q2.units), dimension=self.dimension, dtype=get_dtype(arr.dtype), meta=dict(self.meta))

    # unit-aware arithmetic (strict: requires same unit unless using Pint)
    def _require_same_unit(self, other: "QuantityTensor"):
        if self.unit != other.unit:
            raise ValueError(f"Unit mismatch: {self.unit!r} vs {other.unit!r} (convert first)")

    def __add__(self, other: "QuantityTensor") -> "QuantityTensor":
        self._require_same_unit(other)
        return QuantityTensor.from_data(self.tensor.numpy() + other.tensor.numpy(), unit=self.unit, dimension=self.dimension, dtype=get_dtype(self.tensor.numpy().dtype), meta=dict(self.meta))

    def __sub__(self, other: "QuantityTensor") -> "QuantityTensor":
        self._require_same_unit(other)
        return QuantityTensor.from_data(self.tensor.numpy() - other.tensor.numpy(), unit=self.unit, dimension=self.dimension, dtype=get_dtype(self.tensor.numpy().dtype), meta=dict(self.meta))

    def __mul__(self, other: Any) -> "QuantityTensor":
        # keep it simple: scalar multiply does not change unit (full dimensional analysis is Pint's job)
        if np.isscalar(other):
            return QuantityTensor.from_data(self.tensor.numpy() * other, unit=self.unit, dimension=self.dimension, dtype=get_dtype(self.tensor.numpy().dtype), meta=dict(self.meta))
        raise TypeError("QuantityTensor multiplication supports only scalar in this minimal datatype")

    def __truediv__(self, other: Any) -> "QuantityTensor":
        if np.isscalar(other):
            return QuantityTensor.from_data(self.tensor.numpy() / other, unit=self.unit, dimension=self.dimension, dtype=get_dtype(self.tensor.numpy().dtype), meta=dict(self.meta))
        raise TypeError("QuantityTensor division supports only scalar in this minimal datatype")

    def __repr__(self) -> str:
        dim = f", dimension={self.dimension!r}" if self.dimension else ""
        return f"QuantityTensor(shape={self.tensor.shape}, unit={self.unit!r}{dim})"

@dataclass(frozen=True, slots=True)
class FieldTensor:
    """
    Field = quantity distributed over space/time.

    Example shapes:
      - scalar field on grid: (H, W) or (D, H, W)
      - vector field: (H, W, 3) with components_axis=-1

    components_axis defines where the vector/tensor components live.
    """
    quantity: QuantityTensor
    kind: FieldKind = FieldKind.SCALAR
    components_axis: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        x = self.quantity.tensor.numpy()
        if self.kind == FieldKind.SCALAR:
            if self.components_axis is not None:
                raise ValueError("Scalar field must not set components_axis")
        else:
            if self.components_axis is None:
                raise ValueError("Vector/tensor field requires components_axis")
            ax = int(self.components_axis)
            if not (-x.ndim <= ax < x.ndim):
                raise ValueError("components_axis out of range")

    @property
    def tensor(self) -> Tensor:
        return self.quantity.tensor

    @property
    def unit(self) -> str:
        return self.quantity.unit

    def numpy(self) -> np.ndarray:
        return self.quantity.numpy()

    def magnitude(self) -> QuantityTensor:
        """
        For vector fields: return magnitude (norm) as a scalar QuantityTensor.
        """
        if self.kind != FieldKind.VECTOR:
            raise ValueError("magnitude() is defined for VECTOR fields only")
        x = self.numpy()
        ax = int(self.components_axis)
        mag = np.linalg.norm(x, axis=ax)
        return QuantityTensor.from_data(mag, unit=self.unit, dimension=self.quantity.dimension, dtype=get_dtype(mag.dtype), meta=dict(self.meta))

    def __repr__(self) -> str:
        return f"FieldTensor(kind={self.kind.name}, unit={self.unit!r}, shape={self.quantity.tensor.shape})"
