# pyseed_dtypes/probability.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ._tensors import Tensor
from ..explict import DTypeLike, get_dtype

__all__ = ["DistributionTensor", "ProbabilityTensor", "LogitsTensor"]


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically-stable softmax:
      softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    y = np.exp(x - m)
    return y / np.sum(y, axis=axis, keepdims=True)


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Stable log-softmax computed as:
      log_softmax(x) = (x - max) - log(sum(exp(x - max))) 
    """
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


def _is_normalized(p: np.ndarray, axis: int, tol: float = 1e-5) -> bool:
    if np.any(p < -tol):
        return False
    s = np.sum(p, axis=axis)
    return np.all(np.isfinite(s)) and np.all(np.abs(s - 1.0) <= tol)


@dataclass(frozen=True, slots=True)
class LogitsTensor:
    """
    Unnormalized scores (logits) that can be converted to probabilities via softmax.
    """
    tensor: Tensor
    axis: int = -1
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_data(data: Any, *, axis: int = -1, dtype: DTypeLike | None = None, meta=None) -> "LogitsTensor":
        dt = get_dtype(dtype) if dtype is not None else None
        t = Tensor.from_data(data, dtype=dt, meta=meta)
        return LogitsTensor(tensor=t, axis=axis, meta=meta or {})

    def numpy(self) -> np.ndarray:
        return self.tensor.numpy()

    def softmax(self) -> "ProbabilityTensor":
        p = _softmax(self.numpy(), axis=self.axis)
        return ProbabilityTensor.from_data(p, axis=self.axis, normalized=True, meta=dict(self.meta))

    def log_softmax(self) -> np.ndarray:
        return _log_softmax(self.numpy(), axis=self.axis)

    def __repr__(self) -> str:
        return f"LogitsTensor(shape={self.tensor.shape}, axis={self.axis})"

@dataclass(frozen=True, slots=True)
class ProbabilityTensor:
    """
    Probability distribution stored in a Tensor.
    normalized=True means sum(prob, axis)=1 and prob>=0 (within tolerance).
    """
    tensor: Tensor
    normalized: bool = True
    axis: int = -1
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.normalized:
            p = self.tensor.numpy()
            if not _is_normalized(p, axis=self.axis):
                raise ValueError("ProbabilityTensor(normalized=True) but values are not a valid distribution")

    @staticmethod
    def from_data(
        data: Any,
        *,
        normalized: bool = True,
        axis: int = -1,
        dtype: DTypeLike | None = None,
        meta=None,
        auto_normalize: bool = False,
    ) -> "ProbabilityTensor":
        dt = get_dtype(dtype) if dtype is not None else None
        t = Tensor.from_data(data, dtype=dt, meta=meta)
        if normalized:
            if auto_normalize:
                # force normalization (clip negatives, renormalize)
                p = np.asarray(t.numpy(), dtype=np.float64)
                p = np.clip(p, 0.0, None)
                s = np.sum(p, axis=axis, keepdims=True)
                p = p / np.maximum(s, 1e-12)
                t = Tensor.from_data(p.astype(np.float32), dtype=get_dtype(np.dtype(np.float32)), meta=meta)
            else:
                # validated in __post_init__
                pass
        return ProbabilityTensor(tensor=t, normalized=normalized, axis=axis, meta=meta or {})

    def numpy(self) -> np.ndarray:
        return self.tensor.numpy()

    def entropy(self) -> np.ndarray:
        """
        Shannon entropy along axis: -sum(p * log(p)).
        """
        p = self.numpy()
        p = np.clip(p, 1e-12, 1.0)
        return -np.sum(p * np.log(p), axis=self.axis)

    def argmax(self) -> np.ndarray:
        return np.argmax(self.numpy(), axis=self.axis)

    def __repr__(self) -> str:
        return f"ProbabilityTensor(shape={self.tensor.shape}, axis={self.axis}, normalized={self.normalized})"

@dataclass(frozen=True, slots=True)
class DistributionTensor:
    """
    Generic wrapper that can store one of:
      - probs: ProbabilityTensor
      - logits: LogitsTensor

    Useful when models output logits but evaluation expects probabilities.
    """
    probs: Optional[ProbabilityTensor] = None
    logits: Optional[LogitsTensor] = None
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if (self.probs is None) == (self.logits is None):
            raise ValueError("DistributionTensor requires exactly one of probs or logits")

    @property
    def axis(self) -> int:
        return self.probs.axis if self.probs is not None else self.logits.axis

    def to_probs(self) -> ProbabilityTensor:
        return self.probs if self.probs is not None else self.logits.softmax()

    def to_logits(self) -> LogitsTensor:
        if self.logits is not None:
            return self.logits
        # Convert probs -> logits via log (not unique); choose log(p)
        p = np.clip(self.probs.numpy(), 1e-12, 1.0)
        return LogitsTensor.from_data(np.log(p), axis=self.probs.axis, meta=dict(self.meta))

    def __repr__(self) -> str:
        kind = "probs" if self.probs is not None else "logits"
        return f"DistributionTensor({kind}, axis={self.axis})"
