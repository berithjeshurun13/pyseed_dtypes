# pyseed_dtypes/categorical.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ._tensors import Tensor
from ..explict import get_dtype

__all__ = ["LabelTensor", "CategoryTensor", "OneHotTensor"]


def _as_1d(a, *, name: str) -> np.ndarray:
    x = np.asarray(a)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={x.shape}")
    return x


@dataclass(frozen=True, slots=True)
class LabelTensor:
    """
    Raw labels (strings/ints/etc.) for ML pipelines.

    - labels: object array shape (N,)
    """
    labels: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_data(labels: Any, *, meta: Dict[str, Any] | None = None) -> "LabelTensor":
        x = _as_1d(labels, name="labels").astype(object, copy=False)
        return LabelTensor(labels=x, meta=meta or {})

    @property
    def n(self) -> int:
        return int(self.labels.shape[0])

    def numpy(self) -> np.ndarray:
        return np.asarray(self.labels, dtype=object)

    def unique(self) -> np.ndarray:
        return np.unique(self.labels)

    def to_categories(self, *, categories: Sequence[Any] | None = None) -> "CategoryTensor":
        """
        Map raw labels -> integer codes using either:
          - provided categories (fixed vocabulary)
          - inferred categories via np.unique(..., return_inverse=True)
        """
        x = self.labels
        if categories is None:
            cats, inv = np.unique(x, return_inverse=True)
            return CategoryTensor.from_codes(inv, categories=tuple(cats.tolist()), meta=dict(self.meta))

        cats = tuple(categories)
        index = {c: i for i, c in enumerate(cats)}
        codes = np.empty((x.shape[0],), dtype=np.int64)
        for i, v in enumerate(x):
            if v not in index:
                raise KeyError(f"Unknown category {v!r} (not in provided categories)")
            codes[i] = index[v]
        return CategoryTensor.from_codes(codes, categories=cats, meta=dict(self.meta))

@dataclass(frozen=True, slots=True)
class CategoryTensor:
    """
    Integer-coded categories.

    - codes: int64 array shape (N,)
    - categories: tuple of category values, where categories[code] gives the label.
    """
    codes: np.ndarray
    categories: Tuple[Any, ...]
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_codes(
        codes: Any,
        *,
        categories: Sequence[Any],
        meta: Dict[str, Any] | None = None,
        validate: bool = True,
    ) -> "CategoryTensor":
        c = _as_1d(codes, name="codes").astype(np.int64, copy=False)
        cats = tuple(categories)
        if len(cats) == 0:
            raise ValueError("categories cannot be empty")

        if validate and c.size:
            if c.min() < 0 or c.max() >= len(cats):
                raise IndexError("codes contain out-of-range category id")

        return CategoryTensor(codes=c, categories=cats, meta=meta or {})

    @property
    def n(self) -> int:
        return int(self.codes.shape[0])

    @property
    def num_classes(self) -> int:
        return int(len(self.categories))

    def numpy(self) -> np.ndarray:
        return np.asarray(self.codes, dtype=np.int64)

    def to_labels(self) -> LabelTensor:
        labs = np.asarray([self.categories[i] for i in self.codes], dtype=object)
        return LabelTensor.from_data(labs, meta=dict(self.meta))

    def one_hot(self, *, dtype=np.float32) -> "OneHotTensor":
        """
        Create one-hot matrix of shape (N, C).
        One-hot encoding creates a binary column per category.
        """
        N, C = self.n, self.num_classes
        oh = np.zeros((N, C), dtype=dtype)
        oh[np.arange(N), self.codes] = 1
        return OneHotTensor.from_numpy(oh, categories=self.categories, meta=dict(self.meta))

    def confusion_matrix(
        self,
        pred: "CategoryTensor",
        *,
        normalize: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute confusion matrix counts (C x C).
        This matches the standard y_true/y_pred interpretation used by sklearn
        normalize: None | 'true' | 'pred' | 'all'
        """
        if self.categories != pred.categories:
            raise ValueError("true/pred categories must match exactly")
        if self.n != pred.n:
            raise ValueError("true/pred must have same length")

        C = self.num_classes
        cm = np.zeros((C, C), dtype=np.int64)
        t = self.codes
        p = pred.codes
        np.add.at(cm, (t, p), 1)

        if normalize is None:
            return cm
        cmf = cm.astype(np.float64)
        if normalize == "true":
            denom = cmf.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            denom = cmf.sum(axis=0, keepdims=True)
        elif normalize == "all":
            denom = cmf.sum()
        else:
            raise ValueError("normalize must be None, 'true', 'pred', or 'all'")
        return cmf / np.maximum(denom, 1e-12)

    def __repr__(self) -> str:
        return f"CategoryTensor(n={self.n}, classes={self.num_classes})"

@dataclass(frozen=True, slots=True)
class OneHotTensor:
    """
    One-hot encoded representation.

    - data: float/bool array shape (N, C)
    - categories: tuple describing class index meaning
    """
    data: Tensor
    categories: Tuple[Any, ...]
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_numpy(
        array: Any,
        *,
        categories: Sequence[Any],
        meta: Dict[str, Any] | None = None,
    ) -> "OneHotTensor":
        x = np.asarray(array)
        if x.ndim != 2:
            raise ValueError("one-hot array must be 2D (N, C)")
        cats = tuple(categories)
        if x.shape[1] != len(cats):
            raise ValueError("one-hot second dim must match number of categories")
        t = Tensor.from_data(x, dtype=get_dtype(x.dtype), meta=meta)
        return OneHotTensor(data=t, categories=cats, meta=meta or {})

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.data.shape)  # (N, C)

    def numpy(self) -> np.ndarray:
        return self.data.numpy()

    def argmax(self) -> CategoryTensor:
        codes = np.argmax(self.data.numpy(), axis=1).astype(np.int64)
        return CategoryTensor.from_codes(codes, categories=self.categories, meta=dict(self.meta))

    def __repr__(self) -> str:
        n, c = self.shape
        return f"OneHotTensor(n={n}, classes={c})"
