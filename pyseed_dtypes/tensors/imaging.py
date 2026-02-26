# pyseed_dtypes/imaging.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from PIL import Image, ImageOps, ImageDraw  # type: ignore

from ..explict import DType, DTypeLike, get_dtype
from ._tensors import Tensor

__all__ = [
    "ColorSpace",
    "ImageLayout",
    "PixelFormat",
    "BoxFormat",
    "ImageTensor",
    "PatchTensor",
    "PyramidTensor",
    "BBoxTensor",
    "DepthMapTensor"
]


class ColorSpace(Enum):
    GRAY = auto()   # PIL "L"
    RGB = auto()    # PIL "RGB"
    RGBA = auto()   # PIL "RGBA"
    BGR = auto()
    BGRA = auto()
    HSV = auto()
    LAB = auto()
    YUV = auto()


class ImageLayout(Enum):
    HWC = auto()   # height, width, channels (NumPy/PIL-friendly)
    CHW = auto()   # channels, height, width (torch-style)


class PixelFormat(Enum):
    UINT8 = auto()     # 0..255
    UINT16 = auto()    # 0..65535
    FLOAT32 = auto()   # typically 0..1

class BoxFormat(Enum):
    XYXY = auto()  # (x0, y0, x1, y1)
    XYWH = auto()  # (x0, y0, w, h)  (COCO-style)



_PF_TO_NP = {
    PixelFormat.UINT8: np.dtype(np.uint8),
    PixelFormat.UINT16: np.dtype(np.uint16),
    PixelFormat.FLOAT32: np.dtype(np.float32),
}


_CS_TO_PIL_MODE = {
    ColorSpace.GRAY: "L",
    ColorSpace.RGB: "RGB",
    ColorSpace.RGBA: "RGBA",
}


def _ensure_hwc(arr: np.ndarray, layout: ImageLayout) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError("Expected 2D or 3D array for image data")
    if layout == ImageLayout.HWC:
        return arr
    return np.transpose(arr, (1, 2, 0))  # CHW -> HWC


def _ensure_chw(arr: np.ndarray, layout: ImageLayout) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError("Expected 2D or 3D array for image data")
    if layout == ImageLayout.CHW:
        return arr
    return np.transpose(arr, (2, 0, 1))  # HWC -> CHW


def _normalize_float01_to_uint8(arr: np.ndarray) -> np.ndarray:
    # clips to [0,1] and scales to [0,255]
    a = np.clip(arr, 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class ImageTensor:
    """
    Core image datatype: wraps a dense Tensor + image semantics.

    Invariants:
      - layout describes how tensor axes map to (H,W,C)
      - colorspace controls channel expectations for common modes
      - pixel_format must match tensor numpy dtype (enforced)
    """
    tensor: Tensor
    colorspace: ColorSpace
    layout: ImageLayout = ImageLayout.HWC
    pixel_format: PixelFormat = PixelFormat.UINT8
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        t = self.tensor
        if t.ndim not in (2, 3):
            raise ValueError("ImageTensor must be 2D (gray) or 3D (color)")

        expected_np = _PF_TO_NP[self.pixel_format]
        actual_np = self.tensor.numpy().dtype
        if actual_np != expected_np:
            raise TypeError(f"pixel_format={self.pixel_format.name} requires dtype={expected_np}, got {actual_np}")

        if t.ndim == 2:
            if self.colorspace != ColorSpace.GRAY:
                raise ValueError("2D image must use GRAY colorspace")
            return

        # 3D: validate channels
        if self.layout == ImageLayout.HWC:
            _, _, c = t.shape
        else:
            c, _, _ = t.shape

        expected_c = {
            ColorSpace.GRAY: 1,
            ColorSpace.RGB: 3,
            ColorSpace.BGR: 3,
            ColorSpace.RGBA: 4,
            ColorSpace.BGRA: 4,
        }.get(self.colorspace)

        if expected_c is not None and c != expected_c:
            raise ValueError(f"{self.colorspace.name} expects {expected_c} channels, got {c}")

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.tensor.shape

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    @property
    def height_width(self) -> Tuple[int, int]:
        if self.tensor.ndim == 2:
            h, w = self.tensor.shape
            return int(h), int(w)
        if self.layout == ImageLayout.HWC:
            h, w, _ = self.tensor.shape
        else:
            _, h, w = self.tensor.shape
        return int(h), int(w)

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        *,
        colorspace: ColorSpace,
        layout: ImageLayout = ImageLayout.HWC,
        pixel_format: PixelFormat = PixelFormat.UINT8,
        meta: Dict[str, Any] | None = None,
    ) -> "ImageTensor":
        np_dt = _PF_TO_NP[pixel_format]
        t = Tensor.from_data(np.asarray(array, dtype=np_dt), dtype=get_dtype(np_dt), meta=meta)
        return ImageTensor(tensor=t, colorspace=colorspace, layout=layout, pixel_format=pixel_format, meta=meta or {})

    @staticmethod
    def from_pil(
        img: Image.Image,
        *,
        colorspace: ColorSpace | None = None,
        layout: ImageLayout = ImageLayout.HWC,
        pixel_format: PixelFormat = PixelFormat.UINT8,
        exif_transpose: bool = True,
        meta: Dict[str, Any] | None = None,
    ) -> "ImageTensor":
        if exif_transpose:
            # Fix orientation based on EXIF Orientation tag. [Pillow ImageOps]
            img = ImageOps.exif_transpose(img)

        if colorspace is None:
            # infer from PIL mode
            inv = {v: k for k, v in _CS_TO_PIL_MODE.items()}
            colorspace = inv.get(img.mode, ColorSpace.RGB)

        pil_mode = _CS_TO_PIL_MODE.get(colorspace)
        if pil_mode is None:
            raise ValueError(f"Unsupported colorspace for from_pil: {colorspace.name}")

        if img.mode != pil_mode:
            img = img.convert(pil_mode)  # conversion between modes is standard in Pillow

        arr = np.asarray(img)

        # ensure pixel format
        if pixel_format == PixelFormat.UINT8:
            arr = np.asarray(arr, dtype=np.uint8)
        elif pixel_format == PixelFormat.UINT16:
            arr = np.asarray(arr, dtype=np.uint16)
        else:
            # FLOAT32 in 0..1
            arr = np.asarray(arr, dtype=np.float32) / 255.0

        if layout == ImageLayout.CHW and arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))

        return ImageTensor.from_numpy(arr, colorspace=colorspace, layout=layout, pixel_format=pixel_format, meta=meta)

    def numpy(self, *, layout: ImageLayout | None = None) -> np.ndarray:
        arr = self.tensor.numpy()
        if layout is None or layout == self.layout:
            return arr
        return _ensure_chw(arr, self.layout) if layout == ImageLayout.CHW else _ensure_hwc(arr, self.layout)

    def to_pil(self) -> Image.Image:
        # PIL primarily expects HWC for multi-band images like RGB/RGBA.
        arr = self.numpy(layout=ImageLayout.HWC)

        mode = _CS_TO_PIL_MODE.get(self.colorspace)
        if mode is None:
            raise ValueError(f"PIL conversion not supported for {self.colorspace.name}")

        # Ensure dtype for PIL modes (keep it strict)
        if self.pixel_format == PixelFormat.FLOAT32:
            # interpret as 0..1 floats -> convert to uint8 for PIL modes
            arr = _normalize_float01_to_uint8(arr)

        if arr.dtype != np.uint8 and mode in ("L", "RGB", "RGBA"):
            # keep it strict and predictable for typical Pillow usage
            arr = arr.astype(np.uint8, copy=False)

        return Image.fromarray(arr, mode=mode)

    def convert(self, colorspace: ColorSpace) -> "ImageTensor":
        """
        Convert using Pillow as the canonical converter.
        """
        img = self.to_pil()
        target_mode = _CS_TO_PIL_MODE.get(colorspace)
        if target_mode is None:
            raise ValueError(f"convert() not supported for {colorspace.name}")
        img2 = img.convert(target_mode)
        return ImageTensor.from_pil(
            img2,
            colorspace=colorspace,
            layout=self.layout,
            pixel_format=self.pixel_format,
            exif_transpose=False,
            meta=dict(self.meta),
        )

    def resize(self, size: Tuple[int, int], *, resample: int | None = None) -> "ImageTensor":
        """
        Resize via Pillow.
        size = (width, height) like PIL.
        """
        img = self.to_pil()
        img2 = img.resize(size, resample=resample if resample is not None else Image.BILINEAR)
        return ImageTensor.from_pil(
            img2,
            colorspace=self.colorspace,
            layout=self.layout,
            pixel_format=self.pixel_format,
            exif_transpose=False,
            meta=dict(self.meta),
        )

    def __repr__(self) -> str:
        h, w = self.height_width
        return f"ImageTensor({h}x{w}, colorspace={self.colorspace.name}, layout={self.layout.name}, pf={self.pixel_format.name})"

@dataclass(frozen=True, slots=True)
class PatchTensor:
    """
    Sliding patch view over an ImageTensor (patches extracted from HWC data).

    Stores a 5D array view: (n_y, n_x, patch_h, patch_w, C) for color images
    or (n_y, n_x, patch_h, patch_w) for grayscale.
    """
    image: ImageTensor
    patch_size: Tuple[int, int]           # (patch_h, patch_w)
    stride: Tuple[int, int] = (1, 1)      # (sy, sx)
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def view(self) -> np.ndarray:
        arr = self.image.numpy(layout=ImageLayout.HWC)
        ph, pw = self.patch_size
        sy, sx = self.stride
        h, w = self.image.height_width

        if ph <= 0 or pw <= 0:
            raise ValueError("patch_size must be positive")
        if sy <= 0 or sx <= 0:
            raise ValueError("stride must be positive")
        if ph > h or pw > w:
            raise ValueError("patch_size larger than image")

        # sliding window view over (H,W) axes
        win = np.lib.stride_tricks.sliding_window_view(arr, window_shape=(ph, pw), axis=(0, 1))
        # win shape: (H-ph+1, W-pw+1, ph, pw, C?)  (C remains as is)
        return win[::sy, ::sx]

    def __repr__(self) -> str:
        return f"PatchTensor(patch_size={self.patch_size}, stride={self.stride}, image={self.image})"

@dataclass(frozen=True, slots=True)
class PyramidTensor:
    """
    Image pyramid (multi-scale representations) built via Pillow resize.
    """
    levels: Tuple[ImageTensor, ...]
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_image(
        img: ImageTensor,
        *,
        num_levels: int = 4,
        scale: float = 0.5,
        min_size: int = 8,
        resample: int | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "PyramidTensor":
        if num_levels <= 0:
            raise ValueError("num_levels must be > 0")
        if not (0.0 < scale < 1.0):
            raise ValueError("scale must be in (0,1)")

        levels: List[ImageTensor] = [img]
        cur = img
        for _ in range(1, num_levels):
            h, w = cur.height_width
            nh = max(int(h * scale), min_size)
            nw = max(int(w * scale), min_size)
            if nh == h and nw == w:
                break
            cur = cur.resize((nw, nh), resample=resample)
            levels.append(cur)

        return PyramidTensor(levels=tuple(levels), meta=meta or {})

    def __len__(self) -> int:
        return len(self.levels)

    def __getitem__(self, i: int) -> ImageTensor:
        return self.levels[i]

    def __repr__(self) -> str:
        return f"PyramidTensor(levels={len(self.levels)})"

@dataclass(frozen=True, slots=True)
class BBoxTensor:
    """
    Bounding boxes attached to an image coordinate system.

    Storage:
      - boxes: (N,4) float32
      - labels: (N,) optional (str or int); stored as Python list for flexibility
      - scores: (N,) optional float32

    Coordinates are in pixel space of the associated image (origin at top-left).
    """
    boxes: np.ndarray
    format: BoxFormat = BoxFormat.XYXY
    labels: Tuple[Any, ...] = ()
    scores: np.ndarray | None = None
    image_size: Tuple[int, int] | None = None   # (H, W) for validation/clipping
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_boxes(
        boxes: Any,
        *,
        format: BoxFormat = BoxFormat.XYXY,
        labels: Iterable[Any] | None = None,
        scores: Any | None = None,
        image_size: Tuple[int, int] | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "BBoxTensor":
        b = np.asarray(boxes, dtype=np.float32)
        if b.ndim != 2 or b.shape[1] != 4:
            raise ValueError("boxes must have shape (N,4)")
        lab = tuple(labels) if labels is not None else ()
        sc = None if scores is None else np.asarray(scores, dtype=np.float32)
        if sc is not None and sc.shape != (b.shape[0],):
            raise ValueError("scores must have shape (N,)")

        out = BBoxTensor(
            boxes=b,
            format=format,
            labels=lab,
            scores=sc,
            image_size=image_size,
            meta=meta or {},
        )
        out._validate()
        return out

    def _validate(self) -> None:
        if self.image_size is not None:
            h, w = self.image_size
            if h <= 0 or w <= 0:
                raise ValueError("image_size must be (H>0, W>0)")

    def to_xyxy(self) -> np.ndarray:
        b = self.boxes
        if self.format == BoxFormat.XYXY:
            return b
        # XYWH -> XYXY
        x, y, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return np.stack([x, y, x + bw, y + bh], axis=1)

    def clip_to_image(self) -> "BBoxTensor":
        if self.image_size is None:
            return self
        h, w = self.image_size
        b = self.to_xyxy().copy()
        b[:, 0] = np.clip(b[:, 0], 0, w - 1)
        b[:, 2] = np.clip(b[:, 2], 0, w - 1)
        b[:, 1] = np.clip(b[:, 1], 0, h - 1)
        b[:, 3] = np.clip(b[:, 3], 0, h - 1)
        return BBoxTensor.from_boxes(
            b,
            format=BoxFormat.XYXY,
            labels=self.labels if self.labels else None,
            scores=self.scores,
            image_size=self.image_size,
            meta=dict(self.meta),
        )

    def draw_on(
        self,
        image: "ImageTensor",
        *,
        color: Any = "red",
        width: int = 2,
        with_labels: bool = True,
    ) -> "ImageTensor":
        """
        Returns a new ImageTensor with boxes drawn (via PIL).
        """
        img = image.to_pil().copy()
        draw = ImageDraw.Draw(img)

        b = self.clip_to_image().to_xyxy()
        for i in range(b.shape[0]):
            x0, y0, x1, y1 = map(float, b[i])
            draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
            if with_labels and i < len(self.labels):
                draw.text((x0, y0), str(self.labels[i]), fill=color)

        return ImageTensor.from_pil(
            img,
            colorspace=image.colorspace,
            layout=image.layout,
            pixel_format=image.pixel_format,
            exif_transpose=False,
            meta=dict(image.meta),
        )

    def __repr__(self) -> str:
        return f"BBoxTensor(n={self.boxes.shape[0]}, format={self.format.name})"

@dataclass(frozen=True, slots=True)
class DepthMapTensor:
    """
    Single-channel depth map (float32), aligned to an image.

    Conventions:
      - depth values are in `unit` (default meters)
      - invalid/unknown depth uses `invalid_value` (default 0.0)
    """
    depth: Tensor                       # expected shape (H,W), float32
    unit: str = "m"
    invalid_value: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        d = self.depth.numpy()
        if d.ndim != 2:
            raise ValueError("DepthMapTensor requires shape (H,W)")
        if d.dtype != np.float32:
            raise TypeError(f"DepthMapTensor requires float32, got {d.dtype}")

    @property
    def shape(self) -> Tuple[int, int]:
        h, w = self.depth.shape
        return int(h), int(w)

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        *,
        unit: str = "m",
        invalid_value: float = 0.0,
        meta: Dict[str, Any] | None = None,
    ) -> "DepthMapTensor":
        arr = np.asarray(array, dtype=np.float32)
        t = Tensor.from_data(arr, dtype=get_dtype(np.dtype(np.float32)), meta=meta)
        return DepthMapTensor(depth=t, unit=unit, invalid_value=float(invalid_value), meta=meta or {})

    def valid_mask(self) -> np.ndarray:
        d = self.depth.numpy()
        return d != self.invalid_value

    def to_grayscale(
        self,
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        invert: bool = False,
    ) -> "ImageTensor":
        """
        Visualize depth as an 8-bit grayscale ImageTensor.
        Invalid pixels become black (0).
        """
        d = self.depth.numpy()
        mask = self.valid_mask()

        if vmin is None:
            vmin = float(np.min(d[mask])) if np.any(mask) else 0.0
        if vmax is None:
            vmax = float(np.max(d[mask])) if np.any(mask) else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6

        out = np.zeros_like(d, dtype=np.float32)
        out[mask] = (d[mask] - vmin) / (vmax - vmin)
        out = np.clip(out, 0.0, 1.0)
        if invert:
            out = 1.0 - out

        img_u8 = (out * 255.0 + 0.5).astype(np.uint8)
        return ImageTensor.from_numpy(
            img_u8,
            colorspace=ColorSpace.GRAY,
            layout=ImageLayout.HWC,
            pixel_format=PixelFormat.UINT8,
            meta=dict(self.meta),
        )

    def __repr__(self) -> str:
        h, w = self.shape
        return f"DepthMapTensor({h}x{w}, unit={self.unit!r}, dtype=float32)"
