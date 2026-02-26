from __future__ import annotations

from dataclasses import (
    dataclass, 
    field
)
from typing import (
    Any, 
    Dict, 
    Optional, 
    Tuple, 
    Union
)
from PIL import Image as PILImage
from PIL import ImageEnhance, ImageOps
from enum import Enum, Flag, auto
import numpy as np
from .explict import PathLike
__all__ = [
    "sha1",
    "ImageMode",
    "Image",
    "PILImage"
    "PixelRegion",
]
class ImageMode(Enum):
    L = "L"
    RGB = "RGB"
    RGBA = "RGBA"
    CMYK = "CMYK"
    F = "F"  # 32-bit floating point pixels :]


@dataclass
class Image:
    """
    Production-grade wrapper around PIL.Image.Image.

    Goals:
      - deterministic file/resource behavior
      - explicit EXIF orientation policy
      - strong, typed convenience APIs
      - delegation to Pillow for all core ops
    """
    pil: PILImage.Image
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    @staticmethod
    def open(
        path: PathLike,
        *,
        exif_transpose: bool = True,
        load: bool = True,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "Image":
        """
        - Pillow open is lazy; load=True forces decode.
        - After Image.load(), Pillow may close the underlying file for single-frame images
        """
        img = PILImage.open(path)
        if exif_transpose:
            img = ImageOps.exif_transpose(img)
        if load:
            img.load()
        return Image(img, meta=meta or {"source": str(path)})

    @staticmethod
    def new(
        mode: str | ImageMode,
        size: Tuple[int, int],
        color: Any = 0,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "Image":
        mode_s = mode.value if isinstance(mode, ImageMode) else mode
        return Image(PILImage.new(mode_s, size, color=color), meta=meta or {})

    @staticmethod
    def from_pil(pil_img: PILImage.Image, *, meta: Optional[Dict[str, Any]] = None) -> "Image":
        return Image(pil_img, meta=meta or {})

    def __getattr__(self, name: str):
        # Delegate unknown attributes/methods to PIL.Image.Image
        return getattr(self.pil, name)

    @property
    def size_hw(self) -> Tuple[int, int]:
        w, h = self.pil.size
        return (h, w)

    @property
    def mode(self) -> str:
        return self.pil.mode

    def normalize_orientation(self, *, in_place: bool = False) -> "Image":
        """
        If an image has an EXIF Orientation tag, transpose accordingly and remove orientation data
        """
        if in_place:
            ImageOps.exif_transpose(self.pil, in_place=True)
            return self
        return Image(ImageOps.exif_transpose(self.pil), meta=dict(self.meta))

    def convert(self, mode: str | ImageMode) -> "Image":
        mode_s = mode.value if isinstance(mode, ImageMode) else mode
        return Image(self.pil.convert(mode_s), meta=dict(self.meta))

    def as_rgb(self) -> "Image":
        return self.convert(ImageMode.RGB)

    def enhance_contrast(self, factor: float = 1.2) -> "Image":
        return Image(ImageEnhance.Contrast(self.pil).enhance(factor), meta=dict(self.meta))

    def enhance_brightness(self, factor: float = 1.0) -> "Image":
        return Image(ImageEnhance.Brightness(self.pil).enhance(factor), meta=dict(self.meta))

    def resize_to_fit(self, max_size: int, *, resample: int = PILImage.Resampling.LANCZOS) -> "Image":
        """
        Keeps aspect ratio, does not upscale.
        """
        img = self.pil.copy()
        img.thumbnail((max_size, max_size), resample=resample)
        return Image(img, meta=dict(self.meta))

    def crop_xyxy(self, x0: int, y0: int, x1: int, y1: int) -> "Image":
        return Image(self.pil.crop((x0, y0, x1, y1)), meta=dict(self.meta))

    def to_numpy(self, *, dtype=None) -> "Any":
        """
        Research workflows often need NumPy; Pillow supports np.asarray(image)
        """
        arr = np.asarray(self.pil)
        return arr.astype(dtype) if dtype is not None else arr

    @staticmethod
    def from_numpy(arr, *, mode: str | ImageMode | None = None, meta: Optional[Dict[str, Any]] = None) -> "Image":
        
        a = np.asarray(arr)
        if mode is None:
            pil = PILImage.fromarray(a)
        else:
            mode_s = mode.value if isinstance(mode, ImageMode) else mode
            pil = PILImage.fromarray(a, mode=mode_s)
        return Image(pil, meta=meta or {})

    def save_jpeg(
        self,
        path: PathLike,
        *,
        quality: int = 90,
        optimize: bool = True,
        progressive: bool = True,
    ) -> None:
        """
        Pillow JPEG quality: 0..95 is recommended, values above 95 should be avoided :(
        """
        q = int(quality)
        if q < 0 or q > 95:
            raise ValueError("quality must be in [0,95] (Pillow recommends avoiding >95)")
        self.pil.save(path, format="JPEG", quality=q, optimize=optimize, progressive=progressive)

    def save_png(
        self,
        path: PathLike,
        *,
        optimize: bool = True,
        compress_level: int = 6,
    ) -> None:
        cl = int(compress_level)
        if cl < 0 or cl > 9:
            raise ValueError("compress_level must be in [0,9]")
        self.pil.save(path, format="PNG", optimize=optimize, compress_level=cl)

    def copy(self) -> "Image":
        return Image(self.pil.copy(), meta=dict(self.meta))

    def __repr__(self) -> str:
        w, h = self.pil.size
        return f"Image(mode={self.pil.mode!r}, size=({w},{h}))"

@dataclass
class PixelRegion:
    """Anchor a word to pixels in a manuscript image."""
    word_anchor_id: str
    folio_id: str
    image_path: str
    
    # Bounding box
    x: int
    y: int
    width: int
    height: int
    
    # Metadata
    confidence: float = 1.0  # OCR/ML confidence
    source: str = "manual"  # "manual", "ocr", "ml"
    created_at: str = ""
    
    def as_rect(self) -> Tuple[int, int, int, int]:
        """Return as (x, y, x+w, y+h) for image operations."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def extract_text_region(self) -> Optional[np.ndarray]:
        """Extract image region for OCR or analysis."""
        try:
            img = Image.open(self.image_path)
            return np.array(img.crop(self.as_rect()))
        except Exception:
            return None
