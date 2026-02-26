# pyseed_dtypes/audio.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..explict import DType, DTypeLike, get_dtype
from ._tensors import Tensor

__all__ = [
    "AudioLayout",
    "WaveformTensor",
    "SpectrogramTensor",
    "AudioTensor",
]


class AudioLayout(Enum):
    T = auto()     # (time,)
    TC = auto()    # (time, channels)
    CT = auto()    # (channels, time)


def _require_lib(name: str):
    raise ImportError(f"{name} is required for this operation. Install it with: pip install {name}")


def _as_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _ensure_layout(arr: np.ndarray, layout: AudioLayout) -> np.ndarray:
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError("Waveform must be 1D (T) or 2D (TC/CT)")
    if layout == AudioLayout.TC:
        return arr
    if layout == AudioLayout.CT:
        return np.transpose(arr, (1, 0))
    return arr  # AudioLayout.T but 2D doesn't make sense; validated elsewhere


@dataclass(frozen=True, slots=True)
class WaveformTensor:
    """
    Time-domain waveform.

    data shape:
      - (T,) mono
      - (T, C) if layout=TC
      - (C, T) if layout=CT

    sample_rate is required for most DSP operations.
    """
    tensor: Tensor
    sample_rate: int
    layout: AudioLayout = AudioLayout.T
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        x = self.tensor.numpy()
        if x.ndim == 1:
            if self.layout != AudioLayout.T:
                raise ValueError("1D waveform must use layout=T")
        elif x.ndim == 2:
            if self.layout not in (AudioLayout.TC, AudioLayout.CT):
                raise ValueError("2D waveform must use layout=TC or CT")
        else:
            raise ValueError("WaveformTensor expects ndim 1 or 2")

        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive int")

        # Strongly recommend float32 in [-1,1], but don't force it.

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.tensor.shape

    @property
    def channels(self) -> int:
        x = self.tensor.numpy()
        if x.ndim == 1:
            return 1
        return int(x.shape[1] if self.layout == AudioLayout.TC else x.shape[0])

    @property
    def num_samples(self) -> int:
        x = self.tensor.numpy()
        if x.ndim == 1:
            return int(x.shape[0])
        return int(x.shape[0] if self.layout == AudioLayout.TC else x.shape[1])

    def numpy(self, *, layout: AudioLayout | None = None) -> np.ndarray:
        x = self.tensor.numpy()
        if layout is None or layout == self.layout:
            return x
        if x.ndim == 1:
            return x
        if self.layout == AudioLayout.TC and layout == AudioLayout.CT:
            return np.transpose(x, (1, 0))
        if self.layout == AudioLayout.CT and layout == AudioLayout.TC:
            return np.transpose(x, (1, 0))
        raise ValueError("Invalid layout conversion")

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        *,
        sample_rate: int,
        layout: AudioLayout = AudioLayout.T,
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "WaveformTensor":
        dt = get_dtype(dtype) if dtype is not None else None
        arr = np.asarray(array, dtype=dt.np_dtype if dt else None)
        t = Tensor.from_data(arr, dtype=dt, meta=meta)
        return WaveformTensor(tensor=t, sample_rate=sample_rate, layout=layout, meta=meta or {})

    @staticmethod
    def from_file(
        path: str,
        *,
        always_2d: bool = False,
        dtype: str = "float32",
        meta: Dict[str, Any] | None = None,
    ) -> "WaveformTensor":
        """
        Load audio via soundfile: returns data + samplerate.
        """
        try:
            import soundfile as sf  # type: ignore
        except Exception:
            _require_lib("soundfile")

        data, sr = sf.read(path, always_2d=always_2d, dtype=dtype)
        # soundfile returns shape (frames, channels) when always_2d=True or file is multi-channel
        layout = AudioLayout.TC if data.ndim == 2 else AudioLayout.T
        return WaveformTensor.from_numpy(data, sample_rate=int(sr), layout=layout, dtype=get_dtype(np.dtype(data.dtype)), meta=meta)

    def to_file(self, path: str) -> None:
        try:
            import soundfile as sf  # type: ignore
        except Exception:
            _require_lib("soundfile")

        data = self.numpy(layout=AudioLayout.TC) if self.tensor.ndim == 2 and self.layout == AudioLayout.CT else self.tensor.numpy()
        sf.write(path, data, self.sample_rate)  # common soundfile write API

    def stft_spectrogram(
        self,
        *,
        nperseg: int = 1024,
        noverlap: Optional[int] = None,
        mode: str = "magnitude",
    ) -> "SpectrogramTensor":
        """
        Compute a spectrogram using scipy.signal.spectrogram.
        SciPy supports modes like 'psd', 'complex', 'magnitude', etc.
        """
        try:
            from scipy import signal  # type: ignore
        except Exception:
            _require_lib("scipy")

        x = self.numpy(layout=AudioLayout.T)
        if x.ndim != 1:
            raise ValueError("stft_spectrogram currently supports mono only (layout=T)")

        f, t, Sxx = signal.spectrogram(
            x,
            fs=self.sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            mode=mode,
            axis=-1,  # default is last axis
        )
        # Sxx shape: (freq_bins, time_frames)
        return SpectrogramTensor.from_numpy(
            Sxx,
            sample_rate=self.sample_rate,
            freqs=f,
            times=t,
            kind=f"scipy:{mode}",
            meta=dict(self.meta),
        )

    def mel_spectrogram(
        self,
        *,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        power: float = 2.0,
    ) -> "SpectrogramTensor":
        """
        Mel spectrogram via librosa.feature.melspectrogram.
        """
        try:
            import librosa  # type: ignore
        except Exception:
            _require_lib("librosa")

        x = self.numpy(layout=AudioLayout.T)
        if x.ndim != 1:
            raise ValueError("mel_spectrogram currently supports mono only (layout=T)")

        S = librosa.feature.melspectrogram(
            y=x,
            sr=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
        )  # returns mel spectrogram
        return SpectrogramTensor.from_numpy(
            S,
            sample_rate=self.sample_rate,
            freqs=None,
            times=None,
            kind="mel",
            meta=dict(self.meta),
        )

    def __repr__(self) -> str:
        return f"WaveformTensor(samples={self.num_samples}, sr={self.sample_rate}, channels={self.channels}, layout={self.layout.name})"

@dataclass(frozen=True, slots=True)
class SpectrogramTensor:
    """
    Time-frequency representation.

    Typical shape:
      - (F, T) = (frequency bins, time frames)
      - for mel-spectrogram, F = n_mels.
    """
    tensor: Tensor
    sample_rate: int
    freqs: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    kind: str = "stft"
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._validate()

    def _validate(self):
        S = self.tensor.numpy()
        if S.ndim != 2:
            raise ValueError("SpectrogramTensor expects 2D array (F, T)")
        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive int")

    @property
    def shape(self) -> Tuple[int, int]:
        f, t = self.tensor.shape
        return int(f), int(t)

    def numpy(self) -> np.ndarray:
        return self.tensor.numpy()

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        *,
        sample_rate: int,
        freqs: Optional[np.ndarray],
        times: Optional[np.ndarray],
        kind: str = "stft",
        dtype: DTypeLike | None = None,
        meta: Dict[str, Any] | None = None,
    ) -> "SpectrogramTensor":
        dt = get_dtype(dtype) if dtype is not None else None
        arr = np.asarray(array, dtype=dt.np_dtype if dt else None)
        t = Tensor.from_data(arr, dtype=dt, meta=meta)
        return SpectrogramTensor(tensor=t, sample_rate=sample_rate, freqs=freqs, times=times, kind=kind, meta=meta or {})

    def __repr__(self) -> str:
        f, t = self.shape
        return f"SpectrogramTensor(F={f}, T={t}, kind={self.kind!r}, sr={self.sample_rate})"

@dataclass(frozen=True, slots=True)
class AudioTensor:
    """
    A general container for audio representations.
    Exactly one of waveform/spectrogram should be set.
    """
    waveform: WaveformTensor | None = None
    spectrogram: SpectrogramTensor | None = None
    meta: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if (self.waveform is None) == (self.spectrogram is None):
            raise ValueError("AudioTensor requires exactly one of waveform or spectrogram")

    @property
    def sample_rate(self) -> int:
        return self.waveform.sample_rate if self.waveform is not None else self.spectrogram.sample_rate

    def __repr__(self) -> str:
        if self.waveform is not None:
            return f"AudioTensor({self.waveform!r})"
        return f"AudioTensor({self.spectrogram!r})"
