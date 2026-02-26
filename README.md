# pyseed_dtypes

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Strict, immutable, and semantic data structures for scientific computing, digital humanities, and machine learning.**

`pyseed_dtypes` is a production-grade type system designed to bring **explicit semantics** to Python's data ecosystem. Unlike standard arrays which are agnostic to their content, `pyseed_dtypes` provides specialized containers that enforce logic, units, and structural integrity for domains ranging from Quantum Physics to Biblical Manuscript analysis.

---

## 📦 Features at a Glance

### 1. The Tensor System (`tensors`)

A unified framework wrapping NumPy with immutable, domain-aware logic.
* **Core Tensors:** `Tensor`, `SparseTensor` (COO), `NDSparseTensor`, `MaskedTensor`.
* **Symbolic & Autodiff:** `SymbolicTensor` (SymPy integration) and `GradTensor` (Reverse-mode autodiff).
* **Physics & Quantum:** `QuantityTensor` (Unit-aware), `FieldTensor` (Vector/Scalar fields), `QuantumTensor` (Bra-ket notation).
* **Geometry:** `PointCloud`, `MeshTensor`, `TransformTensor` (Homogeneous coords).
* **Audio & Signal:** `WaveformTensor`, `SpectrogramTensor` (STFT/Mel).
* **Graph:** `AdjacencyTensor`, `GraphTensor`, `EdgeListTensor`.

### 2. Digital Humanities & NLPU (`nlpu`)
A highly specialized engine for computational theology and ancient manuscript analysis.
* **Biblical Analysis:** `BibleReference`, `WordAnchor`, `TextualVariant`.
* **Syntax Trees:** Full recursive `SyntaxTree` with `Clause`, `Phrase`, and `Word` nodes.
* **Manuscript Mapping:** `PixelAnchor` linking text to specific regions on manuscript images.
* **Morphology:** Deep morphological tagging (`Tense`, `Voice`, `Mood`, `Person`) for Greek/Hebrew.

### 3. Imaging Architecture (`imaging`)
Deterministic wrappers around Pillow/PIL.
* **Explicit Orientation:** Auto-handling of EXIF orientation.
* **Strict Typing:** `ImageTensor` enforces ColorSpace (`RGB`, `HSV`, `LAB`) and Layout (`HWC` vs `CHW`).
* **Depth Maps:** Native support for `DepthMapTensor`.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/berithjeshurun13/pyseed_dtypes.git](https://github.com/berithjeshurun13/pyseed_dtypes.git)
cd pyseed_dtypes

# Install strict dependencies
pip install numpy pillow

# Install optional domain-specific dependencies
pip install scipy sympy soundfile librosa pint

```

---

## 🚀 Quick Start

### 1. Semantically Rich Tensors

Stop guessing dimensions. Use tensors that know what they are.

```python
import numpy as np
from pyseed_dtypes import (
    WaveformTensor, 
    AudioLayout, 
    QuantityTensor,
    FieldKind, 
    FieldTensor
)

# --- Audio Example ---
# Create a waveform with explicit sample rate and layout
raw_audio = np.random.uniform(-1, 1, size=(44100, 2))  # 1 second stereo
waveform = WaveformTensor.from_numpy(
    raw_audio, 
    sample_rate=44100, 
    layout=AudioLayout.TC # Time-Channel
)

print(waveform.channels) # 2
print(waveform.num_samples) # 44100

# --- Physics Example ---
# Define a Vector Field (e.g., wind velocity) with units
data = np.random.rand(100, 100, 3) # 100x100 grid, 3 vector components
field = FieldTensor(
    quantity=QuantityTensor.from_data(data, unit="m/s"),
    kind=FieldKind.VECTOR,
    components_axis=-1
)

# Calculate magnitude (automatically returns a Scalar Field)
speed = field.magnitude()
print(f"Max speed: {speed.numpy().max()} {speed.unit}")

```

### 2. Symbolic Computation

Seamlessly mix symbolic math with tensor structures.

```python
import sympy as sp
from pyseed_dtypes import SymbolicTensor

x, y = sp.symbols('x y')
exprs = [[x**2, x*y], [x+y, y**2]]

# Create a symbolic tensor
sym_tensor = SymbolicTensor.from_data(exprs, strict_shape=True)

# Compute gradients or simplify
grad = sym_tensor.diff(x) 

# JIT compile to a fast NumPy function
func = sym_tensor.lambdify([x, y])
result = func(2.0, 3.0) 
print(result)

```

### 3. Digital Humanities (Biblical NLP)

Analyze ancient texts with deep structural data types.

```python
from pyseed_dtypes.nlpu import (
    BibleReference, 
    WordAnchor, 
    SyntaxTree, 
    ClauseType
)

# 1. Define a robust reference
ref = BibleReference.from_string("John 1:1")

# 2. Create a deterministic anchor for a specific word occurrence
# (Useful for linking distinct manuscripts to a canonical text)
anchor = WordAnchor.create(
    manuscript_id="P66",
    book="John", 
    chapter=1, 
    verse=1, 
    word_index=0
)

# 3. Work with Syntax Trees
# (Assuming a tree is loaded)
# Find all subordinate clauses in a sentence
sub_clauses = syntax_tree.find_clauses_by_type(ClauseType.SUBORDINATE)

```

---

## 📂 Architecture Overview

```
pyseed_dtypes/
├── explicit.py          # Core type definitions (Float32, Int64, Tree, Node)
├── imaging.py           # Production-grade PIL wrappers
├── nlpu/                # Natural Language Processing Unit
│   ├── bible.py         # Biblical reference & manuscript systems
│   └── _nlpu.py         # Syntax trees, Morphology, Semantics
├── tensors/             # The Tensor System
│   ├── _tensors.py      # Base Tensor, Sparse, Symbolic, Grad, Quantum
│   ├── audio.py         # Waveform, Spectrogram
│   ├── geometry.py      # PointCloud, Mesh, BBox
│   ├── graph.py         # Adjacency, EdgeList
│   ├── physics.py       # Quantity (Units), Fields
│   └── probability.py   # Distributions, Logits
-
```

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

> **Author:** GliTCH
<br>
> **Contact:** beirthjeshurun13@gmail.com
