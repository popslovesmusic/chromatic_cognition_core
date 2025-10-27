# Project Spec — Chromatic Cognition Core (CPU-First)

## Objective
Build a deterministic Rust engine that represents cognition as an RGB tensor field.
Each cell = (r,g,b) triple + scalar certainty ρ.
The engine must run entirely on CPU but be easily portable to GPU later through Candle.

---

## Core Components

### 1. Chromatic Tensor
- Structure: 4-D array `[rows, cols, layers, 3]` (RGB)
- Type: `f32`
- Backed by `ndarray::Array4<f32>`
- Supports traits: `Add`, `Sub`, `Display`, `Serialize`.

### 2. Primitive Operations
Implement as pure functions returning new tensors:
- `mix(a,b)`: additive coherence → normalize(a + b)
- `filter(a,b)`: subtractive distinction → clamp(a - b, 0.0, 1.0)
- `complement(a)`: rotate hue 180° in (g,b) plane
- `saturate(a,alpha)`: multiply chroma by alpha
All ops must be parallelized with `rayon`.

### 3. Gradient Layer
- Computes visible hue per cell: weighted blend of sub-cells.
- Exposes `to_png(path)` using `plotters` for visualization.

### 4. Training & Loss
- Optional autograd hooks (compatible with Candle tensors later).
- CPU: simple MSE loss between current and target color fields.
- Logging: every iteration outputs JSON `{iter, loss, mean_rgb, variance}`.

### 5. Configuration
- TOML config file under `/config/engine.toml`:
  ```toml
  [engine]
  rows = 64
  cols = 64
  layers = 8
  seed = 42
  device = "cpu"
