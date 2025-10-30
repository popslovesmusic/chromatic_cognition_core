# Architecture Documentation

## Overview

Chromatic Cognition Core is a deterministic Rust engine that represents cognition as an RGB tensor field. The architecture is designed to be modular, testable, and GPU-portable.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                       │
│                  (examples/demo.rs, etc.)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Library Interface                         │
│                    (src/lib.rs)                            │
│                                                             │
│  Exports: ChromaticTensor, Operations, Config, Logging    │
└──┬──────────────┬──────────────┬──────────────┬───────────┘
   │              │              │              │
   │              │              │              │
┌──▼──────┐  ┌───▼───────┐  ┌──▼──────┐  ┌───▼────────┐
│ Tensor  │  │ Operations│  │ Config  │  │  Logging   │
│ Module  │  │  Module   │  │ Module  │  │   Module   │
└─────────┘  └───────────┘  └─────────┘  └────────────┘
```

## Chromatic Semantic Archive Layout

The Chromatic Semantic Archive (CSA) is partitioned into a deterministic **3 × 12 × 12 × 3 processing unit**:

- **3 spectral stacks** capture low-, mid-, and high-frequency energy in the unified modality space (UMS).
- **12 hue categories** per stack align with the mapper's discrete color wheel, guaranteeing deterministic routing for
  retrieval.
- **12 modulation slots** per category preserve recent temporal slices without overlap, enabling fast churn recovery.
- **3 channel features** (RGB) are retained for each slot so the CSA can reconstruct chromatic tensors directly from archive
  state.

This organization mirrors the structure produced by `ModalityMapper::new`, ensuring that UMS encoders, the HNSW index, and
category-aware retrieval (`retrieve_hybrid`) operate on the same fixed lattice.

### Index Management Strategy

CSA stability relies on a churn-aware rebuild policy. The dream pool tracks evictions since the last rebuild and invalidates the
active indexes when churn exceeds **10% of the current entry count**. Light churn (<10%) leaves the HNSW graph intact while
marking ghost nodes, whereas heavy churn triggers `rebuild_active()` to regenerate the ANN graph and drops the linear fallback
index. This mechanism bounds rebuild costs while guaranteeing that stale nodes never exceed the configured safety margin.

## Module Breakdown

### 1. Tensor Module (`src/tensor/`)

The core tensor abstraction and related utilities.

#### `chromatic_tensor.rs`
- **ChromaticTensor**: Main 4D RGB tensor structure
- **TensorStatistics**: Statistical summary of tensor state
- **Methods**:
  - Construction: `new()`, `from_seed()`, `from_arrays()`
  - Manipulation: `normalize()`, `clamp()`, `complement()`, `saturate()`
  - Analysis: `statistics()`, `shape()`
  - Operators: `Add`, `Sub`, `Display`

#### `operations.rs`
- Pure functional operations on tensors
- All operations are parallelized with `rayon`
- Each operation logs its result statistics
- **Functions**:
  - `mix(a, b)` - Additive coherence
  - `filter(a, b)` - Subtractive distinction
  - `complement(a)` - Hue rotation
  - `saturate(a, alpha)` - Chroma adjustment

#### `gradient.rs`
- **GradientLayer**: 2D projection of 3D tensor
- Certainty-weighted averaging across layers
- PNG export via `plotters` backend

### 2. Config Module (`src/config.rs`)

Engine configuration management.

- **EngineConfig**: TOML configuration struct
- **ConfigError**: Error types for config loading
- Defaults: 64×64×8 tensor, seed 42, CPU device

### 3. Logging Module (`src/logging.rs`)

JSON line-delimited logging infrastructure.

- **OperationLogEntry**: Per-operation statistics
- **TrainingLogEntry**: Per-iteration training metrics
- Outputs to:
  - `logs/operations.jsonl` - Operation history
  - `logs/run.jsonl` - Training history

### 4. Training Module (`src/training.rs`)

Loss computation and training metrics.

- **TrainingMetrics**: Loss and tensor statistics
- **Functions**:
  - `mse_loss(current, target)` - Mean squared error

## Data Flow

### Typical Operation Flow

```
1. Load Config
   config/engine.toml → EngineConfig

2. Initialize Tensors
   EngineConfig → ChromaticTensor::from_seed()

3. Apply Operations
   ChromaticTensor → operations → ChromaticTensor
   └─→ log_operation() → logs/operations.jsonl

4. Project to 2D
   ChromaticTensor → GradientLayer → PNG

5. Compute Loss (optional)
   ChromaticTensor + Target → TrainingMetrics
   └─→ log_training_iteration() → logs/run.jsonl
```

## Memory Layout

### ChromaticTensor

```
colors: Array4<f32>
  Dimensions: [rows, cols, layers, 3]
  Layout: Row-major, contiguous
  Memory: rows × cols × layers × 3 × 4 bytes

  Example (64×64×8):
    64 × 64 × 8 × 3 × 4 = 393,216 bytes (~384 KB)

certainty: Array3<f32>
  Dimensions: [rows, cols, layers]
  Layout: Row-major, contiguous
  Memory: rows × cols × layers × 4 bytes

  Example (64×64×8):
    64 × 64 × 8 × 4 = 131,072 bytes (~128 KB)

Total per tensor: ~512 KB (for 64×64×8)
```

## Parallelization Strategy

### Current (CPU)

All operations use `rayon` for data parallelism:

1. **Slice-based parallelism**: When contiguous memory is available
   ```rust
   colors.as_slice_mut()
       .par_iter_mut()
       .for_each(|value| /* operation */);
   ```

2. **Zip parallelism**: For multi-array operations
   ```rust
   Zip::from(&mut output)
       .and(&input_a)
       .and(&input_b)
       .par_for_each(|out, &a, &b| *out = a + b);
   ```

3. **Iterator parallelism**: For non-contiguous views
   ```rust
   tensor.indexed_iter_mut()
       .par_bridge()
       .for_each(|(idx, value)| /* operation */);
   ```

### Future (GPU)

Architecture is designed for easy porting to Candle:

- Replace `ndarray::Array4` with `candle::Tensor`
- Replace `rayon` operations with GPU kernels
- Keep same API surface

## Error Handling

### Strategy

- **Config errors**: Fallback to defaults
- **Dimension mismatches**: Panic with clear message
- **I/O errors**: Propagate via `Result<T, io::Error>`
- **Logging errors**: Print to stderr, continue execution

### Error Types

```rust
ConfigError
├─ Io(std::io::Error)
└─ Parse(String)
```

## Testing Strategy

### Unit Tests (`tests/operations.rs`)

- Test each operation with known inputs/outputs
- Verify mathematical properties
- Check edge cases (zero, one, clamping)

### Integration Tests

- End-to-end demo execution
- Config loading and fallback
- Log file generation

### Property Tests (Future)

- Commutativity of mix
- Idempotency of normalize
- Inverse operations

## Performance Considerations

### Current Optimizations

1. **Parallel operations**: All tensor ops use rayon
2. **Contiguous memory**: Array4/Array3 use contiguous layout
3. **In-place when safe**: Operations clone only when necessary
4. **SIMD**: Enabled via rustc flags (target-cpu=native)

### Profiling Points

- [ ] Operation timings per tensor size
- [ ] Memory allocation patterns
- [ ] Parallel scaling (1-16 cores)
- [ ] Cache efficiency

### Bottlenecks

Current bottlenecks (64×64×8):
1. `complement()` - ~15ms (nested loops)
2. `saturate()` - ~25ms (nested loops)
3. PNG export - ~10ms (plotters overhead)

Optimization opportunities:
- Replace nested loops with parallel iterators
- Use SIMD intrinsics for color operations
- Cache statistics computation

## Configuration Schema

### TOML Format

```toml
[engine]
rows = 64           # Tensor height (positive integer)
cols = 64           # Tensor width (positive integer)
layers = 8          # Tensor depth (positive integer)
seed = 42           # Random seed (u64)
device = "cpu"      # Target device (string)
```

### Defaults

All fields are optional and fall back to sensible defaults:

```rust
EngineConfig {
    rows: 64,
    cols: 64,
    layers: 8,
    seed: 42,
    device: "cpu".to_string(),
}
```

## Logging Schema

### Operations Log (`logs/operations.jsonl`)

```json
{
  "operation": "mix",
  "timestamp_ms": 1761535834354,
  "mean_rgb": [0.53, 0.53, 0.53],
  "variance": 0.065,
  "certainty_mean": 0.55
}
```

### Training Log (`logs/run.jsonl`)

```json
{
  "iteration": 0,
  "loss": 0.32,
  "mean_rgb": [0.054, 0.946, 0.946],
  "variance": 0.022,
  "timestamp_ms": 1761535816560
}
```

## Extension Points

### Adding New Operations

1. Implement in `src/tensor/operations.rs`
2. Add logging call
3. Export from `src/lib.rs`
4. Add unit test

### Adding New Loss Functions

1. Implement in `src/training.rs`
2. Return `TrainingMetrics`
3. Export from `src/lib.rs`

### Adding New Export Formats

1. Implement in `src/tensor/gradient.rs`
2. Add feature flag to `Cargo.toml` if needed
3. Document in README

## Dependencies Rationale

- **ndarray** (0.15): N-dimensional array library, industry standard
- **rayon** (1.8): Data parallelism, excellent CPU scaling
- **serde** (1.0): Serialization framework, ubiquitous
- **serde_json** (1.0): JSON logging format
- **toml** (0.8): Configuration file format
- **plotters** (0.3): Plotting and image export

## Future Roadmap

### Phase 1: Optimization (Current)
- [ ] Profile and optimize hot paths
- [ ] Add SIMD operations
- [ ] Reduce allocations

### Phase 2: Features
- [ ] Additional operations (rotate, scale, blend)
- [ ] Multiple loss functions
- [ ] Training loop with backpropagation
- [ ] Checkpoint save/load

### Phase 3: GPU Port
- [ ] Replace ndarray with Candle tensors
- [ ] Implement GPU kernels
- [ ] Benchmark CPU vs GPU
- [ ] Multi-GPU support

### Phase 4: Applications
- [ ] Image processing examples
- [ ] Neural network integration
- [ ] Real-time visualization
- [ ] Interactive demo
