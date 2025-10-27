# API Reference

This document provides a comprehensive API reference for the Chromatic Cognition Core library.

## Table of Contents

- [Core Types](#core-types)
- [Operations](#operations)
- [Configuration](#configuration)
- [Logging](#logging)
- [Training](#training)

---

## Core Types

### ChromaticTensor

```rust
pub struct ChromaticTensor {
    pub colors: Array4<f32>,      // [rows, cols, layers, 3]
    pub certainty: Array3<f32>,   // [rows, cols, layers]
}
```

A 4-dimensional chromatic tensor representing an RGB color field with certainty weights.

#### Constructors

##### `new(rows, cols, layers) -> Self`

Creates a zero-initialized tensor.

```rust
let tensor = ChromaticTensor::new(64, 64, 8);
```

##### `from_seed(seed, rows, cols, layers) -> Self`

Creates a deterministic random tensor using LCG.

```rust
let tensor = ChromaticTensor::from_seed(42, 64, 64, 8);
```

**Parameters:**
- `seed: u64` - Random seed (0 uses default seed of 1)
- `rows: usize` - Number of rows
- `cols: usize` - Number of columns
- `layers: usize` - Number of depth layers

**Returns:** Initialized ChromaticTensor

##### `from_arrays(colors, certainty) -> Self`

Creates a tensor from existing arrays.

```rust
let colors = Array4::zeros((64, 64, 8, 3));
let certainty = Array3::ones((64, 64, 8));
let tensor = ChromaticTensor::from_arrays(colors, certainty);
```

**Panics:** If dimensions don't match

#### Methods

##### `shape(&self) -> (usize, usize, usize, usize)`

Returns tensor dimensions as (rows, cols, layers, channels).

```rust
let (rows, cols, layers, channels) = tensor.shape();
assert_eq!(channels, 3); // Always 3 for RGB
```

##### `normalize(&self) -> Self`

Normalizes color values to [0.0, 1.0] range.

```rust
let normalized = tensor.normalize();
```

**Returns:** New normalized tensor

##### `clamp(&self, min, max) -> Self`

Clamps all color values to specified range.

```rust
let clamped = tensor.clamp(0.0, 1.0);
```

**Parameters:**
- `min: f32` - Minimum value
- `max: f32` - Maximum value

**Returns:** New clamped tensor

##### `complement(&self) -> Self`

Inverts green and blue channels (180° hue rotation).

```rust
let complemented = tensor.complement();
```

**Returns:** New complemented tensor

##### `saturate(&self, alpha) -> Self`

Adjusts saturation by scaling deviation from mean.

```rust
let saturated = tensor.saturate(1.5);   // Increase saturation
let desaturated = tensor.saturate(0.5); // Decrease saturation
```

**Parameters:**
- `alpha: f32` - Saturation multiplier

**Returns:** New saturated tensor

##### `statistics(&self) -> TensorStatistics`

Computes statistical summary of tensor.

```rust
let stats = tensor.statistics();
println!("Mean RGB: {:?}", stats.mean_rgb);
println!("Variance: {}", stats.variance);
println!("Mean certainty: {}", stats.mean_certainty);
```

**Returns:** TensorStatistics struct

#### Operators

##### `Add`

Element-wise addition with certainty averaging.

```rust
let result = tensor_a + tensor_b;
```

**Panics:** If dimensions don't match

##### `Sub`

Element-wise subtraction with certainty averaging.

```rust
let result = tensor_a - tensor_b;
```

**Panics:** If dimensions don't match

##### `Display`

Formatted output with statistics.

```rust
println!("{}", tensor);
// Output: ChromaticTensor 64x64x8 mean_rgb=(0.531,0.531,0.530) variance=0.06486
```

---

### TensorStatistics

```rust
pub struct TensorStatistics {
    pub mean_rgb: [f32; 3],
    pub variance: f32,
    pub mean_certainty: f32,
}
```

Statistical summary of a chromatic tensor.

**Fields:**
- `mean_rgb` - Average RGB values across all cells
- `variance` - Color variance across all channels
- `mean_certainty` - Average certainty weight

---

### GradientLayer

```rust
pub struct GradientLayer {
    pub image: Array3<f32>,  // [rows, cols, 3]
}
```

2D projection of a 3D chromatic tensor using certainty-weighted averaging.

#### Constructors

##### `from_tensor(tensor) -> Self`

Projects a ChromaticTensor to 2D using weighted averaging.

```rust
let gradient = GradientLayer::from_tensor(&tensor);
```

**Parameters:**
- `tensor: &ChromaticTensor` - Source tensor to project

**Returns:** GradientLayer with 2D image

#### Methods

##### `to_png<P: AsRef<Path>>(&self, path) -> io::Result<()>`

Exports the gradient layer as a PNG image.

```rust
gradient.to_png("output/frame_0001.png")?;
```

**Parameters:**
- `path` - Output file path

**Returns:** Result with I/O error on failure

---

## Operations

All operations are pure functions in the `tensor::operations` module.

### mix

```rust
pub fn mix(a: &ChromaticTensor, b: &ChromaticTensor) -> ChromaticTensor
```

Combines two tensors through additive coherence.

**Algorithm:**
1. Element-wise addition: `out = a + b`
2. Normalize to [0.0, 1.0]
3. Average certainty: `cert = (a.cert + b.cert) / 2`

**Example:**

```rust
use chromatic_cognition_core::{ChromaticTensor, mix};

let a = ChromaticTensor::from_seed(42, 32, 32, 4);
let b = ChromaticTensor::from_seed(100, 32, 32, 4);
let result = mix(&a, &b);
```

**Panics:** If tensor dimensions don't match

**Logs:** Operation statistics to `logs/operations.jsonl`

---

### filter

```rust
pub fn filter(a: &ChromaticTensor, b: &ChromaticTensor) -> ChromaticTensor
```

Applies subtractive filtering between two tensors.

**Algorithm:**
1. Element-wise subtraction: `out = a - b`
2. Clamp to [0.0, 1.0]
3. Average certainty: `cert = (a.cert + b.cert) / 2`

**Example:**

```rust
use chromatic_cognition_core::{ChromaticTensor, filter};

let a = ChromaticTensor::from_seed(42, 32, 32, 4);
let b = ChromaticTensor::from_seed(100, 32, 32, 4);
let result = filter(&a, &b);
```

**Panics:** If tensor dimensions don't match

**Logs:** Operation statistics to `logs/operations.jsonl`

---

### complement

```rust
pub fn complement(a: &ChromaticTensor) -> ChromaticTensor
```

Computes chromatic complement by rotating hue 180°.

**Algorithm:**
- Red channel: unchanged
- Green channel: `g' = 1.0 - g`
- Blue channel: `b' = 1.0 - b`

**Example:**

```rust
use chromatic_cognition_core::{ChromaticTensor, complement};

let tensor = ChromaticTensor::from_seed(42, 32, 32, 4);
let result = complement(&tensor);
```

**Logs:** Operation statistics to `logs/operations.jsonl`

---

### saturate

```rust
pub fn saturate(a: &ChromaticTensor, alpha: f32) -> ChromaticTensor
```

Adjusts color saturation by scaling distance from mean.

**Algorithm:**
1. Compute mean: `mean = (r + g + b) / 3`
2. Scale deviation: `r' = mean + (r - mean) * alpha`
3. Clamp to [0.0, 1.0]

**Example:**

```rust
use chromatic_cognition_core::{ChromaticTensor, saturate};

let tensor = ChromaticTensor::from_seed(42, 32, 32, 4);
let more_saturated = saturate(&tensor, 1.5);
let desaturated = saturate(&tensor, 0.5);
```

**Parameters:**
- `a` - Input tensor
- `alpha` - Saturation multiplier (>1 increases, <1 decreases)

**Logs:** Operation statistics to `logs/operations.jsonl`

---

## Configuration

### EngineConfig

```rust
pub struct EngineConfig {
    pub rows: usize,
    pub cols: usize,
    pub layers: usize,
    pub seed: u64,
    pub device: String,
}
```

Engine configuration loaded from TOML files.

#### Constructors

##### `load_from_file<P: AsRef<Path>>(path) -> Result<Self, ConfigError>`

Loads configuration from a TOML file.

```rust
use chromatic_cognition_core::EngineConfig;

let config = EngineConfig::load_from_file("config/engine.toml")?;
```

**Returns:**
- `Ok(EngineConfig)` on success
- `Err(ConfigError)` on failure

##### `from_str(toml_str) -> Result<Self, ConfigError>`

Parses configuration from a TOML string.

```rust
let toml = r#"
[engine]
rows = 128
cols = 128
layers = 16
seed = 42
device = "cpu"
"#;

let config = EngineConfig::from_str(toml)?;
```

##### `default() -> Self`

Returns default configuration.

```rust
let config = EngineConfig::default();
// rows=64, cols=64, layers=8, seed=42, device="cpu"
```

#### TOML Format

```toml
[engine]
rows = 64          # Grid height
cols = 64          # Grid width
layers = 8         # Depth layers
seed = 42          # Random seed
device = "cpu"     # Target device
```

---

### ConfigError

```rust
pub enum ConfigError {
    Io(std::io::Error),
    Parse(String),
}
```

Configuration loading errors.

**Variants:**
- `Io` - File I/O error
- `Parse` - TOML parsing error

---

## Logging

### log_operation

```rust
pub fn log_operation(operation: &str, stats: &TensorStatistics) -> io::Result<()>
```

Logs an operation to `logs/operations.jsonl`.

**Not typically called directly** - operations call this automatically.

**Log Format:**

```json
{
  "operation": "mix",
  "timestamp_ms": 1761535834354,
  "mean_rgb": [0.53, 0.53, 0.53],
  "variance": 0.065,
  "certainty_mean": 0.55
}
```

---

### log_training_iteration

```rust
pub fn log_training_iteration(iteration: usize, metrics: &TrainingMetrics) -> io::Result<()>
```

Logs a training iteration to `logs/run.jsonl`.

**Example:**

```rust
use chromatic_cognition_core::{mse_loss, logging};

let current = ChromaticTensor::from_seed(42, 64, 64, 8);
let target = ChromaticTensor::from_seed(100, 64, 64, 8);

let metrics = mse_loss(&current, &target);
logging::log_training_iteration(0, &metrics)?;
```

**Log Format:**

```json
{
  "iteration": 0,
  "loss": 0.32,
  "mean_rgb": [0.054, 0.946, 0.946],
  "variance": 0.022,
  "timestamp_ms": 1761535816560
}
```

---

## Training

### TrainingMetrics

```rust
pub struct TrainingMetrics {
    pub loss: f32,
    pub mean_rgb: [f32; 3],
    pub variance: f32,
}
```

Training iteration metrics.

**Fields:**
- `loss` - Computed loss value
- `mean_rgb` - Average RGB of current tensor
- `variance` - Color variance of current tensor

---

### mse_loss

```rust
pub fn mse_loss(current: &ChromaticTensor, target: &ChromaticTensor) -> TrainingMetrics
```

Computes mean squared error between two tensors.

**Algorithm:**

```
MSE = Σ(current - target)² / N
```

**Example:**

```rust
use chromatic_cognition_core::{ChromaticTensor, mse_loss};

let current = ChromaticTensor::from_seed(42, 64, 64, 8);
let target = ChromaticTensor::from_seed(100, 64, 64, 8);

let metrics = mse_loss(&current, &target);
println!("Loss: {}", metrics.loss);
```

**Panics:** If tensor dimensions don't match

**Returns:** TrainingMetrics with loss and current tensor statistics

---

## Type Aliases

```rust
use ndarray::{Array3, Array4};

// Commonly used types
type ColorTensor = Array4<f32>;   // [rows, cols, layers, 3]
type CertaintyTensor = Array3<f32>; // [rows, cols, layers]
type ImageArray = Array3<f32>;    // [rows, cols, 3]
```

---

## Error Handling

Most functions use Rust's standard error handling:

- **Panics:** For programming errors (mismatched dimensions)
- **Result:** For recoverable errors (I/O, parsing)
- **Option:** Not currently used

### Common Panic Scenarios

```rust
// Dimension mismatch
let a = ChromaticTensor::new(64, 64, 8);
let b = ChromaticTensor::new(32, 32, 4);
let result = mix(&a, &b); // PANIC: dimensions don't match

// Invalid arrays
let colors = Array4::zeros((64, 64, 8, 3));
let certainty = Array3::zeros((32, 32, 4));
let tensor = ChromaticTensor::from_arrays(colors, certainty); // PANIC
```

### Error Recovery

```rust
// Config loading with fallback
let config = EngineConfig::load_from_file("config/engine.toml")
    .unwrap_or_else(|err| {
        eprintln!("Config error: {}", err);
        EngineConfig::default()
    });

// Logging errors (non-fatal)
if let Err(e) = logging::log_operation("custom", &stats) {
    eprintln!("Failed to log: {}", e);
    // Continue execution
}
```

---

## Performance Notes

### Parallelization

All operations use `rayon` for automatic parallelization:

- Optimal for CPU-bound workloads
- Scales with available cores
- No manual thread management needed

### Memory

Tensors are heap-allocated with contiguous layout:

```rust
// Memory usage for 64×64×8 tensor
// colors:    64 × 64 × 8 × 3 × 4 = 393,216 bytes (~384 KB)
// certainty: 64 × 64 × 8 × 4     = 131,072 bytes (~128 KB)
// Total:                          ~512 KB per tensor
```

### Cloning

Most operations clone the tensor. For large tensors, consider:

```rust
// Avoid unnecessary clones
let a = ChromaticTensor::from_seed(42, 256, 256, 32); // ~50 MB
let b = a.clone(); // Another 50 MB allocated

// Better: pass by reference
fn process(tensor: &ChromaticTensor) -> ChromaticTensor {
    // Only clone when necessary
    tensor.normalize()
}
```
