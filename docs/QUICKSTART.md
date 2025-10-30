# Quick Start Guide

Get up and running with Chromatic Cognition Core in 5 minutes.

## Installation

### Prerequisites

- Rust 1.70 or later with Cargo (bundled with Rustup)
- MUSL target support for Rust:
  ```bash
  rustup target add x86_64-unknown-linux-musl
  ```
- MUSL tooling installed on the host (e.g., `musl-tools` or `musl-gcc`) to enable static linking

### Clone and Build

```bash
git clone <repository-url>
cd chromatic_cognition_core
cargo build --release
```

## Your First Tensor

### Basic Usage

Create a simple Rust program:

```rust
use chromatic_cognition_core::ChromaticTensor;

fn main() {
    // Create a deterministic random tensor
    let tensor = ChromaticTensor::from_seed(42, 64, 64, 8);

    // Display statistics
    println!("{}", tensor);
    // Output: ChromaticTensor 64x64x8 mean_rgb=(0.531,0.531,0.530) variance=0.06486

    // Get detailed statistics
    let stats = tensor.statistics();
    println!("Mean RGB: {:?}", stats.mean_rgb);
    println!("Variance: {}", stats.variance);
    println!("Mean Certainty: {}", stats.mean_certainty);
}
```

### Running the Example

```bash
cargo run
```

## Applying Operations

### Color Operations

```rust
use chromatic_cognition_core::{ChromaticTensor, mix, filter, complement, saturate};

fn main() {
    // Create two tensors
    let a = ChromaticTensor::from_seed(42, 64, 64, 8);
    let b = ChromaticTensor::from_seed(100, 64, 64, 8);

    // Mix them together (additive)
    let mixed = mix(&a, &b);
    println!("Mixed: {}", mixed);

    // Apply subtractive filter
    let filtered = filter(&mixed, &b);
    println!("Filtered: {}", filtered);

    // Complement the colors
    let complemented = complement(&filtered);
    println!("Complemented: {}", complemented);

    // Increase saturation by 50%
    let saturated = saturate(&complemented, 1.5);
    println!("Saturated: {}", saturated);
}
```

All operations automatically log to `logs/operations.jsonl`.

## Visualization

### Export to PNG

```rust
use chromatic_cognition_core::{ChromaticTensor, GradientLayer};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a tensor
    let tensor = ChromaticTensor::from_seed(42, 256, 256, 8);

    // Project to 2D and export
    let gradient = GradientLayer::from_tensor(&tensor);
    gradient.to_png(PathBuf::from("output/my_visualization.png"))?;

    println!("Saved visualization to output/my_visualization.png");
    Ok(())
}
```

The PNG will show a certainty-weighted projection of all layers.

## Configuration

### Using Config Files

Create `config/my_config.toml`:

```toml
[engine]
rows = 128
cols = 128
layers = 16
seed = 12345
device = "cpu"
```

Load it in your code:

```rust
use chromatic_cognition_core::{EngineConfig, ChromaticTensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load config
    let config = EngineConfig::load_from_file("config/my_config.toml")?;

    // Create tensor with config dimensions
    let tensor = ChromaticTensor::from_seed(
        config.seed,
        config.rows,
        config.cols,
        config.layers
    );

    println!("Created {}x{}x{} tensor", config.rows, config.cols, config.layers);
    Ok(())
}
```

### Defaults and Fallback

```rust
use chromatic_cognition_core::EngineConfig;

fn main() {
    // Try to load, fallback to defaults on error
    let config = EngineConfig::load_from_file("config/engine.toml")
        .unwrap_or_else(|err| {
            eprintln!("Using default config: {}", err);
            EngineConfig::default()
        });

    println!("Config: rows={} cols={} layers={}",
        config.rows, config.cols, config.layers);
}
```

## Training Example

### Compute Loss

```rust
use chromatic_cognition_core::{ChromaticTensor, mse_loss, logging};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create current and target tensors
    let current = ChromaticTensor::from_seed(42, 64, 64, 8);
    let target = ChromaticTensor::from_seed(100, 64, 64, 8);

    // Compute MSE loss
    let metrics = mse_loss(&current, &target);

    println!("Loss: {}", metrics.loss);
    println!("Mean RGB: {:?}", metrics.mean_rgb);
    println!("Variance: {}", metrics.variance);

    // Log the iteration
    logging::log_training_iteration(0, &metrics)?;

    println!("Logged to logs/run.jsonl");
    Ok(())
}
```

## Running the Demo

The included demo showcases the full pipeline:

```bash
cargo run --example demo
```

This will:
1. Load `config/engine.toml` (or use defaults)
2. Create two random tensors
3. Apply all operations in sequence
4. Generate `out/frame_0001.png`
5. Log operations to `logs/operations.jsonl`
6. Log training metrics to `logs/run.jsonl`

## Common Patterns

### Chaining Operations

```rust
use chromatic_cognition_core::{ChromaticTensor, mix, filter, complement, saturate};

fn process_tensor(input: &ChromaticTensor) -> ChromaticTensor {
    let noise = ChromaticTensor::from_seed(999, input.shape().0, input.shape().1, input.shape().2);

    mix(input, &noise)               // Add noise
        |> |t| filter(t, &noise)      // Remove some noise
        |> |t| complement(t)          // Invert colors
        |> |t| saturate(t, 1.3)       // Boost saturation
        |> |t| t.normalize()          // Ensure valid range
}
```

Or more idiomatically:

```rust
fn process_tensor(input: &ChromaticTensor) -> ChromaticTensor {
    let noise = ChromaticTensor::from_seed(999, input.shape().0, input.shape().1, input.shape().2);

    let step1 = mix(input, &noise);
    let step2 = filter(&step1, &noise);
    let step3 = complement(&step2);
    let step4 = saturate(&step3, 1.3);
    step4.normalize()
}
```

### Batch Processing

```rust
use chromatic_cognition_core::{ChromaticTensor, complement, GradientLayer};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let seeds = vec![42, 100, 200, 300, 400];

    for (i, seed) in seeds.iter().enumerate() {
        // Create tensor
        let tensor = ChromaticTensor::from_seed(*seed, 128, 128, 8);

        // Process
        let processed = complement(&tensor);

        // Export
        let gradient = GradientLayer::from_tensor(&processed);
        let path = PathBuf::from(format!("output/batch_{:03}.png", i));
        gradient.to_png(path)?;

        println!("Processed frame {}", i);
    }

    Ok(())
}
```

### Custom Statistics

```rust
use chromatic_cognition_core::ChromaticTensor;

fn analyze_tensor(tensor: &ChromaticTensor) {
    let stats = tensor.statistics();

    // Compute additional metrics
    let brightness = (stats.mean_rgb[0] + stats.mean_rgb[1] + stats.mean_rgb[2]) / 3.0;
    let color_balance = stats.mean_rgb[0] / (stats.mean_rgb[1] + 0.001); // Avoid divide by zero

    println!("Brightness: {:.3}", brightness);
    println!("Color Balance (R/G): {:.3}", color_balance);
    println!("Variance: {:.5}", stats.variance);
    println!("Certainty: {:.3}", stats.mean_certainty);
}
```

## Performance Tips

### 1. Avoid Unnecessary Clones

```rust
// Bad - clones the entire tensor
let tensor = ChromaticTensor::from_seed(42, 256, 256, 32);
let copy = tensor.clone(); // 32 MB copied!

// Good - pass by reference
fn process(tensor: &ChromaticTensor) -> ChromaticTensor {
    tensor.normalize() // Only clones when needed
}
```

### 2. Use Appropriate Sizes

```rust
// For testing/prototyping - small and fast
let small = ChromaticTensor::from_seed(42, 32, 32, 4);  // ~12 KB

// For production - balanced
let medium = ChromaticTensor::from_seed(42, 128, 128, 8);  // ~1.5 MB

// For high quality - slower
let large = ChromaticTensor::from_seed(42, 512, 512, 16);  // ~48 MB
```

### 3. Parallel Processing is Automatic

All operations use rayon for parallelism. To control thread count:

```bash
RAYON_NUM_THREADS=4 cargo run --release
```

## Debugging

### Check Logs

```bash
# View operation history
cat logs/operations.jsonl | jq .

# View training history
cat logs/run.jsonl | jq .

# Watch logs in real-time
tail -f logs/operations.jsonl
```

### Verbose Output

```rust
use chromatic_cognition_core::{ChromaticTensor, mix};

fn main() {
    let a = ChromaticTensor::from_seed(42, 64, 64, 8);
    let b = ChromaticTensor::from_seed(100, 64, 64, 8);

    println!("Tensor A: {}", a);
    println!("Tensor B: {}", b);

    let result = mix(&a, &b);

    println!("Mixed: {}", result);

    // Detailed breakdown
    let stats = result.statistics();
    println!("R: {:.3}, G: {:.3}, B: {:.3}",
        stats.mean_rgb[0], stats.mean_rgb[1], stats.mean_rgb[2]);
}
```

## Testing

### Run All Tests

```bash
cargo test
```

### Run Specific Test

```bash
cargo test mix_adds_and_normalizes
```

### Run with Output

```bash
cargo test -- --nocapture
```

## Next Steps

- Read the [Architecture Documentation](./ARCHITECTURE.md) to understand the system design
- Check the [API Reference](./API.md) for detailed function documentation
- Explore `examples/demo.rs` for a complete working example
- View generated rustdoc: `cargo doc --open`

## Common Issues

### Q: Tests fail with "contiguous" panic

Make sure you're on the latest code with the statistics fix:

```rust
// Fixed version uses as_slice() with Option fallback
let sum = if let Some(slice) = channel_view.as_slice() {
    slice.par_iter().sum()
} else {
    channel_view.iter().sum()
};
```

### Q: PNG export fails

Ensure output directory exists or the code creates it:

```rust
if let Some(parent) = path.parent() {
    std::fs::create_dir_all(parent)?;
}
```

### Q: Config file not found

Use fallback pattern:

```rust
let config = EngineConfig::load_from_file("config/engine.toml")
    .unwrap_or_else(|_| EngineConfig::default());
```

## Support

- Documentation: `cargo doc --open`
- Examples: `examples/demo.rs`
- Tests: `tests/operations.rs`
- Issues: [GitHub Issues](link-to-issues)
