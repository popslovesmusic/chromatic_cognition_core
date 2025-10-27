# Chromatic Neural Network Design

## Architecture Overview

### Novel Concept
A neural network where **color tensors are the fundamental computational unit**, replacing traditional scalar activations with RGB triplets + certainty weights.

## Network Structure

### Layer Types

#### 1. ChromaticLayer
```rust
pub struct ChromaticLayer {
    weights: ChromaticTensor,      // Learnable color filters
    bias: ChromaticTensor,          // Learnable color bias
    operation: ChromaticOp,         // Operation type
}
```

**Forward pass:**
```
output = operation(mix(input, weights), bias)
```

#### 2. Supported Operations
- **MixLayer**: Additive combination with learned weights
- **FilterLayer**: Subtractive filtering with learned masks
- **ComplementLayer**: Learned hue rotation (parameter: angle)
- **SaturateLayer**: Learned saturation adjustment (parameter: alpha)

### Network Architecture

```
Input (32×32×4 chromatic tensor)
    ↓
ChromaticLayer (mix + saturate)     [32×32×4 → 32×32×8]
    ↓
ChromaticLayer (filter + complement) [32×32×8 → 16×16×16]
    ↓
ChromaticLayer (mix + saturate)     [16×16×16 → 8×8×32]
    ↓
GradientProjection                  [8×8×32 → 8×8×3]
    ↓
GlobalPooling                       [8×8×3 → 1×1×3]
    ↓
ColorClassifier                     [RGB → N classes]
```

## Task: Color Pattern Classification

### Dataset Design

#### Synthetic Color Patterns
1. **Primary Colors** (3 classes)
   - Predominantly red, green, or blue patterns

2. **Color Harmonies** (6 classes)
   - Complementary (opposite hues)
   - Analogous (adjacent hues)
   - Triadic (120° apart)
   - Split-complementary
   - Tetradic
   - Monochromatic

3. **Color Temperature** (2 classes)
   - Warm (reds, oranges, yellows)
   - Cool (blues, greens, purples)

4. **Saturation Levels** (3 classes)
   - Vivid (high saturation)
   - Muted (medium saturation)
   - Grayscale (zero saturation)

**Start with:** Primary colors (simple 3-class problem)

### Data Generation

```rust
pub struct ColorPattern {
    pub tensor: ChromaticTensor,
    pub label: usize,
    pub description: String,
}

pub fn generate_primary_color_dataset(
    samples_per_class: usize,
    rows: usize,
    cols: usize,
    layers: usize,
) -> Vec<ColorPattern>
```

**Generation strategy:**
- Base color from class (red/green/blue)
- Add noise and variations
- Apply random transformations
- Ensure class separability

## Backpropagation for Chromatic Operations

### Gradients for Each Operation

#### Mix Operation
```
forward:  out = normalize(a + b)
backward: ∂L/∂a = ∂L/∂out * (1 / max_val)
          ∂L/∂b = ∂L/∂out * (1 / max_val)
```

#### Filter Operation
```
forward:  out = clamp(a - b, 0, 1)
backward: ∂L/∂a = ∂L/∂out * (mask where a > b)
          ∂L/∂b = -∂L/∂out * (mask where a > b)
```

#### Complement Operation
```
forward:  out[g] = 1 - in[g], out[b] = 1 - in[b]
backward: ∂L/∂in[r] = ∂L/∂out[r]
          ∂L/∂in[g] = -∂L/∂out[g]
          ∂L/∂in[b] = -∂L/∂out[b]
```

#### Saturate Operation
```
forward:  out = mean + (in - mean) * alpha
backward: ∂L/∂in = ∂L/∂out * alpha
          ∂L/∂alpha = ∂L/∂out * (in - mean)
```

### Gradient Tensor

```rust
pub struct GradientTensor {
    pub color_grad: Array4<f32>,     // Gradient w.r.t. colors
    pub certainty_grad: Array3<f32>, // Gradient w.r.t. certainty
}
```

## Training Loop

### Optimizer: SGD with Momentum

```rust
pub struct SGDOptimizer {
    learning_rate: f32,
    momentum: f32,
    velocity: HashMap<String, ChromaticTensor>,
}
```

### Training Steps

1. **Forward Pass**
   - Input → Layer1 → Layer2 → ... → Output
   - Compute predictions

2. **Loss Computation**
   - Cross-entropy for classification
   - Color distance loss (optional regularization)

3. **Backward Pass**
   - Compute ∂L/∂output
   - Backpropagate through each layer
   - Accumulate gradients

4. **Parameter Update**
   - Update weights using optimizer
   - Update bias terms
   - Update operation parameters

### Training Configuration

```rust
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub checkpoint_every: usize,
}
```

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: % correct predictions
- **Per-class precision/recall**
- **Confusion matrix**

### Chromatic Metrics
- **Color distance**: Average distance in RGB space
- **Certainty statistics**: Mean/variance of learned certainties
- **Pattern coherence**: Spatial consistency of colors

### Visualization
- **Learned filters**: Visualize weight tensors
- **Activations**: Layer-by-layer activation maps
- **Attention maps**: Certainty heatmaps
- **Training curves**: Loss/accuracy over time

## Implementation Plan

### Phase 1: Core Components (2 hours)
1. Gradient computation for operations
2. ChromaticLayer struct
3. Forward/backward pass
4. SGD optimizer

### Phase 2: Dataset & Training (2 hours)
1. Color pattern generator
2. Dataset loader
3. Training loop
4. Loss functions

### Phase 3: Experimentation (2 hours)
1. Train on primary colors
2. Evaluate and visualize
3. Hyperparameter tuning
4. Document results

## Expected Results

### Hypothesis
- Network should learn color-specific filters
- Certainty should focus on discriminative regions
- Operations should specialize (e.g., filter for edges, saturate for intensity)

### Success Metrics
- **>80% accuracy** on 3-class primary colors
- **Interpretable learned filters** (visible color patterns)
- **Faster convergence** than random baseline

## Research Extensions

### Future Experiments
1. **Deeper networks**: 5-10 chromatic layers
2. **Attention mechanisms**: Learn certainty weights
3. **Multi-task learning**: Classify color + saturation + harmony
4. **Transfer learning**: Pre-train on synthetic, fine-tune on real images
5. **Adversarial robustness**: Test against color perturbations

## File Structure

```
src/
├── neural/
│   ├── mod.rs
│   ├── layer.rs           # ChromaticLayer
│   ├── network.rs         # ChromaticNetwork
│   ├── gradient.rs        # Gradient computation
│   ├── optimizer.rs       # SGD, Adam, etc.
│   └── loss.rs           # Loss functions
├── data/
│   ├── mod.rs
│   ├── pattern.rs        # Pattern generation
│   └── dataset.rs        # Dataset loader
examples/
├── train_color_classifier.rs
└── visualize_network.rs
```

## Next Steps

1. Implement gradient computation
2. Build ChromaticLayer
3. Create color pattern dataset
4. Implement training loop
5. Train and evaluate
6. Visualize results
7. Write research report
