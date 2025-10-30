# Changelog

All notable changes to Chromatic Cognition Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed - Dream Pool Diagnostics
- Logged HNSW construction failures in `SimpleDreamPool::rebuild_soft_index`, ensuring
  ANN fallback decisions capture actionable error details while reverting to the
  linear index budget settings.
- Unified `SimpleDreamPool` add flows behind `internal_add`, keeping memory-budget
  evictions and index bookkeeping consistent across coherence-aware inserts.
- Removed noisy HNSW eviction-path warnings so hot-path tracing remains clean while
  the ANN rebuild diagnostics retain their detailed instrumentation.
<<<<<<< ours
=======
- Added an eviction-threshold counter so ANN/soft indices are only invalidated after
  churn surpasses 10% of the current pool, resetting rebuild markers once indices
  are refreshed.
>>>>>>> theirs

### Added - Spectral Accumulation Determinism
- Introduced Q16.48 fixed-point accumulator with Neumaier-compensated, fixed tree
  reduction for spectral energy sums.
- Updated FFT feature extraction to rely on deterministic rounding when converting
  accumulated spectra back to `f32`, ensuring platform-stable entropy and band
  energy metrics.

### Added - Hue Continuity Bridge
- Spectral bridge adds deterministic hue normalization, seam blending, and reversible
  ChromaticTensor ↔ SpectralTensor conversion with frequency storage.

### Added - Bridge Configuration Loader
- Introduced `BridgeConfig` parser for deterministic hue ↔ frequency parameters,
  including spectral accumulation and reversibility guard-rails (`src/config.rs`).
- Documented `config/bridge.toml` usage alongside engine configuration in the README.

### Added - Modality Mapper Wrapper
- Added `bridge::ModalityMapper` as a deterministic façade over the spectral bridge,
  exposing encode/decode helpers that respect the loaded `BridgeConfig` and emit
  operation logs for tensor conversions (`src/bridge/modality_map.rs`).
- Re-exported the mapper from the crate root so downstream integrations can request
  chromatic ↔ spectral conversions without touching lower-level seam parameters.

### Added - Unified Modality Space Projection
- Implemented `bridge::modality_ums` with a deterministic 512D Unified Modality Space
  vector that embeds spectral categories, replicated HSL channels, and temporal priors
  normalised by Chronicle statistics.
- Added round-trip helpers (`encode_to_ums`, `decode_from_ums`) and unit coverage to
  validate spectral category normalisation and HSL recovery for Phase 7A integration.

### Changed - Unified Modality Space Projection
- Replaced categorical replication with deterministic 2049→256 spectral downsampling,
  per-channel Chronicle μ/σ normalization, and hue consensus checks via
  `ModalityMapper::map_hue_to_category`.
- Hardened the UMS encoder to require the 2049-bin Chronicle layout, average spectral
  energy before block means, seed the affective prior block from Chronicle μ, and add
  regression tests covering spectral aggregation and fixed block windowing.

### Added - Native Rust Solver

### Added - Phase 5A Awareness & Prediction
- **meta::awareness** - `Awareness` collector with bounded history and deterministic statistics (`observe`).
- **meta::predict** - AR(2) `Predictor` with bounded forecasts for coherence, entropy, and gradient energy.
- **Phase5AConfig** - Parses `[p5a]` sections with horizon and feature set defaults.
- **Config/Docs** - Updated README and engine TOML to expose Phase 5A controls.

### Added - Phase 5C Ethics Filter & Meta-Adapter
- **meta::ethics** - Configurable `EthicsGuard` with clipping verdicts for learning rate, tint, and augmentation controls.
- **meta::adapter** - Deterministic adapter that applies reflection plans, supports rollbacks, and exports `MetaCycleReport`.
- **meta::log** - JSONL meta-journal with ordered sequence numbers and sampling via `log_every`.
- **Phase5CConfig** - Parses `[p5c]` safety bounds (`lr_damp_max`, `cool_tint_max`, `pause_aug_max_steps`, `ethics_hue_jump_deg`).
- **TrainingControls** - Canonical self-regulation state shared with Instinct Kernel integrations.

### Added - Phase 6C Continuity Control
- **meta::continuity** - Temporal regulator translating trend slopes into bounded learning-rate and dream-pool adjustments with cooldowns.
- **Phase6CConfig** - Parses `[p6c]` cadence and action bounds for the continuity loop.

### Added - Phase 6D Predictive Diagnostics
- **meta::diagnostics** - Normalizes trend slopes into deterministic `DiagnosticModel` instances with repeatable risk scoring.
- **Phase6DConfig** - Parses `[p6d]` thresholds and risk weights, keeping actions bounded by continuity limits.
- **Continuity integration** - `plan_temporal_action` now queries Phase 6D before applying heuristic adjustments, enabling damp and reset pre-emption.
- **Documentation** - Added `DIAGNOSTICS_SPEC.md`, validation report, and integration log covering oscillation and drift case studies.

#### Solver Module
- **`Solver` trait** - Interface for chromatic field evaluators
- **`SolverResult`** - Standardized result structure (energy, coherence, violation, gradients)
- **`ChromaticNativeSolver`** - Color-theory-informed metric implementation

#### Color-Space Metrics
- **Energy**: Total variation (spatial smoothness) + saturation penalty
- **Coherence**: Color harmony based on complementary balance and hue consistency
- **Violation**: Gamut clipping, extreme saturation, local discontinuities
- **Gradients**: Analytical derivatives for all metrics (fast, accurate)

#### Color Space Utilities
- `rgb_distance()` - Euclidean distance between RGB colors
- `rgb_saturation()` - Saturation computation (max-min)/max
- `rgb_saturation_gradient()` - Gradient of saturation w.r.t. RGB
- `rgb_to_hsv()` - RGB to hue-saturation-value conversion
- `angle_difference()` - Shortest angular distance between hue angles

#### ChromaticTensor Enhancements
- `get_rgb()` - Access RGB values at specific cell
- `dims()` - Get tensor dimensions (rows, cols, layers)
- `mean_rgb()` - Compute mean RGB across all cells
- `total_cells()` - Get total number of cells
- `rows()`, `cols()`, `layers()` - Dimension accessors

#### Examples
- **solver_demo.rs** - Comprehensive solver demonstration
  - Smooth random fields
  - High contrast patterns (checkerboard)
  - Pure RGB colors (saturation analysis)
  - Out-of-gamut violation detection
  - Gradient computation example

#### Testing
- Unit tests for color space utilities (rgb_distance, rgb_saturation, rgb_to_hsv, angle_difference)
- All tests passing (4/4 solver tests)

### Dependencies
- **anyhow 1.0** - Error handling for solver trait

### Updated
- Synchronized training examples with the new `TrainingConfig::retrieval_mode` field and current solver API for example builds.
- Reworked the Phase 3B validation workflow to perform class-aware dream mixing, accumulate utility feedback, and persist bias profile artifacts.
- Refreshed dream module documentation snippets and removed unused imports to keep doctests compiling without warnings.

### Analysis

#### Dream Pool Specification Evaluation
- **DREAM_POOL_EVALUATION.md** - Pre-implementation assessment
- Evaluated proposed long-term semantic memory system for chromatic tensors
- Analyzed coherence definition (spectral entropy vs. existing color harmony metric)
- Assessed chromatic tokenization (manual hue/saturation vs. embedding-based)
- Implementation complexity: 4800 LOC, 1 month, 8 new dependencies (FFT, NLP, DB)
- **Value proposition concerns:** Unclear use case, unvalidated assumptions
- **Recommendation:** ❌ Defer full implementation
- **Alternative path:** SimpleDreamPool prototype (1 week) → validation → decision gate
- **Identified risks:** Coherence metric collision, high opportunity cost, technical complexity

#### DASE Integration Assessment
- **DASE_INTEGRATION_ANALYSIS.md** - Comprehensive feasibility study
- Analyzed DASE (Discrete Analog Simulation Engine) at `D:\isoG\New-folder\sase_amp_fixed`
- Identified architectural mismatch: DASE is analog circuit simulator, not constraint solver
- Evaluated 4 integration options (IGSOA Adapter, Phase 4B, Native Rust, Hybrid)
- **Recommendation:** Implement native Rust solver for color-space metrics
- Future research opportunity: IGSOA quantum-inspired physics for color reasoning

## [0.2.0] - 2025-10-27

### Added - Chromatic Neural Network

#### Neural Network Components
- **Gradient computation** for all chromatic operations (mix, filter, complement, saturate)
- **ChromaticLayer** - Neural network layer with learnable weights and biases
- **ChromaticNetwork** - Multi-layer network for classification
- **SGDOptimizer** - Stochastic gradient descent with momentum
- **AdamOptimizer** - Adaptive moment estimation optimizer
- **Loss functions** - Cross-entropy and MSE with gradients
- **Accuracy metrics** - Classification evaluation

#### Data Generation
- **ColorPattern** dataset structure
- **Primary color dataset generator** - Synthetic red/green/blue patterns
- **Dataset splitting** - Train/validation split
- **Dataset shuffling** - Randomized sampling

#### Training Infrastructure
- **Forward pass** through multi-layer networks
- **Backward pass** with gradient computation
- **Parameter updates** via optimizer
- **Batch evaluation** for validation
- **Per-class performance** metrics

#### Examples
- **train_color_classifier** - Complete training pipeline
- Achieves **100% accuracy** on 3-class color classification
- Generates visualization of predictions

#### Documentation
- **NEURAL_NETWORK_DESIGN.md** - Architecture specification
- **RESEARCH_RESULTS.md** - Experimental findings and analysis
- Comprehensive API documentation for neural components

### Results

**Breakthrough Achievement:**
- Trained chromatic neural network on color classification
- **100% training accuracy** (120 samples)
- **100% validation accuracy** (30 samples)
- **100% per-class accuracy** (red, green, blue)
- Loss decreased from 0.9858 to 0.9708 over 20 epochs
- Stable training with no overfitting

### Performance

Network specifications:
- Input: 16×16×4 chromatic tensors
- Architecture: 2 chromatic layers
- Operations: Saturate + Mix
- Training time: ~2 seconds (20 epochs, 120 samples)

## [0.1.0] - 2025-10-26

### Added - Milestone 1: Chromatic Tensor Core

#### Core Tensor System
- **ChromaticTensor** struct with 4D RGB tensor and 3D certainty weights
- Deterministic random initialization via `from_seed()` using LCG
- Zero initialization via `new()`
- Construction from existing arrays via `from_arrays()`
- `normalize()` method to clamp values to [0.0, 1.0]
- `clamp()` method for arbitrary range limiting
- `statistics()` method for mean RGB, variance, and certainty analysis
- `Display` trait implementation for readable tensor summaries
- Arithmetic operators: `Add` and `Sub` with certainty averaging

#### Primitive Operations
- **mix()** - Additive coherence with normalization
- **filter()** - Subtractive distinction with clamping
- **complement()** - 180° hue rotation (inverts G and B channels)
- **saturate()** - Chroma adjustment by scaling deviation from mean
- All operations parallelized with rayon
- Automatic operation logging to JSON

#### Gradient Projection
- **GradientLayer** for certainty-weighted 3D → 2D projection
- PNG export via plotters backend
- `to_png()` method for visualization output
- Automatic directory creation for output files

#### Configuration System
- TOML-based configuration via `EngineConfig`
- Fields: rows, cols, layers, seed, device
- `load_from_file()` for loading config files
- `from_str()` for parsing TOML strings
- Sensible defaults (64×64×8, seed 42, CPU)
- Graceful fallback to defaults on error

#### Logging Infrastructure
- JSON line-delimited logging format (JSONL)
- **OperationLogEntry** with timestamp, statistics, and operation name
- **TrainingLogEntry** with iteration, loss, and metrics
- Separate log files: `logs/operations.jsonl` and `logs/run.jsonl`
- Automatic log directory creation
- Non-blocking logging (errors printed to stderr)

#### Training Support
- **TrainingMetrics** struct with loss and tensor statistics
- **mse_loss()** function for mean squared error computation
- Parallelized loss calculation with rayon
- Integration with logging system

#### Testing
- Unit tests for all operations (mix, filter, complement, saturate)
- Gradient layer projection tests
- MSE loss computation tests
- Test utilities for creating sample tensors
- All 6 tests passing

#### Examples & Demos
- `examples/demo.rs` showcasing full pipeline:
  - Config loading
  - Tensor initialization
  - Operation chaining (mix → filter → complement → saturate)
  - Gradient projection and PNG export
  - Loss computation and logging

#### Documentation
- Comprehensive README.md with Quick Start guide
- Architecture documentation in `docs/ARCHITECTURE.md`
- API reference in `docs/API.md`
- Inline rustdoc comments on all public APIs
- Code examples in documentation

#### Project Infrastructure
- Cargo.toml with all dependencies
- .gitignore for Rust projects (target/, out/, logs/)
- TOML config file: `config/engine.toml`
- Example output directories: out/, logs/

### Fixed

- **Cargo.toml**: Changed edition from "2024" to "2021"
- **Cargo.toml**: Added missing `toml` dependency for config parsing
- **Cargo.toml**: Fixed plotters features (`bitmap_backend`, `bitmap_encoder`)
- **chromatic_tensor.rs**: Replaced `axis_iter_mut` with direct indexing in `complement()`
- **chromatic_tensor.rs**: Replaced `axis_iter_mut` with direct indexing in `saturate()`
- **chromatic_tensor.rs**: Fixed `statistics()` to handle non-contiguous arrays
- **operations.rs**: Replaced deprecated `par_apply` with `par_for_each`
- **tests/operations.rs**: Corrected gradient_layer test expectations

### Dependencies

- ndarray 0.15 (N-dimensional arrays with rayon support)
- rayon 1.8 (Data parallelism)
- serde 1.0 (Serialization framework)
- serde_json 1.0 (JSON serialization)
- toml 0.8 (TOML configuration parsing)
- plotters 0.3 (PNG visualization)

### Performance

Current benchmarks on 64×64×8 tensor (130,560 cells):
- Random initialization: ~5ms
- Mix operation: ~2ms
- Filter operation: ~2ms
- Complement operation: ~15ms
- Saturate operation: ~25ms
- Gradient projection: ~50ms
- PNG export: ~10ms

### Known Limitations

- CPU-only (GPU support planned for future release)
- Nested loops in `complement()` and `saturate()` (optimization opportunity)
- Limited to f32 precision
- No gradient computation for backpropagation yet

## [Unreleased]

### Added

- Deterministic CIE ΔE94 color difference metric with fixed D65/2° parameters and round-trip tolerance constant (`src/spectral/color.rs`).
- Explicit sRGB → CIELAB conversion helper exported for chromatic reversibility checks (`src/spectral/mod.rs`).

### Changed

- Re-exported spectral utilities now include perceptual distance helpers for reuse across dream and solver modules.

---

### Added

- Phase 5B dissonance scoring module with meta-log output and adaptive reflection planner driven by configurable thresholds.
- Regression testing documentation capturing `cargo test` results and reproduction steps (`docs/TEST_REPORT.md`).

### Changed

- Hardened Phase 4 memory budget and HNSW integration: ANN-aware eviction math, deterministic distance clamping, and safe fallback wiring (`src/dream/memory.rs`, `src/dream/hnsw_index.rs`, `src/dream/simple_pool.rs`).
- Expanded the Phase 4 audit report with line-level findings, recommendations, and follow-up actions (`docs/PHASE4_MEMORY_HNSW_AUDIT.md`).

### Planned for Milestone 2: Gradient Projection + Logger

- [ ] Enhanced gradient computation
- [ ] Advanced logging features
- [ ] Performance profiling tools
- [ ] Additional visualization options
- [ ] Benchmark suite

### Planned for Milestone 3: Training Loop

- [ ] Gradient descent implementation
- [ ] Multiple loss functions (L1, cross-entropy, etc.)
- [ ] Training callbacks and hooks
- [ ] Checkpoint saving and loading
- [ ] Learning rate scheduling

### Future: GPU Support

- [ ] Port to Candle framework
- [ ] CUDA backend support
- [ ] Metal backend support (macOS)
- [ ] Performance comparison CPU vs GPU
- [ ] Multi-GPU training support

---

## Initial Commit - 2025-10-25

- Initialize Rust crate with ChromaticTensor data model and certainty tracking
- Implement parallel tensor primitives with deterministic logging
- Add gradient visualization, CPU MSE loss, and logging utilities
- Provide example demo, configuration defaults, report, and tests
