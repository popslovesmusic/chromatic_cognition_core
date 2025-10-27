//! # Chromatic Cognition Core
//!
//! A deterministic Rust engine that represents cognition as an RGB tensor field.
//! Each cell contains an (r,g,b) triple with scalar certainty, enabling novel
//! approaches to neural computation through color-space operations.
//!
//! ## Quick Start
//!
//! ```rust
//! use chromatic_cognition_core::{ChromaticTensor, mix, filter, complement, saturate};
//!
//! // Create two random tensors
//! let a = ChromaticTensor::from_seed(42, 64, 64, 8);
//! let b = ChromaticTensor::from_seed(100, 64, 64, 8);
//!
//! // Apply operations
//! let mixed = mix(&a, &b);
//! let filtered = filter(&mixed, &b);
//! let complemented = complement(&filtered);
//! let saturated = saturate(&complemented, 1.25);
//!
//! // Get statistics
//! let stats = saturated.statistics();
//! println!("Mean RGB: {:?}", stats.mean_rgb);
//! ```
//!
//! ## Core Modules
//!
//! - [`config`] - Engine configuration via TOML
//! - [`tensor`] - Chromatic tensor types and operations
//! - [`logging`] - JSON line-delimited logging
//! - [`training`] - Loss functions and training metrics

pub mod config;
pub mod data;
pub mod dream;
pub mod learner;
pub mod logging;
pub mod neural;
pub mod solver;
pub mod tensor;
pub mod training;

pub use config::EngineConfig;
pub use dream::{SimpleDreamPool};
pub use learner::{ColorClassifier, MLPClassifier, ClassifierConfig};
pub use learner::training::{TrainingConfig, TrainingResult, train_with_dreams, train_baseline};
pub use solver::{Solver, SolverResult};
pub use solver::native::ChromaticNativeSolver;
pub use tensor::ChromaticTensor;
pub use tensor::gradient::GradientLayer;
pub use tensor::operations::{complement, filter, mix, saturate};
pub use training::{TrainingMetrics, mse_loss};
