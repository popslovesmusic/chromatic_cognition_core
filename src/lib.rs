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

pub mod bridge;
pub mod config;
pub mod data;
pub mod dream;
pub mod learner;
pub mod logging;
pub mod meta;
pub mod neural;
pub mod solver;
pub mod spectral;
pub mod tensor;
pub mod training;

pub use bridge::{
    decode_from_ums, encode_to_ums, ModalityMapper, UMSVector, UnifiedModalityVector,
};
pub use config::{
    BridgeConfig, EngineConfig, Phase5AConfig, Phase5BConfig, Phase5CConfig, Phase6BConfig,
};
pub use dream::{BiasProfile, ChromaBias, ClassBias, SimpleDreamPool, SpectralBias};
pub use learner::feedback::{ClassUtilityStats, FeedbackRecord, UtilityAggregator};
pub use learner::training::{train_baseline, train_with_dreams, TrainingConfig, TrainingResult};
pub use learner::{ClassifierConfig, ColorClassifier, MLPClassifier};
pub use meta::{
    detect_anomaly, fit_trend, plan_reflection, Awareness, ControlStateSnapshot, CycleRecord,
    Dissonance, DissonanceWeights, EthicsBounds, EthicsGuard, EthicsVerdict, Feature,
    FeatureForecast, InterventionSpec, MetaActionStatus, MetaAdapter, MetaCycleReport,
    MetaLogEntry, MetaLogger, Observation, Plan, PlanStep, PredictionSet, Predictor,
    ReflectionAction, TrainingControls, TrendModel, VerdictStatus,
};
pub use solver::native::ChromaticNativeSolver;
pub use solver::{Solver, SolverResult};
pub use spectral::{
    canonical_hue, circular_mean, compute_spectral_entropy, extract_spectral_features,
    SpectralFeatures, SpectralTensor, WindowFunction,
};
pub use tensor::gradient::GradientLayer;
pub use tensor::operations::{complement, filter, mix, saturate};
pub use tensor::ChromaticTensor;
pub use training::{mse_loss, TrainingMetrics};
