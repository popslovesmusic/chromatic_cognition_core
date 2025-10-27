//! Chromatic Cognition Core library interface.

pub mod config;
pub mod logging;
pub mod tensor;
pub mod training;

pub use config::EngineConfig;
pub use tensor::ChromaticTensor;
pub use tensor::gradient::GradientLayer;
pub use tensor::operations::{complement, filter, mix, saturate};
pub use training::{TrainingMetrics, mse_loss};
