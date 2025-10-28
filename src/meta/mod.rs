//! Meta cognition utilities for awareness and prediction pipelines.
//!
//! This module exposes deterministic self-observation tools (`awareness`)
//! alongside short-term time series prediction utilities (`predict`).

pub mod awareness;
pub mod predict;

pub use awareness::{Awareness, Observation};
pub use predict::{Feature, FeatureForecast, PredictionSet, Predictor};
