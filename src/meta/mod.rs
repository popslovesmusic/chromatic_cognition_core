//! Meta cognition utilities for awareness and prediction pipelines.
//!
//! This module exposes deterministic self-observation tools (`awareness`)
//! alongside short-term time series prediction utilities (`predict`).

pub mod awareness;
pub mod dissonance;
pub mod predict;
pub mod reflection;

pub use awareness::{Awareness, Observation};
pub use dissonance::{Dissonance, DissonanceWeights};
pub use predict::{Feature, FeatureForecast, PredictionSet, Predictor};
pub use reflection::{plan_reflection, Plan, PlanStep, ReflectionAction};
