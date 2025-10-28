//! Meta cognition utilities for awareness and prediction pipelines.
//!
//! This module exposes deterministic self-observation tools (`awareness`)
//! alongside short-term time series prediction utilities (`predict`).

pub mod adapter;
pub mod awareness;
pub mod dissonance;
pub mod ethics;
pub mod log;
pub mod predict;
pub mod reflection;
pub mod trend;

pub use adapter::{MetaAdapter, MetaCycleReport, TrainingControls};
pub use awareness::{Awareness, Observation};
pub use dissonance::{Dissonance, DissonanceWeights};
pub use ethics::{EthicsBounds, EthicsGuard, EthicsVerdict, InterventionSpec, VerdictStatus};
pub use log::{ControlStateSnapshot, MetaActionStatus, MetaLogEntry, MetaLogger};
pub use predict::{Feature, FeatureForecast, PredictionSet, Predictor};
pub use reflection::{plan_reflection, Plan, PlanStep, ReflectionAction};
pub use trend::{detect_anomaly, fit_trend, CycleRecord, TrendModel};
