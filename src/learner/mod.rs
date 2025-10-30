//! Learner module - analytical half of the Dreamer-Learner system
//!
//! The Learner extracts structure from Dream Pool entries, evaluates utility,
//! and provides feedback to bias the Dreamer via the retrieval mechanism.
//!
//! This is the Minimal Viable Learner (MVP) that implements:
//! - Color classification via MLP
//! - Gradient descent training
//! - Dream Pool integration for retrieval-based seeding
//! - Basic feedback collection (Δloss tracking)
//!
//! Future expansion path to full LEARNER MANIFEST v1.0 features.

pub mod classifier;
pub mod feedback;
pub mod training;

pub use classifier::{ClassifierConfig, ColorClassifier, MLPClassifier};
pub use feedback::{ClassUtilityStats, FeedbackRecord, UtilityAggregator};
pub use training::{train_with_dreams, TrainingConfig, TrainingResult};
