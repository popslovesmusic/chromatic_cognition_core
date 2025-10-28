//! Prelude module for convenient dream pool imports.
//!
//! This module provides a curated set of commonly-used types and functions
//! for working with the dream pool system.
//!
//! # Usage
//!
//! ```rust
//! use chromatic_cognition_core::dream::prelude::*;
//!
//! // Now you have access to the most common types:
//! let config = PoolConfig::default();
//! let pool = SimpleDreamPool::new(config);
//! let mapper = EmbeddingMapper::new(64);
//! let weights = RetrievalWeights::default();
//! ```
//!
//! # What's Included
//!
//! **Core Types:**
//! - `SimpleDreamPool` - Main dream storage and retrieval
//! - `DreamEntry` - Individual dream with tensor and metrics
//! - `PoolConfig` - Configuration for pool behavior
//!
//! **Phase 4 (Soft Retrieval):**
//! - `EmbeddingMapper` - Feature fusion to embeddings
//! - `RetrievalWeights` - Multi-objective scoring weights
//! - `RetrievalMode` - Hard/Soft/Hybrid mode selection
//!
//! **Phase 3B (Bias & Experiments):**
//! - `BiasProfile` - Utility-driven bias synthesis
//! - `ExperimentHarness` - Validation framework
//!
//! # Advanced Features
//!
//! For advanced usage, import from specific submodules:
//! ```rust
//! // Advanced retrieval
//! use chromatic_cognition_core::dream::soft_index::SoftIndex;
//! use chromatic_cognition_core::dream::hybrid_scoring::rerank_hybrid;
//!
//! // Analysis tools
//! use chromatic_cognition_core::dream::analysis::compare_experiments;
//!
//! // Diversity functions
//! use chromatic_cognition_core::dream::diversity::retrieve_diverse_mmr;
//! ```

// Re-export the most commonly used types

// Core pool types
pub use crate::dream::simple_pool::{DreamEntry, PoolConfig, SimpleDreamPool};

// Phase 4: Soft retrieval essentials
pub use crate::dream::embedding::EmbeddingMapper;
pub use crate::dream::hybrid_scoring::RetrievalWeights;
pub use crate::dream::retrieval_mode::RetrievalMode;

// Phase 3B: Essential for experiments
pub use crate::dream::bias::BiasProfile;
pub use crate::dream::experiment::ExperimentHarness;

// Commonly used enums
pub use crate::dream::soft_index::Similarity;
pub use crate::data::ColorClass;
