//! Dream Pool module for long-term semantic memory
//!
//! This module implements a memory system for storing and retrieving
//! high-coherence ChromaticTensor states (dreams) to accelerate solver
//! convergence through retrieval-based seeding.
//!
//! # Quick Start
//!
//! For most use cases, import from the prelude:
//! ```rust
//! use chromatic_cognition_core::dream::prelude::*;
//!
//! let config = PoolConfig::default();
//! let mut pool = SimpleDreamPool::new(config);
//! let mapper = EmbeddingMapper::new(64);
//! ```
//!
//! # Module Organization
//!
//! - **Core:** `simple_pool` - Dream storage and retrieval
//! - **Phase 4:** `embedding`, `soft_index`, `hybrid_scoring`, `retrieval_mode`
//! - **Phase 3B:** `bias`, `diversity`, `experiment`, `analysis`
//!
//! # API Tiers
//!
//! **Tier 1 (Prelude):** Essential types for 80% of use cases
//! - Import via: `use dream::prelude::*;`
//!
//! **Tier 2 (Direct):** Advanced features, explicitly imported
//! - Import via: `use dream::soft_index::SoftIndex;`
//!
//! **Tier 3 (Modules):** Implementation details, typically not needed
//! - Accessible but not re-exported at top level

// === Public Modules ===

pub mod analysis;
pub mod bias;
pub mod diversity;
pub mod embedding;
pub mod experiment;
pub mod hybrid_scoring;
pub mod prelude;
pub mod retrieval_mode;
pub mod simple_pool;
pub mod soft_index;

// === Tier 1: Core API (Recommended re-exports) ===

/// Main dream pool for storage and retrieval
pub use simple_pool::SimpleDreamPool;

/// Dream pool configuration
pub use simple_pool::PoolConfig;

/// Individual dream entry
pub use simple_pool::DreamEntry;

/// Phase 4: Embedding mapper
pub use embedding::EmbeddingMapper;

/// Phase 4: Retrieval mode selection
pub use retrieval_mode::RetrievalMode;

/// Phase 3B: Bias profile synthesis
pub use bias::BiasProfile;

// === Tier 2: Advanced API (Commonly needed) ===

/// Phase 3B: Bias components (needed for lib.rs re-exports)
pub use bias::{ChromaBias, ClassBias, SpectralBias};

/// Phase 4: Hybrid scoring weights
pub use hybrid_scoring::RetrievalWeights;

/// Phase 4: Soft index types
pub use soft_index::{EntryId, Similarity, SoftIndex};

/// Phase 3B: Experiment harness
pub use experiment::ExperimentHarness;

// === Tier 3: Advanced API (Import from submodules when needed) ===
// - use dream::hybrid_scoring::rerank_hybrid;
// - use dream::embedding::QuerySignature;
// - use dream::analysis::compare_experiments;
// - use dream::diversity::retrieve_diverse_mmr;
