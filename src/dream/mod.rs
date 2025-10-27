//! Dream Pool module for long-term semantic memory
//!
//! This module implements a memory system for storing and retrieving
//! high-coherence ChromaticTensor states (dreams) to accelerate solver
//! convergence through retrieval-based seeding.

pub mod analysis;
pub mod bias;
pub mod diversity;
pub mod embedding;
pub mod experiment;
pub mod simple_pool;

pub use analysis::{compare_experiments, generate_report, ExperimentComparison, Statistics};
pub use bias::{BiasProfile, ClassBias, SpectralBias, ChromaBias, ProfileMetadata};
pub use diversity::{chroma_dispersion, mmr_score, retrieve_diverse_mmr, DiversityStats};
pub use embedding::{EmbeddingMapper, QuerySignature};
pub use experiment::{ExperimentConfig, ExperimentHarness, ExperimentResult, SeedingStrategy};
pub use simple_pool::SimpleDreamPool;
