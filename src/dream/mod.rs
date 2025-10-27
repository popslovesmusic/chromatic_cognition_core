//! Dream Pool module for long-term semantic memory
//!
//! This module implements a memory system for storing and retrieving
//! high-coherence ChromaticTensor states (dreams) to accelerate solver
//! convergence through retrieval-based seeding.

pub mod analysis;
pub mod experiment;
pub mod simple_pool;

pub use analysis::{compare_experiments, generate_report, ExperimentComparison, Statistics};
pub use experiment::{ExperimentConfig, ExperimentHarness, ExperimentResult, SeedingStrategy};
pub use simple_pool::SimpleDreamPool;
