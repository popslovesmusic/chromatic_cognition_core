/// Solver trait and implementations for chromatic field evaluation
///
/// This module provides a trait-based interface for evaluating chromatic tensor fields
/// and computing metrics like energy, coherence, and constraint violations.

pub mod native;

use crate::tensor::ChromaticTensor;
use serde_json::Value;
use anyhow::Result;

/// Result of evaluating a chromatic field
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Total field energy (lower is better)
    /// Combines smoothness (total variation) and saturation penalties
    pub energy: f64,

    /// Field coherence score (0-1, higher is better)
    /// Measures color harmony and consistency
    pub coherence: f64,

    /// Constraint violation score (0-1, lower is better)
    /// Measures out-of-gamut colors, extreme saturation, discontinuities
    pub violation: f64,

    /// Gradient with respect to RGB values (optional)
    /// Length: rows * cols * layers * 3
    pub grad: Option<Vec<f32>>,

    /// Per-cell penalty/attention mask (optional)
    /// Length: rows * cols * layers
    pub mask: Option<Vec<f32>>,

    /// Additional metadata (timings, diagnostics, etc.)
    pub meta: Value,
}

/// Trait for chromatic field solvers/evaluators
///
/// Implementations provide different approaches to evaluating chromatic fields,
/// such as native Rust metrics, external physics engines, or learned evaluators.
pub trait Solver {
    /// Evaluate a chromatic field and optionally compute gradients
    ///
    /// # Arguments
    /// * `field` - The chromatic tensor to evaluate
    /// * `with_grad` - Whether to compute gradients (more expensive)
    ///
    /// # Returns
    /// SolverResult containing energy, coherence, violation, and optional gradients
    fn evaluate(&mut self, field: &ChromaticTensor, with_grad: bool) -> Result<SolverResult>;

    /// Reset internal state to known baseline (if applicable)
    ///
    /// # Arguments
    /// * `seed` - Random seed for deterministic evaluation
    fn reset(&mut self, seed: u64) -> Result<()> {
        let _ = seed; // Default implementation ignores seed
        Ok(())
    }

    /// Get solver name for logging/debugging
    fn name(&self) -> &str {
        "UnknownSolver"
    }
}
