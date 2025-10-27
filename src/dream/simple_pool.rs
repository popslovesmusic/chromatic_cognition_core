//! SimpleDreamPool - In-memory dream storage with cosine similarity retrieval
//!
//! This is a minimal implementation designed for validation experiments.
//! It stores ChromaticTensor dreams with their evaluation metrics and provides
//! retrieval based on chromatic signature similarity.

use crate::tensor::ChromaticTensor;
use crate::solver::SolverResult;
use std::collections::VecDeque;

/// A stored dream entry with tensor and evaluation metrics
#[derive(Clone)]
pub struct DreamEntry {
    pub tensor: ChromaticTensor,
    pub result: SolverResult,
    pub chroma_signature: [f32; 3],
}

impl DreamEntry {
    /// Create a new dream entry from a tensor and its evaluation result
    pub fn new(tensor: ChromaticTensor, result: SolverResult) -> Self {
        let chroma_signature = tensor.mean_rgb();
        Self {
            tensor,
            result,
            chroma_signature,
        }
    }
}

/// Configuration for SimpleDreamPool
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of dreams to store in memory
    pub max_size: usize,
    /// Minimum coherence threshold for persistence (0.0-1.0)
    pub coherence_threshold: f64,
    /// Number of similar dreams to retrieve
    pub retrieval_limit: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_size: 1000,
            coherence_threshold: 0.75,
            retrieval_limit: 3,
        }
    }
}

/// In-memory dream pool with cosine similarity retrieval
///
/// Stores high-coherence ChromaticTensor states and retrieves similar dreams
/// based on chromatic signature (mean RGB) for retrieval-based seeding.
///
/// # Example
///
/// ```rust
/// use chromatic_cognition_core::dream::{SimpleDreamPool, PoolConfig};
/// use chromatic_cognition_core::{ChromaticTensor, ChromaticNativeSolver, Solver};
///
/// let config = PoolConfig::default();
/// let mut pool = SimpleDreamPool::new(config);
///
/// // Evaluate and store a dream
/// let tensor = ChromaticTensor::from_seed(42, 32, 32, 4);
/// let mut solver = ChromaticNativeSolver::default();
/// let result = solver.evaluate(&tensor, false).unwrap();
///
/// pool.add_if_coherent(tensor.clone(), result);
///
/// // Retrieve similar dreams
/// let query_signature = tensor.mean_rgb();
/// let similar = pool.retrieve_similar(&query_signature, 3);
/// ```
pub struct SimpleDreamPool {
    entries: VecDeque<DreamEntry>,
    config: PoolConfig,
}

impl SimpleDreamPool {
    /// Create a new dream pool with the given configuration
    pub fn new(config: PoolConfig) -> Self {
        Self {
            entries: VecDeque::with_capacity(config.max_size),
            config,
        }
    }

    /// Add a dream entry if it meets the coherence threshold
    ///
    /// Returns true if the dream was added, false otherwise.
    /// If the pool is at capacity, the oldest dream is removed (FIFO).
    pub fn add_if_coherent(&mut self, tensor: ChromaticTensor, result: SolverResult) -> bool {
        if result.coherence < self.config.coherence_threshold {
            return false;
        }

        let entry = DreamEntry::new(tensor, result);

        // Remove oldest entry if at capacity
        if self.entries.len() >= self.config.max_size {
            self.entries.pop_front();
        }

        self.entries.push_back(entry);
        true
    }

    /// Force add a dream entry regardless of coherence threshold
    ///
    /// Useful for testing or when coherence filtering is not desired.
    pub fn add(&mut self, tensor: ChromaticTensor, result: SolverResult) {
        let entry = DreamEntry::new(tensor, result);

        if self.entries.len() >= self.config.max_size {
            self.entries.pop_front();
        }

        self.entries.push_back(entry);
    }

    /// Retrieve K most similar dreams based on cosine similarity of chroma signatures
    ///
    /// # Arguments
    /// * `query_signature` - Target RGB signature to match against [r, g, b]
    /// * `k` - Number of similar dreams to retrieve
    ///
    /// # Returns
    /// Vector of up to K most similar dreams, sorted by similarity (highest first)
    pub fn retrieve_similar(&self, query_signature: &[f32; 3], k: usize) -> Vec<DreamEntry> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        // Compute cosine similarity for all entries
        let mut scored: Vec<(f32, &DreamEntry)> = self
            .entries
            .iter()
            .map(|entry| {
                let similarity = cosine_similarity(query_signature, &entry.chroma_signature);
                (similarity, entry)
            })
            .collect();

        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top K and clone the entries
        scored
            .into_iter()
            .take(k)
            .map(|(_, entry)| entry.clone())
            .collect()
    }

    /// Get the number of dreams currently stored
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all stored dreams
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        if self.entries.is_empty() {
            return PoolStats {
                count: 0,
                mean_coherence: 0.0,
                mean_energy: 0.0,
                mean_violation: 0.0,
            };
        }

        let count = self.entries.len();
        let sum_coherence: f64 = self.entries.iter().map(|e| e.result.coherence).sum();
        let sum_energy: f64 = self.entries.iter().map(|e| e.result.energy).sum();
        let sum_violation: f64 = self.entries.iter().map(|e| e.result.violation).sum();

        PoolStats {
            count,
            mean_coherence: sum_coherence / count as f64,
            mean_energy: sum_energy / count as f64,
            mean_violation: sum_violation / count as f64,
        }
    }
}

/// Pool statistics for monitoring and analysis
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub count: usize,
    pub mean_coherence: f64,
    pub mean_energy: f64,
    pub mean_violation: f64,
}

/// Compute cosine similarity between two 3D vectors
///
/// Returns a value in [-1, 1] where 1 means identical direction,
/// 0 means orthogonal, and -1 means opposite direction.
fn cosine_similarity(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mag_a = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let mag_b = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        assert!((cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);

        // Orthogonal vectors
        assert!((cosine_similarity(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]) - 0.0).abs() < 1e-6);

        // Opposite vectors
        assert!((cosine_similarity(&[1.0, 0.0, 0.0], &[-1.0, 0.0, 0.0]) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_pool_add_and_retrieve() {
        use crate::ChromaticTensor;
        use crate::solver::SolverResult;
        use serde_json::json;

        let config = PoolConfig {
            max_size: 5,
            coherence_threshold: 0.5,
            retrieval_limit: 3,
        };
        let mut pool = SimpleDreamPool::new(config);

        // Add some dreams with different signatures
        let tensor1 = ChromaticTensor::from_seed(42, 8, 8, 2);
        let result1 = SolverResult {
            energy: 0.1,
            coherence: 0.9,
            violation: 0.05,
            grad: None,
            mask: None,
            meta: json!({}),
        };

        assert!(pool.add_if_coherent(tensor1.clone(), result1));
        assert_eq!(pool.len(), 1);

        // Retrieve similar to tensor1's signature
        let similar = pool.retrieve_similar(&tensor1.mean_rgb(), 1);
        assert_eq!(similar.len(), 1);
    }

    #[test]
    fn test_pool_capacity() {
        use crate::ChromaticTensor;
        use crate::solver::SolverResult;
        use serde_json::json;

        let config = PoolConfig {
            max_size: 3,
            coherence_threshold: 0.0,
            retrieval_limit: 3,
        };
        let mut pool = SimpleDreamPool::new(config);

        // Add 5 dreams to a pool with max_size = 3
        for i in 0..5 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.9,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            pool.add(tensor, result);
        }

        // Should only have 3 dreams (oldest 2 evicted)
        assert_eq!(pool.len(), 3);
    }
}
