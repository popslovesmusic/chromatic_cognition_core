//! SimpleDreamPool - In-memory dream storage with cosine similarity retrieval
//!
//! This is a minimal implementation designed for validation experiments.
//! It stores ChromaticTensor dreams with their evaluation metrics and provides
//! retrieval based on chromatic signature similarity.

use crate::data::ColorClass;
use crate::tensor::ChromaticTensor;
use crate::solver::SolverResult;
use std::collections::VecDeque;
use std::time::SystemTime;

/// A stored dream entry with tensor and evaluation metrics
///
/// Enhanced for Phase 3B with class awareness, utility tracking, and timestamps
/// Enhanced for Phase 4 with spectral features and embeddings
#[derive(Clone)]
pub struct DreamEntry {
    pub tensor: ChromaticTensor,
    pub result: SolverResult,
    pub chroma_signature: [f32; 3],
    /// Optional class label for class-aware retrieval (Phase 3B)
    pub class_label: Option<ColorClass>,
    /// Utility score from feedback (Phase 3B)
    pub utility: Option<f32>,
    /// Timestamp for recency tracking (Phase 3B)
    pub timestamp: SystemTime,
    /// Number of times this dream has been retrieved (Phase 3B)
    pub usage_count: usize,
    /// Spectral features for embedding (Phase 4)
    pub spectral_features: Option<crate::spectral::SpectralFeatures>,
    /// Cached embedding vector (Phase 4)
    pub embed: Option<Vec<f32>>,
    /// Aggregated mean utility (Phase 4)
    pub util_mean: f32,
}

impl DreamEntry {
    /// Create a new dream entry from a tensor and its evaluation result
    pub fn new(tensor: ChromaticTensor, result: SolverResult) -> Self {
        let chroma_signature = tensor.mean_rgb();
        Self {
            tensor,
            result,
            chroma_signature,
            class_label: None,
            utility: None,
            timestamp: SystemTime::now(),
            usage_count: 0,
            spectral_features: None,
            embed: None,
            util_mean: 0.0,
        }
    }

    /// Create a new dream entry with class label (Phase 3B)
    pub fn with_class(
        tensor: ChromaticTensor,
        result: SolverResult,
        class_label: ColorClass,
    ) -> Self {
        let chroma_signature = tensor.mean_rgb();
        Self {
            tensor,
            result,
            chroma_signature,
            class_label: Some(class_label),
            utility: None,
            timestamp: SystemTime::now(),
            usage_count: 0,
            spectral_features: None,
            embed: None,
            util_mean: 0.0,
        }
    }

    /// Update the utility score for this dream (Phase 3B)
    pub fn set_utility(&mut self, utility: f32) {
        self.utility = Some(utility);
    }

    /// Increment usage count when retrieved (Phase 3B)
    pub fn increment_usage(&mut self) {
        self.usage_count += 1;
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

    /// Add a dream entry with class label (Phase 3B)
    ///
    /// # Arguments
    /// * `tensor` - The chromatic tensor to store
    /// * `result` - The solver evaluation result
    /// * `class_label` - The color class this dream represents
    ///
    /// # Returns
    /// true if the dream was added, false if it didn't meet coherence threshold
    pub fn add_with_class(
        &mut self,
        tensor: ChromaticTensor,
        result: SolverResult,
        class_label: ColorClass,
    ) -> bool {
        if result.coherence < self.config.coherence_threshold {
            return false;
        }

        let entry = DreamEntry::with_class(tensor, result, class_label);

        if self.entries.len() >= self.config.max_size {
            self.entries.pop_front();
        }

        self.entries.push_back(entry);
        true
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

    /// Retrieve K most similar dreams filtered by class label (Phase 3B)
    ///
    /// Only retrieves dreams that match the specified class label.
    ///
    /// # Arguments
    /// * `query_signature` - Target RGB signature to match against
    /// * `target_class` - The class to filter by
    /// * `k` - Number of similar dreams to retrieve
    ///
    /// # Returns
    /// Vector of up to K most similar dreams from the target class
    pub fn retrieve_similar_class(
        &self,
        query_signature: &[f32; 3],
        target_class: ColorClass,
        k: usize,
    ) -> Vec<DreamEntry> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        // Filter by class, then compute similarity
        let mut scored: Vec<(f32, &DreamEntry)> = self
            .entries
            .iter()
            .filter(|entry| entry.class_label == Some(target_class))
            .map(|entry| {
                let similarity = cosine_similarity(query_signature, &entry.chroma_signature);
                (similarity, entry)
            })
            .collect();

        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top K and clone
        scored
            .into_iter()
            .take(k)
            .map(|(_, entry)| entry.clone())
            .collect()
    }

    /// Retrieve dreams with balanced representation across classes (Phase 3B)
    ///
    /// Retrieves `k_per_class` dreams from each specified class.
    ///
    /// # Arguments
    /// * `query_signature` - Target RGB signature to match against
    /// * `classes` - List of classes to retrieve from
    /// * `k_per_class` - Number of dreams to retrieve per class
    ///
    /// # Returns
    /// Vector of dreams with balanced class representation
    pub fn retrieve_balanced(
        &self,
        query_signature: &[f32; 3],
        classes: &[ColorClass],
        k_per_class: usize,
    ) -> Vec<DreamEntry> {
        let mut result = Vec::new();

        for &class in classes {
            let class_dreams = self.retrieve_similar_class(query_signature, class, k_per_class);
            result.extend(class_dreams);
        }

        result
    }

    /// Retrieve dreams filtered by utility threshold (Phase 3B)
    ///
    /// Only retrieves dreams with utility >= threshold.
    ///
    /// # Arguments
    /// * `query_signature` - Target RGB signature
    /// * `k` - Number of dreams to retrieve
    /// * `utility_threshold` - Minimum utility score
    ///
    /// # Returns
    /// Vector of high-utility dreams sorted by similarity
    pub fn retrieve_by_utility(
        &self,
        query_signature: &[f32; 3],
        k: usize,
        utility_threshold: f32,
    ) -> Vec<DreamEntry> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        // Filter by utility, then compute similarity
        let mut scored: Vec<(f32, &DreamEntry)> = self
            .entries
            .iter()
            .filter(|entry| {
                entry.utility.map(|u| u >= utility_threshold).unwrap_or(false)
            })
            .map(|entry| {
                let similarity = cosine_similarity(query_signature, &entry.chroma_signature);
                (similarity, entry)
            })
            .collect();

        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top K and clone
        scored
            .into_iter()
            .take(k)
            .map(|(_, entry)| entry.clone())
            .collect()
    }

    /// Retrieve diverse dreams using Maximum Marginal Relevance (Phase 3B)
    ///
    /// Balances relevance to query with diversity from already-selected dreams.
    /// Uses MMR algorithm to avoid near-duplicates and ensure chromatic variety.
    ///
    /// # Arguments
    /// * `query_signature` - Target RGB signature
    /// * `k` - Number of dreams to retrieve
    /// * `lambda` - Relevance vs diversity tradeoff [0=max diversity, 1=max relevance]
    /// * `min_dispersion` - Minimum required chromatic dispersion (0.0 = no constraint)
    ///
    /// # Returns
    /// Vector of diverse dreams selected by MMR
    ///
    /// # Example
    /// ```
    /// # use chromatic_cognition_core::dream::SimpleDreamPool;
    /// # use chromatic_cognition_core::tensor::ChromaticTensor;
    /// # use chromatic_cognition_core::solver::SolverResult;
    /// let mut pool = SimpleDreamPool::new(100, 0.7);
    /// // ... add dreams ...
    /// let query = [1.0, 0.0, 0.0]; // Red query
    /// let diverse = pool.retrieve_diverse(&query, 5, 0.7, 0.1);
    /// // Returns 5 dreams that are relevant to red but diverse from each other
    /// ```
    pub fn retrieve_diverse(
        &self,
        query_signature: &[f32; 3],
        k: usize,
        lambda: f32,
        min_dispersion: f32,
    ) -> Vec<DreamEntry> {
        use crate::dream::diversity::retrieve_diverse_mmr;

        if self.entries.is_empty() {
            return Vec::new();
        }

        // Convert to slice for MMR algorithm
        let candidates: Vec<DreamEntry> = self.entries.iter().cloned().collect();
        retrieve_diverse_mmr(&candidates, query_signature, k, lambda, min_dispersion)
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

    /// Get diversity statistics for the pool (Phase 3B)
    ///
    /// Computes chromatic dispersion and distance metrics across all dreams.
    ///
    /// # Returns
    /// * DiversityStats with mean/min/max pairwise distances
    pub fn diversity_stats(&self) -> crate::dream::diversity::DiversityStats {
        use crate::dream::diversity::DiversityStats;

        if self.entries.is_empty() {
            return DiversityStats {
                mean_dispersion: 0.0,
                min_distance: 0.0,
                max_distance: 0.0,
                count: 0,
            };
        }

        let entries: Vec<DreamEntry> = self.entries.iter().cloned().collect();
        DiversityStats::compute(&entries)
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

    #[test]
    fn test_class_aware_retrieval() {
        use crate::ChromaticTensor;
        use crate::data::ColorClass;
        use crate::solver::SolverResult;
        use serde_json::json;

        let config = PoolConfig {
            max_size: 20,
            coherence_threshold: 0.5,
            retrieval_limit: 5,
        };
        let mut pool = SimpleDreamPool::new(config);

        // Add dreams with different class labels
        for i in 0..5 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.8,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            pool.add_with_class(tensor, result, ColorClass::Red);
        }

        for i in 5..10 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.8,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            pool.add_with_class(tensor, result, ColorClass::Blue);
        }

        assert_eq!(pool.len(), 10);

        // Retrieve only Red class dreams
        let query = [1.0, 0.0, 0.0];
        let red_dreams = pool.retrieve_similar_class(&query, ColorClass::Red, 3);
        assert_eq!(red_dreams.len(), 3);
        assert!(red_dreams.iter().all(|d| d.class_label == Some(ColorClass::Red)));

        // Retrieve only Blue class dreams
        let blue_dreams = pool.retrieve_similar_class(&query, ColorClass::Blue, 3);
        assert_eq!(blue_dreams.len(), 3);
        assert!(blue_dreams.iter().all(|d| d.class_label == Some(ColorClass::Blue)));
    }

    #[test]
    fn test_balanced_retrieval() {
        use crate::ChromaticTensor;
        use crate::data::ColorClass;
        use crate::solver::SolverResult;
        use serde_json::json;

        let config = PoolConfig::default();
        let mut pool = SimpleDreamPool::new(config);

        // Add dreams from three classes
        for i in 0..5 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.8,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            pool.add_with_class(tensor, result, ColorClass::Red);
        }

        for i in 5..10 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.8,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            pool.add_with_class(tensor, result, ColorClass::Green);
        }

        for i in 10..15 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.8,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            pool.add_with_class(tensor, result, ColorClass::Blue);
        }

        // Retrieve balanced across 3 classes, 2 per class
        let query = [0.5, 0.5, 0.5];
        let classes = vec![ColorClass::Red, ColorClass::Green, ColorClass::Blue];
        let balanced = pool.retrieve_balanced(&query, &classes, 2);

        assert_eq!(balanced.len(), 6); // 2 * 3 classes

        // Count per class
        let red_count = balanced.iter().filter(|d| d.class_label == Some(ColorClass::Red)).count();
        let green_count = balanced.iter().filter(|d| d.class_label == Some(ColorClass::Green)).count();
        let blue_count = balanced.iter().filter(|d| d.class_label == Some(ColorClass::Blue)).count();

        assert_eq!(red_count, 2);
        assert_eq!(green_count, 2);
        assert_eq!(blue_count, 2);
    }

    #[test]
    fn test_utility_retrieval() {
        use crate::ChromaticTensor;
        use crate::solver::SolverResult;
        use serde_json::json;

        let config = PoolConfig {
            max_size: 10,
            coherence_threshold: 0.0,
            retrieval_limit: 5,
        };
        let mut pool = SimpleDreamPool::new(config);

        // Add dreams with varying utility
        for i in 0..5 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.8,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            let mut entry = DreamEntry::new(tensor, result);
            entry.set_utility(0.9); // High utility

            pool.entries.push_back(entry);
        }

        for i in 5..10 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.8,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            let mut entry = DreamEntry::new(tensor, result);
            entry.set_utility(0.1); // Low utility

            pool.entries.push_back(entry);
        }

        // Retrieve only high-utility dreams (utility >= 0.5)
        let query = [0.5, 0.5, 0.5];
        let high_utility = pool.retrieve_by_utility(&query, 10, 0.5);

        assert_eq!(high_utility.len(), 5);
        assert!(high_utility.iter().all(|d| d.utility.unwrap_or(0.0) >= 0.5));
    }
}
