//! Memory budget tracking and management for dream pool
//!
//! This module implements memory usage tracking and budget enforcement
//! to prevent unbounded memory growth in large dream pools.

use std::mem;
use crate::dream::simple_pool::DreamEntry;

/// Memory budget tracker for dream pool
///
/// Tracks current memory usage and enforces a maximum budget.
/// Triggers eviction when usage exceeds threshold.
///
/// # Example
///
/// ```ignore
/// let mut budget = MemoryBudget::new(100); // 100 MB limit
/// budget.add_entry(entry_size);
/// if budget.needs_eviction() {
///     // Evict entries
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Maximum memory budget in bytes
    max_bytes: usize,
    /// Current memory usage in bytes
    current_bytes: usize,
    /// Number of entries currently tracked
    entry_count: usize,
    /// Eviction threshold (0.0-1.0), triggers at this fraction of max_bytes
    eviction_threshold: f32,
}

impl MemoryBudget {
    /// Create a new memory budget with given limit in megabytes
    ///
    /// # Arguments
    ///
    /// * `max_mb` - Maximum memory budget in megabytes
    ///
    /// # Example
    ///
    /// ```
    /// # use chromatic_cognition_core::dream::memory::MemoryBudget;
    /// let budget = MemoryBudget::new(100); // 100 MB limit
    /// ```
    pub fn new(max_mb: usize) -> Self {
        Self {
            max_bytes: max_mb * 1024 * 1024,
            current_bytes: 0,
            entry_count: 0,
            eviction_threshold: 0.9, // Trigger at 90%
        }
    }

    /// Create a memory budget with custom eviction threshold
    ///
    /// # Arguments
    ///
    /// * `max_mb` - Maximum memory budget in megabytes
    /// * `threshold` - Eviction threshold (0.0-1.0)
    pub fn with_threshold(max_mb: usize, threshold: f32) -> Self {
        Self {
            max_bytes: max_mb * 1024 * 1024,
            current_bytes: 0,
            entry_count: 0,
            eviction_threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Check if adding an entry of given size would exceed budget
    ///
    /// # Arguments
    ///
    /// * `entry_size` - Size of entry in bytes
    ///
    /// # Returns
    ///
    /// true if entry can be added without exceeding budget
    pub fn can_add(&self, entry_size: usize) -> bool {
        self.current_bytes + entry_size <= self.max_bytes
    }

    /// Add an entry to the budget tracker
    ///
    /// # Arguments
    ///
    /// * `entry_size` - Size of entry in bytes
    pub fn add_entry(&mut self, entry_size: usize) {
        self.current_bytes += entry_size;
        self.entry_count += 1;
    }

    /// Remove an entry from the budget tracker
    ///
    /// # Arguments
    ///
    /// * `entry_size` - Size of entry in bytes
    pub fn remove_entry(&mut self, entry_size: usize) {
        self.current_bytes = self.current_bytes.saturating_sub(entry_size);
        self.entry_count = self.entry_count.saturating_sub(1);
    }

    /// Get current memory usage ratio (0.0-1.0)
    ///
    /// # Returns
    ///
    /// Fraction of budget currently used
    pub fn usage_ratio(&self) -> f32 {
        if self.max_bytes == 0 {
            1.0
        } else {
            self.current_bytes as f32 / self.max_bytes as f32
        }
    }

    /// Check if eviction should be triggered
    ///
    /// # Returns
    ///
    /// true if current usage exceeds eviction threshold
    pub fn needs_eviction(&self) -> bool {
        self.usage_ratio() > self.eviction_threshold
    }

    /// Get current memory usage in bytes
    pub fn current_usage(&self) -> usize {
        self.current_bytes
    }

    /// Get maximum memory budget in bytes
    pub fn max_budget(&self) -> usize {
        self.max_bytes
    }

    /// Get number of entries tracked
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Get average entry size in bytes
    pub fn average_entry_size(&self) -> usize {
        if self.entry_count == 0 {
            0
        } else {
            self.current_bytes / self.entry_count
        }
    }

    /// Reset the budget tracker
    pub fn reset(&mut self) {
        self.current_bytes = 0;
        self.entry_count = 0;
    }

    /// Get memory statistics as a formatted string
    pub fn stats(&self) -> String {
        format!(
            "Memory: {:.2} / {:.2} MB ({:.1}%), {} entries, avg {:.2} KB/entry",
            self.current_bytes as f64 / (1024.0 * 1024.0),
            self.max_bytes as f64 / (1024.0 * 1024.0),
            self.usage_ratio() * 100.0,
            self.entry_count,
            self.average_entry_size() as f64 / 1024.0,
        )
    }
}

/// Estimate memory size of a DreamEntry
///
/// Calculates approximate memory usage including:
/// - ChromaticTensor data (colors Array4 + certainty Array3)
/// - SolverResult
/// - Spectral features
/// - Embedding vector (if present)
/// - Metadata
///
/// # Arguments
///
/// * `entry` - The dream entry to measure
///
/// # Returns
///
/// Estimated size in bytes
pub fn estimate_entry_size(entry: &DreamEntry) -> usize {
    let mut size = 0;

    // ChromaticTensor: colors (4D: rows×cols×layers×3) + certainty (3D: rows×cols×layers)
    let shape = entry.tensor.colors.shape();
    let rows = shape[0];
    let cols = shape[1];
    let layers = shape[2];

    let colors_size = rows * cols * layers * 3 * mem::size_of::<f32>();
    let certainty_size = rows * cols * layers * mem::size_of::<f32>();
    size += colors_size + certainty_size;

    // SolverResult: 3 f64 fields (energy, coherence, violation)
    size += 3 * mem::size_of::<f64>();

    // Chroma signature: 3 f32
    size += 3 * mem::size_of::<f32>();

    // Class label: Option<ColorClass> = Option<u8>
    size += mem::size_of::<Option<u8>>();

    // Utility: Option<f32>
    size += mem::size_of::<Option<f32>>();

    // Timestamp: SystemTime (2 × u64)
    size += 2 * mem::size_of::<u64>();

    // Usage count: usize
    size += mem::size_of::<usize>();

    // Spectral features: 6 fields (5 f32 + array of 3 usize)
    size += 5 * mem::size_of::<f32>() + 3 * mem::size_of::<usize>();

    // Embedding vector: Option<Vec<f32>>
    if let Some(ref embed) = entry.embed {
        size += embed.len() * mem::size_of::<f32>();
        size += mem::size_of::<Vec<f32>>(); // Vec overhead
    } else {
        size += mem::size_of::<Option<Vec<f32>>>();
    }

    // Util mean: f32
    size += mem::size_of::<f32>();

    size
}

/// Calculate required eviction count to free target bytes
///
/// # Arguments
///
/// * `budget` - Current memory budget
/// * `target_bytes` - Number of bytes to free
/// * `avg_entry_size` - Average size per entry
///
/// # Returns
///
/// Number of entries to evict
pub fn calculate_eviction_count(budget: &MemoryBudget, target_bytes: usize, avg_entry_size: usize) -> usize {
    if avg_entry_size == 0 {
        return 0;
    }

    let bytes_to_free = if budget.current_usage() > target_bytes {
        budget.current_usage() - target_bytes
    } else {
        0
    };

    (bytes_to_free + avg_entry_size - 1) / avg_entry_size // Ceiling division
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::ChromaticTensor;
    use crate::solver::SolverResult;

    #[test]
    fn test_memory_budget_creation() {
        let budget = MemoryBudget::new(100);
        assert_eq!(budget.max_budget(), 100 * 1024 * 1024);
        assert_eq!(budget.current_usage(), 0);
        assert_eq!(budget.entry_count(), 0);
    }

    #[test]
    fn test_can_add() {
        let mut budget = MemoryBudget::new(1); // 1 MB
        assert!(budget.can_add(500 * 1024)); // 500 KB - OK

        budget.add_entry(900 * 1024); // Add 900 KB
        assert!(!budget.can_add(200 * 1024)); // 200 KB more would exceed
        assert!(budget.can_add(100 * 1024)); // 100 KB is OK
    }

    #[test]
    fn test_add_remove_entry() {
        let mut budget = MemoryBudget::new(10);

        budget.add_entry(1024);
        assert_eq!(budget.current_usage(), 1024);
        assert_eq!(budget.entry_count(), 1);

        budget.add_entry(2048);
        assert_eq!(budget.current_usage(), 3072);
        assert_eq!(budget.entry_count(), 2);

        budget.remove_entry(1024);
        assert_eq!(budget.current_usage(), 2048);
        assert_eq!(budget.entry_count(), 1);
    }

    #[test]
    fn test_usage_ratio() {
        let mut budget = MemoryBudget::new(10); // 10 MB
        assert_eq!(budget.usage_ratio(), 0.0);

        budget.add_entry(5 * 1024 * 1024); // 5 MB
        assert!((budget.usage_ratio() - 0.5).abs() < 0.01);

        budget.add_entry(5 * 1024 * 1024); // Another 5 MB
        assert!((budget.usage_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_needs_eviction() {
        let mut budget = MemoryBudget::with_threshold(10, 0.8); // 10 MB, 80% threshold

        budget.add_entry(7 * 1024 * 1024); // 7 MB (70%)
        assert!(!budget.needs_eviction());

        budget.add_entry(2 * 1024 * 1024); // 9 MB (90%)
        assert!(budget.needs_eviction());
    }

    #[test]
    fn test_average_entry_size() {
        let mut budget = MemoryBudget::new(10);

        budget.add_entry(1000);
        budget.add_entry(2000);
        budget.add_entry(3000);

        assert_eq!(budget.average_entry_size(), 2000); // (1000+2000+3000)/3
    }

    #[test]
    fn test_reset() {
        let mut budget = MemoryBudget::new(10);
        budget.add_entry(1024);
        budget.add_entry(2048);

        budget.reset();
        assert_eq!(budget.current_usage(), 0);
        assert_eq!(budget.entry_count(), 0);
    }

    #[test]
    fn test_estimate_entry_size() {
        let tensor = ChromaticTensor::new(8, 8, 4);
        let result = SolverResult {
            energy: 1.0,
            coherence: 0.9,
            violation: 0.0,
            grad: None,
            mask: None,
            meta: serde_json::json!({}),
        };
        let entry = DreamEntry::new(tensor, result);

        let size = estimate_entry_size(&entry);

        // Should be at least tensor size + metadata
        // colors: 8×8×4×3×4 = 3072 bytes
        // certainty: 8×8×4×4 = 1024 bytes
        let min_size = 3072 + 1024;
        assert!(size >= min_size);

        // Should be less than 20KB for this small tensor (with spectral features, etc.)
        assert!(size < 20 * 1024);
    }

    #[test]
    fn test_calculate_eviction_count() {
        let mut budget = MemoryBudget::new(10); // 10 MB
        budget.add_entry(5 * 1024 * 1024); // 5 MB used

        // Want to free 2 MB, avg entry is 1 MB
        let count = calculate_eviction_count(&budget, 3 * 1024 * 1024, 1024 * 1024);
        assert_eq!(count, 2); // Need to evict 2 entries
    }

    #[test]
    fn test_stats_string() {
        let mut budget = MemoryBudget::new(100);
        budget.add_entry(50 * 1024 * 1024);

        let stats = budget.stats();
        assert!(stats.contains("50.00"));
        assert!(stats.contains("100.00 MB"));
        assert!(stats.contains("50.0%"));
    }
}
