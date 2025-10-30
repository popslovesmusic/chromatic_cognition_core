//! Memory budget tracking and management for dream pool
//!
//! This module implements memory usage tracking and budget enforcement
//! to prevent unbounded memory growth in large dream pools.

use crate::dream::simple_pool::DreamEntry;
use serde::{Deserialize, Serialize};
use std::convert::TryFrom;
use std::mem;

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
    /// Current memory usage in bytes (raw, without ANN overhead adjustment)
    current_bytes: usize,
    /// Number of entries currently tracked
    entry_count: usize,
    /// Eviction threshold (0.0-1.0), triggers at this fraction of max_bytes
    eviction_threshold: f32,
    /// Additional multiplier applied when ANN indexes mirror entry memory usage (e.g. HNSW ≈ 2×)
    ann_overhead_factor: f32,
}

/// Serializable representation of [`MemoryBudget`].
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct MemoryBudgetSnapshot {
    max_bytes: u64,
    current_bytes: u64,
    entry_count: u64,
    eviction_threshold: f32,
    ann_overhead_factor: f32,
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
            ann_overhead_factor: 1.0,
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
            ann_overhead_factor: 1.0,
        }
    }

    /// Set the multiplier applied to memory usage when ANN indexes mirror dream memory
    ///
    /// `factor` is clamped to the range [1.0, 8.0] to avoid runaway scaling.
    pub fn set_ann_overhead_factor(&mut self, factor: f32) {
        self.ann_overhead_factor = factor.clamp(1.0, 8.0);
    }

    /// Get the current ANN overhead multiplier.
    pub fn ann_overhead_factor(&self) -> f32 {
        self.ann_overhead_factor
    }

    /// Compute the raw memory usage adjusted for ANN overhead.
    fn adjusted_usage_bytes(&self) -> usize {
        if self.current_bytes == 0 {
            return 0;
        }

        let adjusted = (self.current_bytes as f64) * (self.ann_overhead_factor as f64);
        if !adjusted.is_finite() {
            usize::MAX
        } else {
            adjusted.ceil().min(usize::MAX as f64).max(0.0) as usize
        }
    }

    /// Compute the eviction threshold in bytes.
    fn threshold_bytes(&self) -> usize {
        if self.max_bytes == 0 {
            return 0;
        }

        let threshold = (self.max_bytes as f64) * (self.eviction_threshold as f64);
        threshold.floor().clamp(0.0, self.max_bytes as f64) as usize
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
        if self.max_bytes == 0 {
            return false;
        }

        let prospective = self.current_bytes.saturating_add(entry_size);
        let adjusted = (prospective as f64) * (self.ann_overhead_factor as f64);
        adjusted <= self.max_bytes as f64
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
            if self.current_bytes == 0 {
                0.0
            } else {
                1.0
            }
        } else {
            self.adjusted_usage_bytes() as f32 / self.max_bytes as f32
        }
    }

    /// Check if eviction should be triggered
    ///
    /// # Returns
    ///
    /// true if current usage exceeds eviction threshold
    pub fn needs_eviction(&self) -> bool {
        if self.max_bytes == 0 {
            return self.current_bytes > 0;
        }

        self.adjusted_usage_bytes() > self.threshold_bytes()
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

    /// Calculate required eviction count to return below threshold.
    ///
    /// Returns zero when the average entry size is unknown or the budget is already satisfied.
    pub fn calculate_eviction_count(&self, avg_entry_size: usize) -> usize {
        if avg_entry_size == 0 || self.entry_count == 0 {
            return 0;
        }

        let threshold_bytes = self.threshold_bytes();
        let usage = self.adjusted_usage_bytes();

        if usage <= threshold_bytes {
            return 0;
        }

        let bytes_to_free = usage - threshold_bytes;
        let count = (bytes_to_free + avg_entry_size - 1) / avg_entry_size;
        count.min(self.entry_count)
    }

    /// Capture the full budget state for checkpointing.
    pub(crate) fn snapshot(&self) -> MemoryBudgetSnapshot {
        MemoryBudgetSnapshot {
            max_bytes: u64::try_from(self.max_bytes).unwrap_or(u64::MAX),
            current_bytes: u64::try_from(self.current_bytes).unwrap_or(u64::MAX),
            entry_count: u64::try_from(self.entry_count).unwrap_or(u64::MAX),
            eviction_threshold: self.eviction_threshold,
            ann_overhead_factor: self.ann_overhead_factor,
        }
    }

    /// Restore a budget instance from a previously captured snapshot.
    pub(crate) fn from_snapshot(snapshot: MemoryBudgetSnapshot) -> Result<Self, String> {
        let max_bytes = usize::try_from(snapshot.max_bytes)
            .map_err(|_| "Memory budget max_bytes exceeds platform capacity".to_string())?;
        let current_bytes = usize::try_from(snapshot.current_bytes)
            .map_err(|_| "Memory budget current_bytes exceeds platform capacity".to_string())?;
        let entry_count = usize::try_from(snapshot.entry_count)
            .map_err(|_| "Memory budget entry_count exceeds platform capacity".to_string())?;

        if !(0.0..=1.0).contains(&snapshot.eviction_threshold) {
            return Err("Memory budget eviction_threshold out of range".to_string());
        }

        if !(1.0..=8.0).contains(&snapshot.ann_overhead_factor) {
            return Err("Memory budget ann_overhead_factor out of range".to_string());
        }

        if current_bytes > max_bytes {
            return Err("Memory budget current_bytes exceeds max_bytes".to_string());
        }

        Ok(Self {
            max_bytes,
            current_bytes,
            entry_count,
            eviction_threshold: snapshot.eviction_threshold,
            ann_overhead_factor: snapshot.ann_overhead_factor,
        })
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
        let adjusted_mb = self.adjusted_usage_bytes() as f64 / (1024.0 * 1024.0);
        format!(
            "Memory: {:.2} / {:.2} MB ({:.1}%), {} entries, avg {:.2} KB/entry, overhead x{:.1}",
            adjusted_mb,
            self.max_bytes as f64 / (1024.0 * 1024.0),
            self.usage_ratio() * 100.0,
            self.entry_count,
            self.average_entry_size() as f64 / 1024.0,
            self.ann_overhead_factor,
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

    // UMS vector: Vec<f32> with 512 elements (Phase 7)
    size += entry.ums_vector.len() * mem::size_of::<f32>();
    size += mem::size_of::<Vec<f32>>(); // Vec overhead

    // Hue category: usize (Phase 7)
    size += mem::size_of::<usize>();

    size
}

/// Calculate required eviction count to free target bytes
///
/// This helper is kept for backwards compatibility with earlier tooling. It forwards to
/// [`MemoryBudget::calculate_eviction_count`] using the budget's internal threshold and ANN
/// overhead adjustment, ignoring the explicit `target_bytes` argument.
#[deprecated(
    since = "0.2.0",
    note = "Use MemoryBudget::calculate_eviction_count(avg_entry_size) instead."
)]
pub fn calculate_eviction_count(
    budget: &MemoryBudget,
    _target_bytes: usize,
    avg_entry_size: usize,
) -> usize {
    budget.calculate_eviction_count(avg_entry_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SolverResult;
    use crate::tensor::ChromaticTensor;

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
        let mut budget = MemoryBudget::with_threshold(10, 0.5); // 10 MB, 50% threshold
        budget.add_entry(3 * 1024 * 1024); // 3 MB
        budget.add_entry(3 * 1024 * 1024); // 6 MB total -> above 5 MB threshold

        // Average entry size is 3 MB, so evicting one entry should restore budget
        let count = budget.calculate_eviction_count(budget.average_entry_size());
        assert_eq!(count, 1);
    }

    #[test]
    fn test_eviction_with_ann_overhead() {
        let mut budget = MemoryBudget::with_threshold(10, 0.9); // 10 MB, 90% threshold
        budget.set_ann_overhead_factor(2.0); // HNSW ~2× memory

        budget.add_entry(4 * 1024 * 1024); // Raw 4 MB => adjusted 8 MB
        assert!(!budget.needs_eviction());

        budget.add_entry(1 * 1024 * 1024); // Raw 5 MB => adjusted 10 MB (> 9 MB threshold)
        assert!(budget.needs_eviction());

        // Average entry size is 2.5 MB, eviction should drop adjusted usage below threshold
        let count = budget.calculate_eviction_count(budget.average_entry_size());
        assert_eq!(count, 1);
    }

    #[test]
    fn test_zero_budget_handling() {
        let mut budget = MemoryBudget::new(0);
        assert!(!budget.needs_eviction());

        budget.add_entry(512);
        assert!(budget.needs_eviction());
        assert_eq!(budget.calculate_eviction_count(512), 1);
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
