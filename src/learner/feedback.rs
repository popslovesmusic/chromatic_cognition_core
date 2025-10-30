//! Feedback collection and utility scoring for Dream Pool.
//!
//! This module implements ΔLoss-based utility tracking to measure which dreams
//! help or hurt training convergence. Feedback is used to bias future retrieval
//! toward high-utility dreams.
//!
//! **Core Concept:** After training with a dream, compare loss before and after.
//! If loss decreased significantly, the dream was useful. If loss increased or
//! stagnated, the dream was not helpful.

use crate::data::ColorClass;
use crate::spectral::SpectralFeatures;
use std::collections::HashMap;

/// Feedback record for a single dream's impact on training.
///
/// Tracks the change in loss (ΔLoss) after using a dream for augmentation.
#[derive(Debug, Clone)]
pub struct FeedbackRecord {
    /// Chromatic signature of the dream [R, G, B]
    pub chroma_signature: [f32; 3],

    /// Class label if known
    pub class_label: Option<ColorClass>,

    /// Loss before using this dream
    pub loss_before: f32,

    /// Loss after using this dream
    pub loss_after: f32,

    /// Change in loss (negative = improvement)
    pub delta_loss: f32,

    /// Utility score in [-1, 1] (higher = more useful)
    ///
    /// Computed as normalized ΔLoss:
    /// - utility = 1.0 → dream caused maximum loss reduction
    /// - utility = 0.0 → dream had no effect
    /// - utility = -1.0 → dream caused maximum loss increase
    pub utility: f32,

    /// Spectral features of the dream (optional)
    pub spectral_features: Option<SpectralFeatures>,

    /// Epoch when this feedback was recorded
    pub epoch: usize,
}

impl FeedbackRecord {
    /// Create a new feedback record.
    ///
    /// # Arguments
    /// * `chroma_signature` - RGB signature of the dream
    /// * `class_label` - Optional class label
    /// * `loss_before` - Training loss before using the dream
    /// * `loss_after` - Training loss after using the dream
    /// * `epoch` - Current training epoch
    ///
    /// # Returns
    /// * FeedbackRecord with computed ΔLoss and utility
    pub fn new(
        chroma_signature: [f32; 3],
        class_label: Option<ColorClass>,
        loss_before: f32,
        loss_after: f32,
        epoch: usize,
    ) -> Self {
        let delta_loss = loss_after - loss_before;

        // Utility: negative ΔLoss is good (loss decreased)
        // Clamp to reasonable range to avoid outliers dominating
        let utility = -delta_loss.clamp(-1.0, 1.0);

        Self {
            chroma_signature,
            class_label,
            loss_before,
            loss_after,
            delta_loss,
            utility,
            spectral_features: None,
            epoch,
        }
    }

    /// Add spectral features to this feedback record.
    pub fn with_spectral_features(mut self, features: SpectralFeatures) -> Self {
        self.spectral_features = Some(features);
        self
    }

    /// Check if this dream was helpful (utility > 0).
    pub fn was_helpful(&self) -> bool {
        self.utility > 0.0
    }

    /// Check if this dream was harmful (utility < 0).
    pub fn was_harmful(&self) -> bool {
        self.utility < 0.0
    }
}

/// Aggregates feedback records to compute summary statistics.
///
/// Used to identify patterns in useful vs harmful dreams.
#[derive(Debug, Clone)]
pub struct UtilityAggregator {
    /// All feedback records collected
    records: Vec<FeedbackRecord>,

    /// Per-class utility statistics
    class_stats: HashMap<ColorClass, ClassUtilityStats>,
}

/// Utility statistics for a specific color class.
#[derive(Debug, Clone)]
pub struct ClassUtilityStats {
    /// Mean utility for this class
    pub mean_utility: f32,

    /// Number of dreams from this class
    pub count: usize,

    /// Number of helpful dreams (utility > 0)
    pub helpful_count: usize,

    /// Number of harmful dreams (utility < 0)
    pub harmful_count: usize,

    /// Mean spectral entropy (if available)
    pub mean_entropy: Option<f32>,
}

impl UtilityAggregator {
    /// Create a new utility aggregator.
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            class_stats: HashMap::new(),
        }
    }

    /// Add a feedback record.
    pub fn add_record(&mut self, record: FeedbackRecord) {
        // Update class statistics if class label is known
        if let Some(class) = record.class_label {
            let stats = self.class_stats.entry(class).or_insert(ClassUtilityStats {
                mean_utility: 0.0,
                count: 0,
                helpful_count: 0,
                harmful_count: 0,
                mean_entropy: None,
            });

            stats.count += 1;
            if record.was_helpful() {
                stats.helpful_count += 1;
            } else if record.was_harmful() {
                stats.harmful_count += 1;
            }
        }

        self.records.push(record);

        // Recompute class statistics
        self.recompute_stats();
    }

    /// Add multiple feedback records.
    pub fn add_records(&mut self, records: Vec<FeedbackRecord>) {
        for record in records {
            self.add_record(record);
        }
    }

    /// Get all feedback records.
    pub fn records(&self) -> &[FeedbackRecord] {
        &self.records
    }

    /// Get the number of feedback records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if the aggregator is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Get overall mean utility across all dreams.
    pub fn mean_utility(&self) -> f32 {
        if self.records.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.records.iter().map(|r| r.utility).sum();
        sum / self.records.len() as f32
    }

    /// Get utility statistics for a specific class.
    pub fn class_stats(&self, class: ColorClass) -> Option<&ClassUtilityStats> {
        self.class_stats.get(&class)
    }

    /// Get all class statistics.
    pub fn all_class_stats(&self) -> &HashMap<ColorClass, ClassUtilityStats> {
        &self.class_stats
    }

    /// Get the K most helpful dreams (highest utility).
    pub fn top_k_helpful(&self, k: usize) -> Vec<&FeedbackRecord> {
        let mut sorted = self.records.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.utility.partial_cmp(&a.utility).unwrap());
        sorted.into_iter().take(k).collect()
    }

    /// Get the K most harmful dreams (lowest utility).
    pub fn top_k_harmful(&self, k: usize) -> Vec<&FeedbackRecord> {
        let mut sorted = self.records.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| a.utility.partial_cmp(&b.utility).unwrap());
        sorted.into_iter().take(k).collect()
    }

    /// Filter records by minimum utility threshold.
    pub fn filter_by_utility(&self, min_utility: f32) -> Vec<&FeedbackRecord> {
        self.records
            .iter()
            .filter(|r| r.utility >= min_utility)
            .collect()
    }

    /// Get records from a specific class.
    pub fn filter_by_class(&self, class: ColorClass) -> Vec<&FeedbackRecord> {
        self.records
            .iter()
            .filter(|r| r.class_label == Some(class))
            .collect()
    }

    /// Compute correlation between spectral entropy and utility.
    ///
    /// Returns None if insufficient data with spectral features.
    pub fn entropy_utility_correlation(&self) -> Option<f32> {
        let pairs: Vec<(f32, f32)> = self
            .records
            .iter()
            .filter_map(|r| r.spectral_features.as_ref().map(|f| (f.entropy, r.utility)))
            .collect();

        if pairs.len() < 2 {
            return None;
        }

        Some(compute_correlation(&pairs))
    }

    /// Clear all records.
    pub fn clear(&mut self) {
        self.records.clear();
        self.class_stats.clear();
    }

    /// Recompute class statistics from records.
    fn recompute_stats(&mut self) {
        // Reset all stats
        for stats in self.class_stats.values_mut() {
            stats.mean_utility = 0.0;
            stats.mean_entropy = None;
        }

        // Group records by class
        let mut class_records: HashMap<ColorClass, Vec<&FeedbackRecord>> = HashMap::new();
        for record in &self.records {
            if let Some(class) = record.class_label {
                class_records.entry(class).or_default().push(record);
            }
        }

        // Compute mean utility and entropy per class
        for (class, records) in class_records {
            if let Some(stats) = self.class_stats.get_mut(&class) {
                let mean_utility: f32 =
                    records.iter().map(|r| r.utility).sum::<f32>() / records.len() as f32;
                stats.mean_utility = mean_utility;

                // Compute mean entropy if available
                let entropies: Vec<f32> = records
                    .iter()
                    .filter_map(|r| r.spectral_features.as_ref().map(|f| f.entropy))
                    .collect();

                if !entropies.is_empty() {
                    let mean_entropy = entropies.iter().sum::<f32>() / entropies.len() as f32;
                    stats.mean_entropy = Some(mean_entropy);
                }
            }
        }
    }
}

impl Default for UtilityAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute Pearson correlation coefficient between two variables.
///
/// # Arguments
/// * `pairs` - Vec of (x, y) pairs
///
/// # Returns
/// * Correlation coefficient in [-1, 1], or 0.0 if insufficient variance
fn compute_correlation(pairs: &[(f32, f32)]) -> f32 {
    if pairs.len() < 2 {
        return 0.0;
    }

    let n = pairs.len() as f32;
    let mean_x = pairs.iter().map(|(x, _)| x).sum::<f32>() / n;
    let mean_y = pairs.iter().map(|(_, y)| y).sum::<f32>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-10 || var_y < 1e-10 {
        return 0.0; // No variance
    }

    cov / (var_x * var_y).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_record_creation() {
        let record = FeedbackRecord::new([1.0, 0.0, 0.0], Some(ColorClass::Red), 0.5, 0.3, 10);

        assert_eq!(record.chroma_signature, [1.0, 0.0, 0.0]);
        assert_eq!(record.class_label, Some(ColorClass::Red));
        assert_eq!(record.loss_before, 0.5);
        assert_eq!(record.loss_after, 0.3);
        assert!((record.delta_loss + 0.2).abs() < 0.01); // Loss decreased (floating point)
        assert!(record.utility > 0.0); // Positive utility (helpful)
        assert!(record.was_helpful());
        assert!(!record.was_harmful());
    }

    #[test]
    fn test_feedback_record_harmful() {
        let record = FeedbackRecord::new([0.0, 1.0, 0.0], Some(ColorClass::Green), 0.3, 0.5, 10);

        assert!((record.delta_loss - 0.2).abs() < 0.01); // Loss increased (floating point)
        assert!(record.utility < 0.0); // Negative utility (harmful)
        assert!(!record.was_helpful());
        assert!(record.was_harmful());
    }

    #[test]
    fn test_utility_aggregator_add() {
        let mut agg = UtilityAggregator::new();
        assert!(agg.is_empty());

        let record1 = FeedbackRecord::new([1.0, 0.0, 0.0], Some(ColorClass::Red), 0.5, 0.3, 1);
        let record2 = FeedbackRecord::new([0.0, 1.0, 0.0], Some(ColorClass::Green), 0.4, 0.2, 2);

        agg.add_record(record1);
        agg.add_record(record2);

        assert_eq!(agg.len(), 2);
        assert!(!agg.is_empty());
    }

    #[test]
    fn test_utility_aggregator_mean() {
        let mut agg = UtilityAggregator::new();

        // Helpful dream (utility = 0.2)
        agg.add_record(FeedbackRecord::new([1.0, 0.0, 0.0], None, 0.5, 0.3, 1));
        // Harmful dream (utility = -0.2)
        agg.add_record(FeedbackRecord::new([0.0, 1.0, 0.0], None, 0.3, 0.5, 2));

        let mean = agg.mean_utility();
        assert!((mean - 0.0).abs() < 0.01); // Should be ~0 (balanced)
    }

    #[test]
    fn test_utility_aggregator_class_stats() {
        let mut agg = UtilityAggregator::new();

        // Two helpful red dreams
        agg.add_record(FeedbackRecord::new(
            [1.0, 0.0, 0.0],
            Some(ColorClass::Red),
            0.5,
            0.3,
            1,
        ));
        agg.add_record(FeedbackRecord::new(
            [0.9, 0.1, 0.0],
            Some(ColorClass::Red),
            0.4,
            0.2,
            2,
        ));

        // One harmful green dream
        agg.add_record(FeedbackRecord::new(
            [0.0, 1.0, 0.0],
            Some(ColorClass::Green),
            0.3,
            0.5,
            3,
        ));

        let red_stats = agg.class_stats(ColorClass::Red).unwrap();
        assert_eq!(red_stats.count, 2);
        assert_eq!(red_stats.helpful_count, 2);
        assert_eq!(red_stats.harmful_count, 0);
        assert!(red_stats.mean_utility > 0.0);

        let green_stats = agg.class_stats(ColorClass::Green).unwrap();
        assert_eq!(green_stats.count, 1);
        assert_eq!(green_stats.helpful_count, 0);
        assert_eq!(green_stats.harmful_count, 1);
        assert!(green_stats.mean_utility < 0.0);
    }

    #[test]
    fn test_top_k_helpful() {
        let mut agg = UtilityAggregator::new();

        agg.add_record(FeedbackRecord::new([1.0, 0.0, 0.0], None, 0.5, 0.1, 1)); // utility = 0.4
        agg.add_record(FeedbackRecord::new([0.0, 1.0, 0.0], None, 0.5, 0.3, 2)); // utility = 0.2
        agg.add_record(FeedbackRecord::new([0.0, 0.0, 1.0], None, 0.5, 0.4, 3)); // utility = 0.1

        let top2 = agg.top_k_helpful(2);
        assert_eq!(top2.len(), 2);
        assert!(top2[0].utility > top2[1].utility);
        assert!((top2[0].utility - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_filter_by_utility() {
        let mut agg = UtilityAggregator::new();

        agg.add_record(FeedbackRecord::new([1.0, 0.0, 0.0], None, 0.5, 0.1, 1)); // utility = 0.4
        agg.add_record(FeedbackRecord::new([0.0, 1.0, 0.0], None, 0.5, 0.4, 2)); // utility = 0.1
        agg.add_record(FeedbackRecord::new([0.0, 0.0, 1.0], None, 0.5, 0.6, 3)); // utility = -0.1

        let filtered = agg.filter_by_utility(0.15);
        assert_eq!(filtered.len(), 1); // Only the first one
    }

    #[test]
    fn test_compute_correlation() {
        // Perfect positive correlation
        let pairs = vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
        let corr = compute_correlation(&pairs);
        assert!((corr - 1.0).abs() < 0.01);

        // Perfect negative correlation
        let pairs = vec![(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)];
        let corr = compute_correlation(&pairs);
        assert!((corr + 1.0).abs() < 0.01);

        // No correlation
        let pairs = vec![(1.0, 2.0), (1.0, 3.0), (1.0, 1.0)];
        let corr = compute_correlation(&pairs);
        assert!(corr.abs() < 0.1); // Near zero
    }
}
