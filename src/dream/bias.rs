//! Bias Profile synthesis for Dream Pool retrieval.
//!
//! This module implements the synthesis of utility feedback into actionable
//! BiasProfiles that can be used to bias future dream retrieval toward
//! high-utility patterns.
//!
//! **Core Concept:** The Learner aggregates feedback about which dreams helped
//! training, synthesizes patterns into a BiasProfile, and feeds it back to the
//! Dreamer to bias future retrieval/generation.

use crate::data::ColorClass;
use crate::learner::feedback::UtilityAggregator;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Bias profile synthesized from utility feedback.
///
/// Contains actionable biases for dream retrieval:
/// - Class-level biases (which color classes are helpful?)
/// - Spectral biases (which frequency characteristics are helpful?)
/// - Chromatic biases (which RGB regions are helpful?)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasProfile {
    /// Per-class utility biases
    pub class_biases: HashMap<String, ClassBias>,

    /// Spectral feature biases
    pub spectral_bias: SpectralBias,

    /// Chromatic signature biases (RGB region preferences)
    pub chroma_bias: ChromaBias,

    /// Metadata about profile generation
    pub metadata: ProfileMetadata,
}

/// Bias information for a specific color class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassBias {
    /// Mean utility for this class [-1, 1]
    pub mean_utility: f32,

    /// Number of feedback samples for this class
    pub sample_count: usize,

    /// Whether to prefer this class (utility > threshold)
    pub prefer: bool,

    /// Weight for retrieval biasing [0, 1]
    pub weight: f32,
}

/// Spectral feature biases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralBias {
    /// Preferred entropy range [min, max]
    pub entropy_range: Option<(f32, f32)>,

    /// Correlation between entropy and utility
    pub entropy_utility_correlation: Option<f32>,

    /// Preferred low-frequency energy threshold
    pub low_freq_threshold: Option<f32>,

    /// Preferred high-frequency energy threshold
    pub high_freq_threshold: Option<f32>,
}

/// Chromatic signature biases (RGB region preferences).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromaBias {
    /// Preferred red channel range [min, max]
    pub red_range: Option<(f32, f32)>,

    /// Preferred green channel range [min, max]
    pub green_range: Option<(f32, f32)>,

    /// Preferred blue channel range [min, max]
    pub blue_range: Option<(f32, f32)>,
}

/// Metadata about bias profile generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileMetadata {
    /// Total number of feedback records used
    pub total_samples: usize,

    /// Overall mean utility
    pub mean_utility: f32,

    /// Timestamp of profile generation (Unix timestamp)
    pub timestamp: u64,

    /// Utility threshold used for "prefer" decisions
    pub utility_threshold: f32,
}

impl BiasProfile {
    /// Synthesize a bias profile from utility feedback.
    ///
    /// # Arguments
    /// * `aggregator` - UtilityAggregator with collected feedback
    /// * `utility_threshold` - Threshold for "prefer" decisions (default: 0.0)
    ///
    /// # Returns
    /// * BiasProfile with synthesized biases
    ///
    /// # Example
    /// ```
    /// # use chromatic_cognition_core::learner::feedback::{FeedbackRecord, UtilityAggregator};
    /// # use chromatic_cognition_core::dream::bias::BiasProfile;
    /// # use chromatic_cognition_core::data::ColorClass;
    /// let mut agg = UtilityAggregator::new();
    /// // ... add feedback records ...
    /// let profile = BiasProfile::from_aggregator(&agg, 0.1);
    /// println!("Mean utility: {}", profile.metadata.mean_utility);
    /// ```
    pub fn from_aggregator(aggregator: &UtilityAggregator, utility_threshold: f32) -> Self {
        // Synthesize class biases
        let class_biases = synthesize_class_biases(aggregator, utility_threshold);

        // Synthesize spectral biases
        let spectral_bias = synthesize_spectral_biases(aggregator);

        // Synthesize chromatic biases
        let chroma_bias = synthesize_chroma_biases(aggregator, utility_threshold);

        // Create metadata
        let metadata = ProfileMetadata {
            total_samples: aggregator.len(),
            mean_utility: aggregator.mean_utility(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            utility_threshold,
        };

        BiasProfile {
            class_biases,
            spectral_bias,
            chroma_bias,
            metadata,
        }
    }

    /// Save bias profile to JSON file.
    ///
    /// # Arguments
    /// * `path` - Path to save the profile
    ///
    /// # Returns
    /// * Result indicating success or error
    pub fn save_to_json<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load bias profile from JSON file.
    ///
    /// # Arguments
    /// * `path` - Path to load the profile from
    ///
    /// # Returns
    /// * Result with BiasProfile or error
    pub fn load_from_json<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let json = fs::read_to_string(path)?;
        let profile = serde_json::from_str(&json)?;
        Ok(profile)
    }

    /// Get the weight for a specific color class.
    ///
    /// Returns 0.0 if class not found.
    pub fn class_weight(&self, class: ColorClass) -> f32 {
        let class_name = format!("{:?}", class);
        self.class_biases
            .get(&class_name)
            .map(|b| b.weight)
            .unwrap_or(0.0)
    }

    /// Check if a class should be preferred.
    pub fn prefer_class(&self, class: ColorClass) -> bool {
        let class_name = format!("{:?}", class);
        self.class_biases
            .get(&class_name)
            .map(|b| b.prefer)
            .unwrap_or(false)
    }

    /// Get preferred classes (those with prefer = true).
    pub fn preferred_classes(&self) -> Vec<String> {
        self.class_biases
            .iter()
            .filter(|(_, bias)| bias.prefer)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Check if spectral entropy is in preferred range.
    pub fn entropy_in_range(&self, entropy: f32) -> bool {
        if let Some((min, max)) = self.spectral_bias.entropy_range {
            entropy >= min && entropy <= max
        } else {
            true // No constraint
        }
    }
}

/// Synthesize class-level biases from feedback.
fn synthesize_class_biases(
    aggregator: &UtilityAggregator,
    utility_threshold: f32,
) -> HashMap<String, ClassBias> {
    let mut class_biases = HashMap::new();

    for (class, stats) in aggregator.all_class_stats() {
        let class_name = format!("{:?}", class);

        // Prefer if mean utility > threshold
        let prefer = stats.mean_utility > utility_threshold;

        // Weight: normalize utility to [0, 1]
        // utility in [-1, 1] → weight in [0, 1]
        let weight = ((stats.mean_utility + 1.0) / 2.0).clamp(0.0, 1.0);

        class_biases.insert(
            class_name,
            ClassBias {
                mean_utility: stats.mean_utility,
                sample_count: stats.count,
                prefer,
                weight,
            },
        );
    }

    class_biases
}

/// Synthesize spectral biases from feedback.
fn synthesize_spectral_biases(aggregator: &UtilityAggregator) -> SpectralBias {
    // Get entropy-utility correlation
    let entropy_utility_correlation = aggregator.entropy_utility_correlation();

    // Compute entropy range for helpful dreams
    let helpful = aggregator.filter_by_utility(0.1);
    let entropies: Vec<f32> = helpful
        .iter()
        .filter_map(|r| r.spectral_features.as_ref().map(|f| f.entropy))
        .collect();

    let entropy_range = if entropies.len() >= 2 {
        let min = entropies.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = entropies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        Some((min, max))
    } else {
        None
    };

    // Compute frequency energy thresholds for helpful dreams
    let low_freqs: Vec<f32> = helpful
        .iter()
        .filter_map(|r| r.spectral_features.as_ref().map(|f| f.low_freq_energy))
        .collect();

    let high_freqs: Vec<f32> = helpful
        .iter()
        .filter_map(|r| r.spectral_features.as_ref().map(|f| f.high_freq_energy))
        .collect();

    let low_freq_threshold = if !low_freqs.is_empty() {
        Some(low_freqs.iter().sum::<f32>() / low_freqs.len() as f32)
    } else {
        None
    };

    let high_freq_threshold = if !high_freqs.is_empty() {
        Some(high_freqs.iter().sum::<f32>() / high_freqs.len() as f32)
    } else {
        None
    };

    SpectralBias {
        entropy_range,
        entropy_utility_correlation,
        low_freq_threshold,
        high_freq_threshold,
    }
}

/// Synthesize chromatic signature biases from feedback.
fn synthesize_chroma_biases(aggregator: &UtilityAggregator, utility_threshold: f32) -> ChromaBias {
    // Get helpful dreams
    let helpful = aggregator.filter_by_utility(utility_threshold);

    if helpful.is_empty() {
        return ChromaBias {
            red_range: None,
            green_range: None,
            blue_range: None,
        };
    }

    // Extract RGB ranges from helpful dreams
    let reds: Vec<f32> = helpful.iter().map(|r| r.chroma_signature[0]).collect();
    let greens: Vec<f32> = helpful.iter().map(|r| r.chroma_signature[1]).collect();
    let blues: Vec<f32> = helpful.iter().map(|r| r.chroma_signature[2]).collect();

    let red_range = compute_range(&reds);
    let green_range = compute_range(&greens);
    let blue_range = compute_range(&blues);

    ChromaBias {
        red_range,
        green_range,
        blue_range,
    }
}

/// Compute [min, max] range from values.
fn compute_range(values: &[f32]) -> Option<(f32, f32)> {
    if values.is_empty() {
        return None;
    }

    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    Some((min, max))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::feedback::FeedbackRecord;

    #[test]
    fn test_bias_profile_creation() {
        let mut agg = UtilityAggregator::new();

        // Add helpful red dreams
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

        // Add harmful green dream
        agg.add_record(FeedbackRecord::new(
            [0.0, 1.0, 0.0],
            Some(ColorClass::Green),
            0.3,
            0.5,
            3,
        ));

        let profile = BiasProfile::from_aggregator(&agg, 0.0);

        assert_eq!(profile.metadata.total_samples, 3);
        assert!(profile.class_biases.contains_key("Red"));
        assert!(profile.class_biases.contains_key("Green"));

        let red_bias = &profile.class_biases["Red"];
        assert!(red_bias.prefer); // Red is helpful
        assert!(red_bias.weight > 0.5); // High weight

        let green_bias = &profile.class_biases["Green"];
        assert!(!green_bias.prefer); // Green is harmful
        assert!(green_bias.weight < 0.5); // Low weight
    }

    #[test]
    fn test_class_weight() {
        let mut agg = UtilityAggregator::new();
        agg.add_record(FeedbackRecord::new(
            [1.0, 0.0, 0.0],
            Some(ColorClass::Red),
            0.5,
            0.3,
            1,
        ));

        let profile = BiasProfile::from_aggregator(&agg, 0.0);
        let weight = profile.class_weight(ColorClass::Red);
        assert!(weight > 0.0);
    }

    #[test]
    fn test_preferred_classes() {
        let mut agg = UtilityAggregator::new();
        agg.add_record(FeedbackRecord::new(
            [1.0, 0.0, 0.0],
            Some(ColorClass::Red),
            0.5,
            0.3,
            1,
        ));
        agg.add_record(FeedbackRecord::new(
            [0.0, 1.0, 0.0],
            Some(ColorClass::Green),
            0.3,
            0.5,
            2,
        ));

        let profile = BiasProfile::from_aggregator(&agg, 0.0);
        let preferred = profile.preferred_classes();

        assert_eq!(preferred.len(), 1);
        assert!(preferred.contains(&"Red".to_string()));
    }

    #[test]
    fn test_chroma_bias_synthesis() {
        let mut agg = UtilityAggregator::new();

        // Helpful dreams in red region [0.8-1.0, 0.0-0.2, 0.0-0.1]
        agg.add_record(FeedbackRecord::new([1.0, 0.0, 0.0], None, 0.5, 0.3, 1));
        agg.add_record(FeedbackRecord::new([0.9, 0.1, 0.0], None, 0.4, 0.2, 2));
        agg.add_record(FeedbackRecord::new([0.8, 0.2, 0.1], None, 0.3, 0.1, 3));

        let profile = BiasProfile::from_aggregator(&agg, 0.0);

        assert!(profile.chroma_bias.red_range.is_some());
        let (r_min, r_max) = profile.chroma_bias.red_range.unwrap();
        assert!((r_min - 0.8).abs() < 0.01);
        assert!((r_max - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_json_serialization() {
        let mut agg = UtilityAggregator::new();
        agg.add_record(FeedbackRecord::new(
            [1.0, 0.0, 0.0],
            Some(ColorClass::Red),
            0.5,
            0.3,
            1,
        ));

        let profile = BiasProfile::from_aggregator(&agg, 0.0);

        // Serialize to JSON
        let json = serde_json::to_string(&profile).unwrap();
        assert!(json.contains("class_biases"));
        assert!(json.contains("spectral_bias"));
        assert!(json.contains("metadata"));

        // Deserialize back
        let profile2: BiasProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(profile2.metadata.total_samples, 1);
    }

    #[test]
    fn test_save_and_load_json() {
        let mut agg = UtilityAggregator::new();
        agg.add_record(FeedbackRecord::new(
            [1.0, 0.0, 0.0],
            Some(ColorClass::Red),
            0.5,
            0.3,
            1,
        ));

        let profile = BiasProfile::from_aggregator(&agg, 0.0);

        // Save to temp file
        let temp_path = "test_bias_profile.json";
        profile.save_to_json(temp_path).unwrap();

        // Load back
        let loaded = BiasProfile::load_from_json(temp_path).unwrap();
        assert_eq!(loaded.metadata.total_samples, 1);

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }
}
