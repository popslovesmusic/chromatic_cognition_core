//! Hybrid scoring combining similarity, utility, class matching, and diversity.
//!
//! Implements multi-objective retrieval scoring for Phase 4:
//! - α·similarity (from SoftIndex)
//! - β·utility (from ΔLoss tracking)
//! - γ·class_match (optional class conditioning)
//! - δ·MMR_penalty (diversity enforcement)

use crate::dream::simple_pool::DreamEntry;
use crate::dream::soft_index::EntryId;
use crate::data::ColorClass;
use std::collections::HashMap;

/// Weights for hybrid retrieval scoring
#[derive(Debug, Clone, Copy)]
pub struct RetrievalWeights {
    /// Similarity weight (default: 0.65)
    pub alpha: f32,

    /// Utility weight (default: 0.20)
    pub beta: f32,

    /// Class match weight (default: 0.10)
    pub gamma: f32,

    /// Duplicate penalty weight (default: 0.05)
    pub delta: f32,

    /// MMR lambda for diversity (default: 0.7, higher = more diversity)
    pub lambda: f32,
}

impl Default for RetrievalWeights {
    fn default() -> Self {
        Self {
            alpha: 0.65,
            beta: 0.20,
            gamma: 0.10,
            delta: 0.05,
            lambda: 0.7,
        }
    }
}

impl RetrievalWeights {
    /// Create new weights with validation
    pub fn new(alpha: f32, beta: f32, gamma: f32, delta: f32, lambda: f32) -> Self {
        assert!(alpha >= 0.0 && alpha <= 1.0, "alpha must be in [0, 1]");
        assert!(beta >= 0.0 && beta <= 1.0, "beta must be in [0, 1]");
        assert!(gamma >= 0.0 && gamma <= 1.0, "gamma must be in [0, 1]");
        assert!(delta >= 0.0 && delta <= 1.0, "delta must be in [0, 1]");
        assert!(lambda >= 0.0 && lambda <= 1.0, "lambda must be in [0, 1]");

        Self { alpha, beta, gamma, delta, lambda }
    }

    /// Normalize weights so alpha + beta + gamma + delta = 1.0
    pub fn normalize(&mut self) {
        let sum = self.alpha + self.beta + self.gamma + self.delta;
        if sum > 0.0 {
            self.alpha /= sum;
            self.beta /= sum;
            self.gamma /= sum;
            self.delta /= sum;
        }
    }
}

/// Rerank initial SoftIndex hits using hybrid scoring
///
/// # Arguments
/// * `hits` - Initial k-NN results from SoftIndex (id, similarity_score)
/// * `weights` - Hybrid scoring weights
/// * `entries` - Map from EntryId to DreamEntry for utility/class lookup
/// * `query_class` - Optional class hint for class matching bonus
///
/// # Returns
/// * Vec<(EntryId, f32)> sorted by hybrid score (descending)
pub fn rerank_hybrid(
    hits: &[(EntryId, f32)],
    weights: &RetrievalWeights,
    entries: &HashMap<EntryId, DreamEntry>,
    query_class: Option<ColorClass>,
) -> Vec<(EntryId, f32)> {
    if hits.is_empty() {
        return Vec::new();
    }

    // Normalize similarity scores to [0, 1]
    let sim_scores: Vec<f32> = hits.iter().map(|(_, s)| *s).collect();
    let (sim_min, sim_max) = min_max(&sim_scores);
    let sim_range = sim_max - sim_min;

    // Compute base hybrid scores (without diversity penalty)
    let mut scored: Vec<(EntryId, f32, f32)> = hits
        .iter()
        .filter_map(|(id, sim)| {
            entries.get(id).map(|entry| {
                // Normalize similarity
                let norm_sim = if sim_range > 1e-6 {
                    (sim - sim_min) / sim_range
                } else {
                    1.0
                };

                // Utility score (normalized to [0, 1])
                let utility = entry.util_mean.clamp(0.0, 1.0);

                // Class match bonus
                let class_match = match (query_class, entry.class_label) {
                    (Some(q), Some(e)) if q == e => 1.0,
                    (Some(_), Some(_)) => 0.0,
                    _ => 0.5, // Neutral if no class info
                };

                // Base score (before diversity penalty)
                let base_score = weights.alpha * norm_sim
                    + weights.beta * utility
                    + weights.gamma * class_match;

                (*id, base_score, norm_sim)
            })
        })
        .collect();

    // Sort by base score descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Apply MMR diversity penalty
    if weights.lambda > 0.0 {
        scored = apply_mmr_penalty(scored, weights.lambda, weights.delta, entries);
    }

    // Return final scores
    scored.into_iter().map(|(id, score, _)| (id, score)).collect()
}

/// Apply Maximum Marginal Relevance (MMR) penalty for diversity
///
/// Iteratively selects entries, penalizing those too similar to already-selected ones.
fn apply_mmr_penalty(
    mut candidates: Vec<(EntryId, f32, f32)>,
    lambda: f32,
    delta: f32,
    entries: &HashMap<EntryId, DreamEntry>,
) -> Vec<(EntryId, f32, f32)> {
    if candidates.len() <= 1 {
        return candidates;
    }

    let mut selected: Vec<(EntryId, f32, f32)> = Vec::new();
    let mut remaining = candidates;

    // Select first entry (highest base score)
    if let Some(first) = remaining.first() {
        selected.push(*first);
        remaining.remove(0);
    }

    // Iteratively select remaining entries with MMR penalty
    while !remaining.is_empty() {
        let mut best_idx = 0;
        let mut best_mmr_score = f32::NEG_INFINITY;

        for (idx, (id, base_score, _)) in remaining.iter().enumerate() {
            // Compute max similarity to already-selected entries
            let max_sim = selected
                .iter()
                .filter_map(|(sel_id, _, _)| {
                    let entry = entries.get(id)?;
                    let sel_entry = entries.get(sel_id)?;
                    Some(chroma_similarity(&entry.chroma_signature, &sel_entry.chroma_signature))
                })
                .fold(0.0f32, |a, b| a.max(b));

            // MMR score: balance relevance and diversity
            // Higher lambda = more diversity, lower lambda = more relevance
            let mmr_score = lambda * base_score - delta * max_sim;

            if mmr_score > best_mmr_score {
                best_mmr_score = mmr_score;
                best_idx = idx;
            }
        }

        // Move best to selected
        let best = remaining.remove(best_idx);
        selected.push((best.0, best_mmr_score, best.2));
    }

    selected
}

/// Compute cosine similarity between two chromatic signatures
#[inline]
fn chroma_similarity(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Find min and max values in a slice
fn min_max(values: &[f32]) -> (f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for &v in values {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    (min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::ChromaticTensor;
    use crate::solver::SolverResult;

    fn mock_entry(id: EntryId, chroma: [f32; 3], utility: f32, class: Option<ColorClass>) -> (EntryId, DreamEntry) {
        let tensor = ChromaticTensor::new(4, 4, 3);
        let result = SolverResult {
            energy: 0.1,
            coherence: 0.8,
            violation: 0.05,
            grad: None,
            mask: None,
            meta: serde_json::json!({}),
        };

        (id, DreamEntry {
            tensor,
            result,
            chroma_signature: chroma,
            class_label: class,
            utility: Some(utility),
            timestamp: std::time::SystemTime::now(),
            usage_count: 0,
            spectral_features: None,
            embed: None,
            util_mean: utility,
        })
    }

    #[test]
    fn test_weights_default() {
        let w = RetrievalWeights::default();
        assert!((w.alpha - 0.65).abs() < 0.01);
        assert!((w.beta - 0.20).abs() < 0.01);
        assert!((w.gamma - 0.10).abs() < 0.01);
        assert!((w.delta - 0.05).abs() < 0.01);
        assert!((w.lambda - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_weights_normalize() {
        let mut w = RetrievalWeights::new(0.5, 0.3, 0.1, 0.1, 0.7);
        w.normalize();

        let sum = w.alpha + w.beta + w.gamma + w.delta;
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_rerank_hybrid_basic() {
        let id1 = EntryId::new_v4();
        let id2 = EntryId::new_v4();
        let id3 = EntryId::new_v4();

        let mut entries = HashMap::new();
        entries.insert(id1, mock_entry(id1, [1.0, 0.0, 0.0], 0.9, Some(ColorClass::Red)).1);
        entries.insert(id2, mock_entry(id2, [0.0, 1.0, 0.0], 0.5, Some(ColorClass::Green)).1);
        entries.insert(id3, mock_entry(id3, [0.0, 0.0, 1.0], 0.3, Some(ColorClass::Blue)).1);

        let hits = vec![
            (id1, 0.95), // High similarity
            (id2, 0.80), // Medium similarity
            (id3, 0.60), // Low similarity
        ];

        let weights = RetrievalWeights::default();
        let results = rerank_hybrid(&hits, &weights, &entries, Some(ColorClass::Red));

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, id1); // Best: high sim + high utility + class match
    }

    #[test]
    fn test_rerank_utility_boost() {
        let id1 = EntryId::new_v4();
        let id2 = EntryId::new_v4();

        let mut entries = HashMap::new();
        entries.insert(id1, mock_entry(id1, [1.0, 0.0, 0.0], 0.3, None).1); // Low utility
        entries.insert(id2, mock_entry(id2, [0.9, 0.0, 0.0], 0.9, None).1); // High utility

        let hits = vec![
            (id1, 1.0), // Slightly higher similarity
            (id2, 0.95), // Slightly lower similarity
        ];

        let weights = RetrievalWeights {
            alpha: 0.3, // Lower similarity weight
            beta: 0.7,  // Much higher utility weight
            gamma: 0.0,
            delta: 0.0,
            lambda: 0.0,
        };

        let results = rerank_hybrid(&hits, &weights, &entries, None);

        // id2 should win due to much higher utility
        assert_eq!(results[0].0, id2);
    }

    #[test]
    fn test_rerank_class_match() {
        let id1 = EntryId::new_v4();
        let id2 = EntryId::new_v4();

        let mut entries = HashMap::new();
        entries.insert(id1, mock_entry(id1, [1.0, 0.0, 0.0], 0.5, Some(ColorClass::Red)).1);
        entries.insert(id2, mock_entry(id2, [1.0, 0.0, 0.0], 0.5, Some(ColorClass::Blue)).1);

        let hits = vec![
            (id1, 0.9),
            (id2, 0.9),
        ];

        let weights = RetrievalWeights {
            alpha: 0.3,
            beta: 0.3,
            gamma: 0.4, // High class match weight
            delta: 0.0,
            lambda: 0.0,
        };

        let results = rerank_hybrid(&hits, &weights, &entries, Some(ColorClass::Red));

        // id1 should win due to class match
        assert_eq!(results[0].0, id1);
    }

    #[test]
    fn test_mmr_diversity() {
        let id1 = EntryId::new_v4();
        let id2 = EntryId::new_v4();
        let id3 = EntryId::new_v4();

        let mut entries = HashMap::new();
        entries.insert(id1, mock_entry(id1, [1.0, 0.0, 0.0], 0.8, None).1);
        entries.insert(id2, mock_entry(id2, [0.98, 0.02, 0.0], 0.78, None).1); // Very similar to id1
        entries.insert(id3, mock_entry(id3, [0.0, 1.0, 0.0], 0.76, None).1); // Different

        let hits = vec![
            (id1, 0.95),
            (id2, 0.94), // Similar to id1, slightly lower sim
            (id3, 0.93), // Different, slightly lower sim
        ];

        let weights = RetrievalWeights {
            alpha: 0.5,
            beta: 0.5,
            gamma: 0.0,
            delta: 0.5,  // Higher penalty for duplicates
            lambda: 0.9, // Very high diversity preference
        };

        let results = rerank_hybrid(&hits, &weights, &entries, None);

        // With high lambda, id3 should rank higher than id2 despite similar base scores
        assert_eq!(results[0].0, id1); // Best overall

        // Check that id3 ranks higher than id2 due to diversity
        let id2_rank = results.iter().position(|(id, _)| *id == id2).unwrap();
        let id3_rank = results.iter().position(|(id, _)| *id == id3).unwrap();
        assert!(id3_rank < id2_rank, "id3 should rank higher than id2 due to diversity");
    }

    #[test]
    fn test_chroma_similarity() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let sim = chroma_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.01); // Identical

        let c = [0.0, 1.0, 0.0];
        let sim = chroma_similarity(&a, &c);
        assert!((sim - 0.0).abs() < 0.01); // Orthogonal
    }

    #[test]
    fn test_empty_hits() {
        let entries = HashMap::new();
        let hits = vec![];
        let weights = RetrievalWeights::default();
        let results = rerank_hybrid(&hits, &weights, &entries, None);
        assert!(results.is_empty());
    }
}
