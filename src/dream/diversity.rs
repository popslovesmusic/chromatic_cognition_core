/// Diversity enforcement for Dream Pool retrieval.
///
/// This module implements diversity metrics and Maximum Marginal Relevance (MMR)
/// to ensure retrieved dreams have sufficient chromatic variation.
///
/// **Problem:** Cosine similarity can return near-duplicates, reducing effective
/// batch diversity and limiting training data variation.
///
/// **Solution:** MMR balances relevance (similarity to query) with diversity
/// (dissimilarity to already-selected dreams).

use crate::dream::simple_pool::DreamEntry;

/// Compute pairwise chromatic dispersion between dream entries.
///
/// **Dispersion** = mean pairwise Euclidean distance in RGB space.
///
/// # Arguments
/// * `dreams` - Set of dream entries to analyze
///
/// # Returns
/// * Mean pairwise L2 distance across all RGB channels
/// * Returns 0.0 for empty or single-element sets
///
/// # Example
/// ```
/// # use chromatic_cognition_core::dream::diversity::chroma_dispersion;
/// # use chromatic_cognition_core::dream::simple_pool::DreamEntry;
/// # use chromatic_cognition_core::tensor::ChromaticTensor;
/// # use chromatic_cognition_core::solver::SolverResult;
/// # use serde_json::json;
/// # let tensor = ChromaticTensor::new(2, 2, 4);
/// # let result = SolverResult { energy: 0.1, coherence: 0.8, violation: 0.0, grad: None, mask: None, meta: json!({}) };
/// let mut dreams = vec![
///     DreamEntry::new(tensor.clone(), result.clone()),
///     DreamEntry::new(tensor.clone(), result.clone()),
/// ];
/// // Manually set signatures for testing
/// dreams[0].chroma_signature = [1.0, 0.0, 0.0];
/// dreams[1].chroma_signature = [0.0, 1.0, 0.0];
/// let dispersion = chroma_dispersion(&dreams);
/// assert!(dispersion > 0.0); // Red and Green are dispersed
/// ```
pub fn chroma_dispersion(dreams: &[DreamEntry]) -> f32 {
    if dreams.len() <= 1 {
        return 0.0;
    }

    let n = dreams.len();
    let mut total_distance = 0.0;
    let mut count = 0;

    // Compute pairwise L2 distances
    for i in 0..n {
        for j in (i + 1)..n {
            let sig_i = &dreams[i].chroma_signature;
            let sig_j = &dreams[j].chroma_signature;

            let dist = euclidean_distance(sig_i, sig_j);
            total_distance += dist;
            count += 1;
        }
    }

    if count == 0 {
        0.0
    } else {
        total_distance / count as f32
    }
}

/// Compute Euclidean distance between two chromatic signatures.
///
/// # Arguments
/// * `a` - First RGB signature [r, g, b]
/// * `b` - Second RGB signature [r, g, b]
///
/// # Returns
/// * L2 distance: sqrt((r1-r2)^2 + (g1-g2)^2 + (b1-b2)^2)
#[inline]
fn euclidean_distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dr = a[0] - b[0];
    let dg = a[1] - b[1];
    let db = a[2] - b[2];
    (dr * dr + dg * dg + db * db).sqrt()
}

/// Compute cosine similarity between two chromatic signatures.
///
/// # Arguments
/// * `a` - First RGB signature [r, g, b]
/// * `b` - Second RGB signature [r, g, b]
///
/// # Returns
/// * Cosine similarity in [-1, 1], where 1 = identical direction
#[inline]
fn cosine_similarity(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm_a = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let norm_b = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();

    if norm_a < 1e-8 || norm_b < 1e-8 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Maximum Marginal Relevance (MMR) scoring.
///
/// **MMR** balances relevance to query with diversity from already-selected dreams.
///
/// **Score** = λ × Similarity(dream, query) - (1-λ) × max_j Similarity(dream, selected_j)
///
/// # Arguments
/// * `candidate` - Dream being considered for selection
/// * `query_sig` - Target chromatic signature
/// * `selected` - Dreams already selected
/// * `lambda` - Relevance vs diversity tradeoff [0=max diversity, 1=max relevance]
///
/// # Returns
/// * MMR score (higher = better candidate)
///
/// # Example
/// ```
/// # use chromatic_cognition_core::dream::diversity::mmr_score;
/// # use chromatic_cognition_core::dream::simple_pool::DreamEntry;
/// # use chromatic_cognition_core::tensor::ChromaticTensor;
/// # use chromatic_cognition_core::solver::SolverResult;
/// # use serde_json::json;
/// # let tensor = ChromaticTensor::new(2, 2, 4);
/// # let result = SolverResult { energy: 0.1, coherence: 0.8, violation: 0.0, grad: None, mask: None, meta: json!({}) };
/// let mut candidate = DreamEntry::new(tensor.clone(), result.clone());
/// candidate.chroma_signature = [0.5, 0.5, 0.0];
/// let query = [1.0, 0.0, 0.0]; // Red query
/// let mut selected_dream = DreamEntry::new(tensor.clone(), result.clone());
/// selected_dream.chroma_signature = [0.9, 0.1, 0.0];
/// let selected = vec![selected_dream];
///
/// let score = mmr_score(&candidate, &query, &selected, 0.7);
/// // Score balances relevance to red vs diversity from already-selected red-ish dream
/// ```
pub fn mmr_score(
    candidate: &DreamEntry,
    query_sig: &[f32; 3],
    selected: &[DreamEntry],
    lambda: f32,
) -> f32 {
    // Relevance: similarity to query
    let relevance = cosine_similarity(&candidate.chroma_signature, query_sig);

    // Diversity: maximum similarity to any already-selected dream
    let max_similarity = if selected.is_empty() {
        0.0
    } else {
        selected
            .iter()
            .map(|s| cosine_similarity(&candidate.chroma_signature, &s.chroma_signature))
            .fold(f32::NEG_INFINITY, f32::max)
    };

    // MMR: balance relevance and diversity
    lambda * relevance - (1.0 - lambda) * max_similarity
}

/// Greedy MMR selection from a candidate pool.
///
/// Iteratively selects dreams that maximize MMR score, ensuring both
/// relevance to the query and diversity from already-selected dreams.
///
/// # Arguments
/// * `candidates` - Pool of dream entries to select from
/// * `query_sig` - Target chromatic signature
/// * `k` - Number of dreams to select
/// * `lambda` - MMR tradeoff parameter [0=max diversity, 1=max relevance]
/// * `min_dispersion` - Minimum required dispersion (0.0 = no constraint)
///
/// # Returns
/// * Vec of selected dreams (may be < k if dispersion constraint not met)
///
/// # Example
/// ```
/// # use chromatic_cognition_core::dream::diversity::retrieve_diverse_mmr;
/// # use chromatic_cognition_core::dream::simple_pool::DreamEntry;
/// # use chromatic_cognition_core::tensor::ChromaticTensor;
/// # use chromatic_cognition_core::solver::SolverResult;
/// # use serde_json::json;
/// # let tensor = ChromaticTensor::new(2, 2, 4);
/// # let result = SolverResult { energy: 0.1, coherence: 0.8, violation: 0.0, grad: None, mask: None, meta: json!({}) };
/// let mut d1 = DreamEntry::new(tensor.clone(), result.clone());
/// d1.chroma_signature = [1.0, 0.0, 0.0];
/// let mut d2 = DreamEntry::new(tensor.clone(), result.clone());
/// d2.chroma_signature = [0.9, 0.1, 0.0];
/// let mut d3 = DreamEntry::new(tensor.clone(), result.clone());
/// d3.chroma_signature = [0.0, 1.0, 0.0];
/// let candidates = vec![d1, d2, d3];
/// let query = [1.0, 0.0, 0.0];
/// let selected = retrieve_diverse_mmr(&candidates, &query, 2, 0.7, 0.1);
/// // Should select [1.0, 0.0, 0.0] (most relevant) and [0.0, 1.0, 0.0] (diverse)
/// assert_eq!(selected.len(), 2);
/// ```
pub fn retrieve_diverse_mmr(
    candidates: &[DreamEntry],
    query_sig: &[f32; 3],
    k: usize,
    lambda: f32,
    min_dispersion: f32,
) -> Vec<DreamEntry> {
    if candidates.is_empty() || k == 0 {
        return Vec::new();
    }

    let mut selected = Vec::with_capacity(k);
    let mut remaining: Vec<&DreamEntry> = candidates.iter().collect();

    // Greedy selection: pick best MMR score at each step
    for _ in 0..k {
        if remaining.is_empty() {
            break;
        }

        // Find candidate with highest MMR score
        let (best_idx, _) = remaining
            .iter()
            .enumerate()
            .map(|(idx, candidate)| {
                let score = mmr_score(candidate, query_sig, &selected, lambda);
                (idx, score)
            })
            .max_by(|(_, score_a), (_, score_b)| {
                score_a.partial_cmp(score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        // Add to selected set
        let chosen = remaining.remove(best_idx).clone();
        selected.push(chosen);

        // Check dispersion constraint
        if min_dispersion > 0.0 && selected.len() >= 2 {
            let dispersion = chroma_dispersion(&selected);
            if dispersion < min_dispersion {
                // Constraint violated, stop early
                break;
            }
        }
    }

    selected
}

/// Statistics about dream set diversity.
#[derive(Debug, Clone)]
pub struct DiversityStats {
    /// Mean pairwise chromatic dispersion
    pub mean_dispersion: f32,
    /// Minimum pairwise distance
    pub min_distance: f32,
    /// Maximum pairwise distance
    pub max_distance: f32,
    /// Number of dreams in set
    pub count: usize,
}

impl DiversityStats {
    /// Compute diversity statistics for a set of dreams.
    ///
    /// # Arguments
    /// * `dreams` - Set of dream entries to analyze
    ///
    /// # Returns
    /// * Diversity statistics struct
    pub fn compute(dreams: &[DreamEntry]) -> Self {
        if dreams.len() <= 1 {
            return Self {
                mean_dispersion: 0.0,
                min_distance: 0.0,
                max_distance: 0.0,
                count: dreams.len(),
            };
        }

        let n = dreams.len();
        let mut distances = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = euclidean_distance(
                    &dreams[i].chroma_signature,
                    &dreams[j].chroma_signature,
                );
                distances.push(dist);
            }
        }

        let mean = distances.iter().sum::<f32>() / distances.len() as f32;
        let min = distances.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = distances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        Self {
            mean_dispersion: mean,
            min_distance: min,
            max_distance: max,
            count: dreams.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SolverResult;
    use crate::tensor::ChromaticTensor;
    use serde_json::json;

    fn make_dream(sig: [f32; 3]) -> DreamEntry {
        let tensor = ChromaticTensor::new(2, 2, 4);
        let result = SolverResult {
            energy: 0.1,
            coherence: 0.8,
            violation: 0.0,
            grad: None,
            mask: None,
            meta: json!({}),
        };
        let mut entry = DreamEntry::new(tensor, result);
        entry.chroma_signature = sig;
        entry
    }

    #[test]
    fn test_euclidean_distance() {
        let a = [1.0, 0.0, 0.0]; // Red
        let b = [0.0, 1.0, 0.0]; // Green
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 1.414).abs() < 0.01); // sqrt(2)

        let c = [1.0, 0.0, 0.0]; // Red
        let dist_same = euclidean_distance(&a, &c);
        assert!(dist_same < 1e-6); // Should be ~0
    }

    #[test]
    fn test_chroma_dispersion() {
        // High dispersion: Red, Green, Blue corners
        let diverse = vec![
            make_dream([1.0, 0.0, 0.0]),
            make_dream([0.0, 1.0, 0.0]),
            make_dream([0.0, 0.0, 1.0]),
        ];
        let high_dispersion = chroma_dispersion(&diverse);
        assert!(high_dispersion > 1.0); // Should be ~1.414

        // Low dispersion: Similar reds
        let similar = vec![
            make_dream([1.0, 0.0, 0.0]),
            make_dream([0.9, 0.1, 0.0]),
            make_dream([0.95, 0.05, 0.0]),
        ];
        let low_dispersion = chroma_dispersion(&similar);
        assert!(low_dispersion < 0.2); // Should be very small

        assert!(high_dispersion > low_dispersion);
    }

    #[test]
    fn test_mmr_score_relevance() {
        let query = [1.0, 0.0, 0.0]; // Red query

        // Candidate very similar to query
        let relevant = make_dream([0.9, 0.1, 0.0]);
        let selected = vec![]; // No prior selection

        // With lambda=1.0 (pure relevance), score should be high
        let score_pure_relevance = mmr_score(&relevant, &query, &selected, 1.0);
        assert!(score_pure_relevance > 0.8);
    }

    #[test]
    fn test_mmr_score_diversity() {
        let query = [1.0, 0.0, 0.0]; // Red query

        // Candidate similar to already-selected dream
        let candidate = make_dream([0.9, 0.1, 0.0]);
        let selected = vec![make_dream([0.95, 0.05, 0.0])]; // Already selected similar red

        // With lambda=0.0 (pure diversity), score should be penalized
        let score_pure_diversity = mmr_score(&candidate, &query, &selected, 0.0);
        assert!(score_pure_diversity < 0.0); // Negative due to high similarity to selected

        // With lambda=1.0 (pure relevance), diversity penalty ignored
        let score_pure_relevance = mmr_score(&candidate, &query, &selected, 1.0);
        assert!(score_pure_relevance > 0.8); // Still high relevance
    }

    #[test]
    fn test_retrieve_diverse_mmr() {
        let candidates = vec![
            make_dream([1.0, 0.0, 0.0]),   // Red (most relevant)
            make_dream([0.95, 0.05, 0.0]), // Red-ish (similar to red)
            make_dream([0.0, 1.0, 0.0]),   // Green (diverse but less relevant)
            make_dream([0.0, 0.0, 1.0]),   // Blue (diverse but less relevant)
        ];
        let query = [1.0, 0.0, 0.0]; // Red query

        // With lambda=0.5 (equal balance relevance/diversity), should select red + diverse color
        let selected = retrieve_diverse_mmr(&candidates, &query, 2, 0.5, 0.0);
        assert_eq!(selected.len(), 2);

        // First should be pure red (most relevant)
        let sig0 = selected[0].chroma_signature;
        assert!((sig0[0] - 1.0).abs() < 0.1);

        // Second should be diverse (green or blue, not red-ish)
        let sig1 = selected[1].chroma_signature;
        // Either green (sig1[1] > 0.5) or blue (sig1[2] > 0.5)
        assert!(sig1[1] > 0.5 || sig1[2] > 0.5,
                "Second selection should be diverse (green or blue), got {:?}", sig1);
    }

    #[test]
    fn test_retrieve_diverse_mmr_with_min_dispersion() {
        let candidates = vec![
            make_dream([1.0, 0.0, 0.0]),   // Red
            make_dream([0.99, 0.01, 0.0]), // Almost identical red
            make_dream([0.98, 0.02, 0.0]), // Almost identical red
        ];
        let query = [1.0, 0.0, 0.0];

        // With high min_dispersion, should stop early due to constraint
        let selected = retrieve_diverse_mmr(&candidates, &query, 3, 0.7, 0.5);
        assert!(selected.len() < 3); // Stopped due to dispersion constraint
    }

    #[test]
    fn test_diversity_stats() {
        let dreams = vec![
            make_dream([1.0, 0.0, 0.0]),
            make_dream([0.0, 1.0, 0.0]),
            make_dream([0.0, 0.0, 1.0]),
        ];

        let stats = DiversityStats::compute(&dreams);
        assert_eq!(stats.count, 3);
        assert!(stats.mean_dispersion > 1.0);
        assert!(stats.min_distance > 1.0);
        assert!(stats.max_distance > 1.0);
    }

    #[test]
    fn test_diversity_stats_empty() {
        let dreams = vec![];
        let stats = DiversityStats::compute(&dreams);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean_dispersion, 0.0);
    }
}
