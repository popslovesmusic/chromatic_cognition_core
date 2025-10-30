//! Soft index for approximate nearest neighbor search with embeddings.
//!
//! In-memory ANN-lite supporting cosine and euclidean similarity.

use crate::dream::error::{DreamError, DreamResult};
use serde::{Deserialize, Serialize};

/// Unique identifier for indexed entries
pub type EntryId = uuid::Uuid;

/// Similarity metric for retrieval
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Similarity {
    /// Cosine similarity: dot(a,b) / (||a|| * ||b||)
    Cosine,
    /// Euclidean distance: ||a - b||
    Euclidean,
}

/// In-memory soft index for semantic retrieval
pub struct SoftIndex {
    /// Embedding dimension
    dim: usize,

    /// Entry IDs
    ids: Vec<EntryId>,

    /// Embedding vectors
    vecs: Vec<Vec<f32>>,

    /// Pre-computed norms for cosine similarity
    norms: Vec<f32>,
}

/// Serializable snapshot of a [`SoftIndex`].
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct SoftIndexSnapshot {
    dim: usize,
    ids: Vec<EntryId>,
    vecs: Vec<Vec<f32>>,
    norms: Vec<f32>,
}

impl SoftIndex {
    /// Create a new soft index
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            ids: Vec::new(),
            vecs: Vec::new(),
            norms: Vec::new(),
        }
    }

    /// Add an entry to the index
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if vector dimension doesn't match index dimension
    pub fn add(&mut self, id: EntryId, vec: Vec<f32>) -> DreamResult<()> {
        if vec.len() != self.dim {
            return Err(DreamError::dimension_mismatch(
                self.dim,
                vec.len(),
                "SoftIndex add",
            ));
        }

        self.ids.push(id);
        self.vecs.push(vec);
        self.norms.push(0.0); // Will be computed in build()
        Ok(())
    }

    /// Build the index (compute norms)
    pub fn build(&mut self) {
        self.norms = self.vecs.iter().map(|v| l2_norm(v)).collect();
    }

    /// Query for K nearest neighbors
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if query dimension doesn't match index dimension
    pub fn query(
        &self,
        query: &[f32],
        k: usize,
        mode: Similarity,
    ) -> DreamResult<Vec<(EntryId, f32)>> {
        if query.len() != self.dim {
            return Err(DreamError::dimension_mismatch(
                self.dim,
                query.len(),
                "SoftIndex query",
            ));
        }

        if self.ids.is_empty() {
            return Ok(Vec::new());
        }

        // Compute similarities
        let query_norm = l2_norm(query);
        let mut scores: Vec<(usize, f32)> = self
            .vecs
            .iter()
            .enumerate()
            .map(|(idx, vec)| {
                let score = match mode {
                    Similarity::Cosine => cosine_sim(query, vec, query_norm, self.norms[idx]),
                    Similarity::Euclidean => -euclidean_dist(query, vec), // Negative for sorting
                };
                (idx, score)
            })
            .collect();

        // Sort by score (descending)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-K
        Ok(scores
            .into_iter()
            .take(k)
            .map(|(idx, score)| (self.ids[idx], score))
            .collect())
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.ids.clear();
        self.vecs.clear();
        self.norms.clear();
    }

    /// Capture the full state of the index for checkpointing.
    pub(crate) fn snapshot(&self) -> SoftIndexSnapshot {
        SoftIndexSnapshot {
            dim: self.dim,
            ids: self.ids.clone(),
            vecs: self.vecs.clone(),
            norms: self.norms.clone(),
        }
    }

    /// Rebuild an index from a previously captured snapshot.
    pub(crate) fn from_snapshot(snapshot: SoftIndexSnapshot) -> Self {
        Self {
            dim: snapshot.dim,
            ids: snapshot.ids,
            vecs: snapshot.vecs,
            norms: snapshot.norms,
        }
    }
}

/// Compute L2 norm of a vector
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Compute cosine similarity with pre-computed norms
#[inline]
fn cosine_sim(a: &[f32], b: &[f32], norm_a: f32, norm_b: f32) -> f32 {
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (norm_a * norm_b)
}

/// Compute Euclidean distance
#[inline]
fn euclidean_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soft_index_creation() {
        let index = SoftIndex::new(64);
        assert_eq!(index.dim, 64);
        assert!(index.is_empty());
    }

    #[test]
    fn test_add_and_query() {
        let mut index = SoftIndex::new(3);

        let id1 = EntryId::new_v4();
        let id2 = EntryId::new_v4();
        let id3 = EntryId::new_v4();

        index.add(id1, vec![1.0, 0.0, 0.0]).unwrap();
        index.add(id2, vec![0.0, 1.0, 0.0]).unwrap();
        index.add(id3, vec![0.0, 0.0, 1.0]).unwrap();
        index.build();

        assert_eq!(index.len(), 3);

        // Query with red vector
        let query = vec![1.0, 0.0, 0.0];
        let results = index.query(&query, 2, Similarity::Cosine).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, id1); // Most similar
        assert!((results[0].1 - 1.0).abs() < 0.01); // Cosine sim = 1
    }

    #[test]
    fn test_cosine_similarity() {
        let mut index = SoftIndex::new(3);

        let id1 = EntryId::new_v4();
        let id2 = EntryId::new_v4();

        index.add(id1, vec![1.0, 0.0, 0.0]).unwrap();
        index.add(id2, vec![0.5, 0.5, 0.0]).unwrap();
        index.build();

        let query = vec![1.0, 0.0, 0.0];
        let results = index.query(&query, 2, Similarity::Cosine).unwrap();

        assert_eq!(results[0].0, id1); // Exact match first
        assert!(results[0].1 > results[1].1); // Higher similarity
    }

    #[test]
    fn test_euclidean_distance() {
        let mut index = SoftIndex::new(3);

        let id1 = EntryId::new_v4();
        let id2 = EntryId::new_v4();

        index.add(id1, vec![1.0, 0.0, 0.0]).unwrap();
        index.add(id2, vec![10.0, 0.0, 0.0]).unwrap();
        index.build();

        let query = vec![1.5, 0.0, 0.0];
        let results = index.query(&query, 2, Similarity::Euclidean).unwrap();

        assert_eq!(results[0].0, id1); // Closer in Euclidean space
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        let norm = l2_norm(&v);
        assert!((norm - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_query() {
        let index = SoftIndex::new(3);
        let query = vec![1.0, 0.0, 0.0];
        let results = index.query(&query, 5, Similarity::Cosine).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_dimension_mismatch_add() {
        let mut index = SoftIndex::new(3);
        let id = EntryId::new_v4();

        let result = index.add(id, vec![1.0, 0.0]); // Wrong dimension
        assert!(result.is_err());
        match result {
            Err(DreamError::DimensionMismatch { expected, got, .. }) => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_dimension_mismatch_query() {
        let index = SoftIndex::new(3);
        let query = vec![1.0, 0.0]; // Wrong dimension

        let result = index.query(&query, 5, Similarity::Cosine);
        assert!(result.is_err());
        match result {
            Err(DreamError::DimensionMismatch { expected, got, .. }) => {
                assert_eq!(expected, 3);
                assert_eq!(got, 2);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }
}
