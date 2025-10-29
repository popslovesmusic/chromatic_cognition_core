//! Query embedding caching for dream pool retrieval
//!
//! This module implements an LRU cache for query embeddings to avoid
//! redundant recomputation when the same query is issued multiple times.
//!
//! Caches the expensive `encode_query` operation which converts a QuerySignature
//! to a dense embedding vector.

use lru::LruCache;
use std::num::NonZeroUsize;

/// RGB query key with fixed precision for hashing
///
/// Converts f32 RGB values to u32 with 3 decimal places of precision.
/// This allows similar queries to hit the cache even with slight floating-point variations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryKey([u32; 3]);

impl QueryKey {
    /// Create a query key from RGB values
    ///
    /// Precision: 0.001 (3 decimal places)
    /// Example: [0.856, 0.234, 0.112] → [856, 234, 112]
    pub fn from_rgb(rgb: &[f32; 3]) -> Self {
        Self([
            (rgb[0] * 1000.0).round() as u32,
            (rgb[1] * 1000.0).round() as u32,
            (rgb[2] * 1000.0).round() as u32,
        ])
    }

    /// Get the RGB values as u32 array
    pub fn as_array(&self) -> [u32; 3] {
        self.0
    }
}

/// LRU cache for query embeddings
///
/// Caches the expensive `encode_query` operation to avoid
/// redundant computation when the same query is issued multiple times.
///
/// # Memory
///
/// Each cache entry stores:
/// - Key: 12 bytes (3 × u32)
/// - Value: D × 4 bytes (embedding vector, D=64 default = 256 bytes)
/// Total: ~268 bytes per entry
///
/// Default capacity of 128 entries = ~34 KB
pub struct QueryCache {
    cache: LruCache<QueryKey, Vec<f32>>,
    hits: u64,
    misses: u64,
}

impl QueryCache {
    /// Create a new query cache with given capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of cached queries (default: 128)
    ///
    /// # Memory
    ///
    /// capacity × 312 bytes ≈ memory usage
    /// - 128 entries ≈ 40 KB
    /// - 256 entries ≈ 80 KB
    /// - 512 entries ≈ 160 KB
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(capacity).unwrap()),
            hits: 0,
            misses: 0,
        }
    }

    /// Get a cached query embedding or compute it
    ///
    /// # Arguments
    ///
    /// * `query` - RGB query vector [R, G, B] in [0, 1]
    /// * `compute` - Function to compute embedding if not cached
    ///
    /// # Returns
    ///
    /// The query embedding vector (cached or freshly computed)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let embedding = cache.get_or_compute(&[1.0, 0.0, 0.0], |q| {
    ///     let query_sig = QuerySignature::from_chroma(*q);
    ///     mapper.encode_query(&query_sig, None)
    /// });
    /// ```
    pub fn get_or_compute<F>(&mut self, query: &[f32; 3], compute: F) -> Vec<f32>
    where
        F: FnOnce(&[f32; 3]) -> Vec<f32>,
    {
        let key = QueryKey::from_rgb(query);

        if let Some(cached) = self.cache.get(&key) {
            self.hits += 1;
            return cached.clone();
        }

        // Cache miss: compute embedding
        self.misses += 1;
        let embedding = compute(query);
        self.cache.put(key, embedding.clone());
        embedding
    }

    /// Get the number of cache hits
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Get the number of cache misses
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Get the cache hit rate
    ///
    /// Returns value in [0.0, 1.0] representing hit/(hit+miss)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Clear the cache and reset statistics
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get the current number of cached entries
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.len() == 0
    }

    /// Get the maximum capacity of the cache
    pub fn capacity(&self) -> usize {
        self.cache.cap().get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dream::embedding::{EmbeddingMapper, QuerySignature};

    fn create_test_mapper() -> EmbeddingMapper {
        EmbeddingMapper::new(64)
    }

    fn encode_rgb(mapper: &EmbeddingMapper, rgb: &[f32; 3]) -> Vec<f32> {
        let query_sig = QuerySignature::from_chroma(*rgb);
        mapper.encode_query(&query_sig, None)
    }

    #[test]
    fn test_query_key_creation() {
        let key1 = QueryKey::from_rgb(&[1.0, 0.0, 0.0]);
        let key2 = QueryKey::from_rgb(&[1.0, 0.0, 0.0]);
        assert_eq!(key1, key2);

        let key3 = QueryKey::from_rgb(&[0.9995, 0.0, 0.0]);
        assert_eq!(key1, key3); // Within precision tolerance (rounds to 1000)

        let key4 = QueryKey::from_rgb(&[0.5, 0.0, 0.0]);
        assert_ne!(key1, key4);
    }

    #[test]
    fn test_cache_hit() {
        let mut cache = QueryCache::new(16);
        let mapper = create_test_mapper();

        let query = [1.0, 0.0, 0.0];

        // First access: cache miss
        let embed1 = cache.get_or_compute(&query, |q| encode_rgb(&mapper, q));
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 1);

        // Second access: cache hit
        let embed2 = cache.get_or_compute(&query, |q| encode_rgb(&mapper, q));
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);

        // Embeddings should be identical
        assert_eq!(embed1.len(), embed2.len());
        assert_eq!(embed1, embed2);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = QueryCache::new(2); // Small capacity
        let mapper = create_test_mapper();

        // Add 3 entries (will evict first)
        cache.get_or_compute(&[1.0, 0.0, 0.0], |q| encode_rgb(&mapper, q));
        cache.get_or_compute(&[0.0, 1.0, 0.0], |q| encode_rgb(&mapper, q));
        cache.get_or_compute(&[0.0, 0.0, 1.0], |q| encode_rgb(&mapper, q));

        assert_eq!(cache.len(), 2); // LRU evicted first entry

        // First query should be cache miss (evicted)
        let _ = cache.get_or_compute(&[1.0, 0.0, 0.0], |q| encode_rgb(&mapper, q));
        assert_eq!(cache.misses(), 4); // 3 initial + 1 re-miss
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = QueryCache::new(16);
        let mapper = create_test_mapper();

        assert_eq!(cache.hit_rate(), 0.0); // No queries yet

        // 1 miss
        cache.get_or_compute(&[1.0, 0.0, 0.0], |q| encode_rgb(&mapper, q));
        assert_eq!(cache.hit_rate(), 0.0); // 0/1

        // 1 hit
        cache.get_or_compute(&[1.0, 0.0, 0.0], |q| encode_rgb(&mapper, q));
        assert_eq!(cache.hit_rate(), 0.5); // 1/2

        // 2 more hits
        cache.get_or_compute(&[1.0, 0.0, 0.0], |q| encode_rgb(&mapper, q));
        cache.get_or_compute(&[1.0, 0.0, 0.0], |q| encode_rgb(&mapper, q));
        assert_eq!(cache.hit_rate(), 0.75); // 3/4
    }

    #[test]
    fn test_clear() {
        let mut cache = QueryCache::new(16);
        let mapper = create_test_mapper();

        cache.get_or_compute(&[1.0, 0.0, 0.0], |q| encode_rgb(&mapper, q));
        cache.get_or_compute(&[1.0, 0.0, 0.0], |q| encode_rgb(&mapper, q));

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.hits(), 1);

        cache.clear();

        assert_eq!(cache.len(), 0);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn test_precision_tolerance() {
        let mut cache = QueryCache::new(16);
        let mapper = create_test_mapper();

        // First query
        let embed1 = cache.get_or_compute(&[0.856, 0.234, 0.112], |q| {
            encode_rgb(&mapper, q)
        });

        // Second query with minor floating-point variation (within 0.001)
        let embed2 = cache.get_or_compute(&[0.8564, 0.2336, 0.1124], |q| {
            encode_rgb(&mapper, q)
        });

        // Should be cache hit (same after rounding)
        assert_eq!(cache.hits(), 1);
        assert_eq!(embed1, embed2);
    }
}
