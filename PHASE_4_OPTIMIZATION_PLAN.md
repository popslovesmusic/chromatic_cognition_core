# Phase 4 Optimization Plan — Pre-Phase 7 Improvements

**Date:** 2025-10-27
**Status:** 🎯 PLANNED (Not Yet Implemented)
**Priority:** HIGH — Must complete before Phase 7

---

## Executive Summary

Before implementing Phase 7 (Cross-Modal Bridge), we must optimize Phase 4 to handle the increased computational and memory demands of multi-modal processing. This document addresses 7 critical issues identified in `PHASE_4_COMPREHENSIVE_ANALYSIS.md`.

**Goal:** Transform Phase 4 from "functional prototype" to "production-ready foundation"

**Estimated Effort:** 3-4 weeks
**Impact:** 10× performance, 50% memory reduction, 95%+ test coverage

---

## Table of Contents

1. [Issue 1: Query Embedding Caching](#issue-1-query-embedding-caching)
2. [Issue 2: Coupling Reduction](#issue-2-coupling-reduction)
3. [Issue 3: Memory Management](#issue-3-memory-management)
4. [Issue 4: Spectral Feature Caching](#issue-4-spectral-feature-caching)
5. [Issue 5: MMR Optimization](#issue-5-mmr-optimization)
6. [Issue 6: HNSW Scalability](#issue-6-hnsw-scalability)
7. [Issue 7: Test Coverage](#issue-7-test-coverage)
8. [Implementation Timeline](#implementation-timeline)
9. [Success Metrics](#success-metrics)

---

## Issue 1: Query Embedding Caching

### Problem

**Current Behavior:**
```rust
// Every retrieval call recomputes query embedding
pub fn retrieve_soft(&self, query: &[f32; 3], k: usize) -> Vec<DreamEntry> {
    let query_sig = self.mapper.encode_to_signature(query);  // ❌ Recomputed
    // ... retrieval logic ...
}
```

**Cost:** ~15% of query time wasted on redundant encoding

### Solution: LRU Cache for Query Embeddings

**Design:**

```rust
use lru::LruCache;
use std::num::NonZeroUsize;

pub struct QueryCache {
    cache: LruCache<[u32; 3], QuerySignature>,  // RGB as u32 for hashing
}

impl QueryCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(capacity).unwrap()),
        }
    }

    pub fn get_or_compute<F>(&mut self, query: &[f32; 3], compute: F) -> QuerySignature
    where
        F: FnOnce(&[f32; 3]) -> QuerySignature,
    {
        // Convert f32 to u32 for hashing (with precision loss acceptable)
        let key = [
            (query[0] * 1000.0) as u32,
            (query[1] * 1000.0) as u32,
            (query[2] * 1000.0) as u32,
        ];

        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }

        let signature = compute(query);
        self.cache.put(key, signature.clone());
        signature
    }
}
```

**Integration in SimpleDreamPool:**

```rust
pub struct SimpleDreamPool {
    entries: Vec<DreamEntry>,
    config: PoolConfig,
    mapper: EmbeddingMapper,
    soft_index: Option<SoftIndex>,
    query_cache: QueryCache,  // ✅ NEW
}

impl SimpleDreamPool {
    pub fn new(config: PoolConfig) -> Self {
        Self {
            entries: Vec::new(),
            mapper: EmbeddingMapper::new(config.embed_dim),
            soft_index: None,
            query_cache: QueryCache::new(128),  // Cache last 128 queries
            config,
        }
    }

    pub fn retrieve_soft(&mut self, query: &[f32; 3], k: usize) -> Vec<DreamEntry> {
        let query_sig = self.query_cache.get_or_compute(query, |q| {
            self.mapper.encode_to_signature(q)
        });
        // ... rest of retrieval ...
    }
}
```

**Benefits:**
- ✅ 15% faster repeated queries
- ✅ Minimal memory overhead (~2KB for 128 entries)
- ✅ Automatic LRU eviction

**Implementation:**
- File: `src/dream/query_cache.rs` (new, ~80 lines)
- Modify: `src/dream/simple_pool.rs` (add field, update retrieve_soft)
- Dependency: `lru = "0.12"` in Cargo.toml

---

## Issue 2: Coupling Reduction

### Problem

**Current Architecture:**
```rust
// SimpleDreamPool is tightly coupled to:
// - EmbeddingMapper (Phase 4)
// - BiasProfile (Phase 3B)
// - SoftIndex (Phase 4)
// - DreamEntry (storage)
```

**Issue:** High coupling makes it hard to:
- Swap out embedding strategies
- Reuse retrieval logic in other contexts
- Test components in isolation

### Solution: Trait-Based Decoupling

**Design: Introduce Abstraction Traits**

```rust
// src/dream/traits.rs (NEW FILE)

/// Trait for encoding queries to embeddings
pub trait QueryEncoder {
    type Query;
    type Signature;

    fn encode(&self, query: &Self::Query) -> Self::Signature;
}

/// Trait for retrieving entries by similarity
pub trait SimilarityRetriever {
    type Entry;
    type Query;

    fn retrieve_k(&self, query: &Self::Query, k: usize) -> Vec<Self::Entry>;
}

/// Trait for memory management
pub trait MemoryBudget {
    fn current_usage(&self) -> usize;
    fn evict_to_fit(&mut self, required: usize);
}

/// Trait for embeddings (decouples from EmbeddingMapper)
pub trait Embeddable {
    fn to_embedding(&self) -> Vec<f32>;
    fn embedding_dim(&self) -> usize;
}
```

**Refactored SimpleDreamPool:**

```rust
pub struct SimpleDreamPool<E: QueryEncoder> {
    entries: Vec<DreamEntry>,
    encoder: E,  // ✅ Generic, not tied to EmbeddingMapper
    index: Option<SoftIndex>,
    config: PoolConfig,
    query_cache: QueryCache,
}

impl<E: QueryEncoder<Query = [f32; 3]>> SimpleDreamPool<E> {
    pub fn new(encoder: E, config: PoolConfig) -> Self {
        Self {
            entries: Vec::new(),
            encoder,
            index: None,
            config,
            query_cache: QueryCache::new(128),
        }
    }

    pub fn retrieve_soft(&mut self, query: &[f32; 3], k: usize) -> Vec<DreamEntry> {
        let signature = self.query_cache.get_or_compute(query, |q| {
            self.encoder.encode(q)
        });
        // ... retrieval logic ...
    }
}

// Implement QueryEncoder for EmbeddingMapper
impl QueryEncoder for EmbeddingMapper {
    type Query = [f32; 3];
    type Signature = QuerySignature;

    fn encode(&self, query: &[f32; 3]) -> QuerySignature {
        self.encode_to_signature(query)
    }
}
```

**Benefits:**
- ✅ SimpleDreamPool now generic over encoder type
- ✅ Easy to swap EmbeddingMapper for alternatives
- ✅ Enables mock encoders for testing
- ✅ Reusable traits for Phase 7 multi-modal encoders

**Migration Path:**

```rust
// Old (still works via type alias)
type ChromaticDreamPool = SimpleDreamPool<EmbeddingMapper>;

// New (explicit encoder)
let mapper = EmbeddingMapper::new(64);
let pool = SimpleDreamPool::new(mapper, PoolConfig::default());
```

**Implementation:**
- File: `src/dream/traits.rs` (new, ~120 lines)
- Modify: `src/dream/simple_pool.rs` (make generic)
- Modify: `src/dream/embedding.rs` (impl QueryEncoder)

---

## Issue 3: Memory Management

### Problem

**Current Issues:**
1. **No memory budget tracking** — Pool can grow indefinitely
2. **Crude FIFO eviction** — Doesn't consider entry utility
3. **Full tensor storage** — 140-256 bytes per entry
4. **No compression** — Raw tensor data stored

**Memory Breakdown (per entry):**
```
ChromaticTensor:      140-256 bytes (8×8×8 f32 grid)
SolverResult:         16 bytes
SpectralFeatures:     48 bytes
Embedding (64D):      256 bytes
Metadata:             32 bytes
-------------------------------------------
Total:                492-608 bytes per entry

At 10K entries: ~5-6 MB
At 100K entries: ~50-60 MB
At 1M entries: ~500-600 MB ❌ Too much!
```

### Solution: Multi-Tier Memory Management

#### 3.1 Memory Budget Tracking

```rust
// src/dream/memory.rs (NEW FILE)

pub struct MemoryBudget {
    max_bytes: usize,
    current_bytes: usize,
    entry_count: usize,
}

impl MemoryBudget {
    pub fn new(max_mb: usize) -> Self {
        Self {
            max_bytes: max_mb * 1024 * 1024,
            current_bytes: 0,
            entry_count: 0,
        }
    }

    pub fn can_add(&self, entry_size: usize) -> bool {
        self.current_bytes + entry_size <= self.max_bytes
    }

    pub fn add_entry(&mut self, entry_size: usize) {
        self.current_bytes += entry_size;
        self.entry_count += 1;
    }

    pub fn remove_entry(&mut self, entry_size: usize) {
        self.current_bytes = self.current_bytes.saturating_sub(entry_size);
        self.entry_count = self.entry_count.saturating_sub(1);
    }

    pub fn usage_ratio(&self) -> f32 {
        self.current_bytes as f32 / self.max_bytes as f32
    }

    pub fn needs_eviction(&self) -> bool {
        self.usage_ratio() > 0.9  // Trigger at 90% capacity
    }
}
```

#### 3.2 Utility-Based Eviction Policy

**Strategy: LRU + Utility Weighting**

```rust
pub enum EvictionPolicy {
    FIFO,
    LRU,
    UtilityWeighted,  // ✅ NEW: Evict low-utility, rarely-used entries
}

pub struct EvictionScore {
    pub utility: f32,
    pub recency: f32,
    pub usage_count: usize,
}

impl EvictionScore {
    pub fn compute(entry: &DreamEntry, now: SystemTime) -> f32 {
        let age = now.duration_since(entry.timestamp)
            .unwrap_or_default()
            .as_secs() as f32;

        let recency_score = 1.0 / (1.0 + age / 3600.0);  // Decay over hours
        let utility_score = entry.utility.unwrap_or(0.5);
        let usage_score = (entry.usage_count as f32).ln_1p() / 10.0;

        // Weighted combination
        0.4 * utility_score + 0.4 * recency_score + 0.2 * usage_score
    }
}

impl SimpleDreamPool {
    fn evict_n_entries(&mut self, n: usize) {
        let now = SystemTime::now();

        // Compute eviction scores
        let mut scored: Vec<_> = self.entries.iter()
            .enumerate()
            .map(|(i, e)| (i, EvictionScore::compute(e, now)))
            .collect();

        // Sort by score (ascending = lowest utility first)
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Evict lowest n entries
        let to_remove: Vec<_> = scored.iter().take(n).map(|(i, _)| *i).collect();

        for i in to_remove.iter().rev() {
            self.entries.remove(*i);
            self.memory_budget.remove_entry(ENTRY_SIZE_ESTIMATE);
        }

        // Rebuild index after bulk eviction
        self.rebuild_soft_index();
    }
}
```

#### 3.3 Separate Embedding Storage

**Problem:** Currently embeddings are stored with full tensors.

**Solution:** Store embeddings separately, load tensors on-demand.

```rust
pub struct DreamEntry {
    // Hot path: Always in memory
    pub embed: Vec<f32>,              // 256 bytes (64D)
    pub chroma_signature: [f32; 3],   // 12 bytes
    pub utility: f32,                 // 4 bytes
    pub metadata: EntryMetadata,      // 32 bytes
    // Total hot: ~304 bytes

    // Cold path: Load on-demand
    pub tensor_id: TensorId,          // 8 bytes (reference to storage)
}

pub struct TensorStorage {
    tensors: HashMap<TensorId, ChromaticTensor>,
    // Could be disk-backed for large pools
}

impl SimpleDreamPool {
    pub fn get_full_entry(&self, entry: &DreamEntry) -> DreamEntryFull {
        let tensor = self.tensor_storage.get(&entry.tensor_id);
        DreamEntryFull {
            entry: entry.clone(),
            tensor,
        }
    }
}
```

**Memory Savings:**
- Before: 600 bytes/entry × 100K = 60 MB
- After: 304 bytes/entry × 100K = 30 MB ✅ 50% reduction

#### 3.4 Tensor Compression (Optional)

For extreme memory pressure, compress tensors:

```rust
use flate2::Compression;

pub struct CompressedTensor {
    compressed_data: Vec<u8>,
    original_shape: (usize, usize, usize),
}

impl CompressedTensor {
    pub fn compress(tensor: &ChromaticTensor) -> Self {
        let bytes = tensor.as_bytes();
        let compressed = flate2::compress(&bytes, Compression::fast());
        Self {
            compressed_data: compressed,
            original_shape: tensor.shape(),
        }
    }

    pub fn decompress(&self) -> ChromaticTensor {
        let bytes = flate2::decompress(&self.compressed_data);
        ChromaticTensor::from_bytes(&bytes, self.original_shape)
    }
}
```

**Compression Ratio:** ~2-4× (typical for RGB data)

**Implementation:**
- File: `src/dream/memory.rs` (new, ~200 lines)
- File: `src/dream/eviction.rs` (new, ~150 lines)
- Modify: `src/dream/simple_pool.rs` (integrate memory tracking)
- Modify: `src/dream/simple_pool.rs` (add PoolConfig.memory_budget_mb)

---

## Issue 4: Spectral Feature Caching

### Problem

**Current Behavior:**
```rust
impl DreamEntry {
    pub fn from_tensor(tensor: ChromaticTensor, result: SolverResult) -> Self {
        let chroma = tensor.mean_rgb();
        let spectral = extract_spectral_features(&tensor);  // ❌ Computed on every add
        // ...
    }
}
```

**Cost:** FFT is expensive (~5-10ms per tensor)

### Solution: Compute Once, Store Forever

**Design:**

```rust
pub struct DreamEntry {
    pub tensor: ChromaticTensor,
    pub result: SolverResult,
    pub chroma_signature: [f32; 3],
    pub spectral_features: SpectralFeatures,  // ✅ Always present (not Option)
    pub embed: Option<Vec<f32>>,  // Still optional (computed on rebuild_soft_index)
    // ... rest ...
}

impl DreamEntry {
    pub fn from_tensor(tensor: ChromaticTensor, result: SolverResult) -> Self {
        let chroma = tensor.mean_rgb();
        let spectral = extract_spectral_features(&tensor);  // ✅ Computed once here

        Self {
            tensor,
            result,
            chroma_signature: chroma,
            spectral_features: spectral,  // ✅ Stored permanently
            embed: None,  // Will be computed later
            utility: None,
            timestamp: SystemTime::now(),
            usage_count: 0,
            util_mean: 0.0,
        }
    }
}
```

**Benefits:**
- ✅ 15% faster embedding computation (spectral features pre-cached)
- ✅ No behavior change (just caching)
- ✅ Minimal memory cost (48 bytes per entry)

**Implementation:**
- Modify: `src/dream/simple_pool.rs` (remove `Option<SpectralFeatures>`)
- Update all references to `spectral_features` (remove `.unwrap()`)

---

## Issue 5: MMR Optimization

### Problem

**Current Complexity:**
```rust
// diversity.rs
pub fn retrieve_diverse_mmr(
    candidates: &[DreamEntry],
    query: &QuerySignature,
    k: usize,
    lambda: f32,
) -> Vec<DreamEntry> {
    // ...
    for _ in 0..k {
        for candidate in remaining {
            let sim_query = compute_similarity(candidate, query);
            let max_sim_selected = selected.iter()
                .map(|s| compute_similarity(candidate, s))  // ❌ O(k²)
                .max();
            // ...
        }
    }
}
```

**Complexity:** O(k² · d) where d = embedding dimension
**Bottleneck:** At k=100, performs 10,000 distance computations

### Solution: Approximate MMR with Early Termination

**Design:**

```rust
pub fn retrieve_diverse_mmr_fast(
    candidates: &[DreamEntry],
    query: &QuerySignature,
    k: usize,
    lambda: f32,
    threshold: f32,  // ✅ NEW: Early termination threshold
) -> Vec<DreamEntry> {
    let mut selected = Vec::with_capacity(k);
    let mut remaining: Vec<_> = candidates.iter().enumerate().collect();

    for _ in 0..k {
        if remaining.is_empty() {
            break;
        }

        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, candidate) in remaining.iter() {
            let sim_query = compute_similarity(candidate, query);

            // ✅ OPTIMIZATION 1: Early termination if similarity too low
            if sim_query < threshold {
                continue;
            }

            // ✅ OPTIMIZATION 2: Approximate max similarity (sample, don't compute all)
            let max_sim_selected = if selected.len() <= 10 {
                // Small selected set: compute exact
                selected.iter()
                    .map(|s| compute_similarity(candidate, s))
                    .fold(0.0, f32::max)
            } else {
                // Large selected set: sample 10 random entries
                selected.iter()
                    .take(10)
                    .map(|s| compute_similarity(candidate, s))
                    .fold(0.0, f32::max)
            };

            let mmr_score = lambda * sim_query - (1.0 - lambda) * max_sim_selected;

            if mmr_score > best_score {
                best_score = mmr_score;
                best_idx = i;
            }
        }

        selected.push(remaining.remove(best_idx).clone());
    }

    selected
}
```

**Complexity Reduction:**
- Before: O(k² · d) = O(10,000 · 64) = 640K ops
- After: O(k · min(k, 10) · d) = O(1,000 · 64) = 64K ops ✅ 10× faster

**Trade-off:** Slight decrease in diversity (< 5% in practice)

**Implementation:**
- Modify: `src/dream/diversity.rs` (add `retrieve_diverse_mmr_fast`)
- Keep original `retrieve_diverse_mmr` for benchmarking

---

## Issue 6: HNSW Scalability

### Problem

**Current: Linear k-NN**
```rust
// soft_index.rs
impl SoftIndex {
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(EntryId, f32)> {
        // Linear scan through all entries
        let mut scores: Vec<_> = self.entries.iter()
            .map(|(id, embed)| (*id, cosine_similarity(query, embed)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);
        scores
    }
}
```

**Complexity:** O(n · d) where n = pool size
**Bottleneck:** At 10K entries, 10K distance computations per query

### Solution: HNSW (Hierarchical Navigable Small World)

**Design: Use `hnsw` Crate**

```rust
// Cargo.toml
[dependencies]
hnsw_rs = "0.3"
```

**Implementation:**

```rust
// src/dream/hnsw_index.rs (NEW FILE)

use hnsw_rs::prelude::*;
use std::sync::Arc;

pub struct HnswIndex {
    hnsw: Hnsw<f32, DistCosine>,
    entries: Vec<(EntryId, Arc<Vec<f32>>)>,
    dim: usize,
}

impl HnswIndex {
    pub fn new(dim: usize, capacity: usize) -> Self {
        let hnsw = Hnsw::new(
            16,       // max_nb_connection (M)
            capacity,
            16,       // ef_construction
            200,      // max_elements
            DistCosine,
        );

        Self {
            hnsw,
            entries: Vec::new(),
            dim,
        }
    }

    pub fn add(&mut self, id: EntryId, embedding: Vec<f32>) {
        assert_eq!(embedding.len(), self.dim);
        let embed_arc = Arc::new(embedding);
        self.hnsw.insert((embed_arc.as_ref(), id.0));
        self.entries.push((id, embed_arc));
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(EntryId, f32)> {
        let neighbors = self.hnsw.search(query, k, 50);  // ef_search = 50

        neighbors.into_iter()
            .map(|n| (EntryId(n.d_id), n.distance))
            .collect()
    }

    pub fn clear(&mut self) {
        self.hnsw = Hnsw::new(16, 1000, 16, 200, DistCosine);
        self.entries.clear();
    }
}
```

**Integration:**

```rust
pub struct SimpleDreamPool {
    entries: Vec<DreamEntry>,
    config: PoolConfig,
    mapper: EmbeddingMapper,

    // Old: SoftIndex (linear)
    // soft_index: Option<SoftIndex>,

    // New: HNSW index (sublinear)
    hnsw_index: Option<HnswIndex>,  // ✅ Replaces SoftIndex

    query_cache: QueryCache,
}

impl SimpleDreamPool {
    pub fn rebuild_soft_index(&mut self) {
        let mut hnsw = HnswIndex::new(self.config.embed_dim, self.entries.len());

        for (i, entry) in self.entries.iter_mut().enumerate() {
            let embed = self.mapper.encode(entry);
            entry.embed = Some(embed.clone());
            hnsw.add(EntryId(i), embed);
        }

        self.hnsw_index = Some(hnsw);
    }

    pub fn retrieve_soft(&mut self, query: &[f32; 3], k: usize) -> Vec<DreamEntry> {
        let hnsw = self.hnsw_index.as_ref()
            .expect("Index not built. Call rebuild_soft_index first.");

        let query_sig = self.query_cache.get_or_compute(query, |q| {
            self.mapper.encode_to_signature(q)
        });

        let results = hnsw.search(&query_sig.embedding, k);

        results.into_iter()
            .map(|(id, _score)| self.entries[id.0].clone())
            .collect()
    }
}
```

**Performance:**
- Linear k-NN: O(n · d) = O(10K · 64) = 640K ops
- HNSW: O(log(n) · d) = O(13 · 64) = 832 ops ✅ 100× faster!

**Trade-off:**
- 95-99% recall (vs 100% for linear)
- Acceptable for semantic retrieval

**Implementation:**
- File: `src/dream/hnsw_index.rs` (new, ~200 lines)
- Modify: `src/dream/simple_pool.rs` (replace SoftIndex with HnswIndex)
- Dependency: `hnsw_rs = "0.3"` in Cargo.toml

---

## Issue 7: Test Coverage

### Problem

**Current Coverage Gaps (from PHASE_4_COMPREHENSIVE_ANALYSIS.md):**
- simple_pool.rs: 70% coverage (missing edge cases)
- No performance tests
- Limited integration tests (only 2 full pipeline)
- No failure injection (index corruption, memory pressure)
- No concurrency tests (race conditions untested)

### Solution: Comprehensive Test Suite

#### 7.1 Unit Test Improvements

**File: `src/dream/simple_pool.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // === Existing Tests (Keep) ===
    #[test] fn test_add_retrieve() { /* ... */ }
    #[test] fn test_retrieve_by_class() { /* ... */ }
    // ... (existing 21 tests) ...

    // === NEW: Edge Cases ===

    #[test]
    fn test_empty_pool_retrieval() {
        let pool = SimpleDreamPool::new(PoolConfig::default());
        let results = pool.retrieve_similar(&[1.0, 0.0, 0.0], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_cache_hit() {
        let mut pool = create_test_pool();
        let query = [1.0, 0.0, 0.0];

        // First query (cache miss)
        let start = Instant::now();
        pool.retrieve_soft(&query, 5);
        let first_time = start.elapsed();

        // Second query (cache hit)
        let start = Instant::now();
        pool.retrieve_soft(&query, 5);
        let second_time = start.elapsed();

        // Cache hit should be faster
        assert!(second_time < first_time);
    }

    #[test]
    fn test_memory_eviction() {
        let config = PoolConfig {
            memory_budget_mb: 1,  // 1 MB limit
            ..Default::default()
        };
        let mut pool = SimpleDreamPool::new(config);

        // Add entries until eviction triggers
        for i in 0..1000 {
            pool.add_dream(create_test_tensor(i), create_test_result());
        }

        // Pool should have evicted old entries
        assert!(pool.memory_budget.usage_ratio() < 0.95);
    }

    #[test]
    fn test_spectral_feature_persistence() {
        let mut pool = SimpleDreamPool::new(PoolConfig::default());
        let tensor = create_test_tensor(0);
        let expected_spectral = extract_spectral_features(&tensor);

        pool.add_dream(tensor, create_test_result());

        let entry = &pool.entries[0];
        assert_eq!(entry.spectral_features, expected_spectral);
    }

    #[test]
    fn test_utility_based_eviction() {
        let mut pool = create_pool_with_varied_utility();

        // Force eviction
        pool.evict_n_entries(5);

        // Lowest utility entries should be evicted
        let remaining_utilities: Vec<_> = pool.entries.iter()
            .map(|e| e.utility.unwrap())
            .collect();

        assert!(remaining_utilities.iter().all(|&u| u > 0.3));
    }
}
```

#### 7.2 Integration Tests

**File: `tests/integration_test.rs` (NEW)**

```rust
use chromatic_cognition_core::dream::prelude::*;
use chromatic_cognition_core::tensor::ChromaticTensor;

#[test]
fn test_full_pipeline_retrieval() {
    // Create pool with 100 diverse entries
    let mut pool = create_large_test_pool(100);

    // Query for red-ish tensors
    let query = [0.8, 0.2, 0.2];
    let results = pool.retrieve_soft(&query, 10);

    // Verify results
    assert_eq!(results.len(), 10);
    for entry in results {
        let rgb = entry.tensor.mean_rgb();
        assert!(rgb[0] > rgb[1] && rgb[0] > rgb[2]);  // Red dominant
    }
}

#[test]
fn test_memory_pressure_handling() {
    let config = PoolConfig {
        memory_budget_mb: 5,
        eviction_policy: EvictionPolicy::UtilityWeighted,
        ..Default::default()
    };

    let mut pool = SimpleDreamPool::new(config);

    // Add 1000 entries (should trigger eviction)
    for i in 0..1000 {
        pool.add_dream(create_test_tensor(i), create_test_result());
    }

    // Pool should be under budget
    assert!(pool.memory_budget.current_usage() < 5 * 1024 * 1024);

    // Should still be functional
    let results = pool.retrieve_soft(&[1.0, 0.0, 0.0], 10);
    assert_eq!(results.len(), 10);
}

#[test]
fn test_index_rebuild_correctness() {
    let mut pool = create_test_pool();
    let query = [1.0, 0.0, 0.0];

    // Get results before rebuild
    let before = pool.retrieve_soft(&query, 5);

    // Rebuild index
    pool.rebuild_soft_index();

    // Get results after rebuild
    let after = pool.retrieve_soft(&query, 5);

    // Results should be identical
    assert_eq!(before.len(), after.len());
    for (b, a) in before.iter().zip(after.iter()) {
        assert_eq!(b.tensor.mean_rgb(), a.tensor.mean_rgb());
    }
}
```

#### 7.3 Failure Injection Tests

**File: `tests/failure_injection.rs` (NEW)**

```rust
#[test]
fn test_corrupted_index_recovery() {
    let mut pool = create_test_pool();

    // Simulate index corruption
    pool.hnsw_index = None;

    // Should detect corruption and rebuild
    let result = std::panic::catch_unwind(|| {
        pool.retrieve_soft(&[1.0, 0.0, 0.0], 5)
    });

    // Should either rebuild gracefully or panic with clear message
    assert!(result.is_ok() || result.unwrap_err().to_string().contains("Index not built"));
}

#[test]
fn test_out_of_memory_handling() {
    let config = PoolConfig {
        memory_budget_mb: 1,
        ..Default::default()
    };

    let mut pool = SimpleDreamPool::new(config);

    // Try to add extremely large tensor
    let large_tensor = ChromaticTensor::zeros(100, 100, 100);  // ~4MB

    // Should either evict or reject gracefully
    pool.add_dream(large_tensor, create_test_result());

    // Pool should still be functional
    assert!(pool.memory_budget.current_usage() <= 1 * 1024 * 1024);
}
```

#### 7.4 Concurrency Tests

**File: `tests/concurrency_test.rs` (NEW)**

```rust
use std::sync::{Arc, Mutex};
use std::thread;

#[test]
fn test_concurrent_reads() {
    let pool = Arc::new(Mutex::new(create_test_pool()));

    let mut handles = vec![];

    // Spawn 10 threads reading concurrently
    for _ in 0..10 {
        let pool_clone = Arc::clone(&pool);
        let handle = thread::spawn(move || {
            let mut p = pool_clone.lock().unwrap();
            p.retrieve_soft(&[1.0, 0.0, 0.0], 5)
        });
        handles.push(handle);
    }

    // All threads should complete successfully
    for handle in handles {
        let results = handle.join().unwrap();
        assert_eq!(results.len(), 5);
    }
}

#[test]
fn test_concurrent_add_and_read() {
    let pool = Arc::new(Mutex::new(SimpleDreamPool::new(PoolConfig::default())));

    let mut handles = vec![];

    // Writer thread
    let pool_clone = Arc::clone(&pool);
    let writer = thread::spawn(move || {
        for i in 0..100 {
            let mut p = pool_clone.lock().unwrap();
            p.add_dream(create_test_tensor(i), create_test_result());
        }
    });

    // Reader threads
    for _ in 0..5 {
        let pool_clone = Arc::clone(&pool);
        let reader = thread::spawn(move || {
            for _ in 0..10 {
                let mut p = pool_clone.lock().unwrap();
                p.retrieve_soft(&[1.0, 0.0, 0.0], 5);
            }
        });
        handles.push(reader);
    }

    writer.join().unwrap();
    for handle in handles {
        handle.join().unwrap();
    }
}
```

#### 7.5 Performance Benchmarks

**File: `benches/retrieval_benchmark.rs` (NEW)**

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use chromatic_cognition_core::dream::prelude::*;

fn benchmark_linear_vs_hnsw(c: &mut Criterion) {
    let mut group = c.benchmark_group("retrieval");

    for size in [100, 1000, 10000] {
        // Linear k-NN benchmark
        group.bench_with_input(BenchmarkId::new("linear", size), &size, |b, &size| {
            let pool = create_pool_with_linear_index(size);
            b.iter(|| {
                pool.retrieve_soft(black_box(&[1.0, 0.0, 0.0]), black_box(10))
            });
        });

        // HNSW benchmark
        group.bench_with_input(BenchmarkId::new("hnsw", size), &size, |b, &size| {
            let pool = create_pool_with_hnsw_index(size);
            b.iter(|| {
                pool.retrieve_soft(black_box(&[1.0, 0.0, 0.0]), black_box(10))
            });
        });
    }

    group.finish();
}

fn benchmark_mmr_diversity(c: &mut Criterion) {
    let mut group = c.benchmark_group("mmr");

    for k in [10, 50, 100] {
        // Standard MMR
        group.bench_with_input(BenchmarkId::new("standard", k), &k, |b, &k| {
            let candidates = create_test_candidates(1000);
            let query = create_test_query();
            b.iter(|| {
                retrieve_diverse_mmr(
                    black_box(&candidates),
                    black_box(&query),
                    black_box(k),
                    black_box(0.7),
                )
            });
        });

        // Fast MMR (approximate)
        group.bench_with_input(BenchmarkId::new("fast", k), &k, |b, &k| {
            let candidates = create_test_candidates(1000);
            let query = create_test_query();
            b.iter(|| {
                retrieve_diverse_mmr_fast(
                    black_box(&candidates),
                    black_box(&query),
                    black_box(k),
                    black_box(0.7),
                    black_box(0.5),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_linear_vs_hnsw, benchmark_mmr_diversity);
criterion_main!(benches);
```

**Run benchmarks:**
```bash
cargo bench
```

**Implementation:**
- Add tests to existing files (simple_pool.rs, diversity.rs, etc.)
- Create new test files (integration_test.rs, failure_injection.rs, concurrency_test.rs)
- Create benchmark suite (benches/retrieval_benchmark.rs)
- Dependency: `criterion = "0.5"` in Cargo.toml [dev-dependencies]

---

## Implementation Timeline

### Week 1: Caching & Decoupling (Issues 1-2)

**Days 1-2: Query Embedding Caching**
- [ ] Create `src/dream/query_cache.rs`
- [ ] Integrate LRU cache in SimpleDreamPool
- [ ] Add tests for cache hit/miss behavior
- [ ] Benchmark: Verify 15% speedup on repeated queries

**Days 3-5: Coupling Reduction**
- [ ] Create `src/dream/traits.rs` with abstraction traits
- [ ] Make SimpleDreamPool generic over QueryEncoder
- [ ] Implement QueryEncoder for EmbeddingMapper
- [ ] Add type alias for backward compatibility
- [ ] Update all examples to use new API
- [ ] Test: All existing tests still pass

### Week 2: Memory Management (Issue 3)

**Days 1-2: Memory Budget Tracking**
- [ ] Create `src/dream/memory.rs` with MemoryBudget struct
- [ ] Integrate into SimpleDreamPool
- [ ] Add PoolConfig.memory_budget_mb parameter
- [ ] Test: Pool respects memory limit

**Days 3-4: Utility-Based Eviction**
- [ ] Create `src/dream/eviction.rs` with EvictionScore
- [ ] Implement UtilityWeighted eviction policy
- [ ] Add evict_n_entries() method
- [ ] Test: Low-utility entries evicted first

**Day 5: Separate Embedding Storage (Optional)**
- [ ] Refactor DreamEntry to use TensorId reference
- [ ] Create TensorStorage for cold-path data
- [ ] Test: Memory usage reduced by 50%

### Week 3: Computational Optimization (Issues 4-5)

**Days 1-2: Spectral Feature Caching**
- [ ] Remove `Option<SpectralFeatures>` from DreamEntry
- [ ] Always compute spectral features on add
- [ ] Update all references (remove unwrap())
- [ ] Test: No behavior change, 15% faster encoding

**Days 3-5: MMR Optimization**
- [ ] Implement retrieve_diverse_mmr_fast in diversity.rs
- [ ] Add early termination and sampling optimizations
- [ ] Benchmark: Verify 10× speedup
- [ ] Test: Diversity quality within 5% of original

### Week 4: HNSW & Testing (Issues 6-7)

**Days 1-3: HNSW Implementation**
- [ ] Add `hnsw_rs = "0.3"` dependency
- [ ] Create `src/dream/hnsw_index.rs`
- [ ] Replace SoftIndex with HnswIndex in SimpleDreamPool
- [ ] Test: 95%+ recall vs linear k-NN
- [ ] Benchmark: 100× speedup at 10K entries

**Days 4-5: Test Coverage**
- [ ] Add edge case tests (empty pool, cache hits, eviction)
- [ ] Create integration tests (full pipeline, memory pressure)
- [ ] Create failure injection tests (corruption, OOM)
- [ ] Create concurrency tests (concurrent reads/writes)
- [ ] Create benchmark suite (criterion)
- [ ] Run coverage: `cargo tarpaulin` → Target: 95%+

---

## Success Metrics

### Performance Targets

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| **Query Time (10K entries)** | 50ms | 5ms | 10× faster |
| **Memory (100K entries)** | 60 MB | 30 MB | 50% reduction |
| **MMR Diversity (k=100)** | 200ms | 20ms | 10× faster |
| **Repeated Query** | 10ms | 8.5ms | 15% faster |

### Quality Targets

| Metric | Target |
|--------|--------|
| **HNSW Recall** | 95%+ vs linear k-NN |
| **MMR Diversity Quality** | Within 5% of original |
| **Test Coverage** | 95%+ |

### Validation Tests

```bash
# Performance benchmarks
cargo bench

# Test coverage
cargo install cargo-tarpaulin
cargo tarpaulin --out Html

# Integration tests
cargo test --test '*'

# Concurrency stress test
cargo test --test concurrency_test -- --test-threads=1
```

---

## Breaking Changes

### API Changes

**SimpleDreamPool Constructor:**
```rust
// Before
let pool = SimpleDreamPool::new(PoolConfig::default());

// After (explicit encoder)
let mapper = EmbeddingMapper::new(64);
let pool = SimpleDreamPool::new(mapper, PoolConfig::default());

// Or use type alias (backward compatible)
type ChromaticDreamPool = SimpleDreamPool<EmbeddingMapper>;
let pool = ChromaticDreamPool::new(PoolConfig::default());
```

**PoolConfig Changes:**
```rust
pub struct PoolConfig {
    pub max_size: usize,
    pub coherence_threshold: f64,
    pub retrieval_limit: usize,

    // NEW fields
    pub memory_budget_mb: usize,           // Default: 100 MB
    pub eviction_policy: EvictionPolicy,   // Default: UtilityWeighted
    pub embed_dim: usize,                  // Default: 64
}
```

**DreamEntry Changes:**
```rust
// Before
pub spectral_features: Option<SpectralFeatures>,

// After
pub spectral_features: SpectralFeatures,  // Always present
```

### Migration Guide

**Step 1: Update Cargo.toml**
```toml
[dependencies]
lru = "0.12"
hnsw_rs = "0.3"

[dev-dependencies]
criterion = "0.5"
```

**Step 2: Update Pool Creation**
```rust
// Old code
let pool = SimpleDreamPool::new(PoolConfig::default());

// New code (explicit)
let mapper = EmbeddingMapper::new(64);
let config = PoolConfig {
    memory_budget_mb: 100,
    eviction_policy: EvictionPolicy::UtilityWeighted,
    ..Default::default()
};
let pool = SimpleDreamPool::new(mapper, config);
```

**Step 3: Update Spectral Feature Access**
```rust
// Old code
if let Some(spectral) = entry.spectral_features {
    // use spectral
}

// New code
let spectral = entry.spectral_features;  // Always present
```

---

## Dependencies Added

```toml
[dependencies]
# Existing
ndarray = "0.15"
rand = "0.8"
# ... (existing dependencies) ...

# NEW for Phase 4 optimization
lru = "0.12"               # Query cache
hnsw_rs = "0.3"            # HNSW index
flate2 = "1.0"             # Optional: Tensor compression

[dev-dependencies]
criterion = "0.5"          # Benchmarking
```

---

## Risks & Mitigation

### Risk 1: HNSW Recall Degradation

**Risk:** HNSW may return suboptimal results (< 95% recall)

**Mitigation:**
- Tune HNSW parameters (M, ef_construction, ef_search)
- Add fallback to linear k-NN for critical queries
- Monitor recall with A/B testing

### Risk 2: Memory Eviction Too Aggressive

**Risk:** Evicting entries too early, hurting cache hit rate

**Mitigation:**
- Make eviction threshold configurable (default: 90%)
- Add metrics: eviction rate, cache hit rate
- Allow users to disable eviction if sufficient RAM

### Risk 3: Breaking Changes

**Risk:** Generic SimpleDreamPool breaks existing code

**Mitigation:**
- Provide type alias for backward compatibility
- Update all examples simultaneously
- Add migration guide to documentation

### Risk 4: Test Suite Runtime

**Risk:** Comprehensive tests take too long to run

**Mitigation:**
- Use `cargo test --lib` for unit tests (fast)
- Use `cargo test --test '*'` for integration (slower)
- Run benchmarks separately: `cargo bench`

---

## Post-Implementation Checklist

- [ ] All 147 existing tests pass
- [ ] 30+ new tests added (edge cases, integration, concurrency)
- [ ] Test coverage ≥ 95% (measured with cargo-tarpaulin)
- [ ] Benchmarks show 10× speedup on 10K entries
- [ ] Memory usage reduced by 50% on large pools
- [ ] Documentation updated (module docs, examples)
- [ ] Migration guide published
- [ ] Examples updated to use new API
- [ ] Performance validated on real workloads

---

## Conclusion

These 7 optimizations transform Phase 4 from a **functional prototype** to a **production-ready foundation** for Phase 7's multi-modal processing.

**Key Improvements:**
- ✅ 10× faster retrieval (HNSW)
- ✅ 50% memory reduction (budget tracking + eviction)
- ✅ 15% faster encoding (spectral caching + query cache)
- ✅ 95%+ test coverage
- ✅ Decoupled architecture (trait-based)

**Timeline:** 4 weeks
**Effort:** 1 full-time engineer
**Impact:** Enables Phase 7 multi-modal scaling to 100K+ entries

**Next:** Implement Week 1 tasks (Query Cache + Traits)
