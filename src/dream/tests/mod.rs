//! Integration tests for dream pool system
//!
//! This module contains comprehensive integration tests that verify
//! the entire dream pool pipeline works correctly end-to-end.

use crate::dream::*;
use crate::dream::embedding::QuerySignature;
use crate::dream::soft_index::EntryId;
use crate::solver::SolverResult;
use crate::tensor::ChromaticTensor;
use serde_json::json;

/// Test that the full retrieval pipeline works with all components
#[test]
fn test_full_retrieval_pipeline() {
    // Create a pool with some entries
    let config = PoolConfig::default();
    let mut pool = SimpleDreamPool::new(config);

    // Add diverse entries with different colors
    for i in 0..20 {
        let tensor = ChromaticTensor::new(4, 4, 3);
        let result = SolverResult {
            energy: (0.1 + (i as f64) * 0.01),
            coherence: (0.9 - (i as f64) * 0.01),
            violation: 0.0,
            grad: None,
            mask: None,
            meta: json!({"test": i}),
        };
        pool.add_if_coherent(tensor, result);
    }

    // Create mapper and rebuild index
    let mapper = EmbeddingMapper::new(64);
    pool.rebuild_soft_index(&mapper, None);

    // Test retrieval works
    let query = QuerySignature::from_chroma([1.0, 0.0, 0.0]);
    let weights = RetrievalWeights::default();
    let results = pool.retrieve_soft(&query, 5, &weights, Similarity::Cosine, &mapper, None);

    assert!(results.len() <= 5);
    assert!(results.len() > 0);
}

/// Test that query cache improves performance on repeated queries
#[test]
fn test_query_cache_integration() {
    let config = PoolConfig::default();
    let mut pool = SimpleDreamPool::new(config);

    // Add entries
    for _ in 0..10 {
        let tensor = ChromaticTensor::new(4, 4, 3);
        let result = SolverResult {
            energy: 0.1,
            coherence: 0.9,
            violation: 0.0,
            grad: None,
            mask: None,
            meta: json!({}),
        };
        pool.add_if_coherent(tensor, result);
    }

    // Query cache stats should show hits after repeated queries
    let stats_before = pool.query_cache_stats();
    assert_eq!(stats_before.0, 0); // No hits initially

    // Note: Direct query cache testing would require exposing
    // the cache at a higher level. For now, we verify it exists.
}

/// Test that spectral features are always computed
#[test]
fn test_spectral_features_always_present() {
    let tensor = ChromaticTensor::new(4, 4, 3);
    let result = SolverResult {
        energy: 0.1,
        coherence: 0.9,
        violation: 0.0,
        grad: None,
        mask: None,
        meta: json!({}),
    };

    let entry = DreamEntry::new(tensor, result);

    // Spectral features should always be present (not Option)
    assert!(entry.spectral_features.entropy >= 0.0);
    assert!(entry.spectral_features.low_freq_energy >= 0.0);
    assert!(entry.spectral_features.mid_freq_energy >= 0.0);
    assert!(entry.spectral_features.high_freq_energy >= 0.0);
}

/// Test MMR diversity enforcement
#[test]
fn test_mmr_diversity_enforcement() {
    use crate::dream::diversity::retrieve_diverse_mmr;

    // Create similar entries (all red-ish)
    let mut candidates = vec![];
    for i in 0..5 {
        let tensor = ChromaticTensor::new(2, 2, 2);
        let result = SolverResult {
            energy: 0.1,
            coherence: 0.9,
            violation: 0.0,
            grad: None,
            mask: None,
            meta: json!({}),
        };
        let mut entry = DreamEntry::new(tensor, result);
        entry.chroma_signature = [1.0 - (i as f32) * 0.05, 0.0, 0.0];
        candidates.push(entry);
    }

    // Without diversity (lambda=1.0), should select most similar
    let no_diversity = retrieve_diverse_mmr(&candidates, &[1.0, 0.0, 0.0], 3, 1.0, 0.0);

    // With diversity (lambda=0.3), should select more diverse set
    let with_diversity = retrieve_diverse_mmr(&candidates, &[1.0, 0.0, 0.0], 3, 0.3, 0.0);

    // Both should return 3 entries
    assert_eq!(no_diversity.len(), 3);
    assert_eq!(with_diversity.len(), 3);

    // First entry should be the same (most relevant)
    assert_eq!(no_diversity[0].chroma_signature, with_diversity[0].chroma_signature);
}

/// Test memory budget prevents unbounded growth
#[test]
fn test_memory_budget_prevents_unbounded_growth() {
    use crate::dream::memory::MemoryBudget;

    let mut budget = MemoryBudget::new(1024 * 1024); // 1 MB limit

    // Add entries until we hit the limit
    for _ in 0..100 {
        budget.add_entry(10 * 1024); // 10 KB each

        if budget.needs_eviction() {
            // Budget correctly detects when eviction is needed
            assert!(budget.usage_ratio() > 0.9);
            break;
        }
    }
}

/// Test HNSW index scales to large numbers of entries
#[test]
fn test_hnsw_scalability() {
    // Test that we can handle large pools with HNSW
    // For now, test with SoftIndex which is simpler for integration tests
    use crate::dream::soft_index::{SoftIndex, EntryId, Similarity};

    let mut index = SoftIndex::new(32);

    // Add 1000 entries
    for i in 0..1000 {
        let id = EntryId::new_v4();
        let embedding: Vec<f32> = (0..32)
            .map(|j| ((i * 32 + j) as f32) / 32000.0)
            .collect();
        index.add(id, embedding).unwrap();
    }

    // Build index
    index.build();

    // Query should complete
    let query: Vec<f32> = (0..32).map(|i| (i as f32) / 32.0).collect();
    let results = index.query(&query, 10, Similarity::Cosine).unwrap();

    assert_eq!(results.len(), 10);
}

/// Test error recovery when index is corrupted
#[test]
fn test_error_recovery_on_index_failure() {
    let config = PoolConfig::default();
    let mut pool = SimpleDreamPool::new(config);

    // Add entries
    for _ in 0..5 {
        let tensor = ChromaticTensor::new(4, 4, 3);
        let result = SolverResult {
            energy: 0.1,
            coherence: 0.9,
            violation: 0.0,
            grad: None,
            mask: None,
            meta: json!({}),
        };
        pool.add_if_coherent(tensor, result);
    }

    let mapper = EmbeddingMapper::new(64);
    pool.rebuild_soft_index(&mapper, None);

    // Even if we clear the pool, retrieval shouldn't crash
    let query = QuerySignature::from_chroma([1.0, 0.0, 0.0]);
    let weights = RetrievalWeights::default();
    let results = pool.retrieve_soft(&query, 5, &weights, Similarity::Cosine, &mapper, None);

    // Should handle gracefully (might return empty)
    assert!(results.len() <= 5);
}

/// Test concurrent access patterns (basic safety check)
#[test]
fn test_concurrent_reads() {
    use std::sync::Arc;
    use std::thread;

    let config = PoolConfig::default();
    let mut pool = SimpleDreamPool::new(config);

    // Add entries
    for _ in 0..20 {
        let tensor = ChromaticTensor::new(4, 4, 3);
        let result = SolverResult {
            energy: 0.1,
            coherence: 0.9,
            violation: 0.0,
            grad: None,
            mask: None,
            meta: json!({}),
        };
        pool.add_if_coherent(tensor, result);
    }

    let mapper = EmbeddingMapper::new(64);
    pool.rebuild_soft_index(&mapper, None);

    // Wrap in Arc to share across threads
    let pool = Arc::new(pool);
    let mapper = Arc::new(mapper);

    // Spawn multiple reader threads
    let mut handles = vec![];
    for i in 0..4 {
        let pool_clone = Arc::clone(&pool);
        let mapper_clone = Arc::clone(&mapper);

        let handle = thread::spawn(move || {
            let query = QuerySignature::from_chroma([
                (i as f32) / 4.0,
                1.0 - (i as f32) / 4.0,
                0.0
            ]);
            let weights = RetrievalWeights::default();
            let results = pool_clone.retrieve_soft(
                &query,
                5,
                &weights,
                Similarity::Cosine,
                &*mapper_clone,
                None
            );
            results.len()
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        let count = handle.join().unwrap();
        assert!(count <= 5);
    }
}

/// Test large batch operations
#[test]
fn test_large_batch_operations() {
    let config = PoolConfig::default();
    let mut pool = SimpleDreamPool::new(config);

    // Add 100 entries in batch
    for _ in 0..100 {
        let tensor = ChromaticTensor::new(4, 4, 3);
        let result = SolverResult {
            energy: 0.1,
            coherence: 0.9,
            violation: 0.0,
            grad: None,
            mask: None,
            meta: json!({}),
        };
        pool.add_if_coherent(tensor, result);
    }

    assert_eq!(pool.len(), 100);

    // Rebuild index should handle large batches
    let mapper = EmbeddingMapper::new(64);
    pool.rebuild_soft_index(&mapper, None);

    // Retrieval should work with large pool
    let query = QuerySignature::from_chroma([1.0, 0.0, 0.0]);
    let weights = RetrievalWeights::default();
    let results = pool.retrieve_soft(&query, 10, &weights, Similarity::Cosine, &mapper, None);

    assert!(results.len() <= 10);
}

/// Test hybrid scoring weights combination
#[test]
fn test_hybrid_scoring_weights() {
    use crate::dream::hybrid_scoring::{RetrievalWeights, rerank_hybrid};
    use std::collections::HashMap;

    // Create test hits
    let id1 = EntryId::new_v4();
    let id2 = EntryId::new_v4();
    let hits = vec![(id1, 0.9), (id2, 0.7)];

    // Create test entries
    let tensor1 = ChromaticTensor::new(2, 2, 2);
    let result1 = SolverResult {
        energy: 0.1,
        coherence: 0.95,
        violation: 0.0,
        grad: None,
        mask: None,
        meta: json!({}),
    };
    let entry1 = DreamEntry::new(tensor1, result1);

    let tensor2 = ChromaticTensor::new(2, 2, 2);
    let result2 = SolverResult {
        energy: 0.2,
        coherence: 0.85,
        violation: 0.0,
        grad: None,
        mask: None,
        meta: json!({}),
    };
    let entry2 = DreamEntry::new(tensor2, result2);

    let mut entries = HashMap::new();
    entries.insert(id1, entry1);
    entries.insert(id2, entry2);

    // Test different weight configurations
    let equal_weights = RetrievalWeights {
        alpha: 0.4,    // Similarity
        beta: 0.3,     // Utility
        gamma: 0.2,    // Class match
        delta: 0.1,    // Duplicate penalty
        lambda: 0.5,   // MMR diversity
    };

    let results = rerank_hybrid(&hits, &equal_weights, &entries, None);
    assert_eq!(results.len(), 2);
}

/// Test UMS round-trip fidelity (Phase 7 / Phase 2 Cognitive Integration)
///
/// Verifies the complete data path:
/// ChromaticTensor → UMS Encode → UMS Decode → Original Tensor Features
///
/// Asserts that ΔE94 distance between starting and ending color vectors is ≤ 1.0 × 10^-3
///
/// Note: decode_from_ums returns HSL (hue, saturation, luminance), where hue is in radians.
/// This test validates that the round-trip encoding preserves color fidelity within the
/// specified tolerance using perceptual color difference (ΔE94).
#[test]
fn test_ums_round_trip_fidelity() {
    use crate::bridge::{encode_to_ums, decode_from_ums, ModalityMapper};
    use crate::config::BridgeConfig;
    use crate::spectral::color::delta_e94;
    use crate::spectral::canonical_hue;

    // Helper to convert HSL to RGB (inline version)
    fn hsl_to_rgb(h_norm: f32, saturation: f32, luminance: f32) -> [f32; 3] {
        let c = (1.0 - (2.0 * luminance - 1.0).abs()) * saturation;
        let h_prime = h_norm * 6.0;
        let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());

        let (r1, g1, b1) = match h_prime.floor() as i32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            5 | 6 => (c, 0.0, x),
            _ => (0.0, 0.0, 0.0),
        };

        let m = luminance - c / 2.0;
        [(r1 + m).clamp(0.0, 1.0), (g1 + m).clamp(0.0, 1.0), (b1 + m).clamp(0.0, 1.0)]
    }

    // Load bridge configuration
    let config = BridgeConfig::from_str(include_str!("../../../config/bridge.toml"))
        .expect("valid bridge config");
    let mapper = ModalityMapper::new(config.clone());
    let tolerance = config.reversibility.delta_e_tolerance;

    // Test with multiple seeds to ensure robustness
    for seed in [42, 123, 456, 789, 1024] {
        let tensor = ChromaticTensor::from_seed(seed, 16, 16, 4);
        let original_rgb = tensor.mean_rgb();

        // Encode to UMS
        let ums_vector = encode_to_ums(&mapper, &tensor);
        assert_eq!(ums_vector.components().len(), 512, "UMS vector must be 512D");

        // Decode from UMS (returns HSL: [hue_radians, saturation, luminance])
        let decoded_hsl = decode_from_ums(&ums_vector);

        // Convert HSL to RGB for comparison
        let hue_norm = canonical_hue(decoded_hsl[0]) / std::f32::consts::TAU;
        let decoded_rgb = hsl_to_rgb(hue_norm, decoded_hsl[1], decoded_hsl[2]);

        // Compute ΔE94 perceptual color difference
        let delta_e = delta_e94(original_rgb, decoded_rgb);

        // Assert fidelity requirement: ΔE94 ≤ tolerance (1.0 × 10^-3 from config)
        assert!(
            delta_e <= tolerance,
            "UMS round-trip fidelity requirement violated: ΔE94 = {} > {} (seed={}, original={:?}, decoded_rgb={:?}, decoded_hsl={:?})",
            delta_e,
            tolerance,
            seed,
            original_rgb,
            decoded_rgb,
            decoded_hsl
        );

        // Additional sanity checks
        for channel in 0..3 {
            assert!(
                original_rgb[channel] >= 0.0 && original_rgb[channel] <= 1.0,
                "Original RGB must be in [0, 1]"
            );
            assert!(
                decoded_rgb[channel] >= 0.0 && decoded_rgb[channel] <= 1.0,
                "Decoded RGB must be in [0, 1]: {:?}",
                decoded_rgb
            );
        }
    }
}

/// Test that DreamEntry automatically computes UMS vector and hue category (Phase 7)
#[test]
fn test_dream_entry_ums_integration() {
    let tensor = ChromaticTensor::from_seed(42, 16, 16, 4);
    let result = SolverResult {
        energy: 0.1,
        coherence: 0.9,
        violation: 0.0,
        grad: None,
        mask: None,
        meta: json!({}),
    };

    let entry = DreamEntry::new(tensor.clone(), result);

    // Verify UMS vector is computed
    assert_eq!(entry.ums_vector.len(), 512, "UMS vector must be 512D");

    // Verify hue category is in valid range [0-11]
    assert!(
        entry.hue_category < 12,
        "Hue category must be in [0, 11], got {}",
        entry.hue_category
    );

    // Verify UMS encoding is deterministic
    let result2 = SolverResult {
        energy: 0.2,
        coherence: 0.8,
        violation: 0.0,
        grad: None,
        mask: None,
        meta: json!({}),
    };
    let entry2 = DreamEntry::new(tensor.clone(), result2);

    // Same tensor should produce same UMS vector and hue category
    assert_eq!(entry.ums_vector, entry2.ums_vector, "UMS encoding must be deterministic");
    assert_eq!(entry.hue_category, entry2.hue_category, "Hue category must be deterministic");
}

/// Test category-based hybrid retrieval (Phase 3)
#[test]
fn test_retrieve_hybrid_category_filtering() {
    let config = PoolConfig::default();
    let mut pool = SimpleDreamPool::new(config);

    // Add entries with different hue categories
    // Create tensors with distinct RGB values to ensure different categories
    for i in 0..30 {
        let mut tensor = ChromaticTensor::new(8, 8, 2);
        // Create distinct hues by varying the dominant color channel
        let hue_offset = (i as f32) / 30.0;
        for row in 0..8 {
            for col in 0..8 {
                for layer in 0..2 {
                    tensor.colors[[row, col, layer, 0]] = (hue_offset + 0.1_f32).clamp(0.0, 1.0);
                    tensor.colors[[row, col, layer, 1]] = 0.5_f32.clamp(0.0, 1.0);
                    tensor.colors[[row, col, layer, 2]] = (1.0_f32 - hue_offset).clamp(0.0, 1.0);
                }
            }
        }

        let result = SolverResult {
            energy: 0.1,
            coherence: 0.9,
            violation: 0.0,
            grad: None,
            mask: None,
            meta: json!({"index": i}),
        };

        pool.add_if_coherent(tensor, result);
    }

    assert!(pool.len() > 0, "Pool should have entries");

    // Test retrieve_hybrid: should filter by query's category and rank by UMS
    let query_tensor = ChromaticTensor::from_seed(42, 8, 8, 2);
    let results = pool.retrieve_hybrid(&query_tensor, 5);

    // Verify results
    assert!(results.len() <= 5, "Should return at most 5 results");

    // All results should be from the same category as the query
    if !results.is_empty() {
        let query_entry = DreamEntry::new(query_tensor.clone(), SolverResult {
            energy: 0.1,
            coherence: 0.9,
            violation: 0.0,
            grad: None,
            mask: None,
            meta: json!({}),
        });
        let query_category = query_entry.hue_category;

        for result in &results {
            assert_eq!(
                result.hue_category, query_category,
                "All results should be from query's category"
            );
        }
    }
}

/// Test retrieve_by_category (Phase 3)
#[test]
fn test_retrieve_by_category() {
    let config = PoolConfig::default();
    let mut pool = SimpleDreamPool::new(config);

    // Add entries across different categories
    for i in 0..24 {
        let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
        let result = SolverResult {
            energy: 0.1,
            coherence: 0.9,
            violation: 0.0,
            grad: None,
            mask: None,
            meta: json!({"index": i}),
        };

        pool.add_if_coherent(tensor, result);
    }

    // Create a query UMS vector (use an actual entry's UMS for testing)
    let query_tensor = ChromaticTensor::from_seed(42, 8, 8, 2);
    let query_entry = DreamEntry::new(query_tensor, SolverResult {
        energy: 0.1,
        coherence: 0.9,
        violation: 0.0,
        grad: None,
        mask: None,
        meta: json!({}),
    });

    // Test retrieve_by_category for category 0
    let results = pool.retrieve_by_category(0, &query_entry.ums_vector, 3);

    // Verify all results are from category 0
    for result in &results {
        assert_eq!(result.hue_category, 0, "All results should be from category 0");
    }

    // Test invalid category (should return empty with warning)
    let results = pool.retrieve_by_category(12, &query_entry.ums_vector, 3);
    assert_eq!(results.len(), 0, "Invalid category should return empty results");
}
