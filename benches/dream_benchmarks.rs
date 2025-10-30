//! Performance benchmarks for dream pool optimizations
//!
//! Run with: cargo bench --bench dream_benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use chromatic_cognition_core::dream::*;
use chromatic_cognition_core::dream::soft_index::{SoftIndex, EntryId, Similarity};
use chromatic_cognition_core::dream::hnsw_index::HnswIndex;
use chromatic_cognition_core::dream::diversity::{retrieve_diverse_mmr, retrieve_diverse_mmr_fast};
use chromatic_cognition_core::dream::query_cache::QueryCache;
use chromatic_cognition_core::dream::embedding::QuerySignature;
use chromatic_cognition_core::solver::SolverResult;
use chromatic_cognition_core::tensor::ChromaticTensor;
use serde_json::json;

/// Benchmark query cache hit rates
fn bench_query_cache(c: &mut Criterion) {
    let mut cache = QueryCache::new(128);

    let query1 = [1.0, 0.0, 0.0];
    let query2 = [0.0, 1.0, 0.0];

    // Pre-populate cache
    cache.get_or_compute(&query1, |q| vec![q[0], q[1], q[2], 0.5]);

    c.bench_function("query_cache_hit", |b| {
        b.iter(|| {
            black_box(cache.get_or_compute(&query1, |q| vec![q[0], q[1], q[2], 0.5]));
        });
    });

    c.bench_function("query_cache_miss", |b| {
        b.iter(|| {
            black_box(cache.get_or_compute(&query2, |q| vec![q[0], q[1], q[2], 0.5]));
        });
    });
}

/// Benchmark HNSW vs linear k-NN at different scales
fn bench_hnsw_vs_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_search");

    for size in [100, 500, 1000, 5000].iter() {
        // Create test data
        let mut embeddings = vec![];
        for i in 0..*size {
            let id = EntryId::new_v4();
            let embedding: Vec<f32> = (0..64)
                .map(|j| ((i * 64 + j) as f32) / (size * 64) as f32)
                .collect();
            embeddings.push((id, embedding));
        }

        // Build SoftIndex (linear)
        let mut soft_index = SoftIndex::new(64);
        for (id, emb) in &embeddings {
            soft_index.add(*id, emb.clone()).unwrap();
        }
        soft_index.build();

        // Build HNSW
        let mut hnsw = HnswIndex::new(64, *size);
        for (id, emb) in &embeddings {
            hnsw.add(*id, emb.clone()).unwrap();
        }
        hnsw.build(Similarity::Cosine).unwrap();

        let query: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();

        // Benchmark linear search
        group.bench_with_input(
            BenchmarkId::new("linear", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(soft_index.query(&query, 10, Similarity::Cosine).unwrap());
                });
            },
        );

        // Benchmark HNSW search
        group.bench_with_input(
            BenchmarkId::new("hnsw", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(hnsw.search(&query, 10, Similarity::Cosine).unwrap());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark MMR standard vs fast
fn bench_mmr_diversity(c: &mut Criterion) {
    let mut group = c.benchmark_group("mmr_diversity");

    for k in [10, 20, 50].iter() {
        // Create test candidates
        let mut candidates = vec![];
        for i in 0..100 {
            let tensor = ChromaticTensor::new(4, 4, 3);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.9,
                violation: 0.0,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            let mut entry = DreamEntry::new(tensor, result);
            entry.chroma_signature = [
                (i as f32) / 100.0,
                1.0 - (i as f32) / 100.0,
                0.0
            ];
            candidates.push(entry);
        }

        let query = [1.0, 0.0, 0.0];

        // Benchmark standard MMR
        group.bench_with_input(
            BenchmarkId::new("standard", k),
            k,
            |b, k| {
                b.iter(|| {
                    black_box(retrieve_diverse_mmr(&candidates, &query, *k, 0.5, 0.0));
                });
            },
        );

        // Benchmark fast MMR with sampling
        group.bench_with_input(
            BenchmarkId::new("fast", k),
            k,
            |b, k| {
                b.iter(|| {
                    black_box(retrieve_diverse_mmr_fast(&candidates, &query, *k, 0.5, 0.0, 5));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark spectral feature extraction
fn bench_spectral_features(c: &mut Criterion) {
    use chromatic_cognition_core::spectral::{extract_spectral_features, WindowFunction};

    c.bench_function("spectral_extraction", |b| {
        let tensor = ChromaticTensor::new(8, 8, 4);
        b.iter(|| {
            black_box(extract_spectral_features(&tensor, WindowFunction::Hann));
        });
    });
}

/// Benchmark embedding encoding
fn bench_embedding_encoding(c: &mut Criterion) {
    let mapper = EmbeddingMapper::new(64);
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

    c.bench_function("encode_entry", |b| {
        b.iter(|| {
            black_box(mapper.encode_entry(&entry, None));
        });
    });

    let query = QuerySignature::from_chroma([1.0, 0.0, 0.0]);

    c.bench_function("encode_query", |b| {
        b.iter(|| {
            black_box(mapper.encode_query(&query, None));
        });
    });
}

/// Benchmark memory budget operations
fn bench_memory_budget(c: &mut Criterion) {
    use chromatic_cognition_core::dream::memory::MemoryBudget;

    let mut budget = MemoryBudget::new(10 * 1024 * 1024); // 10 MB

    c.bench_function("memory_add_entry", |b| {
        b.iter(|| {
            black_box(budget.add_entry(1024));
            budget.remove_entry(1024);
        });
    });

    c.bench_function("memory_needs_eviction", |b| {
        b.iter(|| {
            black_box(budget.needs_eviction());
        });
    });
}

/// Benchmark full retrieval pipeline
fn bench_full_pipeline(c: &mut Criterion) {
    let config = PoolConfig::default();
    let mut pool = SimpleDreamPool::new(config);

    // Add entries
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

    let mapper = EmbeddingMapper::new(64);
    pool.rebuild_soft_index(&mapper, None);

    let query = QuerySignature::from_chroma([1.0, 0.0, 0.0]);
    let weights = RetrievalWeights::default();

    c.bench_function("full_pipeline", |b| {
        b.iter(|| {
            black_box(pool.retrieve_soft(&query, 10, &weights, Similarity::Cosine, &mapper, None));
        });
    });
}

criterion_group!(
    benches,
    bench_query_cache,
    bench_hnsw_vs_linear,
    bench_mmr_diversity,
    bench_spectral_features,
    bench_embedding_encoding,
    bench_memory_budget,
    bench_full_pipeline,
);

criterion_main!(benches);
