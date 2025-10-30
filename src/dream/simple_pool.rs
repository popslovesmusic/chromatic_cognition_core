//! SimpleDreamPool - In-memory dream storage with cosine similarity retrieval
//!
//! This is a minimal implementation designed for validation experiments.
//! It stores ChromaticTensor dreams with their evaluation metrics and provides
//! retrieval based on chromatic signature similarity.

use crate::bridge::{encode_to_ums, ModalityMapper};
use crate::checkpoint::{CheckpointError, Checkpointable};
use crate::config::{
    BridgeBaseConfig, BridgeConfig, BridgeReversibilityConfig, BridgeSpectralConfig,
};
use crate::data::ColorClass;
use crate::dream::embedding::{EmbeddingMapper, QuerySignature};
use crate::dream::error::{DreamError, DreamResult};
use crate::dream::hnsw_index::{HnswIndex, HnswIndexSnapshot};
use crate::dream::hybrid_scoring::{rerank_hybrid, RetrievalWeights};
use crate::dream::memory::{estimate_entry_size, MemoryBudget, MemoryBudgetSnapshot};
use crate::dream::query_cache::QueryCache;
use crate::dream::soft_index::{EntryId, Similarity, SoftIndex, SoftIndexSnapshot};
use crate::solver::SolverResult;
use crate::spectral::{extract_spectral_features, SpectralFeatures, WindowFunction};
use crate::tensor::ChromaticTensor;
use ndarray::{Array3, Array4};
use serde::{Deserialize, Serialize};
use serde_json;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::TryFrom;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const POOL_CHECKPOINT_VERSION: u32 = 2;

#[derive(Serialize, Deserialize)]
struct SimpleDreamPoolCheckpoint {
    version: u32,
    config: PoolConfig,
    entries: Vec<DreamEntryCheckpoint>,
    entry_ids: Vec<EntryId>,
    id_map: Vec<(EntryId, usize)>,
    soft_index: Option<SoftIndexSnapshot>,
    hnsw_index: Option<HnswIndexSnapshot>,
    memory_budget: Option<MemoryBudgetSnapshot>,
    evictions_since_rebuild: usize,
}

/// A stored dream entry with tensor and evaluation metrics
///
/// Enhanced for Phase 3B with class awareness, utility tracking, and timestamps
/// Enhanced for Phase 4 with spectral features and embeddings
/// Enhanced for Phase 7 (Phase 2 Cognitive Integration) with UMS vector
#[derive(Clone, Serialize, Deserialize)]
pub struct DreamEntry {
    pub tensor: ChromaticTensor,
    pub result: SolverResult,
    pub chroma_signature: [f32; 3],
    /// Optional class label for class-aware retrieval (Phase 3B)
    pub class_label: Option<ColorClass>,
    /// Utility score from feedback (Phase 3B)
    pub utility: Option<f32>,
    /// Timestamp for recency tracking (Phase 3B)
    pub timestamp: SystemTime,
    /// Number of times this dream has been retrieved (Phase 3B)
    pub usage_count: usize,
    /// Spectral features for embedding (Phase 4) - Always computed on creation
    pub spectral_features: SpectralFeatures,
    /// Cached embedding vector (Phase 4)
    pub embed: Option<Vec<f32>>,
    /// Aggregated mean utility (Phase 4)
    pub util_mean: f32,
    /// Unified Modality Space vector (Phase 7 / Phase 2 Cognitive Integration)
    /// 512D deterministic encoding for Chromatic Semantic Archive (CSA)
    pub ums_vector: Vec<f32>,
    /// Hue category index [0-11] for CSA partitioning (Phase 7)
    pub hue_category: usize,
}

#[derive(Serialize, Deserialize)]
struct ChromaticTensorCheckpoint {
    colors: Vec<f32>,
    certainty: Vec<f32>,
    rows: usize,
    cols: usize,
    layers: usize,
}

#[derive(Serialize, Deserialize)]
struct DreamEntryCheckpoint {
    tensor: ChromaticTensorCheckpoint,
    result: SolverResultCheckpoint,
    chroma_signature: [f32; 3],
    class_label: Option<ColorClass>,
    utility: Option<f32>,
    timestamp_secs: i128,
    timestamp_nanos: u32,
    usage_count: usize,
    spectral_features: SpectralFeatures,
    embed: Option<Vec<f32>>,
    util_mean: f32,
    ums_vector: Vec<f32>,
    hue_category: usize,
}

#[derive(Serialize, Deserialize)]
struct SolverResultCheckpoint {
    energy: f64,
    coherence: f64,
    violation: f64,
    grad: Option<Vec<f32>>,
    mask: Option<Vec<f32>>,
    meta_json: Vec<u8>,
}

impl DreamEntry {
    /// Create a new dream entry from a tensor and its evaluation result
    ///
    /// Spectral features are computed immediately using Hann windowing.
    /// This one-time computation enables faster embedding generation later.
    ///
    /// Phase 7: UMS vector is computed immediately for Chromatic Semantic Archive.
    pub fn new(tensor: ChromaticTensor, result: SolverResult) -> Self {
        let chroma_signature = tensor.mean_rgb();
        let spectral_features = extract_spectral_features(&tensor, WindowFunction::Hann);

        // Phase 7: Compute UMS vector and hue category for CSA
        let mapper = Self::default_modality_mapper();
        let ums = encode_to_ums(&mapper, &tensor);
        let ums_vector = ums.components().to_vec();

        // Extract hue from chroma signature and map to category
        let rgb = chroma_signature;
        let hue_radians = Self::rgb_to_hue(rgb);
        let hue_category = mapper.map_hue_to_category(hue_radians);

        Self {
            tensor,
            result,
            chroma_signature,
            class_label: None,
            utility: None,
            timestamp: SystemTime::now(),
            usage_count: 0,
            spectral_features,
            embed: None,
            util_mean: 0.0,
            ums_vector,
            hue_category,
        }
    }

    /// Create a new dream entry with class label (Phase 3B)
    ///
    /// Spectral features are computed immediately using Hann windowing.
    ///
    /// Phase 7: UMS vector is computed immediately for Chromatic Semantic Archive.
    pub fn with_class(
        tensor: ChromaticTensor,
        result: SolverResult,
        class_label: ColorClass,
    ) -> Self {
        let chroma_signature = tensor.mean_rgb();
        let spectral_features = extract_spectral_features(&tensor, WindowFunction::Hann);

        // Phase 7: Compute UMS vector and hue category for CSA
        let mapper = Self::default_modality_mapper();
        let ums = encode_to_ums(&mapper, &tensor);
        let ums_vector = ums.components().to_vec();

        // Extract hue from chroma signature and map to category
        let rgb = chroma_signature;
        let hue_radians = Self::rgb_to_hue(rgb);
        let hue_category = mapper.map_hue_to_category(hue_radians);

        Self {
            tensor,
            result,
            chroma_signature,
            class_label: Some(class_label),
            utility: None,
            timestamp: SystemTime::now(),
            usage_count: 0,
            spectral_features,
            embed: None,
            util_mean: 0.0,
            ums_vector,
            hue_category,
        }
    }

    /// Update the utility score for this dream (Phase 3B)
    pub fn set_utility(&mut self, utility: f32) {
        self.utility = Some(utility);
    }

    /// Increment usage count when retrieved (Phase 3B)
    pub fn increment_usage(&mut self) {
        self.usage_count += 1;
    }

    /// Create default modality mapper for UMS encoding (Phase 7)
    fn default_modality_mapper() -> ModalityMapper {
        let bridge_config = BridgeConfig::from_str(include_str!("../../config/bridge.toml"))
            .unwrap_or_else(|err| {
                tracing::warn!(
                    "Failed to load bridge configuration for UMS encoding; falling back to defaults: {}",
                    err
                );
                BridgeConfig {
                    base: BridgeBaseConfig {
                        f_min: 110.0,
                        octaves: 7.0,
                        gamma: 1.0,
                        sample_rate: 44_100,
                    },
                    spectral: BridgeSpectralConfig {
                        fft_size: 4096,
                        accum_format: "Q16.48".to_string(),
                        reduction_mode: "pairwise_neumaier".to_string(),
                        categorical_count: 12,
                    },
                    reversibility: BridgeReversibilityConfig {
                        delta_e_tolerance: 0.001,
                    },
                }
            });

        ModalityMapper::new(bridge_config)
    }

    /// Convert RGB to hue in radians (Phase 7)
    ///
    /// Uses HSV color space conversion to extract hue angle.
    fn rgb_to_hue(rgb: [f32; 3]) -> f32 {
        let r = rgb[0];
        let g = rgb[1];
        let b = rgb[2];

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        if delta < f32::EPSILON {
            return 0.0; // Achromatic (gray)
        }

        let hue_deg = if (max - r).abs() < f32::EPSILON {
            // Red is max
            60.0 * (((g - b) / delta) % 6.0)
        } else if (max - g).abs() < f32::EPSILON {
            // Green is max
            60.0 * (((b - r) / delta) + 2.0)
        } else {
            // Blue is max
            60.0 * (((r - g) / delta) + 4.0)
        };

        // Convert to radians and ensure positive
        let hue_rad = hue_deg.to_radians();
        if hue_rad < 0.0 {
            hue_rad + std::f32::consts::TAU
        } else {
            hue_rad
        }
    }
}

fn encode_system_time(ts: SystemTime) -> (i128, u32) {
    match ts.duration_since(UNIX_EPOCH) {
        Ok(duration) => (duration.as_secs() as i128, duration.subsec_nanos()),
        Err(err) => {
            let duration = err.duration();
            (-(duration.as_secs() as i128), duration.subsec_nanos())
        }
    }
}

fn decode_system_time(secs: i128, nanos: u32) -> Result<SystemTime, CheckpointError> {
    if nanos >= 1_000_000_000 {
        return Err(CheckpointError::InvalidFormat(
            "Timestamp nanoseconds out of range".to_string(),
        ));
    }

    if secs >= 0 {
        let secs_u64 = u64::try_from(secs).map_err(|_| {
            CheckpointError::InvalidFormat("Timestamp seconds overflow".to_string())
        })?;
        Ok(UNIX_EPOCH + Duration::new(secs_u64, nanos))
    } else {
        if secs == i128::MIN {
            return Err(CheckpointError::InvalidFormat(
                "Timestamp seconds underflow".to_string(),
            ));
        }
        let abs_secs = (-secs) as i128;
        let abs_secs_u64 = u64::try_from(abs_secs).map_err(|_| {
            CheckpointError::InvalidFormat("Timestamp seconds overflow".to_string())
        })?;
        let duration = Duration::new(abs_secs_u64, nanos);
        Ok(UNIX_EPOCH - duration)
    }
}

impl DreamEntryCheckpoint {
    fn capture(entry: &DreamEntry) -> Result<Self, CheckpointError> {
        let (timestamp_secs, timestamp_nanos) = encode_system_time(entry.timestamp);
        Ok(Self {
            tensor: ChromaticTensorCheckpoint::from(&entry.tensor),
            result: SolverResultCheckpoint::capture(&entry.result)?,
            chroma_signature: entry.chroma_signature,
            class_label: entry.class_label,
            utility: entry.utility,
            timestamp_secs,
            timestamp_nanos,
            usage_count: entry.usage_count,
            spectral_features: entry.spectral_features.clone(),
            embed: entry.embed.clone(),
            util_mean: entry.util_mean,
            ums_vector: entry.ums_vector.clone(),
            hue_category: entry.hue_category,
        })
    }

    fn into_entry(self) -> Result<DreamEntry, CheckpointError> {
        let timestamp = decode_system_time(self.timestamp_secs, self.timestamp_nanos)?;
        Ok(DreamEntry {
            tensor: self.tensor.into_tensor()?,
            result: self.result.into_result()?,
            chroma_signature: self.chroma_signature,
            class_label: self.class_label,
            utility: self.utility,
            timestamp,
            usage_count: self.usage_count,
            spectral_features: self.spectral_features,
            embed: self.embed,
            util_mean: self.util_mean,
            ums_vector: self.ums_vector,
            hue_category: self.hue_category,
        })
    }
}

impl SolverResultCheckpoint {
    fn capture(result: &SolverResult) -> Result<Self, CheckpointError> {
        let meta_json = serde_json::to_vec(&result.meta).map_err(|err| {
            CheckpointError::InvalidFormat(format!(
                "Failed to serialize solver metadata for checkpoint: {err}"
            ))
        })?;

        Ok(Self {
            energy: result.energy,
            coherence: result.coherence,
            violation: result.violation,
            grad: result.grad.clone(),
            mask: result.mask.clone(),
            meta_json,
        })
    }

    fn into_result(self) -> Result<SolverResult, CheckpointError> {
        let meta = serde_json::from_slice(&self.meta_json).map_err(|err| {
            CheckpointError::InvalidFormat(format!(
                "Failed to deserialize solver metadata from checkpoint: {err}"
            ))
        })?;

        Ok(SolverResult {
            energy: self.energy,
            coherence: self.coherence,
            violation: self.violation,
            grad: self.grad,
            mask: self.mask,
            meta,
        })
    }
}

impl From<&ChromaticTensor> for ChromaticTensorCheckpoint {
    fn from(tensor: &ChromaticTensor) -> Self {
        let (rows, cols, layers, _) = tensor.colors.dim();
        Self {
            colors: tensor.colors.iter().cloned().collect(),
            certainty: tensor.certainty.iter().cloned().collect(),
            rows,
            cols,
            layers,
        }
    }
}

impl ChromaticTensorCheckpoint {
    fn into_tensor(self) -> Result<ChromaticTensor, CheckpointError> {
        let color_len = self
            .rows
            .checked_mul(self.cols)
            .and_then(|v| v.checked_mul(self.layers))
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| {
                CheckpointError::InvalidFormat("Tensor dimensions overflow".to_string())
            })?;
        if self.colors.len() != color_len {
            return Err(CheckpointError::InvalidFormat(
                "Color tensor length mismatch".to_string(),
            ));
        }

        let certainty_len = self
            .rows
            .checked_mul(self.cols)
            .and_then(|v| v.checked_mul(self.layers))
            .ok_or_else(|| {
                CheckpointError::InvalidFormat("Certainty tensor dimensions overflow".to_string())
            })?;
        if self.certainty.len() != certainty_len {
            return Err(CheckpointError::InvalidFormat(
                "Certainty tensor length mismatch".to_string(),
            ));
        }

        let colors = Array4::from_shape_vec((self.rows, self.cols, self.layers, 3), self.colors)
            .map_err(|_| {
                CheckpointError::InvalidFormat("Invalid color tensor shape".to_string())
            })?;
        let certainty = Array3::from_shape_vec((self.rows, self.cols, self.layers), self.certainty)
            .map_err(|_| {
                CheckpointError::InvalidFormat("Invalid certainty tensor shape".to_string())
            })?;

        Ok(ChromaticTensor::from_arrays(colors, certainty))
    }
}

/// Configuration for SimpleDreamPool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Maximum number of dreams to store in memory
    pub max_size: usize,
    /// Minimum coherence threshold for persistence (0.0-1.0)
    pub coherence_threshold: f64,
    /// Number of similar dreams to retrieve
    pub retrieval_limit: usize,
    /// Use HNSW index for scalable retrieval (Phase 4 optimization)
    /// When false, uses linear SoftIndex (simpler but O(n) search)
    /// Enable this when the pool regularly exceeds ~5,000 entries or when
    /// retrieval latency must remain under 100 ms.
    pub use_hnsw: bool,
    /// Memory budget in megabytes (Phase 4 optimization)
    /// When None, no memory limit is enforced (legacy behavior)
    /// When Some(mb), triggers automatic eviction at 90% of limit
    pub memory_budget_mb: Option<usize>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_size: 1000,
            coherence_threshold: 0.75,
            retrieval_limit: 3,
            use_hnsw: false,        // Default to linear index; HNSW is opt-in
            memory_budget_mb: None, // Unlimited unless specified
        }
    }
}

/// In-memory dream pool with cosine similarity retrieval
///
/// Stores high-coherence ChromaticTensor states and retrieves similar dreams
/// based on chromatic signature (mean RGB) for retrieval-based seeding.
///
/// # Example
///
/// ```rust
/// use chromatic_cognition_core::dream::simple_pool::PoolConfig;
/// use chromatic_cognition_core::dream::SimpleDreamPool;
/// use chromatic_cognition_core::{ChromaticNativeSolver, ChromaticTensor, Solver};
///
/// let config = PoolConfig::default();
/// let mut pool = SimpleDreamPool::new(config);
///
/// // Evaluate and store a dream
/// let tensor = ChromaticTensor::from_seed(42, 32, 32, 4);
/// let mut solver = ChromaticNativeSolver::default();
/// let result = solver.evaluate(&tensor, false).unwrap();
///
/// pool.add_if_coherent(tensor.clone(), result);
///
/// // Retrieve similar dreams
/// let query_signature = tensor.mean_rgb();
/// let similar = pool.retrieve_similar(&query_signature, 3);
/// ```
pub struct SimpleDreamPool {
    entries: VecDeque<DreamEntry>,
    config: PoolConfig,
    modality_mapper: ModalityMapper,
    /// Phase 4: Soft index for semantic retrieval (linear O(n) search)
    soft_index: Option<SoftIndex>,
    /// Phase 4 Optimization: HNSW index for scalable retrieval (O(log n) search)
    hnsw_index: Option<HnswIndex<'static>>,
    /// Phase 4: Mapping from EntryId to DreamEntry for retrieval
    id_to_entry: HashMap<EntryId, DreamEntry>,
    /// Phase 4: Mapping from index in entries VecDeque to EntryId
    entry_ids: VecDeque<EntryId>,
    /// Phase 4 Optimization: LRU cache for query embeddings
    query_cache: QueryCache,
    /// Phase 4 Optimization: Memory budget tracker for automatic eviction
    memory_budget: Option<MemoryBudget>,
    /// Number of entries evicted since the last index rebuild/invalidation
    evictions_since_rebuild: usize,
}

impl SimpleDreamPool {
    /// Create a new dream pool with the given configuration
    pub fn new(config: PoolConfig) -> Self {
        let max_size = config.max_size;
        let memory_budget = config.memory_budget_mb.map(|mb| {
            let mut budget = MemoryBudget::new(mb);
            if config.use_hnsw {
                budget.set_ann_overhead_factor(2.0);
            }
            budget
        });
        let modality_mapper = Self::default_modality_mapper();

        Self {
            entries: VecDeque::with_capacity(max_size),
            config,
            modality_mapper,
            soft_index: None,
            hnsw_index: None,
            id_to_entry: HashMap::new(),
            entry_ids: VecDeque::with_capacity(max_size),
            query_cache: QueryCache::new(128), // Cache last 128 queries (~40 KB)
            memory_budget,
            evictions_since_rebuild: 0,
        }
    }

    fn default_modality_mapper() -> ModalityMapper {
        let bridge_config = BridgeConfig::from_str(include_str!("../../config/bridge.toml"))
            .unwrap_or_else(|err| {
                tracing::warn!(
                    "Failed to load bridge configuration for modality mapper; falling back to defaults: {}",
                    err
                );
                BridgeConfig {
                    base: BridgeBaseConfig {
                        f_min: 110.0,
                        octaves: 7.0,
                        gamma: 1.0,
                        sample_rate: 44_100,
                    },
                    spectral: BridgeSpectralConfig {
                        fft_size: 4096,
                        accum_format: "Q16.48".to_string(),
                        reduction_mode: "pairwise_neumaier".to_string(),
                        categorical_count: 12,
                    },
                    reversibility: BridgeReversibilityConfig {
                        delta_e_tolerance: 0.001,
                    },
                }
            });

        ModalityMapper::new(bridge_config)
    }

    fn attach_semantic_embedding(&self, entry: &mut DreamEntry) -> Vec<f32> {
        let ums = encode_to_ums(&self.modality_mapper, &entry.tensor);
        let embedding = ums.components().to_vec();
        entry.embed = Some(embedding.clone());
        embedding
    }

    fn rebuild_semantic_index_internal(&mut self) -> DreamResult<()> {
        if self.entries.is_empty() {
            if let Some(index) = self.hnsw_index.as_mut() {
                index.clear();
            }
            return Ok(());
        }

        let dim = self
            .entries
            .front()
            .and_then(|entry| entry.embed.as_ref().map(|vec| vec.len()))
            .or_else(|| {
                self.entries.front().map(|entry| {
                    encode_to_ums(&self.modality_mapper, &entry.tensor)
                        .components()
                        .len()
                })
            })
            .unwrap_or(0);

        if dim == 0 {
            return Err(DreamError::index_corrupted(
                "Semantic embedding dimension resolved to zero during HNSW rebuild",
            ));
        }

        let capacity = self.entries.len();
        let needs_new = self
            .hnsw_index
            .as_ref()
            .map_or(true, |index| index.dim() != dim);

        if needs_new {
            self.hnsw_index = Some(HnswIndex::new(dim, capacity));
        }

        let hnsw = self
            .hnsw_index
            .as_mut()
            .expect("HNSW index must exist after initialization");

        hnsw.clear();

        for (entry_id, entry) in self.entry_ids.iter().copied().zip(self.entries.iter()) {
            let vector = entry.embed.as_ref().cloned().unwrap_or_else(|| {
                encode_to_ums(&self.modality_mapper, &entry.tensor)
                    .components()
                    .to_vec()
            });
            hnsw.add(entry_id, vector)?;
        }

        hnsw.build(Similarity::Cosine)?;
        self.evictions_since_rebuild = 0;

        Ok(())
    }

    fn cosine_similarity_dense(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    fn filter_active_results(&self, hits: Vec<(EntryId, f32)>, k: usize) -> Vec<EntryId> {
        hits.into_iter()
            .filter_map(|(id, _)| self.id_to_entry.contains_key(&id).then_some(id))
            .take(k)
            .collect()
    }

    fn linear_semantic_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> DreamResult<Vec<EntryId>> {
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut scored: Vec<(EntryId, f32)> = Vec::with_capacity(self.entry_ids.len());

        for entry_id in self.entry_ids.iter().copied() {
            if let Some(entry) = self.id_to_entry.get(&entry_id) {
                let candidate = entry.embed.as_ref().cloned().unwrap_or_else(|| {
                    encode_to_ums(&self.modality_mapper, &entry.tensor)
                        .components()
                        .to_vec()
                });

                if candidate.len() != query_embedding.len() {
                    return Err(DreamError::dimension_mismatch(
                        query_embedding.len(),
                        candidate.len(),
                        "linear semantic search",
                    ));
                }

                let score = Self::cosine_similarity_dense(query_embedding, &candidate);
                scored.push((entry_id, score));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        Ok(scored.into_iter().take(k).map(|(id, _)| id).collect())
    }

    /// Evict up to `count` entries from the front of the pool while keeping all
    /// auxiliary structures synchronized.
    fn evict_n_entries(&mut self, count: usize) {
        if count == 0 {
            return;
        }

        let mut evicted_count = 0usize;

        for _ in 0..count {
            let old_entry = match self.entries.pop_front() {
                Some(entry) => entry,
                None => break,
            };

            evicted_count = evicted_count.saturating_add(1);

            if let Some(ref mut budget) = self.memory_budget {
                let old_size = estimate_entry_size(&old_entry);
                budget.remove_entry(old_size);
            }

            if let Some(old_id) = self.entry_ids.pop_front() {
                self.id_to_entry.remove(&old_id);

                if let Some(hnsw) = self.hnsw_index.as_mut() {
                    if !hnsw.remove(&old_id) {
                        tracing::warn!("Evicted entry {old_id:?} was missing from HNSW id map");
                    }
                }
            }
        }

        if evicted_count > 0 {
            self.evictions_since_rebuild =
                self.evictions_since_rebuild.saturating_add(evicted_count);
            self.maybe_invalidate_indices();
        }
    }

    fn internal_add(&mut self, entry: DreamEntry, embedding: Vec<f32>) -> bool {
        let entry_size = estimate_entry_size(&entry);
        let initial_evictions = if let Some(budget) = self.memory_budget.as_ref() {
            let needs_space = !budget.can_add(entry_size) || budget.needs_eviction();
            if needs_space {
                let avg_entry_size = if budget.entry_count() > 0 {
                    budget.average_entry_size()
                } else {
                    entry_size
                };

                let mut eviction_count = budget.calculate_eviction_count(avg_entry_size);

                if !budget.can_add(entry_size) && avg_entry_size > 0 {
                    let numerator = entry_size.saturating_add(avg_entry_size).saturating_sub(1);
                    let additional = numerator / avg_entry_size;
                    eviction_count = eviction_count.max(additional);
                }

                if eviction_count == 0 {
                    1
                } else {
                    eviction_count
                }
            } else {
                0
            }
        } else {
            0
        };

        if initial_evictions > 0 {
            self.evict_n_entries(initial_evictions);
        }

        while {
            if let Some(budget) = self.memory_budget.as_ref() {
                !budget.can_add(entry_size)
            } else {
                false
            }
        } && !self.entries.is_empty()
        {
            self.evict_n_entries(1);
        }

        let overflow = self
            .entries
            .len()
            .saturating_add(1)
            .saturating_sub(self.config.max_size);
        if overflow > 0 {
            self.evict_n_entries(overflow);
        }

        let entry_id = EntryId::new_v4();
        self.entry_ids.push_back(entry_id);
        self.id_to_entry.insert(entry_id, entry.clone());
        self.entries.push_back(entry);

        if let Some(ref mut budget) = self.memory_budget {
            budget.add_entry(entry_size);
        }

        if let Some(hnsw) = self.hnsw_index.as_mut() {
            if let Err(err) = hnsw.add(entry_id, embedding.clone()) {
                tracing::warn!(
                    "Failed to add entry {entry_id:?} to HNSW index; semantic search may require rebuild: {}",
                    err
                );
            }
        }

        self.maybe_invalidate_indices();

        true
    }

    /// Drop all retrieval indices when eviction churn exceeds the configured threshold.
    fn maybe_invalidate_indices(&mut self) {
        let threshold = self.entries.len() / 10;
        if self.evictions_since_rebuild > threshold {
            self.invalidate_indices();
        }
    }

    /// Clear both HNSW and linear indices and reset the eviction counter.
    fn invalidate_indices(&mut self) {
        if let Some(hnsw) = self.hnsw_index.as_mut() {
            if let Err(err) = hnsw.rebuild_active() {
                tracing::warn!(
                    "Failed to refresh HNSW index after eviction churn; continuing with existing graph: {}",
                    err
                );
            }
        }
        self.soft_index = None;
        self.evictions_since_rebuild = 0;
    }

    /// Add a dream entry if it meets the coherence threshold
    ///
    /// Returns true if the dream was added, false otherwise.
    /// If the pool is at capacity or memory budget is exceeded, oldest dreams are evicted (FIFO).
    pub fn add_if_coherent(&mut self, tensor: ChromaticTensor, result: SolverResult) -> bool {
        if result.coherence < self.config.coherence_threshold {
            return false;
        }

        let mut entry = DreamEntry::new(tensor, result);
        let embedding = self.attach_semantic_embedding(&mut entry);

        self.internal_add(entry, embedding)
    }

    /// Force add a dream entry regardless of coherence threshold
    ///
    /// Useful for testing or when coherence filtering is not desired.
    /// Respects memory budget and pool capacity limits.
    pub fn add(&mut self, tensor: ChromaticTensor, result: SolverResult) {
        let mut entry = DreamEntry::new(tensor, result);
        let embedding = self.attach_semantic_embedding(&mut entry);

        let _ = self.internal_add(entry, embedding);
    }

    /// Add a dream entry with class label (Phase 3B)
    ///
    /// # Arguments
    /// * `tensor` - The chromatic tensor to store
    /// * `result` - The solver evaluation result
    /// * `class_label` - The color class this dream represents
    ///
    /// # Returns
    /// true if the dream was added, false if it didn't meet coherence threshold
    pub fn add_with_class(
        &mut self,
        tensor: ChromaticTensor,
        result: SolverResult,
        class_label: ColorClass,
    ) -> bool {
        if result.coherence < self.config.coherence_threshold {
            return false;
        }

        let mut entry = DreamEntry::with_class(tensor, result, class_label);
        let embedding = self.attach_semantic_embedding(&mut entry);

        self.internal_add(entry, embedding)
    }

    /// Retrieve K most similar dreams based on cosine similarity of chroma signatures
    ///
    /// # Arguments
    /// * `query_signature` - Target RGB signature to match against [r, g, b]
    /// * `k` - Number of similar dreams to retrieve
    ///
    /// # Returns
    /// Vector of up to K most similar dreams, sorted by similarity (highest first)
    pub fn retrieve_similar(&self, query_signature: &[f32; 3], k: usize) -> Vec<DreamEntry> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        // Compute cosine similarity for all entries
        let mut scored: Vec<(f32, &DreamEntry)> = self
            .entries
            .iter()
            .map(|entry| {
                let similarity = cosine_similarity(query_signature, &entry.chroma_signature);
                (similarity, entry)
            })
            .collect();

        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top K and clone the entries
        scored
            .into_iter()
            .take(k)
            .map(|(_, entry)| entry.clone())
            .collect()
    }

    /// Retrieve K most similar dreams filtered by class label (Phase 3B)
    ///
    /// Only retrieves dreams that match the specified class label.
    ///
    /// # Arguments
    /// * `query_signature` - Target RGB signature to match against
    /// * `target_class` - The class to filter by
    /// * `k` - Number of similar dreams to retrieve
    ///
    /// # Returns
    /// Vector of up to K most similar dreams from the target class
    pub fn retrieve_similar_class(
        &self,
        query_signature: &[f32; 3],
        target_class: ColorClass,
        k: usize,
    ) -> Vec<DreamEntry> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        // Filter by class, then compute similarity
        let mut scored: Vec<(f32, &DreamEntry)> = self
            .entries
            .iter()
            .filter(|entry| entry.class_label == Some(target_class))
            .map(|entry| {
                let similarity = cosine_similarity(query_signature, &entry.chroma_signature);
                (similarity, entry)
            })
            .collect();

        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top K and clone
        scored
            .into_iter()
            .take(k)
            .map(|(_, entry)| entry.clone())
            .collect()
    }

    /// Retrieve dreams with balanced representation across classes (Phase 3B)
    ///
    /// Retrieves `k_per_class` dreams from each specified class.
    ///
    /// # Arguments
    /// * `query_signature` - Target RGB signature to match against
    /// * `classes` - List of classes to retrieve from
    /// * `k_per_class` - Number of dreams to retrieve per class
    ///
    /// # Returns
    /// Vector of dreams with balanced class representation
    pub fn retrieve_balanced(
        &self,
        query_signature: &[f32; 3],
        classes: &[ColorClass],
        k_per_class: usize,
    ) -> Vec<DreamEntry> {
        let mut result = Vec::new();

        for &class in classes {
            let class_dreams = self.retrieve_similar_class(query_signature, class, k_per_class);
            result.extend(class_dreams);
        }

        result
    }

    /// Retrieve dreams filtered by utility threshold (Phase 3B)
    ///
    /// Only retrieves dreams with utility >= threshold.
    ///
    /// # Arguments
    /// * `query_signature` - Target RGB signature
    /// * `k` - Number of dreams to retrieve
    /// * `utility_threshold` - Minimum utility score
    ///
    /// # Returns
    /// Vector of high-utility dreams sorted by similarity
    pub fn retrieve_by_utility(
        &self,
        query_signature: &[f32; 3],
        k: usize,
        utility_threshold: f32,
    ) -> Vec<DreamEntry> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        // Filter by utility, then compute similarity
        let mut scored: Vec<(f32, &DreamEntry)> = self
            .entries
            .iter()
            .filter(|entry| {
                entry
                    .utility
                    .map(|u| u >= utility_threshold)
                    .unwrap_or(false)
            })
            .map(|entry| {
                let similarity = cosine_similarity(query_signature, &entry.chroma_signature);
                (similarity, entry)
            })
            .collect();

        // Sort by similarity (descending)
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top K and clone
        scored
            .into_iter()
            .take(k)
            .map(|(_, entry)| entry.clone())
            .collect()
    }

    /// Retrieve diverse dreams using Maximum Marginal Relevance (Phase 3B)
    ///
    /// Balances relevance to query with diversity from already-selected dreams.
    /// Uses MMR algorithm to avoid near-duplicates and ensure chromatic variety.
    ///
    /// # Arguments
    /// * `query_signature` - Target RGB signature
    /// * `k` - Number of dreams to retrieve
    /// * `lambda` - Relevance vs diversity tradeoff [0=max diversity, 1=max relevance]
    /// * `min_dispersion` - Minimum required chromatic dispersion (0.0 = no constraint)
    ///
    /// # Returns
    /// Vector of diverse dreams selected by MMR
    ///
    /// # Example
    /// ```rust
    /// # use chromatic_cognition_core::dream::simple_pool::PoolConfig;
    /// # use chromatic_cognition_core::dream::SimpleDreamPool;
    /// let config = PoolConfig::default();
    /// let mut pool = SimpleDreamPool::new(config);
    /// // ... add dreams ...
    /// let query = [1.0, 0.0, 0.0]; // Red query
    /// let diverse = pool.retrieve_diverse(&query, 5, 0.7, 0.1);
    /// // Returns 5 dreams that are relevant to red but diverse from each other
    /// ```
    pub fn retrieve_diverse(
        &self,
        query_signature: &[f32; 3],
        k: usize,
        lambda: f32,
        min_dispersion: f32,
    ) -> Vec<DreamEntry> {
        use crate::dream::diversity::retrieve_diverse_mmr;

        if self.entries.is_empty() {
            return Vec::new();
        }

        // Convert to slice for MMR algorithm
        let candidates: Vec<DreamEntry> = self.entries.iter().cloned().collect();
        retrieve_diverse_mmr(&candidates, query_signature, k, lambda, min_dispersion)
    }

    /// Get the number of dreams currently stored
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn evict_for_test(&mut self, count: usize) {
        self.evict_n_entries(count);
    }

    #[cfg(test)]
    pub(crate) fn has_soft_index_for_test(&self) -> bool {
        self.soft_index.is_some()
    }

    #[cfg(test)]
    pub(crate) fn evictions_since_rebuild_for_test(&self) -> usize {
        self.evictions_since_rebuild
    }

    #[cfg(test)]
    pub(crate) fn hnsw_len_for_test(&self) -> Option<usize> {
        self.hnsw_index.as_ref().map(|idx| idx.len())
    }

    #[cfg(test)]
    pub(crate) fn hnsw_built_for_test(&self) -> bool {
        self.hnsw_index
            .as_ref()
            .map(|idx| idx.stats().built)
            .unwrap_or(false)
    }

    #[cfg(test)]
    pub(crate) fn hnsw_ghosts_for_test(&self) -> Option<usize> {
        self.hnsw_index
            .as_ref()
            .map(|idx| idx.ghost_count_for_mode_test(Similarity::Cosine))
    }

    /// Clear all stored dreams
    pub fn clear(&mut self) {
        self.entries.clear();
        self.query_cache.clear();
        if let Some(ref mut budget) = self.memory_budget {
            budget.reset();
        }
    }

    /// Get query cache statistics
    ///
    /// Returns (hits, misses, hit_rate) tuple for performance monitoring
    pub fn query_cache_stats(&self) -> (u64, u64, f64) {
        (
            self.query_cache.hits(),
            self.query_cache.misses(),
            self.query_cache.hit_rate(),
        )
    }

    /// Clear the query cache (useful for benchmarking)
    pub fn clear_query_cache(&mut self) {
        self.query_cache.clear();
    }

    /// Get memory budget statistics (Phase 4 optimization)
    ///
    /// Returns memory usage info: (current_mb, max_mb, usage_ratio, entry_count)
    /// Returns None if memory budget is not enabled.
    pub fn memory_budget_stats(&self) -> Option<(f64, f64, f32, usize)> {
        self.memory_budget.as_ref().map(|budget| {
            let current_mb = budget.current_usage() as f64 / (1024.0 * 1024.0);
            let max_mb = budget.max_budget() as f64 / (1024.0 * 1024.0);
            let usage_ratio = budget.usage_ratio();
            let entry_count = budget.entry_count();
            (current_mb, max_mb, usage_ratio, entry_count)
        })
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        if self.entries.is_empty() {
            return PoolStats {
                count: 0,
                mean_coherence: 0.0,
                mean_energy: 0.0,
                mean_violation: 0.0,
            };
        }

        let count = self.entries.len();
        let sum_coherence: f64 = self.entries.iter().map(|e| e.result.coherence).sum();
        let sum_energy: f64 = self.entries.iter().map(|e| e.result.energy).sum();
        let sum_violation: f64 = self.entries.iter().map(|e| e.result.violation).sum();

        PoolStats {
            count,
            mean_coherence: sum_coherence / count as f64,
            mean_energy: sum_energy / count as f64,
            mean_violation: sum_violation / count as f64,
        }
    }

    /// Get diversity statistics for the pool (Phase 3B)
    ///
    /// Computes chromatic dispersion and distance metrics across all dreams.
    ///
    /// # Returns
    /// * DiversityStats with mean/min/max pairwise distances
    pub fn diversity_stats(&self) -> crate::dream::diversity::DiversityStats {
        use crate::dream::diversity::DiversityStats;

        if self.entries.is_empty() {
            return DiversityStats {
                mean_dispersion: 0.0,
                min_distance: 0.0,
                max_distance: 0.0,
                count: 0,
            };
        }

        let entries: Vec<DreamEntry> = self.entries.iter().cloned().collect();
        DiversityStats::compute(&entries)
    }

    /// Rebuild the soft index using the provided embedding mapper (Phase 4)
    ///
    /// # Arguments
    /// * `mapper` - EmbeddingMapper to encode entries into fixed-dimensional vectors
    /// * `bias` - Optional BiasProfile for query-time conditioning
    ///
    /// This method encodes all current entries and builds either:
    /// - HNSW graph index (O(log n), 100× faster at 10K+ entries) if `config.use_hnsw = true`
    /// - SoftIndex linear scan (O(n), simpler) if `config.use_hnsw = false`
    ///
    /// Should be called after adding a batch of entries or when BiasProfile changes.
    pub fn rebuild_soft_index(
        &mut self,
        mapper: &EmbeddingMapper,
        bias: Option<&crate::dream::bias::BiasProfile>,
    ) {
        let embed_dim = mapper.dim;

        if let Some(budget) = self.memory_budget.as_mut() {
            if self.config.use_hnsw {
                budget.set_ann_overhead_factor(2.0);
            } else {
                budget.set_ann_overhead_factor(1.0);
            }
        }

        let mut embeddings: Vec<(EntryId, Vec<f32>)> = Vec::with_capacity(self.entries.len());
        let mut refreshed_map = HashMap::with_capacity(self.entries.len());

        for (entry_id, entry) in self.entry_ids.iter().copied().zip(self.entries.iter()) {
            let embedding = mapper.encode_entry(entry, bias);
            embeddings.push((entry_id, embedding));
            refreshed_map.insert(entry_id, entry.clone());
        }

        self.id_to_entry = refreshed_map;

        let mut index = SoftIndex::new(embed_dim);
        for (id, emb) in embeddings {
            if let Err(err) = index.add(id, emb) {
                tracing::warn!(
                    "Failed to insert embedding {id:?} into soft index; entry will be skipped: {}",
                    err
                );
            }
        }

        index.build();
        self.soft_index = Some(index);
        self.evictions_since_rebuild = 0;

        if self.config.use_hnsw {
            if let Err(err) = self.rebuild_semantic_index_internal() {
                tracing::warn!(
                    "Failed to rebuild semantic HNSW index; continuing with linear fallback: {}",
                    err
                );
                if let Some(budget) = self.memory_budget.as_mut() {
                    budget.set_ann_overhead_factor(1.0);
                }
            }
        } else {
            self.hnsw_index = None;
        }
    }

    /// Retrieve dreams using soft index with hybrid scoring (Phase 4)
    ///
    /// # Arguments
    /// * `query` - QuerySignature specifying target chromatic features and hints
    /// * `k` - Number of dreams to retrieve
    /// * `weights` - RetrievalWeights for hybrid scoring (α·sim + β·util + γ·class + MMR)
    /// * `mode` - Similarity metric (Cosine or Euclidean)
    /// * `mapper` - EmbeddingMapper to encode the query
    /// * `bias` - Optional BiasProfile for query conditioning
    ///
    /// # Returns
    /// Vec<DreamEntry> ordered by hybrid score (descending)
    ///
    /// Uses HNSW index if available (O(log n)), otherwise uses SoftIndex (O(n)).
    /// Returns empty vec if no index built. Call `rebuild_soft_index` first.
    pub fn retrieve_soft(
        &self,
        query: &QuerySignature,
        k: usize,
        weights: &RetrievalWeights,
        mode: Similarity,
        mapper: &EmbeddingMapper,
        bias: Option<&crate::dream::bias::BiasProfile>,
    ) -> Vec<DreamEntry> {
        // Encode query (with caching)
        let query_embedding = mapper.encode_query(query, bias);

        // Get initial k-NN from either SoftIndex or (dimension-compatible) HNSW
        let hits = if let Some(index) = &self.soft_index {
            // Fall back to linear SoftIndex
            match index.query(&query_embedding, k, mode) {
                Ok(results) => results,
                Err(err) => {
                    tracing::warn!(
                        "Soft index retrieval failed; returning empty result set: {}",
                        err
                    );
                    Vec::new()
                }
            }
        } else if let Some(hnsw) = &self.hnsw_index {
            if hnsw.dim() == query_embedding.len() {
                match hnsw.query(&query_embedding, k, mode) {
                    Ok(results) => results,
                    Err(err) => {
                        tracing::warn!(
                            "HNSW retrieval failed; returning empty result set: {}",
                            err
                        );
                        Vec::new()
                    }
                }
            } else {
                tracing::warn!(
                    "Skipping HNSW retrieval due to dimension mismatch (index={} query={})",
                    hnsw.dim(),
                    query_embedding.len()
                );
                Vec::new()
            }
        } else {
            // No index built yet
            return Vec::new();
        };

        // Apply hybrid scoring with MMR diversity
        let reranked = rerank_hybrid(&hits, weights, &self.id_to_entry, query.class_hint);

        // Map EntryIds back to DreamEntries
        reranked
            .into_iter()
            .filter_map(|(id, _score)| self.id_to_entry.get(&id).cloned())
            .collect()
    }

    /// Retrieve dreams using semantic UMS search (Phase 7 / Phase 3)
    ///
    /// Primary low-latency semantic search utilizing HNSW index for speed.
    ///
    /// # Arguments
    /// * `query_tensor` - ChromaticTensor to search for
    ///
    /// # Returns
    /// Vec<EntryId> of top-K most similar entries (K from config.retrieval_limit)
    ///
    /// # Implementation
    /// 1. Converts query_tensor to UMS vector (512D)
    /// 2. Uses HNSW index for O(log n) search if available
    /// 3. Falls back to linear search if HNSW unavailable or fails
    /// 4. Filters results to active entries (prevents ghost nodes)
    pub fn retrieve_semantic(&self, query_tensor: &ChromaticTensor) -> DreamResult<Vec<EntryId>> {
        if self.entries.is_empty() {
            return Ok(Vec::new());
        }

        let limit = self.config.retrieval_limit;
        if limit == 0 {
            return Ok(Vec::new());
        }

        let ums = encode_to_ums(&self.modality_mapper, query_tensor);
        let query_embedding = ums.components();
        let k = limit.min(self.entries.len());

        if let Some(hnsw) = &self.hnsw_index {
            match hnsw.query(query_embedding, k, Similarity::Cosine) {
                Ok(results) => {
                    let filtered = self.filter_active_results(results, k);
                    if !filtered.is_empty() {
                        return Ok(filtered);
                    }

                    let err = DreamError::index_corrupted(
                        "HNSW semantic search returned only inactive nodes",
                    );
                    tracing::warn!("{}; falling back to linear semantic search", err);
                }
                Err(err) => {
                    tracing::warn!(
                        "HNSW semantic search failed; falling back to linear search: {}",
                        err
                    );
                }
            }
        } else {
            let err = DreamError::index_not_built("semantic HNSW search");
            tracing::warn!(
                "HNSW semantic index unavailable; falling back to linear search: {}",
                err
            );
        }

        self.linear_semantic_search(query_embedding, k)
    }

    /// Retrieve dreams by category with UMS ranking (Phase 7 / Phase 3 Hybrid Retrieval)
    ///
    /// High-fidelity retrieval combining discrete hue category partitioning
    /// with continuous UMS vector similarity ranking.
    ///
    /// # Arguments
    /// * `query_tensor` - ChromaticTensor to search for
    /// * `k` - Number of results to return
    ///
    /// # Returns
    /// Vec<DreamEntry> of top-K most similar entries within the same hue category
    ///
    /// # Implementation
    /// 1. Converts query_tensor to UMS vector and extracts hue category
    /// 2. Filters pool to entries matching query's hue category [0-11]
    /// 3. Ranks filtered entries by cosine similarity in UMS space
    /// 4. Returns top-K entries sorted by similarity (descending)
    ///
    /// # Performance
    /// - Category filtering: O(n) where n = total entries
    /// - UMS ranking: O(m log m) where m = entries in category (~n/12)
    /// - More efficient than full pool search when categories are balanced
    pub fn retrieve_hybrid(&self, query_tensor: &ChromaticTensor, k: usize) -> Vec<DreamEntry> {
        if self.entries.is_empty() || k == 0 {
            return Vec::new();
        }

        // Convert query to UMS and extract hue category
        let ums = encode_to_ums(&self.modality_mapper, query_tensor);
        let query_ums = ums.components();
        let query_rgb = query_tensor.mean_rgb();
        let query_hue = DreamEntry::rgb_to_hue(query_rgb);
        let query_category = self.modality_mapper.map_hue_to_category(query_hue);

        // Filter by category and rank by UMS similarity
        let mut scored: Vec<(f32, &DreamEntry)> = self
            .entries
            .iter()
            .filter(|entry| entry.hue_category == query_category)
            .map(|entry| {
                let similarity = Self::cosine_similarity_dense(query_ums, &entry.ums_vector);
                (similarity, entry)
            })
            .collect();

        // Sort by similarity descending
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        // Return top-K entries
        scored
            .into_iter()
            .take(k)
            .map(|(_, entry)| entry.clone())
            .collect()
    }

    /// Retrieve dreams by specific category with UMS ranking (Phase 7 / Phase 3)
    ///
    /// Allows explicit category selection for fine-grained control.
    ///
    /// # Arguments
    /// * `target_category` - Hue category index [0-11] to search within
    /// * `query_ums` - UMS vector to rank against
    /// * `k` - Number of results to return
    ///
    /// # Returns
    /// Vec<DreamEntry> of top-K most similar entries within the specified category
    pub fn retrieve_by_category(
        &self,
        target_category: usize,
        query_ums: &[f32],
        k: usize,
    ) -> Vec<DreamEntry> {
        if self.entries.is_empty() || k == 0 {
            return Vec::new();
        }

        // Validate category
        if target_category >= 12 {
            tracing::warn!(
                "Invalid category {} (must be 0-11); returning empty results",
                target_category
            );
            return Vec::new();
        }

        // Filter by category and rank by UMS similarity
        let mut scored: Vec<(f32, &DreamEntry)> = self
            .entries
            .iter()
            .filter(|entry| entry.hue_category == target_category)
            .map(|entry| {
                let similarity = Self::cosine_similarity_dense(query_ums, &entry.ums_vector);
                (similarity, entry)
            })
            .collect();

        // Sort by similarity descending
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        // Return top-K entries
        scored
            .into_iter()
            .take(k)
            .map(|(_, entry)| entry.clone())
            .collect()
    }

    /// Check if any index is built (Phase 4)
    pub fn has_soft_index(&self) -> bool {
        self.soft_index.is_some() || self.hnsw_index.is_some()
    }

    /// Get number of entries in the active index (Phase 4)
    pub fn soft_index_size(&self) -> usize {
        if let Some(hnsw) = &self.hnsw_index {
            hnsw.len()
        } else {
            self.soft_index.as_ref().map_or(0, |idx| idx.len())
        }
    }
}

impl Checkpointable for SimpleDreamPool {
    fn save_checkpoint<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), CheckpointError> {
        let entries: Vec<DreamEntryCheckpoint> = self
            .entries
            .iter()
            .map(DreamEntryCheckpoint::capture)
            .collect::<Result<_, _>>()?;

        let mut id_map: Vec<(EntryId, usize)> = self
            .entry_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, entry_id)| (entry_id, idx))
            .collect();
        id_map.sort_by_key(|(entry_id, _)| entry_id.as_u128());

        let soft_index = self.soft_index.as_ref().map(|index| index.snapshot());
        let hnsw_index = if let Some(index) = &self.hnsw_index {
            Some(index.snapshot()?)
        } else {
            None
        };
        let memory_budget = self.memory_budget.as_ref().map(|budget| budget.snapshot());

        let snapshot = SimpleDreamPoolCheckpoint {
            version: POOL_CHECKPOINT_VERSION,
            config: self.config.clone(),
            entries,
            entry_ids: self.entry_ids.iter().copied().collect(),
            id_map,
            soft_index,
            hnsw_index,
            memory_budget,
            evictions_since_rebuild: self.evictions_since_rebuild,
        };

        Self::write_snapshot(&snapshot, path)
    }

    fn load_checkpoint<P: AsRef<std::path::Path>>(path: P) -> Result<Self, CheckpointError> {
        let snapshot: SimpleDreamPoolCheckpoint = Self::read_snapshot(path)?;
        if snapshot.version != POOL_CHECKPOINT_VERSION {
            return Err(CheckpointError::VersionMismatch {
                expected: POOL_CHECKPOINT_VERSION,
                found: snapshot.version,
            });
        }

        if snapshot.entries.len() != snapshot.entry_ids.len() {
            return Err(CheckpointError::InvalidFormat(
                "Entry ID list length does not match entries".to_string(),
            ));
        }

        if snapshot.id_map.len() != snapshot.entry_ids.len() {
            return Err(CheckpointError::InvalidFormat(
                "id_map length does not match stored entry IDs".to_string(),
            ));
        }

        let SimpleDreamPoolCheckpoint {
            version: _,
            config,
            entries,
            entry_ids,
            id_map,
            soft_index,
            hnsw_index,
            memory_budget,
            evictions_since_rebuild,
        } = snapshot;

        let mut pool = SimpleDreamPool::new(config);
        let concrete_entries: Vec<DreamEntry> = entries
            .into_iter()
            .map(DreamEntryCheckpoint::into_entry)
            .collect::<Result<_, _>>()?;
        pool.entries = VecDeque::from(concrete_entries);
        pool.entry_ids = VecDeque::from(entry_ids);

        let mut id_to_entry = HashMap::with_capacity(pool.entries.len());
        for (entry_id, index) in id_map.into_iter() {
            if index >= pool.entries.len() {
                return Err(CheckpointError::InvalidFormat(
                    "id_map index out of range".to_string(),
                ));
            }
            let entry = pool.entries.get(index).cloned().ok_or_else(|| {
                CheckpointError::InvalidFormat(
                    "Failed to resolve entry for id_map index".to_string(),
                )
            })?;
            if id_to_entry.insert(entry_id, entry).is_some() {
                return Err(CheckpointError::InvalidFormat(format!(
                    "Duplicate entry {entry_id} detected while restoring id_map"
                )));
            }
        }

        let id_set: HashSet<EntryId> = pool.entry_ids.iter().copied().collect();
        if !id_to_entry.keys().all(|entry_id| id_set.contains(entry_id)) {
            return Err(CheckpointError::InvalidFormat(
                "id_map contains entries missing from entry_ids".to_string(),
            ));
        }

        pool.id_to_entry = id_to_entry;
        pool.soft_index = soft_index.map(SoftIndex::from_snapshot);
        pool.hnsw_index = match hnsw_index {
            Some(index_snapshot) => Some(HnswIndex::from_snapshot(index_snapshot)?),
            None => None,
        };
        pool.memory_budget = match memory_budget {
            Some(budget_snapshot) => Some(
                MemoryBudget::from_snapshot(budget_snapshot)
                    .map_err(CheckpointError::InvalidFormat)?,
            ),
            None => None,
        };
        pool.evictions_since_rebuild = evictions_since_rebuild;

        Ok(pool)
    }
}

/// Pool statistics for monitoring and analysis
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub count: usize,
    pub mean_coherence: f64,
    pub mean_energy: f64,
    pub mean_violation: f64,
}

/// Compute cosine similarity between two 3D vectors
///
/// Returns a value in [-1, 1] where 1 means identical direction,
/// 0 means orthogonal, and -1 means opposite direction.
fn cosine_similarity(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mag_a = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let mag_b = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        assert!((cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);

        // Orthogonal vectors
        assert!((cosine_similarity(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]) - 0.0).abs() < 1e-6);

        // Opposite vectors
        assert!((cosine_similarity(&[1.0, 0.0, 0.0], &[-1.0, 0.0, 0.0]) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_pool_add_and_retrieve() {
        use crate::solver::SolverResult;
        use crate::ChromaticTensor;
        use serde_json::json;

        let config = PoolConfig {
            max_size: 5,
            coherence_threshold: 0.5,
            retrieval_limit: 3,
            use_hnsw: false,        // Use linear index for simple tests
            memory_budget_mb: None, // No memory limit for simple tests
        };
        let mut pool = SimpleDreamPool::new(config);

        // Add some dreams with different signatures
        let tensor1 = ChromaticTensor::from_seed(42, 8, 8, 2);
        let result1 = SolverResult {
            energy: 0.1,
            coherence: 0.9,
            violation: 0.05,
            grad: None,
            mask: None,
            meta: json!({}),
        };

        assert!(pool.add_if_coherent(tensor1.clone(), result1));
        assert_eq!(pool.len(), 1);

        // Retrieve similar to tensor1's signature
        let similar = pool.retrieve_similar(&tensor1.mean_rgb(), 1);
        assert_eq!(similar.len(), 1);
    }

    #[test]
    fn test_pool_capacity() {
        use crate::solver::SolverResult;
        use crate::ChromaticTensor;
        use serde_json::json;

        let config = PoolConfig {
            max_size: 3,
            coherence_threshold: 0.0,
            retrieval_limit: 3,
            use_hnsw: false,
            memory_budget_mb: None,
        };
        let mut pool = SimpleDreamPool::new(config);

        // Add 5 dreams to a pool with max_size = 3
        for i in 0..5 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.9,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            pool.add(tensor, result);
        }

        // Should only have 3 dreams (oldest 2 evicted)
        assert_eq!(pool.len(), 3);
    }

    #[test]
    fn test_class_aware_retrieval() {
        use crate::data::ColorClass;
        use crate::solver::SolverResult;
        use crate::ChromaticTensor;
        use serde_json::json;

        let config = PoolConfig {
            max_size: 20,
            coherence_threshold: 0.5,
            retrieval_limit: 5,
            use_hnsw: false,
            memory_budget_mb: None,
        };
        let mut pool = SimpleDreamPool::new(config);

        // Add dreams with different class labels
        for i in 0..5 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.8,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            pool.add_with_class(tensor, result, ColorClass::Red);
        }

        for i in 5..10 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.8,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            pool.add_with_class(tensor, result, ColorClass::Blue);
        }

        assert_eq!(pool.len(), 10);

        // Retrieve only Red class dreams
        let query = [1.0, 0.0, 0.0];
        let red_dreams = pool.retrieve_similar_class(&query, ColorClass::Red, 3);
        assert_eq!(red_dreams.len(), 3);
        assert!(red_dreams
            .iter()
            .all(|d| d.class_label == Some(ColorClass::Red)));

        // Retrieve only Blue class dreams
        let blue_dreams = pool.retrieve_similar_class(&query, ColorClass::Blue, 3);
        assert_eq!(blue_dreams.len(), 3);
        assert!(blue_dreams
            .iter()
            .all(|d| d.class_label == Some(ColorClass::Blue)));
    }

    #[test]
    fn test_balanced_retrieval() {
        use crate::data::ColorClass;
        use crate::solver::SolverResult;
        use crate::ChromaticTensor;
        use serde_json::json;

        let config = PoolConfig::default();
        let mut pool = SimpleDreamPool::new(config);

        // Add dreams from three classes
        for i in 0..5 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.8,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            pool.add_with_class(tensor, result, ColorClass::Red);
        }

        for i in 5..10 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.8,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            pool.add_with_class(tensor, result, ColorClass::Green);
        }

        for i in 10..15 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.8,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            pool.add_with_class(tensor, result, ColorClass::Blue);
        }

        // Retrieve balanced across 3 classes, 2 per class
        let query = [0.5, 0.5, 0.5];
        let classes = vec![ColorClass::Red, ColorClass::Green, ColorClass::Blue];
        let balanced = pool.retrieve_balanced(&query, &classes, 2);

        assert_eq!(balanced.len(), 6); // 2 * 3 classes

        // Count per class
        let red_count = balanced
            .iter()
            .filter(|d| d.class_label == Some(ColorClass::Red))
            .count();
        let green_count = balanced
            .iter()
            .filter(|d| d.class_label == Some(ColorClass::Green))
            .count();
        let blue_count = balanced
            .iter()
            .filter(|d| d.class_label == Some(ColorClass::Blue))
            .count();

        assert_eq!(red_count, 2);
        assert_eq!(green_count, 2);
        assert_eq!(blue_count, 2);
    }

    #[test]
    fn test_utility_retrieval() {
        use crate::solver::SolverResult;
        use crate::ChromaticTensor;
        use serde_json::json;

        let config = PoolConfig {
            max_size: 10,
            coherence_threshold: 0.0,
            retrieval_limit: 5,
            use_hnsw: false,
            memory_budget_mb: None,
        };
        let mut pool = SimpleDreamPool::new(config);

        // Add dreams with varying utility
        for i in 0..5 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.8,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            let mut entry = DreamEntry::new(tensor, result);
            entry.set_utility(0.9); // High utility

            pool.entries.push_back(entry);
        }

        for i in 5..10 {
            let tensor = ChromaticTensor::from_seed(i, 8, 8, 2);
            let result = SolverResult {
                energy: 0.1,
                coherence: 0.8,
                violation: 0.05,
                grad: None,
                mask: None,
                meta: json!({}),
            };
            let mut entry = DreamEntry::new(tensor, result);
            entry.set_utility(0.1); // Low utility

            pool.entries.push_back(entry);
        }

        // Retrieve only high-utility dreams (utility >= 0.5)
        let query = [0.5, 0.5, 0.5];
        let high_utility = pool.retrieve_by_utility(&query, 10, 0.5);

        assert_eq!(high_utility.len(), 5);
        assert!(high_utility.iter().all(|d| d.utility.unwrap_or(0.0) >= 0.5));
    }

    #[test]
    fn test_checkpoint_roundtrip_preserves_indices() {
        use crate::dream::embedding::EmbeddingMapper;
        use crate::solver::SolverResult;
        use crate::ChromaticTensor;
        use serde_json::json;
        use uuid::Uuid;

        let mut config = PoolConfig::default();
        config.use_hnsw = true;
        config.max_size = 16;
        let mut pool = SimpleDreamPool::new(config.clone());

        let result = SolverResult {
            energy: 0.1,
            coherence: 0.95,
            violation: 0.02,
            grad: None,
            mask: None,
            meta: json!({}),
        };

        for seed in 0..4 {
            let tensor = ChromaticTensor::from_seed(seed, 8, 8, 2);
            assert!(pool.add_if_coherent(tensor, result.clone()));
        }

        let mapper = EmbeddingMapper::new(64);
        pool.rebuild_soft_index(&mapper, None);
        assert!(pool.soft_index.is_some());
        assert!(pool.hnsw_index.as_ref().map(|idx| idx.len()).unwrap_or(0) > 0);

        let checkpoint_path =
            std::env::temp_dir().join(format!("simple_pool-checkpoint-{}.bin", Uuid::new_v4()));

        pool.save_checkpoint(&checkpoint_path).unwrap();
        let restored = SimpleDreamPool::load_checkpoint(&checkpoint_path).unwrap();
        let _ = std::fs::remove_file(&checkpoint_path);

        assert_eq!(restored.len(), pool.len());
        assert_eq!(
            restored
                .hnsw_index
                .as_ref()
                .map(|idx| idx.len())
                .unwrap_or(0),
            pool.hnsw_index.as_ref().map(|idx| idx.len()).unwrap_or(0)
        );
        assert_eq!(
            restored.evictions_since_rebuild,
            pool.evictions_since_rebuild
        );
        assert_eq!(restored.entry_ids.len(), pool.entry_ids.len());
        assert_eq!(restored.id_to_entry.len(), pool.id_to_entry.len());
        assert!(restored.soft_index.is_some());
    }
}
