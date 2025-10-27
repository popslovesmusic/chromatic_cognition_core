# **Phase 4 — Continuous Embedding / Soft Indexing (Spec v1)**

## **Objective**

Replace hard, ruley retrieval with a **continuous latent index** so dreams are fetched by **semantic proximity** (cosine) rather than class-only or RGB heuristics—while *using* the new spectral \+ ΔLoss utilities you just landed.

f5a27dbf-a206-4743-9b91-fee7316…

## **Inputs we will reuse (stable)**

* Class-aware pool metadata, MMR diversity, spectral features (FFT), ΔLoss utility, BiasProfile.  
   f5a27dbf-a206-4743-9b91-fee7316…

## **Deliverables (D1–D6)**

**D1. Embedding Mapper**

* `EmbeddingMapper` that fuses: chroma\_signature (3), spectral features (e.g., 7–20 dims), optional class one-hot (\<=10), and utility priors.

* Output: fixed‐dim vector (e.g., `D=64`).

`pub struct EmbeddingMapper { pub d: usize }`  
`impl EmbeddingMapper {`  
    `pub fn new(d: usize) -> Self;`  
    `pub fn encode_entry(&self, e: &DreamEntry, bias: Option<&BiasProfile>) -> Vec<f32>;`  
    `pub fn encode_query(&self, q: &QuerySig, bias: Option<&BiasProfile>) -> Vec<f32>;`  
`}`

(Folds in entropy, band-energy, ΔLoss stats as features.)

**D2. Soft Index**

* In-memory ANN-lite with cosine+euclidean.

`pub struct SoftIndex { dim: usize, ids: Vec<Uuid>, vecs: Vec<Vec<f32>> }`  
`impl SoftIndex {`  
    `pub fn add(&mut self, id: Uuid, v: Vec<f32>);`  
    `pub fn build(&mut self); // optional norms`  
    `pub fn query(&self, v: &[f32], k: usize, mode: Similarity) -> Vec<(Uuid, f32)>;`  
`}`  
`pub enum Similarity { Cosine, Euclidean }`

**D3. Scoring & Diversity (hybrid)**

* Final score \= `α * sim + β * utility + γ * class_match - δ * dup_penalty`

* MMR post-filter stays to ensure spread.

`pub struct RetrievalWeights { pub alpha:f32, pub beta:f32, pub gamma:f32, pub delta:f32, pub lambda:f32 }`  
`pub fn rerank_hybrid(hits:&[(Uuid,f32)], w:&RetrievalWeights) -> Vec<(Uuid,f32)>;`

**D4. Pool Integration**

* New API that routes through embeddings by default:

`impl SimpleDreamPool {`  
    `pub fn rebuild_soft_index(&mut self, mapper:&EmbeddingMapper);`  
    `pub fn retrieve_soft(&self, q:&QuerySig, k:usize, w:&RetrievalWeights, mode:Similarity) -> Vec<DreamEntry>;`  
`}`

**D5. Training Loop Hook**

* `train_with_dreams()` gains `retrieval_mode=Hard|Soft|Hybrid`.

* Mid-epoch **dynamic profile update**: refresh BiasProfile every N steps; call `rebuild_soft_index` if drift \> τ.

**D6. Validation Protocol**

* 3-way study: Baseline (no dreams) vs 3B (class/MMR) vs 4 (soft index).

* Metrics: epochs-to-95%, final accuracy, wall clock, “helpful dream rate” (ΔLoss\<0), and *coverage* (unique dream IDs used).  
   f5a27dbf-a206-4743-9b91-fee7316…

## **Data & Schema Changes**

* `DreamEntry` (add):

`pub struct DreamEntry {`  
  `// ...`  
  `pub embed: Option<Vec<f32>>,    // cached 64D`  
  `pub util_mean: f32,             // from FeedbackRecord aggregation`  
`}`

* BiasProfile remains as a conditioning prior (weights/features).

## **Math (summary)**

* **Embedding**: `z = LayerNorm([rgb3, spectral_k, class_onehot, util_stats]) · W + b` (start linear; keep deterministic).

* **Similarity**: cosine default; euclidean optional.

* **Hybrid score**:  
   `score = α·cos(z_q, z_i) + β·utility_i + γ·1[class(q)=class(i)] − δ·max_sim_selected`

## **Config (engine.toml)**

`[phase4]`  
`embed_dim = 64`  
`similarity = "cosine"        # cosine|euclidean`  
`alpha = 0.65`  
`beta  = 0.20`  
`gamma = 0.10`  
`delta = 0.05`  
`mmr_lambda = 0.7`  
`refresh_interval_steps = 500`  
`drift_threshold = 0.08       # retrain/reindex if bias drift exceeds`

## **Tests (add 10–14)**

* Embedding determinism & shape

* Index add/query correctness

* Cosine vs euclidean parity on trivial cases

* Hybrid rerank monotonicity w.r.t. utility

* MMR still increases dispersion on top-K

* End-to-end: **Soft** beats **3B** on at least one of (epochs, accuracy) on harder task (≥100 classes or injected noise).  
   f5a27dbf-a206-4743-9b91-fee7316…

## **Definition of Done**

* ✅ API compiled, docs updated

* ✅ Soft retrieval integrated & switchable

* ✅ Validation run completed with report:

  * Δ(epochs-to-95%) ≤ −10% **or** Δ(final acc) ≥ \+1.0 pt

  * No \>10% wall-clock regression over 3B at same batch size

  * Coverage ↑ (≥+20% unique dreams consumed)

* ✅ New tests green; total test count increases

## **Migration Plan (1 day)**

1. Branch `phase4-soft-index`.

2. Implement `EmbeddingMapper` (linear, no learned weights).

3. Build `SoftIndex`; backfill `embed` for existing entries.

4. Wire training `retrieval_mode`.

5. Run the 3-way benchmark script and write `PHASE4_VALIDATION.md`.

