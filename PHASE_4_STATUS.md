# Phase 4 — Continuous Embedding / Soft Indexing - Status

**Implementation Date:** 2025-10-27
**Status:** 🚧 IN PROGRESS - 2/6 Deliverables Complete
**Goal:** Replace hard class-based retrieval with continuous semantic embeddings

---

## Progress Summary

### ✅ Completed Deliverables (2/6)

**✅ D1: EmbeddingMapper** (COMPLETE)
- **File:** `src/dream/embedding.rs` (426 lines)
- **Tests:** 9 new tests (all passing)
- **Features:**
  - Fuses RGB(3) + spectral(6) + class_onehot(10) + utility(2) → 64D
  - Layer normalization for stable embeddings
  - Deterministic encoding (no learned weights)
  - QuerySignature with optional hints
- **Key API:**
  - `encode_entry(entry, bias) -> Vec<f32>`
  - `encode_query(query, bias) -> Vec<f32>`

**✅ D2: SoftIndex** (COMPLETE)
- **File:** `src/dream/soft_index.rs` (224 lines)
- **Tests:** 6 new tests (all passing)
- **Features:**
  - In-memory ANN with cosine/euclidean
  - Pre-computed norms for efficiency
  - K-NN retrieval with scoring
- **Key API:**
  - `add(id, vec)` - Add entry
  - `build()` - Pre-compute norms
  - `query(vec, k, mode) -> Vec<(Uuid, f32)>`

### 🚧 Remaining Deliverables (4/6)

**D3: Hybrid Scoring & Diversity** (PENDING)
- **Goal:** Combine similarity + utility + class_match + MMR diversity
- **Formula:** `score = α·sim + β·utility + γ·class_match - δ·dup_penalty`
- **Deliverable:**
  ```rust
  pub struct RetrievalWeights {
      pub alpha: f32,   // Similarity weight
      pub beta: f32,    // Utility weight
      pub gamma: f32,   // Class match weight
      pub delta: f32,   // Duplicate penalty
      pub lambda: f32,  // MMR lambda
  }
  pub fn rerank_hybrid(hits, weights) -> Vec<(Uuid, f32)>;
  ```

**D4: Pool Integration** (PENDING)
- **Goal:** Integrate SoftIndex with SimpleDreamPool
- **Deliverable:**
  ```rust
  impl SimpleDreamPool {
      pub fn rebuild_soft_index(&mut self, mapper: &EmbeddingMapper);
      pub fn retrieve_soft(&self, query, k, weights, mode) -> Vec<DreamEntry>;
  }
  ```
- **Steps:**
  1. Add `soft_index: Option<SoftIndex>` field to SimpleDreamPool
  2. Add `id_to_entry: HashMap<EntryId, DreamEntry>` mapping
  3. Implement `rebuild_soft_index()` to encode all entries
  4. Implement `retrieve_soft()` using hybrid scoring

**D5: Training Loop Hook** (PENDING)
- **Goal:** Add retrieval mode switching to training
- **Deliverable:**
  ```rust
  pub enum RetrievalMode {
      Hard,    // Phase 3B class-aware
      Soft,    // Phase 4 embedding-based
      Hybrid,  // Combination
  }
  // Update train_with_dreams() signature
  ```
- **Dynamic profile update:** Refresh BiasProfile every N steps

**D6: Validation Protocol** (PENDING)
- **Goal:** 3-way comparison: Baseline vs Phase 3B vs Phase 4
- **Metrics:**
  - Epochs to 95% accuracy
  - Final accuracy
  - Wall clock time
  - Helpful dream rate (ΔLoss < 0)
  - Coverage (unique dream IDs used)
- **Success Criteria:**
  - Δ(epochs-to-95%) ≤ -10% **OR** Δ(final acc) ≥ +1.0 pt
  - No >10% wall-clock regression
  - Coverage ↑ ≥ +20%

---

## Test Status

| Module | Tests | Status |
|--------|-------|--------|
| dream::embedding | 9 | ✅ All passing |
| dream::soft_index | 6 | ✅ All passing |
| **Total Phase 4** | **15** | **✅ 100%** |
| **Project Total** | **89** | **✅ 100%** |

**Test Growth:**
- Before Phase 4: 74 tests
- After D1+D2: 89 tests (+20%)

---

## Code Statistics

| Deliverable | LoC | Status |
|-------------|-----|--------|
| D1: EmbeddingMapper | 426 | ✅ |
| D2: SoftIndex | 224 | ✅ |
| D3: Hybrid Scoring | ~150 | 🚧 |
| D4: Pool Integration | ~200 | 🚧 |
| D5: Training Hook | ~100 | 🚧 |
| D6: Validation | ~400 | 🚧 |
| **Total** | **~1,500** | **33% Complete** |

---

## Dependencies Added

- `uuid = { version = "1.0", features = ["v4"] }` - Unique entry identifiers

---

## Next Steps

### Immediate (D3)

1. Create `src/dream/hybrid_scoring.rs`
2. Implement `RetrievalWeights` struct
3. Implement `rerank_hybrid()` function
4. Add tests for hybrid scoring
5. Test MMR integration with soft index

### After D3 (D4)

1. Add fields to `SimpleDreamPool`:
   - `soft_index: Option<SoftIndex>`
   - `id_to_entry: HashMap<EntryId, DreamEntry>`
2. Implement `rebuild_soft_index(mapper)`
3. Implement `retrieve_soft(query, k, weights, mode)`
4. Add tests for pool integration

### After D4 (D5)

1. Create `RetrievalMode` enum
2. Update `train_with_dreams()` signature
3. Add mode switching logic
4. Implement dynamic profile refresh
5. Add tests for training integration

### After D5 (D6)

1. Create `examples/phase_4_validation.rs`
2. Run 3-way comparison
3. Measure metrics
4. Generate `PHASE_4_VALIDATION.md` report
5. Validate success criteria

---

## Configuration (engine.toml)

```toml
[phase4]
embed_dim = 64
similarity = "cosine"        # cosine|euclidean
alpha = 0.65                 # Similarity weight
beta  = 0.20                 # Utility weight
gamma = 0.10                 # Class match weight
delta = 0.05                 # Duplicate penalty
mmr_lambda = 0.7             # MMR diversity parameter
refresh_interval_steps = 500 # BiasProfile refresh frequency
drift_threshold = 0.08       # Reindex threshold
```

---

## Architecture Diagram

```
PHASE 4: CONTINUOUS EMBEDDING / SOFT INDEXING
─────────────────────────────────────────────

1. Dream Entry
   ├─ Chromatic signature [R, G, B]
   ├─ Spectral features (entropy, bands)
   ├─ Class label (optional)
   └─ Utility score

2. EmbeddingMapper ✅
   ├─ Fuse features → 64D vector
   ├─ Layer normalization
   └─ Deterministic projection

3. SoftIndex ✅
   ├─ Store embeddings with UUIDs
   ├─ Pre-compute norms
   └─ K-NN query (cosine/euclidean)

4. Hybrid Scoring 🚧
   ├─ α·similarity
   ├─ β·utility
   ├─ γ·class_match
   └─ -δ·MMR_penalty

5. SimpleDreamPool Integration 🚧
   ├─ rebuild_soft_index()
   └─ retrieve_soft()

6. Training Loop 🚧
   ├─ RetrievalMode switch
   └─ Dynamic profile refresh

7. Validation 🚧
   └─ 3-way comparison study
```

---

## Key Innovations

### 1. Continuous Semantic Space
- **Before (Phase 3B):** Hard class boundaries, cosine on RGB only
- **After (Phase 4):** Continuous 64D latent space with fused features
- **Benefit:** Smooth interpolation between similar dreams regardless of class

### 2. Multi-Feature Fusion
- **RGB:** Base chromatic signature
- **Spectral:** Frequency-domain patterns (entropy, bands)
- **Class:** Soft conditioning via one-hot
- **Utility:** Data-driven quality signal

### 3. Flexible Similarity
- **Cosine:** Direction-based (good for normalized features)
- **Euclidean:** Distance-based (good for magnitude-sensitive features)
- **Hybrid:** Combine with utility and diversity

### 4. Deterministic Encoding
- No learned weights (yet)
- Simple linear projection + layer norm
- Reproducible and debuggable
- Foundation for future learned embeddings

---

## Definition of Done

- [x] D1: EmbeddingMapper implemented and tested
- [x] D2: SoftIndex implemented and tested
- [ ] D3: Hybrid scoring implemented and tested
- [ ] D4: Pool integration complete
- [ ] D5: Training loop updated
- [ ] D6: Validation experiment run with passing criteria
- [ ] All tests green (target: +10-14 tests)
- [ ] Documentation updated

**Current Progress:** 33% complete (2/6 deliverables)

---

## Estimated Remaining Work

- **D3 (Hybrid Scoring):** 2-3 hours
- **D4 (Pool Integration):** 3-4 hours
- **D5 (Training Hook):** 2-3 hours
- **D6 (Validation):** 4-5 hours (including experiment runtime)

**Total:** ~11-15 hours remaining

---

## Conclusion

Phase 4 is off to a strong start with solid foundations:
- ✅ EmbeddingMapper provides robust feature fusion
- ✅ SoftIndex enables efficient semantic retrieval
- 🚧 Remaining work focuses on integration and validation

The architecture is clean, well-tested, and ready for the hybrid scoring layer that will tie everything together.

**Status:** 🟢 ON TRACK
