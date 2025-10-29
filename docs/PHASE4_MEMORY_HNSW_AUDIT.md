# Phase 4 Memory + HNSW Audit (QIA Core)

## Scope & Methodology
- **Modules reviewed:** `src/dream/memory.rs`, `src/dream/hnsw_index.rs`, `src/dream/simple_pool.rs`, `src/dream/error.rs`, `src/dream/diversity.rs`, `src/spectral/fft.rs`.
- **Focus areas:** ANN-aware memory budgeting, HNSW integrity, error propagation, numerical determinism, and wiring between `SimpleDreamPool`, HNSW, and `MemoryBudget`.
- **Procedure:** Static analysis with line-level tracing, test coverage review, and verification of determinism constraints for FFT/MMR routines.

## Findings & Fixes

### 1. Memory Budget Enforcement (`src/dream/memory.rs`)
- `MemoryBudget::set_ann_overhead_factor` now clamps caller input to `[1.0, 8.0]`, preventing runaway scaling while still modeling HNSW≈2× overhead (L76-L85).
- `adjusted_usage_bytes` and `needs_eviction` evaluate eviction decisions against ANN-adjusted bytes and zero-budget guard rails, eliminating panic paths when `max_bytes == 0` (L88-L179).
- `calculate_eviction_count` consumes the same adjusted usage and enforces `min(entry_count)` to avoid underflow when the pool shrinks to zero (L199-L214).
- Tests cover ANN scaling, zero-budget handling, and eviction counts, giving regression protection for all branches touched by Phase 4 (L341-L484).
- **Recommendation:** Phase 7 callers should drop the deprecated free function wrapper and invoke `MemoryBudget::calculate_eviction_count` directly so byte math cannot diverge.

### 2. HNSW Stability & ID Synchronization (`src/dream/hnsw_index.rs`)
- `add` validates embedding dimensionality before mutating `id_map`, so mismatched vectors cannot desynchronize the index (L103-L126).
- `build` refuses to run when `id_map.len() != pending_embeddings.len()` and resets stale graphs before inserting, ensuring exact parity with HNSW internals (L138-L190).
- `search` converts distances to similarity with deterministic clamping and reports non-finite values or missing `id_map` slots as `DreamError::IndexCorrupted`, blocking undefined behaviour (L210-L276).
- Regression tests confirm detection of tampering (`test_hnsw_build_detects_mismatch`, `test_hnsw_search_reports_id_map_desync`) and enforce both cosine and L2 happy paths (L347-L500).
- **Recommendation:** Keep `id_map` private; expose synchronization helpers (e.g., `fn rebuild_from(entries: impl Iterator<Item=(EntryId, Vec<f32>)>)`) instead of a public field to avoid concurrent mutation hazards.

### 3. Error Propagation Coverage (`src/dream/error.rs`, `src/dream/simple_pool.rs`)
- `DreamError` enumerates the Phase 4 surface area and exposes builder helpers, enabling `HnswIndex`/`SoftIndex` to return `DreamResult` rather than panic (`DreamError` definition and constructors L11-L206).
- Unit tests assert display/equality behaviour for the most common variants, ensuring error messaging stays descriptive (L209-L269).
- `SimpleDreamPool::rebuild_soft_index` toggles the memory multiplier before building HNSW and gracefully falls back to the linear index when ANN construction fails (L678-L741).
- `retrieve_soft` currently erases ANN/SoftIndex errors via `unwrap_or_else(|_| Vec::new())`, which preserves uptime but hides actionable diagnostics (L772-L779).
  - **Fix in place:** HNSW build failures now automatically drop back to the linear index while restoring the memory multiplier (L701-L729).
  - **Follow-up:** surface retrieval errors via a `Result<Vec<DreamEntry>>` return or structured logging once the caller flow is ready to handle transient ANN faults.

### 4. Numerical Determinism (FFT, MMR)
- Spectral feature extraction uses explicit window coefficients and sequential FFT passes with deterministic `rustfft` planners, avoiding parallel scheduling variance (FFT pipeline L108-L288 in `src/spectral/fft.rs`).
- Fast MMR selection samples the already-selected list via deterministic `step_by` iteration, so approximate diversity scores are architecture-stable without randomness (L303-L336 in `src/dream/diversity.rs`).
- **Recommendation:** Document the assumed floating-point tolerances (e.g., `abs() < 1e-6` in tests) once Phase 7 introduces cross-architecture regression harnesses.

### 5. Integration Wiring (`src/dream/simple_pool.rs`)
- Constructor-level budget provisioning now applies the ANN overhead multiplier whenever `use_hnsw` is enabled, keeping eviction math aligned with the active index (L180-L199).
- During rebuild, the pool encodes all embeddings once, restores `id_to_entry`/`entry_ids`, and only commits the HNSW instance after a successful build, preventing partially-built graphs from leaking (L686-L741).
- Memory budget multiplier reverts to `1.0` when the HNSW path fails, so subsequent linear mode operations do not over-evict (L724-L729).
- **Recommendation:** prefer a builder-style `SimpleDreamPool::rebuild_soft_index_with(mapper, bias) -> DreamResult<()>` that returns rich errors instead of toggling `Option` fields implicitly. This keeps state transitions explicit and reduces the risk of stale handles when future concurrency hooks arrive.

## Outstanding Actions Before Phase 7
1. Replace silent `unwrap_or_else` fallbacks in `retrieve_soft` with structured logging or a `Result` to complete the 80% error propagation requirement.
2. Update eviction callers to use the method-based `calculate_eviction_count` and log evicted bytes for deterministic replay.
3. Add integration tests that rebuild HNSW while memory budgets fluctuate to validate the ANN overhead toggling end-to-end.

