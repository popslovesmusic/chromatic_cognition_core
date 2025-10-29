# Phase 1: Structural Foundation & Bug Fixes

## 🎯 Goal
Resolve critical architectural debt and refactor core data structures to align with the final $3 \times 12 \times 12 \times 3$ processing unit, ensuring stable index management.

## 🛠️ Instructions for Coding Agent

### I. Critical Pool Refactoring (src/dream/simple_pool.rs)

1.  **Unify Add Methods:** Extract and unify the common logic from `add()`, `add_if_coherent()`, and `add_with_class()` into a single, private, entry-accepting function: `fn internal_add(&mut self, entry: DreamEntry, entry_size: usize)`.

2.  **Fix Index Thrashing (CRITICAL BUG FIX):**
    * Add a counter field: `evictions_since_rebuild: usize`.
    * In `fn evict_n_entries()`, increment this counter by `count`.
    * **REMOVE** the unconditional index destruction logic (`self.hnsw_index = None; self.soft_index = None;`).
    * **Replace** it with a conditional check: Only call the index invalidation logic (`self.invalidate_indices()`) if `self.evictions_since_rebuild > self.entries.len() / 10` (i.e., after 10% pool change). Reset the counter afterward.

3.  **Clean Hot Path:** Remove all existing `tracing::warn!()` calls related to HNSW index mutation or eviction counts from the `evict_n_entries` and `add_if_coherent` functions.

### II. Data Structure Alignment

1.  **Tensor Size Update:** Locate the primary tensor definition structure (likely in `src/tensor/chromatic_tensor.rs` or a related config) and update its expected dimensions to $\mathbf{[3, 12, 12, 3]}$ to reflect the final processing unit.

2.  **Memory Budget Alignment:** Ensure the `MemoryBudget` module's usage calculations correctly handle the size estimation for the new $\mathbf{3 \times 12 \times 12 \times 3}$ tensor structure.

### III. Verification

* Ensure all unit tests remain passing (196/196).
* Verify that `cargo test` no longer emits warnings about unused variables or complex logging in the hot path.