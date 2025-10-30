# Phase 3: Archive Finalization (Retrieval)

## 🎯 Goal
Implement the core mission of the Chromatic Semantic Archive (CSA): provide reliable, low-latency Semantic Retrieval by stabilizing the HNSW index and integrating the search function.

---

### Subphase 3.A: HNSW Index Stabilization

**Focus:** Modifying the HNSW index to support efficient incremental updates, avoiding unnecessary full rebuilds.

1.  **Implement HNSW Incremental Add:** Modify the `HnswIndex::add()` function (or an equivalent entry point) to support **true incremental insertion**. The HNSW library should be used to add the new embedding directly to the existing graph structure without forcing the complete destruction and rebuild of the index. This is a fundamental change from the design flaw identified earlier.
2.  **Fix HNSW Ghost Nodes:** Review the logic within `evict_n_entries` or an associated HNSW handler to ensure that when an entry is evicted, its corresponding node is correctly **removed or marked as deleted** within the HNSW graph structure itself (if the library supports it, otherwise, ensure the index is aware of the deletion).

---

### Subphase 3.B: Semantic Retrieval Implementation

**Focus:** Implementing the primary function for low-latency similarity search.

1.  **Implement Retrieval Function:** Create the primary retrieval function, e.g., `fn retrieve_semantic(&self, query_tensor: &ChromaticTensor) -> DreamResult<Vec<EntryId>>` within the `SimpleDreamPool` or a new `CSA` façade.
2.  **Query Path:** This function must deterministically route the query:
    * **Tensor $\rightarrow$ UMS:** Convert the input `query_tensor` into a normalized **UMS Vector**.
    * **UMS $\rightarrow$ HNSW Search:** Use the UMS vector to query the HNSW index for the **top-K** most similar archived entries.
    * **Fallback:** Implement a deterministic fallback to the linear index if the HNSW index is unavailable or fails, logging the failure as a **DreamError**.
3.  **Result Filtering:** Ensure the returned results (which are `EntryId`s) are filtered against the pool's internal `id_to_entry` map to prevent returning **"ghost nodes"** or entries that were recently evicted.

---

### Subphase 3.C: Final Audit and Project Alignment

**Focus:** Finalizing documentation and ensuring the system is configured for the mission.

1.  **Documentation Update:** Update the internal documentation (e.g., in `PoolConfig`) to reflect the **Hybrid Approach (Option C)**: the default setting should be `use_hnsw: false` (Linear Index), making the optimization **opt-in**.
2.  **Project Alignment:** Add documentation to the relevant module (e.g., `HnswIndex`) outlining **"When to use HNSW"**: specifically, when the pool size is $\mathbf{>5000}$ entries or when query latency must be $\mathbf{<100}$ milliseconds.