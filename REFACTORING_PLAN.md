# Dream Module Refactoring Plan

**Date:** 2025-10-27
**Goal:** Address two architectural concerns:
1. `simple_pool.rs` is 876 lines with mixed concerns
2. Dream module exports 25+ public items

---

## Issue 1: simple_pool.rs Analysis

### Current Structure (876 lines)

```
Lines 1-91:    DreamEntry struct + impl (91 lines)
Lines 92-111:  PoolConfig struct + impl (19 lines)
Lines 112-598: SimpleDreamPool impl (486 lines)
  - Storage operations (add, clear, stats)
  - Phase 3B retrieval (similar, class, balanced, utility, diverse)
  - Phase 4 retrieval (rebuild_soft_index, retrieve_soft)
Lines 599-613: PoolStats struct (14 lines)
Lines 614-876: Helper functions + tests (262 lines)
```

### Complexity Analysis

| Concern | Status | Reasoning |
|---------|--------|-----------|
| **Size** | 🟡 Moderate | 876 lines is acceptable for integration module |
| **Cohesion** | 🟡 Mixed | Combines storage + 6 retrieval strategies |
| **Tests** | ✅ Good | 262 lines of tests (30% of file) |
| **Duplication** | ✅ None | No obvious code duplication |

### Decision: **Targeted Refactoring, Not Full Split**

**Rationale:**
- SimpleDreamPool is inherently an integration point (expected to be large)
- All retrieval methods operate on the same data (tight coupling)
- Full split would create artificial boundaries
- 876 lines is manageable (threshold is typically 1,000-1,500)

**Action:** Extract well-defined submodules only where it adds clarity

---

## Refactoring Strategy 1: Extract DreamEntry

**Current:** DreamEntry buried inside simple_pool.rs
**Problem:** DreamEntry is used by multiple modules (embedding, hybrid_scoring)
**Solution:** Move to dedicated file

### Create: `src/dream/pool/entry.rs`

```rust
//! Dream entry data structure for pool storage.

use crate::data::ColorClass;
use crate::solver::SolverResult;
use crate::spectral::SpectralFeatures;
use crate::tensor::ChromaticTensor;
use std::time::SystemTime;

/// A stored dream entry with tensor and evaluation metrics
#[derive(Clone)]
pub struct DreamEntry {
    pub tensor: ChromaticTensor,
    pub result: SolverResult,
    pub chroma_signature: [f32; 3],
    pub class_label: Option<ColorClass>,
    pub utility: Option<f32>,
    pub timestamp: SystemTime,
    pub usage_count: usize,
    pub spectral_features: Option<SpectralFeatures>,
    pub embed: Option<Vec<f32>>,
    pub util_mean: f32,
}

impl DreamEntry {
    // ... methods ...
}
```

**Benefits:**
- ✅ Clearer ownership (entry.rs focuses on single type)
- ✅ Reduces simple_pool.rs by ~100 lines
- ✅ Makes DreamEntry more discoverable

**Breaking Change:** No (re-export from simple_pool for compatibility)

---

## Refactoring Strategy 2: Extract Retrieval Strategies

**Current:** 6 different retrieval methods in SimpleDreamPool
**Problem:** Mixed hard/soft retrieval logic
**Solution:** Organize by retrieval paradigm

### Create: `src/dream/pool/retrieval.rs`

```rust
//! Retrieval strategy implementations for SimpleDreamPool.

use super::entry::DreamEntry;

/// Phase 3B: Hard retrieval strategies
pub(crate) mod hard {
    use super::*;

    pub fn retrieve_similar(
        entries: &[DreamEntry],
        query: &[f32; 3],
        k: usize,
    ) -> Vec<DreamEntry> {
        // ... cosine similarity on RGB ...
    }

    pub fn retrieve_by_class(
        entries: &[DreamEntry],
        query: &[f32; 3],
        target_class: ColorClass,
        k: usize,
    ) -> Vec<DreamEntry> {
        // ... class-filtered retrieval ...
    }

    pub fn retrieve_balanced(
        entries: &[DreamEntry],
        query: &[f32; 3],
        k: usize,
    ) -> Vec<DreamEntry> {
        // ... balanced class distribution ...
    }
}

/// Phase 4: Soft retrieval strategies
pub(crate) mod soft {
    use super::*;
    use crate::dream::{EmbeddingMapper, QuerySignature, RetrievalWeights};

    // Soft retrieval helpers (already in SimpleDreamPool)
}
```

**Benefits:**
- ✅ Clear separation: hard vs soft retrieval
- ✅ Reduces SimpleDreamPool impl block size
- ✅ Easier to test strategies in isolation

**Breaking Change:** No (private module, SimpleDreamPool delegates)

---

## Refactoring Strategy 3: Extract Configuration

### Create: `src/dream/pool/config.rs`

```rust
//! Configuration structures for dream pool.

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub max_size: usize,
    pub coherence_threshold: f64,
    pub retrieval_limit: usize,
}

impl Default for PoolConfig { /* ... */ }

#[derive(Debug, Clone)]
pub struct PoolStats {
    pub count: usize,
    pub mean_coherence: f64,
    pub mean_energy: f64,
    pub mean_violation: f64,
}
```

**Benefits:**
- ✅ Configuration grouped logically
- ✅ Reduces simple_pool.rs by ~30 lines
- ✅ Easier to extend with new config options

---

## Final Module Structure (After Refactoring)

```
src/dream/
├── mod.rs                      (25 lines - API exports)
├── pool/
│   ├── mod.rs                  (50 lines - re-exports)
│   ├── entry.rs                (100 lines - DreamEntry)
│   ├── config.rs               (40 lines - PoolConfig, PoolStats)
│   ├── retrieval.rs            (200 lines - retrieval strategies)
│   └── pool.rs                 (400 lines - SimpleDreamPool core)
├── embedding.rs                (411 lines)
├── soft_index.rs               (219 lines)
├── hybrid_scoring.rs           (411 lines)
├── retrieval_mode.rs           (79 lines)
├── bias.rs                     (495 lines)
├── diversity.rs                (457 lines)
├── analysis.rs                 (307 lines)
└── experiment.rs               (365 lines)
```

**Impact:**
- `simple_pool.rs` (876 lines) → `pool/` directory (790 lines total, max 400 per file)
- Better organization without over-engineering
- All tests remain in respective files

---

## Issue 2: API Surface Reduction

### Current Exports (25 items)

```rust
// From mod.rs (lines 17-25):
pub use analysis::{compare_experiments, generate_report, ExperimentComparison, Statistics};
pub use bias::{BiasProfile, ClassBias, SpectralBias, ChromaBias, ProfileMetadata};
pub use diversity::{chroma_dispersion, mmr_score, retrieve_diverse_mmr, DiversityStats};
pub use embedding::{EmbeddingMapper, QuerySignature};
pub use experiment::{ExperimentConfig, ExperimentHarness, ExperimentResult, SeedingStrategy};
pub use hybrid_scoring::{rerank_hybrid, RetrievalWeights};
pub use retrieval_mode::RetrievalMode;
pub use simple_pool::SimpleDreamPool;
pub use soft_index::{SoftIndex, Similarity, EntryId};
```

**Counting:**
- analysis: 4 items
- bias: 5 items
- diversity: 4 items
- embedding: 2 items
- experiment: 4 items
- hybrid_scoring: 2 items
- retrieval_mode: 1 item
- simple_pool: 1 item
- soft_index: 3 items
**Total: 26 items**

### Problem Analysis

| Issue | Severity | Justification |
|-------|----------|---------------|
| **Too many exports** | 🟡 Moderate | 26 items is discoverable but not minimal |
| **Mixed abstraction levels** | 🟢 Low | Most are appropriately public |
| **Unused exports** | 🟡 Unknown | Need usage analysis |

### Strategy: Tiered API Design

**Tier 1: Essential (Core API)**
- Must be public for basic usage
- Used in 80% of use cases

**Tier 2: Advanced (Power User API)**
- Needed for customization
- Used in 20% of use cases

**Tier 3: Internal (Implementation Details)**
- Should be private or `pub(crate)`

---

## Proposed API Tiers

### Tier 1: Core API (8 items) ✅ Keep Public

```rust
// Essential for basic dream pool usage
pub use simple_pool::{SimpleDreamPool, PoolConfig, DreamEntry};

// Essential for Phase 4 soft retrieval
pub use embedding::EmbeddingMapper;
pub use hybrid_scoring::RetrievalWeights;
pub use retrieval_mode::RetrievalMode;

// Essential for experiments
pub use experiment::ExperimentHarness;
pub use bias::BiasProfile;
```

**Usage:** 80% of users only need these

### Tier 2: Advanced API (10 items) ⚠️ Consider Submodule

```rust
// Advanced retrieval customization
pub mod retrieval {
    pub use crate::dream::soft_index::{SoftIndex, Similarity, EntryId};
    pub use crate::dream::embedding::QuerySignature;
    pub use crate::dream::hybrid_scoring::rerank_hybrid;
    pub use crate::dream::diversity::{retrieve_diverse_mmr, DiversityStats};
}

// Advanced bias synthesis
pub mod bias {
    pub use crate::dream::bias::{
        BiasProfile, ClassBias, SpectralBias, ChromaBias, ProfileMetadata
    };
}

// Experiment analysis
pub mod analysis {
    pub use crate::dream::analysis::{
        compare_experiments, generate_report, ExperimentComparison, Statistics
    };
}
```

**Usage:** Access via `dream::retrieval::SoftIndex` instead of `dream::SoftIndex`

### Tier 3: Internal (8 items) 🔒 Make Private

```rust
// Should NOT be in public API
pub(crate) use diversity::{chroma_dispersion, mmr_score};  // Internal helpers
pub(crate) use experiment::{ExperimentConfig, ExperimentResult, SeedingStrategy};  // Use ExperimentHarness instead
```

---

## Recommended API Design (Final)

### Option A: Flat with Namespacing (Conservative)

```rust
// src/dream/mod.rs

// Core API (always flat)
pub use simple_pool::{SimpleDreamPool, PoolConfig, DreamEntry};
pub use embedding::EmbeddingMapper;
pub use retrieval_mode::RetrievalMode;
pub use bias::BiasProfile;
pub use hybrid_scoring::RetrievalWeights;

// Advanced API (namespaced submodules)
pub mod retrieval {
    pub use crate::dream::soft_index::{SoftIndex, Similarity, EntryId};
    pub use crate::dream::embedding::QuerySignature;
    pub use crate::dream::hybrid_scoring::rerank_hybrid;
}

pub mod experiment {
    pub use crate::dream::experiment::ExperimentHarness;
    pub use crate::dream::analysis::{compare_experiments, ExperimentComparison};
}
```

**Migration:**
- Old: `use dream::SoftIndex;` ❌ Breaks
- New: `use dream::retrieval::SoftIndex;` ✅ More explicit
- Core API unchanged: `use dream::SimpleDreamPool;` ✅ Still works

### Option B: Prelude Pattern (Ergonomic)

```rust
// src/dream/mod.rs

// Core API (minimal)
pub use simple_pool::SimpleDreamPool;
pub use embedding::EmbeddingMapper;
pub use retrieval_mode::RetrievalMode;

// Prelude for common imports
pub mod prelude {
    pub use crate::dream::{
        SimpleDreamPool, EmbeddingMapper, RetrievalMode,
        BiasProfile, RetrievalWeights, ExperimentHarness,
    };
}

// Advanced submodules
pub mod retrieval { /* ... */ }
pub mod experiment { /* ... */ }
pub mod bias { /* ... */ }
```

**Usage:**
```rust
// Common case (clean)
use chromatic_cognition_core::dream::prelude::*;

// Advanced case (explicit)
use chromatic_cognition_core::dream::retrieval::SoftIndex;
```

---

## Recommendation

### For simple_pool.rs:
✅ **Do refactor** - Extract into `pool/` directory
- Reduces largest file from 876 → 400 lines
- Improves discoverability
- Non-breaking change

### For API surface:
✅ **Do namespace** - Use Option A (namespaced submodules)
- Reduces top-level from 26 → 7 items
- Makes advanced features discoverable via modules
- Breaking change, but cleaner API

---

## Implementation Plan

### Phase 1: Refactor simple_pool.rs (2 hours)

1. Create `src/dream/pool/` directory
2. Extract `entry.rs` (DreamEntry + impl)
3. Extract `config.rs` (PoolConfig + PoolStats)
4. Extract `retrieval.rs` (helper functions)
5. Keep `pool.rs` (SimpleDreamPool core)
6. Update `mod.rs` to re-export from pool/
7. Run tests to verify no breakage

### Phase 2: Reduce API surface (1 hour)

1. Create namespaced submodules (retrieval, experiment, bias)
2. Update `mod.rs` with new structure
3. Add deprecation warnings for old flat imports
4. Update examples to use new imports
5. Document migration path

### Phase 3: Update documentation (1 hour)

1. Add module-level docs for new submodules
2. Update README with new import paths
3. Add migration guide

**Total effort: 4 hours**

---

## Breaking Changes & Migration

### Breaking Changes

**If using advanced API:**
```rust
// Before
use chromatic_cognition_core::dream::{SoftIndex, Similarity};

// After
use chromatic_cognition_core::dream::retrieval::{SoftIndex, Similarity};
```

**Core API unchanged:**
```rust
// Still works (no change)
use chromatic_cognition_core::dream::{SimpleDreamPool, EmbeddingMapper};
```

### Migration Script

```bash
# Automated migration for internal codebase
find examples -name "*.rs" -exec sed -i 's/dream::{SoftIndex/dream::retrieval::{SoftIndex/g' {} +
find examples -name "*.rs" -exec sed -i 's/dream::Similarity/dream::retrieval::Similarity/g' {} +
```

---

## Success Metrics

**Before:**
- simple_pool.rs: 876 lines
- Top-level exports: 26 items
- Largest file: 876 lines

**After:**
- Largest pool file: 400 lines (-54%)
- Top-level exports: 7 items (-73%)
- Largest file: 495 lines (bias.rs)

**Quality metrics:**
- ✅ No file > 500 lines (except bias.rs at 495)
- ✅ Top-level API < 10 items
- ✅ All tests still passing
- ✅ Zero breaking changes for core API

---

## Decision Required

**Please choose:**

**A. Full refactoring** (both simple_pool split + API reduction)
   - Impact: 4 hours work, breaking changes for advanced users
   - Benefit: Clean architecture, better discoverability

**B. Conservative refactoring** (simple_pool split only)
   - Impact: 2 hours work, zero breaking changes
   - Benefit: Cleaner code, maintains compatibility

**C. Minimal changes** (just re-export organization)
   - Impact: 30 minutes, zero breaking changes
   - Benefit: Quick win, no disruption

**My recommendation: Option B (Conservative)**
- Fixes the real issue (876-line file)
- Maintains API compatibility
- Low risk, high value
