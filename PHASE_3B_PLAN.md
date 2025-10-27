# Phase 3B: Learner Refinement - Implementation Plan

**Timeline:** 2 weeks (2025-10-27 to 2025-11-10)
**Goal:** Make Dream Pool integration beneficial and prepare Dreamer-Learner coupling
**Status:** 📋 PLANNED

---

## Executive Summary

Phase 3B will transform the Dream Pool from a concept that slows training into an adaptive memory system that accelerates learning. The current MVP proves training works (90% accuracy), but retrieval needs refinement to provide actual benefit.

### Success Criteria

**Primary (Must Achieve):**
1. ✅ Dream Pool retrieval improves final accuracy by >2%
2. ✅ Convergence accelerates (fewer epochs to 95% accuracy)
3. ✅ Bias profile synthesis generates actionable feedback

**Secondary (Nice to Have):**
4. FFT features improve retrieval quality
5. Diversity enforcement prevents mode collapse
6. Utility scoring correlates with actual improvement

---

## Current State Analysis

### What Works ✅
- SimpleDreamPool stores and retrieves dreams
- Training achieves 90% baseline accuracy
- Basic cosine similarity retrieval functional
- Infrastructure for metrics tracking exists

### What Doesn't Work ❌
- Retrieval slows convergence (24 vs 15 epochs)
- No class awareness (retrieves wrong-class dreams)
- No diversity enforcement (duplicates possible)
- No feedback loop (can't learn what helps)
- Coherence ≠ task utility

### Root Cause
**Current retrieval:** Cosine similarity on mean RGB
- Two different classes can have similar mean RGB
- Ignores class labels entirely
- No concept of "useful" vs "harmful" dreams

---

## Deliverable 1: Class-Aware DreamPool Retrieval

**Timeline:** Days 1-3
**Location:** `src/dream/simple_pool.rs` enhancements
**Priority:** 🔥 CRITICAL

### Current Implementation
```rust
pub fn retrieve_similar(&self, query_signature: &[f32; 3], k: usize) -> Vec<DreamEntry>
```
- Only uses mean RGB
- No class information
- No filtering options

### Enhanced Implementation

#### New Structure: ClassAwareDreamEntry
```rust
pub struct ClassAwareDreamEntry {
    pub tensor: ChromaticTensor,
    pub result: SolverResult,
    pub chroma_signature: [f32; 3],
    pub class_label: Option<ColorClass>,  // NEW
    pub utility: Option<f32>,              // NEW (Δloss)
    pub timestamp: SystemTime,             // NEW
    pub usage_count: usize,                // NEW
}
```

#### New Retrieval Methods
```rust
// Class-aware retrieval
pub fn retrieve_similar_class(
    &self,
    query_signature: &[f32; 3],
    target_class: ColorClass,
    k: usize,
) -> Vec<ClassAwareDreamEntry>

// Multi-class balanced retrieval
pub fn retrieve_balanced(
    &self,
    query_signature: &[f32; 3],
    classes: &[ColorClass],
    k_per_class: usize,
) -> Vec<ClassAwareDreamEntry>

// Utility-weighted retrieval
pub fn retrieve_by_utility(
    &self,
    query_signature: &[f32; 3],
    k: usize,
    utility_threshold: f32,
) -> Vec<ClassAwareDreamEntry>
```

### Implementation Tasks
- [ ] Add `ClassAwareDreamEntry` struct
- [ ] Extend `SimpleDreamPool` to track class labels
- [ ] Implement `retrieve_similar_class()`
- [ ] Implement `retrieve_balanced()`
- [ ] Add unit tests (class filtering, balance checking)
- [ ] Benchmark retrieval performance

### Expected Impact
- **Accuracy:** +3-5% (retrieves same-class examples)
- **Convergence:** -30% epochs (better initialization)
- **Overhead:** Minimal (just filtering)

---

## Deliverable 2: Diversity Enforcement

**Timeline:** Days 3-5
**Location:** `src/dream/diversity.rs` (new module)
**Priority:** 🔥 CRITICAL

### Problem
Current retrieval can return near-duplicates:
- Same dream retrieved multiple times
- Very similar dreams cluster together
- Reduces effective batch diversity

### Solution: Diversity Metrics

#### Chroma Dispersion
```rust
pub fn chroma_dispersion(entries: &[DreamEntry]) -> f32 {
    // Compute pairwise mean RGB distances
    // Return normalized dispersion score [0, 1]
}
```

#### Diversity-Enforced Retrieval
```rust
pub fn retrieve_diverse(
    &self,
    query_signature: &[f32; 3],
    k: usize,
    min_dispersion: f32,
) -> Vec<DreamEntry> {
    // 1. Retrieve 2k candidates by similarity
    // 2. Greedily select k dreams maximizing dispersion
    // 3. Ensure pairwise distances > threshold
}
```

#### Maximum Marginal Relevance (MMR)
```rust
fn mmr_score(
    candidate: &DreamEntry,
    query: &[f32; 3],
    selected: &[DreamEntry],
    lambda: f32,  // balance relevance vs diversity
) -> f32 {
    let relevance = cosine_similarity(&candidate.chroma_signature, query);
    let max_similarity = selected.iter()
        .map(|s| cosine_similarity(&candidate.chroma_signature, &s.chroma_signature))
        .max();

    lambda * relevance - (1.0 - lambda) * max_similarity.unwrap_or(0.0)
}
```

### Implementation Tasks
- [ ] Create `diversity.rs` module
- [ ] Implement `chroma_dispersion()`
- [ ] Implement `retrieve_diverse()`
- [ ] Implement MMR-based selection
- [ ] Add diversity monitoring metrics
- [ ] Unit tests (dispersion calculation, MMR scoring)

### Expected Impact
- **Accuracy:** +1-2% (more varied training data)
- **Robustness:** Better generalization
- **Mode Collapse:** Prevented

---

## Deliverable 3: FFT-Based Feature Extraction

**Timeline:** Days 5-8
**Location:** `src/spectral/` (new module)
**Priority:** ⭐ HIGH

### Purpose
Replace mean RGB with richer spectral features:
- Frequency domain representation
- Texture information
- Pattern detection

### Implementation

#### Spectral Analysis Module
```rust
// src/spectral/mod.rs
pub mod fft;
pub mod features;

pub use fft::compute_2d_fft;
pub use features::SpectralFeatures;
```

#### FFT Computation
```rust
// src/spectral/fft.rs
use rustfft::{FftPlanner, num_complex::Complex};

pub fn compute_2d_fft(
    tensor: &ChromaticTensor,
    apply_hann: bool,
) -> Array2<Complex<f32>> {
    // 1. Flatten to 2D (spatial dimensions)
    // 2. Apply Hann window if requested
    // 3. Compute 2D FFT
    // 4. Return frequency domain representation
}

pub fn spectral_entropy(fft: &Array2<Complex<f32>>) -> f32 {
    // Compute Shannon entropy of magnitude spectrum
}
```

#### Spectral Features
```rust
// src/spectral/features.rs
pub struct SpectralFeatures {
    pub entropy: f32,              // Shannon entropy
    pub peak_frequency: (f32, f32), // Dominant frequency
    pub energy_ratio: f32,          // Low freq / high freq
    pub mean_magnitude: f32,        // Average spectral energy
    pub variance: f32,              // Spectral variance
}

impl SpectralFeatures {
    pub fn extract(tensor: &ChromaticTensor) -> Self {
        let fft = compute_2d_fft(tensor, true);
        // Extract all features from FFT
    }

    pub fn distance(&self, other: &Self) -> f32 {
        // Euclidean distance in feature space
    }
}
```

#### Enhanced DreamEntry
```rust
pub struct EnhancedDreamEntry {
    pub tensor: ChromaticTensor,
    pub result: SolverResult,
    pub chroma_signature: [f32; 3],
    pub spectral_features: SpectralFeatures,  // NEW
    pub class_label: Option<ColorClass>,
    pub utility: Option<f32>,
}
```

### Implementation Tasks
- [ ] Add `rustfft` dependency to Cargo.toml
- [ ] Create `src/spectral/mod.rs`
- [ ] Implement `compute_2d_fft()`
- [ ] Implement `spectral_entropy()`
- [ ] Create `SpectralFeatures` struct
- [ ] Implement feature extraction pipeline
- [ ] Add spectral-based retrieval method
- [ ] Unit tests (FFT correctness, entropy bounds)
- [ ] Benchmark FFT performance

### Expected Impact
- **Retrieval Quality:** +10-15% better matches
- **Coherence Definition:** Now scientifically grounded
- **Feature Richness:** Beyond simple RGB mean

---

## Deliverable 4: ΔLoss-Based Utility Scoring

**Timeline:** Days 8-10
**Location:** `src/learner/feedback.rs` (new module)
**Priority:** 🔥 CRITICAL

### Purpose
Track which dreams actually help training:
- Compute Δloss before and after using dream
- Store utility scores in pool
- Bias future retrieval toward helpful dreams

### Implementation

#### Feedback Collection
```rust
// src/learner/feedback.rs
pub struct FeedbackRecord {
    pub dream_id: usize,           // Index in pool
    pub step: usize,               // Training step
    pub loss_before: f32,          // Loss without dream
    pub loss_after: f32,           // Loss with dream
    pub utility: f32,              // Normalized Δloss
    pub contribution: f32,         // SHAP-like attribution
}

impl FeedbackRecord {
    pub fn compute_utility(loss_before: f32, loss_after: f32) -> f32 {
        // Utility = relative improvement
        // Positive = helpful, Negative = harmful
        (loss_before - loss_after) / loss_before
    }
}
```

#### Utility Aggregation
```rust
pub struct UtilityAggregator {
    records: Vec<FeedbackRecord>,
    window_size: usize,  // Only consider recent feedback
}

impl UtilityAggregator {
    pub fn add_feedback(&mut self, record: FeedbackRecord) {
        self.records.push(record);
        if self.records.len() > self.window_size {
            self.records.remove(0);
        }
    }

    pub fn get_dream_utility(&self, dream_id: usize) -> Option<f32> {
        // Average utility for this dream
        let utilities: Vec<f32> = self.records.iter()
            .filter(|r| r.dream_id == dream_id)
            .map(|r| r.utility)
            .collect();

        if utilities.is_empty() {
            None
        } else {
            Some(utilities.iter().sum::<f32>() / utilities.len() as f32)
        }
    }

    pub fn top_k_useful(&self, k: usize) -> Vec<(usize, f32)> {
        // Return (dream_id, avg_utility) for top k dreams
    }
}
```

#### Training Integration
```rust
// In src/learner/training.rs
pub fn train_with_feedback(
    classifier: MLPClassifier,
    train_data: &[ColorSample],
    pool: &mut SimpleDreamPool,
    feedback_aggregator: &mut UtilityAggregator,
) -> TrainingResult {
    for epoch in 0..config.num_epochs {
        for batch in batches {
            // Measure baseline loss
            let (loss_before, _) = classifier.compute_loss(&batch);

            // Augment with retrieved dreams
            let retrieved = pool.retrieve_by_utility(...);
            let augmented_batch = augment(batch, retrieved);

            // Measure augmented loss
            let (loss_after, grads) = classifier.compute_loss(&augmented_batch);

            // Record feedback
            for (i, dream) in retrieved.iter().enumerate() {
                let feedback = FeedbackRecord {
                    dream_id: dream.id,
                    step: epoch * num_batches + batch_idx,
                    loss_before,
                    loss_after,
                    utility: FeedbackRecord::compute_utility(loss_before, loss_after),
                    contribution: grads.norm() / retrieved.len() as f32,
                };
                feedback_aggregator.add_feedback(feedback);
            }

            // Update weights
            classifier.update_weights(&grads, lr);
        }
    }
}
```

### Implementation Tasks
- [ ] Create `src/learner/feedback.rs`
- [ ] Implement `FeedbackRecord` struct
- [ ] Implement `UtilityAggregator`
- [ ] Add utility tracking to `ClassAwareDreamEntry`
- [ ] Modify training loop to collect feedback
- [ ] Add `retrieve_by_utility()` to pool
- [ ] Unit tests (utility computation, aggregation)
- [ ] Integration test (feedback collection during training)

### Expected Impact
- **Adaptive Retrieval:** Pool learns what helps
- **Accuracy:** +2-3% (uses better dreams)
- **Convergence:** -20% epochs (faster learning)

---

## Deliverable 5: Bias Profile Synthesis

**Timeline:** Days 10-12
**Location:** `src/dream/bias.rs` (new module)
**Priority:** ⭐⭐ HIGH

### Purpose
Aggregate feedback into actionable guidance for Dreamer:
- Which chroma regions are useful?
- What spectral characteristics help?
- Which classes need more dreams?

### Implementation

#### BiasProfile Structure
```rust
// src/dream/bias.rs
pub struct BiasProfile {
    pub updated_at: SystemTime,
    pub chroma_prior: [f32; 3],        // Preferred mean RGB
    pub spectral_entropy_range: (f32, f32),  // Desired entropy window
    pub class_weights: HashMap<ColorClass, f32>,  // Per-class importance
    pub utility_stats: UtilityStats,
    pub seed_weights: SeedWeights,
}

pub struct UtilityStats {
    pub mean: f32,
    pub variance: f32,
    pub top_contributors: Vec<(usize, f32)>,  // (dream_id, utility)
}

pub struct SeedWeights {
    pub from_pool: f32,      // Use retrieved dreams
    pub from_class: f32,     // Use class-specific samples
    pub random: f32,         // Random init
    pub diversity: f32,      // Diversity bonus
}
```

#### Synthesis Algorithm
```rust
impl BiasProfile {
    pub fn synthesize(
        pool: &SimpleDreamPool,
        aggregator: &UtilityAggregator,
        config: &BiasConfig,
    ) -> Self {
        // 1. Find high-utility dreams
        let top_dreams = aggregator.top_k_useful(config.top_k);

        // 2. Compute chroma prior (weighted centroid)
        let chroma_prior = compute_weighted_centroid(
            &top_dreams,
            pool,
            |utility| utility.max(0.0),  // Only positive utility
        );

        // 3. Compute spectral entropy range (IQR of useful dreams)
        let entropies = top_dreams.iter()
            .map(|(id, _)| pool.get(*id).spectral_features.entropy)
            .collect();
        let spectral_entropy_range = interquartile_range(&entropies);

        // 4. Compute class weights (which classes need help?)
        let class_weights = compute_class_utility_weights(aggregator, pool);

        // 5. Compute seed weights (how to sample?)
        let seed_weights = SeedWeights {
            from_pool: top_dreams.len() as f32 / pool.len() as f32,
            from_class: 0.5,  // Balance class-aware sampling
            random: 0.2,      // Some exploration
            diversity: 0.3,   // Encourage diversity
        };

        BiasProfile {
            updated_at: SystemTime::now(),
            chroma_prior,
            spectral_entropy_range,
            class_weights,
            utility_stats: UtilityStats::from(aggregator),
            seed_weights,
        }
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let json = self.to_json()?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
```

### Implementation Tasks
- [ ] Create `src/dream/bias.rs`
- [ ] Implement `BiasProfile` struct
- [ ] Implement `synthesize()` algorithm
- [ ] Add chroma bin aggregation
- [ ] Add class weight computation
- [ ] Implement JSON serialization
- [ ] Add `save()` and `load()` methods
- [ ] Unit tests (synthesis correctness, JSON roundtrip)
- [ ] Integration test (full pipeline)

### Expected Impact
- **Dreamer Guidance:** Clear signal on what to generate
- **Adaptivity:** Learns from experience
- **Transparency:** Explainable via JSON export

---

## Deliverable 6: Validation Experiment

**Timeline:** Days 12-14
**Location:** `examples/learner_phase3b_validation.rs`
**Priority:** 🔥 CRITICAL

### Experiment Design

**Goal:** Prove Phase 3B refinements improve Dream Pool benefit

**Setup:**
- Baseline: Original MVP (cosine similarity, no class awareness)
- Refined: Phase 3B (class-aware, diverse, utility-weighted, FFT features)
- Control: No Dream Pool (pure baseline)

**Metrics:**
- Final validation accuracy
- Convergence epoch (95% threshold)
- Training time
- Pool utility statistics
- Bias profile quality

### Success Criteria

✅ **Primary:**
- Refined > Baseline > Control (accuracy)
- Refined converges faster than Baseline
- Bias profile shows clear class preferences

✅ **Secondary:**
- FFT features improve retrieval quality
- Diversity enforcement prevents duplicates
- Utility scores correlate with actual improvement

### Implementation Tasks
- [ ] Create `learner_phase3b_validation.rs`
- [ ] Implement 3-way comparison (Control, Baseline, Refined)
- [ ] Add comprehensive metrics logging
- [ ] Generate comparative plots
- [ ] Export bias profiles for analysis
- [ ] Create validation report generator

---

## Timeline & Dependencies

### Week 1 (Days 1-7)

**Days 1-3:** Class-Aware Retrieval
- Implement `ClassAwareDreamEntry`
- Add class filtering methods
- Unit tests

**Days 3-5:** Diversity Enforcement
- Create diversity module
- Implement MMR selection
- Unit tests

**Days 5-7:** FFT Features (Part 1)
- Add rustfft dependency
- Implement basic FFT computation
- Unit tests

**Milestone 1:** Class-aware, diverse retrieval functional

### Week 2 (Days 8-14)

**Days 8-10:** Utility Scoring
- Create feedback module
- Implement utility aggregation
- Integration with training loop

**Days 10-12:** Bias Synthesis
- Create bias module
- Implement synthesis algorithm
- JSON export

**Days 12-14:** Validation
- Run 3-way comparison experiment
- Analyze results
- Generate report

**Milestone 2:** Full Phase 3B complete, validated

---

## Technical Requirements

### New Dependencies
```toml
[dependencies]
rustfft = "6.2"           # FFT computation
num-complex = "0.4"       # Complex number support
```

### Module Structure
```
src/
├── dream/
│   ├── simple_pool.rs    # Enhanced with class awareness
│   ├── diversity.rs      # NEW: Diversity enforcement
│   └── bias.rs           # NEW: Bias profile synthesis
├── learner/
│   ├── classifier.rs
│   ├── training.rs       # Enhanced with feedback
│   └── feedback.rs       # NEW: Utility tracking
└── spectral/             # NEW MODULE
    ├── mod.rs
    ├── fft.rs            # FFT computation
    └── features.rs       # Spectral feature extraction
```

---

## Risk Mitigation

### Technical Risks

**Risk 1:** FFT computation too slow
- **Mitigation:** Use FFTW or RustFFT (optimized)
- **Fallback:** Compute FFT offline, cache features

**Risk 2:** Diversity enforcement degrades quality
- **Mitigation:** Tune λ parameter (relevance vs diversity)
- **Fallback:** Make diversity optional

**Risk 3:** Utility scores too noisy
- **Mitigation:** Use moving average, longer window
- **Fallback:** Combine with coherence heuristic

### Implementation Risks

**Risk 4:** Scope creep (2 weeks tight)
- **Mitigation:** MVP each deliverable, defer polish
- **Fallback:** Drop FFT features if behind schedule

**Risk 5:** Integration complexity
- **Mitigation:** Incremental integration, test each piece
- **Fallback:** Feature flags to disable components

---

## Expected Outcomes

### Quantitative

| Metric | MVP Baseline | Phase 3B Target | Improvement |
|--------|--------------|-----------------|-------------|
| Final Accuracy | 90% | >92% | +2%+ |
| Convergence Epoch | 15 | <12 | -20%+ |
| Pool Utility | N/A | >0.1 | Positive |
| Retrieval Quality | Random | Class-aware | 100%+ |

### Qualitative

✅ **Dream Pool becomes beneficial**
- Retrieval accelerates training
- Bias profile guides generation
- Feedback loop closes

✅ **Scientifically grounded**
- FFT-based coherence definition
- Utility scores track real improvement
- Bias synthesis is explainable

✅ **Production-ready**
- JSON export for Dreamer integration
- Monitoring dashboards
- Configurable hyperparameters

---

## Documentation Deliverables

1. **Implementation Guide:** Phase 3B feature documentation
2. **Validation Report:** Experimental results and analysis
3. **API Reference:** New modules and methods
4. **Migration Guide:** MVP → Phase 3B upgrade path

---

## Next Phase Preview: Phase 4 - Dreamer Coupling

After Phase 3B completion:
- Dreamer reads BiasProfile
- Weighted dream generation
- Closed-loop optimization
- Full LEARNER MANIFEST v1.0

---

**Phase 3B Status:** 📋 PLANNED
**Ready to Begin:** ✅ YES
**Estimated Completion:** 2025-11-10

Let's build it! 🚀
