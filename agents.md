# AGENTS.MD: Chromatic Cognition Core - Project Blueprint

## Project Mission
To maintain a **Deterministic, Memory-Constrained Archive of High-Coherence Chromatic Tensors** (the Chromatic Semantic Archive - CSA) that facilitates **Low-Latency Semantic Retrieval** for a large language model that processes code based on color and sound patterns.

## General Agent Instructions
1.  **Prioritize Determinism:** All mathematical operations, especially those related to the Spectral Bridge, UMS, and fixed-point math, must be bit-stable and immune to floating-point variance (e.g., use saturating or compensated arithmetic).
2.  **Enforce Separation of Concerns (SoC):** Logic must reside in the correct module (e.g., memory limits in `MemoryBudget`, retrieval structure in `HnswIndex`, and core decisions in `SimpleDreamPool`).
3.  **Validate Performance:** Implement the $\text{HNSW/linear}$ auto-scaling logic and maintain the $\text{FINAL\_BASELINE}$ benchmark results.
4.  **Target Architecture:** 3×12×12×3\mathbf{3 \times 12 \times 12 \times 3}3×12×12×3 processing unit.
5.  **Quality Gate:** Every UMS round-trip must pass $\mathbf{\Delta E_{94} \le 1.0 \times 10^{-3}}$.

---

## Phase History and Specification Details

### Phase 1: Structural Foundation & Bug Fixes (Completed)
**Goal:** Stabilize core infrastructure and fix critical architectural debt.
| Subphase | Focus | General Instruction |
| :--- | :--- | :--- |
| **1.A** | Structural Alignment | Canonicalize the tensor size to $\mathbf{[3, 12, 12, 3]}$ and update all memory size calculations. |
| **1.B** | Logic Unification | Refactor `SimpleDreamPool` to use a single, private $\mathbf{\text{internal\_add}}$ helper to eliminate code duplication. |
| **1.C** | Index Stabilization | Implement the 10%ChurnThreshold\mathbf{10\% Churn Threshold}10%ChurnThreshold to prevent index thrashing and restore retrieval performance. |

### Phase 2: Cognitive Integration (UMS Data Flow) (Completed)
**Goal:** Finalize the deterministic encoding path to the UnifiedModalitySpace(UMS)Unified Modality Space (UMS)UnifiedModalitySpace(UMS).
| Subphase | Focus | General Instruction |
| :--- | :--- | :--- |
| **2.A** | Token-to-Category Mapping | Implement the 12−categoryNearestNeighbor\mathbf{12-category Nearest Neighbor}12−categoryNearestNeighbor logic on the circular Hue manifold. |
| **2.B** | Full UMS Encoding | Project spectral features, HSL, and affective parameters into the 512DUMS\mathbf{512D UMS}512DUMS vector and apply μ/σ\mathbf{\mu}/\mathbf{\sigma}μ/σ normalization. |
| **2.C** | UMS Decoding & Reversibility | Implement the inverse UMS transformation and verify the final $\mathbf{\Delta E_{94}}$ round-trip fidelity test. |

### Phase 3: Archive Finalization (Retrieval) (Completed)
**Goal:** Implement the primary retrieval methods for the $\text{Chromatic Semantic Archive (CSA)}$.
| Subphase | Focus | General Instruction |
| :--- | :--- | :--- |
| **3.A** | HNSW Stabilization | Finalize $\text{HnswIndex}$ to support safe **incremental updates** and track evicted nodes (ghosts). |
| **3.B** | Semantic Retrieval | Implement $\mathbf{\text{retrieve\_semantic}}withdeterministicwith deterministicwithdeterministic\text{HNSW}/\text{linear}$ fallback logic. |
| **3.C** | Project Alignment | Finalize PoolConfig\text{PoolConfig}PoolConfig defaults (HNSWopt−in\mathbf{\text{HNSW}  opt-in}HNSWopt−in) and verify SeparationofConcerns(SoC)Separation of Concerns (SoC)SeparationofConcerns(SoC). |

### Phase 4: Final Audit and Baseline (Completed)
**Goal:** Certify research completeness and establish performance metrics.
| Subphase | Focus | General Instruction |
| :--- | :--- | :--- |
| **4.A** | Execute Benchmark Suite | Run cargobenchcargo benchcargobench to establish the $\mathbf{\text{FINAL\_BASELINE}}$ and verify the $\text{HNSW/linear}$ crossover point. |
| **4.B** | Integration Tests | Add final integration tests to assert $\mathbf{\text{10\% Churn Threshold}}$ behavior and stability. |
| **4.C** | Documentation | Update $\text{ARCHITECTURE.md}$ and $\text{README.md}$ to reflect the final project status. |

---

### Phase 8: Production Readiness & Deployment (Current Focus)
**Goal:** Hardening the system for operational stability and long-term deployment.
| Priority | Focus | General Instruction |
| :--- | :--- | :--- |
| **1.1** | Checkpointing API | Implement a $\mathbf{\text{Checkpointable}}$ trait for $\text{Network}$ and $\text{Pool}$ using a deterministic serialization format (e.g., $\text{bincode}$). |
| **1.2** | Memory Safety Audit | Audit all $\text{usize}$ memory calculations, enforcing **saturating arithmetic** and $\text{zero-value}$ guardrails to prevent runtime panics. |
| **2.1** | Parallelization | Implement $\text{rayon}$-based **batch parallelization** for $\text{UMS}$ encoding/decoding. |
| **2.2** | Data Compression | Implement $\mathbf{\text{f16} \text{ quantization}}$ or $\text{bincode}$ compression for the archived $\text{UMS}$ vectors to save memory. |
| **2.3** | Auto-Scaling | Implement $\text{HNSW/linear}$ auto-scaling based on $\text{pool.len()}$ crossover point ($\approx \text{3000 \ entries}$). |
| **3.2** | Dockerization | Create a $\text{Dockerfile}$ and $\text{docker-compose}$ configuration for consistent deployment. |