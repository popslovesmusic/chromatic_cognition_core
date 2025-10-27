# **Dreamer–Learner Interface Specification**

**Path:** `docs/specs/dreamer_learner_interface.md`  
 **Linked Modules:**

* Dreamer side: `dream::{generation, inject, monitor, pool}`

* Learner side: `training::{dataset, optimizer, evaluation}`

* Shared: `tensor`, `spectral`, `config`, `logging`, `db`

---

## **1\) Purpose**

Define the **contract** between the **Dreamer** (generator of imagined chromatic states) and the **Learner** (optimizer/analyst that consolidates useful patterns).  
 The Dreamer **populates** the Dream Pool; the Learner **retrieves**, **evaluates**, and **feeds back** guidance—without conflating roles.

Goals:

* Clean data schemas & APIs (Rust)

* Deterministic, auditable exchanges (JSON lines \+ SQLite)

* Pluggable retrieval strategies and feedback weighting

* Zero runtime network requirement by default

---

## **2\) Control Loop (High Level)**

┌─────────┐        writes           ┌────────────┐     ranked        ┌───────────┐  
│ Dreamer │ ──────────────────────▶ │ Dream Pool │ ────────────────▶ │  Learner  │  
└─────────┘  (DreamEntry records)   └────────────┘  (retrieve N)     └────┬──────┘  
     ▲                                                                 │  
     └──────────────────────── feedback weights / biases ◀─────────────┘  
                (FeedbackRecord → BiasProfile)

**Cycle:**

1. Dreamer runs a dream episode → stores `DreamEntry` if coherence ≥ threshold.

2. Learner issues `RetrievalQuery` → receives ranked entries → trains/evaluates.

3. Learner emits `FeedbackRecord` → Interface computes/updates `BiasProfile`.

4. Dreamer consumes `BiasProfile` to shape next dream seeds (but does not learn).

---

## **3\) Shared Data Schemas**

### **3.1 `ChromaticTensor` (serialized)**

* Binary: compressed f32 RGB array (NCHW or `[rows, cols, layers, 3]`)

* Hash: `sha256` of raw f32 buffer for integrity

* Stored inside `DreamEntry.tensor_blob`

### **3.2 `DreamEntry`**

pub struct DreamEntry {  
    pub id: Uuid,  
    pub timestamp: chrono::DateTime\<chrono::Utc\>,  
    pub factoid\_ids: Vec\<Uuid\>,  
    pub image\_id: Option\<Uuid\>,

    pub coherence: f32,            // spectral-entropy derived  
    pub energy: f32,               // model-defined energy metric  
    pub chroma\_signature: \[f32;3\], // HSV mean or agreed convention  
    pub spectral\_entropy: f32,

    pub tensor\_sha256: \[u8; 32\],  
    pub tensor\_blob: Vec\<u8\>,      // compressed ChromaticTensor  
    pub metadata\_json: String,     // dream config, metrics, provenance  
}

### **3.3 `RetrievalQuery`**

pub struct RetrievalQuery {  
    pub k: usize,                      // how many to retrieve  
    pub mode: RetrievalMode,           // cosine | euclidean | mixed  
    pub query\_signature: \[f32; 3\],     // chroma seed (HSV or agreed)  
    pub min\_coherence: Option\<f32\>,  
    pub max\_entropy: Option\<f32\>,  
    pub recency\_decay: Option\<f32\>,    // λ (0..)  
    pub allow\_duplicates: bool,        // false \= unique dream lineage  
    pub seed: u64,                     // deterministic sampling  
}

### **3.4 `RetrievedSet`**

pub struct RetrievedSet {  
    pub query\_id: Uuid,  
    pub entries: Vec\<DreamEntry\>,  
    pub stats: RetrievalStats,  
}

pub struct RetrievalStats {  
    pub mode: String,  
    pub avg\_coherence: f32,  
    pub avg\_entropy: f32,  
    pub diversity: f32,     // signature dispersion (e.g., mean pairwise distance)  
}

### **3.5 `LearningBatch`**

pub struct LearningBatch {  
    pub query\_id: Uuid,  
    pub entry\_ids: Vec\<Uuid\>,  
    pub tensors: Vec\<Vec\<u8\>\>,    // decoded or on-demand stream handles  
    pub labels: Option\<Vec\<i32\>\>, // optional (supervised tasks)  
}

### **3.6 `FeedbackRecord`**

pub struct FeedbackRecord {  
    pub query\_id: Uuid,  
    pub entry\_id: Uuid,  
    pub utility: f32,          // task-defined (e.g., Δloss, reward)  
    pub contribution: f32,     // SHAP/gradient or heuristic credit  
    pub notes: Option\<String\>, // freeform audit message  
}

### **3.7 `BiasProfile` (Dreamer guidance)**

pub struct BiasProfile {  
    pub updated\_at: chrono::DateTime\<chrono::Utc\>,  
    pub chroma\_prior: \[f32;3\],       // preferred region  
    pub entropy\_window: (f32, f32),  // desired patch entropy range  
    pub seed\_weights: SeedWeights,   // factoid/image/type weights  
}

pub struct SeedWeights {  
    pub factoid\_weight: f32,  
    pub image\_weight: f32,  
    pub pool\_seed\_weight: f32,  
    pub random\_weight: f32,  
}

---

## **4\) Interface API (Rust)**

**Module Path:** `src/interface/dreamer_learner.rs`

pub trait DreamerLearnerInterface {  
    // DREAMER → POOL  
    fn persist\_dream(\&self, entry: DreamEntry) \-\> anyhow::Result\<()\>;

    // LEARNER → POOL  
    fn retrieve(\&self, q: RetrievalQuery) \-\> anyhow::Result\<RetrievedSet\>;  
    fn materialize\_batch(\&self, r: \&RetrievedSet) \-\> anyhow::Result\<LearningBatch\>;

    // LEARNER → DREAMER (feedback)  
    fn submit\_feedback(\&self, fb: FeedbackRecord) \-\> anyhow::Result\<()\>;  
    fn synthesize\_bias\_profile(\&self) \-\> anyhow::Result\<BiasProfile\>;

    // Introspection / health  
    fn pool\_stats(\&self) \-\> anyhow::Result\<PoolStats\>;  
}

pub struct PoolStats {  
    pub dreams\_total: u64,  
    pub coherence\_mean: f32,  
    pub spectral\_entropy\_mean: f32,  
    pub last\_compaction\_at: Option\<chrono::DateTime\<chrono::Utc\>\>,  
}

**Default impl:** `SqliteDreamerLearnerInterface`

* Storage: SQLite (`dream_pool.sqlite`) with FTS or vector index table for signatures

* Logs: JSONL under `logs/interface/*.jsonl` (append-only)

* Determinism: seeded sampling, reproducible retrieval order

---

## **5\) Retrieval Modes (Semantics)**

* **cosine**: maximize directional similarity in chroma space; apply recency decay

* **euclidean**: minimize absolute distance; good for strict color matching

* **mixed**: top-N cosine \+ top-M by coherence; merged & deduplicated; stable sort by composite score

**Composite score (mixed):**  
 \[  
 S\_i \= \\alpha \\cdot \\text{cosine}(q, s\_i) \+ (1-\\alpha) \\cdot \\text{coherence}\_i  
 \]  
 with recency weight (w\_i \= e^{-\\lambda \\Delta t\_i}). Final score (S'\_i \= w\_i \\cdot S\_i).

---

## **6\) Bias Synthesis (Feedback → Guidance)**

After `submit_feedback` calls:

1. **Aggregate utility** per dream and per chroma bin (e.g., HSV hexagon bins).

2. **Compute priors**:

   * `chroma_prior` \= weighted centroid of high-utility bins

   * `entropy_window` \= interquartile range of entropy among helpful dreams

   * `seed_weights` \= normalized utility by source type (factoid/image/pool/random)

3. Produce `BiasProfile`; write to `data/bias_profile.json`.

4. Dreamer reads the profile before next cycle and reweights seed sampling (no weight updates in Dreamer ops).

---

## **7\) File & Table Layout**

**SQLite tables**

* `dreams(id, ts, chroma_r, chroma_g, chroma_b, coherence, energy, spectral_entropy, tensor_sha256, tensor_blob, metadata_json)`

* `lineage(parent_id, child_id)` *(optional, for ancestry)*

* `feedback(entry_id, query_id, utility, contribution, ts)`

* Indices on `(coherence DESC)`, `(ts DESC)`, `(chroma_r, chroma_g, chroma_b)`

**Logs**

* `logs/interface/persist.jsonl` — `DreamEntry` writes

* `logs/interface/retrieval.jsonl` — queries & stats

* `logs/interface/feedback.jsonl` — feedback stream

* `data/bias_profile.json` — latest profile

---

## **8\) Config Additions (`config/engine.toml`)**

\[interface\]  
retrieval\_default \= "mixed"  
alpha\_cosine \= 0.6  
recency\_decay \= 0.02  
k \= 8

\[bias\]  
update\_interval\_steps \= 200  
min\_feedback \= 20  
chroma\_bins \= 24

---

## **9\) Error Handling & Guarantees**

* **Atomic writes**: use SQLite transactions for persist & feedback batches.

* **Integrity**: verify `tensor_sha256` on deserialize; reject on mismatch.

* **Determinism**: retrieval with same `(query, seed)` returns same ordered set.

* **Isolation**: Dreamer never executes Learner code; only consumes `BiasProfile`.

* **Versioning**: include `schema_version` in `metadata_json`; migrate with Alembic-like scripts in Rust.

---

## **10\) Metrics & Evaluation**

Interface-level KPIs (logged each epoch):

* **selection\_precision**: fraction of retrieved dreams that the Learner marks as *useful* (utility \> τ)

* **utilization\_rate**: fraction of retrieved dreams actually used in a training step

* **improvement\_delta**: Δ(loss/accuracy) per 100 feedback events

* **coverage**: % chroma bins represented in retrieved sets

* **recency\_ratio**: share of dreams \< X days old in retrieved sets

---

## **11\) Testing Checklist**

* Persist/retrieve round-trip: tensor deserialization lossless, hash matches

* Deterministic retrieval under fixed seed

* Mixed mode produces expected composition (cosine top-K ∪ coherence top-M)

* Recency decay changes ranking in controlled fixture

* Feedback aggregation → BiasProfile stable and bounded

* Dreamer seed sampling respects `SeedWeights` within tolerance

* Concurrency: parallel Learner retrievals do not corrupt writes (SQLite WAL mode)

---

## **12\) Example Flows**

### **12.1 Dreamer write**

let entry \= dream::finalize\_episode(\&tensor, \&metrics, \&provenance)?;  
if entry.coherence \>= cfg.pool.coherence\_threshold {  
    iface.persist\_dream(entry)?;  
}

### **12.2 Learner retrieve → train → feedback**

let q \= RetrievalQuery {  
    k: 8,  
    mode: RetrievalMode::Mixed,  
    query\_signature: seed\_chroma,  
    min\_coherence: Some(0.7),  
    max\_entropy: None,  
    recency\_decay: Some(cfg.interface.recency\_decay),  
    allow\_duplicates: false,  
    seed: 12345,  
};  
let set \= iface.retrieve(q)?;  
let batch \= iface.materialize\_batch(\&set)?;  
let outcome \= learner.train\_on(\&batch)?; // returns per-entry utilities

for (entry\_id, util, contrib) in outcome.feedback {  
    iface.submit\_feedback(FeedbackRecord {   
        query\_id: set.stats.query\_id, entry\_id, utility: util, contribution: contrib, notes: None   
    })?;  
}

if should\_update\_bias() {  
    let profile \= iface.synthesize\_bias\_profile()?;  
    fs::write("data/bias\_profile.json", serde\_json::to\_vec\_pretty(\&profile)?)?;  
}

---

## **13\) Non-Goals (by design)**

* No in-Dreamer weight updates or gradient descent.

* No direct Learner control of Dreamer parameters (only *bias*).

* No network calls or external data pulls during interface operations (local-first).

---

## **14\) Security & Transparency**

* All persisted artifacts are **verifiable** (content hash \+ sizes).

* Every retrieval and feedback event is **append-only logged** with timestamps and seeds.

* Bias synthesis is **explainable** (export bin contributions & priors to JSON).

---

### **Summary**

This interface keeps the **Dreamer** purely generative and the **Learner** purely consolidative—while enabling an evidence-driven coupling via **retrieval** and **feedback-to-bias**.  
 It preserves your core principles: **transparency, determinism, and interpretability**—and it’s ready for implementation as-is.

