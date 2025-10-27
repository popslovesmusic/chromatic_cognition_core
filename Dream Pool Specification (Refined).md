# **Dream Pool Specification (Refined)**

**Module Path:** `src/dream/pool/`  
 **Linked Modules:** `dream::generation`, `dream::evaluation`, `dream::inject`, `dream::monitor`, `tensor`, `db`

---

## **🧭 Purpose**

The **Dream Pool** serves as the **long-term semantic memory** of the Chromatic Cognition System.  
 It stores, ranks, and retrieves high-coherence `ChromaticTensor` states (dreams) and their associated metadata for **solver self-inquiry** and **conceptual clustering**.

Each dream is a *traceable cognitive artifact* — a measurable record of imagination and response to perturbations from text, images, or factoids.

---

## **⚙️ Configuration**

`[dream.pool]`  
`database_path = "data/dream_pool.sqlite"`  
`max_size_gb = 10.0`  
`coherence_threshold = 0.75     # Minimum coherence for persistence`  
`retrieval_limit = 5`  
`retrieval_mode = "cosine"      # cosine | euclidean | mixed`  
`recency_decay = 0.02           # optional down-weight for older dreams`

---

## **🧩 Core Structures**

### **1\. DreamEntry**

`pub struct DreamEntry {`  
    `pub id: Uuid,`  
    `pub timestamp: DateTime,`  
    `pub factoid_ids: Vec<Uuid>,`  
    `pub image_id: Option<Uuid>,`  
    `pub coherence: f32,`  
    `pub energy: f32,`  
    `pub chroma_signature: [f32; 3],   // average HSV color of final tensor`  
    `pub tensor_blob: Vec<u8>,         // compressed serialized ChromaticTensor`  
    `pub spectral_entropy: f32,        // derived from FFT spectrum`  
    `pub metadata_json: String,        // config + metric context`  
`}`

### **2\. DreamPool Manager**

`pub struct DreamPool {`  
    `db_conn: Connection,`  
    `config: PoolConfig,`  
`}`

`impl DreamPool {`  
    `pub fn new(config: &PoolConfig) -> Self;`

    `/// Store dream if coherence threshold met`  
    `pub fn save_entry(&self, entry: DreamEntry);`

    `/// Retrieve top-N similar dreams by chroma signature and/or metric`  
    `pub fn retrieve_ranked(`  
        `&self,`  
        `query_signature: &[f32; 3],`  
        `mode: &str`  
    `) -> Vec<DreamEntry>;`

    `/// Convert DreamEntry into active ChromaticTensor for reseeding`  
    `pub fn entry_to_tensor(&self, entry: &DreamEntry) -> ChromaticTensor;`

    `/// Optimize internal index for similarity queries`  
    `pub fn optimize_index(&self);`  
`}`

---

## **🧠 Operational Flow**

### **A. Persistence & Indexing**

1. `dream::evaluation` computes `SolverMetrics` for each dream cycle.

2. If `coherence ≥ threshold`, the final `ChromaticTensor` is serialized and stored.

3. The dream’s `chroma_signature` and `spectral_entropy` are recorded as query features.

4. Indexes on timestamp and color vectors are maintained for fast retrieval.

### **B. Retrieval & Self-Inquiry**

1. A **query signature** is generated (e.g., mean chroma of new inputs).

2. `DreamPool::retrieve_ranked()` executes similarity ranking:

   * **Cosine:** directionally similar conceptual dreams.

   * **Euclidean:** absolute color proximity.

   * **Mixed:** top cosine \+ top coherence hybrid set.

3. Older dreams are down-weighted via recency decay:  
    wi=e−λ(tnow−ti)w\_i \= e^{- \\lambda (t\_{now} \- t\_i)}wi​=e−λ(tnow​−ti​)

Retrieved entries are deserialized and mixed into the solver start state via:

 `tensor = mix(&tensor, &pool.entry_to_tensor(&entry));`

4. 

---

## **🧮 Coherence and Spectral Analysis**

### **A. Coherence Definition**

Coherence is defined as **spectral compactness**—the concentration of energy in low-frequency bands of the dream’s Fourier spectrum:

Coherence=1−HspectralHmax\\text{Coherence} \= 1 \- \\frac{H\_{\\text{spectral}}}{H\_{\\text{max}}}Coherence=1−Hmax​Hspectral​​

where HspectralH\_{\\text{spectral}}Hspectral​ is Shannon entropy of normalized amplitude in the frequency domain.

### **B. Implementation**

`fn compute_spectral_entropy(tensor: &ChromaticTensor) -> f32 {`  
    `let spectrum = fft2d(&tensor);`  
    `shannon_entropy(&spectrum.magnitude())`  
`}`

This definition replaces heuristic coherence scores with a **physically grounded measure** of structural order.

---

## **🧩 Chromatic Tokenization (Text-to-Color)**

### **1\. Linguistic-to-Chromatic Mapping**

To ensure interpretable and consistent token colors, text tokens are mapped as:

| Feature | Color Channel | Description |
| ----- | ----- | ----- |
| **Information density** | Saturation | Entropy of token or TF-IDF weight |
| **Sentiment / valence** | Hue | Polarity from −1→1 mapped to warm↔cool tones |
| **Syntactic role** | Brightness | Grammatical prominence (noun/verb high, stopword low) |

### **2\. Implementation Example**

`fn chromatic_token_mapping(token: &str, pos: &str, sentiment: f32, entropy: f32) -> [f32; 3] {`  
    `let hue = sentiment_to_hue(sentiment, pos);`  
    `let saturation = entropy.clamp(0.1, 1.0);`  
    `let brightness = syntax_to_brightness(pos);`  
    `hsv_to_rgb(hue, saturation, brightness)`  
`}`

This produces **human-interpretable factoid hues**:

* Neutral function words → gray

* Abstract or high-impact concepts → vivid colors

* Emotional words → warm/cool shifts

Dream visualization thus conveys conceptual and affective structure in real time.

---

## **🔄 Context-Aware Injection**

Beyond basic blending, the dream engine supports **context-aware perturbations**.

| Mode | Function | Effect |
| ----- | ----- | ----- |
| `mix()` | Weighted average | Stable, smooth combination |
| `add()` | Summation | High-energy “burst” effect |
| `filter()` | Convolution | Texture transfer from injected tensor |
| `mask()` | Selective patch injection | Contextual adaptation based on solver’s violation map |

Example:

`if metrics.violation > 0.3 {`  
    `tensor = mask_inject(&tensor, &factoid_tensor, &violation_mask);`  
`}`

This ensures that injection occurs *where the dream is least coherent*, simulating focused imagination.

---

## **🧬 Conceptual Clustering & Self-Organization**

Stored dreams form clusters in chromatic-semantic space.  
 Over time, the Dream Pool becomes a **self-organizing archive** of ideas, enabling:

* **Cluster Retrieval:** discover color-space neighborhoods (e.g., “blue calm” or “red energetic” dreams).

* **Dream Lineage Tracking:** trace the evolution of conceptual families over multiple runs.

* **Memory Isolation:** seed new dreams only from chosen chromatic clusters.

---

## **🧪 Testing Checklist**

* Only coherent dreams persist beyond threshold.

* FFT-based `spectral_entropy` computation stable and bounded.

* Cosine and mixed-mode retrieval consistent across query vectors.

* Tensor serialization round-trip lossless.

* Recency decay weighting correctly alters ranking order.

* Cross-modal dreams (text \+ image) produce valid composite tensors.

---

## **🔮 Future Extensions**

| Extension | Description |
| ----- | ----- |
| **SpectralTensor API** | Formalize FFT representations for any tensor, enabling frequency-domain manipulation. |
| **Dream Lens** | Render partial visualizations (coherence-only, energy-only) for interpretability dashboards. |
| **Distributed Pooling** | Cluster-aware dream sharing between nodes (multi-agent cognition). |
| **Reconstruction Metrics** | Quantify creative deviation: dream vs. seed dissimilarity index. |

---

### **✅ Summary**

The refined **Dream Pool Specification** establishes:

* A **scientific definition** of coherence grounded in spectral analysis,

* A **linguistically interpretable** chromatic tokenizer,

* A **context-aware injection framework**,

* And a **self-organizing semantic memory** enabling recursive creativity.

This completes the **Memory Layer** of the Chromatic Cognition architecture — connecting *perception*, *imagination*, and *self-reflection* under a transparent and measurable chromatic logic.

