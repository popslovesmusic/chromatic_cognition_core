

# **LEARNER MANIFEST v1.0**

**Role:** Analytical half of the Dreamer–Learner system.  
 **Goal:** Extract structure from Dream Pool entries, evaluate utility, and bias the Dreamer via feedback.

---

### **LOOP**

while active:  
    q \= RetrievalQuery(cfg)  
    set \= iface.retrieve(q)  
    batch \= iface.materialize\_batch(set)  
    x \= decode(batch.tensors, check\_sha=True)  
    f \= learner.extract\_features(x)        \# FFT \+ chroma  
    r \= learner.optimize(f, labels)        \# loss & grads  
    iface.submit\_feedback(r.to\_feedback()) \# normalized Δloss  
    every N steps \-\> iface.synthesize\_bias\_profile()

---

### **RULES**

**Data**

* Use `split="train"` only.

* No reuse within feedback horizon.

* Skip parent/child lineage in same batch.

**Retrieval**

* Maintain chroma dispersion \> τ.

* Deterministic under fixed seed.

* Fill missing with random baseline.

**Preprocessing**

* Fixed mean/std normalization.

* Apply Hann window before FFT.

* All math in f32.

**Optimization**

* Normalize λ-weights (Σλ=1).

* LR∈\[1e-6,1e-2\]; log effective LR.

* Abort on NaN/Inf or monotonic loss↑.

**Feedback**

* Utility \= Δloss vs baseline; clip 5-95 %.

* Contribution ∈ \[0,1\]; log (entry\_id, utility).

**Bias Profile**

* Update every N steps (non-overlapping).

* Limit drift \< 10 %.

* Record top-N contributors for audit.

---

### **MONITOR**

`loss_mean, utility_mean, dispersion, bias_drift, coverage, throughput`  
 Stop if dispersion \< τ for 3 steps or decode fail \> M %.

---

### **TEST CHECKS**

* Deterministic retrieval (fixed seed).

* Split isolation (no leak).

* FFT entropy sanity.

* Utility baseline validity.

* Bias drift cap.

---

**Principle:**

Dreamer imagines → Learner tests → Interface remembers.  
 Maintain determinism, diversity, bounded drift, and traceable feedback.

