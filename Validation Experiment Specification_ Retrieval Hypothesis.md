## **Validation Experiment Specification: Retrieval Hypothesis**

The core hypothesis is that seeding the $\\text{Dream Cycle}$ with a **retrieved high-coherence tensor** that is chromatically similar to the current input will stabilize or accelerate the convergence of the primary solver (e.g., the $\\text{ChromaticNativeSolver}$ or a small Neural Network).

### **1\. Experimental Setup**

The experiment will use an A/B testing methodology over multiple training runs of the color classification task.

| Group | Seed Tensor Generation Method | Purpose |
| :---- | :---- | :---- |
| **A (Control)** | **Random Noise or Zero Tensor** | Baseline performance using standard initialization. |
| **B (Test)** | **Retrieval-Based Seed** | Uses $\\text{SimpleDreamPool}$ to retrieve and blend past successful tensors. |

**Task:** Train the primary solver (e.g., a simple CNN or Native Solver) on a standard color classification dataset (e.g., classifying $10$ common color swatches).

**Duration:** Each run should continue for a fixed number of epochs (e.g., 50 epochs) or until a plateau in validation accuracy is reached.

### **2\. Experimental Procedure**

The standard $\\text{Dream Cycle}$ structure will be followed for each training step:

$$\\text{Input} \\rightarrow \\text{Seed} \\rightarrow \\text{Dream Cycle} \\rightarrow \\text{Solver Update} \\rightarrow \\text{Evaluation}$$

#### **A. Training Step (Both Groups)**

1. **Input:** Load a batch of data (input image/tensor $T\_{input}$) for the solver update.  
2. **Dream Cycle:** Initialize the $\\text{Dream Cycle}$ with a seed tensor $T\_{seed}$.  
3. **Generation:** Run the dream solver for $N$ iterations to produce a final, high-coherence dream $T\_{final}$.  
4. **Evaluation:** Compute $\\text{SolverResult}$ (including $\\text{coherence}$) for $T\_{final}$.  
5. **Store:** $\\text{SimpleDreamPool::add}(T\_{final}, \\text{Result})$.

#### **B. Seed Generation (Specific to Group)**

This step occurs *before* the Dream Cycle in the next training iteration ($i+1$).

| Group | Procedure for Tseed​ at Step i+1 |
| :---- | :---- |
| **A (Control)** | **$T\_{seed} \= T\_{noise}$** (Generate a new tensor filled with Gaussian noise or zeros). |
| **B (Test)** | **Retrieval and Blending:** 1\. Retrieve $K=3$ tensors $T\_{r}$ similar to the current input $T\_{input}$ using $\\text{retrieve\\\_similar}(T\_{input}, 3)$. 2\. **Blend:** $T\_{seed} \= \\text{mix}(T\_{noise}, \\text{mix}(T\_{r1}, \\text{mix}(T\_{r2}, T\_{r3})))$. |

### **3\. Core Metrics and Decision Gates**

We will focus on measuring **training efficiency** and **final quality** to determine if the complexity of the $\\text{Dream Pool}$ is justified.

| Metric | Measurement | Expected Improvement if Hypothesis is Valid |
| :---- | :---- | :---- |
| **Convergence Speed** | Epochs required to reach 90% of final validation accuracy. | **Group B requires fewer epochs.** The solver starts "smarter." |
| **Final Accuracy** | Peak validation accuracy achieved after convergence. | **Group B achieves equal or higher final accuracy.** |
| **Dream Coherence** | Average $\\text{coherence}$ of stored dreams. | **Group B generates higher average coherence** (retrieval encourages stable, successful states). |

#### **Decision Gate 1 (SimpleDreamPool):**

The experiment must demonstrate that **Group B (Retrieval-Based Seeding)** provides a statistically significant advantage over **Group A (Control)** in at least one of the core metrics (Convergence Speed or Final Accuracy).

* **If Yes ($\\text{p} \\le 0.05$):** The retrieval hypothesis is validated. Proceed to Phase 2 (Persistence, FFT).  
* **If No:** The retrieval hypothesis is not validated. Defer or abandon the $\\text{Dream Pool}$ project in favor of higher-priority solver/GPU work.

### **4\. Preparation Checklist**

Before the validation runs can begin, the following must be ensured in the codebase:

1. **$\\text{SimpleDreamPool}$ Integration:** src/dream/pool/simple\_pool.rs must be integrated and compiled.  
2. **$\\text{ChromaticTensor}$ $\\rightarrow$ Mean RGB:** The $\\text{ChromaticTensor}$ object must have a reliable method to compute its mean RGB signature for the $\\text{retrieve\\\_similar}$ query.  
3. **$\\text{mix()}$ Function:** The $\\text{Dream Cycle}$ must expose a working $\\text{mix()}$ function for blending the retrieved tensors into the seed.  
4. **Logging:** The system must log validation accuracy and the average $\\text{coherence}$

