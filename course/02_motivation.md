# MODULE 2: Motivation — Why We Need Muon

## 2.1 The Problem with Adam

Adam works element-wise. Each parameter gets its own adaptive learning rate via the second moment estimate `v_t`. But this means:

1. **It ignores correlations between parameters** — it treats each weight independently
2. **Memory cost** — it stores two state tensors (m and v) per parameter: **2× the model size**
3. **It's solving a "diagonal" approximation** to the true natural gradient

## 2.2 The Ideal: Natural Gradient / Second-Order Methods

The **natural gradient** uses the Fisher information matrix F:

$$
\theta_{t+1} = \theta_t - \eta \cdot F^{-1} \cdot \nabla L(\theta_t)
$$

This accounts for the geometry of the parameter space. But F is enormous (parameters² × parameters²) and inverting it is intractable.

## 2.3 The Shampoo / SOAP Connection

**Shampoo** approximates the full preconditioner by maintaining left and right preconditioners:

For a weight matrix W of shape `m × n`:
$$
\begin{aligned}
L_t &= \beta \cdot L_{t-1} + G_t \cdot G_t^T \quad &(m \times m) \\
R_t &= \beta \cdot R_{t-1} + G_t^T \cdot G_t \quad &(n \times n) \\
\\
\text{Update:} &\quad L_t^{-1/4} \cdot G_t \cdot R_t^{-1/4}
\end{aligned}
$$

**SOAP** (from Meta) modernized Shampoo by running it in the eigenbasis of the preconditioners.

These are powerful but expensive due to the eigendecompositions / matrix inversions.

## 2.4 Muon's Key Insight

Keller Jordan and collaborators realized:

> **If you take the Shampoo preconditioned gradient and look at what it does in the limit of large batch / infinite data, it converges to the polar decomposition of the gradient.**

In other words, the "ideal" Shampoo update direction is just the **orthogonal polar factor** of the gradient.

**Why?** Because:
- Shampoo's preconditioner `L^{-1/4} G R^{-1/4}` normalizes the singular values of G
- In the limit, all singular values become 1
- A matrix with all singular values = 1 is exactly the orthogonal polar factor

So instead of building expensive preconditioners, **just compute the polar decomposition directly!**

## 2.5 Steepest Descent Under the Spectral Norm

There's another beautiful way to motivate Muon. Consider the general optimization step:

$$
\theta_{t+1} = \theta_t - \eta \cdot \operatorname*{argmax}_{\|\Delta\| \le 1} \langle \nabla L, \Delta \rangle
$$

This asks: "what unit-norm direction gives the most decrease in loss?"

The answer depends on the norm:

| Norm | Steepest Descent Direction | Result |
|------|---------------------------|---------|
| L2 (Frobenius) | $G / \|G\|_F$ | Standard gradient descent |
| L$\infty$ (element-wise) | $\text{sign}(G)$ | SignSGD |
| **Spectral (operator) norm** | **Orthogonal polar factor of G** | **Muon** |

**Muon performs steepest descent under the spectral norm!**

This is ideal for weight matrices because:
- It treats the matrix as a **linear operator**, not a bag of independent numbers
- It pushes **all singular values equally**, preventing some directions from being neglected
- It naturally respects the structure of matrix multiplication

## 2.6 Practical Results

On the `modded-nanogpt` speedrun (GPT-2 124M on OpenWebText):
- **Adam baseline**: ~3.28 validation loss in 10K steps
- **Muon**: reaches the same loss in **~5K steps** (roughly 2× fewer steps)
- Wall-clock time is also faster due to simpler computation

## 2.7 Critique of the Course

This course is strong technically, but it can be improved for clarity, evidence quality, and learner usability.

### Strengths

1. The sequence is well designed: motivation -> theory -> algorithm -> implementation -> scaling.
2. Explanations connect intuition to math effectively (especially SVD/polar factor sections).
3. Practical content is substantial: code, tuning guidance, distributed training notes, and exercises.
4. The course stays focused on one core idea and revisits it consistently from different angles.

### Weaknesses

1. Benchmark claims are mostly presented without explicit citations, configs, or reproducibility links.
2. Some sections are dense and assume strong background without enough checkpoints for weaker learners.
3. Notation and assumptions vary by module (for example matrix shape handling and normalization details), which can confuse readers.
4. Failure modes and tradeoffs are under-developed (for example when Muon loses to AdamW, stability issues, or architecture-specific caveats).
5. There is limited assessment scaffolding: exercises are good prompts but lack expected outputs or grading criteria.

### High-Impact Improvements

1. Add a reference block in each module with paper links, code commits, and experiment settings.
2. Add "learning objectives", "prerequisites", and "time estimate" at the top of each module.
3. Add mini checkpoints every 1-2 sections (short conceptual questions with answers in an appendix).
4. Include one full reproducible benchmark recipe (hardware, batch size, tokenizer, schedule, seed, and exact command).
5. Add a dedicated "limitations and open questions" section to balance the strong pro-Muon narrative.

### Overall Assessment

As a technical deep dive, this is a strong and ambitious course. With stronger citation/reproducibility practices and better learner scaffolding, it could move from good expert notes to excellent teaching material.

---
