

# Full Course: The Muon Optimizer

## From First Principles to Production Implementation

---

# Table of Contents

1. **Module 1: Prerequisites & Background**
2. **Module 2: Motivation — Why We Need Muon**
3. **Module 3: Mathematical Foundations**
4. **Module 3.5: SVD for the Muon Optimizer**
5. **Module 4: The Muon Algorithm — Step by Step**
6. **Module 5: Implementation from Scratch**
7. **Module 6: Newton-Schulz Iterations — Deep Dive**
8. **Module 7: Practical Usage & Hyperparameter Tuning**
9. **Module 8: Muon vs Other Optimizers**
10. **Module 9: Distributed / Multi-GPU Muon**
11. **Module 10: Advanced Topics & Current Research**
12. **Appendix A: Quick Reference Card**
13. **Appendix B: Exercises**
14. **Appendix C: FAQ**

---

# MODULE 1: Prerequisites & Background

## 1.1 What is Muon?

**Muon** stands for **M**oment**U**m **O**rthogonalized by **N**ewton-schulz.

It is an optimizer created by **Keller Jordan** (and collaborators) in the context of the `modded-nanogpt` speedrun project (late 2024). It was designed to train transformers significantly faster than Adam/AdamW.

**Core idea in one sentence:**
> Take the Nesterov momentum buffer, then orthogonalize it via an approximate polar decomposition (computed cheaply using Newton-Schulz iterations), and use that as the update.

## 1.2 Prerequisites You Should Know

Before diving in, make sure you understand:

### 1.2.1 Gradient Descent

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla L(\theta_t)
$$

The simplest optimization: move parameters in the direction of steepest descent.

### 1.2.2 Momentum (Polyak / Heavy Ball)

$$
\begin{aligned}
m_{t+1} &= \beta \cdot m_t + \nabla L(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta \cdot m_{t+1}
\end{aligned}
$$

Momentum accumulates past gradients to accelerate training and dampen oscillations.

### 1.2.3 Nesterov Accelerated Gradient (NAG)

$$
\begin{aligned}
m_{t+1} &= \beta \cdot m_t + \nabla L(\theta_t - \eta \cdot \beta \cdot m_t) \quad \text{\# "look-ahead" gradient} \\
\theta_{t+1} &= \theta_t - \eta \cdot m_{t+1}
\end{aligned}
$$

Or equivalently (the form Muon uses):
$$
\begin{aligned}
m_{t+1} &= \beta \cdot m_t + \nabla L(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta \cdot (\beta \cdot m_{t+1} + \nabla L(\theta_t)) \quad \text{\# Nesterov extrapolation}
\end{aligned}
$$

### 1.2.4 Adam / AdamW

$$
\begin{aligned}
m_t &= \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t \quad &\text{\# first moment} \\
v_t &= \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 \quad &\text{\# second moment} \\
\hat{m}_t &= m_t / (1 - \beta_1^t) \quad &\text{\# bias correction} \\
\hat{v}_t &= v_t / (1 - \beta_2^t) \\
\theta_t &= \theta_{t-1} - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
\end{aligned}
$$

Adam is the current default for deep learning. AdamW adds decoupled weight decay.

### 1.2.5 Key Linear Algebra Concepts

| Concept | Definition | Why It Matters |
|---------|-----------|----------------|
| **SVD** | $A = U \Sigma V^T$ | Decomposes any matrix into rotations + scaling |
| **Singular Values** | Diagonal of $\Sigma$ | Measure "how much" each direction is stretched |
| **Orthogonal Matrix** | $Q^T Q = Q Q^T = I$ | All singular values = 1 |
| **Spectral Norm** | $\|A\|_2 = \sigma_{\max}(A)$ | Largest singular value |
| **Frobenius Norm** | $\|A\|_F = \sqrt{\sum \sigma_i^2}$ | "Size" of a matrix |
| **Polar Decomposition** | $A = Q \cdot S$ | $Q$ orthogonal, $S$ symmetric positive semi-definite |

### 1.2.6 The Polar Decomposition (Critical for Muon)

Any matrix **G** with shape $m \times n$ (where $m \ge n$) can be decomposed as:

$$
\begin{aligned}
G &= U \Sigma V^T \quad \text{(SVD)} \\
  &= (U V^T)(V \Sigma V^T) \\
  &= Q \cdot S \quad \text{(Polar Decomposition)}
\end{aligned}
$$

Where:
- **$Q = U V^T$** is the **orthogonal polar factor** (the "direction" of $G$)
- **$S = V \Sigma V^T$** is symmetric positive semi-definite (the "magnitude")

**The orthogonal polar factor $Q$ is the closest orthogonal matrix to $G$** (in Frobenius norm).

This is the key object Muon computes.

---



---

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


---

# MODULE 3.5: SVD for the Muon Optimizer – What You Need to Know

![SVD intuition for Muon: raw gradient has uneven singular values, while Muon uses UV^T with equalized singular values.](images/03a_svd_for_muon_intro.png)

*Figure: Muon keeps the SVD directions ($U, V^T$) and removes scale imbalance by replacing $\Sigma$ with identity.*

## Introduction

When studying the Muon optimizer, you will frequently encounter the **Singular Value Decomposition (SVD)**. While SVD is a deep topic in linear algebra, you don't need a PhD in math to understand how Muon works. 

This lesson covers *exactly* what you need to know about SVD to grasp why Muon converges so fast, and why we can't just use standard SVD in PyTorch to run it.

## 1. The Intuition of SVD

Every matrix (like a weight gradient $G$ in a neural network) can be thought of as applying a geometric transformation to a space. SVD states that *any* matrix $G$ can be broken down into three simpler steps:

$$
G = U \Sigma V^T
$$

1. **$V^T$**: A rotation (or reflection) of the input space.
2. **$\Sigma$**: A scaling step. This is a diagonal matrix containing the **singular values** ($\sigma_1, \sigma_2, \dots$). It stretches or squishes the space along the new axes.
3. **$U$**: Another rotation into the output space.

**Key Idea**: The singular values ($\Sigma$) tell us the "magnitude" or "steepness" of the gradient in different orthogonal directions. $U$ and $V^T$ tell us what those directions actually are.

![Step-by-step geometric view of SVD: unit circle transformed by V^T, then Sigma scaling, then U rotation.](images/03a_svd_geometry_steps.png)

*Figure: SVD decomposes one complex transformation into rotate/reflect, axis scaling, and rotate again.*

## 2. The Problem with Standard Gradients

In standard Gradient Descent or Adam, we update our weights using the raw gradient matrix $G$.

Because standard gradient updates are proportional to the raw matrix $G = U \Sigma V^T$, the update size in each direction depends on its singular value $\sigma_i$. 

- Directions with **large singular values** get massive updates.
- Directions with **small singular values** get tiny, slow updates.

This causes a problem known as poor conditioning. The optimizer spends all of its energy bouncing back and forth along the "steep" directions (large $\sigma$), while making agonizingly slow progress along the flat directions (small $\sigma$).

![Ill-conditioned optimization landscape showing zig-zag path for standard gradients versus smoother progress with balanced directional updates.](images/03a_conditioning_trajectory.png)

*Figure: With poor conditioning, raw gradients zig-zag along steep axes, while balanced directional updates move more directly toward the minimum.*

## 3. The Muon Solution: Equalizing the Singular Values

What if we could update the weights in the correct *directions* ($U$ and $V^T$), but treat every direction with equal importance? 

Mathematically, we could just replace the scaling matrix $\Sigma$ with the Identity matrix $I$ (where all values on the diagonal are exactly 1). 

If we do this, the update matrix becomes:

$$
\Delta = U I V^T = U V^T
$$

The matrix $Q = U V^T$ is known as the **orthogonal polar factor** of $G$. 
- It preserves all the directional information of the gradient.
- It completely removes the scaling imbalances. Every orthogonal dimension gets updated with exactly the same magnitude.

This is fundamentally why Muon uses orthogonalized updates: it forces the network to learn equally fast in all directions, dramatically improving convergence speed across the loss landscape.

## 4. Why Not Just Run `torch.linalg.svd()`?

To get $U V^T$, the naive approach is to simply calculate the exact SVD of the momentum/gradient matrix during training:

```python
# Naive (and too slow) approach
U, S, Vt = torch.linalg.svd(G, full_matrices=False)
update_matrix = U @ Vt
```

**The Catch**: Computing the exact SVD is extremely computationally intensive. The time complexity is $\mathcal{O}(m n^2)$, where $m$ and $n$ are the dimensions of the weight matrix. 

For a standard large language model layer (e.g., a $4096 \times 4096$ projection matrix), forcing the CPU/GPU to compute the exact SVD on every single forward-backward pass would be so slow that any step-wise convergence benefits would be completely erased by massive wall-clock training times. 

## 5. Newton-Schulz to the Rescue

We need $U V^T$, but we can't afford the time it takes to compute $U$, $\Sigma$, and $V^T$ individually.

This is where the **Newton-Schulz iteration** comes in (which is covered in detail in Module 6). Newton-Schulz is an iterative mathematical algorithm that takes the matrix $G$ and effectively "squashes" its singular values $\Sigma$ towards 1 using nothing but a series of fast matrix multiplications. 

After a few brief iterations (usually just 5 or 6), it outputs a matrix that is extremely close to $U V^T$. 
- **No SVD computation needed.**
- **Relies purely on highly parallelizable matrix multiplications.**
- **Extremely hardware-friendly for GPU Tensor Cores.**

![Newton-Schulz intuition plot showing different initial singular values rapidly converging toward 1 across iterations.](images/03a_newton_schulz_singular_value_squash.png)

*Figure: Newton-Schulz-style iterations repeatedly squash singular values toward 1, approximating the $UV^T$ behavior Muon needs.*

## Summary Cheat Sheet

- **$G = U \Sigma V^T$**: The standard gradient, containing directions ($U, V^T$) and magnitudes ($\Sigma$).
- **The Problem**: Raw gradients prioritize directions with large magnitudes, leading to slow training on flat dimensions.
- **The Ideal Update**: $U V^T$. This treats all directions equally (setting all $\sigma_i = 1$).
- **The Obstacle**: Exact SVD is far too slow for large neural networks.
- **The Muon Innovation**: Use Newton-Schulz iterations to cheaply and quickly approximate $U V^T$ using only matrix multiplications.


---

# MODULE 3: Mathematical Foundations

## 3.1 Singular Value Decomposition (SVD) — Review

For any matrix $G \in \mathbb{R}^{m \times n}$ with $m \ge n$:

$$
G = U \Sigma V^T
$$

Where:
- $U \in \mathbb{R}^{m \times m}$ is orthogonal (left singular vectors)
- $\Sigma \in \mathbb{R}^{m \times n}$ has singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_n \ge 0$ on the diagonal
- $V \in \mathbb{R}^{n \times n}$ is orthogonal (right singular vectors)

## 3.2 Polar Decomposition — Formal Treatment

### Definition

For $G \in \mathbb{R}^{m \times n}$ ($m \ge n$) of full rank:

$$
G = Q \cdot P
$$

Where:
- $Q \in \mathbb{R}^{m \times n}$ is a **partial isometry** ($Q^T Q = I_n$)
- $P \in \mathbb{R}^{n \times n}$ is **symmetric positive definite**

### Relation to SVD

$$
\begin{aligned}
Q &= U V^T \quad &\text{(the orthogonal polar factor)} \\
P &= V \Sigma V^T \quad &\text{(the positive semi-definite factor)}
\end{aligned}
$$

### Key Property

**$Q$ is the nearest orthogonal matrix to $G$:**

$$
Q = \operatorname*{argmin}_{Z: Z^TZ = I} \|G - Z\|_F
$$

This can also be written as:

$$
Q = G (G^TG)^{-1/2}
$$

## 3.3 Why Orthogonal Updates?

Consider a weight matrix $W \in \mathbb{R}^{m \times n}$ in a neural network. When we update:

$$
W_{t+1} = W_t - \eta \cdot \Delta
$$

If $\Delta$ is the orthogonal polar factor of the gradient:
- **All directions get equal treatment**: $\sigma_i(\Delta) = 1$ for all $i$
- **No "rich get richer" problem**: unlike raw gradients where large singular value directions dominate
- **Scale-free updates**: the update magnitude is determined by $\eta$ alone, not by gradient magnitude

## 3.4 Steepest Descent — Formal Proof

**Theorem**: Let $G \in \mathbb{R}^{m \times n}$ have SVD $G = U\Sigma V^T$. Then:

$$
\operatorname*{argmax}_{\|\Delta\|_2 \le 1} \langle G, \Delta \rangle_F = UV^T
$$

where $\|\cdot\|_2$ is the spectral (operator) norm and $\langle \cdot,\cdot \rangle_F$ is the Frobenius inner product (trace inner product).

**Proof**:

$$
\langle G, \Delta \rangle_F = \text{tr}(G^T\Delta) = \text{tr}(V\Sigma U^T\Delta)
$$

Let $M = U^T\Delta V$, so $\Delta = UMV^T$. Since $\|\Delta\|_2 \le 1$, we have $\|M\|_2 \le 1$ (singular values of $M \le 1$).

$$
\text{tr}(V\Sigma U^T \cdot UMV^T) = \text{tr}(V\Sigma MV^T) = \text{tr}(\Sigma M) = \sum_i \sigma_i m_{ii}
$$

This is maximized when all $m_{ii} = 1$, i.e., $M = I$, giving:

$$
\Delta^* = UIV^T = UV^T
$$

This is the orthogonal polar factor. ∎

## 3.5 Computing the Polar Factor

Three main approaches:

### 3.5.1 Via SVD (Exact but Expensive)
```python
U, S, Vt = torch.linalg.svd(G, full_matrices=False)
Q = U @ Vt
```
**Cost**: $\mathcal{O}(mn^2)$ for $m \ge n$. Too expensive for every optimization step.

### 3.5.2 Via Matrix Iteration: Newton's Method
The classic iteration for computing `sign(A)` or the polar factor:

$$
\begin{aligned}
X_0 &= G / \|G\| \\
X_{k+1} &= \frac{3}{2} X_k - \frac{1}{2} X_k X_k^T X_k \quad \text{(for thin matrices)}
\end{aligned}
$$

This converges quadratically to the polar factor Q. Each iteration is a matrix multiply.

### 3.5.3 Via Newton-Schulz Iterations (What Muon Uses)
Higher-order variants that converge faster in fewer iterations. Muon uses a **quintic** (5th order) variant. We'll cover this in detail in Module 6.

---



---

# MODULE 4: The Muon Algorithm — Step by Step

## 4.1 The Algorithm

Here's the complete Muon algorithm:

$$
\begin{aligned}
&\textbf{Algorithm: Muon Optimizer} \\
&\rule{10cm}{0.4pt} \\
&\textbf{Input: } \text{learning rate } \eta, \text{ momentum } \beta, \text{ weight decay } \lambda, \text{ Newton-Schulz steps } K, \text{ parameters } \theta_0 \\
&\textbf{Initialize: } \text{momentum buffer } m_0 = 0 \\
\\
&\textbf{For } t = 1, 2, 3, \dots: \\
&\quad \textbf{1. Compute gradient: } g_t = \nabla L(\theta_t) \\
\\
&\quad \textbf{2. Nesterov momentum:} \\
&\qquad m_t = \beta \cdot m_{t-1} + g_t \\
&\qquad n_t = \beta \cdot m_t + g_t \qquad \text{(\# Nesterov "look-ahead")} \\
\\
&\quad \textbf{3. Orthogonalize via Newton-Schulz:} \\
&\qquad \textbf{If } \theta \text{ is a 2D+ weight matrix:} \\
&\qquad \quad \text{Reshape } n_t \text{ to 2D (if needed)} \\
&\qquad \quad \text{Ensure rows } \ge \text{cols (transpose if needed)} \\
&\qquad \quad Q_t = \text{NewtonSchulz}(n_t, K \text{ steps}) \\
&\qquad \quad \text{Reshape } Q_t \text{ back to original shape} \\
&\qquad \textbf{Else:} \\
&\qquad \quad \text{(Use a different optimizer like Adam for 1D params)} \\
\\
&\quad \textbf{4. Update with weight decay:} \\
&\qquad \theta_{t+1} = (1 - \eta \cdot \lambda) \cdot \theta_t - \eta \cdot Q_t \\
&\rule{10cm}{0.4pt}
\end{aligned}
$$

## 4.2 Step-by-Step Walkthrough

Let's trace through one step for a weight matrix W of shape (768, 512):

### Step 1: Gradient
```python
g = torch.autograd.grad(L, W)  # shape: (768, 512)
```

### Step 2: Nesterov Momentum
```
m = 0.95 * m_prev + g          # accumulate momentum
nesterov = 0.95 * m + g        # look-ahead estimate
```

The Nesterov term combines the current momentum with the current gradient to get a "preview" of where momentum is heading.

### Step 3: Orthogonalize
```python
# nesterov has shape (768, 512), already rows >= cols ✓
# Normalize:
X = nesterov / (torch.linalg.norm(nesterov) + eps) * math.sqrt(768)    # scale for numerical stability

# Apply K=5 Newton-Schulz iterations:
for i in range(5):
    A = X @ X.T                  # (768, 768)
    X = a*X + b*(A @ X) + c*(A @ A @ X)  # quintic update

# Result: X is approximately orthogonal polar factor of nesterov
Q = X                           # shape: (768, 512)
```

### Step 4: Update
$$
W_{t+1} = (1 - \text{lr} \cdot \text{wd}) \cdot W_t - \text{lr} \cdot Q_t
$$

## 4.3 Why Nesterov Momentum?

Muon applies Nesterov momentum **before** the orthogonalization step. This is important:

1. **Momentum accumulates gradient history** — gives a better signal than a single noisy gradient
2. **Nesterov specifically** provides a "lookahead" that is empirically better than Polyak momentum
3. **Orthogonalizing the momentum buffer** (not the raw gradient) means we're orthogonalizing a smoother, more reliable signal

## 4.4 What Happens to Different Parameter Types?

Muon is designed for **2D weight matrices**. Other parameters need different treatment:

| Parameter Type | Shape | Optimizer |
|---------------|-------|-----------|
| Linear weights | (out, in) | **Muon** ✓ |
| Conv weights | (out, in, k, k) | **Muon** (reshape to 2D) ✓ |
| Embedding weights | (vocab, dim) | Adam/AdamW |
| LayerNorm scale | (dim,) | Adam/AdamW |
| Biases | (dim,) | Adam/AdamW |
| Final LM head | (vocab, dim) | Adam/AdamW |

The embedding and final head are typically excluded because:
- They deal with the vocabulary dimension which is "discrete"
- Their gradients have very different structure (often very sparse)

## 4.5 The Role of Weight Decay

Weight decay in Muon is **decoupled** (like AdamW), applied directly to the parameters:

$$
W_{t+1} = (1 - \text{lr} \cdot \text{wd}) \cdot W_t - \text{lr} \cdot Q_t
$$

This shrinks the weights independently of the gradient-based update. Typical values are small (e.g., 0.01 to 0.1).

---



---

# MODULE 5: Implementation from Scratch

## 5.1 Minimal Muon Implementation

```python
import torch
from torch.optim import Optimizer
import torch.distributed as dist


def newton_schulz_5(G, steps=5):
    """
    Compute the orthogonal polar factor of G using 
    Newton-Schulz iterations (quintic variant).
    
    Args:
        G: Input matrix of shape (m, n) where m >= n
        steps: Number of Newton-Schulz iterations
    
    Returns:
        Approximate orthogonal polar factor of G
    """
    assert G.shape[0] >= G.shape[1], "Need rows >= cols"
    
    # Quintic polynomial coefficients (optimized for convergence)
    # These are tuned so the combined iteration converges over 
    # the singular value range [0.6, 1.4] in ~5 steps
    a, b, c = (3.4445, -4.7750,  2.0315)
    # Alternative coefficients for later iterations:
    # (11.3168, -20.3300, 9.7132)  
    # (8.4749,  -13.9590, 6.1843)
    
    # Normalize so singular values are near 1
    X = G / (G.norm() + 1e-7)
    
    # Transpose if needed to make it "tall-skinny" for efficiency
    if G.shape[0] < G.shape[1]:
        X = X.T
    
    for _ in range(steps):
        A = X @ X.T                           # (m, m)
        B = A @ X                             # (m, n) = A @ X
        X = a * X + b * B + c * (A @ B)      # quintic update
    
    return X


class Muon(Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    For use on 2D weight matrices. Should be combined with
    a separate optimizer (e.g., AdamW) for 1D parameters.
    
    Args:
        params: Parameters to optimize (should be 2D weight matrices)
        lr: Learning rate (default: 0.02)
        momentum: Nesterov momentum coefficient (default: 0.95)
        weight_decay: Decoupled weight decay (default: 0.0)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
    """
    
    def __init__(self, params, lr=0.02, momentum=0.95, 
                 weight_decay=0.0, ns_steps=5):
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay,
            ns_steps=ns_steps
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = group['weight_decay']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                g = p.grad
                state = self.state[p]
                
                # ---- State initialization ----
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                state['step'] += 1
                buf = state['momentum_buffer']
                
                # ---- Nesterov momentum ----
                buf.mul_(momentum).add_(g)
                nesterov = buf * momentum + g
                # nesterov = momentum * (momentum * buf_old + g) + g
                # This is the standard Nesterov formulation
                
                # ---- Reshape to 2D if necessary ----
                original_shape = nesterov.shape
                if nesterov.dim() > 2:
                    nesterov = nesterov.view(nesterov.shape[0], -1)
                
                # Ensure rows >= cols
                transposed = False
                if nesterov.shape[0] < nesterov.shape[1]:
                    nesterov = nesterov.T
                    transposed = True
                
                # ---- Newton-Schulz orthogonalization ----
                Q = newton_schulz_5(nesterov, steps=ns_steps)
                
                # ---- Undo transpose if applied ----
                if transposed:
                    Q = Q.T
                
                # ---- Reshape back ----
                Q = Q.view(original_shape)
                
                # ---- Apply update with weight decay ----
                p.mul_(1 - lr * wd)
                p.add_(Q, alpha=-lr)
        
        return loss
```

## 5.2 Usage Example — Training a Small Transformer

```python
import torch
import torch.nn as nn

# ---- Define a simple model ----
class SmallTransformer(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=12, n_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=n_heads, 
                dim_feedforward=4*d_model,
                batch_first=True,
                norm_first=True
            ) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)

model = SmallTransformer().cuda()

# ---- Separate parameters for Muon vs AdamW ----
muon_params = []
adam_params = []

for name, param in model.named_parameters():
    if param.dim() >= 2 and 'embedding' not in name and 'lm_head' not in name:
        muon_params.append(param)
    else:
        adam_params.append(param)

print(f"Muon params: {sum(p.numel() for p in muon_params):,}")
print(f"Adam params: {sum(p.numel() for p in adam_params):,}")

# ---- Create optimizers ----
optimizer_muon = Muon(
    muon_params, 
    lr=0.02,
    momentum=0.95,
    weight_decay=0.01,
    ns_steps=5
)

optimizer_adam = torch.optim.AdamW(
    adam_params,
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1
)

# ---- Training loop ----
for step in range(num_steps):
    x, y = get_batch()  # your data loading
    
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        y.view(-1)
    )
    
    loss.backward()
    
    # Clip gradients (optional, Muon is naturally bounded)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer_muon.step()
    optimizer_adam.step()
    
    optimizer_muon.zero_grad()
    optimizer_adam.zero_grad()
    
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
```

## 5.3 Production Implementation (with Optimizations)

```python
import torch
from torch.optim import Optimizer


@torch.compile
def newton_schulz_5_compiled(G, steps=5):
    """
    Optimized Newton-Schulz with torch.compile and 
    varying coefficients per iteration.
    """
    # Three sets of coefficients, optimized for convergence
    # over singular value range [0.6, 1.4] after normalization
    coeffs = [
        (3.4445, -4.7750,  2.0315),
        (11.3168, -20.3300, 9.7132),
        (8.4749, -13.9590, 6.1843),
        (8.4749, -13.9590, 6.1843),
        (8.4749, -13.9590, 6.1843),
    ]
    
    # Normalize: scale so Frobenius norm ≈ sqrt(nrows)
    # This puts singular values near 1
    X = G.bfloat16()
    nrows = X.shape[0]
    X = X / (X.norm() + 1e-7) * (nrows ** 0.5)
    
    for i in range(steps):
        a, b, c = coeffs[min(i, len(coeffs)-1)]
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * (A @ B)
    
    return X.to(G.dtype)


class MuonOptimized(Optimizer):
    """
    Production Muon with:
    - torch.compile for Newton-Schulz
    - Per-iteration varying coefficients
    - Proper handling of different parameter shapes
    - Optional gradient accumulation support
    """
    
    def __init__(self, params, lr=0.02, momentum=0.95,
                 weight_decay=0.0, ns_steps=5, nesterov=True):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            ns_steps=ns_steps, nesterov=nesterov
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']
            ns_steps = group['ns_steps']
            nesterov = group['nesterov']
            
            # Collect all updates, then apply
            # (enables potential future FSDP/DDP optimizations)
            updates = []
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                g = p.grad
                
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                state['step'] += 1
                buf = state['momentum_buffer']
                
                # Momentum update
                buf.mul_(mu).add_(g)
                
                if nesterov:
                    update = g + mu * buf
                else:
                    update = buf.clone()
                
                # Handle reshaping
                orig_shape = update.shape
                if update.dim() == 1:
                    # Skip 1D params — shouldn't be in Muon group
                    # But handle gracefully: just use sign
                    update = update.sign()
                else:
                    if update.dim() > 2:
                        update = update.view(update.shape[0], -1)
                    
                    transposed = False
                    if update.shape[0] < update.shape[1]:
                        update = update.T
                        transposed = True
                    
                    update = newton_schulz_5_compiled(update, ns_steps)
                    
                    if transposed:
                        update = update.T
                    
                    update = update.view(orig_shape)
                
                # Scale the update to have the right magnitude
                # The polar factor has Frobenius norm = sqrt(min(m,n))
                # We want the update to have norm proportional to sqrt(numel)
                scale = max(1, update.shape[0] / update.shape[1]) ** 0.5
                
                updates.append((p, update, scale))
            
            # Apply all updates
            for p, update, scale in updates:
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr * scale)
        
        return loss
```

## 5.4 Combined Optimizer Helper

```python
def create_muon_optimizer(model, muon_lr=0.02, adam_lr=3e-4,
                          muon_momentum=0.95, weight_decay=0.01,
                          adam_betas=(0.9, 0.95)):
    """
    Creates Muon + AdamW optimizer pair for a transformer model.
    
    Returns a single object that manages both optimizers.
    """
    
    muon_params = []
    adam_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Use Muon for 2D weight matrices (except embeddings)
        if (param.dim() >= 2 
            and 'embedding' not in name 
            and 'embed' not in name
            and 'lm_head' not in name
            and 'head' not in name):
            muon_params.append(param)
        else:
            adam_params.append(param)
    
    print(f"Muon: {len(muon_params)} params, "
          f"{sum(p.numel() for p in muon_params)/1e6:.1f}M elements")
    print(f"Adam: {len(adam_params)} params, "
          f"{sum(p.numel() for p in adam_params)/1e6:.1f}M elements")
    
    optimizers = []
    
    if muon_params:
        optimizers.append(Muon(
            muon_params, lr=muon_lr, momentum=muon_momentum,
            weight_decay=weight_decay
        ))
    
    if adam_params:
        optimizers.append(torch.optim.AdamW(
            adam_params, lr=adam_lr, betas=adam_betas,
            weight_decay=weight_decay
        ))
    
    return CombinedOptimizer(optimizers)


class CombinedOptimizer:
    """Wraps multiple optimizers into a single interface."""
    
    def __init__(self, optimizers):
        self.optimizers = optimizers
    
    def step(self):
        for opt in self.optimizers:
            opt.step()
    
    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]
    
    def load_state_dict(self, state_dicts):
        for opt, sd in zip(self.optimizers, state_dicts):
            opt.load_state_dict(sd)
    
    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups
```

---



---

# MODULE 6: Newton-Schulz Iterations — Deep Dive

## 6.1 The Core Mathematical Problem

Given a matrix $G \in \mathbb{R}^{m \times n}$, compute:

$$
Q = U V^T \quad \text{(orthogonal polar factor)}
$$

where $G = U \Sigma V^T$ is the SVD. Direct SVD costs $\mathcal{O}(mn^2)$, which is too expensive for every optimizer step.

## 6.2 Classical Newton Iteration for Polar Decomposition

The standard Newton iteration:

$$
\begin{aligned}
X_0 &= G / \|G\|_2 \\
X_{k+1} &= \frac{1}{2}(X_k + X_k^{-T}) \quad \text{(for square matrices)}
\end{aligned}
$$

This converges quadratically but requires matrix inversion.

For rectangular matrices, we use the equivalent:

$$
X_{k+1} = \frac{1}{2}(3X_k - X_k X_k^T X_k)
$$

This is inversion-free and only requires matrix multiplications!

## 6.3 Understanding the Iteration as a Scalar Function

The key insight: the Newton-Schulz iteration acts **independently on each singular value**.

If $G = U \Sigma V^T$, then at each step, each singular value $\sigma_i$ is mapped by a scalar function:

$$
\sigma_i \to f(\sigma_i)
$$

For the cubic iteration $X \leftarrow (3X - X X^T X) / 2$:
$$
f(\sigma) = \frac{3\sigma - \sigma^3}{2}
$$

**Goal**: $f$ should map any positive $\sigma \to 1$ (since the polar factor has all singular values = 1).

Let's check:
- $f(1) = (3 - 1) / 2 = 1$ ✓ (fixed point)
- $f(0.5) = (1.5 - 0.125) / 2 = 0.6875$ (getting closer to 1)
- $f(0.9) = (2.7 - 0.729) / 2 = 0.9855$ (getting closer to 1)

The iteration converges if $\sigma \in (0, \sqrt{3})$, which is ensured by normalization.

## 6.4 The Quintic (5th Order) Variant Used in Muon

Muon uses a **higher-order** polynomial iteration:

$$
X_{k+1} = a \cdot X + b \cdot (X X^T) X + c \cdot (X X^T)^2 X
$$

In terms of scalar singular values:
$$
f(\sigma) = a \cdot \sigma + b \cdot \sigma^3 + c \cdot \sigma^5
$$

This is a **quintic polynomial** that maps singular values toward 1.

### Why quintic?

The higher the degree, the flatter the function near the fixed point $\sigma = 1$, meaning **faster convergence**. Compare:

| Method | Polynomial degree | Convergence order |
|--------|------------------|-------------------|
| Cubic (standard NS) | 3 | Quadratic |
| Quintic (Muon's NS) | 5 | Cubic |
| Septic (possible) | 7 | Quartic |

More specifically, we need:
$$
\begin{aligned}
f(1) &= 1 \quad &\text{(fixed point)} \\
f'(1) &= 0 \quad &\text{(superlinear convergence)} \\
f''(1) &= 0 \quad &\text{(even faster, for the quintic)}
\end{aligned}
$$

From $f(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$:
$$
\begin{aligned}
f(1) &= a + b + c = 1 \\
f'(1) &= a + 3b + 5c = 0 \\
f''(1) &= 6b + 20c = 0
\end{aligned}
$$

This gives: $b = -10c/3$, $a = 1 - b - c = 1 + 10c/3 - c = 1 + 7c/3$.

We still have one free parameter (c) that can be optimized for the convergence rate over a specific range of initial singular values.

## 6.5 Coefficient Optimization

The Muon authors optimized the coefficients to maximize convergence speed over the initial singular value range they expect after normalization.

After normalizing $X_0 = G / \|G\|_F \cdot \sqrt{\text{nrows}}$, the singular values are distributed roughly in $[0.5, 1.5]$.

The optimized coefficient sets:

```python
# First iteration: broad convergence, brings σ from [0.5, 1.5] closer to 1
(3.4445, -4.7750, 2.0315)

# Second iteration: tighter convergence  
(11.3168, -20.3300, 9.7132)

# Third+ iteration: polishing
(8.4749, -13.9590, 6.1843)
```

Let's verify the first set satisfies our constraints:
```python
a + b + c = 3.4445 + (-4.7750) + 2.0315 = 0.701  
# Hmm, not exactly 1! 
```

Wait — the coefficients used in practice don't exactly satisfy $f(1)=1$ because they're numerically optimized over a range, not just at the fixed point. The iteration is designed to converge to a matrix with all singular values = 1 after multiple steps, even if each individual step doesn't exactly preserve $\sigma = 1$.

## 6.6 Visualizing the Iterations

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_ns_iteration():
    """Visualize how Newton-Schulz maps singular values."""
    sigma = np.linspace(0.1, 2.0, 1000)
    
    # Three coefficient sets
    coeffs = [
        (3.4445, -4.7750, 2.0315),
        (11.3168, -20.3300, 9.7132),
        (8.4749, -13.9590, 6.1843),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (a, b, c) in enumerate(coeffs):
        f = a * sigma + b * sigma**3 + c * sigma**5
        
        axes[idx].plot(sigma, f, 'b-', linewidth=2, label='f(σ)')
        axes[idx].plot(sigma, sigma, 'k--', alpha=0.3, label='identity')
        axes[idx].axhline(y=1, color='r', linestyle=':', alpha=0.5, label='target')
        axes[idx].axvline(x=1, color='r', linestyle=':', alpha=0.5)
        axes[idx].set_xlim(0, 2)
        axes[idx].set_ylim(-0.5, 2)
        axes[idx].set_xlabel('σ (input singular value)')
        axes[idx].set_ylabel('f(σ) (output singular value)')
        axes[idx].set_title(f'Iteration {idx+1}: a={a}, b={b}, c={c}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ns_iterations.png', dpi=150)
    plt.show()

    # Show convergence over multiple iterations
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    initial_sigmas = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8]
    
    for s0 in initial_sigmas:
        trajectory = [s0]
        s = s0
        for i in range(5):
            a, b, c = coeffs[min(i, len(coeffs)-1)]
            s = a * s + b * s**3 + c * s**5
            trajectory.append(s)
        ax2.plot(trajectory, 'o-', label=f'σ₀={s0}', markersize=4)
    
    ax2.axhline(y=1, color='red', linestyle='--', label='target')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Singular value')
    ax2.set_title('Convergence of Singular Values over NS Iterations')
    ax2.legend(ncol=3)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ns_convergence.png', dpi=150)
    plt.show()

plot_ns_iteration()
```

## 6.7 Computational Cost Analysis

For a matrix $G$ of shape $(m, n)$ with $m \ge n$:

**Per Newton-Schulz iteration:**
```python
A = X @ X.T         # Cost: O(m²n)    → (m, n) × (n, m) = (m, m)
B = A @ X           # Cost: O(m²n)    → (m, m) × (m, n) = (m, n)  
C = A @ B           # Cost: O(m²n)    → (m, m) × (m, n) = (m, n)
X = a*X + b*B + c*C # Cost: O(mn)     → element-wise
```

**Total per iteration**: $\mathcal{O}(m^2 n)$
**Total for $K$ iterations**: $\mathcal{O}(K \cdot m^2 n)$

**Compare to SVD**: $\mathcal{O}(mn^2)$ for $m \ge n$

For a typical linear layer (768, 768):
- NS iteration: $\mathcal{O}(K \times 768^3) \approx 5 \times 4.5 \times 10^8 \approx 2.3 \times 10^9$ FLOPs
- SVD: $\mathcal{O}(768^3) \approx 4.5 \times 10^8$ FLOPs

But! NS iterations are:
1. **GPU-friendly**: just matrix multiplies (highly optimized)
2. **Can use bf16/fp16**: lower precision is fine
3. **Easily parallelizable**: standard GEMM operations

## 6.8 Numerical Considerations

### Normalization Strategy

Before starting NS iterations, we need to normalize G so its singular values are near 1:

```python
# Option 1: Frobenius norm (most common in Muon)
X = G / G.norm()                    # σ_max(X) ≤ 1

# Option 2: Scale to expected norm
X = G / G.norm() * (nrows ** 0.5)   # E[σ²] ≈ 1 for random matrices

# Option 3: Spectral norm estimate
X = G / estimated_spectral_norm(G)
```

Option 2 is what Muon uses. The intuition: for a random matrix of shape $(m, n)$, the Frobenius norm is approximately $\sqrt{mn}$, so dividing by the norm and multiplying by $\sqrt{m}$ makes the average singular value $\approx 1$.

### Precision

Newton-Schulz iterations are numerically stable in **bfloat16**, which is important for:
- Memory efficiency (half the memory of fp32)
- Speed on modern GPUs (tensor cores)

```python
def newton_schulz_bf16(G, steps=5):
    X = G.bfloat16()
    # ... iterations in bf16 ...
    return X.to(G.dtype)
```

## 6.9 Alternative: The "Transposed" Formulation

When m < n (wide matrix), we can either:

1. **Transpose**: Work with $G^T$ (shape $n \times m$, now tall), compute polar factor, transpose back
2. **Use the "right" iteration**: $X_{k+1} = aX + b \cdot X(X^T X) + c \cdot X(X^T X)^2$

Option 1 is simpler and is what Muon does:

```python
transposed = False
if G.shape[0] < G.shape[1]:
    G = G.T
    transposed = True

Q = newton_schulz(G)

if transposed:
    Q = Q.T
```

---



---

# MODULE 7: Practical Usage & Hyperparameter Tuning

## 7.1 Key Hyperparameters

| Hyperparameter | Symbol | Typical Range | Default | Notes |
|---------------|--------|--------------|---------|-------|
| Learning Rate | $\eta$ | 0.005 – 0.05 | 0.02 | Much larger than Adam! |
| Momentum | $\beta$ | 0.85 – 0.99 | 0.95 | Nesterov momentum |
| Weight Decay | $\lambda$ | 0.0 – 0.1 | 0.01 | Decoupled (like AdamW) |
| NS Steps | $K$ | 3 – 10 | 5 | More = more accurate but slower |

## 7.2 Learning Rate

### Why is Muon's LR So Much Higher Than Adam's?

Adam's LR is typically 1e-4 to 1e-3. Muon's is 0.01 to 0.05. Why?

**Adam** divides by $\sqrt{v_t} + \epsilon$, which is roughly the RMS gradient magnitude. For typical neural network gradients, this is ~0.01–0.1, so Adam effectively amplifies the update by 10-100×.

**Muon** uses the polar factor, which has a fixed "magnitude" (all singular values = 1). The Frobenius norm of the update is $\sqrt{\min(m,n)}$. So the effective step size is directly controlled by $\eta$.

### Learning Rate Schedules

Muon works well with standard schedules:

```python
# Cosine decay with warmup
def get_lr(step, warmup_steps=200, max_steps=10000, 
           max_lr=0.02, min_lr=0.002):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# In training loop:
for step in range(max_steps):
    lr = get_lr(step)
    for group in optimizer_muon.param_groups:
        group['lr'] = lr
```

### Different LRs for Muon vs Adam

Typically:
```python
muon_lr = 0.02       # For weight matrices
adam_lr = 3e-4        # For embeddings, layernorms, biases
```

The ratio is roughly 50-100×.

## 7.3 Momentum

**$\beta = 0.95$** is a strong default. Some guidelines:

- **Larger batch → higher momentum** (more stable signal, can leverage more history)
- **Smaller batch → lower momentum** (noisy signal, less history is better)
- **Shorter training → lower momentum** (less time for momentum to build up)

```
Batch size 256:  β ≈ 0.90-0.93
Batch size 512:  β ≈ 0.93-0.95
Batch size 1024: β ≈ 0.95-0.97
```

## 7.4 Weight Decay

Weight decay in Muon is straightforward since it's decoupled:

$$
W \leftarrow (1 - \eta \cdot \lambda) \cdot W - \eta \cdot Q
$$

**Typical values**: 0.0 to 0.05. Start with 0.01.

Note: Since Muon's LR is larger than Adam's, the effective weight decay is also larger. If you're matching an Adam setup with WD=0.1 and LR=3e-4:

```
Adam effective WD per step: 0.1 × 3e-4 = 3e-5
Muon with WD=0.0015 and LR=0.02: 0.0015 × 0.02 = 3e-5  (matched)
```

## 7.5 Newton-Schulz Steps

**$K = 5$** is almost always sufficient. Here's why:

After normalization, singular values are in roughly [0.3, 1.7]. After 5 quintic NS iterations, they converge to within ~0.001 of 1.0.

| $K$ | Approx accuracy | Training quality | Speed overhead |
|---|-----------------|-----------------|----------------|
| 3 | ~0.05 | Good | Minimal |
| 5 | ~0.001 | Excellent | ~3% slowdown |
| 7 | ~1e-6 | Same as 5 | ~5% slowdown |
| 10 | ~machine $\epsilon$ | Same as 5 | ~8% slowdown |

**$K = 5$ is the sweet spot.**

## 7.6 Gradient Clipping

Muon's updates are naturally bounded (the polar factor has bounded norm), so gradient clipping is less critical than with Adam. However, it can still help:

```python
# Optional: clip before Muon processes gradients
torch.nn.utils.clip_grad_norm_(muon_params, max_norm=1.0)
```

Or alternatively, clip the momentum buffer before orthogonalization:

```python
# Inside the optimizer, after momentum update:
buf_norm = buf.norm()
if buf_norm > max_grad_norm:
    buf.mul_(max_grad_norm / buf_norm)
```

## 7.7 Batch Size Scaling

Muon scales well with batch size. Rough guidelines:

```
Base config:  batch_size=512, lr=0.02, momentum=0.95
2× batch:     batch_size=1024, lr=0.028, momentum=0.96  (√2× lr)
4× batch:     batch_size=2048, lr=0.04, momentum=0.97   (2× lr)
```

The standard linear scaling rule works reasonably well, though square-root scaling is often better for Muon.

## 7.8 Complete Training Recipe

Here's a battle-tested recipe for training a GPT-2 scale model:

```python
import math
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# ---- Configuration ----
config = {
    'model_dim': 768,
    'n_heads': 12,
    'n_layers': 12,
    'vocab_size': 50257,
    'max_seq_len': 1024,
    'batch_size': 64,
    'grad_accum_steps': 8,        # effective batch = 512
    'max_steps': 10000,
    'warmup_steps': 200,
    'muon_lr': 0.02,
    'adam_lr': 3e-4,
    'muon_momentum': 0.95,
    'weight_decay': 0.01,
    'adam_betas': (0.9, 0.95),
    'adam_eps': 1e-8,
    'max_grad_norm': 1.0,
    'ns_steps': 5,
}

# ---- Setup model & optimizers ----
model = build_model(config).cuda()

# Parameter grouping
muon_params = []
adam_params_decay = []
adam_params_nodecay = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.dim() >= 2 and 'embed' not in name and 'head' not in name:
        muon_params.append(p)
    elif p.dim() >= 2:
        adam_params_decay.append(p)
    else:
        adam_params_nodecay.append(p)

opt_muon = Muon(
    muon_params,
    lr=config['muon_lr'],
    momentum=config['muon_momentum'],
    weight_decay=config['weight_decay'],
    ns_steps=config['ns_steps'],
)

opt_adam = torch.optim.AdamW([
    {'params': adam_params_decay, 'weight_decay': config['weight_decay']},
    {'params': adam_params_nodecay, 'weight_decay': 0.0},
], lr=config['adam_lr'], betas=config['adam_betas'], eps=config['adam_eps'])

# ---- Learning rate schedule ----
def cosine_schedule(step, warmup, total, max_lr, min_lr_ratio=0.1):
    if step < warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return max_lr * (min_lr_ratio + (1 - min_lr_ratio) * 
                     0.5 * (1 + math.cos(math.pi * progress)))

# ---- Training loop ----
scaler = GradScaler()

for step in range(config['max_steps']):
    # Update learning rates
    muon_lr = cosine_schedule(step, config['warmup_steps'], 
                               config['max_steps'], config['muon_lr'])
    adam_lr = cosine_schedule(step, config['warmup_steps'], 
                              config['max_steps'], config['adam_lr'])
    
    for g in opt_muon.param_groups:
        g['lr'] = muon_lr
    for g in opt_adam.param_groups:
        g['lr'] = adam_lr
    
    # Gradient accumulation
    total_loss = 0
    for micro_step in range(config['grad_accum_steps']):
        x, y = get_batch(config['batch_size'], config['max_seq_len'])
        
        with autocast(dtype=torch.bfloat16):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, config['vocab_size']),
                y.view(-1)
            ) / config['grad_accum_steps']
        
        scaler.scale(loss).backward()
        total_loss += loss.item()
    
    # Gradient clipping
    scaler.unscale_(opt_muon)
    scaler.unscale_(opt_adam)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
    
    # Step both optimizers
    scaler.step(opt_muon)
    scaler.step(opt_adam)
    scaler.update()
    
    opt_muon.zero_grad(set_to_none=True)
    opt_adam.zero_grad(set_to_none=True)
    
    # Logging
    if step % 50 == 0:
        print(f"Step {step:5d} | Loss {total_loss:.4f} | "
              f"Muon LR {muon_lr:.6f} | Adam LR {adam_lr:.6f}")
```

## 7.9 Common Pitfalls & Debugging

### Pitfall 1: Using Muon for Embeddings
```python
# ❌ BAD: Embedding gradients are sparse, Muon doesn't handle this well
muon_params = list(model.parameters())

# ✅ GOOD: Separate embeddings
for name, p in model.named_parameters():
    if 'embed' in name or p.dim() < 2:
        adam_params.append(p)
    else:
        muon_params.append(p)
```

### Pitfall 2: Learning Rate Too Low
```python
# ❌ BAD: Using Adam-scale learning rate
Muon(params, lr=3e-4)   # Way too small!

# ✅ GOOD: Muon needs ~100× larger LR
Muon(params, lr=0.02)
```

### Pitfall 3: Not Using Nesterov
```python
# ❌ SUBOPTIMAL: Regular momentum
buf = momentum * buf + grad
update = orthogonalize(buf)

# ✅ BETTER: Nesterov momentum
buf = momentum * buf + grad
nesterov = momentum * buf + grad    # look-ahead
update = orthogonalize(nesterov)
```

### Pitfall 4: Wrong Parameter Shapes
```python
# Check your parameters
for name, p in model.named_parameters():
    print(f"{name:50s} shape={str(p.shape):20s} dim={p.dim()} "
          f"→ {'MUON' if p.dim() >= 2 and 'embed' not in name else 'Adam'}")
```

---



---

# MODULE 8: Muon vs Other Optimizers

## 8.1 Theoretical Comparison

| Property | SGD+Mom | Adam | LAMB | Shampoo | SOAP | **Muon** |
|----------|---------|------|------|---------|------|----------|
| **Per-param memory** | 1$\times$ | 2$\times$ | 2$\times$ | 1$\times$ + precond | 1$\times$ + precond | 1$\times$ |
| **Per-step cost** | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ | $\mathcal{O}(n + d^3)$ | $\mathcal{O}(n + d^3)$ | $\mathcal{O}(K \cdot m^2 n)$ |
| **Adaptive?** | No | Per-element | Per-element | Per-direction | Per-direction | No (spectral) |
| **Norm type** | L2 | $\infty$-like | L2 (layer) | Mahalanobis | Mahalanobis | **Spectral** |
| **Uses curvature?** | No | Diagonal | Diagonal | Kronecker | Kronecker | **No** (implied) |
| **Invariances** | None | Scale | Scale+layer | Affine | Affine | **Rotational** |

## 8.2 Memory Comparison

For a model with $N$ total parameters (all 2D matrices of avg shape $d \times d$):

| Optimizer | Optimizer States | Total Memory |
|-----------|-----------------|--------------|
| SGD + Momentum | $N$ (momentum buffer) | $2N$ |
| Adam/AdamW | $2N$ ($m + v$) | $3N$ |
| Shampoo | $N + 2 \cdot \text{num\_layers} \cdot d^2$ | ~$N + 2d^2$ per layer |
| **Muon** | **$N$** (momentum only) | **$2N$** |

**Muon uses 33% less optimizer memory than Adam!** (1 buffer vs 2)

## 8.3 Compute Comparison

For a linear layer with weight $W \in \mathbb{R}^{d \times d}$:

| Optimizer | Extra compute per step | Relative to forward pass |
|-----------|----------------------|-------------------------|
| SGD+Mom | $\mathcal{O}(d^2)$ | Negligible |
| Adam | $\mathcal{O}(d^2)$ | Negligible |
| Shampoo | $\mathcal{O}(d^3)$ per precond update | Significant |
| **Muon** | **$\mathcal{O}(K \cdot d^3)$** | Moderate |

The $K$ matrix multiplies of size $d \times d$ cost $\mathcal{O}(K \cdot d^3)$ FLOPs. For $d=768$, $K=5$: ~2.3 billion FLOPs.

The forward+backward pass for the same layer (batch $B$, seq $T$): $\mathcal{O}(B \cdot T \cdot d^2) \approx 64 \cdot 1024 \cdot 768^2 \approx 39$ billion FLOPs.

So Muon's overhead is roughly **6% of the forward/backward cost** for this layer.

## 8.4 Convergence Comparison

### GPT-2 124M on OpenWebText (representative results):

```
Steps to reach val loss 3.28:
  AdamW:           ~10,000 steps
  Muon:            ~5,500 steps     (1.8× fewer)
  Shampoo:         ~6,000 steps     (1.7× fewer)
  SOAP:            ~5,800 steps     (1.7× fewer)

Wall-clock time to reach val loss 3.28 (single A100):
  AdamW:           ~45 minutes
  Muon:            ~28 minutes      (1.6× faster)
  Shampoo:         ~50 minutes      (0.9× - overhead!)
  SOAP:            ~40 minutes      (1.1× faster)
```

**Key insight**: Muon achieves Shampoo-level convergence speed with much lower overhead.

## 8.5 When to Use Muon

### ✅ Use Muon When:
- Training **transformers** (language models, vision transformers)
- Model has mostly **2D weight matrices**
- You want **faster convergence** than Adam
- You want **lower memory** than Adam
- You're doing **large-scale pretraining**

### ❌ Don't Use Muon When:
- Model is mostly **1D parameters** (unlikely for modern architectures)
- Very small matrices (overhead not worth it)
- You need an optimizer for **sparse gradients** (use SparseAdam)
- Proven recipes exist with Adam and changing is risky (fine-tuning BERT, etc.)

## 8.6 Detailed Experiment: Muon vs Adam

```python
"""
Controlled comparison of Muon vs Adam on a small language model.
"""
import torch
import torch.nn as nn
import time
from collections import defaultdict

def run_comparison(model_fn, data_fn, steps=2000):
    results = {}
    
    for opt_name in ['Adam', 'Muon']:
        torch.manual_seed(42)
        model = model_fn().cuda()
        
        if opt_name == 'Adam':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=3e-4, 
                betas=(0.9, 0.95), weight_decay=0.1
            )
        else:
            muon_p, adam_p = [], []
            for n, p in model.named_parameters():
                if p.dim() >= 2 and 'embed' not in n:
                    muon_p.append(p)
                else:
                    adam_p.append(p)
            
            optimizer = CombinedOptimizer([
                Muon(muon_p, lr=0.02, momentum=0.95, weight_decay=0.01),
                torch.optim.AdamW(adam_p, lr=3e-4, weight_decay=0.1)
            ])
        
        losses = []
        times = []
        t0 = time.time()
        
        for step in range(steps):
            x, y = data_fn()
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            if step % 50 == 0:
                losses.append(loss.item())
                times.append(time.time() - t0)
        
        results[opt_name] = {'losses': losses, 'times': times}
        print(f"{opt_name}: final loss = {losses[-1]:.4f}, "
              f"time = {times[-1]:.1f}s")
    
    return results
```

---



---

# MODULE 9: Distributed / Multi-GPU Muon

## 9.1 The Challenge

In distributed training (DDP, FSDP), gradients are all-reduced across GPUs. Muon's Newton-Schulz iterations then run independently on each GPU, which is fine for DDP since each GPU has the full gradient after all-reduce.

However, with **FSDP** (Fully Sharded Data Parallelism), parameters and gradients are sharded. This creates challenges because:
1. Newton-Schulz needs the **full gradient matrix** to compute the polar decomposition
2. Sharded gradients only have a portion of the matrix

## 9.2 Muon with DDP

DDP is straightforward — just wrap the model:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group('nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

model = build_model().cuda(rank)
model = DDP(model, device_ids=[rank])

# Muon works exactly the same — gradients are all-reduced before step()
muon_params = [p for n, p in model.named_parameters() 
               if p.dim() >= 2 and 'embed' not in n]
optimizer = Muon(muon_params, lr=0.02)

for step in range(num_steps):
    loss = compute_loss(model, batch)
    loss.backward()          # DDP handles gradient all-reduce
    optimizer.step()         # Each GPU runs NS iterations independently
    optimizer.zero_grad()
```

## 9.3 Muon with FSDP (Advanced)

With FSDP, we need to handle gradient unsharding for Newton-Schulz:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class MuonFSDP(Optimizer):
    """Muon adapted for FSDP training."""
    
    def __init__(self, params, lr=0.02, momentum=0.95,
                 weight_decay=0.0, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, 
                        weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                g = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                state['step'] += 1
                buf = state['momentum_buffer']
                
                # Momentum
                buf.mul_(group['momentum']).add_(g)
                nesterov = group['momentum'] * buf + g
                
                # Reshape to 2D
                orig_shape = nesterov.shape
                if nesterov.dim() >= 2:
                    nesterov_2d = nesterov.view(nesterov.shape[0], -1)
                    
                    # All-gather the full gradient for NS iterations
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        full_nesterov = self._allgather_tensor(nesterov_2d)
                    else:
                        full_nesterov = nesterov_2d
                    
                    # Run Newton-Schulz on full matrix
                    transposed = False
                    if full_nesterov.shape[0] < full_nesterov.shape[1]:
                        full_nesterov = full_nesterov.T
                        transposed = True
                    
                    Q = newton_schulz_5(full_nesterov, group['ns_steps'])
                    
                    if transposed:
                        Q = Q.T
                    
                    # Extract local shard
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        Q = self._get_local_shard(Q)
                    
                    update = Q.view(orig_shape)
                else:
                    update = nesterov.sign()
                
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update, alpha=-group['lr'])
    
    def _allgather_tensor(self, tensor):
        """All-gather a sharded tensor across ranks."""
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)
    
    def _get_local_shard(self, tensor):
        """Get the local shard of a gathered tensor."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        shard_size = tensor.shape[0] // world_size
        return tensor[rank * shard_size : (rank + 1) * shard_size]
```

## 9.4 Efficient Communication Strategy

The naive approach above all-gathers the entire gradient, which is expensive. A more efficient strategy:

```python
class MuonDistributed(Optimizer):
    """
    Efficient distributed Muon that overlaps NS computation with communication.
    
    Key insight: We can run NS iterations on locally-available gradient data
    and only communicate the final update, rather than the full gradient.
    """
    
    @torch.no_grad()
    def step(self):
        # Phase 1: Compute momentum updates for all params (local)
        updates_to_orthogonalize = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # ... momentum computation ...
                
                updates_to_orthogonalize.append((p, nesterov, group))
        
        # Phase 2: Batch the Newton-Schulz computations
        # Group by shape for efficient batching
        shape_groups = defaultdict(list)
        for p, nesterov, group in updates_to_orthogonalize:
            shape = nesterov.shape
            shape_groups[shape].append((p, nesterov, group))
        
        # Phase 3: Run NS iterations with optional async all-reduce
        for shape, items in shape_groups.items():
            # Stack all updates of the same shape
            stacked = torch.stack([item[1] for item in items])
            
            # Batched Newton-Schulz
            Q_stacked = batched_newton_schulz(stacked)
            
            # Apply updates
            for i, (p, _, group) in enumerate(items):
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(Q_stacked[i], alpha=-group['lr'])


def batched_newton_schulz(G_batch, steps=5):
    """
    Run Newton-Schulz on a batch of matrices simultaneously.
    
    G_batch: shape (B, m, n)
    Returns: shape (B, m, n)
    """
    coeffs = [
        (3.4445, -4.7750, 2.0315),
        (11.3168, -20.3300, 9.7132),
        (8.4749, -13.9590, 6.1843),
    ]
    
    X = G_batch.bfloat16()
    
    # Normalize each matrix independently
    norms = X.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1)
    nrows = X.shape[1]
    X = X / (norms + 1e-7) * (nrows ** 0.5)
    
    for i in range(steps):
        a, b, c = coeffs[min(i, len(coeffs)-1)]
        A = X @ X.transpose(-2, -1)          # (B, m, m)
        B = A @ X                             # (B, m, n)
        X = a * X + b * B + c * (A @ B)      # (B, m, n)
    
    return X.to(G_batch.dtype)
```

## 9.5 Scaling Results

Typical scaling behavior of Muon across GPUs:

```
1 GPU (A100):    baseline throughput
2 GPUs:          ~1.9× throughput (95% scaling efficiency)
4 GPUs:          ~3.7× throughput (93% scaling efficiency)  
8 GPUs:          ~7.2× throughput (90% scaling efficiency)

Compare Adam:
8 GPUs:          ~7.5× throughput (94% scaling efficiency)
```

Muon's scaling is slightly worse than Adam due to the matrix multiply overhead, which doesn't scale with data parallelism. However, the faster convergence more than compensates.

---



---

# MODULE 10: Advanced Topics & Current Research

## 10.1 Theoretical Analysis: Why Does Muon Work So Well?

### 10.1.1 The Spectral Perspective

Weight matrices in neural networks act as **linear operators**. Training them with element-wise methods (Adam) ignores this structure. Muon respects it by:

1. **Equalizing singular values of the update**: All directions of the weight matrix receive equal "push"
2. **Preventing rank collapse**: Adam can cause some singular directions to dominate; Muon prevents this
3. **Scale-free updates**: The update norm is independent of the gradient magnitude

### 10.1.2 Connection to Natural Gradient

For a linear layer `y = Wx`, the Fisher information matrix for W has Kronecker structure:

$$
F_W = \mathbb{E}[xx^T] \otimes \mathbb{E}[(\partial L/\partial y)(\partial L/\partial y)^T]
$$

The natural gradient is:
$$
F_W^{-1} \text{vec}(G) = \mathbb{E}[xx^T]^{-1} G \mathbb{E}[(\partial L/\partial y)(\partial L/\partial y)^T]^{-1}
$$

This is exactly what Shampoo approximates. In the "whitened" case (when inputs and output gradients are white noise), this simplifies to:

$$
\text{natural gradient direction} = G (G^T G)^{-1/2} = \text{polar factor of } G
$$

**So Muon can be viewed as the natural gradient under the assumption of white input/output statistics.**

### 10.1.3 Implicit Regularization

Steepest descent under the spectral norm has an implicit regularization effect:

- It tends to keep weight matrices **well-conditioned** (condition number close to 1)
- It prevents extreme singular values from developing
- This acts as a form of spectral regularization

## 10.2 Variants and Extensions

### 10.2.1 Muon with Adaptive Learning Rate

We can add a scalar adaptive learning rate while keeping the orthogonal update direction:

```python
class MuonAdaptive(Optimizer):
    """Muon with per-layer adaptive learning rate."""
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                # ... compute nesterov, Q as before ...
                
                # Adaptive scaling: use gradient alignment
                state = self.state[p]
                if 'ema_scale' not in state:
                    state['ema_scale'] = 1.0
                
                # How well does Q align with the gradient?
                alignment = (p.grad * Q).sum() / (p.grad.norm() * Q.norm() + 1e-8)
                state['ema_scale'] = 0.99 * state['ema_scale'] + 0.01 * alignment.item()
                
                effective_lr = group['lr'] * max(0.1, state['ema_scale'])
                
                p.mul_(1 - effective_lr * group['weight_decay'])
                p.add_(Q, alpha=-effective_lr)
```

### 10.2.2 Muon with Gradient Filtering (Cautious Muon / C-Muon)

Inspired by "Cautious Optimizers" — only update coordinates where the gradient and momentum agree:

```python
def cautious_muon_step(p, g, buf, momentum, lr, wd, ns_steps):
    """Cautious Muon: mask out coordinates where gradient and update disagree."""
    # Standard Muon momentum
    buf.mul_(momentum).add_(g)
    nesterov = momentum * buf + g
    
    # Get orthogonal update
    Q = newton_schulz_5(nesterov.view(nesterov.shape[0], -1))
    Q = Q.view(nesterov.shape)
    
    # Cautious mask: only update where g and Q agree in sign
    mask = (g * Q > 0).float()
    
    # Rescale to preserve expected update magnitude
    mask = mask * (mask.numel() / (mask.sum() + 1))
    
    Q_cautious = Q * mask
    
    p.mul_(1 - lr * wd)
    p.add_(Q_cautious, alpha=-lr)
```

### 10.2.3 Muon for Convolutions

For convolutional layers with weight shape (C_out, C_in, K, K):

```python
def muon_conv_update(weight_grad, ns_steps=5):
    """
    Handle conv weights by reshaping to 2D.
    
    (C_out, C_in, K, K) → (C_out, C_in * K * K)
    """
    shape = weight_grad.shape
    G = weight_grad.reshape(shape[0], -1)  # (C_out, C_in*K*K)
    
    transposed = False
    if G.shape[0] < G.shape[1]:
        G = G.T
        transposed = True
    
    Q = newton_schulz_5(G, ns_steps)
    
    if transposed:
        Q = Q.T
    
    return Q.reshape(shape)
```

### 10.2.4 Muon with Warmup Scheduling

A sophisticated warmup that gradually transitions from Adam to Muon:

```python
def warmup_muon(step, warmup_steps, p, g, muon_state, adam_state):
    """
    Warm up by blending Adam and Muon updates.
    Starts as Adam, transitions to Muon.
    """
    alpha = min(1.0, step / warmup_steps)  # 0 → 1
    
    # Adam update
    adam_update = compute_adam_update(g, adam_state)
    
    # Muon update
    muon_update = compute_muon_update(g, muon_state)
    
    # Blend
    update = (1 - alpha) * adam_update + alpha * muon_update
    
    return update
```

## 10.3 Relationship to Other Matrix-Aware Methods

### 10.3.1 Connection to Shampoo

**Shampoo** maintains left/right preconditioners:
$$
\begin{aligned}
L &= \sum G_t G_t^T, \quad R = \sum G_t^T G_t \\
\text{Update} &= L^{-1/4} G R^{-1/4}
\end{aligned}
$$

**Muon** effectively does the same but in the limit:
$$
\begin{aligned}
&\text{If } L \to GG^T \text{ and } R \to G^TG \text{ (single step),} \\
&\text{then } L^{-1/4} G R^{-1/4} = (GG^T)^{-1/4} G (G^TG)^{-1/4} \\
&= U \Sigma^{-1/2} U^T \cdot U \Sigma V^T \cdot V \Sigma^{-1/2} V^T \\
&= U V^T \\
&= \text{polar factor of } G
\end{aligned}
$$

So **Muon is Shampoo in the single-step / infinite-batch limit**.

### 10.3.2 Connection to SOAP

SOAP runs Adam in the eigenbasis of the Shampoo preconditioners. Muon can be seen as a simplification where instead of maintaining and updating eigenbases, we just project directly to the nearest orthogonal matrix.

### 10.3.3 Connection to Spectral Training

Some papers have proposed constraining weight matrices to be orthogonal during training. Muon achieves a similar effect implicitly — while it doesn't constrain W to be orthogonal, the **updates** are orthogonal, which tends to keep W well-conditioned.

## 10.4 The Complete Keller Jordan Implementation

Here is the reference implementation (simplified from the `modded-nanogpt` codebase):

```python
"""
Reference Muon implementation based on Keller Jordan's modded-nanogpt.
"""
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / polar factor of G.
    
    We opt to use a quintic iteration whose coefficients are selected to 
    maximize the slope at zero. For the purpose of minimizing steps, 
    it turns out to be empirically effective to simply maximize the slope 
    at zero rather than trying to minimize error over any specific range.
    
    The recurrence is:
        X_{k+1} = a * X_k + b * (X_k @ X_k.T) @ X_k + c * ((X_k @ X_k.T) @ X_k @ X_k.T) @ X_k
    which is a quintic in the singular values.
    
    Coefficients used:
        First iteration:  (3.4445, -4.7750,  2.0315)   - chosen for broad convergence
        Second iteration: (11.3168, -20.3300, 9.7132)   - tighter convergence
        Later iterations: (8.4749, -13.9590, 6.1843)    - polishing
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    
    # Ensure tall matrix
    if G.shape[0] > G.shape[1]:
        X = X.T
    
    # Normalize: singular values should be near 1
    X /= (X.norm() + 1e-7)
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.shape[0] > G.shape[1]:
        X = X.T
    
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    
    Muon internally runs standard SGD-momentum, and then performs an 
    idealized version of Shampoo (Gupta et al. 2018), via a Newton-Schulz 
    iteration, to orthogonalize each update before applying it to the 
    model parameters.
    
    In other words, it is doing steepest descent under the spectral norm, 
    instead of the Frobenius norm (vanilla SGD) or the element-wise 
    L-infinity norm (sign SGD / Adam).
    
    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. 0.02 is a good default for most models.
        momentum: The momentum coefficient. 0.95 is the default.
        nesterov: Whether to use Nesterov-style momentum. Default True.
        ns_steps: The number of Newton-Schulz steps. Default 5.
        adamw_params: The parameters to be optimized by AdamW (1D params).
        adamw_lr: Learning rate for AdamW params.
        adamw_betas: Betas for AdamW.
        adamw_eps: Epsilon for AdamW.
        adamw_wd: Weight decay for AdamW.
    """
    
    def __init__(self, muon_params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, adamw_params=None, adamw_lr=3e-4,
                 adamw_betas=(0.95, 0.95), adamw_eps=1e-8, adamw_wd=0.0):
        
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            adamw_lr=adamw_lr, adamw_betas=adamw_betas,
            adamw_eps=adamw_eps, adamw_wd=adamw_wd,
        )
        
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        
        super().__init__(params + adamw_params, defaults)
        
        # Tag which params use Muon vs AdamW
        self.muon_params = set(id(p) for p in params)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            # -------- Process Muon params --------
            muon_updates = []
            for p in group['params']:
                if id(p) not in self.muon_params or p.grad is None:
                    continue
                
                g = p.grad
                state = self.state[p]
                
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(g)
                
                if group['nesterov']:
                    g = g.add(buf, alpha=group['momentum'])
                else:
                    g = buf
                
                muon_updates.append((p, g))
            
            # Orthogonalize and apply Muon updates
            for p, g in muon_updates:
                g_2d = g.view(g.shape[0], -1) if g.dim() > 1 else g.unsqueeze(0)
                update = zeropower_via_newtonschulz5(g_2d, steps=group['ns_steps'])
                update = update.view(g.shape)
                
                # Scale by sqrt(max(m,n)/min(m,n)) to normalize
                p.add_(update, alpha=-group['lr'])
            
            # -------- Process AdamW params --------
            for p in group['params']:
                if id(p) in self.muon_params or p.grad is None:
                    continue
                
                g = p.grad
                state = self.state[p]
                
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(g)
                    state['exp_avg_sq'] = torch.zeros_like(g)
                
                state['step'] += 1
                
                beta1, beta2 = group['adamw_betas']
                
                state['exp_avg'].mul_(beta1).add_(g, alpha=1-beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(g, g, value=1-beta2)
                
                bias1 = 1 - beta1 ** state['step']
                bias2 = 1 - beta2 ** state['step']
                
                step_size = group['adamw_lr'] / bias1
                denom = (state['exp_avg_sq'] / bias2).sqrt().add_(group['adamw_eps'])
                
                p.mul_(1 - group['adamw_lr'] * group['adamw_wd'])
                p.addcdiv_(state['exp_avg'], denom, value=-step_size)
```

## 10.5 Understanding the "Zeropower" Name

In the Muon codebase, the polar factor computation is called `zeropower_via_newtonschulz5`. Why "zeropower"?

The **matrix sign function** of a positive definite matrix A is:
$$
\text{sign}(A) = A \cdot |A|^{-1} = A \cdot (A^T A)^{-1/2} = UV^T \quad (\text{for } A = U\Sigma V^T)
$$

This is also the **zeroth power** in the sense of:
$$
\begin{aligned}
(A^TA)^0 &= I \quad &(\text{true zeroth power}) \\
A (A^TA)^{-1/2} &= U \Sigma^0 V^T = UV^T \quad &(\text{singular values } \to \sigma^0 = 1)
\end{aligned}
$$

So "zeropower" means "raise the singular values to the power zero" (i.e., make them all 1).

## 10.6 Open Research Questions

### 10.6.1 Optimal NS Coefficients
The current coefficients are heuristically optimized. Is there a principled way to choose them based on the expected singular value distribution?

### 10.6.2 Extension to Attention
Should we apply Muon to the QKV projections differently than to the MLP layers? The gradient structure differs significantly.

### 10.6.3 Scaling Laws
How does Muon's advantage change with model scale? Early results suggest it's even more beneficial for larger models, but systematic scaling law studies are ongoing.

### 10.6.4 Fine-tuning
Muon was designed for pretraining. How well does it work for fine-tuning, where the optimization landscape is different?

### 10.6.5 Non-Transformer Architectures
How well does Muon work for CNNs, RNNs, state-space models, or other architectures?

### 10.6.6 Combination with Other Techniques
- Muon + gradient checkpointing
- Muon + mixed precision training
- Muon + curriculum learning
- Muon + data augmentation strategies

## 10.7 Reading List & References

| Resource | Description |
|----------|-------------|
| **[Keller Jordan's modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)** | Original implementation and speedrun |
| **[Bernstein & Newhouse (2024)](https://arxiv.org/abs/2409.20325)** | "Old Optimizer, New Norm" - theoretical foundations |
| **Gupta et al. (2018)** — Shampoo | The precursor second-order optimizer |
| **Vyas et al. (2024)** — SOAP | Shampoo + Adam fusion (Meta) |
| **Nicholas Higham — "Functions of Matrices"** | Definitive reference for matrix functions |
| **Golub & Van Loan — "Matrix Computations"** | SVD, polar decomposition theory |

---



---

# APPENDIX A: Quick Reference Card

```
╔══════════════════════════════════════════════════╗
║            MUON OPTIMIZER CHEAT SHEET            ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  ALGORITHM:                                      ║
║  1. g = gradient                                 ║
║  2. m = β·m + g           (momentum)             ║
║  3. n = β·m + g           (nesterov)             ║
║  4. Q = polar_factor(n)   (newton-schulz)        ║
║  5. W = (1-η·λ)·W - η·Q  (update)               ║
║                                                  ║
║  DEFAULT HYPERPARAMETERS:                        ║
║  • lr (η):        0.02                           ║
║  • momentum (β):  0.95                           ║
║  • weight_decay:  0.01                           ║
║  • ns_steps:      5                              ║
║                                                  ║
║  PARAMETER ROUTING:                              ║
║  • 2D weights (not embed/head) → Muon            ║
║  • Embeddings, LM head        → AdamW            ║
║  • 1D params (LN, biases)     → AdamW            ║
║                                                  ║
║  MEMORY: 1 buffer per param (vs 2 for Adam)      ║
║  COMPUTE: ~5% overhead vs vanilla SGD             ║
║  SPEEDUP: ~1.5-2× fewer steps than Adam          ║
║                                                  ║
╚══════════════════════════════════════════════════╝
```



---

# APPENDIX B: Exercises

## Exercise 1: Verify the Polar Factor
```python
"""
Compute the polar factor of a random matrix using:
1. SVD (exact)
2. Newton-Schulz (approximate)
Compare the results.
"""
import torch

G = torch.randn(64, 32)

# Method 1: SVD
U, S, Vt = torch.linalg.svd(G, full_matrices=False)
Q_exact = U @ Vt

# Method 2: Newton-Schulz
Q_approx = newton_schulz_5(G, steps=5)

# Compare
print(f"Frobenius error: {(Q_exact - Q_approx).norm():.6f}")
print(f"Max error: {(Q_exact - Q_approx).abs().max():.6f}")

# Verify Q is approximately orthogonal
print(f"Q^T Q \approx I error: {(Q_approx.T @ Q_approx - torch.eye(32)).norm():.6f}")
```

## Exercise 2: Visualize Singular Value Convergence
```python
"""
Track how singular values evolve through NS iterations.
"""
G = torch.randn(64, 32)
U, S, Vt = torch.linalg.svd(G, full_matrices=False)
print(f"Initial singular values: {S[:5].tolist()}")

# Track through iterations
X = G / G.norm() * (64 ** 0.5)
for i in range(10):
    _, S_current, _ = torch.linalg.svd(X, full_matrices=False)
    print(f"Step {i}: sigma_min={S_current[-1]:.4f}, sigma_max={S_current[0]:.4f}")
    
    a, b, c = (3.4445, -4.7750, 2.0315)
    A = X @ X.T
    B = A @ X
    X = a * X + b * B + c * (A @ B)
```

## Exercise 3: Train a Small Model
```python
"""
Train a 2-layer MLP on MNIST with Muon vs Adam.
Compare convergence curves.
"""
# Your code here - implement and compare!
```

## Exercise 4: Implement Septic Newton-Schulz
```python
"""
Implement a 7th-order Newton-Schulz iteration.
$f(\sigma) = a\sigma + b\sigma^3 + c\sigma^5 + d\sigma^7$

Constraints:
$f(1) = 1, f'(1) = 0, f''(1) = 0, f'''(1) = 0$

Solve for a, b, c, d and implement.
"""
# Your code here!
```

## Exercise 5: Ablation Study
```python
"""
Run an ablation study varying:
1. Number of NS steps (1, 3, 5, 7, 10)
2. Momentum (0.8, 0.9, 0.95, 0.99)
3. With/without Nesterov
4. Learning rate (0.005, 0.01, 0.02, 0.05, 0.1)

Plot the results.
"""
# Your code here!
```

---



---

# APPENDIX C: FAQ

**Q: Can I use Muon for the entire model?**
A: No. Use it only for 2D+ weight matrices. Embeddings, biases, and LayerNorm parameters should use Adam.

**Q: Why not use exact SVD instead of Newton-Schulz?**
A: SVD is ~10× slower on GPU because it doesn't parallelize as well as matrix multiplies.

**Q: Does Muon work with fp16 / bf16?**
A: Yes! The NS iterations internally use bf16 for speed. The momentum buffer can also be bf16.

**Q: How does Muon interact with gradient accumulation?**
A: Normally — accumulate gradients as usual, then call `optimizer.step()`. The NS iterations run on the accumulated gradient's momentum.

**Q: Is Muon better than Adam for fine-tuning?**
A: This is still being studied. For pretraining, Muon is clearly better. For fine-tuning, Adam with carefully tuned hyperparameters is still competitive.

**Q: Does Muon have convergence guarantees?**
A: Bernstein & Newhouse (2024) provide theoretical analysis showing that steepest descent under the spectral norm converges for smooth objectives. Formal convergence rates are an active area of research.

**Q: What if my weight matrix is very rectangular (e.g., 50257 × 768)?**
A: This is why embeddings are excluded — the polar factor of a very tall matrix is less meaningful and the NS iterations are expensive. For moderately rectangular matrices (up to ~4:1 ratio), Muon works fine.

---

*End of Course*

*This course covers Muon as understood through early-mid 2025. The field is rapidly evolving — check the latest research for updates.*
