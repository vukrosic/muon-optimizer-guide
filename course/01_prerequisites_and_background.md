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

