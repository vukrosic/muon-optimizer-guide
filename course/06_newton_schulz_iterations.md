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

