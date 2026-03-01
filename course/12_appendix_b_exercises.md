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

