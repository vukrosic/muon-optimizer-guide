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

