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

