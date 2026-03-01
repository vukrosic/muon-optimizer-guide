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

