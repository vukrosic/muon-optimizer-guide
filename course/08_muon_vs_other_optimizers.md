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

