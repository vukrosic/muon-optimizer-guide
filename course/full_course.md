# STOP Using AdamW, EVERYONE Uses Muon Now

Everyone (OpenAI, Meta, DeepSeek, Moonshot AI,...) replaced AdamW with Muon optimizer to train their LLMs.

DeepSeek researcher said that Muon optimizer was one of 2 biggest inventions in 2025.

If you train ANY neural network - you MUST try it!

It should make your training 30% to 2x faster!

# The main idea

Let's start with the basic formula for weight updates:

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_{\theta} \mathcal{L}
$$

- $\theta_t$: Model parameters (all weights) at step $t$.
- $\theta_{t+1}$: Updated parameters after one training step.
- $\eta$: Learning rate (update step size).
- $\mathcal{L}$: Loss function.
- $\nabla_{\theta} \mathcal{L}$: Gradient of the loss with respect to parameters $\theta$.
- $t$: Optimization step index.

If we take a look at the gradients $\nabla_{\theta} \mathcal{L}$, this matrix is subtracted from weights - this is how neural network learns, however, this matrix might be ill-conditioned - some directions might have a strong "pull", while other directions might have a weak "pull".

![Non-Orthogonal Effects](images/01_non_orthogonal_effect.png)

A "direction" or a "pull" is a coordinated change across multiple weights. It controls an input→output pathway through a layer.

Example:
Let's say we are training a neural network - when it sees a lot of green color it should predict "grass", and when it sees a lot of circles, it should predict "wheels".

Due to different types of encoding images, it might turn out that green image gradients have very small singular values (0.01), while circle image gradients have larger singular values (100).

This means that neural network will learn the [green -> grass] pathway (prediction) A LOT SLOWER than the [circle -> wheels] pathway.

It's not that there is less data for green color or circles are more important, it's just about the way data is represented with numbers. [is this correct, does it make sense]

Muon optimizer solves this:

![Orthogonal Effects](images/01_orthogonal_effect.png)


![SVD intuition for Muon: raw gradient has uneven singular values, while Muon uses UV^T with equalized singular values.](images/03a_svd_for_muon_intro.png)

Muon optimizer will make all singular values equal to 1, so it will update / learn [green -> grass] pathway at the same speed as [circle -> wheels] pathway.



### Important distinction

Muon optimizer makes makes **weight update matrices** orthogonal, not the weight matrices.


[i need evidence, support such as paper, needs evidence]
[i need some evidence about some directions becoming near 0 if the weight update matrix is not orthogonalized, and evidence that it hurts leanring?, find papers and add some evidence, how do i add it here and how do i cite or credit it]


![SVD intuition for Muon: raw gradient has uneven singular values, while Muon uses UV^T with equalized singular values.](images/03a_svd_for_muon_intro.png)
[put this picture before, on top, so students have feeling]




----
down and below i need you to purge, just check my optimizers folder, write code for muon and explain it, keep it all short, also keep the tips for implementing muon like learning rate big etc

# MODULE 3.5: Practical Muon — Code & Implementation

## 1. Minimal Implementation (The "Polar Express")
Based on the current implementation in our `optimizers/muon.py`, Muon uses a **"Polar Express"** approach — a compiled Newton-Schulz iteration that's extremely fast on GPUs.

```python
import torch

@torch.compile()
def zeropower_polar_express(G: torch.Tensor, steps: int = 5):
    """Newton-Schulz iteration for matrix orthogonalization"""
    X = G.bfloat16()
    
    # Square/Tall matrices work best
    if X.size(-2) > X.size(-1): 
        X = X.mT 
    
    # Normalize singular values to start near 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    
    # Quintic Newton-Schulz (5th order)
    # Converges singular values to 1 in ~5 steps
    for a, b, c in coeffs_list[:steps]:
        A = X @ X.mT 
        X = a * X + (b * A + c * A @ A) @ X
    
    return X.mT if G.size(-2) > G.size(-1) else X

class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-Schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                
                # 1. Update Nesterov Momentum
                state = self.state[p]
                if "buf" not in state: state["buf"] = torch.zeros_like(p.grad)
                buf = state["buf"]
                buf.lerp_(p.grad, 1 - group["momentum"])
                
                # 2. Orthogonalize the Update
                g = p.grad.lerp_(buf, group["momentum"])
                update = zeropower_polar_express(g.view(g.size(0), -1))
                
                # 3. Apply Multi-Step Update
                # Note: Scaling by sqrt(rows/cols) maintains stability
                scale = (p.size(-2) / p.size(-1))**0.5
                p.add_(update.view_as(p), alpha=-group["lr"] * scale)
```

## 2. Key Implementation Tips

### 🚀 Use a BIG Learning Rate
Unlike AdamW which typically uses `3e-4` or `6e-4`, Muon thrives with much larger learning rates.
*   **Default:** Start with `0.02`.
*   **Range:** `0.01` to `0.05` is common.
*   **Why?** Muon updates are orthogonal (all singular values = 1), so the gradient doesn't "explode" or "vanish" in magnitude. You have total control over the step size.

### 🧩 Parameter Routing
Muon is for **2D weight matrices only**. Do NOT apply it to:
*   **Embeddings:** Use AdamW.
*   **Biases/LayerNorms:** Use AdamW.
*   **Final Head:** Usually AdamW (due to sparse gradients).

### ⚡ Compile It!
Newton-Schulz involves several matrix multiplications ($X@X^T$). Using `@torch.compile()` allows PyTorch to fuse these operations, making Muon's overhead nearly zero on modern GPUs.

### 📏 Scaling the LR
The update is scaled by `sqrt(max(rows, cols) / min(rows, cols))`. This compensates for the "rectangularity" of the weight matrix and is a secret sauce for stability in deep models.

---

*This module covers the core Muon logic as used in production speedruns (like modded-nanoGPT).*
