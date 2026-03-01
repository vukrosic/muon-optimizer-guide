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

