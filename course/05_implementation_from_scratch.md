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

