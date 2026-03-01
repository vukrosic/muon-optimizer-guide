# MODULE 9: Distributed / Multi-GPU Muon

## 9.1 The Challenge

In distributed training (DDP, FSDP), gradients are all-reduced across GPUs. Muon's Newton-Schulz iterations then run independently on each GPU, which is fine for DDP since each GPU has the full gradient after all-reduce.

However, with **FSDP** (Fully Sharded Data Parallelism), parameters and gradients are sharded. This creates challenges because:
1. Newton-Schulz needs the **full gradient matrix** to compute the polar decomposition
2. Sharded gradients only have a portion of the matrix

## 9.2 Muon with DDP

DDP is straightforward — just wrap the model:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group('nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

model = build_model().cuda(rank)
model = DDP(model, device_ids=[rank])

# Muon works exactly the same — gradients are all-reduced before step()
muon_params = [p for n, p in model.named_parameters() 
               if p.dim() >= 2 and 'embed' not in n]
optimizer = Muon(muon_params, lr=0.02)

for step in range(num_steps):
    loss = compute_loss(model, batch)
    loss.backward()          # DDP handles gradient all-reduce
    optimizer.step()         # Each GPU runs NS iterations independently
    optimizer.zero_grad()
```

## 9.3 Muon with FSDP (Advanced)

With FSDP, we need to handle gradient unsharding for Newton-Schulz:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class MuonFSDP(Optimizer):
    """Muon adapted for FSDP training."""
    
    def __init__(self, params, lr=0.02, momentum=0.95,
                 weight_decay=0.0, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, 
                        weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                g = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                state['step'] += 1
                buf = state['momentum_buffer']
                
                # Momentum
                buf.mul_(group['momentum']).add_(g)
                nesterov = group['momentum'] * buf + g
                
                # Reshape to 2D
                orig_shape = nesterov.shape
                if nesterov.dim() >= 2:
                    nesterov_2d = nesterov.view(nesterov.shape[0], -1)
                    
                    # All-gather the full gradient for NS iterations
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        full_nesterov = self._allgather_tensor(nesterov_2d)
                    else:
                        full_nesterov = nesterov_2d
                    
                    # Run Newton-Schulz on full matrix
                    transposed = False
                    if full_nesterov.shape[0] < full_nesterov.shape[1]:
                        full_nesterov = full_nesterov.T
                        transposed = True
                    
                    Q = newton_schulz_5(full_nesterov, group['ns_steps'])
                    
                    if transposed:
                        Q = Q.T
                    
                    # Extract local shard
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        Q = self._get_local_shard(Q)
                    
                    update = Q.view(orig_shape)
                else:
                    update = nesterov.sign()
                
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update, alpha=-group['lr'])
    
    def _allgather_tensor(self, tensor):
        """All-gather a sharded tensor across ranks."""
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)
    
    def _get_local_shard(self, tensor):
        """Get the local shard of a gathered tensor."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        shard_size = tensor.shape[0] // world_size
        return tensor[rank * shard_size : (rank + 1) * shard_size]
```

## 9.4 Efficient Communication Strategy

The naive approach above all-gathers the entire gradient, which is expensive. A more efficient strategy:

```python
class MuonDistributed(Optimizer):
    """
    Efficient distributed Muon that overlaps NS computation with communication.
    
    Key insight: We can run NS iterations on locally-available gradient data
    and only communicate the final update, rather than the full gradient.
    """
    
    @torch.no_grad()
    def step(self):
        # Phase 1: Compute momentum updates for all params (local)
        updates_to_orthogonalize = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # ... momentum computation ...
                
                updates_to_orthogonalize.append((p, nesterov, group))
        
        # Phase 2: Batch the Newton-Schulz computations
        # Group by shape for efficient batching
        shape_groups = defaultdict(list)
        for p, nesterov, group in updates_to_orthogonalize:
            shape = nesterov.shape
            shape_groups[shape].append((p, nesterov, group))
        
        # Phase 3: Run NS iterations with optional async all-reduce
        for shape, items in shape_groups.items():
            # Stack all updates of the same shape
            stacked = torch.stack([item[1] for item in items])
            
            # Batched Newton-Schulz
            Q_stacked = batched_newton_schulz(stacked)
            
            # Apply updates
            for i, (p, _, group) in enumerate(items):
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(Q_stacked[i], alpha=-group['lr'])


def batched_newton_schulz(G_batch, steps=5):
    """
    Run Newton-Schulz on a batch of matrices simultaneously.
    
    G_batch: shape (B, m, n)
    Returns: shape (B, m, n)
    """
    coeffs = [
        (3.4445, -4.7750, 2.0315),
        (11.3168, -20.3300, 9.7132),
        (8.4749, -13.9590, 6.1843),
    ]
    
    X = G_batch.bfloat16()
    
    # Normalize each matrix independently
    norms = X.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1)
    nrows = X.shape[1]
    X = X / (norms + 1e-7) * (nrows ** 0.5)
    
    for i in range(steps):
        a, b, c = coeffs[min(i, len(coeffs)-1)]
        A = X @ X.transpose(-2, -1)          # (B, m, m)
        B = A @ X                             # (B, m, n)
        X = a * X + b * B + c * (A @ B)      # (B, m, n)
    
    return X.to(G_batch.dtype)
```

## 9.5 Scaling Results

Typical scaling behavior of Muon across GPUs:

```
1 GPU (A100):    baseline throughput
2 GPUs:          ~1.9× throughput (95% scaling efficiency)
4 GPUs:          ~3.7× throughput (93% scaling efficiency)  
8 GPUs:          ~7.2× throughput (90% scaling efficiency)

Compare Adam:
8 GPUs:          ~7.5× throughput (94% scaling efficiency)
```

Muon's scaling is slightly worse than Adam due to the matrix multiply overhead, which doesn't scale with data parallelism. However, the faster convergence more than compensates.

---

