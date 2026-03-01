# APPENDIX A: Quick Reference Card

```
╔══════════════════════════════════════════════════╗
║            MUON OPTIMIZER CHEAT SHEET            ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  ALGORITHM:                                      ║
║  1. g = gradient                                 ║
║  2. m = β·m + g           (momentum)             ║
║  3. n = β·m + g           (nesterov)             ║
║  4. Q = polar_factor(n)   (newton-schulz)        ║
║  5. W = (1-η·λ)·W - η·Q  (update)               ║
║                                                  ║
║  DEFAULT HYPERPARAMETERS:                        ║
║  • lr (η):        0.02                           ║
║  • momentum (β):  0.95                           ║
║  • weight_decay:  0.01                           ║
║  • ns_steps:      5                              ║
║                                                  ║
║  PARAMETER ROUTING:                              ║
║  • 2D weights (not embed/head) → Muon            ║
║  • Embeddings, LM head        → AdamW            ║
║  • 1D params (LN, biases)     → AdamW            ║
║                                                  ║
║  MEMORY: 1 buffer per param (vs 2 for Adam)      ║
║  COMPUTE: ~5% overhead vs vanilla SGD             ║
║  SPEEDUP: ~1.5-2× fewer steps than Adam          ║
║                                                  ║
╚══════════════════════════════════════════════════╝
```

