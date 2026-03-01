# APPENDIX C: FAQ

**Q: Can I use Muon for the entire model?**
A: No. Use it only for 2D+ weight matrices. Embeddings, biases, and LayerNorm parameters should use Adam.

**Q: Why not use exact SVD instead of Newton-Schulz?**
A: SVD is ~10× slower on GPU because it doesn't parallelize as well as matrix multiplies.

**Q: Does Muon work with fp16 / bf16?**
A: Yes! The NS iterations internally use bf16 for speed. The momentum buffer can also be bf16.

**Q: How does Muon interact with gradient accumulation?**
A: Normally — accumulate gradients as usual, then call `optimizer.step()`. The NS iterations run on the accumulated gradient's momentum.

**Q: Is Muon better than Adam for fine-tuning?**
A: This is still being studied. For pretraining, Muon is clearly better. For fine-tuning, Adam with carefully tuned hyperparameters is still competitive.

**Q: Does Muon have convergence guarantees?**
A: Bernstein & Newhouse (2024) provide theoretical analysis showing that steepest descent under the spectral norm converges for smooth objectives. Formal convergence rates are an active area of research.

**Q: What if my weight matrix is very rectangular (e.g., 50257 × 768)?**
A: This is why embeddings are excluded — the polar factor of a very tall matrix is less meaningful and the NS iterations are expensive. For moderately rectangular matrices (up to ~4:1 ratio), Muon works fine.

---

*End of Course*

*This course covers Muon as understood through early-mid 2025. The field is rapidly evolving — check the latest research for updates.*