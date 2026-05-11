# Week 10 Glossary

**Scaled dot-product attention**: Core attention function: softmax(QK^T / sqrt(d_k)) V; the 1/sqrt(d_k) prevents softmax saturation.

**Multi-head attention (MHA)**: Running h attention functions in parallel on different projections of Q/K/V, then concatenating.

**d_k**: Dimension of queries and keys in one attention head; equals d_model / num_heads in the paper.

**Self-attention**: Attention where Q, K, V all come from the same sequence; every position attends to every other.

**Cross-attention**: Attention where Q comes from the decoder and K/V come from the encoder; connects decoder to encoder context.

**Causal mask**: Upper-triangular boolean mask that prevents position i from attending to j > i; enforces autoregressive generation.

**Encoder**: Stack of N identical layers each with self-attention + FFN + residual + LayerNorm; produces a contextualized representation.

**Decoder**: Stack of N identical layers each with masked self-attention + cross-attention + FFN; generates output autoregressively.

**Feed-forward sublayer (FFN)**: Position-wise MLP with a 4x expansion: d_model → 4*d_model → d_model; applies per-position nonlinearity.

**Residual connection**: Adding the sublayer input to its output (x + Sublayer(x)); critical for gradient flow through deep networks.

**Layer normalization (LayerNorm)**: Normalizes hidden states across the feature dimension; stabilizes training.

**Post-LN**: Original paper's norm placement: LayerNorm(x + Sublayer(x)). Less stable than Pre-LN for deep models.

**Pre-LN**: Modern convention: x + Sublayer(LayerNorm(x)). More stable gradient flow; used in GPT-2, LLaMA.

**Sinusoidal positional encoding**: Fixed position embeddings using sin/cos at different frequencies; allows length generalization.

**Warmup schedule**: LR that increases linearly for `warmup_steps` then decays as 1/sqrt(step); prevents large early gradient updates.

**Label smoothing**: Softening one-hot targets toward a uniform distribution (epsilon=0.1); prevents overconfidence.
