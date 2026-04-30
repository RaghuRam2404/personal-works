# Week 12 Glossary

**RMSNorm**: Normalization using root-mean-square of activations only; no mean subtraction or beta parameter; faster than LayerNorm.

**SwiGLU**: Gated FFN variant: FFN(x) = down(SiLU(gate(x)) * up(x)); uses three projections with 8/3x intermediate dimension.

**SiLU (Swish)**: Activation function: x * sigmoid(x); smooth, non-monotonic; used in SwiGLU gate.

**GLU (Gated Linear Unit)**: FFN variant that uses a learned gate to modulate information flow through the hidden layer.

**RoPE (Rotary Position Embedding)**: Position encoding by rotating Q and K vectors by position-dependent angles; makes attention scores depend on relative position only.

**Absolute positional embedding**: Position encoding as a lookup table indexed by position; cannot generalize beyond training length.

**Relative position encoding**: Any scheme where attention scores depend on relative offset (m-n) rather than absolute positions m and n.

**GQA (Grouped-Query Attention)**: Attention variant with fewer KV heads than query heads; multiple query heads share one KV head.

**MQA (Multi-Query Attention)**: Extreme GQA with a single KV head; maximum KV cache reduction, slight quality drop.

**KV cache**: Cached key and value tensors from past tokens; enables O(1) per-step inference instead of O(T²).

**Sliding Window Attention (SWA)**: Restricts each token's attention to a local window of W tokens; O(T*W) vs. O(T²) complexity.

**Effective receptive field**: In SWA, the number of tokens reachable after L layers: approximately L * W.

**Pre-LN**: Applying LayerNorm before each sublayer; yields cleaner gradient flow through residual stream.

**Intermediate dimension**: The hidden size inside the FFN; typically 4x d_model for standard FFN, 8/3x for SwiGLU.
