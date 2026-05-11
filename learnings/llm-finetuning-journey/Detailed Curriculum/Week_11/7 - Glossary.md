# Week 11 Glossary

**Decoder-only transformer**: Transformer architecture using only the decoder stack (causal self-attention + FFN); no encoder or cross-attention.

**Causal language modeling (CLM)**: Pretraining objective: predict the next token given all previous tokens; requires no labeled data.

**Autoregressive generation**: Generating tokens one at a time, feeding each generated token as input to produce the next.

**Weight tying**: Sharing the same weight matrix for the input embedding table and the output LM head projection.

**LM head**: The final linear layer that maps d_model → vocab_size to produce logits over the vocabulary.

**Residual stream**: The vector x that persists throughout the network; each sublayer reads from it and writes back via addition.

**GELU**: Gaussian Error Linear Unit activation function; used in GPT MLP blocks instead of ReLU; smoother gradient.

**Block size (context length)**: Maximum number of tokens the model can attend to in a single forward pass.

**Temperature (sampling)**: Scalar dividing logits before softmax; lower = more deterministic; higher = more diverse.

**Pretraining**: Training a model from scratch on a large corpus using a self-supervised objective (e.g., CLM).

**nanoGPT**: Andrej Karpathy's minimal decoder-only GPT implementation in ~300 lines of PyTorch.

**Pre-LN (Pre-LayerNorm)**: Applying LayerNorm before each sublayer rather than after; more stable training gradient flow.
