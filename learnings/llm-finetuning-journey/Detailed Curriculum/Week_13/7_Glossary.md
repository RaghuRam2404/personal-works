# Week 13 Glossary

**KV cache**: Stored past key and value tensors from prior generation steps; eliminates redundant recomputation at inference.

**Autoregressive inference**: Token-by-token generation where each new token is conditioned on all previous tokens.

**Greedy decoding**: Always selecting the highest-probability next token; deterministic and fast but repetitive.

**Temperature**: Scalar divisor applied to logits before softmax; lower = sharper (more deterministic), higher = flatter (more random).

**Top-k sampling**: Restrict sampling to the k highest-probability tokens before softmax renormalization.

**Top-p (nucleus) sampling**: Restrict sampling to the smallest set of tokens whose cumulative probability exceeds p; adapts set size to model confidence.

**Beam search**: Search algorithm maintaining B candidate sequences; expensive for LLMs; produces degenerate output at scale.

**Speculative decoding**: Inference optimization using a small draft model to propose K tokens, verified in parallel by the large model.

**Repetition penalty**: Reduces logits for tokens already present in the context; decreases repetitive generation.

**Tokens per second (tok/s)**: Primary inference speed metric; measures throughput of the generation loop.

**Prefill**: The initial forward pass that processes the full prompt and populates the KV cache.

**Decode**: The subsequent token-by-token generation phase using the KV cache.

**KV cache eviction**: Dropping oldest cached tokens when context exceeds block_size; enables sliding window inference.
