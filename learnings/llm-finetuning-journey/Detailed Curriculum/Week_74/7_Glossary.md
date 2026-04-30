# Week 74 Glossary — Context Extension: LongRoPE and YaRN

**RoPE (Rotary Position Embedding)**: A position encoding method that encodes relative position by rotating query and key vectors; enables the attention score to depend on relative position (m-n) rather than absolute positions.

**Position interpolation**: A naive context extension technique that scales all position indices by old_context/new_context to stay within the trained range; degrades high-frequency positional resolution.

**NTK-aware scaling**: YaRN's context extension technique that applies different scaling factors to different RoPE frequency dimensions, preserving high-frequency resolution while extending low-frequency range.

**YaRN (Yet Another RoPE extensioN)**: A context extension method using non-uniform RoPE interpolation and attention temperature scaling; requires only 400 fine-tuning steps for 2x context extension.

**LongRoPE**: A context extension method that uses evolutionary search to find the optimal per-dimension RoPE rescaling factors; achieves better long-context quality than YaRN at significantly higher compute cost.

**Evolutionary search (LongRoPE)**: A population-based optimization algorithm that iteratively evaluates, mutates, and combines candidate rescaling factor vectors to find the optimal per-dimension RoPE scaling.

**Two-stage training (LongRoPE)**: A context extension training procedure with Stage 1 at the target long context and Stage 2 at an intermediate shorter context to recover short-context performance.

**Flash Attention 2**: An exact attention implementation using tiled computation that reduces memory complexity from O(N²) to O(N), enabling long-context inference on consumer GPUs.

**Schema compression**: A preprocessing technique that selects only the relevant tables from a large database schema based on the user's question, reducing prompt length without requiring context extension.

**Context window**: The maximum number of tokens a transformer model can process in a single forward pass; determined by the position encoding's valid range and the model's training context length.
