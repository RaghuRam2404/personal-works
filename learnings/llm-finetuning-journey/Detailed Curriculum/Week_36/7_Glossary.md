# Week 36 Glossary

**DoRA (Weight-Decomposed Low-Rank Adaptation)**: A LoRA variant that decomposes the pretrained weight into magnitude and direction, adapting both via a learned scalar vector and a low-rank BA update; consistently outperforms standard LoRA on SFT tasks.

**Magnitude decomposition**: Expressing a weight matrix W = m × (V/||V||), where m is a vector of norms (one per row) and V/||V|| is the normalized direction matrix; enables separate optimization of scale and direction.

**Direction update**: In DoRA, the low-rank BA matrix updates the direction component of the weight; analogous to standard LoRA but applied to the normalized weight.

**RSLoRA (Rank-Stabilized LoRA)**: A LoRA variant that uses `alpha / sqrt(r)` as the scaling factor instead of `alpha / r`, preventing the adapter contribution from collapsing at high ranks.

**Rank stabilization**: The property that the LoRA output magnitude remains approximately constant as rank increases (when alpha is fixed); achieved by RSLoRA's `alpha / sqrt(r)` formula.

**LoftQ (LoRA-Fine-Tuning-Aware Quantization)**: An initialization method for QLoRA that pre-initializes the LoRA adapter to approximate the quantization error (W_fp16 - W_nf4) using truncated SVD, reducing the gap from the BF16 baseline at step 0.

**Quantization error compensation**: Adjusting the initial adapter weights to partially cancel the accuracy loss introduced by NF4 quantization; LoftQ's core mechanism.

**Truncated SVD**: Singular value decomposition of a matrix retaining only the top-r singular values and corresponding vectors; gives the best rank-r approximation of a matrix in the Frobenius norm sense.

**`use_dora`**: peft `LoraConfig` boolean flag enabling DoRA decomposition; requires peft >= 0.9.0.

**`use_rslora`**: peft `LoraConfig` boolean flag enabling rank-stabilized scaling (`alpha / sqrt(r)`); no version constraint beyond standard peft.

**`init_lora_weights="loftq"`**: peft `LoraConfig` flag enabling LoftQ initialization; requires specifying `loftq_config=LoftQConfig(...)`.
