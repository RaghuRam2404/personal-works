# Week 17 Glossary — Scaling Laws

**Scaling law**: A power-law relationship between model performance (loss) and model size, dataset size, or compute budget.

**Compute-optimal training**: The (N, D) allocation that minimizes validation loss for a fixed compute budget C.

**Chinchilla scaling**: The finding that optimal N and D scale as C^{0.5}, implying roughly 20 tokens per parameter.

**FLOP (Floating-Point Operation)**: A single multiply or add on floating-point numbers; used to measure computation cost.

**MFU (Model FLOP Utilization)**: Ratio of measured throughput to peak hardware FLOP/s; reflects real-world efficiency losses from memory bandwidth and communication overhead.

**6ND approximation**: The rule C ≈ 6 × N × D for estimating training FLOPs; 2ND for forward, 4ND for backward pass.

**IsoFLOP profile**: A curve of (N, D) pairs that all consume the same total compute C, used to find the loss-minimizing allocation.

**Inference-optimal training**: Training a smaller model on more tokens than Chinchilla recommends, reducing inference cost at the expense of extra training compute.

**Power law**: A relationship of the form y = a × x^b; characterizes scaling laws where loss decreases predictably with resources.

**Compute budget (C)**: Total floating-point operations available for a training run, typically expressed in FLOPs or petaFLOP-days.

**Under-trained model**: A model trained with fewer tokens than Chinchilla-optimal for its parameter count (e.g., GPT-3).

**Over-parameterized model**: A model with more parameters than Chinchilla-optimal for its compute budget; wasteful if inference cost is not a concern.

**Kaplan scaling**: The original (2020) scaling law finding that recommended scaling parameters faster than tokens; later revised by Chinchilla.
