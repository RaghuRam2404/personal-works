# Week 75 Glossary — Iteration: Different Base Models

**Controlled experiment:** Experimental design that varies one factor (base model) while holding all others constant.

**Response template:** Token string marking the start of the assistant turn, used by the data collator to mask prompt loss.

**Chat template:** Model-specific formatting that wraps user/assistant turns in special tokens understood by the pretrained model.

**Sliding-window attention (SWA):** Attention variant where each token attends only to a local window of neighbors, reducing context from O(n²) to O(n·w).

**GQA (Grouped Query Attention):** Attention variant with fewer key/value heads than query heads; reduces memory and increases inference throughput.

**Reasoning distillation:** Training a smaller model to imitate the chain-of-thought outputs of a larger teacher, transferring multi-step reasoning ability.

**Switching threshold:** The minimum EM improvement (here, ≥2 pp) required to justify migrating the full training pipeline to a new base model.

**Token boundary alignment:** Verification that response delimiters in the chat template exactly match the token IDs produced by the tokenizer for that model.

**Architecture family:** Group of models sharing a common design (e.g., Qwen family, LLaMA family) with shared tokenizer and attention variant.

**Seed variance:** Standard deviation of an evaluation metric across runs with different random seeds; measures training stability.

**Length-stratified evaluation:** Measuring performance separately across input length bins to detect context-length-dependent degradation.

**Pipeline compatibility:** The property that a base model can be used in every stage of your training recipe (CPT, SFT, DPO, GRPO) without architecture-specific failures.
