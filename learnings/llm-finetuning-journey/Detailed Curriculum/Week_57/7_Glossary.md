# Week 57 Glossary

**Continued pretraining (CPT):** Running the causal language modeling objective on domain-specific text, starting from a general pretrained model, to shift its distribution toward a target domain.

**Causal language modeling:** A training objective where the model predicts the next token given all preceding tokens; the standard objective for decoder-only transformers like GPT and Qwen.

**Catastrophic forgetting:** The degradation of a model's performance on previously learned tasks when fine-tuned on new data; primary risk of multi-epoch CPT.

**Document packing:** Concatenating multiple training documents (separated by EOS tokens) into fixed-length sequences to maximize GPU utilization during training.

**EOS token:** End-of-sequence token; inserted between documents during packing to prevent the model from learning cross-document continuations.

**Packing efficiency:** The fraction of tokens in packed training sequences that are actual content (not padding); target > 95% for efficient training.

**DataCollatorForLanguageModeling:** A HuggingFace utility that prepares causal language modeling batches; with `mlm=False`, it shifts input_ids right to create labels for next-token prediction.

**Domain perplexity:** Perplexity computed on a held-out sample of domain text (e.g., PostgreSQL documentation); measures how well the model covers the target distribution.

**General perplexity:** Perplexity on non-domain text (e.g., Wikipedia); used as a forgetting monitor during CPT.

**Weight decay:** A regularization technique that penalizes large weights; should be higher for CPT (0.1) than SFT to prevent overfitting to a 100M-token corpus.

**Warmup steps:** The number of initial training steps during which the learning rate is linearly ramped from 0 to the target LR; CPT needs more warmup (100–1000 steps) than SFT to avoid early instability.

**The Stack v2:** HuggingFace's large-scale code corpus with permissive licensing; filtered for `.sql` files to source PostgreSQL SQL examples for CPT.
