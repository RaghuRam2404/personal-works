# Week 22 Glossary — LM Evaluation

**Perplexity**: exp(cross_entropy_loss); measures average model uncertainty per token; lower is better.

**Cross-entropy loss**: The negative log-likelihood per token under the model's distribution; the training and evaluation loss for language models.

**Bits-per-character (BPC)**: Perplexity normalized by character count instead of token count; enables cross-tokenizer comparison.

**Greedy decoding**: Generating text by always selecting the highest-probability next token; deterministic but often repetitive.

**Temperature (sampling)**: A scaling factor applied to logits before softmax; T<1 sharpens the distribution (less diverse), T>1 flattens it (more diverse).

**Top-k sampling**: Restricting next-token candidates to the k highest-probability tokens at each step, setting all others to -infinity.

**Top-p (nucleus) sampling**: Restricting candidates to the smallest set of tokens whose cumulative probability exceeds p; dynamically adapts the candidate set.

**Repetition penalty**: A technique that reduces the logit score of any token that has already appeared in the context, discouraging repetitive output.

**Sliding window perplexity**: Computing perplexity with a stride shorter than context_len, so each token is predicted with maximum available context.

**Validation set**: Held-out data the model never trained on; used to estimate generalization performance and detect overfitting.

**Data contamination**: When validation or test set examples appear in the training data; causes artificially low evaluation perplexity.

**Model card**: A documentation page on HuggingFace Hub describing a model's architecture, training data, evaluation results, and intended use.

**Beam search**: A deterministic decoding algorithm that maintains a beam of the k most likely partial sequences at each step; produces more grammatically correct output than sampling but less diverse.
