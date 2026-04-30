# Week 23 Glossary — LM Evaluation

**lm-evaluation-harness**: EleutherAI's open-source framework for evaluating language models on 60+ standardized benchmarks; uses log-likelihood scoring for multiple-choice tasks.

**MMLU (Massive Multitask Language Understanding)**: A 14,000-question multiple-choice benchmark across 57 subjects; tests factual knowledge and reasoning; random baseline is 25%.

**HellaSwag**: A 70,000-question multiple-choice benchmark testing narrative commonsense (picking the most plausible next sentence); uses length-normalized log-likelihood scoring.

**ARC (AI2 Reasoning Challenge)**: Elementary science questions in multiple-choice format; ARC-Easy tests basic facts; ARC-Challenge requires multi-hop reasoning.

**ARC-Easy / ARC-Challenge**: Two difficulty tiers of the ARC benchmark; ARC-Easy has higher GPT-2 accuracy (43%) than ARC-Challenge (26%).

**Log-likelihood scoring**: Evaluating a model on multiple-choice by selecting the option with the highest log P(option | question); works on any causal language model without instruction tuning.

**0-shot evaluation**: Running a benchmark without providing any example question-answer pairs as context; tests the model's learned knowledge directly.

**5-shot (few-shot) evaluation**: Prepending 5 labeled example pairs to each test question as context; helps models that are instruction-following capable; boosts scores significantly for capable models.

**Benchmark contamination**: When the test set of a benchmark appears in a model's training data, artificially inflating evaluation scores.

**Execution Accuracy (EA)**: The fraction of generated SQL queries that return the correct rows when executed against the reference database; the primary metric for text-to-SQL evaluation.

**Spider benchmark**: A standard cross-domain text-to-SQL benchmark with 200+ databases and 10,000+ questions; widely used to compare NL→SQL models.

**Bits-per-character (BPC)**: A perplexity normalization that divides log-likelihood by character count; enables comparison across models with different tokenizers.

**Exact Match (EM)**: The fraction of generated SQL queries that are identical to the reference query string; a strict but often misleading metric since equivalent SQL can be written many ways.

**Semantic Equivalence**: Two SQL queries are semantically equivalent if they return the same rows for all possible database states; the correct standard for SQL evaluation.
