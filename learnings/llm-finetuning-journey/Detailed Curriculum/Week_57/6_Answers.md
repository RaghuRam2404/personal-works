# Week 57 Answers

## Q1 — Answer: B

**Why:** Continued pretraining uses the same objective as original GPT pretraining: given a sequence of tokens, predict the next token. This is the natural language modeling objective. No labels are required beyond the text itself — every token is both input (context) and target (next token to predict). This is fundamentally different from masked LM (BERT) which predicts specific masked positions, and from SFT which requires instruction/response pairs.

---

## Q2 — Answer: B

**Why:** A 1.2-bit increase in general-text perplexity is a significant signal of catastrophic forgetting. The model is overwriting general language representations with domain-specific patterns. At this point the damage is already substantial. The correct action is to stop training, revert to the epoch-1 checkpoint (which presumably showed less forgetting), and use that as your SFT initialization. Multi-epoch CPT is the primary cause of catastrophic forgetting in domain adaptation.

---

## Q3 — Answer: B

**Why:** Without EOS tokens as document separators, the packed sequence contains documents concatenated directly. The model sees the end of document A and the beginning of document B as a continuous stream. It learns: "after the last line of the PostgreSQL window functions chapter, the next token is 'TimescaleDB' (the first word of a TimescaleDB blog post)." These cross-document continuations are noise — the model should not learn them. EOS tokens signal "this is a valid stopping point; what follows is a new, independent document."

---

## Q4 — Answer: B

**Why:** These gains are not additive in a simple way, but CPT provides a better weight initialization for SFT. Without CPT, SFT must simultaneously teach domain vocabulary AND instruction following — both from the same limited fine-tuning budget. With CPT as initialization, the domain vocabulary is already loaded, and SFT only needs to teach instruction following. The result is typically SFT performance is improved when starting from CPT vs. base — often the combined gain exceeds the sum of individual gains because the synergy is multiplicative.

---

## Q5 — Model Answer

The likely cause is corpus imbalance: 60% of your tokens are StackOverflow-style text, which trains the model to be good at answering conversational questions. But StackOverflow Q&A about PostgreSQL is dominated by common questions ("how to do GROUP BY", "how to handle NULLs") not complex TimescaleDB time-series SQL. TimescaleDB content at 10% is insufficient to meaningfully shift the model's distribution toward time-series SQL.

Fix for a future CPT run: increase TimescaleDB content to at least 25% of the corpus. Sources: all TimescaleDB documentation pages (currently), TimescaleDB GitHub issues and discussions, TimescaleDB community Slack exports (if publicly available), and TimescaleDB-specific Stack Overflow posts (there are ~15K questions tagged [timescaledb]). Also add synthetic TimescaleDB SQL files from your v3 dataset as raw text (without instruction format) — this reinforces TimescaleDB syntax representation.

---

## Q6 — Model Answer

**Packing efficiency** is the fraction of tokens in packed sequences that are actual content tokens (not padding). If you pack to exactly the context length without padding, efficiency is near 100%. High packing efficiency matters because H100 charges by time, not by tokens processed. If 20% of your sequence slots are padding, you are paying for 20% idle GPU computation.

**Data utilization** is the fraction of your corpus tokens that actually make it into training sequences. If your documents are very short (< 50 tokens), they waste packing slots and increase the number of EOS separators. Long documents are more efficiently packed.

With 100M tokens and 2048-length sequences, perfectly packed you get 48,828 training steps. With 90% packing efficiency, you effectively train on 90M tokens — still acceptable. Below 80%, you should investigate short document filtering.

---

## Q7 — Model Answer

Three before/after metrics:

1. **Domain perplexity on a held-out PostgreSQL documentation page** (1,000 tokens not in training): measures how well the model's distribution covers PostgreSQL technical text. Target: ≥ 10% decrease in perplexity (bits/token) after CPT.

2. **TimescaleDB function completion accuracy**: create a test set of 100 prompts that are partial TimescaleDB function calls (e.g., "SELECT time_bucket('1 hour',") and measure the fraction where the model correctly completes the function signature. Target: ≥ 20 percentage point improvement.

3. **General-text perplexity on a Wikipedia sample** (1,000 tokens of general English, not SQL): measures forgetting. Target: < 0.3 bits/token increase — if it exceeds 0.5, the CPT run has caused unacceptable forgetting.

---

## Q8 — Model Answer

Experiment design: run SFT for identical steps (1,000 steps) from each of the three initializations, using the same v3 dataset and hyperparameters. Evaluate on your custom 200-example PostgreSQL/TimescaleDB benchmark and on BIRD-SQL dev set after every 200 steps.

Metrics: execution accuracy on domain benchmark, execution accuracy on BIRD-SQL, training loss curve shape (does it converge faster from one initialization?).

Hypothesis: Option B (CPT checkpoint) will achieve the highest domain benchmark score but may be slightly behind Option C (Phase 5 GRPO model) on BIRD-SQL because the GRPO model already has instruction-following SQL skills. Option A (base model) will be lowest.

Decision criterion: choose the initialization that achieves the highest score on your held-out domain benchmark (not BIRD-SQL) at 1,000 SFT steps. This reflects the actual production use case. If B and C are within 1 percentage point, choose B (fresh initialization avoids potential distribution shift from the GRPO training data). If C is clearly better (> 2 pp), consider: does C's Phase 5 GRPO training data overlap with your v3 SFT data? If yes, C's advantage may be contamination, not genuine capability, and you should use B.
