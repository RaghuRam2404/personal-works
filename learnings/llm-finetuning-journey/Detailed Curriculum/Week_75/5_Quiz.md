# Week 75 Quiz — Iteration: Different Base Models

Difficulty: Senior research engineer. Questions are scenario-driven and require you to reason about experimental design, model internals, and practical trade-offs in NL→SQL fine-tuning.

---

## Multiple Choice

**Q1.** You run SFT on four base models (Qwen2.5-Coder-7B, Llama 3.1 8B, Gemma 2 9B, DeepSeek-R1-Distill-Qwen-7B) using identical hyperparameters: same LoRA rank, same learning rate, same number of steps, same data. Validation loss converges smoothly for Qwen2.5-Coder and Llama, but diverges after step 300 for Gemma 2. The most likely root cause is:

A. Gemma 2 uses a different tokenizer, so the effective batch size in tokens is smaller, causing gradient noise.
B. Gemma 2's GQA architecture requires a different learning rate scaling than models with full MHA.
C. The chat template for Gemma 2 was not applied correctly, so the model sees malformed instruction tokens and the loss is computed over the full sequence including the prompt.
D. Gemma 2 has higher embedding norm by default, which interacts poorly with LoRA's default initialization.

---

**Q2.** DeepSeek-R1-Distill-Qwen-7B was distilled from a reasoning model using chain-of-thought supervision. When you fine-tune it for direct NL→SQL (no reasoning trace in target), which of the following is the most accurate statement about expected behavior?

A. The model will underperform Qwen2.5-Coder-7B because reasoning ability is irrelevant for SQL.
B. The model may outperform on complex queries because latent reasoning capacity influences feature representations even when no CoT is generated, but you risk degrading reasoning ability if SFT steps are too many.
C. The model will always generate reasoning traces regardless of your target format because the behavior is baked into weights.
D. The distillation training means the model has seen fewer tokens, so it will always underperform code-specialized models on SQL.

---

**Q3.** You want to ensure your base model comparison experiment is properly controlled. Which of the following is NOT a valid control variable?

A. Number of SFT gradient steps.
B. LoRA rank, alpha, and target modules (using equivalent module names per architecture).
C. The evaluation benchmark (Custom-200).
D. The tokenizer vocabulary size.

---

**Q4.** After running all four experiments, you find: Qwen2.5-Coder-7B = 83.1% EM, DeepSeek-R1-Distill-Qwen-7B = 85.3% EM, Llama 3.1 8B = 81.4% EM, Gemma 2 9B = 82.0% EM. Your switching threshold is ≥2 pp improvement. You decide to switch to DeepSeek. Which next step is most important before committing to the switch?

A. Re-run SFT on DeepSeek with double the steps to confirm the trend holds at full scale.
B. Verify that the full pipeline (CPT → SFT → DPO → GRPO) can be reproduced end-to-end with DeepSeek, not just SFT alone.
C. Benchmark DeepSeek against Spider 1.0 first because Custom-200 is a biased benchmark.
D. Run quantization (GGUF Q4) on DeepSeek before deciding because quantization degradation varies by architecture.

---

**Q5.** Llama 3.1 8B uses a different special token set than Qwen2.5-Coder-7B. When adapting your training data pipeline, the most critical step is:

A. Re-tokenizing the entire dataset with the Llama tokenizer and verifying no token IDs map to `<unk>`.
B. Using `apply_chat_template` from the model's tokenizer, verifying that SQL tokens are not split unexpectedly, and confirming that `instruction_template` boundaries align with what the model was pretrained on.
C. Padding all sequences to the same length because Llama requires fixed-length inputs during SFT.
D. Adding Llama-specific domain tokens (such as `<SQL>` and `</SQL>`) to the vocabulary before fine-tuning.

---

## Short Answer

**Q6.** Explain the difference between a model that is "code-specialized" (like Qwen2.5-Coder) and one that is "reasoning-distilled" (like DeepSeek-R1-Distill-Qwen-7B). For NL→SQL fine-tuning, under what conditions would you prefer each type? Be specific about query complexity and dataset size.

---

**Q7.** You are comparing Gemma 2 9B and Llama 3.1 8B for the same NL→SQL task. Gemma 2 has 9B parameters and uses sliding-window attention. Llama 3.1 8B uses RoPE with 128K context support. Describe two concrete ways the architectural differences could affect your fine-tuning experiment design and evaluation on TimescaleDB queries (which often involve long schema contexts).

---

**Q8.** You run five-seed evaluation for each model (five different random seeds for SFT, same data) and find that DeepSeek-R1-Distill-Qwen-7B has mean EM = 85.3% but standard deviation = 2.8%, while Qwen2.5-Coder-7B has mean = 83.1% and standard deviation = 0.6%. What does this tell you about the two models, and how does it affect your switching decision?

---

**Q9.** Your colleague argues that you should run hyperparameter search per model (find the optimal LR for each architecture separately) before comparing them. You argue for using a single fixed hyperparameter set across all models. What are the strongest arguments for each position, and which do you favor for this week's experiment? Justify.

---

## Deep Scenario

**Q10.** You run the four-model comparison and notice the following pattern: on simple single-table queries (40% of Custom-200), all four models perform similarly (±1% EM). On complex multi-table JOIN + GROUP BY + HAVING queries (60% of Custom-200), DeepSeek-R1-Distill-Qwen-7B outperforms the others by 4–6% EM. On time-series queries involving TimescaleDB-specific functions (time_bucket, first, last), Qwen2.5-Coder-7B outperforms DeepSeek by 3% EM.

Write a structured analysis (4–6 sentences) that: (a) diagnoses why each model excels in its respective category, (b) proposes a concrete next experiment to test your hypothesis, and (c) recommends what you would do if you had to ship one model today versus if you had two more weeks of compute.
