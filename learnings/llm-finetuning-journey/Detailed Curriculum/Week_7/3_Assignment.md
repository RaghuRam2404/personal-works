# Week 7 — Assignment

## Setup Checklist

- [ ] HuggingFace account created and `huggingface-cli login` works. Verify: `huggingface-cli whoami` prints your username.
- [ ] `pip install transformers datasets huggingface_hub` (or upgrade to latest).
- [ ] HF_TOKEN environment variable set (or logged in via CLI). Check with `python -c "from huggingface_hub import whoami; print(whoami())"`.
- [ ] Colab Free available for the heavier inference tasks. Mac is fine for tokenization tasks.

---

## Task 1 — Load distilgpt2 and Inspect Logits

**Goal:** Become fluent with the `transformers` inference API.

**Requirements:**
- Create `week_07/inspect_gpt2.py`.
- Load `distilgpt2` using `AutoTokenizer` and `AutoModelForCausalLM`.
- Run inference on these 3 prompts:
  ```python
  PROMPTS = [
      "SELECT id FROM",
      "The quick brown fox",
      "PostgreSQL is a database that",
  ]
  ```
- For each prompt:
  1. Tokenize with `return_tensors="pt"`. Print the token count and the token strings (use `convert_ids_to_tokens`).
  2. Forward pass (no generation yet). Extract `logits` from model output. Shape: `(1, seq_len, vocab_size)`.
  3. From `logits[0, -1, :]` (the last token's logit vector), compute `softmax`. Find top-5 predicted next tokens and their probabilities. Print them.
  4. Generate a 30-token continuation using `model.generate()` with `do_sample=True, temperature=0.7`. Print the result.
- Save all output to `week_07/gpt2_inspection.txt`.

**Deliverable:** `week_07/inspect_gpt2.py` + `week_07/gpt2_inspection.txt`.

---

## Task 2 — Load, Explore, and Tokenize the Spider Dataset

**Goal:** Build the data pipeline that will be used in Phase 4 fine-tuning.

**Requirements:**
- Create `week_07/process_spider.py`.
- Load Spider from the HuggingFace Hub:
  ```python
  from datasets import load_dataset
  ds = load_dataset("spider")
  ```
- Print: number of examples in train and validation splits. Print the first 3 examples in full.
- Apply a `map()` to create a new column `"text"` in the format:
  ```
  ### Question: {question}
  ### Database: {db_id}
  ### SQL: {query}
  ```
- Tokenize the `"text"` column using the `Qwen/Qwen2.5-Coder-7B` tokenizer (or `distilgpt2` if you cannot access Qwen on your machine):
  - `max_length=512`, `truncation=True`.
  - Use `batched=True` for speed.
  - Remove columns: `'question'`, `'db_id'`, `'query'`, `'text'`.
- Compute and print: the mean, median, and max token count per example after tokenization.
- Save the tokenized dataset locally: `tokenized.save_to_disk("week_07/spider_tokenized")`.
- Push the tokenized dataset to your HuggingFace account as a private dataset:
  ```python
  tokenized.push_to_hub("YOUR_USERNAME/spider-tokenized-phase1", private=True)
  ```
- Print the Hub URL of the uploaded dataset.

**Deliverable:** `week_07/process_spider.py` + HuggingFace Hub URL in `journal.md`. Commit message: `week-07-hf-onboarding`.

---

## Task 3 — Generation Exploration

**Goal:** Understand the key generation parameters and their effect on output quality.

**Requirements:**
- Create `week_07/generation_exploration.py`.
- Load `distilgpt2` (or a small instruction model if you prefer).
- Use the prompt: `"Question: List all customers who ordered more than 5 times.\nSQL:"`.
- Generate 5 completions (20 tokens each) for each of the following configurations:
  1. `temperature=0.1, do_sample=True` (near-greedy)
  2. `temperature=0.9, do_sample=True` (diverse)
  3. `temperature=1.0, top_k=40, do_sample=True` (top-k filtering)
  4. `temperature=1.0, top_p=0.9, do_sample=True` (nucleus sampling)
  5. `do_sample=False` (greedy)
- Print all 25 completions with their config label.
- In `journal.md`, write 3–5 sentences on: which config produces the most "SQL-like" output? Why?

**Deliverable:** `week_07/generation_exploration.py` + journal notes.

---

## Task 4 — Understanding `labels` for Causal LM

**Goal:** Confirm you understand how the `labels` tensor is constructed for next-token prediction training.

**Requirements:**
- Create `week_07/labels_demo.py`.
- Write a function `prepare_causal_lm_batch(input_ids, attention_mask)` that:
  1. Creates `labels = input_ids.clone()`.
  2. Sets all positions where `attention_mask == 0` to `-100`.
  3. Returns `labels`.
- Test it with a batch of 3 examples of different lengths (padded):
  ```
  Example 1: "SELECT id FROM users"    (short)
  Example 2: "SELECT id, name FROM departments WHERE dept = 'Engineering'"  (medium)
  Example 3: "SELECT"  (very short)
  ```
- Print `input_ids`, `attention_mask`, and `labels` side by side and annotate which positions have `-100`.
- Verify: the positions with `-100` correspond exactly to padding tokens.

**Deliverable:** `week_07/labels_demo.py` with printed output.

---

## Stretch Goals

- Write a `DataCollator` subclass that automatically prepares `labels` from `input_ids` and `attention_mask` using your `prepare_causal_lm_batch` function. Verify it works with a `DataLoader`.
- Load `Qwen/Qwen2.5-Coder-1.5B` (a smaller version accessible without HuggingFace Pro) and compare its SQL inference output on the 3 PROMPTS from Task 1 to distilgpt2. Note differences in SQL keyword awareness.
- Explore `model.push_to_hub("YOUR_USERNAME/my-model")` to push a model (even just distilgpt2 without changes) to your Hub. This is the workflow for saving fine-tuned checkpoints in Phase 4.
