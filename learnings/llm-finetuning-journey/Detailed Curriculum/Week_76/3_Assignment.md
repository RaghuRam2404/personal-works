# Week 76 Assignment — Multi-Turn Agentic SQL with Tool Use

## Setup Checklist

- [ ] PostgreSQL instance running locally (or on Colab via `apt-get install postgresql` + `service postgresql start`)
- [ ] Tables loaded: at minimum `orders`, `customers`, `products`, `timeseries_metrics` (use your existing schema from Week 53 onward)
- [ ] `psycopg2` installed: `pip install psycopg2-binary`
- [ ] Your fine-tuned model (`postgres-sqlcoder-7b-final`) loaded locally or accessible via the HuggingFace Hub
- [ ] W&B project `week-76-agentic-sql` created
- [ ] Python environment with `transformers>=4.40`, `trl>=0.8`, `peft>=0.10`

---

## Task 1 — Implement the Tool-Calling Format

**Goal:** Produce the canonical tool-call conversation format for your base model and verify it round-trips correctly.

**Requirements:**
- [ ] Use `tokenizer.apply_chat_template` with a `tools=` argument containing the `execute_sql` schema.
- [ ] Construct a sample three-turn conversation: user question → assistant tool call → tool result → assistant final answer.
- [ ] Print the full tokenized string (decoded back to text) and verify: (a) tool call appears in the correct position, (b) response delimiters are correct for your model family, (c) no `<unk>` tokens appear in SQL keywords.
- [ ] Identify the exact string that marks the start of each assistant turn (you will need this for loss masking).

**Deliverable:** `week76/tool_format_check.py` — script that prints the full conversation string and the detected response templates.

**Hints:** If `apply_chat_template` does not accept `tools=` for your model version, implement the tool-call format manually using the model card's documented schema. Qwen2.5-Coder uses `<tool_call>` tags; Llama 3.1 Instruct uses a JSON-in-text format surrounded by `<|python_tag|>`.

---

## Task 2 — Build a Multi-Turn Trajectory Dataset

**Goal:** Generate 500 agentic training trajectories using synthetic error injection.

**Requirements:**
- [ ] Start from your existing SFT dataset (single-shot SQL examples).
- [ ] For each example, apply one of three mutations to the correct SQL to generate a "first attempt" error: (a) drop a JOIN clause, (b) replace a column name with a nearby but wrong column name, (c) add a syntax error (extra comma, missing closing parenthesis).
- [ ] Execute the mutated SQL against your PostgreSQL instance and capture the error message (exactly as `psycopg2` returns it).
- [ ] Format each example as a multi-turn conversation: user → assistant (tool call with wrong SQL) → tool (error string) → assistant (tool call with correct SQL) → tool (rows) → assistant (final confirmation).
- [ ] Save to `week76/agentic_train.jsonl` with one JSON object per line, each containing a `messages` key.
- [ ] Verify dataset: at least 60% of trajectories have a genuine error in round 1 (not just a warning).

**Deliverable:** `week76/build_trajectories.py` + `week76/agentic_train.jsonl`.

---

## Task 3 — Multi-Turn SFT with Correct Loss Masking

**Goal:** Fine-tune your model on the agentic trajectory dataset with loss computed only on assistant turns.

**Requirements:**
- [ ] Implement a custom `DataCollatorForMultiTurnCompletion` that: (a) concatenates all turns into a single input, (b) sets labels to -100 for user and tool turns, (c) preserves labels for all assistant turns.
- [ ] Train for 500 steps with LoRA (r=64, alpha=128, same target modules as your existing model).
- [ ] Learning rate: 1e-4 (slightly lower than standard SFT to avoid overwriting single-shot capability).
- [ ] Log to W&B project `week-76-agentic-sql`: training loss, eval loss, and a per-step count of how many tokens were unmasked (to verify masking is working).
- [ ] Checkpoint at step 500.

**Requirements for the collator:**
- Loss should be computed on approximately 30–50% of tokens per batch (assistant turns only). If you see >80% unmasked, your masking is broken.

**Deliverable:** `week76/train_agentic.py` + W&B run link + saved checkpoint at `week76/checkpoints/step-500/`.

---

## Task 4 — Implement and Evaluate the Agentic Inference Loop

**Goal:** Run your agentic model against 40 Custom-200 questions and compare first-attempt vs final EM.

**Requirements:**
- [ ] Implement `agentic_sql_loop(model, tokenizer, question, schema, db_conn, max_rounds=3)` that: generates SQL → executes against PostgreSQL → feeds result back → repeats until non-tool response or max rounds.
- [ ] Run on 40 examples from Custom-200 (same 40 for fair comparison with your week-75 baseline).
- [ ] Compute and log: first-attempt EM, final EM, correction rate (fraction of failures in round 1 that succeed in round 2), mean rounds per example.
- [ ] Log all results to `week76/agentic_eval_results.json`.
- [ ] Compare: does final EM exceed single-shot baseline (83.1%)? Does it exceed first-attempt EM? Write 3-sentence interpretation.

**Deliverable:** `week76/agentic_loop.py` + `week76/agentic_eval_results.json` + interpretation in `week76/results_memo.md`.

---

## Stretch Goals

- Add a second tool: `get_schema(table_name)` that returns column definitions. Train the model to call it when the question implies a table it did not see in the initial prompt. Measure whether schema-lookup reduces hallucinated column names.
- Implement a "think before calling" turn: add a reasoning step between user question and first tool call (format: `<think>...</think><tool_call>...</tool_call>`). Measure whether this reduces first-attempt errors.
- Run the agentic loop with your single-shot (non-agentic-SFT) model and compare its correction rate to your agentic-SFT model's rate. This ablates the value of agentic training vs agentic inference alone.
